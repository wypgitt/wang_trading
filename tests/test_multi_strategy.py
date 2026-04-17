"""Tests for the multi-strategy allocator (§8.4 + §8.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.multi_strategy import (
    MultiStrategyAllocator,
    compute_portfolio_risk_metrics,
)


def _strat_returns(
    strategies: list[str],
    instruments_per_strategy: int = 5,
    n_bars: int = 300,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    out: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(strategies):
        cols = [f"{s}_inst_{j}" for j in range(instruments_per_strategy)]
        data = rng.normal(0, 0.01, size=(n_bars, instruments_per_strategy))
        # small drift to give allocator a signal
        data[:, 0] += 0.0005 * (i + 1)
        out[s] = pd.DataFrame(data, index=idx, columns=cols)
    return out


def _signals_and_sizes(strategies: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_syms = [c for df in strategies.values() for c in df.columns]
    first_df = next(iter(strategies.values()))
    idx = first_df.index
    signals = pd.DataFrame(1, index=idx, columns=all_syms, dtype=float)
    bets = pd.DataFrame(1.0, index=idx, columns=all_syms, dtype=float)
    return signals, bets


class TestConstraints:
    def test_gross_exposure_cap(self):
        strategies = ["momentum", "mean_reversion", "carry"]
        rs = _strat_returns(strategies, instruments_per_strategy=5)
        sig, bet = _signals_and_sizes(rs)
        alloc = MultiStrategyAllocator(max_gross_exposure=1.0)
        tgt = alloc.compute_target_portfolio(rs, sig, bet)
        assert tgt["target_weight"].abs().sum() <= 1.0 + 1e-9

    def test_no_strategy_exceeds_cap(self):
        strategies = ["momentum", "mean_reversion", "carry"]
        rs = _strat_returns(strategies, instruments_per_strategy=5)
        sig, bet = _signals_and_sizes(rs)
        alloc = MultiStrategyAllocator(
            max_strategy_weight=0.25, max_instrument_weight=0.99, max_gross_exposure=10.0
        )
        tgt = alloc.compute_target_portfolio(rs, sig, bet)
        per_strat_gross = tgt.groupby("strategy")["target_weight"].apply(
            lambda s: s.abs().sum()
        )
        assert (per_strat_gross <= 0.25 + 1e-9).all()

    def test_no_instrument_exceeds_cap(self):
        strategies = ["momentum", "mean_reversion"]
        rs = _strat_returns(strategies, instruments_per_strategy=3)
        sig, bet = _signals_and_sizes(rs)
        alloc = MultiStrategyAllocator(max_instrument_weight=0.05)
        tgt = alloc.compute_target_portfolio(rs, sig, bet)
        assert (tgt["target_weight"].abs() <= 0.05 + 1e-9).all()

    def test_crypto_allocation_capped(self):
        strategies = ["momentum", "mean_reversion"]
        rs = _strat_returns(strategies, instruments_per_strategy=4)
        sig, bet = _signals_and_sizes(rs)
        all_syms = [c for df in rs.values() for c in df.columns]
        # mark the momentum bucket as crypto
        asset_map = {s: ("crypto" if s.startswith("momentum") else "equities") for s in all_syms}
        alloc = MultiStrategyAllocator(
            max_crypto_pct=0.20,
            max_instrument_weight=0.99,
            max_strategy_weight=0.99,
            max_gross_exposure=10.0,
            asset_class_map=asset_map,
        )
        tgt = alloc.compute_target_portfolio(rs, sig, bet)
        crypto_gross = (
            tgt.loc[tgt["symbol"].map(lambda s: asset_map[s] == "crypto"), "target_weight"]
            .abs()
            .sum()
        )
        assert crypto_gross <= 0.20 + 1e-9


class TestRegimeTilt:
    def test_trending_boosts_momentum_weight(self):
        strategies = ["momentum", "mean_reversion", "carry"]
        rs = _strat_returns(strategies, instruments_per_strategy=5, seed=1)
        sig, bet = _signals_and_sizes(rs)
        alloc = MultiStrategyAllocator(
            strategy_optimizer="equal_weight",
            instrument_optimizer="equal_weight",
            max_strategy_weight=0.99,
            max_instrument_weight=0.99,
            max_gross_exposure=10.0,
            regime_boost=0.50,
        )
        neutral = alloc.compute_target_portfolio(rs, sig, bet, regime=None)
        trending = alloc.compute_target_portfolio(rs, sig, bet, regime="trending")

        mom_neutral = neutral.loc[neutral["strategy"] == "momentum", "target_weight"].abs().sum()
        mom_trending = trending.loc[trending["strategy"] == "momentum", "target_weight"].abs().sum()
        assert mom_trending > mom_neutral

    def test_mean_reverting_boosts_mean_reversion_weight(self):
        strategies = ["momentum", "mean_reversion", "carry"]
        rs = _strat_returns(strategies, instruments_per_strategy=5, seed=2)
        sig, bet = _signals_and_sizes(rs)
        alloc = MultiStrategyAllocator(
            strategy_optimizer="equal_weight",
            instrument_optimizer="equal_weight",
            max_strategy_weight=0.99,
            max_instrument_weight=0.99,
            max_gross_exposure=10.0,
            regime_boost=0.50,
        )
        neutral = alloc.compute_target_portfolio(rs, sig, bet, regime=None)
        mrv = alloc.compute_target_portfolio(rs, sig, bet, regime="mean_reverting")
        mr_n = neutral.loc[neutral["strategy"] == "mean_reversion", "target_weight"].abs().sum()
        mr_r = mrv.loc[mrv["strategy"] == "mean_reversion", "target_weight"].abs().sum()
        assert mr_r > mr_n


class TestRebalanceTrades:
    def _alloc_and_target(self):
        strategies = ["momentum", "mean_reversion", "carry"]
        rs = _strat_returns(strategies, instruments_per_strategy=3)
        sig, bet = _signals_and_sizes(rs)
        alloc = MultiStrategyAllocator(
            max_instrument_weight=0.50, max_gross_exposure=10.0, max_strategy_weight=1.0
        )
        tgt = alloc.compute_target_portfolio(rs, sig, bet)
        all_syms = list(tgt["symbol"])
        prices = pd.Series(100.0, index=all_syms)
        return alloc, tgt, prices

    def test_new_position_is_a_buy(self):
        alloc, tgt, prices = self._alloc_and_target()
        trades = alloc.compute_rebalance_trades(
            tgt, current_positions={}, prices=prices, nav=100_000
        )
        # At least one row and all are new_position buys (target was positive)
        buys = trades[trades["reason"] == "new_position"]
        assert len(buys) > 0
        assert (buys["side"] == "buy").all()

    def test_exit_position_is_a_sell(self):
        alloc, tgt, prices = self._alloc_and_target()
        # Pretend we're currently long a symbol that's not in the target
        phantom = "phantom_holding"
        prices_aug = pd.concat([prices, pd.Series(50.0, index=[phantom])])
        trades = alloc.compute_rebalance_trades(
            tgt,
            current_positions={phantom: 0.05},
            prices=prices_aug,
            nav=100_000,
        )
        exits = trades[trades["reason"] == "exit"]
        assert len(exits) == 1
        assert exits.iloc[0]["side"] == "sell"

    def test_minimum_trade_size_filter(self):
        alloc, tgt, prices = self._alloc_and_target()
        # Set current position equal to target → delta ≈ 0 → filtered
        current = dict(zip(tgt["symbol"], tgt["target_weight"]))
        trades = alloc.compute_rebalance_trades(
            tgt, current_positions=current, prices=prices, nav=100_000
        )
        assert trades.empty

    def test_bad_nav_rejected(self):
        alloc = MultiStrategyAllocator()
        with pytest.raises(ValueError):
            alloc.compute_rebalance_trades(
                pd.DataFrame({"symbol": ["x"], "target_weight": [0.1]}),
                current_positions={},
                prices=pd.Series([100.0], index=["x"]),
                nav=-1.0,
            )


class TestRiskMetrics:
    def test_diversification_ratio_and_effective_n(self):
        cov = pd.DataFrame(
            np.diag([0.04, 0.04, 0.04, 0.04]),
            index=list("abcd"),
            columns=list("abcd"),
        )
        w = pd.Series([0.25, 0.25, 0.25, 0.25], index=list("abcd"))
        out = compute_portfolio_risk_metrics(w, cov)
        assert out["effective_n"] == pytest.approx(4.0)
        assert out["portfolio_volatility"] > 0
        assert out["diversification_ratio"] >= 1.0 - 1e-9

    def test_concentrated_portfolio_has_low_effective_n(self):
        cov = pd.DataFrame(np.eye(4) * 0.04, index=list("abcd"), columns=list("abcd"))
        w = pd.Series([1.0, 0.0, 0.0, 0.0], index=list("abcd"))
        out = compute_portfolio_risk_metrics(w, cov)
        assert out["effective_n"] == pytest.approx(1.0)


class TestValidation:
    def test_bad_construction_rejected(self):
        with pytest.raises(ValueError):
            MultiStrategyAllocator(max_strategy_weight=0)
        with pytest.raises(ValueError):
            MultiStrategyAllocator(rebalance_frequency=0)

    def test_empty_strategy_returns_rejected(self):
        alloc = MultiStrategyAllocator()
        with pytest.raises(ValueError):
            alloc.compute_target_portfolio({}, pd.DataFrame(), pd.DataFrame())


class TestIntegration:
    def test_three_strategies_five_instruments_each(self):
        strategies = ["momentum", "mean_reversion", "carry"]
        rs = _strat_returns(strategies, instruments_per_strategy=5, seed=42)
        sig, bet = _signals_and_sizes(rs)
        alloc = MultiStrategyAllocator(
            strategy_optimizer="hrp",
            instrument_optimizer="hrp",
            max_strategy_weight=0.40,
            max_instrument_weight=0.08,
            max_gross_exposure=1.50,
        )
        tgt = alloc.compute_target_portfolio(rs, sig, bet)
        assert len(tgt) == 15
        assert set(tgt["strategy"].unique()) == set(strategies)
        assert (tgt["target_weight"].abs() <= 0.08 + 1e-9).all()
        assert tgt["target_weight"].abs().sum() <= 1.50 + 1e-9
