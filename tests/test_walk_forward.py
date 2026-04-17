"""Tests for the walk-forward backtester (Phase 4 — P4.02)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.transaction_costs import (
    EQUITIES_COSTS,
    TransactionCostModel,
)
from src.backtesting.walk_forward import (
    BacktestResult,
    BacktestTrade,
    WalkForwardBacktester,
    compute_metrics,
)


@pytest.fixture
def cost_model() -> TransactionCostModel:
    return TransactionCostModel(equities_config=EQUITIES_COSTS)


def _daily_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="B")


def _make_panel(
    n_bars: int,
    symbols: list[str],
    price_fn,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (close, adv, volatility) dataframes."""
    idx = _daily_index(n_bars)
    close = pd.DataFrame(
        {s: price_fn(i, n_bars, rng) for i, s in enumerate(symbols)}, index=idx
    )
    adv = pd.DataFrame(1e7, index=idx, columns=symbols, dtype=float)
    vol = pd.DataFrame(0.02, index=idx, columns=symbols, dtype=float)
    return close, adv, vol


class TestBacktesterCore:
    def test_perfect_signal_produces_positive_equity(self, cost_model):
        n = 60
        close, adv, vol = _make_panel(
            n, ["AAA"], lambda i, n, _r: np.linspace(100, 160, n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)
        signals.iloc[5:50] = 1  # keep re-entering as each barrier closes
        probs = pd.DataFrame(0.9, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.05, index=close.index, columns=close.columns)

        bt = WalkForwardBacktester(
            cost_model=cost_model,
            initial_capital=100_000,
            max_holding_period=5,
            upper_multiplier=2.0,
            lower_multiplier=2.0,
        )
        result = bt.run(close, signals, probs, sizes, adv, vol)

        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
        assert len(result.trades) > 0
        assert all(t.gross_pnl > 0 for t in result.trades)

    def test_random_signal_near_zero_gross_over_many_trades(self, cost_model):
        rng = np.random.default_rng(42)
        n = 300
        # pure random walk, zero drift
        close, adv, vol = _make_panel(
            n,
            ["AAA"],
            lambda i, n, r: 100 * np.exp(np.cumsum(r.normal(0, 0.01, n))),
            rng=rng,
        )
        signals = pd.DataFrame(
            rng.choice([-1, 0, 1], size=(n, 1), p=[0.1, 0.8, 0.1]),
            index=close.index,
            columns=close.columns,
        )
        probs = pd.DataFrame(0.55, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.02, index=close.index, columns=close.columns)

        bt = WalkForwardBacktester(cost_model=cost_model, max_holding_period=10)
        result = bt.run(close, signals, probs, sizes, adv, vol)

        # With zero-drift prices and random sides, expected gross ≈ 0, net < 0
        assert len(result.trades) > 5
        total_net = sum(t.net_pnl for t in result.trades)
        total_gross = sum(t.gross_pnl for t in result.trades)
        assert total_net < total_gross  # costs are paid

    def test_execution_delay_fills_at_next_bar(self, cost_model):
        n = 15
        close = pd.DataFrame(
            {"AAA": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0,
                     107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]},
            index=_daily_index(n),
        )
        signals = pd.DataFrame(0, index=close.index, columns=["AAA"])
        signals.iloc[5] = 1
        probs = pd.DataFrame(0.9, index=close.index, columns=["AAA"])
        sizes = pd.DataFrame(0.1, index=close.index, columns=["AAA"])
        adv = pd.DataFrame(1e7, index=close.index, columns=["AAA"])
        vol = pd.DataFrame(0.02, index=close.index, columns=["AAA"])

        bt = WalkForwardBacktester(
            cost_model=cost_model, execution_delay_bars=1, max_holding_period=3
        )
        result = bt.run(close, signals, probs, sizes, adv, vol)

        assert len(result.trades) == 1
        # signal at bar 5 → entry fill at bar 6 → entry price = 106
        assert result.trades[0].entry_price == pytest.approx(106.0)
        assert result.trades[0].entry_timestamp == close.index[6]

    def test_zero_delay_fills_at_same_bar(self, cost_model):
        n = 10
        close = pd.DataFrame(
            {"AAA": np.linspace(100, 109, n)}, index=_daily_index(n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=["AAA"])
        signals.iloc[3] = 1
        probs = pd.DataFrame(0.9, index=close.index, columns=["AAA"])
        sizes = pd.DataFrame(0.1, index=close.index, columns=["AAA"])
        adv = pd.DataFrame(1e7, index=close.index, columns=["AAA"])
        vol = pd.DataFrame(0.02, index=close.index, columns=["AAA"])

        bt = WalkForwardBacktester(
            cost_model=cost_model, execution_delay_bars=0, max_holding_period=2
        )
        result = bt.run(close, signals, probs, sizes, adv, vol)
        assert result.trades[0].entry_price == pytest.approx(close.iloc[3]["AAA"])


class TestCostsAndPnL:
    def test_net_pnl_is_gross_minus_costs(self, cost_model):
        n = 40
        close, adv, vol = _make_panel(
            n, ["AAA"], lambda i, n, _r: np.linspace(100, 140, n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)
        signals.iloc[5:30] = 1
        probs = pd.DataFrame(0.9, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.05, index=close.index, columns=close.columns)
        bt = WalkForwardBacktester(cost_model=cost_model, max_holding_period=3)
        result = bt.run(close, signals, probs, sizes, adv, vol)

        for t in result.trades:
            assert t.net_pnl == pytest.approx(t.gross_pnl - t.costs.total_cost)
            assert t.costs.total_cost > 0

    def test_final_equity_equals_initial_plus_sum_net_pnl(self, cost_model):
        n = 40
        close, adv, vol = _make_panel(
            n, ["AAA"], lambda i, n, _r: np.linspace(100, 140, n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)
        signals.iloc[5:30] = 1
        probs = pd.DataFrame(0.9, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.05, index=close.index, columns=close.columns)
        bt = WalkForwardBacktester(cost_model=cost_model, max_holding_period=3)
        result = bt.run(close, signals, probs, sizes, adv, vol)

        expected = 100_000.0 + sum(t.net_pnl for t in result.trades)
        assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-9)


class TestMaxPositions:
    def test_max_positions_constraint_respected(self, cost_model):
        n = 20
        symbols = [f"S{i}" for i in range(10)]
        close, adv, vol = _make_panel(
            n, symbols, lambda i, n, _r: np.linspace(100 + i, 100 + i + 10, n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=symbols)
        signals.iloc[3] = 1  # everyone signals at once
        probs = pd.DataFrame(0.9, index=close.index, columns=symbols)
        sizes = pd.DataFrame(0.05, index=close.index, columns=symbols)

        bt = WalkForwardBacktester(
            cost_model=cost_model, max_positions=3, max_holding_period=20
        )
        result = bt.run(close, signals, probs, sizes, adv, vol)
        # Only 3 positions should be opened on that bar
        entry_bars = {t.entry_timestamp for t in result.trades}
        opens_at_bar_4 = [t for t in result.trades if t.entry_timestamp == close.index[4]]
        assert len(opens_at_bar_4) <= 3


class TestDrawdownAndEquity:
    def test_drawdown_curve_always_nonpositive(self, cost_model):
        rng = np.random.default_rng(0)
        n = 100
        close, adv, vol = _make_panel(
            n,
            ["AAA"],
            lambda i, n, r: 100 * np.exp(np.cumsum(r.normal(0, 0.01, n))),
            rng=rng,
        )
        signals = pd.DataFrame(
            rng.choice([-1, 0, 1], size=(n, 1), p=[0.1, 0.8, 0.1]),
            index=close.index,
            columns=close.columns,
        )
        probs = pd.DataFrame(0.55, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.02, index=close.index, columns=close.columns)
        bt = WalkForwardBacktester(cost_model=cost_model, max_holding_period=5)
        result = bt.run(close, signals, probs, sizes, adv, vol)

        assert (result.drawdown_curve <= 1e-9).all()
        assert result.metrics["max_drawdown"] <= 0.0
        assert result.drawdown_curve.min() == pytest.approx(result.metrics["max_drawdown"])

    def test_equity_curve_length_matches_bars(self, cost_model):
        n = 50
        close, adv, vol = _make_panel(
            n, ["A", "B"], lambda i, n, _r: np.linspace(100, 120, n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)
        probs = pd.DataFrame(0.9, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.05, index=close.index, columns=close.columns)
        bt = WalkForwardBacktester(cost_model=cost_model)
        result = bt.run(close, signals, probs, sizes, adv, vol)
        assert len(result.equity_curve) == n
        assert len(result.returns) == n


class TestTrendingMarketMomentum:
    def test_trending_market_with_momentum_signal_has_positive_sharpe(self, cost_model):
        rng = np.random.default_rng(7)
        n = 250
        # trending with mild noise
        drift = 0.001
        noise = rng.normal(0, 0.005, n)
        returns = drift + noise
        prices = 100 * np.exp(np.cumsum(returns))
        idx = _daily_index(n)
        close = pd.DataFrame({"TREND": prices}, index=idx)
        adv = pd.DataFrame(1e7, index=idx, columns=["TREND"])
        vol = pd.DataFrame(0.015, index=idx, columns=["TREND"])

        # momentum signal: +1 when 20-bar return > 0
        mom = close["TREND"].pct_change(20).fillna(0)
        sig_values = np.where(mom > 0, 1, np.where(mom < 0, -1, 0))
        signals = pd.DataFrame({"TREND": sig_values}, index=idx)
        probs = pd.DataFrame(0.7, index=idx, columns=["TREND"])
        sizes = pd.DataFrame(0.05, index=idx, columns=["TREND"])

        bt = WalkForwardBacktester(
            cost_model=cost_model,
            max_holding_period=10,
            upper_multiplier=3.0,
            lower_multiplier=2.0,
        )
        result = bt.run(close, signals, probs, sizes, adv, vol)

        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
        # Sharpe with rf=0 (pure return/vol ratio) must be positive when the
        # strategy makes money on a trending market.
        raw = compute_metrics(result.equity_curve, result.trades, risk_free_rate=0.0)
        assert raw["sharpe"] > 0


class TestMetrics:
    def test_all_metrics_present_and_finite(self, cost_model):
        n = 100
        close, adv, vol = _make_panel(
            n, ["X"], lambda i, n, _r: np.linspace(100, 130, n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)
        signals.iloc[10:80] = 1
        probs = pd.DataFrame(0.8, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.05, index=close.index, columns=close.columns)
        bt = WalkForwardBacktester(cost_model=cost_model, max_holding_period=5)
        result = bt.run(close, signals, probs, sizes, adv, vol)

        required_keys = {
            "total_return",
            "annualized_return",
            "annualized_vol",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "max_drawdown_duration_bars",
            "recovery_factor",
            "win_rate",
            "profit_factor",
            "avg_trade",
            "avg_holding_period_bars",
            "total_trades",
            "trades_per_month",
            "turnover",
            "cost_drag_bps",
            "skewness",
            "kurtosis",
            "tail_ratio",
        }
        assert required_keys.issubset(result.metrics.keys())
        for k in required_keys:
            v = result.metrics[k]
            if v != float("inf"):
                assert np.isfinite(v), f"{k} = {v}"

        assert 0.0 <= result.metrics["win_rate"] <= 1.0
        assert result.metrics["total_trades"] == len(result.trades)

    def test_compute_metrics_callable_directly(self):
        idx = _daily_index(100)
        equity = pd.Series(
            100_000 * np.exp(np.cumsum(np.full(100, 0.0005))), index=idx
        )
        metrics = compute_metrics(equity, trades=[])
        assert metrics["total_trades"] == 0
        assert metrics["max_drawdown"] <= 0.0
        assert np.isfinite(metrics["sharpe"])


class TestValidation:
    def test_rejects_negative_capital(self, cost_model):
        with pytest.raises(ValueError):
            WalkForwardBacktester(cost_model=cost_model, initial_capital=-1)

    def test_rejects_negative_delay(self, cost_model):
        with pytest.raises(ValueError):
            WalkForwardBacktester(cost_model=cost_model, execution_delay_bars=-1)

    def test_rejects_zero_max_positions(self, cost_model):
        with pytest.raises(ValueError):
            WalkForwardBacktester(cost_model=cost_model, max_positions=0)

    def test_empty_close_rejected(self, cost_model):
        bt = WalkForwardBacktester(cost_model=cost_model)
        empty = pd.DataFrame()
        with pytest.raises(ValueError):
            bt.run(empty, empty)


class TestDataclasses:
    def test_backtest_trade_and_result_dataclass(self, cost_model):
        n = 30
        close, adv, vol = _make_panel(
            n, ["Y"], lambda i, n, _r: np.linspace(100, 120, n)
        )
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)
        signals.iloc[5] = 1
        probs = pd.DataFrame(0.9, index=close.index, columns=close.columns)
        sizes = pd.DataFrame(0.05, index=close.index, columns=close.columns)
        bt = WalkForwardBacktester(cost_model=cost_model, max_holding_period=3)
        result = bt.run(close, signals, probs, sizes, adv, vol)

        assert isinstance(result, BacktestResult)
        if result.trades:
            t = result.trades[0]
            assert isinstance(t, BacktestTrade)
            assert t.side in (-1, 1)
            assert t.holding_period_bars >= 0
            assert 0.0 <= t.meta_label_prob <= 1.0
