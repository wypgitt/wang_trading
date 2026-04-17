"""
Top-level multi-strategy allocator (design-doc §8.4 + §8.5).

Sits above the per-family HRP/risk-parity optimisers and orchestrates the
four-layer hierarchy:

    L1  allocate capital across signal families (strategy-level optimiser)
    L2  allocate within each family across instruments (instrument-level optimiser)
    L3  combine → signed instrument weights, scaled by meta-label bet sizes
    L4  clip to the design-doc §8.5 risk budget (single-position, single-
        family, gross-exposure, crypto cap, with an optional regime tilt)

The allocator is side-agnostic above L3 — signs come from the current signal
state. Regime tilts bump the weight of the family that matches the regime
("trending" → momentum / trend; "mean_reverting" → mean_reversion / stat_arb)
by a configurable fraction before renormalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.portfolio.factor_risk import FactorRiskModel
from src.portfolio.hrp import compute_hrp_weights
from src.portfolio.risk_parity import compute_risk_parity_weights


OptimizerName = Literal["hrp", "risk_parity", "equal_weight", "momentum_weighted"]


# ── allocator primitives ───────────────────────────────────────────────


def _run_optimizer(
    name: str,
    returns: pd.DataFrame,
) -> pd.Series:
    """Dispatch to the requested weight-producer; always sums to 1, ≥ 0."""
    if returns.shape[1] == 1:
        return pd.Series(1.0, index=returns.columns)
    if name == "equal_weight":
        return pd.Series(1.0 / returns.shape[1], index=returns.columns)
    if name == "momentum_weighted":
        mom = returns.tail(min(60, len(returns))).sum()
        w = mom.clip(lower=0)
        total = w.sum()
        if total <= 0:
            return pd.Series(1.0 / returns.shape[1], index=returns.columns)
        return w / total
    if name == "hrp":
        return compute_hrp_weights(returns)
    if name == "risk_parity":
        return compute_risk_parity_weights(returns.cov())
    raise ValueError(f"unknown optimizer {name!r}")


@dataclass
class MultiStrategyAllocator:
    strategy_optimizer: OptimizerName = "hrp"
    instrument_optimizer: OptimizerName = "hrp"
    max_strategy_weight: float = 0.30
    max_instrument_weight: float = 0.10
    max_gross_exposure: float = 1.50
    max_crypto_pct: float = 0.30
    rebalance_frequency: int = 5
    regime_boost: float = 0.20
    min_trade_fraction: float = 0.01
    asset_class_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in [
            ("max_strategy_weight", self.max_strategy_weight),
            ("max_instrument_weight", self.max_instrument_weight),
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_crypto_pct", self.max_crypto_pct),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be >= 1")

    # ── target portfolio ──────────────────────────────────────────────

    def compute_target_portfolio(
        self,
        strategy_returns: dict[str, pd.DataFrame],
        current_signals: pd.DataFrame,
        bet_sizes: pd.DataFrame,
        regime: str | None = None,
        prices: pd.Series | None = None,
        nav: float = 1.0,
        current_positions: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        if not strategy_returns:
            raise ValueError("strategy_returns cannot be empty")
        current_positions = current_positions or {}

        # L1 — strategy-level weights from each strategy's portfolio return
        strat_port_returns = pd.DataFrame(
            {
                name: df.mean(axis=1) for name, df in strategy_returns.items()
            }
        ).dropna(how="all")

        strat_weights = _run_optimizer(
            self.strategy_optimizer, strat_port_returns
        )

        # Optional regime tilt — bump the matching family then renormalise
        if regime is not None:
            strat_weights = self._apply_regime_tilt(strat_weights, regime)

        # L4 partial — cap per-strategy weight. Do NOT renormalise after
        # clipping: the cap is a hard ceiling, so the residual stays in
        # cash rather than being redistributed to other families that may
        # also want to be at the ceiling.
        strat_weights = strat_weights.clip(upper=self.max_strategy_weight)

        # L2 — within-strategy instrument weights
        rows: list[dict] = []
        latest_signals = current_signals.iloc[-1] if not current_signals.empty else pd.Series(dtype=float)
        latest_bets = bet_sizes.iloc[-1] if not bet_sizes.empty else pd.Series(dtype=float)

        for strat_name, returns_df in strategy_returns.items():
            if returns_df.empty:
                continue
            sw = float(strat_weights.get(strat_name, 0.0))
            if sw <= 0:
                continue

            within_w = _run_optimizer(self.instrument_optimizer, returns_df)

            for sym, iw in within_w.items():
                sign = np.sign(float(latest_signals.get(sym, 0.0)))
                bet_mult = abs(float(latest_bets.get(sym, 1.0)))
                raw = sw * float(iw) * bet_mult * sign
                rows.append(
                    {
                        "symbol": sym,
                        "strategy": strat_name,
                        "target_weight": raw,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "strategy",
                    "target_weight",
                    "target_shares",
                    "current_weight",
                    "trade_required",
                ]
            )

        df = pd.DataFrame(rows)

        # L4 — per-instrument cap (preserve sign)
        df["target_weight"] = df["target_weight"].clip(
            lower=-self.max_instrument_weight, upper=self.max_instrument_weight
        )

        # L4 — crypto cap
        if self.asset_class_map:
            is_crypto = df["symbol"].map(
                lambda s: self.asset_class_map.get(s) == "crypto"
            )
            crypto_gross = float(df.loc[is_crypto, "target_weight"].abs().sum())
            if crypto_gross > self.max_crypto_pct and crypto_gross > 0:
                scale = self.max_crypto_pct / crypto_gross
                df.loc[is_crypto, "target_weight"] *= scale

        # L4 — gross exposure cap
        gross = float(df["target_weight"].abs().sum())
        if gross > self.max_gross_exposure and gross > 0:
            df["target_weight"] *= self.max_gross_exposure / gross

        # shares + current weight enrichment
        if prices is not None:
            aligned_prices = df["symbol"].map(prices).astype(float)
            df["target_shares"] = np.where(
                aligned_prices > 0,
                df["target_weight"] * nav / aligned_prices.replace(0, np.nan),
                0.0,
            )
        else:
            df["target_shares"] = df["target_weight"]

        df["current_weight"] = df["symbol"].map(
            lambda s: float(current_positions.get(s, 0.0))
        )
        df["trade_required"] = (
            (df["target_weight"] - df["current_weight"]).abs()
            > self.min_trade_fraction
        )

        return df

    def _apply_regime_tilt(
        self, weights: pd.Series, regime: str
    ) -> pd.Series:
        regime_lc = regime.lower()
        targets: tuple[str, ...]
        if regime_lc in {"trending", "trend", "bull"}:
            targets = ("momentum", "trend")
        elif regime_lc in {"mean_reverting", "mean-reverting", "reverting", "range"}:
            targets = ("mean_reversion", "stat_arb", "arb")
        else:
            return weights
        boosted = weights.copy()
        for name in boosted.index:
            lname = str(name).lower()
            if any(t in lname for t in targets):
                boosted[name] *= 1.0 + self.regime_boost
        total = boosted.sum()
        if total > 0:
            boosted = boosted / total
        return boosted

    # ── rebalance translation ─────────────────────────────────────────

    def compute_rebalance_trades(
        self,
        target: pd.DataFrame,
        current_positions: dict[str, float],
        prices: pd.Series,
        nav: float,
    ) -> pd.DataFrame:
        """Diff current positions against target weights → required orders.

        ``current_positions`` maps symbol → signed *fraction of NAV*
        currently held. The minimum-trade filter is applied on the same
        scale: any delta below ``min_trade_fraction`` is suppressed.
        """
        if nav <= 0:
            raise ValueError("nav must be positive")

        target_map = dict(zip(target["symbol"], target["target_weight"]))
        symbols = set(target_map) | set(current_positions)

        rows: list[dict] = []
        for sym in symbols:
            tgt = float(target_map.get(sym, 0.0))
            cur = float(current_positions.get(sym, 0.0))
            delta = tgt - cur
            if abs(delta) < self.min_trade_fraction:
                continue
            price = float(prices.get(sym, float("nan")))
            if not np.isfinite(price) or price <= 0:
                continue
            notional = delta * nav
            side = "buy" if delta > 0 else "sell"
            if cur == 0 and tgt != 0:
                reason = "new_position"
            elif tgt == 0 and cur != 0:
                reason = "exit"
            else:
                reason = "rebalance"
            rows.append(
                {
                    "symbol": sym,
                    "side": side,
                    "size": abs(notional) / price,
                    "notional": abs(notional),
                    "reason": reason,
                }
            )
        return pd.DataFrame(rows)


# ── portfolio-level risk metrics ───────────────────────────────────────


def compute_portfolio_risk_metrics(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    factor_model: FactorRiskModel | None = None,
    holding_period_bars: int = 20,
    drawdown_z: float = 2.33,
) -> dict:
    w = weights.reindex(cov_matrix.index).fillna(0.0).to_numpy()
    Σ = cov_matrix.to_numpy()
    port_var = float(w @ Σ @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    individual_vols = np.sqrt(np.maximum(np.diag(Σ), 0.0))
    weighted_vol = float(np.abs(w) @ individual_vols)
    diversification_ratio = weighted_vol / port_vol if port_vol > 0 else 0.0

    effective_n = 1.0 / float((w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
    # approximate 1-tail VaR-style DD estimate
    max_drawdown_estimate = port_vol * np.sqrt(holding_period_bars) * drawdown_z

    out: dict = {
        "portfolio_volatility": port_vol,
        "weighted_asset_volatility": weighted_vol,
        "diversification_ratio": diversification_ratio,
        "effective_n": effective_n,
        "max_drawdown_estimate": float(max_drawdown_estimate),
        "gross_exposure": float(np.abs(w).sum()),
        "net_exposure": float(w.sum()),
    }

    if factor_model is not None and factor_model.factor_loadings is not None:
        out["factor_exposures"] = factor_model.get_factor_exposures(weights)
        out["risk_decomposition"] = factor_model.get_risk_decomposition(weights)

    return out
