"""
Risk parity (equal risk contribution) portfolio optimiser — Isichenko,
design-doc §8.4.3.

Each asset's *risk contribution* to the portfolio is

    RC_i = w_i · (Σ w)_i / σ_p²

which sums to one across assets. Risk parity chooses weights so every RC_i
matches a target budget (equal by default). This treats correlated assets
as a combined risk block and spreads capital according to how much risk
each asset actually brings — unlike equal-weight, which over-allocates to
high-vol names, and unlike mean-variance, which hinges on fragile expected
returns.

For diagonal covariances the closed-form solution is inverse-*volatility*
(``w_i ∝ 1/σ_i``) — inverse-variance is an HRP leaf-level heuristic, not the
RP answer. The optimiser below reproduces that closed form when Σ is
diagonal and handles general correlated covariances via SLSQP.

Per design-doc §8.4.3 the live system runs both HRP and RP and picks the
allocator with the better trailing 6-month risk-adjusted performance;
:meth:`RiskParityOptimizer.compare_with_hrp` is the building block for that
selector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.portfolio.hrp import compute_hrp_weights


def marginal_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """MRC_i = (Σ w)_i / σ_p."""
    w = weights.reindex(cov_matrix.index).fillna(0.0).to_numpy()
    Σw = cov_matrix.to_numpy() @ w
    portfolio_vol = float(np.sqrt(max(w @ Σw, 0.0)))
    if portfolio_vol == 0:
        return pd.Series(0.0, index=cov_matrix.index)
    return pd.Series(Σw / portfolio_vol, index=cov_matrix.index, name="mrc")


def risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Percentage contribution of each asset to portfolio risk — sums to 1."""
    w = weights.reindex(cov_matrix.index).fillna(0.0).to_numpy()
    Σw = cov_matrix.to_numpy() @ w
    portfolio_var = float(w @ Σw)
    if portfolio_var <= 0:
        return pd.Series(0.0, index=cov_matrix.index)
    rc = w * Σw / portfolio_var
    return pd.Series(rc, index=cov_matrix.index, name="rc")


def compute_risk_parity_weights(
    cov_matrix: pd.DataFrame,
    budget: pd.Series | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> pd.Series:
    """Solve the risk-parity optimisation (SLSQP under the hood).

    Objective: minimise ``Σ_i (RC_i − budget_i)²`` subject to Σw = 1, w ≥ 0.
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pd.DataFrame")
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be square")

    assets = list(cov_matrix.columns)
    n = len(assets)
    if n < 2:
        raise ValueError("need at least 2 assets")

    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        b = budget.reindex(assets).fillna(0.0).to_numpy()
        if (b < 0).any():
            raise ValueError("budget entries must be non-negative")
        total = b.sum()
        if total <= 0:
            raise ValueError("budget must sum to a positive number")
        b = b / total

    Σ = cov_matrix.to_numpy()

    # Coordinate-descent fixed point (Griveau-Billion, Richard, Roncalli 2013).
    # Each iteration solves, asset by asset, the quadratic
    #
    #     Σ_ii · w_i² + c_i · w_i − b_i = 0         with c_i = (Σw)_i − Σ_ii w_i
    #
    # whose positive root satisfies the stationarity condition
    # ``w_i · (Σw)_i = b_i`` required for RC_i = b_i / Σb. Unconditionally
    # convergent for PSD Σ and respects the positive orthant by construction.

    vols = np.sqrt(np.maximum(np.diag(Σ), 1e-12))
    w = (1.0 / vols)
    w = w / w.sum()

    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            Σ_ii = Σ[i, i]
            if Σ_ii <= 0:
                continue
            c_i = float(Σ[i] @ w - Σ_ii * w[i])
            # positive root of Σ_ii·x² + c_i·x − b_i = 0
            disc = c_i * c_i + 4.0 * Σ_ii * b[i]
            w[i] = (-c_i + np.sqrt(max(disc, 0.0))) / (2.0 * Σ_ii)

        # Convergence check on absolute RC deviation
        Σw = Σ @ w
        pvar = float(w @ Σw)
        if pvar > 0:
            rc = w * Σw / pvar
            if float(np.max(np.abs(rc - b))) < tol:
                break
        if np.max(np.abs(w - w_prev)) < tol:
            break

    w_opt = w / w.sum()

    return pd.Series(w_opt, index=assets, name="weight")


# ── stateful optimiser + HRP comparator ────────────────────────────────


@dataclass
class RiskParityOptimizer:
    lookback: int = 252
    rebalance_frequency: int = 5

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be >= 1")

    def get_weights(self, returns: pd.DataFrame) -> pd.Series:
        if returns.shape[1] < 2:
            raise ValueError("need at least 2 assets")
        window = returns.tail(self.lookback)
        cov = window.cov()
        return compute_risk_parity_weights(cov)

    def compare_with_hrp(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Side-by-side weights + risk contributions for HRP vs RP."""
        window = returns.tail(self.lookback)
        cov = window.cov()
        rp_weights = compute_risk_parity_weights(cov)
        hrp_weights = compute_hrp_weights(window).reindex(rp_weights.index)

        rp_rc = risk_contribution(rp_weights, cov)
        hrp_rc = risk_contribution(hrp_weights, cov)

        return pd.DataFrame(
            {
                "hrp_weight": hrp_weights,
                "risk_parity_weight": rp_weights,
                "hrp_rc": hrp_rc,
                "risk_parity_rc": rp_rc,
            }
        )
