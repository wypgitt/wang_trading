"""
Deflated Sharpe Ratio — Bailey & López de Prado (2014), AFML Ch. 11.

The observed Sharpe ratio of the *best* strategy out of N trials is upwardly
biased: even N strategies with a true Sharpe of zero will, by luck, produce a
"best" Sharpe that looks impressive. The DSR removes that selection-bias
haircut and asks a tighter question:

    Under H0 (true Sharpe is zero) and accounting for
    (i) that this strategy was picked from N,
    (ii) that returns are non-normal (skew + kurtosis),
    how surprising is the observed Sharpe?

Two-step calculation
--------------------
1. *Expected max Sharpe under H0* (Bailey & López de Prado, eq. 13–14):

       E[max(SR)] ≈ σ_SR · [(1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))]

   with γ = Euler–Mascheroni constant. σ_SR is the cross-sectional std of
   Sharpes across the trial pool; when unavailable, 1.0 is the standard
   default (the Sharpe itself is already scaled).

2. *Deflated Sharpe statistic* (Mertens 2002 / Bailey & LdP eq. 9):

       σ_SR_hat = sqrt((1 − γ₃·SR + (γ₄ − 1)/4 · SR²) / T)
       DSR      = (SR_obs − E[max(SR)]) / σ_SR_hat
       p        = 1 − Φ(DSR)

   where γ₃ is skewness, γ₄ is the non-excess kurtosis (3.0 for Gaussian),
   and T is the number of return observations.

Gate 2 of the §9 validation pipeline requires ``p < 0.05``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from src.backtesting.walk_forward import BacktestResult


EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(
    n_trials: int,
    sharpe_std: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Expected maximum Sharpe ratio from ``n_trials`` iid trials under H0.

    The skew/kurt arguments are accepted for API symmetry with
    :func:`deflated_sharpe_ratio` — the Bailey & López de Prado E[max]
    formula itself depends only on ``sharpe_std`` and ``n_trials``; the
    non-normality correction is applied in the DSR denominator, not here.
    """

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sharpe_std < 0:
        raise ValueError("sharpe_std must be non-negative")
    del skewness, kurtosis  # see docstring

    if n_trials == 1:
        return 0.0

    z_high = norm.ppf(1.0 - 1.0 / n_trials)
    z_low = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(
        sharpe_std * ((1.0 - EULER_MASCHERONI) * z_high + EULER_MASCHERONI * z_low)
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_std: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> tuple[float, float]:
    """Return ``(dsr_statistic, p_value)`` for the Bailey-LdP hypothesis test."""

    if n_observations < 2:
        raise ValueError("n_observations must be >= 2")

    expected_max = expected_max_sharpe(n_trials, sharpe_std)

    var_term = (
        1.0
        - skewness * observed_sharpe
        + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    # Mertens adjustment can go negative when kurtosis is very small and
    # skewness is strongly positive against the Sharpe sign; clamp to a
    # tiny positive so the sqrt is well-defined.
    var_term = max(var_term, 1e-12)
    std_corrected = math.sqrt(var_term / n_observations)

    dsr = (observed_sharpe - expected_max) / std_corrected
    p_value = float(1.0 - norm.cdf(dsr))
    return float(dsr), p_value


def compute_dsr_from_backtest(
    backtest_result: BacktestResult,
    n_trials: int,
    sharpe_std: float = 1.0,
) -> dict:
    """Extract SR/skew/kurt from a single backtest and run the DSR test.

    ``metrics["kurtosis"]`` is populated by :func:`compute_metrics` with
    *excess* (Fisher) kurtosis (Gaussian = 0), so we add 3 before feeding
    it into the Pearson-convention formula above.
    """

    m = backtest_result.metrics
    observed_sharpe = float(m.get("sharpe", 0.0))
    skewness = float(m.get("skewness", 0.0))
    kurt_excess = float(m.get("kurtosis", 0.0))
    kurtosis = kurt_excess + 3.0
    n_obs = max(len(backtest_result.returns.dropna()), 2)

    dsr_stat, p_value = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        sharpe_std=sharpe_std,
        n_trials=n_trials,
        n_observations=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    return {
        "observed_sharpe": observed_sharpe,
        "expected_max_sharpe": expected_max_sharpe(n_trials, sharpe_std),
        "dsr_statistic": dsr_stat,
        "p_value": p_value,
        "n_trials": n_trials,
        "n_observations": n_obs,
        "sharpe_std": sharpe_std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "passed": bool(p_value < 0.05),
    }


def compute_dsr_from_cpcv(
    cpcv_results: list[BacktestResult],
    n_total_trials: int,
) -> dict:
    """DSR using mean Sharpe across CPCV paths; each path counts as one obs.

    ``sharpe_std`` is taken directly from the cross-path dispersion of
    Sharpes — a sharper σ_SR estimate than the default 1.0 baseline used by
    :func:`compute_dsr_from_backtest`. Skewness and kurtosis are pooled
    across all paths' per-bar returns.
    """

    if not cpcv_results:
        raise ValueError("cpcv_results is empty")

    sharpes = np.array(
        [r.metrics.get("sharpe", 0.0) for r in cpcv_results], dtype=float
    )
    observed_sharpe = float(sharpes.mean())
    sharpe_std = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 1.0
    if sharpe_std == 0.0:
        sharpe_std = 1.0  # degenerate — fall back to baseline

    pooled_returns = np.concatenate(
        [r.returns.dropna().to_numpy() for r in cpcv_results]
    )
    if len(pooled_returns) >= 3:
        skewness = float(_sample_skew(pooled_returns))
        kurt_excess = float(_sample_kurt(pooled_returns))
    else:
        skewness = 0.0
        kurt_excess = 0.0
    kurtosis = kurt_excess + 3.0

    n_obs = len(cpcv_results)
    dsr_stat, p_value = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        sharpe_std=sharpe_std,
        n_trials=n_total_trials,
        n_observations=max(n_obs, 2),
        skewness=skewness,
        kurtosis=kurtosis,
    )

    return {
        "observed_sharpe": observed_sharpe,
        "expected_max_sharpe": expected_max_sharpe(n_total_trials, sharpe_std),
        "dsr_statistic": dsr_stat,
        "p_value": p_value,
        "n_trials": n_total_trials,
        "n_observations": n_obs,
        "sharpe_std": sharpe_std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "path_sharpes": sharpes.tolist(),
        "passed": bool(p_value < 0.05),
    }


# ── small sample moments (avoid extra deps just for this) ───────────────────


def _sample_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0 or n < 3:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma**3)


def _sample_kurt(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher convention — Gaussian returns 0)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0 or n < 4:
        return 0.0
    return float(((x - mu) ** 4).mean() / sigma**4 - 3.0)
