"""
Volatility Features (Sinclair — Volatility Trading)

GARCH-based conditional volatility plus complementary realized-volatility
features. GARCH forecasts feed three downstream systems in the design doc:
triple-barrier widths (labeling), bet sizing, and ATR position sizing.

Features:
    - fit_garch                  : one-shot GARCH(p,q) fit, tolerant of failure
    - garch_volatility           : rolling GARCH conditional stdev with
                                   periodic re-fit + recursion in between
    - vol_term_structure         : RV(short) / RV(long) contango/backwardation
    - vol_of_vol                 : rolling std of conditional volatility
    - realized_vs_implied_spread : IV - RV
    - compute_volatility_features: convenience aggregator
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from loguru import logger

# Annualization factor for daily bars. For non-daily bars, downstream code can
# rescale; we default to daily because GARCH is typically applied to EOD data.
TRADING_DAYS_PER_YEAR = 252


def _log_returns(series: pd.Series) -> pd.Series:
    """Log returns aligned to ``series`` index; first entry is NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = series / series.shift(1)
    out = np.where(ratios > 0, np.log(ratios), np.nan)
    return pd.Series(out, index=series.index, name="log_returns")


# ---------------------------------------------------------------------------
# One-shot GARCH fit
# ---------------------------------------------------------------------------

def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> dict | None:
    """
    Fit a GARCH(p, q) model to a return series and return its parameters.

    Uses the ``arch`` package with a zero-mean specification (appropriate for
    bar-level returns, which are typically demeaned or near-zero). On
    optimizer failure, returns None and logs a warning.

    Args:
        returns: Return series (log or simple). NaNs are dropped.
        p:       ARCH order (lags of squared innovations).
        q:       GARCH order (lags of conditional variance).

    Returns:
        dict with keys ``omega``, ``alpha`` (list[float], length p),
        ``beta`` (list[float], length q), ``conditional_volatility``
        (pd.Series indexed like the cleaned returns), ``log_likelihood``,
        ``aic``, ``bic`` — or None on failure.
    """
    from arch import arch_model

    if p < 1 or q < 0:
        raise ValueError("p must be >= 1 and q must be >= 0")

    clean = returns.dropna()
    if len(clean) < max(20, 5 * (p + q + 1)):
        logger.warning(
            f"fit_garch: series too short ({len(clean)}) for GARCH({p},{q})"
        )
        return None

    # Degenerate input — variance is zero, GARCH is undefined.
    if clean.std(ddof=0) == 0.0:
        logger.warning("fit_garch: constant returns — GARCH is undefined")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                clean, mean="Zero", vol="GARCH", p=p, q=q, rescale=False
            )
            res = model.fit(disp="off", show_warning=False)
    except Exception as exc:  # noqa: BLE001 — arch can raise various exceptions
        logger.warning(f"fit_garch: optimizer failed: {exc}")
        return None

    params = dict(res.params)
    # arch names parameters "omega", "alpha[1]"..., "beta[1]"...
    try:
        omega = float(params["omega"])
        alpha = [float(params[f"alpha[{i}]"]) for i in range(1, p + 1)]
        beta = [float(params[f"beta[{i}]"]) for i in range(1, q + 1)] if q else []
    except KeyError as exc:
        logger.warning(f"fit_garch: unexpected parameter layout: {exc}")
        return None

    cond_vol = pd.Series(
        np.asarray(res.conditional_volatility), index=clean.index,
        name="conditional_volatility",
    )

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "conditional_volatility": cond_vol,
        "log_likelihood": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }


# ---------------------------------------------------------------------------
# Rolling GARCH volatility (fit + recurse)
# ---------------------------------------------------------------------------

def garch_volatility(
    close: pd.Series,
    window: int = 252,
    p: int = 1,
    q: int = 1,
    refit_interval: int = 20,
) -> pd.Series:
    """
    Rolling GARCH conditional volatility with periodic re-fit.

    Fits GARCH(p, q) on the trailing ``window`` returns every
    ``refit_interval`` bars; between fits, extends the conditional variance
    forward using the GARCH recursion

        sigma2_t = omega + sum_i alpha_i * r_{t-i}^2 + sum_j beta_j * sigma2_{t-j}

    For ``window=252, refit_interval=20`` this cuts cost by ~20x versus
    re-fitting every bar while keeping forecasts well-calibrated.

    Args:
        close:          Close price series.
        window:         Trailing returns used for each GARCH fit.
        p, q:           GARCH orders.
        refit_interval: Bars between re-fits. 1 = refit every bar (slow).

    Returns:
        pd.Series: Conditional standard deviation per bar, aligned to
                   ``close``. First ``window`` entries are NaN.
    """
    if window < 30:
        raise ValueError("window must be >= 30 for GARCH to be identified")
    if refit_interval < 1:
        raise ValueError("refit_interval must be >= 1")
    if p < 1 or q < 0:
        raise ValueError("p must be >= 1 and q must be >= 0")

    log_ret = _log_returns(close).to_numpy()
    n = len(close)
    sigma = np.full(n, np.nan)
    sigma2 = np.full(n, np.nan)

    if n <= window:
        return pd.Series(sigma, index=close.index, name="garch_vol")

    # State carried across bars between re-fits.
    omega = 0.0
    alpha: list[float] = []
    beta: list[float] = []
    bars_since_fit = refit_interval  # force a fit at the first eligible bar

    for t in range(window, n):
        if bars_since_fit >= refit_interval or not alpha:
            window_returns = pd.Series(log_ret[t - window + 1 : t + 1])
            fit = fit_garch(window_returns, p=p, q=q)
            if fit is not None:
                omega = fit["omega"]
                alpha = fit["alpha"]
                beta = fit["beta"]
                # Seed sigma for the current bar from the last fitted value.
                # Earlier bars keep their prior state (NaN during warmup, or
                # previously-recursed values in the post-warmup region). This
                # preserves the standard "first ``window`` entries NaN"
                # convention without throwing away the GARCH anchor.
                cv_last = float(fit["conditional_volatility"].iloc[-1])
                sigma[t] = cv_last
                sigma2[t] = cv_last**2
                bars_since_fit = 0
                continue
            # If the fit failed and we have no previous parameters, skip.
            if not alpha:
                continue

        # Recurse one step: sigma2_t = omega + sum a_i r_{t-i}^2 + sum b_j s2_{t-j}.
        # For the first recursion step after a fit the sigma2_{t-j} slots are
        # either the fit's last value (just set above) or zero if j reaches
        # into the pre-warmup region — we substitute the unconditional
        # variance omega/(1-alpha-beta) in that case.
        val = omega
        sum_ab = sum(alpha) + sum(beta)
        uncond_var = omega / max(1e-12, 1.0 - sum_ab) if sum_ab < 1.0 else omega
        for i, a in enumerate(alpha, start=1):
            r_prev = log_ret[t - i] if t - i >= 0 else 0.0
            if np.isfinite(r_prev):
                val += a * (r_prev**2)
        for j, b in enumerate(beta, start=1):
            s_prev = sigma2[t - j] if t - j >= 0 else np.nan
            if not np.isfinite(s_prev):
                s_prev = uncond_var
            val += b * s_prev
        if not np.isfinite(val) or val <= 0:
            bars_since_fit += 1
            continue
        sigma2[t] = val
        sigma[t] = np.sqrt(val)
        bars_since_fit += 1

    return pd.Series(sigma, index=close.index, name="garch_vol")


# ---------------------------------------------------------------------------
# Realized-vol features
# ---------------------------------------------------------------------------

def realized_volatility(
    close: pd.Series,
    window: int,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Rolling realized vol = std of log-returns (annualized by default)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    lr = _log_returns(close)
    rv = lr.rolling(window=window, min_periods=window).std()
    if annualize:
        rv = rv * np.sqrt(periods_per_year)
    return rv.rename(f"rv_{window}")


def vol_term_structure(
    close: pd.Series,
    short_window: int = 5,
    long_window: int = 30,
) -> pd.Series:
    """
    Rolling short-vol to long-vol ratio: RV(short) / RV(long).

    > 1 : short-term vol exceeds long-term vol (spike / inverted curve).
    < 1 : quiet market; short-term vol compressed vs baseline.
    """
    if short_window < 2 or long_window <= short_window:
        raise ValueError("need 2 <= short_window < long_window")
    rv_short = realized_volatility(close, window=short_window, annualize=False)
    rv_long = realized_volatility(close, window=long_window, annualize=False)
    # Annualization cancels in the ratio; skip it.
    ratio = rv_short / rv_long.replace(0.0, np.nan)
    return ratio.rename("vol_term_structure")


def vol_of_vol(conditional_vol: pd.Series, window: int = 30) -> pd.Series:
    """Rolling standard deviation of the conditional volatility series."""
    if window < 2:
        raise ValueError("window must be >= 2")
    return (
        conditional_vol.rolling(window=window, min_periods=window)
        .std()
        .rename("vol_of_vol")
    )


def realized_vs_implied_spread(
    realized_vol: pd.Series, implied_vol: pd.Series,
) -> pd.Series:
    """IV - RV (the volatility risk premium). Positive → short-vol opportunity."""
    return (implied_vol - realized_vol).rename("rv_iv_spread")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_volatility_features(
    close: pd.Series,
    implied_vol: pd.Series | None = None,
    window: int = 252,
    refit_interval: int = 20,
    short_window: int = 5,
    long_window: int = 30,
    vvol_window: int = 30,
) -> pd.DataFrame:
    """
    Compute GARCH + realized vol features as a DataFrame.

    Always includes garch_vol, vol_term_structure, vol_of_vol; if
    ``implied_vol`` is provided, also includes rv_iv_spread (using the
    ``long_window`` realized vol as RV).
    """
    logger.debug(
        f"volatility: computing features on {len(close)} bars "
        f"(window={window}, refit={refit_interval})"
    )

    garch = garch_volatility(
        close, window=window, refit_interval=refit_interval
    )
    cols: dict[str, pd.Series] = {
        "garch_vol": garch,
        "vol_term_structure": vol_term_structure(
            close, short_window=short_window, long_window=long_window
        ),
        "vol_of_vol": vol_of_vol(garch, window=vvol_window),
    }
    if implied_vol is not None:
        rv_long = realized_volatility(close, window=long_window)
        cols["rv_iv_spread"] = realized_vs_implied_spread(rv_long, implied_vol)

    return pd.DataFrame(cols, index=close.index)
