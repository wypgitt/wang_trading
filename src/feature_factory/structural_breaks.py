"""
Structural Break Features (AFML Ch. 17)

Continuous-valued features that quantify how "broken" a series looks at each
point in time — used as inputs to the ML meta-labeler rather than as event
triggers. Complements `src.data_engine.bars.cusum_filter`, which uses the
same CUSUM ideas but for sampling (binary event decisions).

Features:
    - cusum_statistic : rolling CUSUM magnitude over a trailing window
    - sadf_test       : Supremum ADF (right-tail explosiveness detector)
    - gsadf_test      : Generalized SADF (varies both window endpoints)
    - chow_test       : rolling Chow F-test for a change in trend slope
    - compute_structural_break_features : convenience aggregator

The SADF/GSADF tests follow Phillips, Shi, Yu (2015). Large positive t-stats
indicate explosive (bubble-like) behavior in the trailing data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# CUSUM feature
# ---------------------------------------------------------------------------

def cusum_statistic(series: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling CUSUM magnitude as a continuous feature.

    For each bar t, compute the positive and negative one-sided CUSUM paths
    of the *demeaned* log-return sequence over the trailing ``window`` bars
    and return max(sup_t S_pos, -inf_t S_neg). The larger the value, the
    stronger the evidence of a mean shift inside the window.

    NOTE: this is a feature, not a filter. Unlike
    ``src.data_engine.bars.cusum_filter``, we do *not* reset the CUSUM on
    trigger — the whole trailing window is summarized by its max excursion.

    Args:
        series: Price series (will be converted to log-returns internally).
        window: Trailing window length in bars.

    Returns:
        pd.Series: CUSUM magnitude aligned to ``series`` index. The first
                   ``window`` observations (insufficient history) are NaN.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    n = len(series)
    out = np.full(n, np.nan)
    if n < window + 1:
        return pd.Series(out, index=series.index, name="cusum_stat")

    # log-returns; prepend a 0 so indexing aligns with the price series.
    prices = series.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.concatenate([[0.0], np.log(prices[1:] / prices[:-1])])
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    for t in range(window, n):
        # Use raw log-returns (no demeaning): the feature should respond to
        # persistent drift, which the AFML CUSUM filter also tracks. Demeaning
        # within the window would cancel a sustained drift and miss exactly
        # the regime change we want to detect.
        r = log_ret[t - window + 1 : t + 1]
        s_pos = 0.0
        s_neg = 0.0
        max_excursion = 0.0
        for x in r:
            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)
            ex = max(s_pos, -s_neg)
            if ex > max_excursion:
                max_excursion = ex
        out[t] = max_excursion

    return pd.Series(out, index=series.index, name="cusum_stat")


# ---------------------------------------------------------------------------
# SADF / GSADF
# ---------------------------------------------------------------------------

def _safe_adf_tstat(x: np.ndarray, max_lag: int) -> float:
    """Return ADF t-statistic with a fixed lag; NaN on degenerate input."""
    if len(x) < max_lag + 4:
        return np.nan
    if np.std(x) == 0.0:
        return np.nan
    try:
        result = adfuller(x, maxlag=max_lag, autolag=None, regression="c")
        return float(result[0])
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def sadf_test(
    series: pd.Series,
    min_window: int = 20,
    max_lag: int = 1,
    fast: bool = True,
) -> pd.Series:
    """
    Supremum Augmented Dickey-Fuller statistic (right-tail explosiveness).

    For each time t, runs ADF on trailing windows of size w in
    [min_window, t] and returns the supremum ADF t-statistic over those
    windows. Large positive values indicate explosive (bubble-like) behavior.

    In the naive formulation this is O(n^2) ADF calls per series, which is
    prohibitive for long series. With ``fast=True`` (default), the window
    sizes are subsampled logarithmically (up to ~12 sizes per t), keeping
    runtime < ~5s on 1000-point series while still detecting bubbles.

    Args:
        series:     Input price/level series.
        min_window: Minimum window length for ADF.
        max_lag:    ADF lag order (fixed, not AIC-selected).
        fast:       If True, subsample window sizes; otherwise exhaustive.

    Returns:
        pd.Series: Supremum ADF t-stat per bar. First ``min_window`` entries
                   are NaN.
    """
    if min_window < 5:
        raise ValueError("min_window must be >= 5 for ADF to be defined")

    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)

    for t in range(min_window, n):
        if fast:
            # Up to ~12 log-spaced window sizes, including both endpoints.
            n_sizes = min(12, t - min_window + 1)
            window_sizes = np.unique(
                np.geomspace(min_window, t, num=n_sizes).astype(int)
            )
        else:
            window_sizes = np.arange(min_window, t + 1)

        best = -np.inf
        for w in window_sizes:
            start = t - w + 1
            stat = _safe_adf_tstat(values[start : t + 1], max_lag=max_lag)
            if np.isfinite(stat) and stat > best:
                best = stat
        out[t] = best if np.isfinite(best) else np.nan

    return pd.Series(out, index=series.index, name="sadf_stat")


def gsadf_test(
    series: pd.Series,
    min_window: int = 20,
    max_lag: int = 1,
    max_starts: int = 8,
    max_ends: int = 8,
) -> pd.Series:
    """
    Generalized SADF (Phillips, Shi, Yu 2015).

    For each time t, computes the supremum ADF t-statistic over all (r1, r2)
    sub-windows of [0, t] with r2 - r1 >= min_window. This detects multiple
    bubbles by scanning both endpoints rather than only the start.

    A naive implementation is O(n^3) and infeasible. This version subsamples
    both the start points r1 and the end points r2 to at most ``max_starts``
    and ``max_ends`` per t, so runtime scales as O(n * max_starts * max_ends).

    Args:
        series:     Input series.
        min_window: Minimum sub-window length.
        max_lag:    ADF lag order.
        max_starts: Max number of start-point samples per t.
        max_ends:   Max number of end-point samples per t (r2 in [min_window, t]).

    Returns:
        pd.Series: GSADF t-stat per bar.
    """
    if min_window < 5:
        raise ValueError("min_window must be >= 5")

    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)

    for t in range(min_window, n):
        # Sample end points r2 in [min_window-1, t]
        r2_candidates = np.unique(
            np.linspace(min_window - 1, t, num=max_ends).astype(int)
        )

        best = -np.inf
        for r2 in r2_candidates:
            max_starts_t = min(max_starts, r2 - min_window + 2)
            if max_starts_t <= 0:
                continue
            r1_candidates = np.unique(
                np.linspace(0, r2 - min_window + 1, num=max_starts_t).astype(int)
            )
            for r1 in r1_candidates:
                stat = _safe_adf_tstat(values[r1 : r2 + 1], max_lag=max_lag)
                if np.isfinite(stat) and stat > best:
                    best = stat
        out[t] = best if np.isfinite(best) else np.nan

    return pd.Series(out, index=series.index, name="gsadf_stat")


# ---------------------------------------------------------------------------
# Rolling Chow test
# ---------------------------------------------------------------------------

def chow_test(series: pd.Series, min_period: int = 30) -> pd.Series:
    """
    Rolling Chow F-test for a change in linear-trend parameters at time t.

    At each candidate break point t with at least ``min_period`` observations
    on both sides, fits two separate OLS trend lines y = a + b*x to the left
    and right sub-samples, and one pooled model to the full sample. The Chow
    statistic is:

        F = ((RSS_pooled - RSS_split) / k) / (RSS_split / (n - 2k))

    with k = 2 (intercept + slope). Large F means the slope/intercept
    meaningfully differ between sub-samples — evidence of a structural break.

    The p-value is computed against an F(k, n - 2k) distribution and reported
    via ``chow_stat`` (the F-stat itself is returned as the feature).

    Args:
        series:     Input series.
        min_period: Minimum observations required on each side of the break.

    Returns:
        pd.Series: Chow F-statistic per candidate break point. Entries
                   within ``min_period`` of either endpoint are NaN.
    """
    if min_period < 2:
        raise ValueError("min_period must be >= 2")

    y = series.to_numpy(dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    if n < 2 * min_period + 1:
        return pd.Series(out, index=series.index, name="chow_stat")

    x = np.arange(n, dtype=float)

    # Pre-compute pooled RSS once; it depends only on the full sample.
    pooled_slope, pooled_intercept = np.polyfit(x, y, 1)
    rss_pooled = float(np.sum((y - (pooled_slope * x + pooled_intercept)) ** 2))

    k = 2  # parameters per segment
    for t in range(min_period, n - min_period):
        x1, y1 = x[:t], y[:t]
        x2, y2 = x[t:], y[t:]

        # Guard: degenerate segments (constant x or y) skip the fit.
        if np.std(y1) == 0.0 or np.std(y2) == 0.0:
            continue

        s1, i1 = np.polyfit(x1, y1, 1)
        s2, i2 = np.polyfit(x2, y2, 1)
        rss_split = float(
            np.sum((y1 - (s1 * x1 + i1)) ** 2)
            + np.sum((y2 - (s2 * x2 + i2)) ** 2)
        )

        denom_df = n - 2 * k
        if rss_split <= 0 or denom_df <= 0:
            continue
        numerator = (rss_pooled - rss_split) / k
        denominator = rss_split / denom_df
        if denominator <= 0:
            continue
        f_stat = numerator / denominator
        out[t] = f_stat

    return pd.Series(out, index=series.index, name="chow_stat")


def chow_test_pvalue(f_stat: float, n: int, k: int = 2) -> float:
    """Helper: convert a Chow F-statistic to a p-value."""
    if not np.isfinite(f_stat) or n <= 2 * k:
        return np.nan
    return float(1.0 - stats.f.cdf(f_stat, dfn=k, dfd=n - 2 * k))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_structural_break_features(
    close_prices: pd.Series,
    window: int = 50,
    min_window_sadf: int = 20,
    min_period_chow: int = 30,
    include_gsadf: bool = False,
) -> pd.DataFrame:
    """
    Compute the full structural-break feature block as a single DataFrame.

    Args:
        close_prices:     Input price series.
        window:           CUSUM trailing window.
        min_window_sadf:  Minimum ADF window for SADF.
        min_period_chow:  Minimum side length for Chow.
        include_gsadf:    If True, also compute GSADF (slow).

    Returns:
        pd.DataFrame with columns cusum_stat, sadf_stat, chow_stat
        (+ gsadf_stat when requested), indexed like ``close_prices``.
    """
    logger.debug(
        f"structural_breaks: computing features on {len(close_prices)} bars "
        f"(gsadf={'on' if include_gsadf else 'off'})"
    )

    cols = {
        "cusum_stat": cusum_statistic(close_prices, window=window),
        "sadf_stat": sadf_test(close_prices, min_window=min_window_sadf),
        "chow_stat": chow_test(close_prices, min_period=min_period_chow),
    }
    if include_gsadf:
        cols["gsadf_stat"] = gsadf_test(close_prices, min_window=min_window_sadf)

    return pd.DataFrame(cols, index=close_prices.index)
