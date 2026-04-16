"""
Fractional Differentiation (AFML Ch. 5)

Integer differencing (returns) achieves stationarity but destroys long-range
memory. No differencing preserves memory but leaves a non-stationary series.
Fractional differencing finds the minimum order d in [0, 1] that achieves
stationarity while preserving maximum memory.

This module implements the Fixed-Width Window Fracdiff (FFD) method: weights
from the binomial expansion are truncated once their magnitude falls below a
threshold, producing a fixed-window convolution with deterministic lookback.

Usage:
    from src.feature_factory.fractional_diff import frac_diff_ffd, find_min_d

    d_star = find_min_d(price_series)
    stationary = frac_diff_ffd(price_series, d=d_star)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute binomial-series weights for fractional differencing order ``d``.

    Uses the recurrence w_k = -w_{k-1} * (d - k + 1) / k, starting with w_0 = 1.
    Weights are truncated once |w_k| falls below ``threshold``.

    Args:
        d:         Fractional differencing order (typically in [0, 1]).
        threshold: Minimum absolute weight retained in the series.

    Returns:
        np.ndarray: Weights ordered [w_0, w_1, ..., w_K] with w_0 = 1.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        # Safety cap: with a positive threshold the recurrence must terminate,
        # but guard against pathological inputs (e.g., d exactly integer > 0
        # causes w_k to become exactly 0 once k > d; break in that case too).
        if w_k == 0.0:
            break

    return np.asarray(weights, dtype=float)


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply Fixed-Width Window fractional differencing to a series.

    For each time t, computes X_t^d = sum_{k=0}^{K} w_k * X_{t-k}, where K is
    determined by the weight truncation threshold. Values at the start of the
    series that lack a full window are dropped from the output.

    Args:
        series:    Input series (prices, volumes, etc.). Must be numeric.
        d:         Fractional differencing order.
        threshold: Weight truncation threshold passed to ``get_weights_ffd``.

    Returns:
        pd.Series: FFD-transformed series, aligned to the input index. The
                   leading (K-1) observations are dropped; NaNs in the input
                   propagate into the window and are dropped from the output.
    """
    weights = get_weights_ffd(d, threshold=threshold)
    width = len(weights)

    if series.empty or len(series) < width:
        return pd.Series(dtype=float, index=series.index[:0], name=series.name)

    values = series.to_numpy(dtype=float)
    nan_mask = np.isnan(values)

    # Vectorised sliding dot product via scipy.signal.lfilter.
    #
    # The FFD recurrence is X_t^d = sum_{k=0..width-1} w_k * X_{t-k}, which
    # is exactly a direct-form FIR filter with numerator coefficients
    # ``weights`` and denominator [1.0]. lfilter runs in compiled C; this
    # replaces the inner-Python loop and is ~50× faster on long series.
    # NaNs are handled by (a) zeroing them before the filter so they don't
    # poison subsequent outputs and (b) masking any window that touched a
    # NaN via a rolling NaN-count convolution.
    from scipy.signal import lfilter

    clean = np.where(nan_mask, 0.0, values)
    filtered = lfilter(weights, [1.0], clean)
    out = np.full(len(values), np.nan, dtype=float)
    out[width - 1 :] = filtered[width - 1 :]

    if nan_mask.any():
        # nan_count[t] = number of NaNs in the window [t-width+1, t].
        nan_count_valid = np.convolve(
            nan_mask.astype(float), np.ones(width), mode="valid",
        )
        # Align to the full-length output axis.
        bad = np.zeros(len(values), dtype=bool)
        bad[width - 1 :] = nan_count_valid > 0
        out[bad] = np.nan

    result = pd.Series(out, index=series.index, name=series.name)
    return result.dropna()


def find_min_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    p_value: float = 0.05,
    threshold: float = 1e-5,
) -> float:
    """
    Find the minimum ``d`` that makes ``series`` stationary under an ADF test.

    Scans d from ``d_range[0]`` up to ``d_range[1]`` in increments of ``step``
    and returns the first d whose FFD-differenced series rejects the ADF null
    of a unit root at the given ``p_value``. Falls back to 1.0 if no d in the
    grid achieves stationarity.

    Args:
        series:    Input series.
        d_range:   (low, high) bounds for the d search, inclusive.
        step:      Grid step size.
        p_value:   ADF p-value threshold for stationarity.
        threshold: Weight truncation threshold passed through to FFD.

    Returns:
        float: Smallest d in the grid achieving stationarity, else 1.0.
    """
    lo, hi = d_range
    if lo < 0 or hi < lo:
        raise ValueError(f"invalid d_range={d_range}")
    if step <= 0:
        raise ValueError("step must be positive")

    # Build the grid inclusively; np.arange can drop the right endpoint due
    # to float drift, so use a small fudge factor.
    n_steps = int(round((hi - lo) / step)) + 1
    grid = [round(lo + i * step, 10) for i in range(n_steps)]

    for d in grid:
        diffed = frac_diff_ffd(series, d=d, threshold=threshold)
        # ADF needs enough points and non-constant data to be meaningful.
        if len(diffed) < 20 or diffed.std(ddof=0) == 0:
            continue
        try:
            _, pval, *_ = adfuller(diffed.values, autolag="AIC")
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug(f"ADF failed at d={d}: {exc}")
            continue
        if pval < p_value:
            return float(d)

    return 1.0


def frac_diff_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_d: float = 1.0,
    p_value: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Apply FFD column-wise to a DataFrame, finding the optimal d per column.

    For each column, ``find_min_d`` searches ``[0, max_d]`` for the smallest
    d that produces a stationary series (ADF p < ``p_value``). That d is
    applied via ``frac_diff_ffd``. The resulting DataFrame is aligned on the
    intersection of the per-column FFD outputs, so all columns share a common
    index (rows where every transformed column has a value).

    Args:
        df:       Input DataFrame.
        columns:  Columns to transform. If None, all numeric columns are used.
        max_d:    Upper bound of the d search.
        p_value:  ADF stationarity threshold.

    Returns:
        (pd.DataFrame, dict): Transformed DataFrame and {column: optimal_d}.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    d_map: dict[str, float] = {}
    transformed: dict[str, pd.Series] = {}

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in DataFrame")
        series = df[col]
        d_star = find_min_d(series, d_range=(0.0, max_d), p_value=p_value)
        d_map[col] = d_star
        transformed[col] = frac_diff_ffd(series, d=d_star)

    if not transformed:
        return pd.DataFrame(index=df.index[:0]), d_map

    # Align on the intersection so each row has all transformed values.
    result = pd.concat(transformed, axis=1).dropna(how="any")
    return result, d_map
