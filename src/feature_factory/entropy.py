"""
Entropy Features (AFML Ch. 18)

Rolling entropy estimators over a returns series. Low entropy indicates
predictable / regular behavior (tradeable patterns); high entropy indicates
randomness.

Features:
    - shannon_entropy    : discretized (quantile-binned) return entropy
    - lempel_ziv_entropy : LZ76 complexity on a binarized return sequence
    - approx_entropy     : Approximate Entropy (Pincus 1991)
    - sample_entropy     : Sample Entropy (Richman & Moorman 2000)
    - compute_entropy_features : convenience aggregator

Implementation notes:
    ApEn and SampEn are both O(N^2) per window and are vectorized via numpy
    broadcasting. With window ~100 this is a few milliseconds per window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Log-return helper (robust to zeros/negatives and leading NaN)
# ---------------------------------------------------------------------------

def _log_returns(series: pd.Series) -> np.ndarray:
    """Log returns aligned to ``series`` index; first entry is NaN."""
    prices = series.to_numpy(dtype=float)
    out = np.full(len(prices), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = prices[1:] / prices[:-1]
        out[1:] = np.where(ratios > 0, np.log(ratios), np.nan)
    return out


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def shannon_entropy(
    series: pd.Series,
    n_bins: int = 10,
    window: int = 100,
) -> pd.Series:
    """
    Rolling Shannon entropy of quantile-binned log-returns.

    Quantile bin edges are computed ONCE over the full return series so that
    the bins partition the overall distribution evenly. For each trailing
    ``window``-bar slice we then count how samples fall across those fixed
    bins and return H = -sum(p_i * log2(p_i)).

    The rationale: computing quantile edges per-window would produce a
    constant H = log2(n_bins) by construction (equal-frequency bins are
    uniform within the sample that defined them). Fixed global edges instead
    measure how *narrow* the local distribution is relative to the full
    history — a strongly trending window concentrates into a handful of bins
    and yields low entropy, while a random window spreads across all bins
    and approaches log2(n_bins).

    Args:
        series: Price series.
        n_bins: Number of quantile bins.
        window: Trailing window in bars.

    Returns:
        pd.Series: Shannon entropy per bar; NaN before enough history.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if window < 2:
        raise ValueError("window must be >= 2")

    log_ret = _log_returns(series)
    n = len(series)
    out = np.full(n, np.nan)

    valid = log_ret[np.isfinite(log_ret)]
    if len(valid) < n_bins:
        return pd.Series(out, index=series.index, name="shannon_entropy")

    # Global equal-frequency (quantile) bin edges, open at the endpoints so
    # samples outside the observed range still get binned.
    edges = np.quantile(valid, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Guard against degenerate edges (all returns identical).
    if not np.all(np.diff(edges) > 0):
        return pd.Series(out, index=series.index, name="shannon_entropy")

    for t in range(window, n):
        r = log_ret[t - window + 1 : t + 1]
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        counts, _ = np.histogram(r, bins=edges)
        total = counts.sum()
        if total == 0:
            continue
        p = counts[counts > 0] / total
        out[t] = float(-np.sum(p * np.log2(p)))

    return pd.Series(out, index=series.index, name="shannon_entropy")


# ---------------------------------------------------------------------------
# Lempel-Ziv entropy (LZ76)
# ---------------------------------------------------------------------------

def _lz76_complexity(bits: str) -> int:
    """
    LZ76 complexity c(n): count the number of phrases produced when parsing
    ``bits`` left-to-right, where each phrase is the shortest substring not
    previously seen in the parsed prefix.

    For i.i.d. uniform binary strings, c(n) * log2(n) / n → 1 as n → ∞.
    For periodic strings (e.g. 0101...), c(n) stays very small.
    """
    n = len(bits)
    if n == 0:
        return 0
    i = 0
    c = 0
    while i < n:
        # Grow l until the new phrase is not a substring of what we've seen.
        l = 1
        while i + l <= n and bits[i : i + l] in bits[: i + l - 1]:
            l += 1
        c += 1
        i += l
    return c


def lempel_ziv_entropy(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling LZ76 entropy estimate.

    For each trailing ``window``-bar slice of log-returns, binarizes into
    "1" (positive return) / "0" (non-positive) and returns the Kontoyiannis-style
    normalized complexity h = c(n) * log2(n) / n, which lies in roughly
    [0, 1+] and tends to 1 for i.i.d. uniform binary noise.

    Args:
        series: Price series.
        window: Trailing window in bars (must be >= 2).

    Returns:
        pd.Series: Normalized LZ entropy per bar.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    log_ret = _log_returns(series)
    n = len(series)
    out = np.full(n, np.nan)
    log2_n = np.log2(max(window, 2))

    for t in range(window, n):
        r = log_ret[t - window + 1 : t + 1]
        # NaN (e.g. non-positive prices) treated as 0 for the binarization.
        bits = "".join("1" if (np.isfinite(x) and x > 0) else "0" for x in r)
        c = _lz76_complexity(bits)
        out[t] = c * log2_n / len(bits)

    return pd.Series(out, index=series.index, name="lz_entropy")


# ---------------------------------------------------------------------------
# Approximate Entropy (Pincus 1991)
# ---------------------------------------------------------------------------

def _apen_single(u: np.ndarray, m: int, r: float) -> float:
    N = len(u)
    if N < m + 1 or r <= 0:
        return np.nan

    def phi(mm: int) -> float:
        n_vec = N - mm + 1
        if n_vec < 2:
            return np.nan
        # Build (n_vec, mm) template matrix.
        X = np.lib.stride_tricks.sliding_window_view(u, mm)[:n_vec]
        # Pairwise Chebyshev distances; self-matches are included for ApEn.
        D = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        C = np.sum(D <= r, axis=1) / n_vec
        # Guard against log(0) — C >= 1/n_vec because of self-match.
        return float(np.mean(np.log(C)))

    return phi(m) - phi(m + 1)


def approx_entropy(
    series: pd.Series,
    m: int = 2,
    r_mult: float = 0.2,
    window: int = 100,
) -> pd.Series:
    """
    Rolling Approximate Entropy on log-returns.

    Args:
        series: Price series.
        m:      Embedding dimension.
        r_mult: Tolerance multiplier; r = r_mult * std(window_returns).
        window: Trailing window length.

    Returns:
        pd.Series: ApEn per bar.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if window < m + 2:
        raise ValueError(f"window ({window}) must be >= m+2 ({m + 2})")

    log_ret = _log_returns(series)
    n = len(series)
    out = np.full(n, np.nan)

    for t in range(window, n):
        r = log_ret[t - window + 1 : t + 1]
        r = r[np.isfinite(r)]
        if len(r) < m + 2:
            continue
        tol = r_mult * float(np.std(r))
        if tol <= 0:
            continue
        out[t] = _apen_single(r, m=m, r=tol)

    return pd.Series(out, index=series.index, name="approx_entropy")


# ---------------------------------------------------------------------------
# Sample Entropy (Richman & Moorman 2000)
# ---------------------------------------------------------------------------

def _sampen_single(u: np.ndarray, m: int, r: float) -> float:
    N = len(u)
    if N < m + 2 or r <= 0:
        return np.nan

    # N - m vectors each of length m (and m+1), enabling identical pair counts.
    n_vec = N - m
    if n_vec < 2:
        return np.nan

    X_m = np.lib.stride_tricks.sliding_window_view(u, m)[:n_vec]
    X_m1 = np.lib.stride_tricks.sliding_window_view(u, m + 1)[:n_vec]

    D_m = np.max(np.abs(X_m[:, None, :] - X_m[None, :, :]), axis=2)
    D_m1 = np.max(np.abs(X_m1[:, None, :] - X_m1[None, :, :]), axis=2)

    # Count pairs with i < j and distance <= r (no self-matches).
    iu = np.triu_indices(n_vec, k=1)
    B = int(np.sum(D_m[iu] <= r))
    A = int(np.sum(D_m1[iu] <= r))

    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


def sample_entropy(
    series: pd.Series,
    m: int = 2,
    r_mult: float = 0.2,
    window: int = 100,
) -> pd.Series:
    """
    Rolling Sample Entropy — bias-corrected ApEn that excludes self-matches.

    Same arguments as ``approx_entropy``. Lower SampEn indicates more regular
    behavior; higher SampEn indicates more random.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if window < m + 2:
        raise ValueError(f"window ({window}) must be >= m+2 ({m + 2})")

    log_ret = _log_returns(series)
    n = len(series)
    out = np.full(n, np.nan)

    for t in range(window, n):
        r = log_ret[t - window + 1 : t + 1]
        r = r[np.isfinite(r)]
        if len(r) < m + 2:
            continue
        tol = r_mult * float(np.std(r))
        if tol <= 0:
            continue
        out[t] = _sampen_single(r, m=m, r=tol)

    return pd.Series(out, index=series.index, name="sample_entropy")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_entropy_features(
    close_prices: pd.Series,
    window: int = 100,
    n_bins_shannon: int = 10,
    include_apen: bool = False,
) -> pd.DataFrame:
    """
    Compute the standard entropy feature block.

    Args:
        close_prices:   Input prices.
        window:         Trailing window for all entropy features.
        n_bins_shannon: Number of quantile bins for Shannon entropy.
        include_apen:   Whether to compute Approximate Entropy (off by default
                        because SampEn is strictly preferred).

    Returns:
        pd.DataFrame with columns shannon_entropy, lz_entropy, sample_entropy
        (+ approx_entropy if requested).
    """
    logger.debug(
        f"entropy: computing features on {len(close_prices)} bars "
        f"(window={window}, apen={'on' if include_apen else 'off'})"
    )
    cols = {
        "shannon_entropy": shannon_entropy(
            close_prices, n_bins=n_bins_shannon, window=window
        ),
        "lz_entropy": lempel_ziv_entropy(close_prices, window=window),
        "sample_entropy": sample_entropy(close_prices, window=window),
    }
    if include_apen:
        cols["approx_entropy"] = approx_entropy(close_prices, window=window)
    return pd.DataFrame(cols, index=close_prices.index)
