"""
Layer 1 of the Bet-Sizing cascade — AFML Ch. 10 probability→size.

Converts a calibrated meta-labeler probability into a bet size via the AFML
z-score formulation:

    z(p)    = (p − 0.5) / sqrt(p · (1 − p))
    size(p) = (2 · Φ(z(p)) − 1) · max_size

(where Φ is the standard-normal CDF). Properties:

    * ``p = 0.5`` → size = 0 (no bet).
    * ``p → 1``   → size → max_size (concave — returns diminish near certainty).
    * Function is concave in p: marginal gains from confidence fall off,
      naturally limiting exposure on barely-positive signals.

Sign convention for this codebase (*differs from AFML*): in AFML's original
framework ``p < 0.5`` produces a NEGATIVE size, i.e. the model takes the
OPPOSITE side. Here the Signal Battery already commits to a direction (the
``side`` column), and the meta-labeler answers a binary yes/no: "given
side=+1, will this trade be profitable?" So ``p < 0.5`` means "no, skip this
trade" — NOT "reverse it". We therefore clip the raw AFML size to
``[0, max_size]`` so sub-50%-probability signals collapse to zero size.
Caller multiplies by ``side`` afterwards (see :func:`bet_size_with_side`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


# Default discretisation — matches the design-doc recommendation
# (20%-step tiers from 0% to 100% of max_size).
_DEFAULT_TIERS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# ---------------------------------------------------------------------------
# Probability → size
# ---------------------------------------------------------------------------

def bet_size_from_probability(
    prob: float | np.ndarray,
    max_size: float = 1.0,
) -> float | np.ndarray:
    """
    Map meta-labeler probability to a bet size in ``[0, max_size]``.

    Uses the AFML Ch. 10 inverse-normal-CDF mapping parameterised by the
    sample z-statistic ``(p − 0.5) / sqrt(p(1−p))``. For a single trial this
    is the standardised deviation of a binomial-with-probability-p from
    "fair coin" (p = 0.5). The higher the z, the higher the bet.

    Args:
        prob:     Calibrated probability (scalar or array-like) in ``[0, 1]``.
        max_size: Cap applied after the CDF mapping. Defaults to 1.0 so the
                  function returns a fraction-of-budget that downstream
                  layers (Kelly, vol-adj, risk budget) can further scale.

    Returns:
        Same shape as ``prob`` (scalar → float, array → ndarray).
    """
    if max_size < 0:
        raise ValueError(f"max_size must be >= 0 (got {max_size})")

    p_arr = np.asarray(prob, dtype=float)
    if not np.all((p_arr >= 0.0) & (p_arr <= 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    # Clip away from 0 and 1 to avoid the division-by-zero at the endpoints;
    # at p=0 or p=1 the result saturates to ±max_size anyway.
    eps = 1e-12
    p_clip = np.clip(p_arr, eps, 1.0 - eps)
    z = (p_clip - 0.5) / np.sqrt(p_clip * (1.0 - p_clip))
    raw = (2.0 * stats.norm.cdf(z) - 1.0) * max_size
    clipped = np.clip(raw, 0.0, max_size)

    # Preserve the caller's scalar-vs-array expectation.
    if np.ndim(prob) == 0:
        return float(clipped)
    return np.asarray(clipped)


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def discretize_bet_size(
    continuous_size: float,
    tiers: Sequence[float] | None = None,
) -> float:
    """
    Round a continuous bet size to the nearest tier.

    Reduces portfolio turnover when the meta-labeler's probability wobbles
    around a decision boundary — without discretisation a 0.51 → 0.49
    probability flicker could produce a constant stream of small resizing
    trades. The default tier grid is {0, 20, 40, 60, 80, 100}% of max_size.

    Args:
        continuous_size: The raw (possibly already-clipped) size.
        tiers:           Monotonic non-negative tier list; defaults to
                         ``_DEFAULT_TIERS``. Ties resolve to the first
                         (lower) tier — standard ``numpy.argmin`` behaviour.
    """
    tier_list = tuple(tiers) if tiers is not None else _DEFAULT_TIERS
    if not tier_list:
        raise ValueError("tiers must be non-empty")
    arr = np.asarray(tier_list, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(continuous_size))))
    return float(arr[idx])


# ---------------------------------------------------------------------------
# Signed position size
# ---------------------------------------------------------------------------

def bet_size_with_side(
    prob: float,
    side: int,
    max_size: float = 1.0,
) -> float:
    """
    Apply the primary-model direction on top of the AFML size.

    Because :func:`bet_size_from_probability` clips to ``[0, max_size]``,
    the returned value has magnitude ∈ ``[0, max_size]`` and sign exactly
    equal to ``side``. ``side = 0`` returns 0 (neutral signal).
    """
    if side not in (-1, 0, 1):
        raise ValueError(f"side must be -1, 0, or +1 (got {side})")
    if side == 0:
        return 0.0
    magnitude = bet_size_from_probability(prob, max_size=max_size)
    return float(side) * float(magnitude)
