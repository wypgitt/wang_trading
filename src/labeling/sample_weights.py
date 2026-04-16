"""
Sample Weights and Uniqueness (AFML Ch. 4)

Triple-barrier labels are **not** i.i.d. — two events whose label periods
overlap share price information, so naively treating them as independent
samples over-counts evidence and inflates measured accuracy. AFML Ch. 4
patches this with four complementary weighting mechanisms that this module
assembles into a single ``sample_weight`` vector consumed by the meta-labeler
and the purged-CV evaluator:

    1. **Average uniqueness** — down-weights samples whose label period
       overlaps with many concurrent labels.
    2. **Sequential bootstrap** — replaces i.i.d. bootstrap with a draw that
       respects temporal overlap (AFML Algorithm 4.5.2).
    3. **Return attribution** — scales weights by the absolute return earned
       during the label period so high-impact bars matter more.
    4. **Time decay** — biases weight toward recent samples so the model
       tracks the prevailing regime.

See design doc §6.4 (Sample Weights and Uniqueness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers: resolving DatetimeIndex-valued event boundaries to integer positions
# ---------------------------------------------------------------------------

def _resolve_positions(
    index: pd.Index,
    timestamps: pd.Index,
    *,
    name: str,
) -> np.ndarray:
    """Map ``timestamps`` to integer positions in ``index``; raise on misses."""
    positions = np.asarray(index.get_indexer(timestamps))
    if (positions < 0).any():
        missing = timestamps[positions < 0]
        raise ValueError(
            f"{name}: {len(missing)} timestamps not found in reference index "
            f"(first offender: {missing[0]})"
        )
    return positions


# ---------------------------------------------------------------------------
# 1. Concurrent-events count
# ---------------------------------------------------------------------------

def get_num_concurrent_events(
    close_index: pd.DatetimeIndex,
    event_starts: pd.DatetimeIndex,
    event_ends: pd.DatetimeIndex,
) -> pd.Series:
    """
    Count how many labeled events are alive at each bar.

    An event with range ``[start, end]`` is alive at every bar in that closed
    interval. Computed in O(N + T) via a difference-array trick (increment at
    ``start``, decrement at ``end + 1``, cumulative sum) rather than the naive
    O(N·T) inner-loop sweep.

    Args:
        close_index:   Bar clock (all bars that could host an event).
        event_starts:  Timestamps at which events fire.
        event_ends:    Label-period expiry per event (e.g. ``exit_timestamp``
                       from :func:`src.labeling.triple_barrier.make_labels`).

    Returns:
        pd.Series indexed by ``close_index``: concurrent event count per bar.
    """
    if len(event_starts) != len(event_ends):
        raise ValueError(
            f"event_starts ({len(event_starts)}) and event_ends "
            f"({len(event_ends)}) must have the same length"
        )
    n = len(close_index)
    if n == 0:
        return pd.Series([], dtype=np.int64, index=close_index,
                         name="concurrent_events")

    if len(event_starts) == 0:
        return pd.Series(np.zeros(n, dtype=np.int64), index=close_index,
                         name="concurrent_events")

    starts = _resolve_positions(close_index, event_starts, name="event_starts")
    ends = _resolve_positions(close_index, event_ends, name="event_ends")
    if (ends < starts).any():
        raise ValueError("every event_end must be >= its event_start")

    # Difference array: +1 at start, -1 one bar past end; cumulative sum yields
    # the count at every bar in O(N + T).
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends + 1, -1)
    counts = np.cumsum(delta[:-1])
    return pd.Series(counts, index=close_index, name="concurrent_events")


# ---------------------------------------------------------------------------
# 2. Average uniqueness per event
# ---------------------------------------------------------------------------

def get_average_uniqueness(
    event_starts: pd.DatetimeIndex,
    event_ends: pd.DatetimeIndex,
    close_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Mean 1/concurrent_count over the bars each event is alive.

    An event entirely unique to its label period has uniqueness 1.0; two events
    that fully overlap each score ~0.5; three mutually concurrent events each
    score ~0.33. The cumulative-sum trick gives O(N + T) for all N events.

    Returns:
        pd.Series indexed by ``event_starts`` with values in (0, 1].
    """
    if len(event_starts) == 0:
        return pd.Series([], dtype=float, index=event_starts, name="uniqueness")

    concurrent = get_num_concurrent_events(close_index, event_starts, event_ends)
    # Reciprocal; bars with zero concurrent events (shouldn't happen where
    # events live, but guard) contribute zero rather than infinity.
    c = concurrent.to_numpy().astype(float)
    reciprocal = np.where(c > 0, 1.0 / np.maximum(c, 1.0), 0.0)
    cumsum = np.concatenate([[0.0], np.cumsum(reciprocal)])

    starts = _resolve_positions(close_index, event_starts, name="event_starts")
    ends = _resolve_positions(close_index, event_ends, name="event_ends")
    lengths = (ends - starts + 1).astype(float)
    sums = cumsum[ends + 1] - cumsum[starts]
    avg_u = sums / np.maximum(lengths, 1.0)
    return pd.Series(avg_u, index=event_starts, name="uniqueness")


# ---------------------------------------------------------------------------
# 3. Sequential bootstrap (AFML Algorithm 4.5.2)
# ---------------------------------------------------------------------------

def sequential_bootstrap(
    uniqueness: pd.Series,
    event_starts: pd.DatetimeIndex | None = None,
    event_ends: pd.DatetimeIndex | None = None,
    close_index: pd.DatetimeIndex | None = None,
    n_samples: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    AFML Algorithm 4.5.2 — bootstrap that respects label overlap.

    The standard bootstrap assumes i.i.d. samples, which fails here: two
    overlapping labels are correlated, so a standard draw that happens to
    pick them both double-counts the same underlying evidence. The sequential
    bootstrap fixes this by updating draw probabilities after each pick so
    that subsequent draws preferentially avoid samples that overlap with
    already-picked ones.

    Args:
        uniqueness:    Average-uniqueness series from :func:`get_average_uniqueness`.
                       Its index defines the candidate sample set.
        event_starts,
        event_ends,
        close_index:   When all three are supplied, the function runs the full
                       AFML algorithm — at each draw ``j`` it recomputes
                       conditional uniqueness
                           ū_i^{φ_j} = mean over t ∈ [s_i, e_i] of 1 / (1 + Σ_{k∈φ} 1_{t,k})
                       and samples i ∝ ū_i^{φ_j}. The cumulative-sum trick
                       keeps each draw O(T + N), so 1000 draws over 1000 events
                       runs in well under a second.
                       If any of the three is ``None``, the function falls
                       back to probability-weighted resampling using
                       ``uniqueness`` as initial weights — cheaper but does
                       **not** update probabilities after each draw.
        n_samples:     Number of draws. Defaults to ``len(uniqueness)``.
        random_state:  Seed for reproducibility.

    Returns:
        np.ndarray of length ``n_samples`` holding integer indices into
        ``uniqueness`` (positions, 0-based). May contain duplicates.
    """
    if len(uniqueness) == 0:
        return np.empty(0, dtype=np.int64)
    if n_samples is None:
        n_samples = len(uniqueness)
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")

    rng = np.random.default_rng(random_state)

    full_info = (
        event_starts is not None
        and event_ends is not None
        and close_index is not None
    )

    if not full_info:
        logger.debug(
            "sequential_bootstrap: falling back to uniqueness-weighted "
            "resampling (no event ranges supplied)"
        )
        weights = uniqueness.to_numpy().astype(float)
        weights = np.where(weights > 0, weights, 0.0)
        total = weights.sum()
        if total <= 0:
            # Degenerate — return uniform draws.
            return rng.integers(0, len(uniqueness), size=n_samples)
        probs = weights / total
        return rng.choice(len(uniqueness), size=n_samples, replace=True, p=probs)

    # ------------------------------------------------------------------
    # Full AFML Algorithm 4.5.2 — narrow the optional types locally.
    # ------------------------------------------------------------------
    assert event_starts is not None  # pragma: no cover — guarded by full_info
    assert event_ends is not None    # pragma: no cover
    assert close_index is not None   # pragma: no cover

    if len(event_starts) != len(uniqueness) or len(event_ends) != len(uniqueness):
        raise ValueError(
            "event_starts/event_ends must align 1:1 with uniqueness"
        )

    starts = _resolve_positions(close_index, event_starts, name="event_starts")
    ends = _resolve_positions(close_index, event_ends, name="event_ends")
    if (ends < starts).any():
        raise ValueError("every event_end must be >= its event_start")

    T = len(close_index)
    N = len(uniqueness)
    lengths = (ends - starts + 1).astype(float)
    # ``overlap[t]`` = count of already-drawn samples that cover bar t.
    overlap = np.zeros(T, dtype=np.int64)
    drawn = np.empty(n_samples, dtype=np.int64)

    for j in range(n_samples):
        reciprocal = 1.0 / (1.0 + overlap.astype(float))
        cumsum = np.concatenate([[0.0], np.cumsum(reciprocal)])
        # Conditional average uniqueness for every candidate in one shot.
        sums = cumsum[ends + 1] - cumsum[starts]
        avg_u = sums / lengths

        total = avg_u.sum()
        if total <= 0 or not np.isfinite(total):
            # All candidates saturated — fall back to uniform.
            idx = int(rng.integers(0, N))
        else:
            probs = avg_u / total
            idx = int(rng.choice(N, p=probs))
        drawn[j] = idx
        overlap[starts[idx] : ends[idx] + 1] += 1

    return drawn


# ---------------------------------------------------------------------------
# 4. Return-attribution weights
# ---------------------------------------------------------------------------

def get_return_attribution_weights(
    returns: pd.Series,
    event_starts: pd.DatetimeIndex,
    event_ends: pd.DatetimeIndex,
    close_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Weight events by the absolute return earned during their label period,
    divided by concurrent-event count per bar.

    Formula (AFML Ch. 4):

        w_i = |Σ_{t ∈ [s_i, e_i]}  r_t / c_t|

    where ``r_t`` is the per-bar return and ``c_t`` is the concurrent-event
    count. Weights are then rescaled so that ``Σ w_i = N``, matching the
    standard convention that weights average to 1.

    Args:
        returns:       Per-bar return series (e.g. ``close.pct_change()``);
                       will be reindexed onto ``close_index``.
        event_starts,
        event_ends:    Label-period boundaries per event.
        close_index:   Bar clock.

    Returns:
        pd.Series indexed by ``event_starts`` (values ≥ 0, mean 1).
    """
    if len(event_starts) == 0:
        return pd.Series([], dtype=float, index=event_starts,
                         name="return_attribution_weight")

    concurrent = get_num_concurrent_events(close_index, event_starts, event_ends)
    aligned = returns.reindex(close_index).fillna(0.0).to_numpy().astype(float)
    c = concurrent.to_numpy().astype(float)
    per_bar = np.where(c > 0, aligned / np.maximum(c, 1.0), 0.0)
    cumsum = np.concatenate([[0.0], np.cumsum(per_bar)])

    starts = _resolve_positions(close_index, event_starts, name="event_starts")
    ends = _resolve_positions(close_index, event_ends, name="event_ends")
    event_return = cumsum[ends + 1] - cumsum[starts]
    weights = np.abs(event_return)

    total = weights.sum()
    if total > 0:
        weights = weights * (len(event_starts) / total)
    else:
        # All returns zero — fall back to uniform weights of 1.
        weights = np.ones_like(weights)
    return pd.Series(weights, index=event_starts,
                     name="return_attribution_weight")


# ---------------------------------------------------------------------------
# 5. Time-decay weights
# ---------------------------------------------------------------------------

def get_time_decay_weights(
    uniqueness: pd.Series,
    oldest_weight: float = 1.0,
    newest_weight: float = 1.0,
    decay_type: str = "linear",
) -> pd.Series:
    """
    Apply a linear or exponential decay so recent samples dominate the fit.

    Samples are sorted chronologically by their index and assigned a
    coordinate ``x`` in ``[0, 1]`` — ``x = 0`` for the oldest, ``x = 1`` for
    the newest. Weights interpolate between ``oldest_weight`` and
    ``newest_weight`` over ``x``:

        linear       :  w(x) = oldest + (newest - oldest) * x
        exponential  :  log w(x) = log(oldest) + (log(newest) - log(oldest)) * x

    Args:
        uniqueness:     Output of :func:`get_average_uniqueness` (index =
                        event timestamps; must be sortable chronologically).
        oldest_weight:  Weight multiplier applied to the oldest sample.
                        Pass 1.0 to disable decay.
        newest_weight:  Weight multiplier applied to the newest sample
                        (defaults to 1.0, i.e. no amplification).
        decay_type:     ``"linear"`` (default) or ``"exponential"``. Linear
                        interpolates the weight directly; exponential
                        interpolates in log-space, giving a constant
                        relative drop-off per rank unit.

    Returns:
        pd.Series indexed like ``uniqueness``, ready to multiply with other
        weight components.
    """
    if len(uniqueness) == 0:
        return pd.Series([], dtype=float, index=uniqueness.index,
                         name="time_decay_weight")
    if decay_type not in ("linear", "exponential"):
        raise ValueError(
            f"decay_type must be 'linear' or 'exponential' (got {decay_type!r})"
        )
    if decay_type == "exponential" and (oldest_weight <= 0 or newest_weight <= 0):
        raise ValueError(
            "exponential decay requires oldest_weight and newest_weight > 0"
        )

    sorted_u = uniqueness.sort_index()
    n = len(sorted_u)
    # Chronological rank mapped to [0, 1]; single-sample case collapses to
    # the newest_weight.
    x = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])

    if decay_type == "linear":
        weights = oldest_weight + (newest_weight - oldest_weight) * x
        weights = np.clip(weights, 0.0, None)
    else:  # exponential
        log_old = float(np.log(oldest_weight))
        log_new = float(np.log(newest_weight))
        weights = np.exp(log_old + (log_new - log_old) * x)

    out = pd.Series(weights, index=sorted_u.index, name="time_decay_weight")
    return out.reindex(uniqueness.index)


# ---------------------------------------------------------------------------
# 6. Convenience aggregator
# ---------------------------------------------------------------------------

def compute_sample_weights(
    labels_df: pd.DataFrame,
    close: pd.Series,
    time_decay: float = 1.0,
) -> pd.Series:
    """
    Combine uniqueness, return attribution, and linear time decay.

    Expects ``labels_df`` as produced by
    :func:`src.labeling.triple_barrier.make_labels` — specifically requires an
    ``exit_timestamp`` column and an event-timestamp index. The final weights
    are rescaled to sum to ``len(labels_df)`` so they drop directly into
    scikit-learn / LightGBM's ``sample_weight`` parameter.

    Args:
        labels_df:   Labeled events; index must be event timestamps.
        close:       Price series used to compute bar returns.
        time_decay:  ``oldest_weight`` in the linear decay (1.0 = no decay).

    Returns:
        pd.Series indexed by event timestamp with mean weight 1.
    """
    if "exit_timestamp" not in labels_df.columns:
        raise ValueError("labels_df must contain an 'exit_timestamp' column")
    if len(labels_df) == 0:
        return pd.Series([], dtype=float, name="sample_weight",
                         index=labels_df.index)

    event_starts = pd.DatetimeIndex(labels_df.index)
    event_ends = pd.DatetimeIndex(labels_df["exit_timestamp"])
    close_index = pd.DatetimeIndex(close.index)

    uniqueness = get_average_uniqueness(event_starts, event_ends, close_index)
    returns = close.pct_change().fillna(0.0)
    ret_attr = get_return_attribution_weights(
        returns, event_starts, event_ends, close_index,
    )
    decay = get_time_decay_weights(
        uniqueness,
        oldest_weight=time_decay,
        newest_weight=1.0,
        decay_type="linear",
    )

    combined = uniqueness * ret_attr * decay
    total = float(combined.sum())
    if total > 0:
        combined = combined * (len(combined) / total)
    else:
        combined = pd.Series(1.0, index=combined.index)
    return combined.rename("sample_weight")
