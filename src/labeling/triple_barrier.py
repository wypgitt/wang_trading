"""
Triple-Barrier Labeling (AFML Ch. 3)

Defines what the meta-labeler should predict. For each CUSUM-triggered event
with a direction from the Signal Battery, three barriers bound the trade
outcome:

    * Upper barrier (take profit for longs / stop loss for shorts):
        entry * (1 + upper_multiplier * volatility)
    * Lower barrier (stop loss for longs / take profit for shorts):
        entry * (1 - lower_multiplier * volatility)
    * Vertical barrier (time expiry):
        event_timestamp + max_holding_period bars.

The first barrier touched determines the label. If neither horizontal barrier
is hit before the vertical barrier, the label is the sign of the side-adjusted
unrealized return at expiry. Labels are then consumed by:

    * the meta-labeler as the binary target (1 = profitable),
    * the sample-weights module for return attribution and uniqueness,
    * the bet-sizing layer (indirectly, via the calibrated probability).

See design doc §6.1 (Triple-Barrier Method) and §6.2 (Meta-Labeling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Volatility estimator (used to scale the horizontal barriers)
# ---------------------------------------------------------------------------

def get_daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """
    Exponentially weighted moving standard deviation of simple returns.

    This is the "daily vol" surrogate from AFML Ch. 3 — each event's horizontal
    barrier width is set to ``multiplier * vol * entry_price`` using this
    series, so the labels adapt automatically to the prevailing regime
    rather than anchoring on a fixed price distance.

    Args:
        close: Close price series (any bar frequency).
        span:  EWM span. AFML defaults to 100 bars.

    Returns:
        pd.Series: EWM std of ``close.pct_change()``, aligned to ``close.index``.
        The first value is NaN (no prior return); subsequent values warm up
        gradually as the EWM accumulates observations.
    """
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pandas Series")
    if span < 2:
        raise ValueError("span must be >= 2")
    returns = close.pct_change()
    return returns.ewm(span=span).std().rename("daily_vol")


# ---------------------------------------------------------------------------
# Vertical barrier (time expiry)
# ---------------------------------------------------------------------------

def get_vertical_barriers(
    timestamps: pd.DatetimeIndex,
    close: pd.Series,
    max_holding_period: int,
) -> pd.Series:
    """
    Map each event timestamp to the bar timestamp that expires its trade.

    The vertical barrier is ``max_holding_period`` bars forward of the event.
    When an event fires near the end of the series such that there are fewer
    than ``max_holding_period`` bars remaining, the expiry is clamped to the
    final available timestamp. Events whose timestamp does not appear in
    ``close.index`` are located via ``searchsorted`` (nearest following bar).

    Args:
        timestamps:         Event timestamps (typically CUSUM-triggered).
        close:              Price series defining the bar clock.
        max_holding_period: Forward bar count for the vertical barrier.

    Returns:
        pd.Series indexed by ``timestamps`` whose values are the expiry
        timestamps drawn from ``close.index``.
    """
    if not isinstance(timestamps, (pd.DatetimeIndex, pd.Index)):
        raise ValueError("timestamps must be a pandas Index")
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pandas Series")
    if max_holding_period < 1:
        raise ValueError("max_holding_period must be >= 1")

    close_idx = close.index
    n = len(close_idx)
    if n == 0:
        raise ValueError("close series is empty")

    positions = np.asarray(close_idx.get_indexer(timestamps))
    missing = positions == -1
    if missing.any():
        # Fallback: for timestamps not on the bar grid, map to the next bar.
        fallback = np.asarray(close_idx.searchsorted(
            np.asarray(list(timestamps))[missing]
        ))
        fallback = np.clip(fallback, 0, n - 1)
        positions = positions.copy()
        positions[missing] = fallback

    expiry_positions = np.clip(positions + max_holding_period, 0, n - 1)
    return pd.Series(
        close_idx[expiry_positions],
        index=timestamps,
        name="vertical_barrier",
    )


# ---------------------------------------------------------------------------
# Core triple-barrier labeler
# ---------------------------------------------------------------------------

def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    upper_multiplier: float = 2.0,
    lower_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Walk each event forward to its earliest barrier touch.

    Args:
        close:            Close price series indexed by bar timestamp.
        events:           DataFrame indexed by event timestamp with columns:
                              side              (int, ±1),
                              volatility        (float, ≥ 0),
                              vertical_barrier  (timestamp in close.index).
        upper_multiplier: Physical-upper barrier width in units of volatility.
                          For long trades this is the take-profit multiplier;
                          for shorts it is the stop-loss multiplier. Set
                          asymmetrically to tune the trade profile (e.g.
                          momentum favours wider upper_multiplier).
        lower_multiplier: Physical-lower barrier width. Mirror role: stop-loss
                          for longs, take-profit for shorts.

    Returns:
        DataFrame indexed by event timestamp with columns:
            return           side-adjusted realised return at exit,
            label            +1 profitable, -1 unprofitable, 0 flat at expiry,
            barrier_touched  "upper" / "lower" / "vertical",
            exit_timestamp   bar timestamp of the exit,
            holding_period   number of bars between event and exit.
    """
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pandas Series")
    if not isinstance(events, pd.DataFrame):
        raise ValueError("events must be a pandas DataFrame")
    missing = {"side", "volatility", "vertical_barrier"} - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns {sorted(missing)}")
    if upper_multiplier <= 0 or lower_multiplier <= 0:
        raise ValueError("barrier multipliers must be > 0")

    close_idx = close.index
    # Cache integer positions for O(1) holding-period math. Duplicate bar
    # timestamps would break get_loc; the data engine already enforces unique
    # indices, but we verify here since bad data would produce silent bugs.
    if not close_idx.is_unique:
        raise ValueError("close.index must be unique")

    records: list[dict] = []
    n_skipped = 0

    for event_time, row in events.iterrows():
        side = int(row["side"])
        vol = float(row["volatility"])
        vertical = row["vertical_barrier"]

        if side not in (-1, 1):
            n_skipped += 1
            continue
        if not np.isfinite(vol) or vol <= 0.0:
            # Vol undefined → barriers collapse to the entry price; skip.
            n_skipped += 1
            continue
        if event_time not in close_idx:
            n_skipped += 1
            continue

        entry_price = float(close.loc[event_time])
        upper_barrier = entry_price * (1.0 + upper_multiplier * vol)
        lower_barrier = entry_price * (1.0 - lower_multiplier * vol)

        # Forward path (exclude the entry bar — barriers are defined relative
        # to the entry price itself, so t=0 can't "touch" them).
        path = close.loc[event_time:vertical]
        if len(path) <= 1:
            # No forward bars to evaluate — treat as immediate vertical exit.
            exit_time = event_time
            exit_price = entry_price
            barrier_touched = "vertical"
            label = 0
            ret = 0.0
            holding = 0
        else:
            forward = path.iloc[1:]
            upper_hits = forward.index[forward.to_numpy() >= upper_barrier]
            lower_hits = forward.index[forward.to_numpy() <= lower_barrier]
            first_upper = upper_hits[0] if len(upper_hits) else None
            first_lower = lower_hits[0] if len(lower_hits) else None

            if first_upper is not None and (
                first_lower is None or first_upper <= first_lower
            ):
                exit_time = first_upper
                exit_price = float(close.loc[exit_time])
                barrier_touched = "upper"
                # Physical upper: profitable for long (+1), loss for short (-1).
                label = 1 if side == 1 else -1
            elif first_lower is not None:
                exit_time = first_lower
                exit_price = float(close.loc[exit_time])
                barrier_touched = "lower"
                label = -1 if side == 1 else 1
            else:
                exit_time = forward.index[-1]
                exit_price = float(close.loc[exit_time])
                barrier_touched = "vertical"
                signed_ret = ((exit_price / entry_price) - 1.0) * side
                label = 1 if signed_ret > 0 else -1 if signed_ret < 0 else 0

            ret = ((exit_price / entry_price) - 1.0) * side
            holding = close_idx.get_loc(exit_time) - close_idx.get_loc(event_time)

        records.append({
            "event_timestamp": event_time,
            "return": ret,
            "label": label,
            "barrier_touched": barrier_touched,
            "exit_timestamp": exit_time,
            "holding_period": int(holding),
        })

    if n_skipped:
        logger.debug(
            f"apply_triple_barrier: skipped {n_skipped} events with "
            "invalid side/vol/timestamp"
        )

    if not records:
        return pd.DataFrame(
            columns=["return", "label", "barrier_touched",
                     "exit_timestamp", "holding_period"],
            index=pd.DatetimeIndex([], name="event_timestamp"),
        )

    out = pd.DataFrame(records).set_index("event_timestamp")
    return out


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def make_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    sides: pd.Series,
    volatility: pd.Series,
    max_holding_period: int = 50,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Assemble the events DataFrame and run the triple-barrier labeler.

    The main entry point for the labeling pipeline. Callers supply the price
    series, the event timestamps from the CUSUM filter, the side predicted by
    the Signal Battery, and a volatility series (typically
    :func:`get_daily_volatility` or the GARCH forecast from
    ``src.feature_factory.volatility.garch_volatility``).

    Args:
        close:              Close price series.
        events:             Event timestamps (CUSUM-triggered).
        sides:              ±1 direction per event (indexed like ``events``).
        volatility:         Volatility per bar (indexed like ``close``).
        max_holding_period: Vertical barrier distance in bars.
        upper_mult:         Upper horizontal barrier multiplier.
        lower_mult:         Lower horizontal barrier multiplier.

    Returns:
        DataFrame from :func:`apply_triple_barrier`.
    """
    if not isinstance(events, pd.DatetimeIndex):
        events = pd.DatetimeIndex(events)

    side_series = sides.reindex(events).astype("float")
    if side_series.isna().any():
        raise ValueError("sides must provide a value for every event timestamp")

    vol_at_events = volatility.reindex(events)
    vertical = get_vertical_barriers(events, close, max_holding_period)

    events_df = pd.DataFrame(
        {
            "side": side_series.astype(int),
            "volatility": vol_at_events,
            "vertical_barrier": vertical,
        },
        index=events,
    )
    return apply_triple_barrier(
        close, events_df, upper_multiplier=upper_mult, lower_multiplier=lower_mult,
    )


def get_meta_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    sides: pd.Series,
    volatility: pd.Series,
    max_holding_period: int = 50,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
) -> pd.Series:
    """
    Binary meta-label target: 1 if the primary-model trade was profitable.

    The Signal Battery (primary model) supplies the side. The meta-labeler
    learns to predict this target — "given that the primary said LONG, should
    I actually take this trade?" — from the feature matrix. See design doc
    §6.2 for the two-stage architecture.

    Returns:
        pd.Series indexed by event timestamp, values ∈ {0, 1}.
    """
    labeled = make_labels(
        close, events, sides, volatility,
        max_holding_period=max_holding_period,
        upper_mult=upper_mult,
        lower_mult=lower_mult,
    )
    return (labeled["label"] > 0).astype(int).rename("meta_label")
