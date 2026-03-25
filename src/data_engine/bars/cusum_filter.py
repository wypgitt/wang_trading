"""
CUSUM Event Filter (AFML Ch. 17)

Filters bars to identify structurally significant events worth
evaluating with the Signal Battery. Not every bar warrants a
trading decision — the CUSUM filter triggers only when the
cumulative deviation from the mean exceeds a threshold.

This prevents the system from evaluating every bar (generating noise)
and focuses computation on bars where something significant may have
occurred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def cusum_filter(
    prices: pd.Series,
    threshold: float,
) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM filter on log-returns.

    Monitors cumulative positive and negative deviations from the
    mean return. When either exceeds the threshold, an event is
    triggered and the cumulative sum resets.

    Args:
        prices:    Price series (index is datetime)
        threshold: CUSUM threshold h. Events fire when |cumulative
                   deviation| > h. Typically set to 1-2x the daily
                   return standard deviation.

    Returns:
        DatetimeIndex of event timestamps (subset of input index)
    """
    if len(prices) < 2:
        return pd.DatetimeIndex([])

    log_returns = np.log(prices / prices.shift(1)).dropna()

    events = []
    s_pos = 0.0  # positive CUSUM
    s_neg = 0.0  # negative CUSUM

    for dt, r in log_returns.items():
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)

        if s_pos > threshold:
            events.append(dt)
            s_pos = 0.0  # reset after trigger
        elif s_neg < -threshold:
            events.append(dt)
            s_neg = 0.0

    logger.info(
        f"CUSUM filter: {len(events)} events from {len(log_returns)} bars "
        f"(threshold={threshold:.6f})"
    )
    return pd.DatetimeIndex(events)


def compute_cusum_threshold(
    prices: pd.Series,
    multiplier: float = 1.5,
    lookback: int = 252,
) -> float:
    """
    Compute the CUSUM threshold as a multiple of daily return volatility.

    Args:
        prices:     Price series
        multiplier: How many standard deviations for the threshold
        lookback:   Number of bars for volatility estimation

    Returns:
        Threshold value h for the CUSUM filter
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    recent = log_returns.iloc[-lookback:] if len(log_returns) > lookback else log_returns
    std = recent.std()
    threshold = std * multiplier
    logger.debug(f"CUSUM threshold: {threshold:.6f} ({multiplier}x std={std:.6f})")
    return threshold
