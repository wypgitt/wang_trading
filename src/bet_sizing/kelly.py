"""
Layer 2 of the Bet-Sizing cascade — Kelly Criterion (Chan, design-doc §8.2).

Kelly's optimal fraction for a binary bet:

    f* = (p · W − q · L) / (W · L)
       = (win_prob · avg_win − loss_prob · avg_loss) / (avg_win · avg_loss)

where ``W = avg_win`` and ``L = avg_loss`` are positive magnitudes of the
typical winning and losing trade. Maximises long-run geometric growth.

**Why fractional Kelly**: full Kelly is optimal in expectation but draws
down brutally in practice — realised edge and variance are estimated, and
every estimation error maps to sizing error. The design doc clips to
``¼–½`` of Kelly; ``0.25`` is the default here. The Kelly value returned
by this module is used as a HARD CAP by the bet-sizing cascade — the final
position size is ``min(afml_size, fractional_kelly)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Point-estimate Kelly
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_prob: float, avg_win: float, avg_loss: float,
) -> float:
    """
    Full Kelly fraction for a binary bet.

    Inputs:
        win_prob:  Probability of a winning trade, in ``[0, 1]``.
        avg_win:   Mean return of winning trades (positive magnitude).
        avg_loss:  Mean |return| of losing trades (positive magnitude).

    Returns:
        Kelly fraction ∈ ``[0, 1]``. Negative f* (no edge) is clipped to 0.
        f* > 1 is clipped to 1 so this value is always a usable cap.

    Raises:
        ValueError on malformed inputs (``win_prob`` out of range,
        non-positive ``avg_win`` or ``avg_loss``).
    """
    if win_prob < 0.0 or win_prob > 1.0:
        raise ValueError(f"win_prob must be in [0, 1] (got {win_prob})")
    if avg_win <= 0.0 or avg_loss <= 0.0:
        raise ValueError(
            f"avg_win and avg_loss must be positive magnitudes "
            f"(got {avg_win}, {avg_loss})"
        )

    loss_prob = 1.0 - win_prob
    f_star = (win_prob * avg_win - loss_prob * avg_loss) / (avg_win * avg_loss)
    return float(max(0.0, min(f_star, 1.0)))


def fractional_kelly(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly = ``fraction × full Kelly``.

    A 25% Kelly schedule roughly halves growth vs full Kelly but also
    roughly halves the maximum drawdown — strongly preferred in practice
    for estimated-edge strategies.
    """
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in [0, 1] (got {fraction})")
    return float(fraction * kelly_fraction(win_prob, avg_win, avg_loss))


# ---------------------------------------------------------------------------
# Rolling Kelly from realised returns
# ---------------------------------------------------------------------------

def rolling_kelly(
    returns: pd.Series,
    window: int = 252,
    fraction: float = 0.25,
) -> pd.Series:
    """
    Time-varying fractional Kelly computed on a rolling window of returns.

    For each bar ``t`` from ``window - 1`` onward, win_prob/avg_win/avg_loss
    are estimated from the trailing ``window`` returns and fed into
    :func:`fractional_kelly`. Returns that are exactly 0 are excluded from
    both numerator and denominator of ``win_prob`` (pushes / no-trade bars
    shouldn't drag the win rate toward zero).

    Args:
        returns:   Per-bar or per-trade return series.
        window:    Rolling window length.
        fraction:  Fractional-Kelly multiplier, forwarded to
                   :func:`fractional_kelly`.

    Returns:
        pd.Series indexed like ``returns``; the first ``window - 1`` rows
        are NaN (insufficient history).
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")
    if window < 2:
        raise ValueError("window must be >= 2")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in [0, 1] (got {fraction})")

    n = len(returns)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=returns.index, name="rolling_kelly")

    arr = returns.to_numpy(dtype=float)
    for t in range(window - 1, n):
        w = arr[t - window + 1 : t + 1]
        finite = w[np.isfinite(w)]
        if len(finite) == 0:
            out[t] = 0.0
            continue
        wins = finite[finite > 0.0]
        losses = finite[finite < 0.0]
        n_trades = len(wins) + len(losses)
        if n_trades == 0 or len(wins) == 0 or len(losses) == 0:
            # No edge information available on this window.
            out[t] = 0.0
            continue
        win_prob = len(wins) / n_trades
        avg_win = float(wins.mean())
        avg_loss = float(-losses.mean())  # positive magnitude
        out[t] = fractional_kelly(
            win_prob, avg_win, avg_loss, fraction=fraction,
        )

    return pd.Series(out, index=returns.index, name="rolling_kelly")


# ---------------------------------------------------------------------------
# Meta-labeler bridge
# ---------------------------------------------------------------------------

def kelly_from_meta_labeler(
    prob: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Plug the Tier-1 meta-labeler's calibrated probability straight into Kelly.

    ``avg_win`` / ``avg_loss`` come from the triple-barrier backtest for the
    relevant signal family (magnitudes, not signed). The return is a Kelly
    cap ready to ``min(..., afml_size)`` inside the bet-sizing cascade.
    """
    return fractional_kelly(prob, avg_win, avg_loss, fraction=fraction)
