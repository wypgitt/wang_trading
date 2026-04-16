"""
Trend Following Signals (Clenow — Following the Trend + Turtle system)

Two families:
    - MovingAverageCrossoverSignal : dual or triple EMA crossover
    - DonchianBreakoutSignal       : Turtle-style channel breakout

Plus ``atr_position_size``, Clenow/Turtle ATR-normalized unit sizing that
the Bet Sizing layer uses as a max-position cap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.signal_battery.base_signal import BaseSignalGenerator, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with Wilder-style alpha = 2/(period+1)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range = max(H-L, |H - prev_close|, |L - prev_close|).

    Operates bar-by-bar and returns a Series aligned to ``close``; the first
    bar has TR = H - L (no prev close).
    """
    prev_close = close.shift(1)
    hl = (high - low).abs()
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    # First bar's TR defaults to H - L.
    tr.iloc[0] = (high.iloc[0] - low.iloc[0])
    return tr


# ---------------------------------------------------------------------------
# Moving Average Crossover
# ---------------------------------------------------------------------------

class MovingAverageCrossoverSignal(BaseSignalGenerator):
    """
    EMA crossover signal with optional triple-MA 2-of-3 voting.

    Parameters (via ``params``):
        fast_period:   Fast EMA span (default 20).
        slow_period:   Slow EMA span (default 50).
        medium_period: Optional middle EMA. When provided, the signal fires
                       when at least 2 of 3 crossover predicates agree:
                       fast>medium, fast>slow, medium>slow.
    """

    REQUIRED_COLUMNS = ("close",)
    DEFAULT_PARAMS = {"fast_period": 20, "slow_period": 50, "medium_period": None}

    def __init__(
        self,
        name: str = "ma_crossover",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)

        fast = int(self.params["fast_period"])
        slow = int(self.params["slow_period"])
        if fast >= slow:
            raise ValueError("fast_period must be < slow_period")
        med = self.params["medium_period"]
        if med is not None:
            med = int(med)
            if not (fast < med < slow):
                raise ValueError("medium_period must satisfy fast < medium < slow")

    # ------------------------------------------------------------------ API --

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
        **kwargs: Any,
    ) -> list[Signal]:
        if bars is None:
            raise ValueError(f"{self.name}: bars DataFrame is required")
        self.validate_input(bars)
        close = bars["close"].astype(float)

        fast_p = int(self.params["fast_period"])
        slow_p = int(self.params["slow_period"])
        med_p = self.params["medium_period"]
        med_p = int(med_p) if med_p is not None else None

        fast = _ema(close, fast_p)
        slow = _ema(close, slow_p)
        medium = _ema(close, med_p) if med_p is not None else None

        signals: list[Signal] = []
        for t in close.index:
            f, s = fast.loc[t], slow.loc[t]
            if not (np.isfinite(f) and np.isfinite(s)):
                continue

            if medium is not None:
                m = medium.loc[t]
                if not np.isfinite(m):
                    continue
                votes = [f > m, f > s, m > s]
                # Triple-MA: side from majority vote among the three signs.
                up_votes = sum(votes)
                down_votes = sum(not v for v in votes)
                if up_votes >= 2:
                    side = 1
                elif down_votes >= 2:
                    side = -1
                else:
                    side = 0  # shouldn't happen with 3 bools, but keep safe
            else:
                if f > s:
                    side = 1
                elif f < s:
                    side = -1
                else:
                    side = 0

            price = float(close.loc[t])
            if price <= 0:
                continue
            # Confidence = |fast - slow| as a fraction of price, capped at 1.
            confidence = float(min(abs(f - s) / price, 1.0))
            if side == 0:
                confidence = 0.0

            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=symbol,
                    family=self.name,
                    side=int(side),
                    confidence=confidence,
                    metadata={
                        "fast_ema": float(f),
                        "slow_ema": float(s),
                        "medium_ema": float(medium.loc[t]) if medium is not None else None,
                        "price": price,
                    },
                )
            )
        return signals


# ---------------------------------------------------------------------------
# Donchian Channel Breakout
# ---------------------------------------------------------------------------

class DonchianBreakoutSignal(BaseSignalGenerator):
    """
    Turtle-style Donchian channel breakout.

    Entry:
        Long  : close > highest high of prior ``entry_period`` bars.
        Short : close < lowest  low  of prior ``entry_period`` bars.
    Exit:
        Long exit : close < lowest  low  of prior ``exit_period`` bars (signal side=0).
        Short exit: close > highest high of prior ``exit_period`` bars (signal side=0).

    Confidence uses the half-range distance from the channel midpoint:
        confidence = |price - channel_mid| / (channel_range / 2),
    clipped to [0, 1]. A clean break of the upper channel yields ~1.0.

    Parameters:
        entry_period: Lookback for entry channel (default 55).
        exit_period:  Lookback for exit channel (default 20).
    """

    REQUIRED_COLUMNS = ("close", "high", "low")
    DEFAULT_PARAMS = {"entry_period": 55, "exit_period": 20}

    def __init__(
        self,
        name: str = "donchian_breakout",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        if int(self.params["entry_period"]) < 2:
            raise ValueError("entry_period must be >= 2")
        if int(self.params["exit_period"]) < 2:
            raise ValueError("exit_period must be >= 2")

    # ------------------------------------------------------------------ API --

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
        **kwargs: Any,
    ) -> list[Signal]:
        if bars is None:
            raise ValueError(f"{self.name}: bars DataFrame is required")
        self.validate_input(bars)
        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)

        ent_p = int(self.params["entry_period"])
        exit_p = int(self.params["exit_period"])

        # Prior-bar channels (shifted) so the current bar's own H/L don't
        # count toward the channel it might be breaking out of.
        entry_high = high.shift(1).rolling(window=ent_p, min_periods=ent_p).max()
        entry_low = low.shift(1).rolling(window=ent_p, min_periods=ent_p).min()
        exit_high = high.shift(1).rolling(window=exit_p, min_periods=exit_p).max()
        exit_low = low.shift(1).rolling(window=exit_p, min_periods=exit_p).min()

        signals: list[Signal] = []
        current_side = 0  # stateful: track whether we're in a position for exit logic

        for t in close.index:
            price = float(close.loc[t])
            eh, el = entry_high.loc[t], entry_low.loc[t]
            xh, xl = exit_high.loc[t], exit_low.loc[t]
            if not (np.isfinite(eh) and np.isfinite(el) and np.isfinite(xh) and np.isfinite(xl)):
                continue

            # Determine entry / exit events.
            event: str | None = None
            side = current_side

            if current_side <= 0 and price > eh:
                event = "entry"
                side = 1
            elif current_side >= 0 and price < el:
                event = "entry"
                side = -1
            elif current_side > 0 and price < xl:
                event = "exit"
                side = 0
            elif current_side < 0 and price > xh:
                event = "exit"
                side = 0

            if event is None:
                continue

            channel_mid = (eh + el) / 2.0
            channel_half = (eh - el) / 2.0
            if channel_half <= 0 or not np.isfinite(channel_half):
                confidence = 0.0
            else:
                confidence = float(min(abs(price - channel_mid) / channel_half, 1.0))
            if side == 0:
                confidence = 0.0

            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=symbol,
                    family=self.name,
                    side=int(side),
                    confidence=confidence,
                    metadata={
                        "event": event,
                        "price": price,
                        "entry_high": float(eh),
                        "entry_low": float(el),
                        "exit_high": float(xh),
                        "exit_low": float(xl),
                    },
                )
            )
            current_side = int(side)

        return signals


# ---------------------------------------------------------------------------
# ATR position sizing
# ---------------------------------------------------------------------------

def atr_position_size(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    risk_per_trade: float = 0.02,
    account_value: float = 100_000.0,
    atr_period: int = 20,
    atr_multiplier: float = 2.0,
) -> pd.Series:
    """
    Clenow/Turtle ATR-based unit sizing.

    ATR is the EMA of True Range over ``atr_period``. The number of units
    sized such that a ``atr_multiplier`` × ATR move costs
    ``account_value × risk_per_trade``:

        units = (account_value * risk_per_trade) / (atr_multiplier * ATR)

    Returns a Series of unit counts (NaN during the ATR warmup). Zero or
    negative ATR bars map to NaN.
    """
    if atr_period < 2:
        raise ValueError("atr_period must be >= 2")
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive")
    if risk_per_trade <= 0 or account_value <= 0:
        raise ValueError("risk_per_trade and account_value must be positive")

    tr = _true_range(high, low, close)
    atr = _ema(tr, atr_period)
    risk_budget = account_value * risk_per_trade
    denom = (atr_multiplier * atr).replace(0.0, np.nan)
    units = risk_budget / denom
    return units.rename("atr_units")
