"""
Core data models for the trading system.

These dataclasses define the canonical representations of market data
as it flows through the pipeline: Tick → Bar → Feature → Signal → Order.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Enums ──

class Side(Enum):
    BUY = 1
    SELL = -1
    UNKNOWN = 0


class BarType(Enum):
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"
    TICK_IMBALANCE = "tib"
    VOLUME_IMBALANCE = "vib"
    TICK_RUN = "tick_run"
    TIME = "time"  # standard time bars for reference


class AssetClass(Enum):
    EQUITIES = "equities"
    CRYPTO = "crypto"
    FUTURES = "futures"


# ── Tick ──

@dataclass(slots=True)
class Tick:
    """
    A single trade event. The atomic unit of market data.

    All bar types are constructed from sequences of Ticks.
    """
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: Side = Side.UNKNOWN  # classified by tick rule if not provided
    exchange: str = ""
    trade_id: str = ""

    @property
    def dollar_volume(self) -> float:
        return self.price * self.volume


# ── Bar ──

@dataclass(slots=True)
class Bar:
    """
    An OHLCV bar produced by the Data Engine.

    Contains standard OHLCV plus metadata about how the bar was formed
    (bar type, duration, number of ticks, imbalance statistics).
    """
    timestamp: datetime          # bar close time
    open_time: datetime          # bar open time
    symbol: str
    bar_type: BarType

    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    tick_count: int

    # Imbalance metadata (for TIB/VIB bars)
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_ticks: int = 0
    sell_ticks: int = 0
    imbalance: float = 0.0       # cumulative imbalance that triggered this bar
    threshold: float = 0.0       # the threshold that was exceeded

    # Computed
    vwap: float = 0.0
    bar_duration_seconds: float = 0.0

    @property
    def returns(self) -> float:
        """Simple bar return."""
        if self.open == 0:
            return 0.0
        return (self.close - self.open) / self.open

    @property
    def log_returns(self) -> float:
        """Log bar return."""
        if self.open <= 0 or self.close <= 0:
            return 0.0
        return math.log(self.close / self.open)

    @property
    def volume_imbalance(self) -> float:
        """Buy volume minus sell volume."""
        return self.buy_volume - self.sell_volume

    @property
    def tick_imbalance_ratio(self) -> float:
        """(buy_ticks - sell_ticks) / total_ticks."""
        if self.tick_count == 0:
            return 0.0
        return (self.buy_ticks - self.sell_ticks) / self.tick_count


# ── Bar Accumulator ──

@dataclass
class BarAccumulator:
    """
    Mutable state for building a bar from a stream of ticks.

    Used internally by bar constructors. When the bar is "complete"
    (threshold met), call .to_bar() to produce an immutable Bar.
    """
    symbol: str
    bar_type: BarType
    open_time: Optional[datetime] = None
    open: float = 0.0
    high: float = float('-inf')
    low: float = float('inf')
    close: float = 0.0
    volume: float = 0.0
    dollar_volume: float = 0.0
    tick_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_ticks: int = 0
    sell_ticks: int = 0
    cumulative_imbalance: float = 0.0
    vwap_numerator: float = 0.0  # sum(price * volume)
    last_timestamp: Optional[datetime] = None
    threshold: float = 0.0

    def add_tick(self, tick: Tick) -> None:
        """Accumulate a tick into the bar being built."""
        if self.tick_count == 0:
            self.open_time = tick.timestamp
            self.open = tick.price
            self.high = tick.price
            self.low = tick.price

        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        self.close = tick.price
        self.volume += tick.volume
        self.dollar_volume += tick.dollar_volume
        self.tick_count += 1
        self.vwap_numerator += tick.price * tick.volume
        self.last_timestamp = tick.timestamp

        if tick.side == Side.BUY:
            self.buy_volume += tick.volume
            self.buy_ticks += 1
        elif tick.side == Side.SELL:
            self.sell_volume += tick.volume
            self.sell_ticks += 1

    def to_bar(self) -> Bar:
        """Finalize the accumulated state into an immutable Bar."""
        vwap = self.vwap_numerator / self.volume if self.volume > 0 else self.close
        duration = 0.0
        if self.open_time and self.last_timestamp:
            duration = (self.last_timestamp - self.open_time).total_seconds()

        return Bar(
            timestamp=self.last_timestamp or datetime.now(timezone.utc),
            open_time=self.open_time or datetime.now(timezone.utc),
            symbol=self.symbol,
            bar_type=self.bar_type,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            dollar_volume=self.dollar_volume,
            tick_count=self.tick_count,
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            buy_ticks=self.buy_ticks,
            sell_ticks=self.sell_ticks,
            imbalance=self.cumulative_imbalance,
            threshold=self.threshold,
            vwap=vwap,
            bar_duration_seconds=duration,
        )

    def reset(self) -> None:
        """Reset accumulator for the next bar."""
        self.open_time = None
        self.open = 0.0
        self.high = float('-inf')
        self.low = float('inf')
        self.close = 0.0
        self.volume = 0.0
        self.dollar_volume = 0.0
        self.tick_count = 0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.buy_ticks = 0
        self.sell_ticks = 0
        self.cumulative_imbalance = 0.0
        self.vwap_numerator = 0.0
        self.last_timestamp = None
        self.threshold = 0.0
