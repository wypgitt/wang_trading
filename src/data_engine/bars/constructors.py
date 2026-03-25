"""
Bar Constructors (AFML Ch. 2)

Implements all bar types from AFML:
  - TickBarConstructor:    Sample every N ticks
  - VolumeBarConstructor:  Sample every V units of volume
  - DollarBarConstructor:  Sample every $D in notional value
  - TIBConstructor:        Tick Imbalance Bars (dynamic threshold)
  - VIBConstructor:        Volume Imbalance Bars (dynamic threshold)

Each constructor is a stateful object that processes ticks one at a time
and emits completed bars. This streaming design supports both historical
batch processing and real-time WebSocket ingestion.

Usage:
    constructor = TIBConstructor(symbol="AAPL", ewma_span=100)
    for tick in tick_stream:
        bar = constructor.process_tick(tick)
        if bar is not None:
            store(bar)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from src.data_engine.models import (
    Tick, Bar, BarType, Side, BarAccumulator,
)
from src.data_engine.bars.tick_rule import TickRuleClassifier


class BaseBarConstructor(ABC):
    """
    Abstract base for all bar constructors.

    Handles tick classification, accumulation, and bar emission.
    Subclasses implement _is_bar_complete() to define their sampling rule.
    """

    def __init__(self, symbol: str, bar_type: BarType):
        self.symbol = symbol
        self.bar_type = bar_type
        self._classifier = TickRuleClassifier()
        self._accumulator = BarAccumulator(symbol=symbol, bar_type=bar_type)
        self._bar_count = 0

    def process_tick(self, tick: Tick) -> Optional[Bar]:
        """
        Process a single tick. Returns a Bar if one was completed, else None.
        """
        # Classify side if not provided
        classified_side = self._classifier.classify(tick)
        if tick.side == Side.UNKNOWN:
            tick = Tick(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                price=tick.price,
                volume=tick.volume,
                side=classified_side,
                exchange=tick.exchange,
                trade_id=tick.trade_id,
            )

        # Accumulate
        self._accumulator.add_tick(tick)

        # Check if bar is complete
        if self._is_bar_complete(tick):
            bar = self._accumulator.to_bar()
            self._on_bar_complete(bar)
            self._accumulator.reset()
            self._bar_count += 1
            return bar

        return None

    def process_ticks(self, ticks: list[Tick]) -> list[Bar]:
        """Process a batch of ticks. Returns list of completed bars."""
        bars = []
        for tick in ticks:
            bar = self.process_tick(tick)
            if bar is not None:
                bars.append(bar)
        return bars

    @abstractmethod
    def _is_bar_complete(self, tick: Tick) -> bool:
        """Return True if the current accumulation should form a bar."""
        ...

    def _on_bar_complete(self, bar: Bar) -> None:
        """Hook for subclasses to update state when a bar completes."""
        pass

    @property
    def bars_produced(self) -> int:
        return self._bar_count

    def reset(self) -> None:
        """Reset all state."""
        self._classifier.reset()
        self._accumulator.reset()
        self._bar_count = 0


# ── Fixed-Threshold Bar Constructors ──

class TickBarConstructor(BaseBarConstructor):
    """
    Tick Bars (AFML Ch. 2.3): Sample every N trades.

    The simplest information-driven bar. Produces bars with equal
    numbers of trades, regardless of time elapsed.
    """

    def __init__(self, symbol: str, bar_size: int = 500):
        super().__init__(symbol, BarType.TICK)
        self.bar_size = bar_size

    def _is_bar_complete(self, tick: Tick) -> bool:
        return self._accumulator.tick_count >= self.bar_size


class VolumeBarConstructor(BaseBarConstructor):
    """
    Volume Bars (AFML Ch. 2.3): Sample every V units of volume.

    Normalizes for trade size: a single large block trade and many
    small retail trades produce bars at different rates.
    """

    def __init__(self, symbol: str, bar_size: float = 50_000):
        super().__init__(symbol, BarType.VOLUME)
        self.bar_size = bar_size

    def _is_bar_complete(self, tick: Tick) -> bool:
        return self._accumulator.volume >= self.bar_size


class DollarBarConstructor(BaseBarConstructor):
    """
    Dollar Bars (AFML Ch. 2.3): Sample every $D in notional value.

    Best for cross-asset normalization: a $1M threshold produces
    comparable bar frequencies for AAPL ($150/share) and BRK.A
    ($500K/share) despite very different price levels.
    """

    def __init__(self, symbol: str, bar_size: float = 1_000_000):
        super().__init__(symbol, BarType.DOLLAR)
        self.bar_size = bar_size

    def _is_bar_complete(self, tick: Tick) -> bool:
        return self._accumulator.dollar_volume >= self.bar_size


# ── Information-Driven Bar Constructors (AFML Ch. 2.4) ──

class TIBConstructor(BaseBarConstructor):
    """
    Tick Imbalance Bars (AFML Ch. 2.4)

    The primary bar type in the system. Samples when the cumulative
    signed tick imbalance exceeds a dynamically adjusted threshold.

    The key insight: bars form faster during periods of informed trading
    (high imbalance) and slower during noise (balanced buy/sell flow).
    This naturally aligns sampling with information arrival.

    The dynamic threshold is an EWMA of the absolute imbalance at which
    previous bars were triggered. This prevents the threshold from
    drifting during extended quiet or volatile periods.

    Parameters:
        symbol:       Instrument identifier
        ewma_span:    Number of bars for EWMA of threshold (default: 100)
        initial_threshold: Starting threshold before EWMA has data (default: 50)
    """

    def __init__(
        self,
        symbol: str,
        ewma_span: int = 100,
        initial_threshold: float = 50.0,
    ):
        super().__init__(symbol, BarType.TICK_IMBALANCE)
        self.ewma_span = ewma_span
        self._alpha = 2.0 / (ewma_span + 1)

        # Dynamic threshold state
        self._expected_imbalance = initial_threshold
        self._cumulative_theta = 0.0  # running signed tick sum

        # Track expected bar length for threshold calibration
        self._expected_ticks = initial_threshold  # E[T]
        self._bar_tick_counts: deque[int] = deque(maxlen=ewma_span)
        self._bar_imbalances: deque[float] = deque(maxlen=ewma_span)

    def _is_bar_complete(self, tick: Tick) -> bool:
        # Update cumulative imbalance with signed tick (+1 buy, -1 sell)
        sign = 1.0 if tick.side == Side.BUY else -1.0
        self._cumulative_theta += sign

        # Store threshold on accumulator for metadata
        self._accumulator.cumulative_imbalance = self._cumulative_theta
        self._accumulator.threshold = self._expected_imbalance

        # Bar triggers when |cumulative imbalance| exceeds threshold
        return abs(self._cumulative_theta) >= self._expected_imbalance

    def _on_bar_complete(self, bar: Bar) -> None:
        """Update EWMA threshold after bar completion."""
        # Record this bar's stats
        self._bar_tick_counts.append(bar.tick_count)
        abs_imbalance = abs(self._cumulative_theta)
        self._bar_imbalances.append(abs_imbalance)

        # Update expected ticks per bar (EWMA)
        self._expected_ticks = (
            self._alpha * bar.tick_count
            + (1 - self._alpha) * self._expected_ticks
        )

        # Update expected imbalance (EWMA of |imbalance| at trigger)
        # The threshold is E[T] * E[|imbalance per tick|]
        if len(self._bar_imbalances) > 1:
            avg_imb_per_tick = abs_imbalance / max(bar.tick_count, 1)
            expected_imb_per_tick = (
                self._alpha * avg_imb_per_tick
                + (1 - self._alpha) * (self._expected_imbalance / max(self._expected_ticks, 1))
            )
            self._expected_imbalance = self._expected_ticks * expected_imb_per_tick
        else:
            # Not enough data yet, use simple EWMA of imbalance
            self._expected_imbalance = (
                self._alpha * abs_imbalance
                + (1 - self._alpha) * self._expected_imbalance
            )

        # Enforce minimum threshold to prevent degenerate 1-tick bars
        self._expected_imbalance = max(self._expected_imbalance, 5.0)

        # Reset cumulative imbalance for next bar
        self._cumulative_theta = 0.0

    def reset(self) -> None:
        super().reset()
        self._cumulative_theta = 0.0
        self._bar_tick_counts.clear()
        self._bar_imbalances.clear()


class VIBConstructor(BaseBarConstructor):
    """
    Volume Imbalance Bars (AFML Ch. 2.4)

    Like TIBs but uses volume-weighted imbalance rather than tick count.
    A single large buy trade contributes more to the imbalance than
    many small buys. Better for equities where trade sizes vary widely.

    Parameters:
        symbol:       Instrument identifier
        ewma_span:    Number of bars for EWMA of threshold
        initial_threshold: Starting threshold before EWMA has data
    """

    def __init__(
        self,
        symbol: str,
        ewma_span: int = 100,
        initial_threshold: float = 50_000.0,
    ):
        super().__init__(symbol, BarType.VOLUME_IMBALANCE)
        self.ewma_span = ewma_span
        self._alpha = 2.0 / (ewma_span + 1)
        self._expected_imbalance = initial_threshold
        self._cumulative_theta = 0.0
        self._bar_imbalances: deque[float] = deque(maxlen=ewma_span)

    def _is_bar_complete(self, tick: Tick) -> bool:
        sign = 1.0 if tick.side == Side.BUY else -1.0
        self._cumulative_theta += sign * tick.volume

        self._accumulator.cumulative_imbalance = self._cumulative_theta
        self._accumulator.threshold = self._expected_imbalance

        return abs(self._cumulative_theta) >= self._expected_imbalance

    def _on_bar_complete(self, bar: Bar) -> None:
        abs_imbalance = abs(self._cumulative_theta)
        self._bar_imbalances.append(abs_imbalance)

        self._expected_imbalance = (
            self._alpha * abs_imbalance
            + (1 - self._alpha) * self._expected_imbalance
        )
        self._expected_imbalance = max(self._expected_imbalance, 100.0)
        self._cumulative_theta = 0.0

    def reset(self) -> None:
        super().reset()
        self._cumulative_theta = 0.0
        self._bar_imbalances.clear()


# ── Factory ──

def create_bar_constructor(
    symbol: str,
    bar_type: str,
    **kwargs,
) -> BaseBarConstructor:
    """
    Factory function to create bar constructors by type name.

    Args:
        symbol:   Instrument symbol
        bar_type: One of 'tick', 'volume', 'dollar', 'tib', 'vib'
        **kwargs: Constructor-specific parameters (bar_size, ewma_span, etc.)
    """
    constructors = {
        "tick": TickBarConstructor,
        "volume": VolumeBarConstructor,
        "dollar": DollarBarConstructor,
        "tib": TIBConstructor,
        "vib": VIBConstructor,
    }

    if bar_type not in constructors:
        raise ValueError(
            f"Unknown bar type '{bar_type}'. Valid types: {list(constructors.keys())}"
        )

    cls = constructors[bar_type]

    # Map generic kwargs to constructor-specific params
    if bar_type in ("tick", "volume", "dollar") and "bar_size" not in kwargs:
        size_key = f"{bar_type}_bar_size"
        if size_key in kwargs:
            kwargs["bar_size"] = kwargs.pop(size_key)

    # Filter kwargs to only those the constructor accepts
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(symbol=symbol, **filtered)
