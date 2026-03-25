"""
Tick Rule: Classify trades as buy or sell.

The tick rule classifies each trade based on price movement relative
to the previous trade:
  - Price up from previous trade → BUY
  - Price down from previous trade → SELL
  - Price unchanged → same as previous classification

This is the standard method from AFML Ch. 2 for classifying trade
direction when the exchange does not provide it.
"""

from __future__ import annotations

from src.data_engine.models import Tick, Side


class TickRuleClassifier:
    """
    Stateful tick rule classifier.

    Maintains the last known price and side to handle unchanged prices
    (which inherit the previous classification).
    """

    def __init__(self):
        self._last_price: float | None = None
        self._last_side: Side = Side.BUY  # default for first tick

    def classify(self, tick: Tick) -> Side:
        """
        Classify a tick as BUY or SELL using the tick rule.

        If the tick already has a side from the exchange, returns it as-is.
        Otherwise applies the tick rule based on price movement.
        """
        if tick.side != Side.UNKNOWN:
            self._last_price = tick.price
            self._last_side = tick.side
            return tick.side

        if self._last_price is None:
            self._last_price = tick.price
            return self._last_side

        if tick.price > self._last_price:
            side = Side.BUY
        elif tick.price < self._last_price:
            side = Side.SELL
        else:
            side = self._last_side  # unchanged → carry forward

        self._last_price = tick.price
        self._last_side = side
        return side

    def reset(self):
        self._last_price = None
        self._last_side = Side.BUY
