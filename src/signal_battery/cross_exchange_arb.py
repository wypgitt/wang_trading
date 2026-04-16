"""
Cross-Exchange Crypto Arbitrage (Signal Battery §4.6)

Two pieces:
    - CrossExchangeArbSignal     : bar-level arb detection on a prices frame
                                   (one column per exchange)
    - MultiExchangePriceTracker  : real-time snapshot store with a staleness
                                   filter and an arb-opportunity query

The signal fires when the spread between the best bid and best ask across
venues, net of round-trip fees, exceeds a threshold. Side is always +1 —
these are delta-neutral arbs (buy low on one venue, sell high on another).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.signal_battery.base_signal import BaseSignalGenerator, Signal


# ---------------------------------------------------------------------------
# CrossExchangeArbSignal
# ---------------------------------------------------------------------------

class CrossExchangeArbSignal(BaseSignalGenerator):
    """
    Bar-level cross-exchange arbitrage detector.

    Accepts a bars DataFrame where each column is an exchange name and
    each value is the current price on that exchange (mid, or a single
    quoted price). At each bar:

        best_bid  = max(prices across exchanges)   # where you can sell
        best_ask  = min(prices across exchanges)   # where you can buy
        spread_bps = (best_bid - best_ask) / best_ask * 10000

    An arb signal fires when ``spread_bps`` exceeds ``min_spread_bps`` PLUS
    the estimated round-trip fees. Side is always +1 (delta-neutral);
    confidence scales with the profit net of fees.

    Parameters (via ``params``):
        min_spread_bps:   Minimum additional spread beyond fees to emit
                          a signal (default 10 bps).
        fee_estimate_bps: Estimated round-trip fees + slippage (default 20 bps).

    Required input columns: at least 2 exchange-price columns. Columns with
    non-finite or non-positive values at a given bar are skipped for that bar.
    """

    # Column requirements are dynamic (any 2+ numeric columns), so we use a
    # custom validator rather than REQUIRED_COLUMNS.
    REQUIRED_COLUMNS: tuple[str, ...] = ()
    DEFAULT_PARAMS = {"min_spread_bps": 10.0, "fee_estimate_bps": 20.0}

    def __init__(
        self,
        name: str = "cross_exchange_arb",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        if float(self.params["min_spread_bps"]) < 0:
            raise ValueError("min_spread_bps must be >= 0")
        if float(self.params["fee_estimate_bps"]) < 0:
            raise ValueError("fee_estimate_bps must be >= 0")

    # ------------------------------------------------------------------ API --

    def validate_input(self, bars: pd.DataFrame) -> bool:
        if not isinstance(bars, pd.DataFrame):
            raise ValueError("bars must be a pandas DataFrame")
        if bars.empty:
            raise ValueError(f"{self.name}: bars DataFrame is empty")
        if bars.shape[1] < 2:
            raise ValueError(
                f"{self.name}: need at least 2 exchange-price columns, "
                f"got {bars.shape[1]}"
            )
        return True

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
        **kwargs: Any,
    ) -> list[Signal]:
        if bars is None:
            raise ValueError(f"{self.name}: bars DataFrame is required")
        self.validate_input(bars)

        min_bps = float(self.params["min_spread_bps"])
        fee_bps = float(self.params["fee_estimate_bps"])
        threshold = min_bps + fee_bps

        exchanges = list(bars.columns)
        prices = bars.to_numpy(dtype=float)

        signals: list[Signal] = []
        for i, t in enumerate(bars.index):
            row = prices[i]
            # Consider only exchanges with a valid positive price this bar.
            mask = np.isfinite(row) & (row > 0)
            if mask.sum() < 2:
                continue
            valid_prices = row[mask]
            valid_names = [exchanges[j] for j in range(len(exchanges)) if mask[j]]

            sell_idx = int(np.argmax(valid_prices))
            buy_idx = int(np.argmin(valid_prices))
            if sell_idx == buy_idx:
                continue
            sell_px = float(valid_prices[sell_idx])
            buy_px = float(valid_prices[buy_idx])
            if buy_px <= 0:
                continue

            spread_bps = (sell_px - buy_px) / buy_px * 10_000.0
            if spread_bps <= threshold:
                continue

            est_profit_bps = spread_bps - fee_bps
            confidence = float(np.clip(est_profit_bps / 100.0, 0.0, 1.0))

            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=symbol,
                    family=self.name,
                    side=1,
                    confidence=confidence,
                    metadata={
                        "buy_exchange": valid_names[buy_idx],
                        "sell_exchange": valid_names[sell_idx],
                        "buy_price": buy_px,
                        "sell_price": sell_px,
                        "spread_bps": float(spread_bps),
                        "fee_estimate_bps": fee_bps,
                        "estimated_profit_bps": float(est_profit_bps),
                    },
                )
            )
        return signals


# ---------------------------------------------------------------------------
# MultiExchangePriceTracker
# ---------------------------------------------------------------------------

@dataclass
class _Quote:
    bid: float
    ask: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        """Midpoint of the bid/ask."""
        return (self.bid + self.ask) / 2.0


class MultiExchangePriceTracker:
    """
    In-memory real-time tracker for top-of-book quotes across exchanges.

    Intended use: feed tick-level bid/ask updates from each exchange, then
    query ``get_snapshot`` or ``get_arb_opportunities`` to drive execution
    decisions. A quote is considered stale if it hasn't been refreshed
    within ``stale_after`` seconds (default 5).

    Uses true bid/ask for arb detection: buy at the cheapest ask, sell at
    the richest bid. Profit is ``(best_bid - best_ask) / best_ask * 10000``
    basis points; when positive and above threshold, an opportunity exists.
    """

    def __init__(self, stale_after: float = 5.0) -> None:
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        self.stale_after = float(stale_after)
        # {exchange: {symbol: _Quote}}
        self._quotes: dict[str, dict[str, _Quote]] = {}

    # -- feeder API --

    def update(
        self,
        exchange: str,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: datetime,
    ) -> None:
        """Record the latest bid/ask on ``exchange`` for ``symbol``."""
        if bid <= 0 or ask <= 0:
            raise ValueError("bid and ask must be positive")
        if ask < bid:
            raise ValueError("ask must be >= bid")
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        self._quotes.setdefault(exchange, {})[symbol] = _Quote(
            bid=float(bid), ask=float(ask), timestamp=timestamp,
        )

    # -- query API --

    def get_snapshot(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> dict[str, dict]:
        """
        Fresh per-exchange snapshot for ``symbol``: dict of
        {exchange: {bid, ask, mid, timestamp}}. Stale quotes are excluded.

        Args:
            symbol: Instrument ticker.
            now:    Optional "current" time (for deterministic tests).
        """
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        max_age = timedelta(seconds=self.stale_after)

        snapshot: dict[str, dict] = {}
        for exchange, by_sym in self._quotes.items():
            q = by_sym.get(symbol)
            if q is None:
                continue
            if now - q.timestamp > max_age:
                continue
            snapshot[exchange] = {
                "bid": q.bid,
                "ask": q.ask,
                "mid": q.mid,
                "timestamp": q.timestamp,
            }
        return snapshot

    def get_arb_opportunities(
        self,
        symbol: str,
        min_spread_bps: float = 10.0,
        now: datetime | None = None,
    ) -> list[dict]:
        """
        Current cross-exchange arbs for ``symbol``.

        Returns a list of opportunity dicts (there's at most one simultaneous
        best-bid/best-ask arb per symbol — but the return type is a list for
        forward-compat). Each dict has keys: buy_exchange, sell_exchange,
        buy_price, sell_price, spread_bps.
        """
        snap = self.get_snapshot(symbol, now=now)
        if len(snap) < 2:
            return []

        # Buy at lowest ask; sell at highest bid.
        best_ask_ex = min(snap, key=lambda ex: snap[ex]["ask"])
        best_bid_ex = max(snap, key=lambda ex: snap[ex]["bid"])
        if best_ask_ex == best_bid_ex:
            return []

        buy_price = snap[best_ask_ex]["ask"]
        sell_price = snap[best_bid_ex]["bid"]
        if buy_price <= 0 or sell_price <= buy_price:
            return []

        spread_bps = (sell_price - buy_price) / buy_price * 10_000.0
        if spread_bps < float(min_spread_bps):
            return []

        return [{
            "buy_exchange": best_ask_ex,
            "sell_exchange": best_bid_ex,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "spread_bps": float(spread_bps),
        }]
