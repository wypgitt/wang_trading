"""Broker adapters (Phase 5 / P5.04).

Defines the abstract broker interface used by the execution engine and
provides concrete adapters:
    PaperBrokerAdapter  – simulated fills with slippage/delay (default for tests)
    AlpacaBrokerAdapter – equities (paper + live) via alpaca-trade-api
    CCXTBrokerAdapter   – crypto via ccxt (Binance/Coinbase/Kraken)

Only PaperBrokerAdapter is fully implemented here. The Alpaca/CCXT adapters
are thin skeletons that construct their SDK client lazily so unit tests can
import this module without the optional dependencies installed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from src.execution.models import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)

log = logging.getLogger(__name__)


class BrokerNetworkError(Exception):
    """Wraps transport errors (timeout, connection refused, rate limit) so
    callers can catch a single exception type regardless of the underlying
    SDK. Concrete adapters translate SDK errors into this."""


# ── Base interface ─────────────────────────────────────────────────────

class BaseBrokerAdapter(ABC):
    """Abstract broker adapter.

    `submit_order` returns the Order with any immediate fills already attached
    (fills also reported incrementally via `poll_fills`). `get_order_status`
    returns the latest snapshot.
    """

    @abstractmethod
    async def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order: ...

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    async def get_account(self) -> dict[str, Any]: ...

    @abstractmethod
    async def get_quote(self, symbol: str) -> dict[str, float]: ...

    @abstractmethod
    async def heartbeat(self) -> bool: ...

    # Convenience helpers used by execution algorithms.
    async def get_mid_price(self, symbol: str) -> float:
        q = await self.get_quote(symbol)
        return q["mid"]

    _poll_cursor: dict[str, int]

    async def poll_fills(self, order_id: str) -> list[Fill]:
        """Incremental fills since the last poll. Default impl diffs against
        `get_order_status(order_id).fills`. Concrete adapters may override for
        efficiency or to support streaming fill queues."""
        cursor_map = getattr(self, "_poll_cursor", None)
        if cursor_map is None:
            cursor_map = {}
            self._poll_cursor = cursor_map
        order = await self.get_order_status(order_id)
        cursor = cursor_map.get(order_id, 0)
        new = list(order.fills[cursor:])
        cursor_map[order_id] = len(order.fills)
        return new


# ── Position reconciliation helper ─────────────────────────────────────

@dataclass
class ReconciliationDiff:
    symbol: str
    internal_qty: float
    broker_qty: float
    internal_side: int
    broker_side: int

    @property
    def delta(self) -> float:
        return (self.broker_side * self.broker_qty) - (self.internal_side * self.internal_qty)


def reconcile_positions(
    internal: PortfolioState, broker_positions: dict[str, Position], *, tol: float = 1e-6
) -> list[ReconciliationDiff]:
    """Return the list of position differences between internal ledger and broker."""
    diffs: list[ReconciliationDiff] = []
    symbols = set(internal.positions) | set(broker_positions)
    for sym in symbols:
        ip = internal.positions.get(sym)
        bp = broker_positions.get(sym)
        iq, bq = (ip.quantity if ip else 0.0), (bp.quantity if bp else 0.0)
        iside, bside = (ip.side if ip else 0), (bp.side if bp else 0)
        signed_i = iside * iq
        signed_b = bside * bq
        if abs(signed_i - signed_b) > tol:
            diffs.append(ReconciliationDiff(sym, iq, bq, iside, bside))
    return diffs


# ── Paper broker ───────────────────────────────────────────────────────

class PaperBrokerAdapter(BaseBrokerAdapter):
    """In-memory simulated broker used for paper trading + tests."""

    def __init__(
        self,
        *,
        initial_cash: float = 100_000.0,
        slippage_bps: float = 2.0,
        fill_delay_ms: int = 100,
        price_feed: Callable[[str], float] | None = None,
        spread_bps: float = 5.0,
    ) -> None:
        self.cash = initial_cash
        self.slippage_bps = slippage_bps
        self.fill_delay_s = fill_delay_ms / 1000.0
        self.price_feed = price_feed or (lambda sym: 100.0)
        self.spread_bps = spread_bps
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}

    # Core interface
    async def submit_order(self, order: Order) -> Order:
        self.orders[order.order_id] = order
        order.status = OrderStatus.SUBMITTED
        if self.fill_delay_s:
            await asyncio.sleep(self.fill_delay_s)

        mid = float(self.price_feed(order.symbol))
        side = 1 if order.quantity > 0 else -1
        slip = mid * self.slippage_bps / 10_000.0
        fill_price = mid + side * slip

        # Respect limit: don't fill a buy above limit, or a sell below limit
        if order.order_type in (OrderType.LIMIT, OrderType.LIMIT_AT_MID, OrderType.ICEBERG):
            if order.limit_price is not None:
                if side > 0 and fill_price > order.limit_price:
                    # No fill; order stays SUBMITTED
                    return order
                if side < 0 and fill_price < order.limit_price:
                    return order
                fill_price = order.limit_price if order.order_type == OrderType.LIMIT_AT_MID \
                    else fill_price

        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            timestamp=datetime.now(timezone.utc),
            price=fill_price,
            quantity=order.quantity,
            commission=0.0,
            exchange="PAPER",
            is_maker=order.order_type != OrderType.MARKET,
        )
        order.add_fill(fill)
        self._apply_fill_to_ledger(order, fill)
        return order

    async def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order is None:
            return False
        if order.is_active:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order_status(self, order_id: str) -> Order:
        return self.orders[order_id]

    async def get_positions(self) -> dict[str, Position]:
        return dict(self.positions)

    async def get_account(self) -> dict[str, Any]:
        gross = sum(abs(p.market_value) for p in self.positions.values())
        net = sum(p.market_value for p in self.positions.values())
        nav = self.cash + net
        return {
            "cash": self.cash,
            "nav": nav,
            "gross_exposure": gross,
            "net_exposure": net,
            "position_count": len(self.positions),
        }

    async def get_quote(self, symbol: str) -> dict[str, float]:
        mid = float(self.price_feed(symbol))
        half_spread = mid * self.spread_bps / 20_000.0  # spread_bps / 2 / 10000
        return {
            "bid": mid - half_spread,
            "ask": mid + half_spread,
            "mid": mid,
            "spread": 2 * half_spread,
        }

    async def heartbeat(self) -> bool:
        return True

    # Internal helpers
    def _apply_fill_to_ledger(self, order: Order, fill: Fill) -> None:
        self.cash -= fill.quantity * fill.price + fill.commission
        pos = self.positions.get(order.symbol)
        if pos is None:
            side = 1 if fill.quantity > 0 else -1
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                side=side,
                quantity=abs(fill.quantity),
                avg_entry_price=fill.price,
                entry_timestamp=fill.timestamp,
                signal_family=order.signal_family,
                current_price=fill.price,
            )
        else:
            pos.apply_fill(fill)
            if pos.quantity <= 1e-12:
                del self.positions[order.symbol]


# ── Alpaca (skeleton) ──────────────────────────────────────────────────

class AlpacaBrokerAdapter(BaseBrokerAdapter):
    """Equities broker via Alpaca (paper + live).

    Lightweight skeleton: constructs the SDK client lazily. Full mapping of
    Order/Fill ↔ Alpaca schema is fleshed out in later phases; unit tests
    cover the interface contract only.
    """

    _ORDER_TYPE_MAP = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.LIMIT_AT_MID: "limit",
        OrderType.BRACKET: "limit",
    }

    def __init__(self, *, api_key: str, api_secret: str, paper: bool = True) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "alpaca-py not installed; pip install alpaca-py"
                ) from exc
            self._client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        return self._client

    async def submit_order(self, order: Order) -> Order:  # pragma: no cover - integration
        raise NotImplementedError("Alpaca live submit wired up in integration tests")

    async def cancel_order(self, order_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def get_order_status(self, order_id: str) -> Order:  # pragma: no cover
        raise NotImplementedError

    async def get_positions(self) -> dict[str, Position]:  # pragma: no cover
        raise NotImplementedError

    async def get_account(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    async def get_quote(self, symbol: str) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    async def heartbeat(self) -> bool:  # pragma: no cover
        try:
            _ = self.client
            return True
        except Exception:
            return False

    def reconcile_against(
        self, internal: PortfolioState, broker_positions: dict[str, Position]
    ) -> list[ReconciliationDiff]:
        return reconcile_positions(internal, broker_positions)


# ── CCXT (skeleton) ────────────────────────────────────────────────────

class CCXTBrokerAdapter(BaseBrokerAdapter):
    """Crypto broker via CCXT (Binance/Coinbase/Kraken).

    Skeleton: constructs exchange client lazily. Fleshed out in live-trading
    integration tests.
    """

    def __init__(
        self,
        *,
        exchange: str,
        api_key: str = "",
        api_secret: str = "",
        lot_size_map: dict[str, float] | None = None,
        min_order_size_map: dict[str, float] | None = None,
    ) -> None:
        self.exchange = exchange
        self.api_key = api_key
        self.api_secret = api_secret
        self.lot_size_map = lot_size_map or {}
        self.min_order_size_map = min_order_size_map or {}
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import ccxt.async_support as ccxt  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ccxt not installed; pip install ccxt") from exc
            cls = getattr(ccxt, self.exchange)
            self._client = cls({"apiKey": self.api_key, "secret": self.api_secret})
        return self._client

    def round_to_lot(self, symbol: str, qty: float) -> float:
        lot = self.lot_size_map.get(symbol, 0.0)
        if lot <= 0:
            return qty
        n = round(qty / lot)
        return n * lot

    def meets_min_size(self, symbol: str, qty: float) -> bool:
        min_sz = self.min_order_size_map.get(symbol, 0.0)
        return abs(qty) >= min_sz

    async def submit_order(self, order: Order) -> Order:  # pragma: no cover - integration
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def get_order_status(self, order_id: str) -> Order:  # pragma: no cover
        raise NotImplementedError

    async def get_positions(self) -> dict[str, Position]:  # pragma: no cover
        raise NotImplementedError

    async def get_account(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    async def get_quote(self, symbol: str) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    async def heartbeat(self) -> bool:  # pragma: no cover
        return self._client is not None
