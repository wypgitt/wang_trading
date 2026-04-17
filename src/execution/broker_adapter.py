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
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from src.execution.models import (
    ExecutionAlgo,
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

    The SDK client is constructed lazily on first access so this module can be
    imported without ``alpaca-py`` installed. Live trading is gated by the
    ``WANG_ALLOW_LIVE_TRADING=yes`` environment variable.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    LIVE_ENV_GATE = "WANG_ALLOW_LIVE_TRADING"

    _ORDER_TYPE_MAP = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.LIMIT_AT_MID: "limit",
        OrderType.BRACKET: "limit",
    }

    _TIF_MAP = {
        ExecutionAlgo.IMMEDIATE: "day",
        ExecutionAlgo.TWAP: "day",
        ExecutionAlgo.VWAP: "day",
        ExecutionAlgo.ICEBERG: "day",
    }

    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str | None = None,
        paper: bool = True,
        base_url: str | None = None,
        api_secret: str | None = None,  # backward-compat alias
    ) -> None:
        secret = secret_key if secret_key is not None else api_secret
        if secret is None:
            raise TypeError("secret_key is required")
        self.api_key = api_key
        self.secret_key = secret
        # Kept as attribute under both names for backwards compatibility.
        self.api_secret = secret
        self.paper = paper
        if not paper:
            self._assert_live_allowed()
        self.base_url = base_url or (self.PAPER_URL if paper else self.LIVE_URL)
        self._client: Any = None
        self._data_client: Any = None
        self._fill_cursor: dict[str, float] = {}
        self._bracket_children: dict[str, list[str]] = {}

    # ── Live-trading gate ─────────────────────────────────────────────

    @classmethod
    def _assert_live_allowed(cls) -> None:
        if os.environ.get(cls.LIVE_ENV_GATE, "") != "yes":
            raise ValueError(
                f"Live trading blocked: set env var {cls.LIVE_ENV_GATE}=yes to enable."
            )

    # ── Lazy SDK clients ──────────────────────────────────────────────

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "alpaca-py not installed; pip install alpaca-py"
                ) from exc
            self._client = TradingClient(
                self.api_key, self.secret_key, paper=self.paper
            )
        return self._client

    @property
    def data_client(self) -> Any:
        if self._data_client is None:
            try:
                from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "alpaca-py not installed; pip install alpaca-py"
                ) from exc
            self._data_client = StockHistoricalDataClient(
                self.api_key, self.secret_key
            )
        return self._data_client

    # ── Request builders ──────────────────────────────────────────────

    def _build_order_request(self, order: Order) -> Any:
        """Translate our Order into an alpaca-py request object."""
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore
        from alpaca.trading.requests import (  # type: ignore
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        side_enum = OrderSide.BUY if order.side > 0 else OrderSide.SELL
        tif_str = self._TIF_MAP.get(order.execution_algo, "day").upper()
        tif = getattr(TimeInForce, tif_str, TimeInForce.DAY)
        qty = abs(order.quantity)

        is_bracket = (
            order.order_type == OrderType.BRACKET
            or (order.take_profit_price is not None and order.stop_price is not None)
        )

        if is_bracket:
            if order.limit_price is None:
                raise ValueError("Bracket order requires limit_price")
            if order.take_profit_price is None or order.stop_price is None:
                raise ValueError(
                    "Bracket order requires both take_profit_price and stop_price"
                )
            return LimitOrderRequest(
                symbol=order.symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                limit_price=order.limit_price,
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                stop_loss=StopLossRequest(stop_price=order.stop_price),
            )

        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=order.symbol, qty=qty, side=side_enum, time_in_force=tif,
            )

        if order.order_type in (OrderType.LIMIT, OrderType.LIMIT_AT_MID, OrderType.ICEBERG):
            if order.limit_price is None:
                raise ValueError(f"{order.order_type} order requires limit_price")
            return LimitOrderRequest(
                symbol=order.symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                limit_price=order.limit_price,
            )

        if order.stop_price is not None and order.limit_price is not None:
            return StopLimitOrderRequest(
                symbol=order.symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )

        # Fallback: market order
        return MarketOrderRequest(
            symbol=order.symbol, qty=qty, side=side_enum, time_in_force=tif,
        )

    # ── Response parsing ──────────────────────────────────────────────

    @staticmethod
    def _alpaca_status_to_ours(status: Any) -> OrderStatus:
        s = str(getattr(status, "value", status)).lower()
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL_FILL,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(s, OrderStatus.SUBMITTED)

    def _apply_alpaca_response(self, order: Order, resp: Any) -> Order:
        order.order_id = str(getattr(resp, "id", order.order_id))
        order.status = self._alpaca_status_to_ours(getattr(resp, "status", "new"))
        filled_qty = float(getattr(resp, "filled_qty", 0) or 0)
        if filled_qty > 0:
            avg_price = float(getattr(resp, "filled_avg_price", 0) or 0)
            signed = filled_qty if order.side > 0 else -filled_qty
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                timestamp=datetime.now(timezone.utc),
                price=avg_price,
                quantity=signed,
                commission=0.0,
                exchange="ALPACA",
                is_maker=False,
            )
            order.add_fill(fill)
        legs = getattr(resp, "legs", None) or []
        if legs:
            self._bracket_children[order.order_id] = [str(getattr(leg, "id", "")) for leg in legs]
        return order

    # ── Broker interface ──────────────────────────────────────────────

    async def submit_order(self, order: Order) -> Order:
        req = self._build_order_request(order)
        try:
            resp = await asyncio.to_thread(self.client.submit_order, req)
        except Exception as exc:
            raise BrokerNetworkError(f"Alpaca submit_order failed: {exc}") from exc
        return self._apply_alpaca_response(order, resp)

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await asyncio.to_thread(self.client.cancel_order_by_id, order_id)
            return True
        except Exception as exc:
            raise BrokerNetworkError(f"Alpaca cancel_order failed: {exc}") from exc

    async def get_order_status(self, order_id: str) -> Order:
        try:
            resp = await asyncio.to_thread(self.client.get_order_by_id, order_id)
        except Exception as exc:
            raise BrokerNetworkError(f"Alpaca get_order_by_id failed: {exc}") from exc
        qty = float(getattr(resp, "qty", 0) or 0)
        side_str = str(getattr(getattr(resp, "side", ""), "value", getattr(resp, "side", ""))).lower()
        side = 1 if side_str.startswith("b") else -1
        order = Order(
            order_id=str(getattr(resp, "id", order_id)),
            timestamp=datetime.now(timezone.utc),
            symbol=str(getattr(resp, "symbol", "")),
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty * side,
        )
        return self._apply_alpaca_response(order, resp)

    async def get_positions(self) -> dict[str, Position]:
        try:
            resp = await asyncio.to_thread(self.client.get_all_positions)
        except Exception as exc:
            raise BrokerNetworkError(f"Alpaca get_all_positions failed: {exc}") from exc
        positions: dict[str, Position] = {}
        for p in resp or []:
            qty = float(getattr(p, "qty", 0) or 0)
            side = 1 if qty >= 0 else -1
            symbol = str(getattr(p, "symbol", ""))
            positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=abs(qty),
                avg_entry_price=float(getattr(p, "avg_entry_price", 0) or 0),
                entry_timestamp=datetime.now(timezone.utc),
                signal_family="",
                current_price=float(getattr(p, "current_price", 0) or 0),
                unrealized_pnl=float(getattr(p, "unrealized_pl", 0) or 0),
            )
        return positions

    async def get_account(self) -> dict[str, Any]:
        try:
            acct = await asyncio.to_thread(self.client.get_account)
        except Exception as exc:
            raise BrokerNetworkError(f"Alpaca get_account failed: {exc}") from exc
        return {
            "cash": float(getattr(acct, "cash", 0) or 0),
            "nav": float(getattr(acct, "equity", 0) or 0),
            "buying_power": float(getattr(acct, "buying_power", 0) or 0),
            "gross_exposure": float(getattr(acct, "long_market_value", 0) or 0)
            + abs(float(getattr(acct, "short_market_value", 0) or 0)),
            "net_exposure": float(getattr(acct, "long_market_value", 0) or 0)
            + float(getattr(acct, "short_market_value", 0) or 0),
            "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
        }

    async def get_quote(self, symbol: str) -> dict[str, float]:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("alpaca-py not installed") from exc
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        try:
            resp = await asyncio.to_thread(self.data_client.get_stock_latest_quote, req)
        except Exception as exc:
            raise BrokerNetworkError(f"Alpaca get_stock_latest_quote failed: {exc}") from exc
        q = resp[symbol] if isinstance(resp, dict) else resp
        bid = float(getattr(q, "bid_price", 0) or 0)
        ask = float(getattr(q, "ask_price", 0) or 0)
        mid = (bid + ask) / 2.0 if (bid and ask) else (bid or ask)
        return {"bid": bid, "ask": ask, "mid": mid, "spread": max(ask - bid, 0.0)}

    async def heartbeat(self) -> bool:
        try:
            await asyncio.to_thread(self.client.get_clock)
            return True
        except Exception:
            return False

    async def poll_fills(self, order_id: str) -> list[Fill]:
        """Return fills observed since the last poll.

        Alpaca's REST endpoint exposes only the cumulative filled quantity and
        average fill price; we track the cumulative filled quantity in an
        internal cursor and emit a synthetic Fill for the delta.
        """
        order = await self.get_order_status(order_id)
        new: list[Fill] = []
        for f in order.fills:
            seen = self._fill_cursor.get(order_id, 0.0)
            delta = abs(f.quantity) - seen
            if delta > 1e-9:
                synthetic = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order_id,
                    timestamp=f.timestamp,
                    price=f.price,
                    quantity=delta if f.quantity > 0 else -delta,
                    commission=0.0,
                    exchange=f.exchange,
                )
                new.append(synthetic)
                self._fill_cursor[order_id] = abs(f.quantity)
        return new

    def reconcile_against(
        self, internal: PortfolioState, broker_positions: dict[str, Position]
    ) -> list[ReconciliationDiff]:
        return reconcile_positions(internal, broker_positions)


# ── CCXT (skeleton) ────────────────────────────────────────────────────

class CCXTBrokerAdapter(BaseBrokerAdapter):
    """Crypto broker via CCXT (Binance/Coinbase/Kraken/Bybit).

    Uses ``ccxt.async_support`` so all calls are awaitable. Live trading is
    gated by the ``WANG_ALLOW_LIVE_CRYPTO=yes`` environment variable.
    """

    LIVE_ENV_GATE = "WANG_ALLOW_LIVE_CRYPTO"
    SUPPORTED_EXCHANGES = {"binance", "coinbase", "coinbasepro", "kraken", "bybit"}

    _ORDER_TYPE_MAP = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.LIMIT_AT_MID: "limit",
        OrderType.TWAP: "limit",
        OrderType.VWAP: "limit",
        OrderType.ICEBERG: "limit",
        OrderType.BRACKET: "limit",
    }

    def __init__(
        self,
        *,
        exchange_name: str | None = None,
        api_key: str = "",
        secret_key: str | None = None,
        sandbox: bool = True,
        passphrase: str | None = None,
        options: dict[str, Any] | None = None,
        # Backwards-compatibility for P5 skeleton call sites / tests:
        exchange: str | None = None,
        api_secret: str | None = None,
        lot_size_map: dict[str, float] | None = None,
        min_order_size_map: dict[str, float] | None = None,
    ) -> None:
        name = exchange_name if exchange_name is not None else exchange
        if name is None:
            raise TypeError("exchange_name is required")
        self.exchange_name = name
        self.exchange = name
        self.api_key = api_key
        secret = secret_key if secret_key is not None else api_secret
        self.secret_key = secret or ""
        self.api_secret = self.secret_key
        self.sandbox = sandbox
        self.passphrase = passphrase
        self.options = dict(options) if options else {}
        self.lot_size_map = lot_size_map or {}
        self.min_order_size_map = min_order_size_map or {}
        self._client: Any = None
        self._markets_loaded = False
        self._fill_cursor: dict[str, float] = {}

        if not sandbox:
            self._assert_live_allowed()

    # ── Live-trading gate ─────────────────────────────────────────────

    @classmethod
    def _assert_live_allowed(cls) -> None:
        if os.environ.get(cls.LIVE_ENV_GATE, "") != "yes":
            raise ValueError(
                f"Live crypto trading blocked: set env var {cls.LIVE_ENV_GATE}=yes to enable."
            )

    # ── Lazy client ───────────────────────────────────────────────────

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import ccxt.async_support as ccxt  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ccxt not installed; pip install ccxt") from exc
            cls = getattr(ccxt, self.exchange_name)
            config: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.secret_key,
                "enableRateLimit": True,
                "options": self.options,
            }
            if self.passphrase is not None:
                config["password"] = self.passphrase
            self._client = cls(config)
            if self.sandbox and hasattr(self._client, "set_sandbox_mode"):
                try:
                    self._client.set_sandbox_mode(True)
                except Exception:  # pragma: no cover - not all exchanges support it
                    pass
        return self._client

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            try:
                await self.client.load_markets()
            except Exception as exc:
                raise BrokerNetworkError(f"CCXT load_markets failed: {exc}") from exc
            self._markets_loaded = True

    # ── Symbol handling ───────────────────────────────────────────────

    def normalize_symbol(self, symbol: str) -> str:
        """Translate exchange-specific symbols into CCXT's unified format.

        Kraken publishes ``XBTUSD`` / ``XBT/USD`` for Bitcoin but CCXT
        normalizes to ``BTC/USD``. Symbols already in ``BASE/QUOTE`` unified
        format pass through unchanged.
        """
        s = symbol.strip()
        if self.exchange_name == "kraken":
            s = s.replace("XBT", "BTC")
        if "/" in s:
            return s
        # Best-effort split for concatenated symbols (e.g. "BTCUSD").
        for quote in ("USDT", "USDC", "USD", "EUR", "BTC", "ETH"):
            if s.endswith(quote) and len(s) > len(quote):
                return f"{s[:-len(quote)]}/{quote}"
        return s

    # ── Lot + min-size helpers ────────────────────────────────────────

    def round_to_lot(self, symbol: str, qty: float) -> float:
        lot = self.lot_size_map.get(symbol, 0.0)
        if lot <= 0:
            return qty
        n = round(qty / lot)
        return n * lot

    def meets_min_size(self, symbol: str, qty: float) -> bool:
        min_sz = self.min_order_size_map.get(symbol, 0.0)
        return abs(qty) >= min_sz

    def _check_market_min(self, symbol: str, qty: float, price: float | None) -> str | None:
        """Return a rejection reason string if order is below exchange minimum,
        else ``None``. Uses CCXT's ``markets[symbol]['limits']`` structure."""
        markets = getattr(self.client, "markets", {}) or {}
        info = markets.get(symbol)
        if not info:
            return None
        limits = info.get("limits", {}) or {}
        amt_min = (limits.get("amount") or {}).get("min")
        if amt_min is not None and abs(qty) < float(amt_min):
            return f"below min amount {amt_min}"
        cost_min = (limits.get("cost") or {}).get("min")
        if cost_min is not None and price is not None:
            if abs(qty) * price < float(cost_min):
                return f"below min cost {cost_min}"
        return None

    def _apply_precision(self, symbol: str, qty: float, price: float | None) -> tuple[float, float | None]:
        c = self.client
        try:
            qty = float(c.amount_to_precision(symbol, qty))
        except Exception:
            pass
        if price is not None:
            try:
                price = float(c.price_to_precision(symbol, price))
            except Exception:
                pass
        return qty, price

    async def _mid_price(self, symbol: str) -> float:
        try:
            ob = await self.client.fetch_order_book(symbol, limit=5)
        except Exception as exc:
            raise BrokerNetworkError(f"CCXT fetch_order_book failed: {exc}") from exc
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            raise BrokerNetworkError(f"Empty order book for {symbol}")
        return (float(bids[0][0]) + float(asks[0][0])) / 2.0

    # ── Broker interface ──────────────────────────────────────────────

    async def submit_order(self, order: Order) -> Order:
        await self._ensure_markets()
        symbol = self.normalize_symbol(order.symbol)
        side = "buy" if order.side > 0 else "sell"
        ccxt_type = self._ORDER_TYPE_MAP.get(order.order_type, "limit")

        price: float | None = order.limit_price
        if order.order_type == OrderType.LIMIT_AT_MID:
            price = await self._mid_price(symbol)

        qty = abs(order.quantity)
        qty, price = self._apply_precision(symbol, qty, price)

        reason = self._check_market_min(symbol, qty, price)
        if reason is not None:
            order.status = OrderStatus.REJECTED
            log.warning("CCXT rejecting %s qty=%s: %s", symbol, qty, reason)
            return order

        params: dict[str, Any] = {}
        try:
            resp = await self.client.create_order(
                symbol, ccxt_type, side, qty, price, params
            )
        except Exception as exc:
            raise BrokerNetworkError(f"CCXT create_order failed: {exc}") from exc

        return self._apply_ccxt_response(order, resp)

    @staticmethod
    def _ccxt_status_to_ours(status: str) -> OrderStatus:
        mapping = {
            "open": OrderStatus.SUBMITTED,
            "new": OrderStatus.SUBMITTED,
            "partial": OrderStatus.PARTIAL_FILL,
            "partially_filled": OrderStatus.PARTIAL_FILL,
            "closed": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get((status or "").lower(), OrderStatus.SUBMITTED)

    def _apply_ccxt_response(self, order: Order, resp: dict[str, Any]) -> Order:
        order.order_id = str(resp.get("id") or order.order_id)
        order.status = self._ccxt_status_to_ours(str(resp.get("status", "")))
        filled_qty = float(resp.get("filled") or 0.0)
        avg_price = float(resp.get("average") or resp.get("price") or 0.0)
        fee = resp.get("fee") or {}
        commission = float(fee.get("cost") or 0.0)
        if filled_qty > 0:
            signed = filled_qty if order.side > 0 else -filled_qty
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                timestamp=datetime.now(timezone.utc),
                price=avg_price,
                quantity=signed,
                commission=commission,
                exchange=self.exchange_name.upper(),
            )
            order.add_fill(fill)
        return order

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        try:
            await self.client.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            raise BrokerNetworkError(f"CCXT cancel_order failed: {exc}") from exc

    async def get_order_status(self, order_id: str, symbol: str | None = None) -> Order:
        try:
            resp = await self.client.fetch_order(order_id, symbol)
        except Exception as exc:
            raise BrokerNetworkError(f"CCXT fetch_order failed: {exc}") from exc
        side_str = str(resp.get("side") or "buy").lower()
        side = 1 if side_str.startswith("b") else -1
        qty = float(resp.get("amount") or 0.0)
        order = Order(
            order_id=str(resp.get("id") or order_id),
            timestamp=datetime.now(timezone.utc),
            symbol=str(resp.get("symbol") or (symbol or "")),
            side=side,
            order_type=OrderType.LIMIT,
            quantity=qty * side,
            limit_price=resp.get("price"),
        )
        return self._apply_ccxt_response(order, resp)

    async def get_positions(self) -> dict[str, Position]:
        """Return spot balances as Positions. For perpetuals, merge in the
        exchange's positions endpoint when available."""
        positions: dict[str, Position] = {}
        try:
            bal = await self.client.fetch_balance()
        except Exception as exc:
            raise BrokerNetworkError(f"CCXT fetch_balance failed: {exc}") from exc

        totals = bal.get("total") or {}
        for asset, amount in totals.items():
            qty = float(amount or 0.0)
            if qty <= 0 or asset in ("USD", "USDT", "USDC", "EUR"):
                continue
            positions[asset] = Position(
                symbol=asset,
                side=1,
                quantity=qty,
                avg_entry_price=0.0,
                entry_timestamp=datetime.now(timezone.utc),
                signal_family="",
            )

        # Derivatives: fetch_positions where supported.
        if getattr(self.client, "has", {}).get("fetchPositions"):
            try:
                perp_positions = await self.client.fetch_positions()
            except Exception:
                perp_positions = []
            for p in perp_positions or []:
                symbol = str(p.get("symbol") or "")
                contracts = float(p.get("contracts") or 0.0)
                if contracts == 0 or not symbol:
                    continue
                side_str = str(p.get("side") or "long").lower()
                side = 1 if side_str.startswith("l") else -1
                positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=abs(contracts),
                    avg_entry_price=float(p.get("entryPrice") or 0.0),
                    entry_timestamp=datetime.now(timezone.utc),
                    signal_family="",
                    current_price=float(p.get("markPrice") or 0.0),
                    unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
                )
        return positions

    async def get_account(self) -> dict[str, Any]:
        try:
            bal = await self.client.fetch_balance()
        except Exception as exc:
            raise BrokerNetworkError(f"CCXT fetch_balance failed: {exc}") from exc
        quote = "USDT"
        free = (bal.get("free") or {}).get(quote, 0.0)
        total_quote = (bal.get("total") or {}).get(quote, 0.0)
        return {
            "cash": float(free or 0.0),
            "nav": float(total_quote or 0.0),
            "exchange": self.exchange_name,
        }

    async def get_quote(self, symbol: str) -> dict[str, float]:
        sym = self.normalize_symbol(symbol)
        client = self.client
        if getattr(client, "has", {}).get("fetchTicker", True):
            try:
                t = await client.fetch_ticker(sym)
            except Exception as exc:
                raise BrokerNetworkError(f"CCXT fetch_ticker failed: {exc}") from exc
            bid = float(t.get("bid") or 0.0)
            ask = float(t.get("ask") or 0.0)
            mid = (bid + ask) / 2.0 if (bid and ask) else float(t.get("last") or 0.0)
            return {"bid": bid, "ask": ask, "mid": mid, "spread": max(ask - bid, 0.0)}
        # Fallback via order book
        mid = await self._mid_price(sym)
        return {"bid": mid, "ask": mid, "mid": mid, "spread": 0.0}

    async def heartbeat(self) -> bool:
        try:
            if getattr(self.client, "has", {}).get("fetchStatus"):
                status = await self.client.fetch_status()
                return str((status or {}).get("status", "ok")).lower() == "ok"
            await self.client.fetch_time()
            return True
        except Exception:
            return False

    async def get_funding_rate(self, symbol: str) -> float:
        """Current annualized funding rate for a perpetual contract."""
        sym = self.normalize_symbol(symbol)
        try:
            resp = await self.client.fetch_funding_rate(sym)
        except Exception as exc:
            raise BrokerNetworkError(f"CCXT fetch_funding_rate failed: {exc}") from exc
        rate = resp.get("fundingRate")
        if rate is None:
            return 0.0
        # 8-hour funding → 3 periods/day → annualized.
        interval_hours = float(resp.get("fundingInterval") or 8.0)
        periods_per_year = (24.0 / interval_hours) * 365.0
        return float(rate) * periods_per_year

    async def get_supported_symbols(self) -> list[str]:
        await self._ensure_markets()
        markets = getattr(self.client, "markets", {}) or {}
        return sorted(markets.keys())

    async def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception:  # pragma: no cover
                pass
