"""Interactive Brokers adapter for futures trading (P6.03).

Uses ``ib_insync`` to talk to TWS / IB Gateway. Live trading is gated by the
``WANG_ALLOW_LIVE_FUTURES=yes`` environment variable. Contract specifications
are loaded from ``config/futures_contracts.yaml`` (see the example file).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from src.execution.broker_adapter import (
    BaseBrokerAdapter,
    BrokerNetworkError,
    ReconciliationDiff,
    reconcile_positions,
)
from src.execution.models import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)

log = logging.getLogger(__name__)


# ── Contract-month helpers ────────────────────────────────────────────────

MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
CODE_FOR_MONTH = {m: code for code, m in MONTH_CODES.items()}


def _third_friday(year: int, month: int) -> datetime:
    """Third Friday of ``(year, month)`` at 00:00 UTC — the standard expiry
    convention for index futures / options."""
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    # Monday=0 … Friday=4
    days_ahead = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_ahead)
    return first_friday + timedelta(days=14)


def contract_expiry(symbol: str) -> datetime:
    """Approximate expiry date for a contract symbol like ``ESZ25``.

    Uses the 3rd-Friday convention. Good enough for roll-detection heuristics;
    exact expiry can be queried from IBKR via ``reqContractDetails``.
    """
    if len(symbol) < 4:
        raise ValueError(f"Invalid contract symbol: {symbol!r}")
    # Trailing 1 or 2 digit year
    year_part = ""
    i = len(symbol) - 1
    while i >= 0 and symbol[i].isdigit():
        year_part = symbol[i] + year_part
        i -= 1
    month_code = symbol[i]
    base = symbol[:i]
    if month_code not in MONTH_CODES or not year_part or not base:
        raise ValueError(f"Invalid contract symbol: {symbol!r}")
    year = int(year_part)
    if year < 100:
        year += 2000
    return _third_friday(year, MONTH_CODES[month_code])


# ── Futures contract registry ─────────────────────────────────────────────

class FuturesContractRegistry:
    """Loads futures contract specs from YAML and answers queries about
    tick sizes, session hours, and front/next month symbols."""

    DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "futures_contracts.yaml"
    EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "config" / "futures_contracts.example.yaml"

    def __init__(self, path: str | Path | None = None,
                 specs: dict[str, dict[str, Any]] | None = None) -> None:
        if specs is not None:
            self._specs = dict(specs)
            self._path = None
            return
        if path is not None:
            self._path = Path(path)
        elif self.DEFAULT_PATH.exists():
            self._path = self.DEFAULT_PATH
        else:
            self._path = self.EXAMPLE_PATH
        with self._path.open() as fh:
            self._specs = yaml.safe_load(fh) or {}

    # ── Spec lookup ───────────────────────────────────────────────────

    def symbols(self) -> list[str]:
        return list(self._specs.keys())

    def get_spec(self, symbol: str) -> dict[str, Any]:
        base = self._base_symbol(symbol)
        if base not in self._specs:
            raise KeyError(f"Unknown futures symbol: {symbol!r}")
        return dict(self._specs[base])

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        """``ESZ25`` → ``ES`` (strip trailing month-code + year digits)."""
        s = symbol.rstrip("0123456789")
        if s and s[-1] in MONTH_CODES:
            s = s[:-1]
        return s or symbol

    # ── Front / next month ────────────────────────────────────────────

    def get_front_month(self, symbol: str, now: datetime | None = None) -> str:
        spec = self.get_spec(symbol)
        base = self._base_symbol(symbol)
        now = now or datetime.now(timezone.utc)
        months = [MONTH_CODES[c] for c in spec["months"]]
        year = now.year
        for m in months:
            expiry = _third_friday(year, m)
            if expiry > now:
                return f"{base}{CODE_FOR_MONTH[m]}{year % 100:02d}"
        # All months of this year are past — roll to first month of next year
        first = months[0]
        return f"{base}{CODE_FOR_MONTH[first]}{(year + 1) % 100:02d}"

    def get_next_month(self, symbol: str, now: datetime | None = None) -> str:
        """Contract immediately after the current front month."""
        spec = self.get_spec(symbol)
        base = self._base_symbol(symbol)
        now = now or datetime.now(timezone.utc)
        months = [MONTH_CODES[c] for c in spec["months"]]
        year = now.year
        found_front = False
        for m in months:
            expiry = _third_friday(year, m)
            if found_front:
                return f"{base}{CODE_FOR_MONTH[m]}{year % 100:02d}"
            if expiry > now:
                found_front = True
        # Next contract is in the following year
        first = months[0]
        return f"{base}{CODE_FOR_MONTH[first]}{(year + 1) % 100:02d}"

    # ── Session hours ─────────────────────────────────────────────────

    def is_within_rth(self, symbol: str, now: datetime | None = None) -> bool:
        spec = self.get_spec(symbol)
        hours = spec.get("session_hours") or {}
        tz = ZoneInfo(hours.get("timezone", "UTC"))
        now = now or datetime.now(timezone.utc)
        local = now.astimezone(tz)
        if local.weekday() >= 5:  # Saturday / Sunday
            return False
        start = dtime.fromisoformat(hours.get("rth_start", "00:00"))
        end = dtime.fromisoformat(hours.get("rth_end", "23:59"))
        return start <= local.time() <= end


# ── Adapter ───────────────────────────────────────────────────────────────

@dataclass
class RollCandidate:
    symbol: str
    expiry: datetime
    days_left: int
    next_symbol: str


class IBKRBrokerAdapter(BaseBrokerAdapter):
    """Futures broker via Interactive Brokers (TWS / IB Gateway)."""

    LIVE_ENV_GATE = "WANG_ALLOW_LIVE_FUTURES"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        client_id: int = 1,
        account_id: str = "",
        live: bool = False,
        allow_extended: bool = False,
        registry: FuturesContractRegistry | None = None,
    ) -> None:
        self.host = host
        self.port = port if port is not None else (7496 if live else 7497)
        self.client_id = client_id
        self.account_id = account_id
        self.live = live
        self.allow_extended = allow_extended
        self.registry = registry or FuturesContractRegistry()
        self._ib: Any = None
        self._orders: dict[str, Order] = {}
        self._fill_cursor: dict[str, int] = {}
        # Tracked contracts for roll detection: symbol → expiry datetime
        self._active_contracts: dict[str, datetime] = {}

        if live:
            self._assert_live_allowed()

    # ── Live gate ─────────────────────────────────────────────────────

    @classmethod
    def _assert_live_allowed(cls) -> None:
        if os.environ.get(cls.LIVE_ENV_GATE, "") != "yes":
            raise ValueError(
                f"Live futures trading blocked: set {cls.LIVE_ENV_GATE}=yes to enable."
            )

    # ── Lazy IB client ────────────────────────────────────────────────

    @property
    def ib(self) -> Any:
        if self._ib is None:
            try:
                from ib_insync import IB  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ib_insync not installed; pip install ib_insync") from exc
            self._ib = IB()
            try:
                self._ib.connect(self.host, self.port, clientId=self.client_id)
            except Exception as exc:
                raise BrokerNetworkError(f"IBKR connect failed: {exc}") from exc
        return self._ib

    # ── Contract specs ────────────────────────────────────────────────

    async def get_contract_specs(self, symbol: str) -> dict[str, Any]:
        spec = self.registry.get_spec(symbol)
        return {
            "tick_size": spec["tick_size"],
            "multiplier": spec["multiplier"],
            "exchange": spec["exchange"],
            "currency": spec["currency"],
            "contract_month": self.registry.get_front_month(symbol),
            "margin_requirement": spec.get("margin_requirement"),
        }

    # ── Session hours ─────────────────────────────────────────────────

    def _session_check(self, symbol: str, now: datetime | None = None) -> bool:
        if self.allow_extended:
            return True
        try:
            return self.registry.is_within_rth(symbol, now=now)
        except KeyError:
            return True  # Unknown symbol — let the exchange decide

    # ── Contract tracking (for roll detection) ───────────────────────

    def track_contract(self, symbol: str, expiry: datetime | None = None) -> None:
        self._active_contracts[symbol] = expiry or contract_expiry(symbol)

    async def check_roll_needed(
        self, *, threshold_days: int = 10, now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Contracts held whose expiry is within ``threshold_days``."""
        now = now or datetime.now(timezone.utc)
        out: list[dict[str, Any]] = []
        for symbol, expiry in self._active_contracts.items():
            days = (expiry - now).days
            if days <= threshold_days:
                try:
                    nxt = self.registry.get_next_month(symbol, now=now)
                except Exception:
                    nxt = symbol
                out.append({
                    "symbol": symbol,
                    "expiry": expiry,
                    "days_left": days,
                    "next_symbol": nxt,
                })
        return out

    async def execute_roll(
        self, old_contract: str, new_contract: str, position: Position,
    ) -> list[Order]:
        """Close the position in ``old_contract`` and open an equivalent one in
        ``new_contract``. Returns the two submitted orders.
        """
        close_qty = -position.side * position.quantity
        open_qty = position.side * position.quantity
        close_order = Order(
            order_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol=old_contract,
            side=-position.side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
            signal_family=position.signal_family,
        )
        open_order = Order(
            order_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol=new_contract,
            side=position.side,
            order_type=OrderType.MARKET,
            quantity=open_qty,
            signal_family=position.signal_family,
        )
        close_order = await self.submit_order(close_order)
        open_order = await self.submit_order(open_order)
        # Rotate internal tracking
        self._active_contracts.pop(old_contract, None)
        try:
            self._active_contracts[new_contract] = contract_expiry(new_contract)
        except ValueError:
            pass
        return [close_order, open_order]

    # ── Broker interface ──────────────────────────────────────────────

    def _build_ib_contract(self, symbol: str) -> Any:
        from ib_insync import Future  # type: ignore
        spec = self.registry.get_spec(symbol)
        base = FuturesContractRegistry._base_symbol(symbol)
        # "ESZ25" → lastTradeDateOrContractMonth = "202512"
        expiry = contract_expiry(symbol)
        return Future(
            symbol=base,
            lastTradeDateOrContractMonth=f"{expiry.year:04d}{expiry.month:02d}",
            exchange=spec["exchange"],
            currency=spec["currency"],
        )

    def _build_ib_order(self, order: Order) -> Any:
        from ib_insync import LimitOrder, MarketOrder, StopOrder  # type: ignore
        action = "BUY" if order.side > 0 else "SELL"
        qty = abs(order.quantity)
        if order.order_type == OrderType.MARKET:
            return MarketOrder(action, qty)
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            return LimitOrder(action, qty, order.limit_price)
        if order.stop_price is not None:
            return StopOrder(action, qty, order.stop_price)
        return MarketOrder(action, qty)

    async def submit_order(self, order: Order) -> Order:
        if not self._session_check(order.symbol):
            order.status = OrderStatus.REJECTED
            log.warning("IBKR rejecting %s: outside RTH", order.symbol)
            return order

        contract = self._build_ib_contract(order.symbol)
        ib_order = self._build_ib_order(order)
        try:
            trade = await asyncio.to_thread(self.ib.placeOrder, contract, ib_order)
        except Exception as exc:
            raise BrokerNetworkError(f"IBKR placeOrder failed: {exc}") from exc

        order.order_id = str(getattr(getattr(trade, "order", None), "orderId", order.order_id))
        status_obj = getattr(trade, "orderStatus", None)
        status_str = str(getattr(status_obj, "status", "") or "").lower()
        order.status = self._ib_status_to_ours(status_str)
        filled = float(getattr(status_obj, "filled", 0) or 0)
        if filled > 0:
            avg = float(getattr(status_obj, "avgFillPrice", 0) or 0)
            signed = filled if order.side > 0 else -filled
            order.add_fill(Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                timestamp=datetime.now(timezone.utc),
                price=avg,
                quantity=signed,
                commission=0.0,
                exchange="IBKR",
            ))
        self._orders[order.order_id] = order
        return order

    @staticmethod
    def _ib_status_to_ours(status: str) -> OrderStatus:
        mapping = {
            "pendingsubmit": OrderStatus.PENDING,
            "presubmitted": OrderStatus.SUBMITTED,
            "submitted": OrderStatus.SUBMITTED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "apicancelled": OrderStatus.CANCELLED,
            "pendingcancel": OrderStatus.SUBMITTED,
            "inactive": OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.SUBMITTED)

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        try:
            await asyncio.to_thread(self.ib.cancelOrder, order)
            order.status = OrderStatus.CANCELLED
            return True
        except Exception as exc:
            raise BrokerNetworkError(f"IBKR cancelOrder failed: {exc}") from exc

    async def get_order_status(self, order_id: str) -> Order:
        if order_id in self._orders:
            return self._orders[order_id]
        raise KeyError(order_id)

    async def get_positions(self) -> dict[str, Position]:
        try:
            rows = await asyncio.to_thread(self.ib.positions)
        except Exception as exc:
            raise BrokerNetworkError(f"IBKR positions failed: {exc}") from exc
        positions: dict[str, Position] = {}
        for row in rows or []:
            contract = getattr(row, "contract", None)
            symbol = str(getattr(contract, "localSymbol", None) or getattr(contract, "symbol", ""))
            qty = float(getattr(row, "position", 0) or 0)
            if qty == 0 or not symbol:
                continue
            side = 1 if qty > 0 else -1
            positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=abs(qty),
                avg_entry_price=float(getattr(row, "avgCost", 0) or 0),
                entry_timestamp=datetime.now(timezone.utc),
                signal_family="",
            )
        return positions

    async def get_account(self) -> dict[str, Any]:
        try:
            rows = await asyncio.to_thread(self.ib.accountSummary, self.account_id)
        except Exception as exc:
            raise BrokerNetworkError(f"IBKR accountSummary failed: {exc}") from exc
        out: dict[str, Any] = {}
        for row in rows or []:
            tag = str(getattr(row, "tag", ""))
            val = getattr(row, "value", None)
            try:
                out[tag] = float(val)
            except (TypeError, ValueError):
                out[tag] = val
        return out

    async def get_quote(self, symbol: str) -> dict[str, float]:
        contract = self._build_ib_contract(symbol)
        try:
            ticker = await asyncio.to_thread(self.ib.reqMktData, contract)
        except Exception as exc:
            raise BrokerNetworkError(f"IBKR reqMktData failed: {exc}") from exc
        bid = float(getattr(ticker, "bid", 0) or 0)
        ask = float(getattr(ticker, "ask", 0) or 0)
        mid = (bid + ask) / 2.0 if (bid and ask) else float(getattr(ticker, "last", 0) or 0)
        return {"bid": bid, "ask": ask, "mid": mid, "spread": max(ask - bid, 0.0)}

    async def heartbeat(self) -> bool:
        try:
            return bool(self.ib.isConnected())
        except Exception:
            return False

    def reconcile_against(
        self, internal: PortfolioState, broker_positions: dict[str, Position],
    ) -> list[ReconciliationDiff]:
        return reconcile_positions(internal, broker_positions)
