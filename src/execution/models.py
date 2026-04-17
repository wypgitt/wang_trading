"""Order, Fill, Position, and PortfolioState data models (Phase 5).

See design doc §10 (Execution Engine). These are the core state objects
passed between order routing, fills, TCA, and monitoring.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_AT_MID = "limit_at_mid"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    BRACKET = "bracket"


class ExecutionAlgo(Enum):
    IMMEDIATE = "immediate"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


@dataclass
class Fill:
    fill_id: str
    order_id: str
    timestamp: datetime
    price: float
    quantity: float
    commission: float
    exchange: str
    is_maker: bool = False


@dataclass
class Order:
    order_id: str
    timestamp: datetime
    symbol: str
    side: int  # +1 buy, -1 sell
    order_type: OrderType
    quantity: float
    filled_quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    execution_algo: ExecutionAlgo = ExecutionAlgo.IMMEDIATE
    parent_order_id: str | None = None
    signal_family: str = ""
    meta_label_prob: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    @property
    def is_complete(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILL,
        )

    @property
    def fill_pct(self) -> float:
        if self.quantity == 0:
            return 0.0
        return abs(self.filled_quantity) / abs(self.quantity)

    @property
    def avg_fill_price(self) -> float:
        if not self.fills:
            return 0.0
        total_qty = sum(f.quantity for f in self.fills)
        if total_qty == 0:
            return 0.0
        return sum(f.price * f.quantity for f in self.fills) / total_qty

    @property
    def notional_value(self) -> float:
        price = self.avg_fill_price if self.fills else (self.limit_price or 0.0)
        return abs(self.quantity) * price

    def add_fill(self, fill: Fill) -> None:
        """Append a fill and update status/filled_quantity."""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.updated_at = fill.timestamp
        if abs(self.filled_quantity) >= abs(self.quantity) - 1e-9:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL_FILL


@dataclass
class Position:
    symbol: str
    side: int  # +1 long, -1 short
    quantity: float  # always positive; side encodes direction
    avg_entry_price: float
    entry_timestamp: datetime
    signal_family: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    vertical_barrier: datetime | None = None

    @property
    def market_value(self) -> float:
        price = self.current_price if self.current_price else self.avg_entry_price
        return self.side * self.quantity * price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_entry_price

    @property
    def return_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        price = self.current_price if self.current_price else self.avg_entry_price
        return self.side * (price - self.avg_entry_price) / self.avg_entry_price

    @property
    def holding_period_seconds(self) -> float:
        return (_utcnow() - self.entry_timestamp).total_seconds()

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.unrealized_pnl = self.side * self.quantity * (price - self.avg_entry_price)

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill against this position.

        Fill quantity is signed (+buy, -sell). A fill in the same direction
        as `side` increases the position; opposite direction reduces or flips.
        """
        signed_qty = fill.quantity  # signed
        current_signed = self.side * self.quantity
        new_signed = current_signed + signed_qty

        if signed_qty * self.side > 0:
            # Same direction: weighted-average entry
            new_qty = abs(new_signed)
            self.avg_entry_price = (
                self.quantity * self.avg_entry_price + abs(signed_qty) * fill.price
            ) / new_qty
            self.quantity = new_qty
        else:
            # Opposite direction: realize P&L on closed portion
            close_qty = min(abs(signed_qty), self.quantity)
            self.realized_pnl += self.side * close_qty * (fill.price - self.avg_entry_price)
            remaining = abs(new_signed)
            if remaining < 1e-12:
                self.quantity = 0.0
            elif new_signed * self.side > 0:
                self.quantity = remaining
            else:
                # Position flipped direction
                self.side = -self.side
                self.quantity = remaining
                self.avg_entry_price = fill.price

        self.update_price(fill.price)


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    nav: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    open_orders: list[Order] = field(default_factory=list)
    daily_pnl: float = 0.0
    peak_nav: float = 0.0
    drawdown: float = 0.0

    def __post_init__(self) -> None:
        if self.nav == 0.0:
            self.nav = self.cash + sum(p.market_value for p in self.positions.values())
        if self.peak_nav == 0.0:
            self.peak_nav = self.nav
        self._recompute_exposures()

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def long_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values() if p.side > 0)

    @property
    def short_exposure(self) -> float:
        return sum(-p.market_value for p in self.positions.values() if p.side < 0)

    def _recompute_exposures(self) -> None:
        self.gross_exposure = sum(abs(p.market_value) for p in self.positions.values())
        self.net_exposure = sum(p.market_value for p in self.positions.values())

    def _recompute_nav(self) -> None:
        self.nav = self.cash + sum(p.market_value for p in self.positions.values())
        if self.nav > self.peak_nav:
            self.peak_nav = self.nav
        self.drawdown = (
            (self.peak_nav - self.nav) / self.peak_nav if self.peak_nav > 0 else 0.0
        )

    def update_prices(self, prices: dict[str, float]) -> None:
        for symbol, price in prices.items():
            pos = self.positions.get(symbol)
            if pos is not None:
                pos.update_price(price)
        self._recompute_exposures()
        self._recompute_nav()

    def record_fill(self, fill: Fill) -> None:
        """Apply a fill to cash + positions. Fill.quantity is signed."""
        symbol = None
        # Find symbol via order_id in open_orders
        for order in self.open_orders:
            if order.order_id == fill.order_id:
                symbol = order.symbol
                break

        cash_delta = -fill.quantity * fill.price - fill.commission
        self.cash += cash_delta

        if symbol is None:
            # Without an order reference we can't map to a position; update cash only.
            self._recompute_exposures()
            self._recompute_nav()
            return

        pos = self.positions.get(symbol)
        if pos is None:
            side = 1 if fill.quantity > 0 else -1
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=abs(fill.quantity),
                avg_entry_price=fill.price,
                entry_timestamp=fill.timestamp,
                signal_family=next(
                    (o.signal_family for o in self.open_orders if o.order_id == fill.order_id),
                    "",
                ),
                current_price=fill.price,
            )
        else:
            pos.apply_fill(fill)
            if pos.quantity == 0:
                del self.positions[symbol]

        self._recompute_exposures()
        self._recompute_nav()
