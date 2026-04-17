"""Execution algorithms (Phase 5 / P5.03).

Johnson's institutional execution algorithms adapted for a solo operator:
  ImmediateAlgo  – limit-at-mid with market fallback
  TWAPAlgo       – evenly-spaced child slices
  VWAPAlgo       – volume-profile-weighted child slices
  IcebergAlgo    – only a slice of quantity visible at a time

The `select_execution_algo()` router picks an algorithm from order size
relative to ADV and urgency.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.execution.broker_base import BaseBrokerAdapter
from src.execution.models import (
    ExecutionAlgo,
    Fill,
    Order,
    OrderStatus,
    OrderType,
)


# ── Base class ─────────────────────────────────────────────────────────

class BaseExecutionAlgo(ABC):
    def __init__(self, order: Order, broker: BaseBrokerAdapter) -> None:
        self.order = order
        self.broker = broker
        self.child_orders: list[Order] = []
        self.fills: list[Fill] = []
        self._cancelled = False

    @abstractmethod
    async def execute(self) -> list[Fill]:
        ...

    async def cancel(self) -> None:
        self._cancelled = True
        for c in self.child_orders:
            if c.is_active:
                try:
                    await self.broker.cancel_order(c.order_id)
                except Exception:
                    pass
                c.status = OrderStatus.CANCELLED

    @property
    def is_complete(self) -> bool:
        return (
            abs(self.order.filled_quantity) >= abs(self.order.quantity) - 1e-9
            or self._cancelled
        )

    @property
    def progress(self) -> float:
        if self.order.quantity == 0:
            return 1.0
        return min(1.0, abs(self.order.filled_quantity) / abs(self.order.quantity))

    # Helpers
    def _child_order(
        self,
        qty_signed: float,
        order_type: OrderType,
        limit_price: float | None = None,
    ) -> Order:
        child = Order(
            order_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol=self.order.symbol,
            side=self.order.side,
            order_type=order_type,
            quantity=qty_signed,
            limit_price=limit_price,
            parent_order_id=self.order.order_id,
            signal_family=self.order.signal_family,
            meta_label_prob=self.order.meta_label_prob,
        )
        self.child_orders.append(child)
        return child

    def _apply_child_fills(self, child: Order, new_fills: list[Fill]) -> None:
        for f in new_fills:
            child.add_fill(f)
            self.fills.append(f)
            self.order.filled_quantity += f.quantity
        if new_fills:
            self.order.updated_at = new_fills[-1].timestamp
        if abs(self.order.filled_quantity) >= abs(self.order.quantity) - 1e-9:
            self.order.status = OrderStatus.FILLED
        elif self.order.filled_quantity != 0:
            self.order.status = OrderStatus.PARTIAL_FILL


# ── Immediate (limit-at-mid with market fallback) ──────────────────────

class ImmediateAlgo(BaseExecutionAlgo):
    def __init__(
        self,
        order: Order,
        broker: BaseBrokerAdapter,
        *,
        timeout_seconds: float = 30.0,
        use_market_fallback: bool = True,
    ) -> None:
        super().__init__(order, broker)
        self.timeout_seconds = timeout_seconds
        self.use_market_fallback = use_market_fallback

    async def execute(self) -> list[Fill]:
        if self.order.order_type == OrderType.MARKET:
            child = self._child_order(self.order.quantity, OrderType.MARKET)
            await self.broker.submit_order(child)
            fills = await self.broker.poll_fills(child.order_id)
            self._apply_child_fills(child, fills)
            return self.fills

        mid = await self.broker.get_mid_price(self.order.symbol)
        child = self._child_order(self.order.quantity, OrderType.LIMIT_AT_MID, limit_price=mid)
        await self.broker.submit_order(child)
        await asyncio.sleep(self.timeout_seconds)
        fills = await self.broker.poll_fills(child.order_id)
        self._apply_child_fills(child, fills)

        unfilled = self.order.quantity - self.order.filled_quantity
        if abs(unfilled) > 1e-9 and self.use_market_fallback and not self._cancelled:
            await self.broker.cancel_order(child.order_id)
            child.status = OrderStatus.CANCELLED
            fallback = self._child_order(unfilled, OrderType.MARKET)
            await self.broker.submit_order(fallback)
            fills = await self.broker.poll_fills(fallback.order_id)
            self._apply_child_fills(fallback, fills)
        return self.fills


# ── TWAP ───────────────────────────────────────────────────────────────

class TWAPAlgo(BaseExecutionAlgo):
    def __init__(
        self,
        order: Order,
        broker: BaseBrokerAdapter,
        *,
        duration_minutes: float = 10.0,
        n_slices: int | None = None,
        max_slice_size: float | None = None,
        child_timeout_seconds: float = 5.0,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__(order, broker)
        total = abs(order.quantity)
        if n_slices is None:
            if max_slice_size and max_slice_size > 0:
                n_slices = max(1, math.ceil(total / max_slice_size))
            else:
                n_slices = 10
        self.n_slices = n_slices
        self.duration_minutes = duration_minutes
        self.child_timeout_seconds = child_timeout_seconds
        self.time_scale = time_scale
        self.slice_prices: list[float] = []  # mid-prices observed at each slice

    async def execute(self) -> list[Fill]:
        total_signed = self.order.quantity
        base_slice_signed = total_signed / self.n_slices
        interval_s = (self.duration_minutes * 60.0 / self.n_slices) * self.time_scale

        carry_signed = 0.0
        for i in range(self.n_slices):
            if self._cancelled:
                break
            remaining = total_signed - self.order.filled_quantity
            if abs(remaining) < 1e-9:
                break
            # Target this slice: base + carried unfilled
            target = base_slice_signed + carry_signed
            # Last slice sweeps any remainder
            if i == self.n_slices - 1:
                target = remaining
            # Don't over-order vs remaining
            if abs(target) > abs(remaining):
                target = remaining

            mid = await self.broker.get_mid_price(self.order.symbol)
            self.slice_prices.append(mid)
            child = self._child_order(target, OrderType.LIMIT_AT_MID, limit_price=mid)
            await self.broker.submit_order(child)
            await asyncio.sleep(self.child_timeout_seconds * self.time_scale)
            fills = await self.broker.poll_fills(child.order_id)
            self._apply_child_fills(child, fills)

            child_unfilled = target - child.filled_quantity
            if abs(child_unfilled) > 1e-9:
                await self.broker.cancel_order(child.order_id)
                child.status = OrderStatus.CANCELLED
            carry_signed = child_unfilled

            # Wait until next slice
            if i < self.n_slices - 1:
                gap = max(0.0, interval_s - self.child_timeout_seconds * self.time_scale)
                if gap > 0:
                    await asyncio.sleep(gap)

        return self.fills

    @property
    def twap_benchmark(self) -> dict[str, float]:
        """Realized weighted-avg fill price vs arithmetic mean of observed mid-prices."""
        realized = 0.0
        if self.fills:
            total_qty = sum(f.quantity for f in self.fills)
            if total_qty:
                realized = sum(f.price * f.quantity for f in self.fills) / total_qty
        arithmetic = float(np.mean(self.slice_prices)) if self.slice_prices else 0.0
        return {"realized": realized, "arithmetic_mean": arithmetic,
                "slippage_vs_twap": realized - arithmetic}


# ── VWAP ───────────────────────────────────────────────────────────────

def default_u_shape_profile(n_bins: int = 10) -> pd.Series:
    """U-shaped intraday volume profile — higher at open/close."""
    x = np.linspace(-1.0, 1.0, n_bins)
    w = 0.4 + 0.6 * x**2  # symmetric U
    w = w / w.sum()
    return pd.Series(w, name="volume_fraction")


class VWAPAlgo(BaseExecutionAlgo):
    def __init__(
        self,
        order: Order,
        broker: BaseBrokerAdapter,
        *,
        duration_minutes: float = 30.0,
        volume_profile: pd.Series | None = None,
        n_bins: int = 10,
        child_timeout_seconds: float = 5.0,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__(order, broker)
        if volume_profile is None:
            volume_profile = default_u_shape_profile(n_bins)
        profile = np.asarray(volume_profile, dtype=float)
        profile = profile / profile.sum()
        self.volume_profile = profile
        self.duration_minutes = duration_minutes
        self.child_timeout_seconds = child_timeout_seconds
        self.time_scale = time_scale
        self.slice_prices: list[float] = []

    async def execute(self) -> list[Fill]:
        total_signed = self.order.quantity
        n = len(self.volume_profile)
        interval_s = (self.duration_minutes * 60.0 / n) * self.time_scale
        carry_signed = 0.0

        for i, frac in enumerate(self.volume_profile):
            if self._cancelled:
                break
            remaining = total_signed - self.order.filled_quantity
            if abs(remaining) < 1e-9:
                break
            base = total_signed * float(frac)
            target = base + carry_signed
            if i == n - 1:
                target = remaining
            if abs(target) > abs(remaining):
                target = remaining

            mid = await self.broker.get_mid_price(self.order.symbol)
            self.slice_prices.append(mid)
            child = self._child_order(target, OrderType.LIMIT_AT_MID, limit_price=mid)
            await self.broker.submit_order(child)
            await asyncio.sleep(self.child_timeout_seconds * self.time_scale)
            fills = await self.broker.poll_fills(child.order_id)
            self._apply_child_fills(child, fills)

            child_unfilled = target - child.filled_quantity
            if abs(child_unfilled) > 1e-9:
                await self.broker.cancel_order(child.order_id)
                child.status = OrderStatus.CANCELLED
            carry_signed = child_unfilled

            if i < n - 1:
                gap = max(0.0, interval_s - self.child_timeout_seconds * self.time_scale)
                if gap > 0:
                    await asyncio.sleep(gap)

        return self.fills

    @property
    def vwap_benchmark(self) -> dict[str, float]:
        realized = 0.0
        if self.fills:
            total_qty = sum(f.quantity for f in self.fills)
            if total_qty:
                realized = sum(f.price * f.quantity for f in self.fills) / total_qty
        if self.slice_prices:
            weights = self.volume_profile[: len(self.slice_prices)]
            weights = weights / weights.sum()
            vwap = float(np.dot(self.slice_prices, weights))
        else:
            vwap = 0.0
        return {"realized": realized, "vwap_reference": vwap,
                "slippage_vs_vwap": realized - vwap}


# ── Iceberg ────────────────────────────────────────────────────────────

class IcebergAlgo(BaseExecutionAlgo):
    def __init__(
        self,
        order: Order,
        broker: BaseBrokerAdapter,
        *,
        visible_pct: float = 0.10,
        visible_size: float | None = None,
        price_offset: float = 0.0,
        child_timeout_seconds: float = 5.0,
        time_scale: float = 1.0,
        max_tranches: int = 100,
    ) -> None:
        super().__init__(order, broker)
        total = abs(order.quantity)
        if visible_size is None:
            visible_size = max(total * visible_pct, 1.0 if total >= 1 else total)
        self.visible_size = visible_size
        self.price_offset = price_offset
        self.child_timeout_seconds = child_timeout_seconds
        self.time_scale = time_scale
        self.max_tranches = max_tranches

    async def execute(self) -> list[Fill]:
        side = 1 if self.order.quantity > 0 else -1
        for _ in range(self.max_tranches):
            if self._cancelled:
                break
            remaining = self.order.quantity - self.order.filled_quantity
            if abs(remaining) < 1e-9:
                break
            tranche = side * min(self.visible_size, abs(remaining))

            mid = await self.broker.get_mid_price(self.order.symbol)
            limit = mid + side * self.price_offset
            child = self._child_order(tranche, OrderType.LIMIT, limit_price=limit)
            await self.broker.submit_order(child)
            await asyncio.sleep(self.child_timeout_seconds * self.time_scale)
            fills = await self.broker.poll_fills(child.order_id)
            self._apply_child_fills(child, fills)

            child_unfilled = tranche - child.filled_quantity
            if abs(child_unfilled) > 1e-9:
                await self.broker.cancel_order(child.order_id)
                child.status = OrderStatus.CANCELLED
                # If nothing filled at all, break out to avoid infinite loop
                if abs(child.filled_quantity) < 1e-9:
                    break
        return self.fills


# ── Router ─────────────────────────────────────────────────────────────

def select_execution_algo(
    order: Order,
    adv: float,
    urgency: str = "normal",
    *,
    asset_class: str = "equities",
    order_book_depth: float | None = None,
) -> type[BaseExecutionAlgo]:
    """Pick an execution algo from order size vs ADV and urgency.

    Returns the algo *class* (not instance) for the caller to parameterize.
    """
    if urgency == "high":
        return ImmediateAlgo

    size = abs(order.quantity)
    adv_ratio = size / adv if adv > 0 else float("inf")

    if asset_class == "crypto" and order_book_depth and order_book_depth > 0:
        if size / order_book_depth > 0.005:
            return IcebergAlgo

    if adv_ratio < 0.001:
        return ImmediateAlgo
    if adv_ratio <= 0.01:
        return TWAPAlgo
    return VWAPAlgo
