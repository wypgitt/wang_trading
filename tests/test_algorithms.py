"""Tests for execution algorithms (Phase 5 / P5.03)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.execution.algorithms import (
    BaseExecutionAlgo,
    IcebergAlgo,
    ImmediateAlgo,
    TWAPAlgo,
    VWAPAlgo,
    default_u_shape_profile,
    select_execution_algo,
)
from src.execution.broker_base import BaseBrokerAdapter
from src.execution.models import Fill, Order, OrderStatus, OrderType


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _make_order(qty: float = 1000.0, side: int = 1, symbol: str = "AAPL",
                order_type: OrderType = OrderType.LIMIT) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=_ts(),
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=qty * side,
        limit_price=100.0,
    )


class MockBrokerAdapter(BaseBrokerAdapter):
    """Deterministic in-memory broker for tests.

    - `fill_ratio`: fraction of each child order that fills (0..1)
    - `mid_prices`: per-symbol mid-price sequence (cycled)
    - `delay`: async sleep before fills become visible
    """

    def __init__(
        self,
        *,
        fill_ratio: float = 1.0,
        mid_prices: dict[str, list[float]] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.fill_ratio = fill_ratio
        self.mid_prices = mid_prices or {"AAPL": [100.0]}
        self.delay = delay
        self.orders: dict[str, Order] = {}
        self._fills_pending: dict[str, list[Fill]] = {}
        self._mid_idx: dict[str, int] = {}
        self.submit_calls: list[str] = []
        self.cancel_calls: list[str] = []

    async def submit_order(self, order: Order) -> Order:
        await asyncio.sleep(self.delay)
        self.orders[order.order_id] = order
        self.submit_calls.append(order.order_id)
        fill_qty = order.quantity * self.fill_ratio
        if abs(fill_qty) > 1e-12:
            if order.limit_price is not None:
                fill_price = order.limit_price
            else:
                fill_price = await self._next_mid(order.symbol)
            self._fills_pending[order.order_id] = [
                Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    timestamp=_ts(),
                    price=fill_price,
                    quantity=fill_qty,
                    commission=0.0,
                    exchange="MOCK",
                )
            ]
        else:
            self._fills_pending[order.order_id] = []
        return order

    async def cancel_order(self, order_id: str) -> bool:
        self.cancel_calls.append(order_id)
        order = self.orders.get(order_id)
        if order is not None and order.is_active:
            order.status = OrderStatus.CANCELLED
        return True

    async def poll_fills(self, order_id: str) -> list[Fill]:
        return self._fills_pending.pop(order_id, [])

    async def _next_mid(self, symbol: str) -> float:
        seq = self.mid_prices.get(symbol, [100.0])
        idx = self._mid_idx.get(symbol, 0)
        price = seq[idx % len(seq)]
        self._mid_idx[symbol] = idx + 1
        return price

    async def get_quote(self, symbol: str) -> dict[str, float]:
        mid = await self._next_mid(symbol)
        return {"bid": mid - 0.01, "ask": mid + 0.01, "mid": mid, "spread": 0.02}

    async def get_order_status(self, order_id: str) -> Order:
        return self.orders[order_id]

    async def get_positions(self) -> dict:
        return {}

    async def get_account(self) -> dict:
        return {"cash": 0.0, "nav": 0.0}

    async def heartbeat(self) -> bool:
        return True


# ── Immediate ──────────────────────────────────────────────────────────

class TestImmediateAlgo:
    @pytest.mark.asyncio
    async def test_full_fill_at_mid(self):
        broker = MockBrokerAdapter(fill_ratio=1.0, mid_prices={"AAPL": [101.0]})
        order = _make_order(qty=100, side=1)
        algo = ImmediateAlgo(order, broker, timeout_seconds=0.0)
        fills = await algo.execute()
        assert algo.is_complete
        assert sum(f.quantity for f in fills) == pytest.approx(100)
        assert fills[0].price == 101.0

    @pytest.mark.asyncio
    async def test_market_fallback_on_partial(self):
        broker = MockBrokerAdapter(fill_ratio=0.5, mid_prices={"AAPL": [100.0]})
        order = _make_order(qty=100, side=1)
        algo = ImmediateAlgo(order, broker, timeout_seconds=0.0)
        await algo.execute()
        # Original filled 50%; fallback market order fills another 25% (50% of 50)
        assert len(broker.cancel_calls) == 1
        assert len(algo.child_orders) == 2


# ── TWAP ───────────────────────────────────────────────────────────────

class TestTWAPAlgo:
    @pytest.mark.asyncio
    async def test_splits_into_n_slices(self):
        broker = MockBrokerAdapter(fill_ratio=1.0)
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, broker, n_slices=5, duration_minutes=0.0,
                        child_timeout_seconds=0.0, time_scale=0.0)
        await algo.execute()
        assert len(algo.child_orders) == 5
        # Each child targets 200
        for c in algo.child_orders:
            assert c.quantity == pytest.approx(200.0)
        assert order.filled_quantity == pytest.approx(1000.0)

    @pytest.mark.asyncio
    async def test_carries_unfilled_to_next_slice(self):
        # First child fills 0%, second fills 100%, etc.
        class SequencedBroker(MockBrokerAdapter):
            def __init__(self):
                super().__init__()
                self.ratios = iter([0.0, 1.0, 1.0, 1.0, 1.0])
            async def submit_order(self, order):
                self.fill_ratio = next(self.ratios, 1.0)
                return await super().submit_order(order)

        broker = SequencedBroker()
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, broker, n_slices=5, duration_minutes=0.0,
                        child_timeout_seconds=0.0, time_scale=0.0)
        await algo.execute()
        # First slice (200) unfilled; second slice should target 200 base + 200 carry = 400
        assert algo.child_orders[0].quantity == pytest.approx(200)
        assert algo.child_orders[1].quantity == pytest.approx(400)
        # Ultimately last slice sweeps any leftover
        assert order.filled_quantity == pytest.approx(1000)

    @pytest.mark.asyncio
    async def test_twap_benchmark_reports(self):
        broker = MockBrokerAdapter(
            fill_ratio=1.0, mid_prices={"AAPL": [99.0, 100.0, 101.0, 102.0]},
        )
        order = _make_order(qty=400)
        algo = TWAPAlgo(order, broker, n_slices=4, duration_minutes=0.0,
                        child_timeout_seconds=0.0, time_scale=0.0)
        await algo.execute()
        bench = algo.twap_benchmark
        assert bench["arithmetic_mean"] == pytest.approx(100.5)


# ── VWAP ───────────────────────────────────────────────────────────────

class TestVWAPAlgo:
    @pytest.mark.asyncio
    async def test_weights_by_volume_profile(self):
        broker = MockBrokerAdapter(fill_ratio=1.0)
        order = _make_order(qty=1000)
        profile = pd.Series([0.5, 0.3, 0.2])
        algo = VWAPAlgo(order, broker, volume_profile=profile,
                        duration_minutes=0.0, child_timeout_seconds=0.0, time_scale=0.0)
        await algo.execute()
        qtys = [c.quantity for c in algo.child_orders]
        # First = 500, second = 300, third = 200 (last sweep picks up exactly remaining)
        assert qtys[0] == pytest.approx(500)
        assert qtys[1] == pytest.approx(300)
        assert qtys[2] == pytest.approx(200)
        assert order.filled_quantity == pytest.approx(1000)

    def test_default_u_shape_sums_to_one(self):
        prof = default_u_shape_profile(10)
        assert prof.sum() == pytest.approx(1.0)
        # Ends higher than middle
        assert prof.iloc[0] > prof.iloc[5]
        assert prof.iloc[-1] > prof.iloc[5]


# ── Iceberg ────────────────────────────────────────────────────────────

class TestIcebergAlgo:
    @pytest.mark.asyncio
    async def test_shows_only_visible_size(self):
        broker = MockBrokerAdapter(fill_ratio=1.0)
        order = _make_order(qty=1000)
        algo = IcebergAlgo(order, broker, visible_size=100,
                          child_timeout_seconds=0.0, time_scale=0.0)
        await algo.execute()
        # 10 tranches of 100
        assert len(algo.child_orders) == 10
        for c in algo.child_orders:
            assert c.quantity == pytest.approx(100)
        assert order.filled_quantity == pytest.approx(1000)

    @pytest.mark.asyncio
    async def test_stops_on_unfilled_tranche(self):
        broker = MockBrokerAdapter(fill_ratio=0.0)
        order = _make_order(qty=1000)
        algo = IcebergAlgo(order, broker, visible_size=100,
                          child_timeout_seconds=0.0, time_scale=0.0, max_tranches=10)
        await algo.execute()
        # Nothing fills → breaks after first tranche
        assert len(algo.child_orders) == 1
        assert order.filled_quantity == 0.0


# ── Router ─────────────────────────────────────────────────────────────

class TestSelectExecutionAlgo:
    def test_small_order_picks_immediate(self):
        order = _make_order(qty=100)  # 0.01% of 1M
        assert select_execution_algo(order, adv=1_000_000) is ImmediateAlgo

    def test_medium_order_picks_twap(self):
        order = _make_order(qty=5_000)  # 0.5% of 1M
        assert select_execution_algo(order, adv=1_000_000) is TWAPAlgo

    def test_large_order_picks_vwap(self):
        order = _make_order(qty=50_000)  # 5% of 1M
        assert select_execution_algo(order, adv=1_000_000) is VWAPAlgo

    def test_high_urgency_picks_immediate(self):
        order = _make_order(qty=50_000)
        assert select_execution_algo(order, adv=1_000_000, urgency="high") is ImmediateAlgo

    def test_crypto_thin_book_picks_iceberg(self):
        order = _make_order(qty=1_000, symbol="BTC-USD")
        algo = select_execution_algo(
            order, adv=10_000_000, asset_class="crypto", order_book_depth=100_000,
        )
        assert algo is IcebergAlgo


# ── Common ─────────────────────────────────────────────────────────────

class TestAlgoCommon:
    @pytest.mark.asyncio
    async def test_cancel_stops_algo(self):
        broker = MockBrokerAdapter(fill_ratio=0.0)
        order = _make_order(qty=1000)
        algo = TWAPAlgo(order, broker, n_slices=5, duration_minutes=0.0,
                        child_timeout_seconds=0.0, time_scale=0.0)
        await algo.cancel()
        fills = await algo.execute()
        assert fills == []
