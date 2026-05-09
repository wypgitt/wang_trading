"""Unit tests for IBKR broker adapter safety hooks."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.execution.ibkr_adapter import IBKRBrokerAdapter
from src.execution.models import OrderStatus


class _FakeIB:
    def __init__(self):
        self.raw_order = SimpleNamespace(
            orderId=123,
            action="BUY",
            orderType="LMT",
            totalQuantity=2,
            lmtPrice=5010.25,
        )
        self.trade = SimpleNamespace(
            order=self.raw_order,
            contract=SimpleNamespace(localSymbol="ESM26", symbol="ES"),
            orderStatus=SimpleNamespace(
                status="Submitted",
                filled=0,
                avgFillPrice=0,
            ),
        )
        self.cancelled = None

    def openTrades(self):
        return [self.trade]

    def cancelOrder(self, order):
        self.cancelled = order


def _adapter(fake_ib: _FakeIB) -> IBKRBrokerAdapter:
    adapter = IBKRBrokerAdapter(live=False)
    adapter._ib = fake_ib
    return adapter


def test_ibkr_get_open_orders_maps_open_trades():
    adapter = _adapter(_FakeIB())

    orders = asyncio.run(adapter.get_open_orders(["ESM26"]))

    assert len(orders) == 1
    assert orders[0].order_id == "123"
    assert orders[0].symbol == "ESM26"
    assert orders[0].status == OrderStatus.SUBMITTED


def test_ibkr_cancel_uses_raw_ib_order_from_open_trades():
    fake_ib = _FakeIB()
    adapter = _adapter(fake_ib)

    cancelled = asyncio.run(adapter.cancel_order("123"))

    assert cancelled is True
    assert fake_ib.cancelled is fake_ib.raw_order
    assert asyncio.run(adapter.get_order_status("123")).status == OrderStatus.CANCELLED
