"""Tests for the IBKR futures adapter (P6.03).

All ib_insync interactions are mocked via `adapter._ib = MagicMock()`, so no
real TWS connection is ever attempted. Tagged ``live_broker`` so CI can skip.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.broker_adapter import BrokerNetworkError
from src.execution.ibkr_adapter import (
    FuturesContractRegistry,
    IBKRBrokerAdapter,
    contract_expiry,
)
from src.execution.models import (
    Order,
    OrderStatus,
    OrderType,
    Position,
)

pytestmark = pytest.mark.live_broker


# ── Helpers ───────────────────────────────────────────────────────────────

def _adapter(**kwargs) -> IBKRBrokerAdapter:
    adapter = IBKRBrokerAdapter(**kwargs)
    adapter._ib = MagicMock()
    adapter._ib.isConnected.return_value = True
    return adapter


def _order(symbol: str = "ESZ25", qty: float = 1.0, side: int = 1,
           order_type: OrderType = OrderType.MARKET) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=qty * side,
        limit_price=4500.0 if order_type == OrderType.LIMIT else None,
    )


# ── Live-trading gate ─────────────────────────────────────────────────────

class TestLiveGate:
    def test_paper_default_ok(self):
        a = IBKRBrokerAdapter()
        assert a.live is False
        assert a.port == 7497

    def test_live_blocked_without_env(self, monkeypatch):
        monkeypatch.delenv("WANG_ALLOW_LIVE_FUTURES", raising=False)
        with pytest.raises(ValueError, match="Live futures trading blocked"):
            IBKRBrokerAdapter(live=True)

    def test_live_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("WANG_ALLOW_LIVE_FUTURES", "yes")
        a = IBKRBrokerAdapter(live=True)
        assert a.live is True
        assert a.port == 7496


# ── Contract registry ─────────────────────────────────────────────────────

class TestRegistry:
    def test_loads_example_defaults(self):
        reg = FuturesContractRegistry()
        specs = reg.symbols()
        for expected in ("ES", "NQ", "CL", "GC", "ZN", "ZC"):
            assert expected in specs

    def test_get_spec_es(self):
        reg = FuturesContractRegistry()
        spec = reg.get_spec("ES")
        assert spec["tick_size"] == 0.25
        assert spec["multiplier"] == 50
        assert spec["exchange"] == "CME"

    def test_get_spec_accepts_full_contract(self):
        reg = FuturesContractRegistry()
        spec = reg.get_spec("ESZ25")
        assert spec["exchange"] == "CME"

    def test_front_and_next_month(self):
        reg = FuturesContractRegistry()
        # Force now to 2026-04-17 (from system context)
        now = datetime(2026, 4, 17, tzinfo=timezone.utc)
        front = reg.get_front_month("ES", now=now)
        nxt = reg.get_next_month("ES", now=now)
        # ES uses H,M,U,Z. April 17, 2026 → next expiry is June 2026 (M26)
        assert front == "ESM26"
        assert nxt == "ESU26"


# ── Session hours ─────────────────────────────────────────────────────────

class TestSessionHours:
    def test_within_rth(self):
        reg = FuturesContractRegistry()
        # 10:00 ET on a Wednesday (2026-04-15 14:00 UTC == 10:00 ET)
        now = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)
        assert reg.is_within_rth("ES", now=now) is True

    def test_outside_rth(self):
        reg = FuturesContractRegistry()
        # 03:00 ET Wednesday (07:00 UTC)
        now = datetime(2026, 4, 15, 7, 0, tzinfo=timezone.utc)
        assert reg.is_within_rth("ES", now=now) is False

    def test_weekend_rejected(self):
        reg = FuturesContractRegistry()
        now = datetime(2026, 4, 18, 14, 0, tzinfo=timezone.utc)  # Saturday
        assert reg.is_within_rth("ES", now=now) is False


class TestSubmitSessionGate:
    def test_rejects_outside_rth(self, monkeypatch):
        # Freeze the session-check to return False
        adapter = _adapter()

        def always_reject(symbol, now=None):
            return False

        monkeypatch.setattr(adapter, "_session_check", always_reject)
        result = asyncio.run(adapter.submit_order(_order()))
        assert result.status == OrderStatus.REJECTED
        adapter._ib.placeOrder.assert_not_called()

    def test_allow_extended_bypasses_session_check(self, monkeypatch):
        adapter = _adapter(allow_extended=True)
        # allow_extended=True makes _session_check return True unconditionally
        called = {}

        def fake_place_order(contract, order):
            called["placed"] = True
            trade = MagicMock()
            trade.order.orderId = 42
            trade.orderStatus.status = "Submitted"
            trade.orderStatus.filled = 0
            return trade

        adapter._ib.placeOrder.side_effect = fake_place_order
        # Stub the contract/order builders so we don't import ib_insync
        monkeypatch.setattr(adapter, "_build_ib_contract", lambda s: object())
        monkeypatch.setattr(adapter, "_build_ib_order", lambda o: object())

        result = asyncio.run(adapter.submit_order(_order()))
        assert result.status == OrderStatus.SUBMITTED
        assert called.get("placed") is True


# ── Roll detection ────────────────────────────────────────────────────────

class TestRoll:
    def test_check_roll_needed_within_threshold(self):
        adapter = _adapter()
        now = datetime(2026, 4, 17, tzinfo=timezone.utc)
        # ESM26 expires around 2026-06-19 (3rd Friday June). Far away.
        # ESH26 expired 2026-03-20. Use ESM26 with artificial near-expiry.
        adapter._active_contracts["ESM26"] = now + timedelta(days=7)
        adapter._active_contracts["ESU26"] = now + timedelta(days=95)

        out = asyncio.run(adapter.check_roll_needed(threshold_days=10, now=now))
        syms = [r["symbol"] for r in out]
        assert "ESM26" in syms
        assert "ESU26" not in syms
        row = next(r for r in out if r["symbol"] == "ESM26")
        assert row["days_left"] == 7
        assert row["next_symbol"].startswith("ES")

    def test_no_roll_when_far(self):
        adapter = _adapter()
        now = datetime(2026, 4, 17, tzinfo=timezone.utc)
        adapter._active_contracts["ESU26"] = now + timedelta(days=60)
        out = asyncio.run(adapter.check_roll_needed(threshold_days=10, now=now))
        assert out == []

    def test_execute_roll_submits_two_orders(self, monkeypatch):
        adapter = _adapter()

        async def fake_submit(order):
            order.status = OrderStatus.SUBMITTED
            return order

        monkeypatch.setattr(adapter, "submit_order", fake_submit)

        adapter._active_contracts["ESM26"] = datetime(2026, 6, 19, tzinfo=timezone.utc)
        position = Position(
            symbol="ESM26", side=1, quantity=2,
            avg_entry_price=4500, entry_timestamp=datetime.now(timezone.utc),
            signal_family="trend",
        )
        orders = asyncio.run(adapter.execute_roll("ESM26", "ESU26", position))
        assert len(orders) == 2
        assert orders[0].side == -1  # close long
        assert orders[1].side == 1   # open new long
        assert "ESM26" not in adapter._active_contracts
        assert "ESU26" in adapter._active_contracts


# ── Misc interface ───────────────────────────────────────────────────────

class TestMisc:
    def test_contract_expiry_helper(self):
        exp = contract_expiry("ESZ25")
        # December 2025 3rd Friday = 2025-12-19
        assert exp.year == 2025 and exp.month == 12 and exp.day == 19

    def test_get_contract_specs(self):
        adapter = _adapter()
        specs = asyncio.run(adapter.get_contract_specs("ES"))
        assert specs["tick_size"] == 0.25
        assert specs["exchange"] == "CME"
        assert specs["multiplier"] == 50

    def test_heartbeat(self):
        adapter = _adapter()
        assert asyncio.run(adapter.heartbeat()) is True
        adapter._ib.isConnected.return_value = False
        assert asyncio.run(adapter.heartbeat()) is False

    def test_cancel_missing_order_returns_false(self):
        adapter = _adapter()
        assert asyncio.run(adapter.cancel_order("missing")) is False
