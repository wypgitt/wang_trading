"""Tests for execution data models (Phase 5 / P5.01)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.execution.models import (
    ExecutionAlgo,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _make_order(
    symbol: str = "AAPL",
    side: int = 1,
    qty: float = 100.0,
    order_type: OrderType = OrderType.LIMIT,
    limit: float | None = 150.0,
) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=_ts(),
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=qty * side,
        limit_price=limit,
    )


def _make_fill(order: Order, qty: float, price: float, commission: float = 0.0) -> Fill:
    return Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        timestamp=_ts(),
        price=price,
        quantity=qty,
        commission=commission,
        exchange="TEST",
    )


class TestOrderLifecycle:
    def test_pending_to_filled_transition(self):
        order = _make_order(qty=100.0, side=1)
        assert order.status == OrderStatus.PENDING
        assert order.is_active

        order.status = OrderStatus.SUBMITTED
        assert order.is_active and not order.is_complete

        order.add_fill(_make_fill(order, qty=40.0, price=150.0))
        assert order.status == OrderStatus.PARTIAL_FILL
        assert order.fill_pct == pytest.approx(0.4)

        order.add_fill(_make_fill(order, qty=60.0, price=151.0))
        assert order.status == OrderStatus.FILLED
        assert order.is_complete
        assert order.fill_pct == pytest.approx(1.0)
        # weighted average
        assert order.avg_fill_price == pytest.approx((40 * 150 + 60 * 151) / 100)

    def test_notional_value_uses_avg_fill(self):
        order = _make_order(qty=10, side=1, limit=100.0)
        order.add_fill(_make_fill(order, qty=10, price=105.0))
        assert order.notional_value == pytest.approx(1050.0)

    def test_is_complete_for_cancelled(self):
        order = _make_order()
        order.status = OrderStatus.CANCELLED
        assert order.is_complete and not order.is_active


class TestPosition:
    def test_update_price_long(self):
        pos = Position(
            symbol="AAPL",
            side=1,
            quantity=100,
            avg_entry_price=150.0,
            entry_timestamp=_ts(),
            signal_family="momentum",
        )
        pos.update_price(155.0)
        assert pos.current_price == 155.0
        assert pos.unrealized_pnl == pytest.approx(500.0)
        assert pos.return_pct == pytest.approx(5.0 / 150)

    def test_update_price_short(self):
        pos = Position(
            symbol="AAPL",
            side=-1,
            quantity=100,
            avg_entry_price=150.0,
            entry_timestamp=_ts(),
            signal_family="mean_rev",
        )
        pos.update_price(140.0)
        assert pos.unrealized_pnl == pytest.approx(1000.0)
        assert pos.return_pct == pytest.approx(10.0 / 150)
        assert pos.market_value == pytest.approx(-14000.0)

    def test_apply_fill_partial_close_long(self):
        pos = Position(
            symbol="AAPL",
            side=1,
            quantity=100,
            avg_entry_price=150.0,
            entry_timestamp=_ts(),
            signal_family="",
        )
        order = _make_order(symbol="AAPL", side=-1, qty=30)
        fill = _make_fill(order, qty=-30, price=160.0)
        pos.apply_fill(fill)
        assert pos.quantity == pytest.approx(70)
        assert pos.side == 1
        assert pos.avg_entry_price == pytest.approx(150.0)  # unchanged on reduce
        assert pos.realized_pnl == pytest.approx(30 * 10.0)

    def test_apply_fill_add_to_long_updates_avg(self):
        pos = Position(
            symbol="AAPL",
            side=1,
            quantity=100,
            avg_entry_price=150.0,
            entry_timestamp=_ts(),
            signal_family="",
        )
        order = _make_order(symbol="AAPL", side=1, qty=50)
        fill = _make_fill(order, qty=50, price=160.0)
        pos.apply_fill(fill)
        assert pos.quantity == pytest.approx(150)
        assert pos.avg_entry_price == pytest.approx((100 * 150 + 50 * 160) / 150)


class TestPortfolioState:
    def test_update_prices_and_nav(self):
        pos = Position(
            symbol="AAPL",
            side=1,
            quantity=100,
            avg_entry_price=150.0,
            entry_timestamp=_ts(),
            signal_family="",
            current_price=150.0,
        )
        pf = PortfolioState(cash=10_000.0, positions={"AAPL": pos})
        initial_nav = pf.nav
        assert initial_nav == pytest.approx(25_000.0)

        pf.update_prices({"AAPL": 160.0})
        assert pf.nav == pytest.approx(26_000.0)
        assert pf.peak_nav == pytest.approx(26_000.0)
        assert pf.drawdown == pytest.approx(0.0)

        pf.update_prices({"AAPL": 140.0})
        assert pf.nav == pytest.approx(24_000.0)
        assert pf.peak_nav == pytest.approx(26_000.0)
        assert pf.drawdown == pytest.approx((26_000 - 24_000) / 26_000)

    def test_record_fill_adjusts_cash_and_position(self):
        pf = PortfolioState(cash=100_000.0)
        order = _make_order(symbol="AAPL", side=1, qty=100, limit=150.0)
        pf.open_orders.append(order)
        fill = _make_fill(order, qty=100, price=150.0, commission=1.0)
        pf.record_fill(fill)
        assert pf.cash == pytest.approx(100_000 - 100 * 150 - 1)
        assert "AAPL" in pf.positions
        assert pf.positions["AAPL"].quantity == 100
        assert pf.positions["AAPL"].avg_entry_price == pytest.approx(150.0)

    def test_gross_and_net_exposure_mixed(self):
        long_pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=150.0,
            entry_timestamp=_ts(), signal_family="", current_price=150.0,
        )
        short_pos = Position(
            symbol="TSLA", side=-1, quantity=50, avg_entry_price=200.0,
            entry_timestamp=_ts(), signal_family="", current_price=200.0,
        )
        pf = PortfolioState(
            cash=0.0,
            positions={"AAPL": long_pos, "TSLA": short_pos},
        )
        assert pf.long_exposure == pytest.approx(15_000.0)
        assert pf.short_exposure == pytest.approx(10_000.0)
        assert pf.gross_exposure == pytest.approx(25_000.0)
        assert pf.net_exposure == pytest.approx(5_000.0)
        assert pf.position_count == 2

    def test_drawdown_tracks_rise_and_fall(self):
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = PortfolioState(cash=0.0, positions={"AAPL": pos})
        assert pf.peak_nav == pytest.approx(10_000)
        assert pf.drawdown == 0.0

        pf.update_prices({"AAPL": 120.0})  # up
        assert pf.peak_nav == pytest.approx(12_000)
        assert pf.drawdown == 0.0

        pf.update_prices({"AAPL": 110.0})  # down from peak
        assert pf.peak_nav == pytest.approx(12_000)
        assert pf.drawdown == pytest.approx((12_000 - 11_000) / 12_000)

        pf.update_prices({"AAPL": 130.0})  # new peak resets dd
        assert pf.peak_nav == pytest.approx(13_000)
        assert pf.drawdown == 0.0

    def test_close_position_via_record_fill(self):
        pf = PortfolioState(cash=100_000.0)
        buy = _make_order(symbol="AAPL", side=1, qty=100)
        pf.open_orders.append(buy)
        pf.record_fill(_make_fill(buy, qty=100, price=150.0))
        assert "AAPL" in pf.positions

        sell = _make_order(symbol="AAPL", side=-1, qty=100)
        pf.open_orders.append(sell)
        pf.record_fill(_make_fill(sell, qty=-100, price=160.0))
        # Position closed
        assert "AAPL" not in pf.positions
        # Cash: 100k - 15000 (buy) + 16000 (sell) = 101000
        assert pf.cash == pytest.approx(101_000.0)


class TestEnumsAndDefaults:
    def test_order_defaults(self):
        o = _make_order()
        assert o.execution_algo == ExecutionAlgo.IMMEDIATE
        assert o.filled_quantity == 0.0
        assert o.fills == []
        assert o.parent_order_id is None

    def test_order_types_and_statuses(self):
        assert OrderType.TWAP.value == "twap"
        assert OrderStatus.FILLED.value == "filled"
        assert ExecutionAlgo.ICEBERG.value == "iceberg"
