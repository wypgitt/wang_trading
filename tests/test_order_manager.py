"""Tests for OrderManager (Phase 5 / P5.05)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.models import (
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)
from src.execution.order_manager import OrderManager


def _ts() -> datetime:
    return datetime.now(timezone.utc)


EQ_COST_CFG = {
    "commission_per_share": 0.0,
    "min_commission": 0.0,
    "spread_bps": 1.0,
    "slippage_bps": 1.0,
    "impact_coefficient": 0.1,
}


def _make_manager(
    *,
    prices: dict[str, float] | None = None,
    portfolio: PortfolioState | None = None,
    cb_overrides: dict | None = None,
) -> tuple[OrderManager, PaperBrokerAdapter, PortfolioState]:
    prices = prices or {"AAPL": 100.0, "TSLA": 200.0, "MSFT": 300.0}
    portfolio = portfolio if portfolio is not None else PortfolioState(cash=1_000_000.0)
    broker = PaperBrokerAdapter(
        initial_cash=1_000_000.0, slippage_bps=0.0, fill_delay_ms=0,
        price_feed=lambda s: prices.get(s, 100.0),
    )
    cb_kwargs = dict(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0, daily_loss_limit_pct=0.02,
    )
    cb_kwargs.update(cb_overrides or {})
    cbs = CircuitBreakerManager(**cb_kwargs)
    cost_model = TransactionCostModel(equities_config=EQ_COST_CFG)
    om = OrderManager(broker, cbs, cost_model, portfolio,
                     adv_map={"AAPL": 1e7, "TSLA": 1e7, "MSFT": 1e7})
    return om, broker, portfolio


# ── execute_target_portfolio ───────────────────────────────────────────

class TestExecuteTargetPortfolio:
    @pytest.mark.asyncio
    async def test_creates_orders_for_three_new_positions(self):
        prices = {"AAPL": 100.0, "TSLA": 200.0, "MSFT": 300.0}
        om, broker, pf = _make_manager(prices=prices)
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.10, "strategy": "mom"},
            {"symbol": "TSLA", "target_weight": 0.10, "strategy": "mom"},
            {"symbol": "MSFT", "target_weight": 0.10, "strategy": "mom"},
        ])
        orders = await om.execute_target_portfolio(target, prices)
        assert len(orders) == 3
        assert all(o.status == OrderStatus.FILLED for o in orders)
        assert set(pf.positions.keys()) == {"AAPL", "TSLA", "MSFT"}

    @pytest.mark.asyncio
    async def test_exits_processed_before_entries(self):
        prices = {"AAPL": 100.0, "TSLA": 200.0}
        # Start with AAPL position; target wants to fully exit AAPL and enter TSLA
        existing = Position(
            symbol="AAPL", side=1, quantity=1000, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = PortfolioState(cash=900_000.0, positions={"AAPL": existing})
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        # AAPL currently ~10% of NAV. Target: AAPL=0, TSLA=0.10
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.0, "strategy": "mom"},
            {"symbol": "TSLA", "target_weight": 0.10, "strategy": "mom"},
        ])
        orders = await om.execute_target_portfolio(target, prices)
        # Exit order (AAPL sell) comes before entry (TSLA buy)
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == -1
        assert orders[1].symbol == "TSLA"
        assert orders[1].side == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_oversized_order(self):
        prices = {"AAPL": 100.0}
        # Set fat-finger threshold low enough that a 10% position is rejected
        om, broker, pf = _make_manager(
            prices=prices, cb_overrides={"max_order_pct": 0.05},
        )
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.10, "strategy": "mom"},  # 10% > 5%
        ])
        orders = await om.execute_target_portfolio(target, prices)
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_trivial_rebalance_filtered(self):
        prices = {"AAPL": 100.0}
        # Existing 10% position; target is 10.5% — delta 0.5% < min 1%
        pos = Position(
            symbol="AAPL", side=1, quantity=1000, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = PortfolioState(cash=900_000.0, positions={"AAPL": pos})
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.105, "strategy": "mom"},
        ])
        orders = await om.execute_target_portfolio(target, prices)
        assert orders == []


# ── check_position_exits (triple-barrier) ─────────────────────────────

class TestPositionExits:
    @pytest.mark.asyncio
    async def test_stop_loss_triggers_market_order(self):
        prices = {"AAPL": 90.0}
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
            stop_loss=95.0,
        )
        pf = PortfolioState(cash=990_000.0, positions={"AAPL": pos})
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        orders = await om.check_position_exits(prices)
        assert len(orders) == 1
        assert orders[0].order_type == OrderType.MARKET
        assert orders[0].side == -1
        assert orders[0].signal_family == "exit:stop_loss"
        assert "AAPL" not in pf.positions

    @pytest.mark.asyncio
    async def test_take_profit_triggers_limit_order(self):
        prices = {"AAPL": 120.0}
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
            take_profit=115.0,
        )
        pf = PortfolioState(cash=990_000.0, positions={"AAPL": pos})
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        orders = await om.check_position_exits(prices)
        assert len(orders) == 1
        assert orders[0].order_type == OrderType.LIMIT
        assert orders[0].signal_family == "exit:take_profit"
        assert "AAPL" not in pf.positions

    @pytest.mark.asyncio
    async def test_time_expiry_exit(self):
        prices = {"AAPL": 102.0}
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
            vertical_barrier=_ts() - timedelta(minutes=1),
        )
        pf = PortfolioState(cash=990_000.0, positions={"AAPL": pos})
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        orders = await om.check_position_exits(prices)
        assert len(orders) == 1
        assert orders[0].signal_family == "exit:time_expiry"
        assert "AAPL" not in pf.positions

    @pytest.mark.asyncio
    async def test_no_exit_when_barriers_not_hit(self):
        prices = {"AAPL": 105.0}
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
            stop_loss=90.0, take_profit=120.0,
        )
        pf = PortfolioState(cash=990_000.0, positions={"AAPL": pos})
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        orders = await om.check_position_exits(prices)
        assert orders == []
        assert "AAPL" in pf.positions


# ── Reconciliation ────────────────────────────────────────────────────

class TestReconciliation:
    @pytest.mark.asyncio
    async def test_detects_broker_position_not_internally(self):
        prices = {"AAPL": 100.0}
        om, broker, pf = _make_manager(prices=prices)
        # Broker has a rogue AAPL position we don't know about
        broker.positions["AAPL"] = Position(
            symbol="AAPL", side=1, quantity=50, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="",
        )
        diffs = await om.reconcile_positions()
        assert len(diffs) == 1
        assert diffs[0]["symbol"] == "AAPL"
        assert diffs[0]["broker_signed_qty"] == 50
        assert diffs[0]["internal_signed_qty"] == 0

    @pytest.mark.asyncio
    async def test_no_diff_when_matched(self):
        prices = {"AAPL": 100.0}
        pos = Position(
            symbol="AAPL", side=1, quantity=10, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = PortfolioState(cash=990_000.0, positions={"AAPL": pos.__class__(**pos.__dict__)})
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        broker.positions["AAPL"] = Position(
            symbol="AAPL", side=1, quantity=10, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="",
        )
        assert await om.reconcile_positions() == []


# ── run_cycle ─────────────────────────────────────────────────────────

class TestRunCycle:
    @pytest.mark.asyncio
    async def test_full_cycle_with_paper_broker(self):
        prices = {"AAPL": 100.0, "TSLA": 200.0}
        om, broker, pf = _make_manager(prices=prices)
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.05, "strategy": "mom"},
            {"symbol": "TSLA", "target_weight": 0.05, "strategy": "mom"},
        ])
        summary = await om.run_cycle(prices, target=target)
        assert len(summary["rebalance_orders"]) == 2
        assert summary["exits"] == []
        assert set(pf.positions.keys()) == {"AAPL", "TSLA"}
        assert summary["reconciliation"] == []  # paper broker mirrors us? not necessarily

    @pytest.mark.asyncio
    async def test_cycle_halts_rebalance_on_halt_action(self):
        prices = {"AAPL": 100.0}
        # Position with 25% drawdown baked in
        pos = Position(
            symbol="AAPL", side=1, quantity=1000, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = PortfolioState(cash=900_000.0, positions={"AAPL": pos})
        pf.peak_nav = pf.nav / 0.75  # 25% drawdown
        pf.drawdown = 0.25
        om, broker, pf = _make_manager(prices=prices, portfolio=pf)
        target = pd.DataFrame([
            {"symbol": "TSLA", "target_weight": 0.05, "strategy": "mom"},
        ])
        summary = await om.run_cycle(prices, target=target)
        # No rebalance because halt
        assert summary["rebalance_orders"] == []
        assert any(b.action == "HALT_AND_FLATTEN" for b in summary["breakers"])
