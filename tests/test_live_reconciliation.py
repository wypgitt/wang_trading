"""Focused tests for live startup/shutdown execution reconciliation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.live_trading import LiveTradingPipeline
from src.execution.models import Order, OrderStatus, OrderType, PortfolioState, Position
from src.execution.order_manager import OrderManager
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector


class _Preflight:
    async def run_all_checks(self):
        return []

    def summary(self, checks):
        return {"all_passed": True, "blockers_failed": 0, "checks": []}


class _Deployment:
    _halted = False

    def get_current_phase(self):
        return SimpleNamespace(
            name="paper_rehearsal",
            position_size_multiplier=1.0,
            to_dict=lambda: {"name": "paper_rehearsal", "position_size_multiplier": 1.0},
        )

    def get_size_multiplier(self):
        return 1.0


class _Alerts:
    def __init__(self):
        self.sent = []

    async def send_alert(self, alert):
        self.sent.append(alert)


class _ExternalOpenOrderBroker(PaperBrokerAdapter):
    def __init__(self):
        super().__init__(fill_delay_ms=0)
        self.external_open_order = Order(
            order_id="external-open-1",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            side=1,
            order_type=OrderType.LIMIT,
            quantity=3.0,
            status=OrderStatus.SUBMITTED,
        )

    async def get_open_orders(self, symbols=None):
        if symbols and self.external_open_order.symbol not in symbols:
            return []
        if self.external_open_order.is_active:
            return [self.external_open_order]
        return []

    async def cancel_order(self, order_id: str) -> bool:
        if order_id == self.external_open_order.order_id:
            self.external_open_order.status = OrderStatus.CANCELLED
            return True
        return await super().cancel_order(order_id)


def _cost_model() -> TransactionCostModel:
    cfg = {
        "commission_per_share": 0.0,
        "min_commission": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "impact_coefficient": 0.0,
    }
    return TransactionCostModel(equities_config=cfg)


def _pipeline(tmp_path, broker: PaperBrokerAdapter, portfolio: PortfolioState):
    operator = tmp_path / "operator_checkin"
    operator.write_text("ok", encoding="utf-8")
    order_manager = OrderManager(
        broker,
        CircuitBreakerManager(),
        _cost_model(),
        portfolio,
    )
    return LiveTradingPipeline(
        broker_factory=SimpleNamespace(),
        preflight_checker=_Preflight(),
        deployment_controller=_Deployment(),
        halt_file=tmp_path / "halt",
        operator_checkin_path=operator,
        compliance_log_path=tmp_path / "compliance.log",
        data_adapter=None,
        bar_constructors={},
        feature_assembler=None,
        signal_battery=None,
        meta_pipeline=None,
        meta_labeler=None,
        bet_sizing=None,
        portfolio_optimizer=None,
        order_manager=order_manager,
        metrics=MetricsCollector(),
        alert_manager=_Alerts(),
        drift_detector=FeatureDriftDetector(),
    )


def test_startup_reconciliation_blocks_unmanaged_broker_position(tmp_path):
    broker = PaperBrokerAdapter(fill_delay_ms=0)
    broker.positions["AAPL"] = Position(
        symbol="AAPL",
        side=1,
        quantity=5.0,
        avg_entry_price=100.0,
        entry_timestamp=datetime.now(timezone.utc),
        signal_family="external",
        current_price=100.0,
    )
    pipeline = _pipeline(tmp_path, broker, PortfolioState(cash=100_000.0))

    with pytest.raises(RuntimeError, match="startup reconciliation failed"):
        asyncio.run(pipeline.startup_sequence())


def test_reconciliation_detects_order_status_and_fill_mismatch(tmp_path):
    broker = PaperBrokerAdapter(fill_delay_ms=0)
    portfolio = PortfolioState(cash=100_000.0)
    internal = Order(
        order_id="open-1",
        timestamp=datetime.now(timezone.utc),
        symbol="AAPL",
        side=1,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        status=OrderStatus.SUBMITTED,
    )
    broker.orders["open-1"] = Order(
        order_id="open-1",
        timestamp=internal.timestamp,
        symbol="AAPL",
        side=1,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        filled_quantity=5.0,
        status=OrderStatus.PARTIAL_FILL,
    )
    portfolio.open_orders.append(internal)
    pipeline = _pipeline(tmp_path, broker, portfolio)

    summary = asyncio.run(
        pipeline._reconcile_execution_state("unit", strict=False)
    )

    assert summary["order_diffs"]
    fields = summary["order_diffs"][0]["fields"]
    assert "status" in fields
    assert "filled_quantity" in fields


def test_reconciliation_uses_broker_open_order_endpoint(tmp_path):
    broker = _ExternalOpenOrderBroker()
    pipeline = _pipeline(tmp_path, broker, PortfolioState(cash=100_000.0))

    summary = asyncio.run(
        pipeline._reconcile_execution_state("unit", strict=False)
    )
    verify = asyncio.run(pipeline.verify_flat())

    assert summary["order_diffs"][0]["order_id"] == "external-open-1"
    assert verify["flat"] is False
    assert verify["broker_open_orders"][0]["order_id"] == "external-open-1"


def test_shutdown_cancels_broker_open_orders_not_in_internal_ledger(tmp_path):
    broker = _ExternalOpenOrderBroker()
    pipeline = _pipeline(tmp_path, broker, PortfolioState(cash=100_000.0))

    asyncio.run(pipeline.shutdown())
    verify = asyncio.run(pipeline.verify_flat())

    assert verify["flat"] is True
    assert broker.external_open_order.status == OrderStatus.CANCELLED
