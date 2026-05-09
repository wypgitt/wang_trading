"""Tests for explicit operator kill-switch helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.live_trading import LiveTradingPipeline, write_halt_file
from src.execution.models import PortfolioState, Position
from src.execution.order_manager import OrderManager
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector


class _Alerts:
    async def send_alert(self, alert):
        return None


class _Preflight:
    async def run_all_checks(self):
        return []

    def summary(self, checks):
        return {"all_passed": True, "blockers_failed": 0, "checks": []}


class _Deployment:
    _halted = False

    def get_size_multiplier(self):
        return 1.0


def _pipeline(tmp_path, broker, portfolio):
    cfg = {
        "commission_per_share": 0.0,
        "min_commission": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "impact_coefficient": 0.0,
    }
    om = OrderManager(
        broker,
        CircuitBreakerManager(),
        TransactionCostModel(equities_config=cfg),
        portfolio,
    )
    return LiveTradingPipeline(
        broker_factory=SimpleNamespace(),
        preflight_checker=_Preflight(),
        deployment_controller=_Deployment(),
        halt_file=tmp_path / "halt",
        operator_checkin_path=tmp_path / "checkin",
        compliance_log_path=tmp_path / "compliance.log",
        data_adapter=None,
        bar_constructors={},
        feature_assembler=None,
        signal_battery=None,
        meta_pipeline=None,
        meta_labeler=None,
        bet_sizing=None,
        portfolio_optimizer=None,
        order_manager=om,
        metrics=MetricsCollector(registry=CollectorRegistry()),
        alert_manager=_Alerts(),
        drift_detector=FeatureDriftDetector(),
    )


def test_write_halt_file_records_reason(tmp_path):
    path = write_halt_file(tmp_path / "halt", reason="test_halt")
    text = path.read_text(encoding="utf-8")

    assert "reason=test_halt" in text
    assert "emergency=False" in text


def test_verify_flat_detects_broker_position(tmp_path):
    broker = PaperBrokerAdapter(fill_delay_ms=0)
    broker.positions["AAPL"] = Position(
        symbol="AAPL",
        side=1,
        quantity=10.0,
        avg_entry_price=100.0,
        entry_timestamp=datetime.now(timezone.utc),
        signal_family="manual",
    )
    pipeline = _pipeline(tmp_path, broker, PortfolioState(cash=100_000.0))

    result = asyncio.run(pipeline.verify_flat())

    assert not result["flat"]
    assert result["broker_positions"][0]["symbol"] == "AAPL"


def test_flatten_uses_broker_positions_not_only_internal_ledger(tmp_path):
    broker = PaperBrokerAdapter(fill_delay_ms=0, price_feed=lambda _: 100.0)
    broker.positions["AAPL"] = Position(
        symbol="AAPL",
        side=1,
        quantity=10.0,
        avg_entry_price=100.0,
        entry_timestamp=datetime.now(timezone.utc),
        signal_family="manual",
        current_price=100.0,
    )
    pipeline = _pipeline(tmp_path, broker, PortfolioState(cash=100_000.0))

    asyncio.run(pipeline.shutdown(emergency=True))
    result = asyncio.run(pipeline.verify_flat())

    assert result["flat"]
