"""Tests for LiveTradingPipeline (P6.07)."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.capital_deployment import CapitalDeploymentController
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.live_trading import LiveTradingPipeline, _ScaledBetSizing
from src.execution.models import (
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PipelineConfig
from src.execution.preflight import PreflightCheck
from src.backtesting.transaction_costs import TransactionCostModel

EQ_COST_CFG = {
    "commission_per_share": 0.0, "min_commission": 0.0,
    "spread_bps": 1.0, "slippage_bps": 1.0, "impact_coefficient": 0.1,
}
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector

pytestmark = pytest.mark.integration


# ── Fixtures / factories ─────────────────────────────────────────────────

def _fresh_paths(tmp_path: Path) -> dict[str, Path]:
    halt = tmp_path / "halt.lock"
    checkin = tmp_path / "checkin.lock"
    checkin.write_text("ok")  # fresh
    compliance = tmp_path / "compliance.log"
    return {"halt": halt, "checkin": checkin, "compliance": compliance}


class _Preflight:
    def __init__(self, *, passes: bool = True) -> None:
        self.passes = passes
        self._checks = [
            PreflightCheck("c1", "x", severity="blocker", passed=passes),
        ]

    async def run_all_checks(self):
        return self._checks

    @staticmethod
    def summary(checks):
        blockers = sum(1 for c in checks if c.severity == "blocker" and not c.passed)
        return {"all_passed": blockers == 0, "blockers_failed": blockers,
                "warnings_failed": 0, "checks": [c.to_dict() for c in checks]}


def _make_pipeline(
    tmp_path: Path, *, preflight_passes: bool = True,
    bets: pd.DataFrame | None = None,
    portfolio: PortfolioState | None = None,
) -> tuple[LiveTradingPipeline, OrderManager, PaperBrokerAdapter, PortfolioState]:
    paths = _fresh_paths(tmp_path)
    portfolio = portfolio or PortfolioState(cash=1_000_000.0)
    broker = PaperBrokerAdapter(
        initial_cash=1_000_000.0, slippage_bps=0.0, fill_delay_ms=0,
        price_feed=lambda s: 100.0,
    )
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST_CFG)
    om = OrderManager(broker, cbs, cost, portfolio)
    metrics = MetricsCollector(registry=CollectorRegistry())
    alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity})

    class _InnerSizing:
        def compute(self, meta, features):
            return bets if bets is not None else pd.DataFrame({"symbol": [], "size": []})

    controller = CapitalDeploymentController(portfolio=portfolio)
    broker_factory = MagicMock()
    broker_factory.heartbeat_all = AsyncMock(return_value={"crypto": True})

    pipeline = LiveTradingPipeline(
        broker_factory=broker_factory,
        preflight_checker=_Preflight(passes=preflight_passes),
        deployment_controller=controller,
        halt_file=paths["halt"],
        operator_checkin_path=paths["checkin"],
        compliance_log_path=paths["compliance"],
        data_adapter=None,
        bar_constructors={},
        feature_assembler=None,
        signal_battery=None,
        meta_pipeline=None,
        meta_labeler=None,
        bet_sizing=_InnerSizing(),
        portfolio_optimizer=None,
        order_manager=om,
        metrics=metrics,
        alert_manager=alerts,
        drift_detector=FeatureDriftDetector(),
        config=PipelineConfig(max_cycles=1, sleep_seconds=0.0, drift_check_every=1000),
    )
    return pipeline, om, broker, portfolio


# ── Tests ────────────────────────────────────────────────────────────────

class TestStartup:
    def test_preflight_failure_blocks(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path, preflight_passes=False)
        with pytest.raises(RuntimeError, match="preflight failed"):
            asyncio.run(pipeline.startup_sequence())

    def test_halt_file_blocks_startup(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        pipeline.halt_file.write_text("halted")
        with pytest.raises(RuntimeError, match="HALT file"):
            asyncio.run(pipeline.startup_sequence())

    def test_stale_operator_checkin_blocks(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        old = time.time() - 2 * 3600
        import os
        os.utime(pipeline.operator_checkin_path, (old, old))
        with pytest.raises(RuntimeError, match="Operator check-in"):
            asyncio.run(pipeline.startup_sequence())

    def test_happy_path_returns_summary(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        out = asyncio.run(pipeline.startup_sequence())
        assert out["preflight"]["all_passed"] is True
        assert out["phase"]["phase_id"] == 1


# ── Deployment multiplier ───────────────────────────────────────────────

class TestDeploymentMultiplier:
    def test_scales_final_size(self):
        bets = pd.DataFrame({"symbol": ["AAPL"], "final_size": [1.0]})

        class _Inner:
            def compute(self, meta, features):
                return bets

        controller = MagicMock()
        controller.get_size_multiplier = MagicMock(return_value=0.25)
        wrapped = _ScaledBetSizing(_Inner(), controller)
        out = wrapped.compute(None, None)
        assert out["final_size"].iloc[0] == pytest.approx(0.25)

    def test_multiplier_one_is_passthrough(self):
        bets = pd.DataFrame({"symbol": ["AAPL"], "final_size": [1.0]})

        class _Inner:
            def compute(self, meta, features):
                return bets

        controller = MagicMock()
        controller.get_size_multiplier = MagicMock(return_value=1.0)
        wrapped = _ScaledBetSizing(_Inner(), controller)
        out = wrapped.compute(None, None)
        assert out["final_size"].iloc[0] == 1.0


# ── Emergency flatten ───────────────────────────────────────────────────

class TestShutdownFlatten:
    def test_emergency_flatten_submits_exit_orders(self, tmp_path):
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=datetime.now(timezone.utc), signal_family="",
            current_price=100.0,
        )
        pf = PortfolioState(cash=500_000.0, positions={"AAPL": pos})
        pipeline, om, broker, pf = _make_pipeline(tmp_path, portfolio=pf)
        # Put an open order to be cancelled
        live_order = Order(
            order_id="o1", timestamp=datetime.now(timezone.utc),
            symbol="AAPL", side=1, order_type=OrderType.LIMIT,
            quantity=10, limit_price=100.0, status=OrderStatus.SUBMITTED,
        )
        broker.orders[live_order.order_id] = live_order
        pf.open_orders.append(live_order)

        asyncio.run(pipeline.shutdown(emergency=True))

        # Open order cancelled
        assert live_order.status == OrderStatus.CANCELLED
        # HALT file written
        assert pipeline.halt_file.exists()
        # Flatten submitted an AAPL sell order
        aapl_sells = [
            o for o in broker.orders.values()
            if o.symbol == "AAPL" and o.side == -1 and o.order_id != "o1"
        ]
        assert aapl_sells, "expected flatten to submit an AAPL sell order"

    def test_normal_shutdown_writes_halt_but_keeps_positions(self, tmp_path):
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=datetime.now(timezone.utc), signal_family="",
            current_price=100.0,
        )
        pf = PortfolioState(cash=500_000.0, positions={"AAPL": pos})
        pipeline, om, broker, pf = _make_pipeline(tmp_path, portfolio=pf)
        # Seed the broker with an existing position so flatten can act
        broker.positions["AAPL"] = pos

        asyncio.run(pipeline.shutdown())
        assert pipeline.halt_file.exists()
        assert broker.positions.get("AAPL") is pos  # untouched in non-emergency


# ── Halt state gating cycles ────────────────────────────────────────────

class TestCycleHaltGate:
    def test_halted_cycle_returns_early(self, tmp_path):
        pipeline, *_ = _make_pipeline(tmp_path)
        pipeline.deployment_controller.halt("test")
        result = asyncio.run(pipeline.run_cycle())
        assert result.get("halted") is True
