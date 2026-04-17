"""Production smoke test (P6.17).

End-to-end validation that exercises every Phase 6 system against a paper
broker configured like production: preflight → startup → 60 cycles of
trading → circuit-breaker trigger → snapshot taken → audit log verified →
alerts fired (mock Telegram) → graceful shutdown.

Target wall time: < 10 minutes. In practice it completes in seconds because
the pipeline config uses ``sleep_seconds=0``.

Run:
    python scripts/production_smoke_test.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.audit_log import ComplianceAuditLogger
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.capital_deployment import CapitalDeploymentController
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.disaster_recovery import RecoveryManager, SnapshotManager
from src.execution.live_trading import LiveTradingPipeline
from src.execution.models import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PipelineConfig
from src.execution.preflight import PreflightCheck
from src.monitoring.alerting import AlertManager, AlertSeverity, Alert
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("smoke")


EQ_COST = {
    "commission_per_share": 0.0,
    "min_commission": 0.0,
    "spread_bps": 1.0,
    "slippage_bps": 1.0,
    "impact_coefficient": 0.1,
}


# ── Test doubles ─────────────────────────────────────────────────────────

class _MockTelegramChannel:
    """Captures every alert the pipeline sends instead of hitting Telegram."""
    def __init__(self) -> None:
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return True


class _PassingPreflight:
    async def run_all_checks(self):
        return [PreflightCheck("ok", "smoke", severity="blocker", passed=True)]

    @staticmethod
    def summary(checks):
        return {"all_passed": True, "blockers_failed": 0, "warnings_failed": 0,
                "checks": [c.to_dict() for c in checks]}


class _OptimizerStub:
    """Rotates through two symbols across cycles to force churn + fills."""
    def __init__(self) -> None:
        self._cycle = 0

    def compute_target_portfolio(self, **kwargs) -> pd.DataFrame:
        self._cycle += 1
        weight = 0.05 if self._cycle % 2 == 0 else 0.02
        return pd.DataFrame([
            {"symbol": "AAPL", "target_weight": weight, "strategy": "mom"},
            {"symbol": "MSFT", "target_weight": weight / 2, "strategy": "mom"},
        ])


class _SignalBatteryStub:
    def generate(self, features):
        return pd.DataFrame({"symbol": ["AAPL"], "family": ["mom"]})


class _MetaPipelineStub:
    def predict(self, features, signals):
        return pd.DataFrame({"symbol": ["AAPL"], "meta_prob": [0.7]})


# ── Setup ────────────────────────────────────────────────────────────────

def _build(tmp_dir: Path) -> tuple[LiveTradingPipeline, PaperBrokerAdapter,
                                   PortfolioState, _MockTelegramChannel,
                                   ComplianceAuditLogger, SnapshotManager,
                                   MetricsCollector]:
    halt = tmp_dir / "halt.lock"
    checkin = tmp_dir / "checkin.lock"
    checkin.write_text("ok")
    compliance_log = tmp_dir / "compliance.log"

    portfolio = PortfolioState(cash=1_000_000.0)
    prices = {"AAPL": 100.0, "MSFT": 200.0}
    broker = PaperBrokerAdapter(
        initial_cash=1_000_000.0, slippage_bps=0.5, fill_delay_ms=0,
        price_feed=lambda s: prices.get(s, 100.0),
    )
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST)
    om = OrderManager(broker, cbs, cost, portfolio)

    metrics = MetricsCollector(registry=CollectorRegistry())

    telegram = _MockTelegramChannel()
    alerts = AlertManager(channel_map={s: [telegram] for s in AlertSeverity})

    controller = CapitalDeploymentController(portfolio=portfolio)
    broker_factory = type("BF", (), {
        "heartbeat_all": staticmethod(lambda: asyncio.sleep(0, result={"crypto": True})),
        "get_all_brokers": lambda self: {"crypto": broker},
    })()

    snapshot_mgr = SnapshotManager(directory=tmp_dir / "snapshots")
    recovery = RecoveryManager(
        snapshot_manager=snapshot_mgr,
        halt_file=halt,
        crash_file=tmp_dir / "crash.lock",
    )
    audit = ComplianceAuditLogger(signing_key="smoke-test-signing-key")

    pipeline = LiveTradingPipeline(
        broker_factory=broker_factory,
        preflight_checker=_PassingPreflight(),
        deployment_controller=controller,
        halt_file=halt,
        operator_checkin_path=checkin,
        compliance_log_path=compliance_log,
        recovery_manager=None,  # handler install requires main thread
        data_adapter=None,
        bar_constructors={},
        feature_assembler=None,
        signal_battery=_SignalBatteryStub(),
        meta_pipeline=_MetaPipelineStub(),
        meta_labeler=None,
        bet_sizing=None,
        portfolio_optimizer=_OptimizerStub(),
        order_manager=om,
        metrics=metrics,
        alert_manager=alerts,
        drift_detector=FeatureDriftDetector(),
        config=PipelineConfig(sleep_seconds=0.0, drift_check_every=1000),
    )
    return pipeline, broker, portfolio, telegram, audit, snapshot_mgr, metrics


# ── Checks ───────────────────────────────────────────────────────────────

def _check(cond: bool, label: str) -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}")
    if not cond:
        raise AssertionError(label)


async def _drive_cycles(pipeline: LiveTradingPipeline, n: int,
                        audit: ComplianceAuditLogger,
                        prices: dict[str, float]) -> None:
    for i in range(n):
        features = pd.DataFrame({"ret": [0.001, 0.002]})
        result = await pipeline.run_cycle(features=features, prices=prices)
        if i == 0:
            audit.log_operator_action("pipeline_started", operator="smoke")
        # Audit-log every order submitted this cycle
        for order in result.get("orders", []) or []:
            audit.log_order(order)
        for exit_order in result.get("exits", []) or []:
            audit.log_order(exit_order)


async def _trigger_circuit_breaker(
    pipeline: LiveTradingPipeline,
    portfolio: PortfolioState,
    audit: ComplianceAuditLogger,
) -> None:
    # Force a deep drawdown to wake the NAV_DRAWDOWN breaker
    portfolio.peak_nav = portfolio.nav / 0.75
    portfolio.drawdown = 0.25
    result = await pipeline.run_cycle(prices={"AAPL": 100.0, "MSFT": 200.0})
    for breaker in result.get("breakers", []) or []:
        audit.log_breaker(breaker)


# ── Main ────────────────────────────────────────────────────────────────

async def main() -> int:
    print("=" * 68)
    print("wang_trading production smoke test — P6.17")
    print("=" * 68)

    t0 = time.monotonic()
    import tempfile
    with tempfile.TemporaryDirectory() as td_str:
        tmp_dir = Path(td_str)
        pipeline, broker, portfolio, telegram, audit, snapshot_mgr, metrics = _build(tmp_dir)

        # ── Phase 1: startup ────────────────────────────────────────
        print("\n[1/6] Startup sequence")
        startup = await pipeline.startup_sequence()
        _check(startup["preflight"]["all_passed"] is True, "preflight all_passed")
        _check(startup["phase"]["phase_id"] == 1, "starts at deployment phase 1")
        _check(len(telegram.sent) >= 1, "startup alert fired")

        # ── Phase 2: 60 cycles of trading ──────────────────────────
        print("\n[2/6] Drive 60 trading cycles")
        await _drive_cycles(pipeline, 60, audit,
                            prices={"AAPL": 100.0, "MSFT": 200.0})
        _check(pipeline.cycle_count == 60, "60 cycles completed")
        _check(pipeline._orders_placed_count > 0, "orders placed")
        _check(len(broker.orders) > 0, "broker received orders")

        # ── Phase 3: circuit breaker trigger ───────────────────────
        print("\n[3/6] Trigger a circuit breaker")
        breaker_alerts_before = len(telegram.sent)
        await _trigger_circuit_breaker(pipeline, portfolio, audit)
        _check(len(telegram.sent) > breaker_alerts_before,
               "breaker alert fan-out")

        # ── Phase 4: snapshot + audit chain ────────────────────────
        print("\n[4/6] Take snapshot + verify audit chain")
        snap = await snapshot_mgr.take_snapshot(pipeline)
        _check(snap.checksum != "", "snapshot checksum written")
        chain_result = snapshot_mgr.verify_snapshot_chain()
        _check(chain_result["ok"] is True, "snapshot chain clean")

        audit_verify = audit.verify_chain()
        _check(audit_verify["ok"] is True, "audit log chain clean")
        _check(audit_verify["total"] >= 2, "audit log populated")

        # ── Phase 5: metrics exposed ───────────────────────────────
        print("\n[5/6] Metrics exposed")
        snapshot = metrics.snapshot()
        _check("wang_trading_portfolio_nav" in snapshot,
               "portfolio_nav metric exported")
        _check(any("circuit_breaker" in k for k in snapshot),
               "circuit_breaker metric exported")

        # ── Phase 6: graceful shutdown ─────────────────────────────
        print("\n[6/6] Graceful shutdown")
        await pipeline.shutdown()
        _check(pipeline.halt_file.exists(), "HALT file written on shutdown")
        _check(any("shutdown" in (a.title or "").lower() for a in telegram.sent),
               "shutdown alert fired")

    elapsed = time.monotonic() - t0
    print("\n" + "=" * 68)
    print(f"SMOKE TEST PASSED in {elapsed:.2f}s "
          f"({elapsed / 60:.2f} min, budget 10 min)")
    print("=" * 68)
    if elapsed > 600:
        print("WARNING: exceeded the 10-minute budget.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
