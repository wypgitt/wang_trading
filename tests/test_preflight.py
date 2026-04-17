"""Tests for the pre-flight check system (P6.05)."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.preflight import (
    PreflightCheck,
    PreflightChecker,
    main,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _good_broker(cash: float = 10_000, can_trade: bool = True) -> MagicMock:
    b = MagicMock()
    b.heartbeat = AsyncMock(return_value=True)
    b.get_account = AsyncMock(return_value={"buying_power": cash, "cash": cash, "nav": cash})
    b.can_trade = can_trade
    b.__class__.__name__ = "CCXTBrokerAdapter"
    return b


def _good_factory() -> MagicMock:
    f = MagicMock()
    b = _good_broker()
    f.heartbeat_all = AsyncMock(return_value={"crypto": True})
    f.get_all_brokers = MagicMock(return_value={"crypto": b})
    return f


def _good_model_registry() -> MagicMock:
    r = MagicMock()
    r.get_production_model = MagicMock(return_value={
        "trained_at": datetime.now(timezone.utc) - timedelta(days=5),
        "n_training_events": 1200,
        "gates": {"cpcv": True, "dsr": True, "pbo": True},
    })
    regime = MagicMock()
    regime.is_ready = MagicMock(return_value=True)
    r.regime_detector = regime
    return r


def _good_paper_stats() -> dict:
    return {
        "weeks_history": 10, "sharpe": 1.5, "max_drawdown": 0.08,
        "win_rate": 0.55, "n_trades": 120,
    }


def _good_infra() -> dict:
    return {
        "db_reachable": True, "db_disk_pct": 42.0,
        "prometheus_up": True, "grafana_up": True, "alerts_ok": True,
        "mlflow_up": True, "feature_freshness_h": 2.0,
    }


def _good_risk(tmp_path: Path) -> dict:
    dms = tmp_path / "dms.lock"
    dms.write_text("ok")
    return {
        "max_single_position": 0.05,
        "max_daily_loss": 0.015,
        "circuit_breakers": {"enabled": True},
        "dead_mans_switch_path": str(dms),
    }


def _good_operator() -> dict:
    return {
        "risk_acknowledged": True,
        "emergency_contact": "ops@example.com",
        "within_working_hours": True,
    }


def _full_checker(tmp_path: Path) -> PreflightChecker:
    return PreflightChecker(
        broker_factory=_good_factory(),
        model_registry=_good_model_registry(),
        paper_stats=_good_paper_stats(),
        infra=_good_infra(),
        risk_config=_good_risk(tmp_path),
        operator_config=_good_operator(),
    )


# ── Category checks ──────────────────────────────────────────────────────

class TestBrokerConnectivity:
    def test_all_pass(self):
        checker = PreflightChecker(broker_factory=_good_factory())
        checks = asyncio.run(checker.check_broker_connectivity())
        names = {c.name: c for c in checks}
        assert names["broker.heartbeat"].passed
        assert names["broker.buying_power"].passed
        assert names["broker.trading_permissions"].passed
        assert names["broker.market_open"].passed

    def test_heartbeat_failure(self):
        f = _good_factory()
        f.heartbeat_all = AsyncMock(return_value={"crypto": False})
        checker = PreflightChecker(broker_factory=f)
        checks = asyncio.run(checker.check_broker_connectivity())
        hb = next(c for c in checks if c.name == "broker.heartbeat")
        assert not hb.passed

    def test_low_buying_power(self):
        f = _good_factory()
        b = _good_broker(cash=10.0)
        f.get_all_brokers.return_value = {"crypto": b}
        f.heartbeat_all = AsyncMock(return_value={"crypto": True})
        checker = PreflightChecker(broker_factory=f)
        checks = asyncio.run(checker.check_broker_connectivity())
        bp = next(c for c in checks if c.name == "broker.buying_power")
        assert not bp.passed

    def test_missing_factory(self):
        checker = PreflightChecker()
        checks = asyncio.run(checker.check_broker_connectivity())
        assert checks[0].passed is False


class TestModelReadiness:
    def test_all_pass(self):
        checker = PreflightChecker(model_registry=_good_model_registry())
        checks = asyncio.run(checker.check_model_readiness())
        for c in checks:
            assert c.passed, f"{c.name}: {c.message}"

    def test_old_model_fails_age(self):
        r = _good_model_registry()
        r.get_production_model.return_value = {
            "trained_at": datetime.now(timezone.utc) - timedelta(days=60),
            "n_training_events": 1000,
            "gates": {"cpcv": True, "dsr": True, "pbo": True},
        }
        checker = PreflightChecker(model_registry=r)
        checks = asyncio.run(checker.check_model_readiness())
        age = next(c for c in checks if c.name == "model.age")
        assert not age.passed

    def test_missing_gate_fails(self):
        r = _good_model_registry()
        r.get_production_model.return_value = {
            "trained_at": datetime.now(timezone.utc) - timedelta(days=5),
            "n_training_events": 1000,
            "gates": {"cpcv": True, "dsr": False, "pbo": True},
        }
        checker = PreflightChecker(model_registry=r)
        checks = asyncio.run(checker.check_model_readiness())
        g = next(c for c in checks if c.name == "model.gates")
        assert not g.passed
        assert "dsr" in g.details["missing"]


class TestPaperValidation:
    def test_good_stats_all_pass(self):
        checker = PreflightChecker(paper_stats=_good_paper_stats())
        checks = asyncio.run(checker.check_paper_validation())
        assert all(c.passed for c in checks)

    def test_low_sharpe_fails(self):
        stats = _good_paper_stats() | {"sharpe": 0.3}
        checker = PreflightChecker(paper_stats=stats)
        checks = asyncio.run(checker.check_paper_validation())
        sharpe = next(c for c in checks if c.name == "paper.sharpe")
        assert not sharpe.passed


class TestInfrastructure:
    def test_all_pass(self):
        checker = PreflightChecker(infra=_good_infra())
        checks = asyncio.run(checker.check_infrastructure())
        assert all(c.passed for c in checks), [c.name for c in checks if not c.passed]

    def test_disk_too_full(self):
        infra = _good_infra() | {"db_disk_pct": 95.0}
        checker = PreflightChecker(infra=infra)
        checks = asyncio.run(checker.check_infrastructure())
        db = next(c for c in checks if c.name == "infra.timescaledb")
        assert not db.passed


class TestRiskLimits:
    def test_all_pass(self, tmp_path):
        checker = PreflightChecker(risk_config=_good_risk(tmp_path))
        checks = asyncio.run(checker.check_risk_limits())
        assert all(c.passed for c in checks)

    def test_excessive_position_limit_fails(self, tmp_path):
        risk = _good_risk(tmp_path) | {"max_single_position": 0.5}
        checker = PreflightChecker(risk_config=risk)
        checks = asyncio.run(checker.check_risk_limits())
        c = next(c for c in checks if c.name == "risk.max_single_position")
        assert not c.passed

    def test_missing_dead_mans_switch(self, tmp_path):
        risk = _good_risk(tmp_path) | {
            "dead_mans_switch_path": str(tmp_path / "does-not-exist")
        }
        checker = PreflightChecker(risk_config=risk)
        checks = asyncio.run(checker.check_risk_limits())
        c = next(c for c in checks if c.name == "risk.dead_mans_switch")
        assert not c.passed

    def test_old_dead_mans_switch_fails(self, tmp_path):
        dms = tmp_path / "stale.lock"
        dms.write_text("x")
        old_ts = time.time() - 48 * 3600
        import os
        os.utime(dms, (old_ts, old_ts))
        risk = _good_risk(tmp_path) | {"dead_mans_switch_path": str(dms)}
        checker = PreflightChecker(risk_config=risk)
        checks = asyncio.run(checker.check_risk_limits())
        c = next(c for c in checks if c.name == "risk.dead_mans_switch")
        assert not c.passed


class TestOperator:
    def test_all_pass(self):
        checker = PreflightChecker(operator_config=_good_operator())
        checks = asyncio.run(checker.check_operator())
        assert all(c.passed for c in checks)
        assert all(c.severity == "warning" for c in checks)


# ── Summary + CLI ────────────────────────────────────────────────────────

class TestSummary:
    def test_all_pass_sets_all_passed(self, tmp_path):
        checks = asyncio.run(_full_checker(tmp_path).run_all_checks())
        summary = PreflightChecker.summary(checks)
        assert summary["all_passed"] is True
        assert summary["blockers_failed"] == 0

    def test_blocker_fail_blocks_go(self):
        checks = [
            PreflightCheck("b.fail", "x", severity="blocker", passed=False),
            PreflightCheck("b.ok", "x", severity="blocker", passed=True),
        ]
        s = PreflightChecker.summary(checks)
        assert s["all_passed"] is False
        assert s["blockers_failed"] == 1

    def test_warning_only_all_passed_true(self):
        checks = [
            PreflightCheck("w.fail", "x", severity="warning", passed=False),
            PreflightCheck("b.ok", "x", severity="blocker", passed=True),
        ]
        s = PreflightChecker.summary(checks)
        assert s["all_passed"] is True
        assert s["warnings_failed"] == 1


class TestCLI:
    def test_exit_code_1_when_missing_deps(self, capsys):
        code = main(["--full-check"])
        assert code == 1  # no broker_factory / registry → blockers fail
        captured = capsys.readouterr()
        assert "all_passed" in captured.out

    def test_category_runs(self, capsys):
        code = main(["--check", "operator"])
        # operator-only checks are warnings → blockers_failed=0, warnings>0 → exit 2
        assert code == 2
