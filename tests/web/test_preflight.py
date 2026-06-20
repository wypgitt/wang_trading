"""Tests for ``GET /api/v1/preflight`` — the go-live checklist.

The route now runs the real ``PreflightChecker`` vector. We inject a fake
checker (no broker/model/infra deps) so the test maps engine checks → DTO
status and the overall verdict without any live infrastructure.
"""

from __future__ import annotations

from src.web.app import app
from src.web.deps import get_preflight_service
from src.web.services.preflight_service import PreflightService


class _FakeCheck:
    def __init__(self, name, passed, severity="blocker", message=""):
        self.name = name
        self.passed = passed
        self.severity = severity
        self.message = message


class _FakeChecker:
    def __init__(self, checks):
        self._checks = checks

    async def run_all_checks(self):
        return self._checks


def _override(service: PreflightService):
    app.dependency_overrides[get_preflight_service] = lambda: service


def _clear():
    app.dependency_overrides.pop(get_preflight_service, None)


def test_preflight_blocked_when_a_blocker_fails(client):
    checks = [
        _FakeCheck("broker_connectivity", False, "blocker", "broker not configured"),
        _FakeCheck("model_readiness", True, "blocker", "production model loaded"),
    ]
    _override(PreflightService(checker_factory=lambda: _FakeChecker(checks)))
    try:
        resp = client.get("/api/v1/preflight")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["overall"] == "BLOCKED"
        by_name = {c["name"]: c for c in data["checks"]}
        assert by_name["broker_connectivity"]["status"] == "FAIL"
        assert by_name["broker_connectivity"]["reason"] == "broker not configured"
        assert by_name["model_readiness"]["status"] == "PASS"
        # camelCase + always-present envelope metadata.
        assert "evaluatedAt" in by_name["model_readiness"]
        assert "staleness_seconds" in resp.json()
    finally:
        _clear()


def test_preflight_ready_when_all_pass(client):
    checks = [
        _FakeCheck("broker_connectivity", True, "blocker", ""),
        _FakeCheck("infrastructure", True, "warning", ""),
    ]
    _override(PreflightService(checker_factory=lambda: _FakeChecker(checks)))
    try:
        data = client.get("/api/v1/preflight").json()["data"]
        assert data["overall"] == "READY"
        assert data["infraProbes"] == {}  # no probe wired ⇒ empty, not fabricated
    finally:
        _clear()


def test_preflight_unknown_when_checker_unavailable(client):
    def _boom():
        raise RuntimeError("preflight stack import failed")

    _override(PreflightService(checker_factory=_boom))
    try:
        resp = client.get("/api/v1/preflight")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["overall"] == "UNKNOWN"
    finally:
        _clear()


def test_preflight_failing_warning_does_not_block(client):
    # A FAILED *warning*-severity check surfaces per-check but must NOT block
    # go-live: only a failed blocker flips overall to BLOCKED.
    checks = [
        _FakeCheck("broker_connectivity", True, "blocker", ""),
        _FakeCheck("ntp_drift", False, "warning", "clock drift 1.2s"),
    ]
    _override(PreflightService(checker_factory=lambda: _FakeChecker(checks)))
    try:
        data = client.get("/api/v1/preflight").json()["data"]
        assert data["overall"] == "READY"
        by_name = {c["name"]: c for c in data["checks"]}
        assert by_name["ntp_drift"]["status"] == "FAIL"
    finally:
        _clear()


def test_preflight_infra_probes_populated_when_wired(client):
    class _FakeProbe:
        async def collect(self):
            return {"db_reachable": True, "mlflow_up": False, "db_disk_pct": 41.2}

    checks = [_FakeCheck("broker_connectivity", True, "blocker", "")]
    _override(
        PreflightService(
            checker_factory=lambda: _FakeChecker(checks),
            probe_factory=lambda: _FakeProbe(),
        )
    )
    try:
        data = client.get("/api/v1/preflight").json()["data"]
        # Probe results are stringified into infraProbes (no fabrication).
        assert data["infraProbes"]["db_reachable"] == "True"
        assert data["infraProbes"]["mlflow_up"] == "False"
    finally:
        _clear()
