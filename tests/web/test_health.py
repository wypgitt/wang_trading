"""Tests for the freshness SLO: ``/livez`` (liveness), ``/healthz`` (liveness +
freshness vector), ``/readyz`` (readiness — gates on a fresh snapshot).

No infra: the HealthService is overridden with readers over a seeded tmpfs
snapshot, a no-DB bars gateway, and a fake model probe.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.web.app import app
from src.web.deps import get_health_service
from src.web.dtos import ModelResponse
from src.web.services.bars_gateway import BarsGateway
from src.web.services.health_service import HealthService
from src.web.services.trade_ideas_service import TradeIdeasService


class _FakeModel:
    def __init__(self, version=None):
        self._v = version

    def get_model(self):
        return ModelResponse(version=self._v), []


def _seed(tmp_path, *, age_seconds):
    as_of = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    report = {"idea_count": 0, "totals": {}, "ideas": []}
    path = tmp_path / "trade_ideas.json"
    path.write_text(json.dumps({"schema_version": 1, "as_of": as_of.isoformat(), "report": report}))
    return path


def _install(path=None, *, model_version=None):
    ideas = TradeIdeasService(tmpfs_path=str(path)) if path else TradeIdeasService(tmpfs_path="/no/such/file")
    svc = HealthService(
        ideas=ideas,
        gateway=BarsGateway(fetch=lambda s, bt, n: None),  # no DB ⇒ last_bar_age None
        model=_FakeModel(model_version),
    )
    app.dependency_overrides[get_health_service] = lambda: svc


@pytest.fixture(autouse=True)
def _clear():
    yield
    app.dependency_overrides.clear()


def test_livez_always_ok(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json()["data"]["ok"] is True


def test_readyz_503_when_snapshot_absent(client):
    _install(None)
    r = client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["data"]["ready"] is False
    assert body["data"]["snapshot_present"] is False
    assert body["errors"][0]["code"] == "SNAPSHOT_UNAVAILABLE"


def test_readyz_503_when_snapshot_stale(client, tmp_path):
    _install(_seed(tmp_path, age_seconds=300))  # > 90s threshold
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["data"]["snapshot_stale"] is True


def test_readyz_200_when_snapshot_fresh(client, tmp_path):
    _install(_seed(tmp_path, age_seconds=5))
    r = client.get("/readyz")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["ready"] is True
    assert data["snapshot_present"] is True


def test_healthz_reports_freshness_without_gating(client, tmp_path):
    # Even a stale snapshot ⇒ /healthz stays 200 (liveness) but reports it.
    _install(_seed(tmp_path, age_seconds=300), model_version="meta_v1")
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["snapshot_present"] is True
    assert body["data"]["snapshot_stale"] is True
    assert body["data"]["model_loaded"] is True
    # Freshness rides in the envelope's source_freshness too.
    assert "snapshot" in (body.get("source_freshness") or {})
