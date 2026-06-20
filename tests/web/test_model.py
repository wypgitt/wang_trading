"""Tests for ``GET /api/v1/model`` — the production meta-labeler card.

No MLflow/DB needed: the ``ModelService`` is fully injectable. We override
the provider with a service whose ``registry_factory`` returns a fake
registry, and assert that REAL fields (cvScore, trainAcc, trainingEvents,
gates, retrainTimeline) flow through camelCased, that the COMING fields
(auc/brier/ece, calibration, featureImportance, drift) stay null/empty, and
that the envelope carries ``modelVersion``. Two degrade paths and the
meta-prob histogram are covered too.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.web.app import app
from src.web.deps import get_model_service
from src.web.services.model_service import ModelService


class _FakeRegistry:
    """Stand-in for ``ModelRegistry`` with canned production metadata."""

    def __init__(self, *, production=None, history=None, raise_on_get=False):
        self._production = production
        self._history = history if history is not None else []
        self._raise_on_get = raise_on_get

    def get_production_model(self):
        if self._raise_on_get:
            raise RuntimeError("mlflow store unreachable")
        return self._production

    def get_best_model(self, metric="mean_cv_score", n=1):
        return self._history


def _override(service: ModelService):
    app.dependency_overrides[get_model_service] = lambda: service


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.pop(get_model_service, None)


def _production_info(trained_at):
    return {
        "run_id": "run-abc123",
        "version": "7",
        "name": "meta-labeler",
        "stage": "Production",
        "alias": "production",
        "source": "alias",
        "params": {"model_type": "lightgbm", "n_features": "42"},
        "metrics": {"mean_cv_score": 0.612, "train_accuracy": 0.731},
        "gates": {"cpcv": True, "dsr": True, "pbo": False},
        "n_training_events": 1840,
        "trained_at": trained_at,
    }


def test_model_card_maps_real_fields(client):
    trained_at = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
    now = datetime(2026, 6, 19, 0, 0, tzinfo=timezone.utc)  # +12h
    history = [
        {"run_id": "run-abc123", "params": {}, "metrics": {"mean_cv_score": 0.612}},
        {"run_id": "run-old999", "params": {}, "metrics": {"mean_cv_score": 0.588}},
    ]
    registry = _FakeRegistry(
        production=_production_info(trained_at), history=history,
    )
    service = ModelService(
        registry_factory=lambda: registry,
        now_fn=lambda: now,
    )
    _override(service)

    resp = client.get("/api/v1/model")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]

    # REAL fields, camelCased out.
    assert data["version"] == "7"
    assert data["runId"] == "run-abc123"
    assert data["type"] == "lightgbm"
    assert data["cvScore"] == 0.612
    assert data["trainAcc"] == 0.731
    assert data["trainingEvents"] == 1840
    assert data["gates"] == {"cpcv": True, "dsr": True, "pbo": False}
    assert round(data["lastRetrainHours"], 3) == 12.0

    # Retrain timeline is held empty (COMING): get_best_model is a CV-score
    # ranking with no run dates, so serving it as a chronological promote/
    # reject timeline would be misleading. Empty until a time-ordered query lands.
    assert data["retrainTimeline"] == []
    assert any("retrain timeline" in w for w in body["warnings"])

    # COMING / not-logged: absent (exclude_none drops them) or empty.
    assert "auc" not in data
    assert "brier" not in data
    assert "ece" not in data
    assert data["featureImportance"] == []
    assert data["drift"] == []
    assert data["calibration"] == []
    assert data["metaProbHist"] == []  # default provider returns []

    # Envelope carries the model version and the honesty warning.
    assert body["model_version"] == "7"
    assert any("not produced by the engine yet" in w for w in body["warnings"])


def test_model_card_no_production_model(client):
    registry = _FakeRegistry(production=None)
    service = ModelService(registry_factory=lambda: registry)
    _override(service)

    resp = client.get("/api/v1/model")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]

    # All-null card => the client's MODEL_REQUIRED state.
    assert "version" not in data
    assert "cvScore" not in data
    assert data["retrainTimeline"] == []
    assert data["metaProbHist"] == []
    assert body.get("model_version") is None
    assert any("no production model" in w for w in body["warnings"])


def test_model_card_registry_unavailable(client):
    registry = _FakeRegistry(raise_on_get=True)
    service = ModelService(registry_factory=lambda: registry)
    _override(service)

    resp = client.get("/api/v1/model")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" not in body["data"]
    assert any("model registry unavailable" in w for w in body["warnings"])


def test_model_card_meta_prob_histogram(client):
    registry = _FakeRegistry(production=_production_info(None))
    service = ModelService(
        registry_factory=lambda: registry,
        meta_probs_provider=lambda: [0.05, 0.15, 0.55, 0.62],
    )
    _override(service)

    resp = client.get("/api/v1/model")
    assert resp.status_code == 200
    hist = {b["bucket"]: b["count"] for b in resp.json()["data"]["metaProbHist"]}

    # Ten bins '0.0'..'0.9'; the four probs land in 0.0, 0.1, 0.5, 0.6.
    assert len(hist) == 10
    assert hist["0.0"] == 1
    assert hist["0.1"] == 1
    assert hist["0.5"] == 1
    assert hist["0.6"] == 1
    assert hist["0.9"] == 0
