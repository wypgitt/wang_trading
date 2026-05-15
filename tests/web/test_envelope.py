"""Envelope serialisation tests (api_contracts_v2 §0)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.web.envelope import (
    ApiEnvelope,
    ApiError,
    RegimeProbabilities,
    RegimeSnapshot,
    envelope,
)


def test_envelope_basic_roundtrip() -> None:
    payload = {"hello": "world", "count": 3}
    env = envelope(payload, source="test")
    assert env["source"] == "test"
    assert env["data"] == payload
    assert "as_of" in env
    # Timestamp parseable + UTC.
    as_of = datetime.fromisoformat(env["as_of"].replace("Z", "+00:00"))
    assert as_of.tzinfo is not None


def test_envelope_excludes_none_fields() -> None:
    env = envelope({"x": 1}, source="test")
    # None fields per §0.1 must be omitted when meaningless.
    assert "staleness_seconds" not in env
    assert "model_version" not in env
    assert "regime" not in env
    assert "source_freshness" not in env
    # Defaulted empty list fields are present (allowed) but empty.
    assert env.get("warnings", []) == []
    assert env.get("errors", []) == []


def test_envelope_includes_freshness_when_given() -> None:
    env = envelope(
        {"k": "v"},
        source="test",
        staleness_seconds=12.3,
        source_freshness={"bars": 2.1, "signals": 4.7},
        model_version="meta_v1.7.2",
    )
    assert env["staleness_seconds"] == pytest.approx(12.3)
    assert env["source_freshness"]["bars"] == pytest.approx(2.1)
    assert env["model_version"] == "meta_v1.7.2"


def test_envelope_includes_regime() -> None:
    regime = RegimeSnapshot(
        label="trending_up",
        probabilities=RegimeProbabilities(
            trending_up=0.7,
            trending_down=0.1,
            mean_reverting=0.15,
            high_volatility=0.05,
        ),
        as_of=datetime(2026, 5, 15, 18, 44, tzinfo=timezone.utc),
    )
    env = envelope({"x": 1}, source="test", regime=regime)
    assert env["regime"]["label"] == "trending_up"
    assert env["regime"]["probabilities"]["trending_up"] == pytest.approx(0.7)


def test_envelope_errors_warnings_emitted() -> None:
    err = ApiError(code="STALE_MODEL", message="degraded")
    env = envelope(
        {"x": 1},
        source="test",
        warnings=["sentiment older than 60s"],
        errors=[err],
    )
    assert env["warnings"] == ["sentiment older than 60s"]
    assert env["errors"] == [{"code": "STALE_MODEL", "message": "degraded"}]


def test_apienvelope_typed_roundtrip() -> None:
    model = ApiEnvelope[dict](
        as_of=datetime.now(timezone.utc),
        source="typed",
        warnings=[],
        errors=[],
        data={"k": 1},
    )
    blob = model.model_dump(mode="json", exclude_none=True)
    assert blob["source"] == "typed"
    assert blob["data"] == {"k": 1}
    # Reconstruct via the model and assert it round-trips.
    rebuilt = ApiEnvelope[dict].model_validate(blob)
    assert rebuilt.source == "typed"
    assert rebuilt.data == {"k": 1}
