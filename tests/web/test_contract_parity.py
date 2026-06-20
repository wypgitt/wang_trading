"""Frozen-contract tests (aperture_backend_design §10.4 #3).

Two boundaries the design locks:

1. The ``ApiEnvelope`` field set is identical across ``envelope.py`` and the
   client ``envelope.ts`` (and the iOS model). This test pins the Python side
   to the frozen set; if a field is added/removed here, update all three.
2. The tmpfs snapshot carries a ``schema_version`` the engine publisher and the
   BFF reader agree on, and the reader parses a versioned snapshot.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.execution.trade_idea_publisher import SNAPSHOT_SCHEMA_VERSION as PUB_VERSION
from src.web.envelope import ApiEnvelope, ApiError
from src.web.services.trade_ideas_service import (
    SNAPSHOT_SCHEMA_VERSION as BFF_VERSION,
    TmpfsTradeIdeasCache,
)

# The frozen ApiEnvelope contract — mirrors envelope.ts (and the iOS model).
# Keep this in lockstep with apps/aperture-web/src/data/envelope.ts.
_FROZEN_ENVELOPE_FIELDS = {
    "data",
    "as_of",
    "source",
    "staleness_seconds",
    "source_freshness",
    "model_version",
    "regime",
    "warnings",
    "errors",
    "request_id",
}
_FROZEN_ERROR_FIELDS = {"code", "message", "field"}


def test_envelope_field_set_is_frozen() -> None:
    assert set(ApiEnvelope.model_fields.keys()) == _FROZEN_ENVELOPE_FIELDS


def test_envelope_error_field_set_is_frozen() -> None:
    assert set(ApiError.model_fields.keys()) == _FROZEN_ERROR_FIELDS


def test_publisher_and_bff_agree_on_snapshot_schema_version() -> None:
    assert PUB_VERSION == BFF_VERSION


def _payload(*, version=PUB_VERSION, with_version=True):
    report = {
        "idea_count": 0,
        "totals": {
            "buy": 0, "sell": 0, "watch": 0,
            "model_required": 0, "no_data": 0, "error": 0,
            "gross_target_weight": 0.0, "net_target_weight": 0.0,
        },
        "ideas": [],
    }
    payload = {"as_of": datetime.now(timezone.utc).isoformat(), "report": report}
    if with_version:
        payload["schema_version"] = version
    return payload


def test_bff_reads_a_versioned_snapshot(tmp_path) -> None:
    path = tmp_path / "trade_ideas.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    result = TmpfsTradeIdeasCache(path=str(path)).read()
    assert result is not None
    response, staleness = result
    assert response.idea_count == 0
    assert staleness >= 0.0


def test_bff_tolerates_snapshot_without_version(tmp_path) -> None:
    # Snapshots predating the field still parse (version None ⇒ no warning).
    path = tmp_path / "trade_ideas.json"
    path.write_text(json.dumps(_payload(with_version=False)), encoding="utf-8")
    assert TmpfsTradeIdeasCache(path=str(path)).read() is not None
