"""Tests for ``GET /api/v1/trade-ideas`` and ``/{symbol}``.

The highest-traffic surface and the core of the read-only invariant: it reads
the engine's tmpfs snapshot and NEVER regenerates. Verifies the enveloped list
shape, the symbols filter, the degrade-but-ship stale warning, the honest 503
on a missing snapshot, and the detail resolve-from-snapshot behavior. No infra:
the snapshot is seeded via ``WANG_TRADE_IDEAS_PATH``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


def _idea(symbol, action="BUY", weight=0.05):
    return {
        "symbol": symbol,
        "action": action,
        "target_weight": weight,
        "target_notional": weight * 1_000_000,
        "reason": "test idea",
        "stage_latency_seconds": {"data_fetch": 0.01},
        "errors": [],
    }


def _seed(tmp_path, monkeypatch, ideas, *, age_seconds=5.0):
    as_of = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    report = {
        "idea_count": len(ideas),
        "totals": {
            "buy": sum(i["action"] == "BUY" for i in ideas),
            "sell": sum(i["action"] == "SELL" for i in ideas),
            "watch": 0, "model_required": 0, "no_data": 0, "error": 0,
            "gross_target_weight": 0.0, "net_target_weight": 0.0,
        },
        "ideas": ideas,
    }
    path = tmp_path / "trade_ideas.json"
    path.write_text(
        json.dumps({"schema_version": 1, "as_of": as_of.isoformat(), "report": report}),
        encoding="utf-8",
    )
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(path))


def test_list_returns_enveloped_camelcase(client, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_idea("NVDA"), _idea("TSLA", "SELL", -0.03)])
    body = client.get("/api/v1/trade-ideas").json()
    assert body["source"] == "trade_ideas_service"
    data = body["data"]
    # Rich response shape, camelCased.
    assert data["ideaCount"] == 2
    assert {i["symbol"] for i in data["ideas"]} == {"NVDA", "TSLA"}
    assert data["totals"]["buy"] == 1
    # Envelope metadata present + staleness a real number.
    assert isinstance(body["staleness_seconds"], (int, float))


def test_list_symbols_filter(client, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_idea("NVDA"), _idea("TSLA", "SELL", -0.03)])
    data = client.get("/api/v1/trade-ideas?symbols=NVDA").json()["data"]
    assert [i["symbol"] for i in data["ideas"]] == ["NVDA"]
    assert data["ideaCount"] == 1


def test_list_stale_snapshot_ships_with_warning(client, tmp_path, monkeypatch):
    # Degrade but ship: a >90s-old snapshot is served (not 503) + flagged.
    _seed(tmp_path, monkeypatch, [_idea("NVDA")], age_seconds=180.0)
    resp = client.get("/api/v1/trade-ideas")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["ideaCount"] == 1
    assert body["staleness_seconds"] > 90
    assert any("stale" in w for w in body["warnings"])


def test_list_missing_snapshot_is_503(client, tmp_path, monkeypatch):
    # Read-only: no regenerate. A missing snapshot is an honest, enveloped 503.
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(tmp_path / "does_not_exist.json"))
    resp = client.get("/api/v1/trade-ideas")
    assert resp.status_code == 503
    body = resp.json()
    assert body["data"] is None
    assert body["errors"][0]["code"] == "SNAPSHOT_UNAVAILABLE"
    # No leaked exception text, enveloped with a request id.
    assert body["request_id"].startswith("req_")


def test_detail_resolves_idea_from_snapshot(client, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_idea("NVDA"), _idea("TSLA", "SELL", -0.03)])
    data = client.get("/api/v1/trade-ideas/NVDA").json()["data"]
    assert data["symbol"] == "NVDA"
    assert data["action"] == "BUY"


def test_detail_unknown_symbol_is_200_null(client, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_idea("NVDA")])
    resp = client.get("/api/v1/trade-ideas/ZZZZ")
    assert resp.status_code == 200
    assert resp.json()["data"] is None  # present snapshot, no match → honest null


def test_detail_missing_snapshot_is_200_null_with_warning(client, tmp_path, monkeypatch):
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(tmp_path / "nope.json"))
    resp = client.get("/api/v1/trade-ideas/NVDA")
    assert resp.status_code == 200  # detail never 500s/503s
    body = resp.json()
    assert body["data"] is None
    assert any("unavailable" in w for w in body["warnings"])
