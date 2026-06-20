"""Tests for ``GET /api/v1/overview`` — Command Center aggregates.

Verifies the route aggregates action counts, the top actionable ideas,
and per-stage latency from the trade-ideas snapshot, leaves portfolio
metrics null (no persisted portfolio yet), and degrades rather than 500
when the ideas source is unavailable. No DB/MLflow/broker needed: a
snapshot is seeded via ``WANG_TRADE_IDEAS_PATH``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def _idea(symbol, action, target_weight, latency):
    return {
        "symbol": symbol,
        "action": action,
        "target_weight": target_weight,
        "target_notional": target_weight * 1_000_000,
        "estimated_quantity": None,
        "latest_price": 100.0,
        "latest_bar_at": None,
        "bars_loaded": 500,
        "feature_rows": 480,
        "signal_count": 2,
        "top_signal_family": "ts_momentum",
        "top_signal_side": 1,
        "top_signal_confidence": 0.7,
        "avg_signal_confidence": 0.6,
        "meta_probability": 0.66,
        "calibrated_probability": 0.61,
        "bet_size": target_weight,
        "strategy": "ts_momentum",
        "reason": "test idea",
        "stage_latency_seconds": latency,
        "errors": [],
    }


def _seed_snapshot(tmp_path, monkeypatch, ideas, totals):
    path = tmp_path / "trade_ideas.json"
    report = {"idea_count": len(ideas), "totals": totals, "ideas": ideas}
    payload = {"as_of": datetime.now(timezone.utc).isoformat(), "report": report}
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(path))


def test_overview_aggregates_from_snapshot(client, tmp_path, monkeypatch):
    ideas = [
        _idea("NVDA", "BUY", 0.092, {"data_fetch": 0.01, "meta_inference": 0.02}),
        _idea("AAPL", "BUY", 0.020, {"data_fetch": 0.01, "meta_inference": 0.03}),
        _idea("TSLA", "SELL", -0.052, {"data_fetch": 0.02}),
        _idea("AVAX", "WATCH", 0.0, {"data_fetch": 0.01}),
    ]
    totals = {
        "buy": 2, "sell": 1, "watch": 1,
        "model_required": 0, "no_data": 0, "error": 0,
        "gross_target_weight": 0.164, "net_target_weight": 0.06,
    }
    _seed_snapshot(tmp_path, monkeypatch, ideas, totals)

    resp = client.get("/api/v1/overview")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]

    assert data["actionCounts"] == {
        "buy": 2, "sell": 1, "watch": 1,
        "modelRequired": 0, "noData": 0,
    }

    # Top actionable = BUY/SELL only (WATCH excluded), sorted by |weight| desc.
    symbols = [i["symbol"] for i in data["topActionable"]]
    assert symbols == ["NVDA", "TSLA", "AAPL"]
    assert len(data["topActionable"]) <= 5

    # Engine pulse: per-stage latency summed across every idea.
    pulse = {s["stage"]: s["seconds"] for s in data["enginePulse"]["stages"]}
    assert round(pulse["data_fetch"], 6) == 0.05
    assert round(pulse["meta_inference"], 6) == 0.05

    # Portfolio metrics stay unavailable (no persisted portfolio).
    assert data.get("nav") is None
    assert any("no persisted portfolio" in w for w in body["warnings"])


def test_overview_caps_top_actionable_at_five(client, tmp_path, monkeypatch):
    ideas = [_idea(f"SYM{i}", "BUY", 0.10 - i * 0.01, {"data_fetch": 0.01}) for i in range(8)]
    totals = {
        "buy": 8, "sell": 0, "watch": 0,
        "model_required": 0, "no_data": 0, "error": 0,
        "gross_target_weight": 0.0, "net_target_weight": 0.0,
    }
    _seed_snapshot(tmp_path, monkeypatch, ideas, totals)

    resp = client.get("/api/v1/overview")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data["topActionable"]) == 5
    # Highest absolute weight first.
    assert data["topActionable"][0]["symbol"] == "SYM0"


def test_overview_degrades_when_ideas_unavailable(client, monkeypatch):
    # Force the read-only snapshot read to raise (e.g. an unreadable file).
    # The route must still 200 and degrade — patch the method the route
    # ACTUALLY calls (read_snapshot), not the removed regenerate path.
    from src.web.services import trade_ideas_service as svc

    def _boom(self, **kwargs):
        raise RuntimeError("snapshot read failed")

    monkeypatch.setattr(svc.TradeIdeasService, "read_snapshot", _boom)

    resp = client.get("/api/v1/overview")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    assert data["actionCounts"]["buy"] == 0
    assert data["topActionable"] == []
    assert data["enginePulse"]["stages"] == []
    assert any("aggregation unavailable" in w for w in body["warnings"])
