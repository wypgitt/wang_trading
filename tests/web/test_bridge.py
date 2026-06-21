"""End-to-end engine->BFF bridge.

The publisher (the bridge) writes the tmpfs snapshot; the BFF reads + serves it.
This locks the two halves together on one file + one shape, with NO infra — the
exact contract that, in production, the supervised publisher daemon keeps fresh.
"""

from __future__ import annotations

from src.execution.trade_idea_publisher import TradeIdeaPublisher


def _idea(sym, action, w):
    return {
        "symbol": sym, "action": action, "target_weight": w, "target_notional": w * 1_000_000,
        "strategy": "ts_momentum", "top_signal_family": "ts_momentum",
        "meta_probability": 0.7, "calibrated_probability": 0.66,
        "stage_latency_seconds": {"data_fetch": 0.01, "meta_inference": 0.02}, "errors": [],
    }


def _report(ideas):
    return {
        "idea_count": len(ideas),
        "totals": {
            "buy": sum(i["action"] == "BUY" for i in ideas),
            "sell": sum(i["action"] == "SELL" for i in ideas),
            "watch": 0, "model_required": 0, "no_data": 0, "error": 0,
            "gross_target_weight": 0.0, "net_target_weight": 0.0,
        },
        "ideas": ideas,
    }


def test_publisher_writes_and_bff_serves(client, tmp_path, monkeypatch):
    path = tmp_path / "trade_ideas.json"
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(path))

    # The bridge: publish a report (the publish_report write path the post-cycle
    # hook uses — no pipeline run needed for the test).
    TradeIdeaPublisher(output_path=path).publish_report(
        _report([_idea("NVDA", "BUY", 0.09), _idea("TSLA", "SELL", -0.05)])
    )

    # The BFF serves exactly what was published.
    overview = client.get("/api/v1/overview").json()["data"]
    assert overview["actionCounts"]["buy"] == 1
    assert overview["actionCounts"]["sell"] == 1
    assert {i["symbol"] for i in overview["topActionable"]} == {"NVDA", "TSLA"}

    ideas = client.get("/api/v1/trade-ideas").json()["data"]["ideas"]
    assert {i["symbol"] for i in ideas} == {"NVDA", "TSLA"}

    # A re-publish is picked up (the daemon's steady state) — staleness resets.
    TradeIdeaPublisher(output_path=path).publish_report(_report([_idea("BTC", "BUY", 0.04)]))
    again = client.get("/api/v1/trade-ideas").json()["data"]["ideas"]
    assert {i["symbol"] for i in again} == {"BTC"}
