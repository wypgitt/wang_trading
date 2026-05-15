"""DTO round-trip tests (web_app_design_v2 §27, api_contracts_v2)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.web.dtos import (
    AffectedIdea,
    ScenarioResult,
    SizingLayer,
    SizingWaterfall,
    SuggestedHedge,
    TradeIdea,
    TradeIdeasResponse,
    TradeIdeasTotals,
)


def test_trade_idea_minimal_roundtrip() -> None:
    raw = {
        "symbol": "BTC-USD",
        "action": "BUY",
        "target_weight": 0.05,
        "target_notional": 50_000.0,
        "latest_price": 64_321.0,
        "latest_bar_at": "2026-05-15T18:45:00+00:00",
        "bar_type": "volume",
        "bars_loaded": 1024,
        "feature_rows": 1024,
        "signal_count": 6,
        "top_signal_family": "ts_momentum",
        "top_signal_side": 1,
        "top_signal_confidence": 0.72,
        "avg_signal_confidence": 0.61,
        "meta_probability": 0.68,
        "calibrated_probability": 0.65,
        "regime_fit_score": 0.81,
        "bet_size": 0.05,
        "sizing_constraints_applied": ["afml_cap", "vol_target"],
        "strategy": "ts_momentum_v3",
        "reason": "TS momentum aligned with trending_up regime",
        "expected_cost_bps": 8.2,
    }

    idea = TradeIdea.model_validate(raw)
    assert idea.symbol == "BTC-USD"
    assert idea.action == "BUY"
    assert idea.top_signal_side == 1
    assert idea.bars_loaded == 1024

    # Round-trip through JSON and back.
    blob = idea.model_dump(mode="json", exclude_none=True)
    rebuilt = TradeIdea.model_validate(blob)
    assert rebuilt == idea


def test_trade_ideas_response_aggregation() -> None:
    idea = TradeIdea(
        symbol="ETH-USD",
        action="WATCH",
        target_weight=0.0,
        target_notional=0.0,
    )
    totals = TradeIdeasTotals(buy=0, sell=0, watch=1)
    resp = TradeIdeasResponse(idea_count=1, totals=totals, ideas=[idea])
    blob = resp.model_dump(mode="json", exclude_none=True)
    assert blob["idea_count"] == 1
    assert blob["totals"]["watch"] == 1
    # Stage latency / errors default to empty containers.
    assert blob["ideas"][0]["stage_latency_seconds"] == {}
    assert blob["ideas"][0]["errors"] == []


def test_sizing_waterfall_roundtrip() -> None:
    waterfall = SizingWaterfall(
        layers=[
            SizingLayer(name="afml", value=0.12, capped=False),
            SizingLayer(name="kelly", value=0.09, capped=False),
            SizingLayer(name="vol", value=0.07, capped=True, cap_reason="vol_target"),
            SizingLayer(name="atr", value=0.05, capped=True, cap_reason="atr_cap"),
            SizingLayer(name="final", value=0.05, capped=False),
        ],
        constraints_applied=["vol_target", "atr_cap"],
        side=1,
        final=0.05,
    )
    blob = waterfall.model_dump(mode="json", exclude_none=True)
    assert blob["side"] == 1
    assert blob["final"] == 0.05
    assert blob["layers"][0]["name"] == "afml"
    assert blob["layers"][2]["cap_reason"] == "vol_target"
    rebuilt = SizingWaterfall.model_validate(blob)
    assert rebuilt == waterfall


def test_scenario_result_roundtrip() -> None:
    result = ScenarioResult(
        pnl_impact_usd=-12_500.0,
        drawdown_impact=0.024,
        factor_exposure_deltas={"momentum": -0.18, "value": 0.05},
        breaker_headroom={"daily_loss": 0.42, "var_99": 0.31},
        affected_ideas=[
            AffectedIdea(symbol="BTC-USD", old_target=0.05, new_target=0.02, flipped=False),
            AffectedIdea(symbol="ETH-USD", old_target=0.03, new_target=-0.01, flipped=True),
        ],
        suggested_hedges=[SuggestedHedge(long="BTC-USD", short="ETH-USD", ratio=1.4)],
        warnings=["factor model stale"],
    )
    blob = result.model_dump(mode="json", exclude_none=True)
    assert blob["pnl_impact_usd"] == -12_500.0
    assert len(blob["affected_ideas"]) == 2
    assert blob["affected_ideas"][1]["flipped"] is True
    assert blob["suggested_hedges"][0]["ratio"] == 1.4
    rebuilt = ScenarioResult.model_validate(blob)
    assert rebuilt == result


def test_trade_idea_omits_none_when_dumped() -> None:
    idea = TradeIdea(
        symbol="SOL-USD",
        action="NO_DATA",
        target_weight=0.0,
        target_notional=0.0,
    )
    blob = idea.model_dump(mode="json", exclude_none=True)
    assert "meta_probability" not in blob
    assert "regime" not in blob
    assert "top_shap_feature" not in blob
    # Required fields still present.
    assert blob["symbol"] == "SOL-USD"
    assert blob["action"] == "NO_DATA"
    # Lifted defaults present (empty containers).
    assert blob["stage_latency_seconds"] == {}


def test_trade_idea_now_field_validates_timestamp() -> None:
    """Confirm DTO accepts an ISO-8601 timestamp string for datetime fields."""

    now = datetime.now(timezone.utc).isoformat()
    idea = TradeIdea(
        symbol="BTC-USD",
        action="BUY",
        target_weight=0.01,
        target_notional=1000.0,
        latest_bar_at=now,  # type: ignore[arg-type] — pydantic parses string
    )
    assert idea.latest_bar_at is not None
    assert idea.latest_bar_at.tzinfo is not None
