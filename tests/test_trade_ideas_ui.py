from __future__ import annotations

import json

import pandas as pd

from src.ui.trade_ideas import TradeIdeaReport, _current_signal_slice, _idea_from_artifacts


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 14:30", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [98.0, 99.0, 100.0],
            "high": [99.0, 100.0, 102.0],
            "low": [97.0, 98.5, 99.0],
            "close": [98.5, 100.0, 101.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=idx,
    )


def test_trade_idea_artifacts_render_buy_target():
    bars = _bars()
    ts = bars.index[-1]
    signals = pd.DataFrame(
        {
            "timestamp": [ts],
            "symbol": ["AAPL"],
            "family": ["ts_momentum"],
            "side": [1],
            "confidence": [0.82],
        }
    )
    meta = signals.assign(meta_prob=0.74, calibrated_prob=0.71)
    bets = pd.DataFrame(
        {
            "timestamp": [ts],
            "symbol": ["AAPL"],
            "family": ["ts_momentum"],
            "final_size": [0.05],
        }
    )
    target = pd.DataFrame(
        {"symbol": ["AAPL"], "target_weight": [0.05], "strategy": ["ts_momentum"]}
    )

    idea = _idea_from_artifacts(
        symbol="AAPL",
        bars=bars,
        features=pd.DataFrame({"f0": [1.0]}, index=[ts]),
        signals=signals,
        meta=meta,
        bets=bets,
        target=target,
        nav=100_000.0,
        min_abs_weight=0.0025,
        model_source="mlflow_production",
        stage_latency_seconds={"data_fetch": 0.01},
        errors=[],
    )

    assert idea.action == "BUY"
    assert idea.target_weight == 0.05
    assert idea.target_notional == 5000.0
    assert idea.estimated_quantity == 5000.0 / 101.0
    assert idea.top_signal_family == "ts_momentum"
    assert idea.top_signal_confidence == 0.82
    assert idea.meta_probability == 0.74


def test_trade_idea_marks_model_required_when_signals_exist_without_model():
    bars = _bars()
    ts = bars.index[-1]
    signals = pd.DataFrame(
        {
            "timestamp": [ts],
            "symbol": ["MSFT"],
            "family": ["mean_reversion"],
            "side": [-1],
            "confidence": [0.66],
        }
    )

    idea = _idea_from_artifacts(
        symbol="MSFT",
        bars=bars,
        features=pd.DataFrame({"f0": [1.0]}, index=[ts]),
        signals=signals,
        meta=pd.DataFrame(),
        bets=pd.DataFrame(),
        target=pd.DataFrame(),
        nav=100_000.0,
        min_abs_weight=0.0025,
        model_source="none",
        stage_latency_seconds={},
        errors=[],
    )

    assert idea.action == "MODEL_REQUIRED"
    assert "Production meta model is unavailable" in idea.reason
    assert idea.target_weight == 0.0


def test_trade_idea_report_is_json_serializable_and_order_safe():
    bars = _bars()
    ts = bars.index[-1]
    idea = _idea_from_artifacts(
        symbol="AAPL",
        bars=bars,
        features=pd.DataFrame({"f0": [1.0]}, index=[ts]),
        signals=pd.DataFrame(),
        meta=pd.DataFrame(),
        bets=pd.DataFrame(),
        target=pd.DataFrame(),
        nav=100_000.0,
        min_abs_weight=0.0025,
        model_source="mlflow_production",
        stage_latency_seconds={"data_fetch": 0.01, "feature_compute": 0.02},
        errors=[],
    )
    report = TradeIdeaReport(
        generated_at="2026-01-01T15:00:00+00:00",
        mode="paper_production_readonly",
        config_path="config/live_trading.yaml",
        symbols=["AAPL"],
        bar_type="tib",
        nav=100_000.0,
        model_source="mlflow_production",
        live_orders_sent=0,
        allow_confidence_fallback=False,
        ideas=[idea],
        stage_latency_seconds={"data_fetch": 0.01, "feature_compute": 0.02},
    )

    payload = report.to_dict()
    json.dumps(payload)
    assert payload["live_orders_sent"] == 0
    assert payload["totals"]["watch"] == 1


def test_current_signal_slice_keeps_only_latest_timestamp():
    idx = pd.date_range("2026-01-01 14:30", periods=3, freq="1h", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": [idx[0], idx[1], idx[1]],
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "family": ["old", "fresh_a", "fresh_b"],
            "side": [1, 1, -1],
            "confidence": [0.9, 0.6, 0.7],
        }
    )

    current = _current_signal_slice(signals)

    assert set(current["family"]) == {"fresh_a", "fresh_b"}
