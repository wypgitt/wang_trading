"""Phase-2 post-cycle publish hook.

``build_report_from_cycle`` assembles a TradeIdeaReport from the live cycle's
already-computed PANEL artifacts (no shadow-pipeline re-run), reusing the same
per-symbol assembly the read-only rehearsal uses. The hook is then proven
end-to-end into the BFF reader via ``publish_report``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.execution.trade_idea_publisher import TradeIdeaPublisher
from src.ui.trade_ideas import build_report_from_cycle


class ModelMetaPipeline:  # name matters: _model_source -> "mlflow_production"
    pass


class _Bar:
    def __init__(self, close, ts):
        self.close = close
        self.timestamp = ts


def _panels():
    ts = datetime(2026, 6, 20, 14, 0, tzinfo=timezone.utc)
    bars = {"NVDA": _Bar(120.0, ts), "TSLA": _Bar(250.0, ts)}
    signals = pd.DataFrame([
        {"symbol": "NVDA", "family": "ts_momentum", "side": 1, "confidence": 0.74, "timestamp": ts},
        {"symbol": "TSLA", "family": "mean_reversion", "side": -1, "confidence": 0.61, "timestamp": ts},
    ])
    meta = pd.DataFrame([
        {"symbol": "NVDA", "meta_prob": 0.72, "calibrated_prob": 0.69},
        {"symbol": "TSLA", "meta_prob": 0.66, "calibrated_prob": 0.63},
    ])
    bets = pd.DataFrame([
        {"symbol": "NVDA", "final_size": 0.09, "family": "ts_momentum"},
        {"symbol": "TSLA", "final_size": -0.05, "family": "mean_reversion"},
    ])
    target = pd.DataFrame([
        {"symbol": "NVDA", "target_weight": 0.09, "strategy": "ts_momentum"},
        {"symbol": "TSLA", "target_weight": -0.05, "strategy": "mean_reversion"},
    ])
    return bars, signals, meta, bets, target


def _report():
    bars, signals, meta, bets, target = _panels()
    return build_report_from_cycle(
        symbols=["NVDA", "TSLA"],
        bars=bars,
        features=pd.DataFrame({"x": [1, 2, 3]}),
        signals=signals,
        meta=meta,
        bets=bets,
        target=target,
        nav=2_000_000.0,
        bar_type="tib",
        meta_pipeline=ModelMetaPipeline(),
    )


def test_build_report_from_cycle_slices_panel_per_symbol():
    d = _report().to_dict()
    by = {i["symbol"]: i for i in d["ideas"]}
    assert set(by) == {"NVDA", "TSLA"}
    # Per-symbol slicing: each idea gets its OWN target/meta/signal/price.
    assert by["NVDA"]["target_weight"] == 0.09
    assert by["NVDA"]["meta_probability"] == 0.72
    assert by["NVDA"]["calibrated_probability"] == 0.69
    assert by["NVDA"]["latest_price"] == 120.0
    assert by["NVDA"]["action"] == "BUY"
    assert by["TSLA"]["target_weight"] == -0.05
    assert by["TSLA"]["top_signal_family"] == "mean_reversion"
    assert by["TSLA"]["action"] == "SELL"
    # Produced by the LIVE cycle, not the rehearsal.
    assert d["mode"] == "paper_production_live"
    assert d["model_source"] == "mlflow_production"
    assert d["totals"]["buy"] == 1 and d["totals"]["sell"] == 1


def test_hook_report_publishes_and_bff_reader_parses(tmp_path):
    # The full phase-2 chain: cycle artifacts -> build_report_from_cycle ->
    # publish_report (the sink) -> the BFF's snapshot reader.
    from src.web.services.trade_ideas_service import TmpfsTradeIdeasCache

    path = tmp_path / "trade_ideas.json"
    TradeIdeaPublisher(output_path=path).publish_report(_report())

    result = TmpfsTradeIdeasCache(path=str(path)).read()
    assert result is not None
    resp, staleness = result
    assert resp.idea_count == 2
    assert {i.symbol for i in resp.ideas} == {"NVDA", "TSLA"}
    assert staleness >= 0.0


def test_build_report_handles_empty_cycle():
    # No artifacts this tick (e.g. cold start) -> an honest empty report, no crash.
    report = build_report_from_cycle(
        symbols=[], bars={}, features=None, signals=None, meta=None,
        bets=None, target=None, nav=1_000_000.0, bar_type="tib",
    )
    assert report.to_dict()["idea_count"] == 0
