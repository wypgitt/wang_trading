"""Unit tests for :mod:`src.execution.trade_idea_publisher`.

These tests monkeypatch the live report generator so they do not boot
the pipeline; they exercise file shape, atomic-rename behaviour, and
the JSON contract the BFF's ``TmpfsTradeIdeasCache`` relies on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.execution import trade_idea_publisher as publisher_mod
from src.execution.trade_idea_publisher import (
    SNAPSHOT_SCHEMA_VERSION,
    TradeIdeaPublisher,
    default_output_path,
)
from src.ui import trade_ideas as trade_ideas_module


# ── Fake report ───────────────────────────────────────────────────────


@dataclass
class _FakeIdea:
    symbol: str
    action: str = "WATCH"
    target_weight: float = 0.0
    target_notional: float = 0.0
    estimated_quantity: float | None = None
    latest_price: float | None = None
    latest_bar_at: str | None = None
    bars_loaded: int = 0
    feature_rows: int = 0
    signal_count: int = 0
    top_signal_family: str | None = None
    top_signal_side: int | None = None
    top_signal_confidence: float | None = None
    avg_signal_confidence: float | None = None
    meta_probability: float | None = None
    calibrated_probability: float | None = None
    bet_size: float | None = None
    strategy: str | None = None
    reason: str = ""
    stage_latency_seconds: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class _FakeReport:
    ideas: list[_FakeIdea]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": "2026-05-15T15:00:00+00:00",
            "mode": "paper_production_readonly",
            "config_path": None,
            "symbols": [idea.symbol for idea in self.ideas],
            "bar_type": "1h",
            "nav": 100_000.0,
            "model_source": "mlflow_production",
            "live_orders_sent": 0,
            "allow_confidence_fallback": False,
            "idea_count": len(self.ideas),
            "totals": {
                "buy": sum(1 for i in self.ideas if i.action == "BUY"),
                "sell": sum(1 for i in self.ideas if i.action == "SELL"),
                "watch": sum(1 for i in self.ideas if i.action == "WATCH"),
                "model_required": 0,
                "error": 0,
                "gross_target_weight": sum(abs(i.target_weight) for i in self.ideas),
                "net_target_weight": sum(i.target_weight for i in self.ideas),
            },
            "warnings": [],
            "errors": [],
            "stage_latency_seconds": {},
            "ideas": [
                {
                    "symbol": idea.symbol,
                    "action": idea.action,
                    "target_weight": idea.target_weight,
                    "target_notional": idea.target_notional,
                    "estimated_quantity": idea.estimated_quantity,
                    "latest_price": idea.latest_price,
                    "latest_bar_at": idea.latest_bar_at,
                    "bars_loaded": idea.bars_loaded,
                    "feature_rows": idea.feature_rows,
                    "signal_count": idea.signal_count,
                    "top_signal_family": idea.top_signal_family,
                    "top_signal_side": idea.top_signal_side,
                    "top_signal_confidence": idea.top_signal_confidence,
                    "avg_signal_confidence": idea.avg_signal_confidence,
                    "meta_probability": idea.meta_probability,
                    "calibrated_probability": idea.calibrated_probability,
                    "bet_size": idea.bet_size,
                    "strategy": idea.strategy,
                    "reason": idea.reason,
                    "stage_latency_seconds": dict(idea.stage_latency_seconds),
                    "errors": list(idea.errors),
                }
                for idea in self.ideas
            ],
        }


def _install_fake_generator(monkeypatch, report: _FakeReport) -> dict[str, Any]:
    """Patch ``generate_trade_idea_report_sync`` and capture its kwargs."""

    captured: dict[str, Any] = {}

    def _fake_generator(**kwargs: Any) -> _FakeReport:
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(
        trade_ideas_module,
        "generate_trade_idea_report_sync",
        _fake_generator,
    )
    return captured


# ── default_output_path ───────────────────────────────────────────────


def test_default_output_path_honours_env_var(tmp_path, monkeypatch):
    override = tmp_path / "custom" / "ideas.json"
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(override))
    result = default_output_path()
    assert result == override
    # The publisher contract guarantees the parent dir exists.
    assert result.parent.is_dir()


def test_default_output_path_uses_xdg_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("WANG_TRADE_IDEAS_PATH", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    result = default_output_path()
    assert result == tmp_path / "wang" / "trade_ideas.json"
    assert result.parent.is_dir()


# ── publish_once ──────────────────────────────────────────────────────


def test_publish_once_writes_expected_json_shape(tmp_path, monkeypatch):
    report = _FakeReport(
        ideas=[
            _FakeIdea(symbol="AAPL", action="BUY", target_weight=0.04, target_notional=4_000.0),
            _FakeIdea(symbol="MSFT"),
        ],
    )
    captured = _install_fake_generator(monkeypatch, report)

    output = tmp_path / "trade_ideas.json"
    publisher = TradeIdeaPublisher(
        symbols=["AAPL", "MSFT"],
        bar_limit=250,
        min_abs_weight=0.005,
        allow_confidence_fallback=True,
        output_path=output,
    )

    returned = publisher.publish_once()
    assert returned is report
    assert captured["kwargs"] == {
        "config_path": None,
        "symbols": ["AAPL", "MSFT"],
        "bar_limit": 250,
        "min_abs_weight": 0.005,
        "allow_confidence_fallback": True,
    }

    # File contract: {"schema_version": <int>, "as_of": <iso>, "report": <dict>}
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"schema_version", "as_of", "report"}
    assert payload["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert isinstance(payload["as_of"], str) and payload["as_of"]
    assert payload["report"]["idea_count"] == 2
    symbols_in_payload = [idea["symbol"] for idea in payload["report"]["ideas"]]
    assert symbols_in_payload == ["AAPL", "MSFT"]


def test_publish_once_is_atomic_temp_file_removed(tmp_path, monkeypatch):
    report = _FakeReport(ideas=[_FakeIdea(symbol="AAPL")])
    _install_fake_generator(monkeypatch, report)

    output = tmp_path / "trade_ideas.json"
    publisher = TradeIdeaPublisher(output_path=output)
    publisher.publish_once()

    # The tmp sibling is the only intermediate; after rename it must be gone.
    tmp_candidates = list(tmp_path.glob("trade_ideas.json.tmp"))
    assert tmp_candidates == []
    # And the rename target exists.
    assert output.exists()


def test_publish_once_round_trips_through_json(tmp_path, monkeypatch):
    """The file must parse back into the shape the BFF expects."""

    report = _FakeReport(ideas=[_FakeIdea(symbol="AAPL", action="BUY", target_weight=0.03)])
    _install_fake_generator(monkeypatch, report)

    output = tmp_path / "trade_ideas.json"
    TradeIdeaPublisher(output_path=output).publish_once()

    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    assert "as_of" in parsed
    assert "report" in parsed
    report_dict = parsed["report"]
    assert isinstance(report_dict, dict)
    assert report_dict["ideas"][0]["symbol"] == "AAPL"
    assert report_dict["ideas"][0]["action"] == "BUY"


def test_publish_once_overwrites_previous_snapshot(tmp_path, monkeypatch):
    output = tmp_path / "trade_ideas.json"

    first_report = _FakeReport(ideas=[_FakeIdea(symbol="AAPL")])
    _install_fake_generator(monkeypatch, first_report)
    TradeIdeaPublisher(output_path=output).publish_once()
    first_payload = json.loads(output.read_text(encoding="utf-8"))

    second_report = _FakeReport(ideas=[_FakeIdea(symbol="MSFT", action="SELL", target_weight=-0.02)])
    _install_fake_generator(monkeypatch, second_report)
    TradeIdeaPublisher(output_path=output).publish_once()
    second_payload = json.loads(output.read_text(encoding="utf-8"))

    assert first_payload["report"]["ideas"][0]["symbol"] == "AAPL"
    assert second_payload["report"]["ideas"][0]["symbol"] == "MSFT"
    assert second_payload["as_of"] >= first_payload["as_of"]


def test_publisher_creates_parent_directory(tmp_path, monkeypatch):
    nested = tmp_path / "deeply" / "nested" / "wang" / "trade_ideas.json"
    _install_fake_generator(monkeypatch, _FakeReport(ideas=[_FakeIdea(symbol="AAPL")]))

    publisher = TradeIdeaPublisher(output_path=nested)
    assert nested.parent.is_dir()
    publisher.publish_once()
    assert nested.exists()


# ── CLI smoke ─────────────────────────────────────────────────────────


def test_cli_once_writes_file(tmp_path, monkeypatch):
    _install_fake_generator(monkeypatch, _FakeReport(ideas=[_FakeIdea(symbol="AAPL")]))
    output = tmp_path / "ideas.json"

    rc = publisher_mod.main(["--once", "--output", str(output), "--symbols", "AAPL"])
    assert rc == 0
    assert output.exists()
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["report"]["ideas"][0]["symbol"] == "AAPL"
