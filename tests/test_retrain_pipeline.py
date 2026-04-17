"""Tests for RetrainPipeline end-to-end wiring (C2)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.retrain_pipeline import RetrainPipeline, RetrainRun


# ── Fixture factory ─────────────────────────────────────────────────────

def _pipeline(**overrides) -> RetrainPipeline:
    meta = MagicMock()
    X = pd.DataFrame({"f1": np.arange(10.0), "f2": np.arange(10.0) * 2})
    y = pd.Series([1] * 10)
    sw = pd.Series([1.0] * 10)
    meta.prepare_training_data = MagicMock(return_value=(X, y, sw))

    assembler = MagicMock()
    assembler.compute = MagicMock(return_value=pd.DataFrame({"f1": [1]}))

    signals = MagicMock()
    signals.generate = MagicMock(return_value=pd.DataFrame({"family": ["trend"]}))

    gate = MagicMock()
    gate.validate = MagicMock(return_value={"passed": True, "cpcv": 0.6,
                                            "dsr": 1.2, "pbo": 0.1})

    registry = MagicMock()
    registry.get_production_model = MagicMock(return_value=None)
    registry.log_training_run = MagicMock(return_value="new-run-123")
    registry.promote_model = MagicMock()

    alert = MagicMock()
    alert.send_alert = AsyncMock()

    defaults = {
        "meta_labeling_pipeline": meta,
        "feature_assembler": assembler,
        "signal_battery": signals,
        "gate": gate,
        "registry": registry,
        "cost_model": MagicMock(),
        "alert_manager": alert,
        "trainer": MagicMock(return_value=(MagicMock(name="new-model"), 0.65)),
        "scorer": MagicMock(return_value=0.50),
        "min_improvement_pct": 0.05,
    }
    defaults.update(overrides)
    return RetrainPipeline(**defaults)


# ── Happy path ─────────────────────────────────────────────────────────

class TestRunSuccess:
    def test_promotion_when_gate_passes(self):
        pipe = _pipeline()
        close = pd.Series(np.arange(100.0))
        bars = pd.DataFrame({"close": close})
        run = asyncio.run(pipe.run("AAPL", close, bars))
        assert isinstance(run, RetrainRun)
        assert run.promoted is True
        assert run.new_version == "new-run-123"
        assert run.cv_score == 0.65
        assert run.gate_results.get("passed") is True
        pipe.registry.promote_model.assert_called_once_with("new-run-123")

    def test_fills_metadata(self):
        pipe = _pipeline()
        run = asyncio.run(pipe.run("MSFT", pd.Series([100.0]), pd.DataFrame()))
        assert run.symbol == "MSFT"
        assert run.trigger == "scheduled"
        assert run.completed_at is not None
        assert run.training_rows == 10


# ── Rejection paths ────────────────────────────────────────────────────

class TestRejections:
    def test_rejects_when_new_model_insufficient_improvement(self):
        # Incumbent scores 0.60; new scores 0.61 → only 1.6% gain < 5% threshold
        pipe = _pipeline(
            trainer=MagicMock(return_value=(MagicMock(), 0.61)),
            scorer=MagicMock(return_value=0.60),
        )
        pipe.registry.get_production_model.return_value = {"run_id": "incumbent"}
        run = asyncio.run(pipe.run("AAPL", pd.Series([100.0]), pd.DataFrame()))
        assert run.promoted is False
        assert "insufficient_improvement" in run.rejection_reason
        pipe.registry.promote_model.assert_not_called()

    def test_rejects_when_gate_fails(self):
        pipe = _pipeline()
        pipe.gate.validate = MagicMock(return_value={
            "passed": False, "failing_gates": ["pbo"], "pbo": 0.9,
        })
        run = asyncio.run(pipe.run("AAPL", pd.Series([100.0]), pd.DataFrame()))
        assert run.promoted is False
        assert "gate_failed" in run.rejection_reason
        pipe.registry.promote_model.assert_not_called()

    def test_rejects_when_no_training_rows(self):
        pipe = _pipeline()
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)
        empty_w = pd.Series(dtype=float)
        pipe.meta_labeling_pipeline.prepare_training_data.return_value = (
            empty_X, empty_y, empty_w,
        )
        run = asyncio.run(pipe.run("AAPL", pd.Series([100.0]), pd.DataFrame()))
        assert run.promoted is False
        assert run.rejection_reason == "no_training_rows"

    def test_rejects_when_no_trainer(self):
        pipe = _pipeline(trainer=None)
        run = asyncio.run(pipe.run("AAPL", pd.Series([100.0]), pd.DataFrame()))
        assert run.promoted is False
        assert run.rejection_reason == "no_trainer_configured"


# ── Emergency retrain ─────────────────────────────────────────────────

class TestEmergencyRetrain:
    def test_emergency_bypasses_improvement_threshold(self):
        # New model underperforms, but emergency should still run the gate
        pipe = _pipeline(
            trainer=MagicMock(return_value=(MagicMock(), 0.10)),
            scorer=MagicMock(return_value=0.90),
        )
        pipe.registry.get_production_model.return_value = {"run_id": "incumbent"}
        run = asyncio.run(pipe.emergency_retrain(
            "AAPL", reason="drift", close=pd.Series([100.0]), bars=pd.DataFrame(),
        ))
        # With gate passing, emergency should promote despite underperformance
        assert run.trigger == "emergency"
        assert run.promoted is True
        assert run.gate_results.get("emergency_reason") == "drift"


# ── Batch path ────────────────────────────────────────────────────────

class TestRunAllSymbols:
    def test_per_symbol_failures_do_not_abort(self):
        pipe = _pipeline()

        def loader(symbol):
            if symbol == "BAD":
                raise RuntimeError("no data")
            return pd.Series([100.0]), pd.DataFrame()

        runs = asyncio.run(pipe.run_all_symbols(["AAPL", "BAD", "MSFT"], loader))
        assert len(runs) == 3
        bad = next(r for r in runs if r.symbol == "BAD")
        assert "data_loader_failed" in bad.rejection_reason
        good = [r for r in runs if r.symbol != "BAD"]
        assert all(r.promoted for r in good)


# ── Scheduler delegation ──────────────────────────────────────────────

class TestSchedulerDelegation:
    def test_scheduler_delegates_to_pipeline(self):
        from src.execution.retrain_scheduler import RetrainScheduler
        from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
        pipe = _pipeline()
        scheduler = RetrainScheduler(
            alert_manager=AlertManager(
                channel_map={s: [LogChannel()] for s in AlertSeverity}
            ),
            retrain_pipeline=pipe,
        )
        run = asyncio.run(scheduler.retrain_via_pipeline(
            "AAPL", pd.Series([100.0]), pd.DataFrame(),
        ))
        assert run.promoted is True
        assert scheduler.status.promotions == 1

    def test_scheduler_without_pipeline_raises(self):
        from src.execution.retrain_scheduler import RetrainScheduler
        from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
        scheduler = RetrainScheduler(
            alert_manager=AlertManager(
                channel_map={s: [LogChannel()] for s in AlertSeverity}
            ),
        )
        with pytest.raises(RuntimeError, match="retrain_pipeline not configured"):
            asyncio.run(scheduler.retrain_via_pipeline(
                "AAPL", pd.Series([100.0]), pd.DataFrame(),
            ))
