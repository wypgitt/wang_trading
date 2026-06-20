"""Tests for RetrainPipeline end-to-end wiring (C2)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtesting.gate_orchestrator import StrategyGate
from src.backtesting.transaction_costs import EQUITIES_COSTS, TransactionCostModel
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
    gate.evaluate_candidate = MagicMock(return_value={
        "passed": True,
        "gate_1_cpcv": {"passed": True, "positive_paths": 30,
                        "total_paths": 45, "pct": 30 / 45},
        "gate_2_dsr": {"passed": True, "statistic": 3.0, "p_value": 0.001},
        "gate_3_pbo": {"passed": True, "pbo_value": 0.1},
    })

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
        logged = pipe.registry.log_training_run.call_args.kwargs
        assert logged["X"].shape == (10, 2)
        assert logged["y"].shape[0] == 10
        assert logged["metrics"]["gate_cpcv"] == 1.0
        assert logged["metrics"]["gate_dsr"] == 1.0
        assert logged["metrics"]["gate_pbo"] == 1.0
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
        pipe.gate.evaluate_candidate = MagicMock(return_value={
            "passed": False,
            "gate_1_cpcv": {"passed": True, "positive_paths": 30,
                            "total_paths": 45, "pct": 30 / 45},
            "gate_2_dsr": {"passed": True, "statistic": 3.0, "p_value": 0.001},
            "gate_3_pbo": {"passed": False, "pbo_value": 0.9},
        })
        run = asyncio.run(pipe.run("AAPL", pd.Series([100.0]), pd.DataFrame()))
        assert run.promoted is False
        assert "gate_failed" in run.rejection_reason
        # The rejection names the gate that actually blocked it (not an empty
        # list and not the historical uniform "gate_unavailable").
        assert "gate_3_pbo" in run.rejection_reason
        assert "gate_unavailable" not in run.rejection_reason
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


# ── Promotion gate produces REAL verdicts (regression for the historical
#    `gate_unavailable` fall-through in retrain_pipeline._run_gate) ─────────

def _real_gate_pipeline(*, drift: float, vol: float, n: int = 500, seed: int = 0):
    """A pipeline wired to a *real* ``StrategyGate`` over synthetic but
    production-shaped data (tz-aware UTC bars, flat signal-battery frame), so
    the gate actually backtests the candidate and computes CPCV/DSR/PBO."""
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    price = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    close = pd.Series(price, index=idx)
    bars = pd.DataFrame({"close": price}, index=idx)
    features = pd.DataFrame(
        {"f1": price, "f2": np.r_[0.0, np.diff(price)]}, index=idx
    )
    flat_signals = pd.DataFrame(
        {"timestamp": idx, "symbol": "AAA", "family": "trend",
         "side": 1, "confidence": 0.5}
    )
    X = features.reset_index(drop=True)
    y = pd.Series((np.r_[0.0, np.diff(price)] > 0).astype(int))
    sw = pd.Series(np.ones(n))

    meta = MagicMock()
    meta.prepare_training_data = MagicMock(return_value=(X, y, sw))
    meta.max_holding_period = 25

    assembler = MagicMock()
    assembler.compute = MagicMock(return_value=features)
    battery = MagicMock()
    battery.generate = MagicMock(return_value=flat_signals)

    registry = MagicMock()
    registry.get_production_model = MagicMock(return_value=None)
    registry.log_training_run = MagicMock(return_value="run-xyz")
    registry.promote_model = MagicMock()

    pipe = RetrainPipeline(
        meta_labeling_pipeline=meta,
        feature_assembler=assembler,
        signal_battery=battery,
        gate=StrategyGate(),
        registry=registry,
        cost_model=TransactionCostModel(equities_config=EQUITIES_COSTS),
        trainer=MagicMock(return_value=(MagicMock(name="candidate"), 0.9)),
        scorer=MagicMock(return_value=0.5),
        min_improvement_pct=0.05,
    )
    return pipe, close, bars


class _SpyGate:
    """Records the kwargs ``_run_gate`` threads into ``evaluate_candidate``."""

    def __init__(self, result: dict) -> None:
        self.calls: list[dict] = []
        self._result = result

    def evaluate_candidate(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


class _ExplodingGate:
    def evaluate_candidate(self, **kwargs):
        raise RuntimeError("cpcv blew up")


class _NoMethodGate:
    """A gate object missing ``evaluate_candidate`` entirely."""


class TestGateProducesRealVerdicts:
    def test_real_gate_passes_and_persists_real_flags(self):
        pipe, close, bars = _real_gate_pipeline(drift=0.004, vol=0.004)
        run = asyncio.run(pipe.run("AAA", close, bars))

        assert run.promoted is True
        # Real CPCV/DSR/PBO verdicts flowed through — never the stub.
        assert run.gate_results["gate_1_cpcv"]["passed"] is True
        assert run.gate_results["gate_2_dsr"]["passed"] is True
        assert run.gate_results["gate_3_pbo"]["passed"] is True
        assert isinstance(run.gate_results["gate_2_dsr"]["p_value"], float)
        assert "gate_unavailable" not in str(run.gate_results)

        # The three MLflow gate flags persisted on the promoted model are REAL
        # (1.0), not the historically hardcoded 0.0.
        logged = pipe.registry.log_training_run.call_args.kwargs
        assert logged["metrics"]["gate_cpcv"] == 1.0
        assert logged["metrics"]["gate_dsr"] == 1.0
        assert logged["metrics"]["gate_pbo"] == 1.0
        assert logged["params"]["gates"] == {"cpcv": True, "dsr": True, "pbo": True}
        pipe.registry.promote_model.assert_called_once_with("run-xyz")

        # Decode exactly as the Research/Backtests screen does (the function the
        # BFF reads through): the persisted flags resolve to REAL verdicts, not
        # the historically uniform 0/False.
        from src.ml_layer.model_registry import ModelRegistry
        decoded = ModelRegistry._extract_gate_flags(
            logged["params"], logged["metrics"]
        )
        assert decoded == {"cpcv": True, "dsr": True, "pbo": True}

    def test_real_gate_fails_with_real_verdict_and_zero_flags(self):
        pipe, close, bars = _real_gate_pipeline(drift=0.0, vol=0.01, seed=3)
        run = asyncio.run(pipe.run("AAA", close, bars))

        assert run.promoted is False
        # DSR is a genuine computed False, not None/stub/gate_unavailable.
        assert run.gate_results["gate_2_dsr"]["passed"] is False
        assert run.gate_results["gate_2_dsr"]["p_value"] >= 0.05
        assert "gate_failed" in run.rejection_reason
        assert "gate_2_dsr" in run.rejection_reason
        assert "gate_unavailable" not in run.rejection_reason
        assert "gate_unavailable" not in str(run.gate_results)
        pipe.registry.promote_model.assert_not_called()
        pipe.registry.log_training_run.assert_not_called()

    def test_run_gate_threads_real_data_into_evaluate_candidate(self):
        pipe = _pipeline()
        spy = _SpyGate({
            "passed": True,
            "gate_1_cpcv": {"passed": True}, "gate_2_dsr": {"passed": True},
            "gate_3_pbo": {"passed": True},
        })
        pipe.gate = spy
        pipe.meta_labeling_pipeline.max_holding_period = 25
        idx = pd.date_range("2021-01-01", periods=3, freq="B", tz="UTC")
        run = asyncio.run(pipe.run("AAA", pd.Series([100.0, 101.0, 102.0], index=idx),
                                   pd.DataFrame()))
        assert run.promoted is True
        assert len(spy.calls) == 1
        kw = spy.calls[0]
        # The gate received the data it needs — wide close/signals panel, bet
        # sizes, features and the cost model — keyed by name (the bug was that
        # NONE of this was threaded, so the gate always raised).
        assert set(kw) >= {
            "close", "signals", "bet_sizes", "features", "cost_model",
            "cpcv_horizon",
        }
        assert list(kw["close"].columns) == ["AAA"]
        assert len(kw["close"]) == 3
        assert kw["cpcv_horizon"] == 25  # from meta.max_holding_period

    def test_gate_exception_surfaces_loudly_not_gate_unavailable(self):
        pipe = _pipeline()
        pipe.gate = _ExplodingGate()
        run = asyncio.run(pipe.run("AAA", pd.Series([100.0]), pd.DataFrame()))
        assert run.promoted is False
        # The error is surfaced with a real reason + traceback, NOT degraded to
        # a uniform `gate_unavailable` reject.
        assert "cpcv blew up" in run.rejection_reason
        assert "trace" in run.gate_results
        assert "gate_unavailable" not in str(run.gate_results)
        pipe.registry.promote_model.assert_not_called()

    def test_missing_gate_method_is_distinct_from_gate_unavailable(self):
        pipe = _pipeline()
        pipe.gate = _NoMethodGate()
        run = asyncio.run(pipe.run("AAA", pd.Series([100.0]), pd.DataFrame()))
        assert run.promoted is False
        # A config error reads as `gate_not_configured`, never the misleading
        # `gate_unavailable` (which used to fire on EVERY healthy retrain).
        assert "gate_not_configured" in run.rejection_reason
        assert "gate_unavailable" not in run.rejection_reason
        pipe.registry.promote_model.assert_not_called()
