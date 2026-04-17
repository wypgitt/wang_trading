"""Tests for RetrainScheduler (Phase 5 / P5.14)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.execution.retrain_scheduler import RetrainScheduler
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel


class _SpyAlertManager(AlertManager):
    def __init__(self):
        super().__init__(channel_map={s: [LogChannel()] for s in AlertSeverity},
                         default_cooldown_seconds=0)
        self.sent = []

    async def send_alert(self, alert, *, cooldown_seconds=None):
        self.sent.append(alert)
        return await super().send_alert(alert, cooldown_seconds=cooldown_seconds)


class FakeModel:
    def __init__(self, score: float):
        self.score = score


def _make_scheduler(
    *,
    trainer=None, evaluator=None, retrain_interval_days=7, min_new_bars=100,
):
    alerts = _SpyAlertManager()
    sched = RetrainScheduler(
        retrain_interval_days=retrain_interval_days,
        min_new_bars=min_new_bars,
        alert_manager=alerts,
        trainer=trainer,
        evaluator=evaluator,
    )
    return sched, alerts


# ── should_retrain ────────────────────────────────────────────────────

class TestShouldRetrain:
    @pytest.mark.asyncio
    async def test_true_when_aged_and_enough_bars(self):
        sched, _ = _make_scheduler(retrain_interval_days=7, min_new_bars=100)
        assert await sched.should_retrain(
            current_model_age_hours=24 * 8, new_bars_since_retrain=500,
        )

    @pytest.mark.asyncio
    async def test_false_when_fresh(self):
        sched, _ = _make_scheduler(retrain_interval_days=7, min_new_bars=100)
        assert not await sched.should_retrain(
            current_model_age_hours=24 * 1, new_bars_since_retrain=500,
        )

    @pytest.mark.asyncio
    async def test_false_when_not_enough_bars(self):
        sched, _ = _make_scheduler(retrain_interval_days=7, min_new_bars=100)
        assert not await sched.should_retrain(
            current_model_age_hours=24 * 10, new_bars_since_retrain=50,
        )


# ── retrain ───────────────────────────────────────────────────────────

@pytest.fixture
def small_data():
    close = pd.Series([100.0, 101.0, 99.0, 100.5], name="close")
    features = pd.DataFrame({"f1": [0.1, -0.1, 0.2, 0.0]})
    signals = pd.DataFrame({"side": [1, -1, 1, 0]})
    return close, features, signals


class TestRetrain:
    @pytest.mark.asyncio
    async def test_promotes_better_model(self, small_data):
        close, features, signals = small_data
        current = FakeModel(score=0.55)
        new = FakeModel(score=0.70)

        def trainer(c, f, s, mp):
            return new, new.score

        def evaluator(model, X, y):
            return model.score

        sched, alerts = _make_scheduler(trainer=trainer, evaluator=evaluator)
        eval_X = pd.DataFrame({"x": [1, 2, 3]})
        eval_y = pd.Series([0, 1, 0])

        result = await sched.retrain(
            close, features, signals, meta_pipeline=None,
            current_model=current, eval_X=eval_X, eval_y=eval_y,
        )
        assert result is new
        assert sched.status.promotions == 1
        assert any(a.title == "Model updated" for a in alerts.sent)

    @pytest.mark.asyncio
    async def test_keeps_current_when_worse(self, small_data):
        close, features, signals = small_data
        current = FakeModel(score=0.70)
        new = FakeModel(score=0.55)

        def trainer(c, f, s, mp):
            return new, new.score

        def evaluator(model, X, y):
            return model.score

        sched, alerts = _make_scheduler(trainer=trainer, evaluator=evaluator)
        eval_X = pd.DataFrame({"x": [1, 2]})
        eval_y = pd.Series([0, 1])

        result = await sched.retrain(
            close, features, signals, meta_pipeline=None,
            current_model=current, eval_X=eval_X, eval_y=eval_y,
        )
        assert result is None
        assert sched.status.rejected == 1
        assert any("did not improve" in a.title for a in alerts.sent)

    @pytest.mark.asyncio
    async def test_trainer_exception_alerts_critical(self, small_data):
        close, features, signals = small_data

        def trainer(*args, **kwargs):
            raise RuntimeError("boom")

        sched, alerts = _make_scheduler(trainer=trainer)
        result = await sched.retrain(close, features, signals,
                                     meta_pipeline=None, current_model=None)
        assert result is None
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts.sent)

    @pytest.mark.asyncio
    async def test_no_trainer_configured_is_noop(self, small_data):
        close, features, signals = small_data
        sched, alerts = _make_scheduler(trainer=None)
        result = await sched.retrain(close, features, signals,
                                     meta_pipeline=None, current_model=None)
        assert result is None
        assert any("skipped" in a.title.lower() for a in alerts.sent)


class TestStatus:
    def test_status_initial_state(self):
        sched, _ = _make_scheduler()
        st = sched.get_retrain_status()
        assert st["last_retrain"] is None
        assert st["promotions"] == 0
        assert st["rejected"] == 0

    @pytest.mark.asyncio
    async def test_status_after_promotion(self, small_data):
        close, features, signals = small_data

        def trainer(c, f, s, mp):
            return FakeModel(0.9), 0.9

        sched, _ = _make_scheduler(trainer=trainer,
                                   evaluator=lambda m, X, y: 0.1)
        await sched.retrain(close, features, signals, meta_pipeline=None,
                            current_model=FakeModel(0.1),
                            eval_X=pd.DataFrame({"x": [1]}),
                            eval_y=pd.Series([0]))
        st = sched.get_retrain_status()
        assert st["last_retrain"] is not None
        assert st["next_scheduled"] is not None
        assert st["promotions"] == 1
