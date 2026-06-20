"""Tests for drift-triggered auto-retrain integration (C3)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.models import PortfolioState
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PaperTradingPipeline, PipelineConfig
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector


EQ_COST = {
    "commission_per_share": 0.0, "min_commission": 0.0,
    "spread_bps": 1.0, "slippage_bps": 1.0, "impact_coefficient": 0.1,
}


def _pipeline(*, retrain_pipeline=None) -> PaperTradingPipeline:
    portfolio = PortfolioState(cash=100_000.0)
    broker = PaperBrokerAdapter(initial_cash=100_000.0, slippage_bps=0.0,
                                 fill_delay_ms=0, price_feed=lambda s: 100.0)
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST)
    om = OrderManager(broker, cbs, cost, portfolio)
    metrics = MetricsCollector(registry=CollectorRegistry())
    alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity})
    drift = FeatureDriftDetector()
    return PaperTradingPipeline(
        data_adapter=None, bar_constructors={},
        feature_assembler=None, signal_battery=None, meta_pipeline=None,
        meta_labeler=None, bet_sizing=None, portfolio_optimizer=None,
        order_manager=om, metrics=metrics, alert_manager=alerts,
        drift_detector=drift,
        config=PipelineConfig(max_cycles=1, sleep_seconds=0.0,
                              drift_check_every=1),
        retrain_pipeline=retrain_pipeline,
    )


# ── Fakes for the sizing path (drift-haircut application) ───────────────

class _FakeSignalBattery:
    def generate(self, frame, symbol=None):
        return pd.DataFrame({"symbol": ["AAPL"], "family": ["trend"], "side": [1]})


class _FakeMeta:
    def predict(self, features, signals):
        return pd.DataFrame({
            "symbol": ["AAPL"], "family": ["trend"], "side": [1],
            "meta_prob": [0.8],
        })


class _FakeBetSizing:
    """Returns a constant bet of final_size=RAW_SIZE on every compute call."""

    RAW_SIZE = 0.10

    def compute(self, meta, features):
        return pd.DataFrame({
            "symbol": ["AAPL"], "family": ["trend"], "side": [1],
            "final_size": [self.RAW_SIZE],
        })


class _RecordingOptimizer:
    """Captures the bet_sizes that actually reach the portfolio optimizer."""

    def __init__(self) -> None:
        self.seen_bet_sizes: list = []

    def compute_target_portfolio(self, **kwargs):
        bets = kwargs.get("bet_sizes")
        self.seen_bet_sizes.append(None if bets is None else bets.copy())
        return None  # no target → order_manager runs with target=None


def _sizing_pipeline(optimizer) -> PaperTradingPipeline:
    """Like `_pipeline` but wired with real fakes through the sizing path."""
    portfolio = PortfolioState(cash=100_000.0)
    broker = PaperBrokerAdapter(initial_cash=100_000.0, slippage_bps=0.0,
                                 fill_delay_ms=0, price_feed=lambda s: 100.0)
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST)
    om = OrderManager(broker, cbs, cost, portfolio)
    metrics = MetricsCollector(registry=CollectorRegistry())
    alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity})
    drift = FeatureDriftDetector()
    return PaperTradingPipeline(
        data_adapter=None, bar_constructors={},
        feature_assembler=None,
        signal_battery=_FakeSignalBattery(),
        meta_pipeline=_FakeMeta(), meta_labeler=None,
        bet_sizing=_FakeBetSizing(), portfolio_optimizer=optimizer,
        order_manager=om, metrics=metrics, alert_manager=alerts,
        drift_detector=drift,
        config=PipelineConfig(max_cycles=10, sleep_seconds=0.0,
                              drift_check_every=1),
        retrain_pipeline=None,
    )


# ── Severity branches ──────────────────────────────────────────────────

class TestDriftHandling:
    def test_severe_drift_triggers_emergency_retrain(self):
        async def _go():
            retrain = MagicMock()
            retrain.emergency_retrain = AsyncMock(return_value=MagicMock(
                promoted=True, rejection_reason=None,
            ))
            pipeline = _pipeline(retrain_pipeline=retrain)
            drifted = [f"f{i}" for i in range(6)]  # 60% of 10 features
            await pipeline._handle_drift(drifted, drift_pct=0.6)
            # Give the scheduled task a chance to run
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return retrain, pipeline
        retrain, pipeline = asyncio.run(_go())
        retrain.emergency_retrain.assert_awaited()
        assert pipeline._last_drift_retrain is not None
        # Bet sizes haircut active
        assert pipeline.drift_size_multiplier == 0.5

    def test_warning_band_no_retrain(self):
        async def _go():
            retrain = MagicMock()
            retrain.emergency_retrain = AsyncMock()
            pipeline = _pipeline(retrain_pipeline=retrain)
            await pipeline._handle_drift(["f1", "f2"], drift_pct=0.30)
            await asyncio.sleep(0)
            return retrain, pipeline
        retrain, pipeline = asyncio.run(_go())
        retrain.emergency_retrain.assert_not_awaited()
        assert pipeline._last_drift_retrain is None
        assert pipeline.drift_size_multiplier == 1.0

    def test_mild_drift_no_alert_no_retrain(self):
        async def _go():
            retrain = MagicMock()
            retrain.emergency_retrain = AsyncMock()
            pipeline = _pipeline(retrain_pipeline=retrain)
            # Patch alert_manager.send_alert to detect calls
            pipeline.alert_manager.send_alert = AsyncMock()
            await pipeline._handle_drift(["f1"], drift_pct=0.05)
            await asyncio.sleep(0)
            return retrain, pipeline
        retrain, pipeline = asyncio.run(_go())
        retrain.emergency_retrain.assert_not_awaited()
        pipeline.alert_manager.send_alert.assert_not_awaited()


# ── Cooldown ─────────────────────────────────────────────────────────

class TestCooldown:
    def test_24h_cooldown_blocks_second_retrain(self):
        async def _go():
            retrain = MagicMock()
            retrain.emergency_retrain = AsyncMock()
            pipeline = _pipeline(retrain_pipeline=retrain)
            drifted = [f"f{i}" for i in range(6)]
            # First call triggers
            await pipeline._handle_drift(drifted, drift_pct=0.6)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            first_count = retrain.emergency_retrain.await_count
            # Second immediate call should be skipped
            await pipeline._handle_drift(drifted, drift_pct=0.65)
            await asyncio.sleep(0)
            second_count = retrain.emergency_retrain.await_count
            return first_count, second_count
        first, second = asyncio.run(_go())
        assert first == 1
        assert second == 1  # no additional retrain

    def test_retrain_fires_again_after_cooldown(self):
        async def _go():
            retrain = MagicMock()
            retrain.emergency_retrain = AsyncMock()
            pipeline = _pipeline(retrain_pipeline=retrain)
            drifted = [f"f{i}" for i in range(6)]
            await pipeline._handle_drift(drifted, drift_pct=0.6)
            await asyncio.sleep(0)
            # Rewind the cooldown marker past the window
            pipeline._last_drift_retrain = datetime.now(timezone.utc) - timedelta(hours=25)
            await pipeline._handle_drift(drifted, drift_pct=0.6)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return retrain
        retrain = asyncio.run(_go())
        assert retrain.emergency_retrain.await_count == 2


# ── Non-blocking ─────────────────────────────────────────────────────

class TestNonBlocking:
    def test_slow_retrain_does_not_block_handle_drift(self):
        async def _go():
            retrain = MagicMock()

            slow_started = asyncio.Event()
            slow_done = asyncio.Event()

            async def slow(**kwargs):
                slow_started.set()
                await asyncio.sleep(0.5)
                slow_done.set()
                return MagicMock(promoted=True)

            retrain.emergency_retrain = slow
            pipeline = _pipeline(retrain_pipeline=retrain)
            drifted = [f"f{i}" for i in range(6)]

            # This should return immediately even though retrain takes 0.5s
            import time
            t0 = time.monotonic()
            await pipeline._handle_drift(drifted, drift_pct=0.6)
            elapsed = time.monotonic() - t0

            # Give the scheduled task a chance to start
            await asyncio.sleep(0.01)
            return elapsed, slow_started.is_set(), slow_done.is_set()

        elapsed, started, done = asyncio.run(_go())
        assert elapsed < 0.1  # handle_drift returned quickly
        assert started is True  # but the retrain task did fire
        # done may or may not be set; the point is handle_drift didn't wait.


# ── Constructor surface ──────────────────────────────────────────────

class TestConstructor:
    def test_defaults(self):
        pipeline = _pipeline()
        assert pipeline.retrain_pipeline is None
        assert pipeline.drift_retrain_threshold == 0.50
        assert pipeline.drift_warning_threshold == 0.20
        assert pipeline.drift_retrain_cooldown_hours == 24.0
        assert pipeline._last_drift_retrain is None

    def test_size_multiplier_expires(self):
        pipeline = _pipeline()
        pipeline._drift_size_haircut_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert pipeline.drift_size_multiplier == 1.0
        assert pipeline._drift_size_haircut_until is None


# ── Haircut actually reaches order sizing (the live gap) ───────────────

class TestDriftHaircutApplied:
    """The protective haircut must change real order sizing, not just a
    property read by tests. Drive two cycles: a severe-drift cycle opens the
    24h window, and the *next* cycle's bet sizes must be scaled by 0.5."""

    def test_next_cycle_bet_sizes_halved_after_severe_drift(self):
        async def _go():
            optimizer = _RecordingOptimizer()
            pipeline = _sizing_pipeline(optimizer)
            features = pd.DataFrame(
                {"a": [1.0, 1.1], "b": [2.0, 2.1],
                 "c": [3.0, 3.1], "d": [4.0, 4.1]}
            )
            # Force every drift check to report 100% severe drift.
            pipeline.drift_detector.get_drifted_features = MagicMock(
                return_value=["a", "b", "c", "d"]
            )
            prices = {"AAPL": 100.0}
            # Cycle 1: sizing runs *before* drift handling, so it is NOT
            # haircut; the severe-drift handler then opens the 24h window.
            await pipeline.run_cycle(prices=prices, features=features)
            # Cycle 2: the haircut window is now active.
            await pipeline.run_cycle(prices=prices, features=features)
            await asyncio.sleep(0)
            return optimizer, pipeline

        optimizer, pipeline = asyncio.run(_go())

        assert pipeline.drift_size_multiplier == 0.5  # window still active
        assert len(optimizer.seen_bet_sizes) == 2
        raw = _FakeBetSizing.RAW_SIZE
        # Cycle 1 sized at full size (haircut not yet active during sizing).
        assert optimizer.seen_bet_sizes[0]["final_size"].iloc[0] == pytest.approx(raw)
        # Cycle 2 sized at the 50% haircut — the gap this change closes.
        assert optimizer.seen_bet_sizes[1]["final_size"].iloc[0] == pytest.approx(raw * 0.5)


class TestApplyDriftHaircut:
    """Unit coverage for the `_apply_drift_haircut` helper branches."""

    def _active(self) -> PaperTradingPipeline:
        pipeline = _pipeline()
        pipeline._drift_size_haircut_until = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        )
        return pipeline

    def test_scales_final_size_when_active(self):
        pipeline = self._active()
        bets = pd.DataFrame({"symbol": ["AAPL"], "final_size": [0.2]})
        out = pipeline._apply_drift_haircut(bets)
        assert out["final_size"].iloc[0] == pytest.approx(0.1)
        # Input frame is not mutated (copy-on-write).
        assert bets["final_size"].iloc[0] == pytest.approx(0.2)

    def test_noop_when_window_inactive(self):
        pipeline = _pipeline()  # no haircut window
        bets = pd.DataFrame({"symbol": ["AAPL"], "final_size": [0.2]})
        out = pipeline._apply_drift_haircut(bets)
        assert out["final_size"].iloc[0] == pytest.approx(0.2)

    def test_falls_back_to_size_column(self):
        pipeline = self._active()
        bets = pd.DataFrame({"symbol": ["AAPL"], "size": [0.4]})
        out = pipeline._apply_drift_haircut(bets)
        assert out["size"].iloc[0] == pytest.approx(0.2)

    def test_handles_none_and_empty(self):
        pipeline = self._active()
        assert pipeline._apply_drift_haircut(None) is None
        out = pipeline._apply_drift_haircut(pd.DataFrame())
        assert out.empty

    def test_no_weight_column_is_left_unchanged(self):
        pipeline = self._active()
        bets = pd.DataFrame({"symbol": ["AAPL"], "afml_size": [0.2]})
        out = pipeline._apply_drift_haircut(bets)
        assert out["afml_size"].iloc[0] == pytest.approx(0.2)
