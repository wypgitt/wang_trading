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
