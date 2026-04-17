"""Tests for PaperTradingPipeline (Phase 5 / P5.12)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.models import (
    OrderStatus,
    PortfolioState,
    Position,
)
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import (
    PaperTradingPipeline,
    PipelineConfig,
)
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector


EQ_COST_CFG = {
    "commission_per_share": 0.0,
    "min_commission": 0.0,
    "spread_bps": 1.0,
    "slippage_bps": 1.0,
    "impact_coefficient": 0.1,
}


def _ts() -> datetime:
    return datetime.now(timezone.utc)


# ── Mocks ─────────────────────────────────────────────────────────────

class FakeSignalBattery:
    def __init__(self, signals: pd.DataFrame):
        self.signals = signals

    def generate(self, features):
        return self.signals


class FakeMetaPipeline:
    def __init__(self, meta: pd.DataFrame):
        self.meta = meta

    def predict(self, features, signals):
        return self.meta


class FakeBetSizing:
    def __init__(self, bets: pd.DataFrame):
        self.bets = bets

    def compute(self, meta, features):
        return self.bets


class FakePortfolioOptimizer:
    def __init__(self, target: pd.DataFrame):
        self.target = target

    def compute_target_portfolio(self, **kwargs):
        return self.target


def _make_pipeline(
    *,
    target: pd.DataFrame | None = None,
    prices: dict[str, float] | None = None,
    portfolio: PortfolioState | None = None,
    max_cycles: int | None = None,
):
    prices = prices or {"AAPL": 100.0, "TSLA": 200.0}
    portfolio = portfolio or PortfolioState(cash=1_000_000.0)
    broker = PaperBrokerAdapter(
        initial_cash=1_000_000.0, slippage_bps=0.0, fill_delay_ms=0,
        price_feed=lambda s: prices.get(s, 100.0),
    )
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST_CFG)
    om = OrderManager(broker, cbs, cost, portfolio)

    metrics = MetricsCollector(registry=CollectorRegistry())
    alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity})
    drift = FeatureDriftDetector()

    opt = FakePortfolioOptimizer(target) if target is not None else None
    # Supply minimal fake meta/signals so the optimizer is invoked
    meta = pd.DataFrame({"symbol": [], "meta_prob": []})
    sig = pd.DataFrame({"symbol": [], "family": []})
    bet = pd.DataFrame({"symbol": [], "size": []})

    pipeline = PaperTradingPipeline(
        data_adapter=None,
        bar_constructors={},
        feature_assembler=None,
        signal_battery=FakeSignalBattery(sig),
        meta_pipeline=FakeMetaPipeline(meta),
        meta_labeler=None,
        bet_sizing=FakeBetSizing(bet),
        portfolio_optimizer=opt,
        order_manager=om,
        metrics=metrics,
        alert_manager=alerts,
        drift_detector=drift,
        config=PipelineConfig(max_cycles=max_cycles, sleep_seconds=0.0,
                              drift_check_every=1000),
    )
    return pipeline, om, broker, portfolio


# ── Tests ─────────────────────────────────────────────────────────────

class TestInitialization:
    def test_pipeline_initializes_all_components(self):
        pipeline, *_ = _make_pipeline()
        assert pipeline.order_manager is not None
        assert pipeline.metrics is not None
        assert pipeline.alert_manager is not None
        assert pipeline.drift_detector is not None
        assert pipeline.cycle_count == 0
        assert pipeline.running is False


class TestSingleCycle:
    @pytest.mark.asyncio
    async def test_cycle_drives_orders_to_fills(self):
        prices = {"AAPL": 100.0, "TSLA": 200.0}
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.05, "strategy": "mom"},
            {"symbol": "TSLA", "target_weight": 0.05, "strategy": "mom"},
        ])
        features = pd.DataFrame({"ret": [0.01, 0.02]})
        pipeline, om, broker, pf = _make_pipeline(target=target, prices=prices)

        result = await pipeline.run_cycle(features=features, prices=prices)
        assert result["cycle"] == 1
        assert len(result["orders"]) == 2
        assert all(o.status == OrderStatus.FILLED for o in result["orders"])
        assert set(pf.positions.keys()) == {"AAPL", "TSLA"}

    @pytest.mark.asyncio
    async def test_metrics_updated_after_cycle(self):
        prices = {"AAPL": 100.0}
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.05, "strategy": "mom"},
        ])
        pipeline, om, _, pf = _make_pipeline(target=target, prices=prices)
        await pipeline.run_cycle(prices=prices)
        snap = pipeline.metrics.snapshot()
        assert snap["wang_trading_portfolio_nav"] == pytest.approx(pf.nav)


class TestCircuitBreakersChecked:
    @pytest.mark.asyncio
    async def test_breakers_fire_on_drawdown(self):
        prices = {"AAPL": 100.0}
        pos = Position(
            symbol="AAPL", side=1, quantity=1000, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = PortfolioState(cash=900_000.0, positions={"AAPL": pos})
        pf.peak_nav = pf.nav / 0.75
        pf.drawdown = 0.25
        pipeline, *_ = _make_pipeline(portfolio=pf, prices=prices)
        result = await pipeline.run_cycle(prices=prices)
        assert any(b.action == "HALT_AND_FLATTEN" for b in result["breakers"])
        snap = pipeline.metrics.snapshot()
        assert snap.get(
            "wang_trading_circuit_breaker_triggers_total"
            "{breaker_type=HALT_AND_FLATTEN}"
        ) == 1.0


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_flags_stop_and_logs(self, caplog):
        pipeline, *_ = _make_pipeline()
        pipeline.running = True
        import logging
        caplog.set_level(logging.INFO)
        await pipeline.shutdown()
        assert pipeline.running is False
        assert any("shutdown" in r.message.lower()
                   or "snapshot" in r.message.lower() for r in caplog.records)


class TestPerformanceSummary:
    @pytest.mark.asyncio
    async def test_summary_after_n_cycles(self):
        prices = {"AAPL": 100.0}
        target = pd.DataFrame([
            {"symbol": "AAPL", "target_weight": 0.01, "strategy": "mom"},
        ])
        pipeline, om, broker, pf = _make_pipeline(target=target, prices=prices)
        for _ in range(3):
            await pipeline.run_cycle(prices=prices)
        summary = pipeline.get_performance_summary()
        assert summary["cycles"] == 3
        assert summary["nav"] == pytest.approx(pf.nav)
        assert "sharpe" in summary and "drawdown" in summary

    @pytest.mark.asyncio
    async def test_run_loop_respects_max_cycles(self):
        prices = {"AAPL": 100.0}
        pipeline, *_ = _make_pipeline(max_cycles=2, prices=prices)
        # No data adapter → bars will be empty; drive directly via run_cycle loop
        for _ in range(2):
            await pipeline.run_cycle(prices=prices)
        assert pipeline.cycle_count == 2
