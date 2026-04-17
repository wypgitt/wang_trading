"""Tests for RL integration with the live pipeline (P6.11)."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.capital_deployment import CapitalDeploymentController
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.live_trading import (
    LiveTradingPipeline,
    _ShadowAwareOptimizer,
    _weights,
)
from src.execution.models import PortfolioState
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PipelineConfig
from src.execution.preflight import PreflightCheck
from src.ml_layer.rl_shadow import ShadowComparisonEngine
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector

pytestmark = pytest.mark.integration


EQ_COST_CFG = {
    "commission_per_share": 0.0, "min_commission": 0.0,
    "spread_bps": 1.0, "slippage_bps": 1.0, "impact_coefficient": 0.1,
}


class _Preflight:
    async def run_all_checks(self):
        return [PreflightCheck("ok", "", severity="blocker", passed=True)]

    @staticmethod
    def summary(checks):
        return {"all_passed": True, "blockers_failed": 0, "warnings_failed": 0,
                "checks": [c.to_dict() for c in checks]}


def _hrp_target() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": "AAPL", "target_weight": 0.05}])


def _rl_target() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": "MSFT", "target_weight": 0.05}])


class _FakeOptimizer:
    def compute_target_portfolio(self, **kwargs):
        return _hrp_target()


class _FakeRLShadow:
    def get_shadow_target_portfolio(self, context):
        return _rl_target()


def _make_pipeline(tmp_path: Path, *, rl_is_production: bool = False,
                   rl_agent=None, shadow_engine=None, promotion_controller=None,
                   portfolio=None):
    halt = tmp_path / "halt.lock"
    checkin = tmp_path / "checkin.lock"
    checkin.write_text("ok")
    compliance = tmp_path / "compliance.log"

    pf = portfolio or PortfolioState(cash=1_000_000.0)
    broker = PaperBrokerAdapter(
        initial_cash=1_000_000.0, slippage_bps=0.0, fill_delay_ms=0,
        price_feed=lambda s: 100.0,
    )
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST_CFG)
    om = OrderManager(broker, cbs, cost, pf)
    metrics = MetricsCollector(registry=CollectorRegistry())
    alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity})

    controller = CapitalDeploymentController(portfolio=pf)
    broker_factory = MagicMock()
    broker_factory.heartbeat_all = AsyncMock(return_value={"crypto": True})

    pipeline = LiveTradingPipeline(
        broker_factory=broker_factory,
        preflight_checker=_Preflight(),
        deployment_controller=controller,
        halt_file=halt,
        operator_checkin_path=checkin,
        compliance_log_path=compliance,
        shadow_rl_agent=rl_agent,
        shadow_comparison=shadow_engine,
        rl_is_production=rl_is_production,
        promotion_controller=promotion_controller,
        data_adapter=None,
        bar_constructors={},
        feature_assembler=None,
        signal_battery=None,
        meta_pipeline=None,
        meta_labeler=None,
        bet_sizing=None,
        portfolio_optimizer=_FakeOptimizer(),
        order_manager=om,
        metrics=metrics,
        alert_manager=alerts,
        drift_detector=FeatureDriftDetector(),
        config=PipelineConfig(max_cycles=1, sleep_seconds=0.0, drift_check_every=1000),
    )
    return pipeline, om, broker, pf


# ── Optimizer wrapper ───────────────────────────────────────────────────

class TestShadowAwareOptimizer:
    def test_shadow_mode_returns_hrp_and_records_rl(self):
        pipe = MagicMock()
        pipe.rl_is_production = False
        pipe.shadow_rl_agent = _FakeRLShadow()
        pipe.shadow_comparison = MagicMock()

        wrapper = _ShadowAwareOptimizer(pipe, _FakeOptimizer())
        out = wrapper.compute_target_portfolio(nav=100_000)
        assert list(out["symbol"]) == ["AAPL"]  # HRP wins in shadow mode
        pipe.shadow_comparison.record_decision.assert_called_once()
        kwargs = pipe.shadow_comparison.record_decision.call_args.kwargs
        assert kwargs["hrp_target"] == {"AAPL": 0.05}
        assert kwargs["rl_target"] == {"MSFT": 0.05}
        assert kwargs["executed_target"] == {"AAPL": 0.05}

    def test_production_mode_returns_rl_and_records_hrp(self):
        pipe = MagicMock()
        pipe.rl_is_production = True
        pipe.shadow_rl_agent = _FakeRLShadow()
        pipe.shadow_comparison = MagicMock()

        wrapper = _ShadowAwareOptimizer(pipe, _FakeOptimizer())
        out = wrapper.compute_target_portfolio(nav=100_000)
        assert list(out["symbol"]) == ["MSFT"]  # RL wins in production
        kwargs = pipe.shadow_comparison.record_decision.call_args.kwargs
        assert kwargs["executed_target"] == {"MSFT": 0.05}

    def test_no_shadow_agent_behaves_like_inner(self):
        pipe = MagicMock()
        pipe.rl_is_production = False
        pipe.shadow_rl_agent = None
        pipe.shadow_comparison = None
        wrapper = _ShadowAwareOptimizer(pipe, _FakeOptimizer())
        out = wrapper.compute_target_portfolio()
        assert list(out["symbol"]) == ["AAPL"]

    def test_weights_helper_handles_edge_cases(self):
        assert _weights(None) == {}
        assert _weights(pd.DataFrame({"symbol": ["A"], "target_weight": [0.1]})) == {"A": 0.1}


# ── Auto-revert on fast drawdown ────────────────────────────────────────

class TestAutoRevert:
    def test_revert_triggered_on_drawdown(self, tmp_path):
        pf = PortfolioState(cash=100_000.0)
        promotion = MagicMock()
        promotion.revert_to_hrp = AsyncMock(return_value={"current": "hrp"})
        pipeline, *_ = _make_pipeline(
            tmp_path, rl_is_production=True, rl_agent=_FakeRLShadow(),
            promotion_controller=promotion, portfolio=pf,
        )
        # Simulate 7% drawdown since promotion
        pipeline._rl_nav_at_promotion = 100_000.0
        pf.cash = 93_000.0
        pf.nav = 93_000.0

        asyncio.run(pipeline._maybe_auto_revert_rl(datetime.now(timezone.utc)))

        promotion.revert_to_hrp.assert_awaited_once()
        assert pipeline.rl_is_production is False

    def test_no_revert_within_threshold(self, tmp_path):
        pf = PortfolioState(cash=100_000.0)
        promotion = MagicMock()
        promotion.revert_to_hrp = AsyncMock()
        pipeline, *_ = _make_pipeline(
            tmp_path, rl_is_production=True, rl_agent=_FakeRLShadow(),
            promotion_controller=promotion, portfolio=pf,
        )
        pipeline._rl_nav_at_promotion = 100_000.0
        pf.cash = 98_000.0
        pf.nav = 98_000.0  # 2% dd < 5% threshold

        asyncio.run(pipeline._maybe_auto_revert_rl(datetime.now(timezone.utc)))
        promotion.revert_to_hrp.assert_not_awaited()
        assert pipeline.rl_is_production is True

    def test_no_revert_outside_window(self, tmp_path):
        pf = PortfolioState(cash=100_000.0)
        promotion = MagicMock()
        promotion.revert_to_hrp = AsyncMock()
        pipeline, *_ = _make_pipeline(
            tmp_path, rl_is_production=True, rl_agent=_FakeRLShadow(),
            promotion_controller=promotion, portfolio=pf,
        )
        pipeline._rl_promoted_at = datetime.now(timezone.utc) - timedelta(days=10)
        pipeline._rl_nav_at_promotion = 100_000.0
        pf.cash = 90_000.0
        pf.nav = 90_000.0

        asyncio.run(pipeline._maybe_auto_revert_rl(datetime.now(timezone.utc)))
        promotion.revert_to_hrp.assert_not_awaited()


# ── Operator approval required for promotion ───────────────────────────

class TestApproval:
    def test_approve_rl_promotion_calls_controller(self, tmp_path):
        promotion = MagicMock()
        promotion.promote_rl_to_production = AsyncMock(
            return_value={"current": "rl"}
        )
        pipeline, *_ = _make_pipeline(
            tmp_path, rl_agent=_FakeRLShadow(), promotion_controller=promotion,
        )
        result = asyncio.run(pipeline.approve_rl_promotion(
            gates={"cpcv": True, "dsr": True, "pbo": True},
        ))
        assert pipeline.rl_is_production is True
        assert pipeline._rl_promoted_at is not None
        assert result["current"] == "rl"
        promotion.promote_rl_to_production.assert_awaited_once()

    def test_no_auto_promotion_even_when_eligible(self, tmp_path):
        engine = ShadowComparisonEngine()
        promotion = MagicMock()
        # Make eligibility check return True
        engine.check_promotion_eligibility = MagicMock(return_value={
            "eligible": True, "reasons": [], "comparison": None,
        })
        pipeline, *_ = _make_pipeline(
            tmp_path, rl_agent=_FakeRLShadow(),
            shadow_engine=engine, promotion_controller=promotion,
        )
        asyncio.run(pipeline._check_rl_eligibility())
        # Alert fires but no auto-promotion
        promotion.promote_rl_to_production.assert_not_called()
        assert pipeline.rl_is_production is False

    def test_revert_clears_production_flag(self, tmp_path):
        promotion = MagicMock()
        promotion.revert_to_hrp = AsyncMock(return_value={"current": "hrp"})
        pipeline, *_ = _make_pipeline(
            tmp_path, rl_is_production=True, rl_agent=_FakeRLShadow(),
            promotion_controller=promotion,
        )
        asyncio.run(pipeline.revert_to_hrp("manual"))
        assert pipeline.rl_is_production is False
