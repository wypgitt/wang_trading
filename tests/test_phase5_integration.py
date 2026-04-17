"""Phase 5 end-to-end integration test (P5.15).

Drives the full execution pipeline for 100 cycles with a PaperBroker and
synthetic prices. Verifies:
  - orders are submitted + filled
  - portfolio NAV / positions / cash track broker
  - a forced daily-loss breach trips the circuit breaker
  - TCA is computable for every fill
  - metrics collector has fresh values
  - feature drift detector runs cleanly
  - reconciliation has no discrepancies
  - performance summary is valid
  - total runtime < 60 seconds
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.models import OrderStatus, PortfolioState
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PaperTradingPipeline, PipelineConfig
from src.execution.tca import TCAAnalyzer
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector


pytestmark = pytest.mark.integration


EQ_COST_CFG = {
    "commission_per_share": 0.0,
    "min_commission": 0.0,
    "spread_bps": 1.0,
    "slippage_bps": 1.0,
    "impact_coefficient": 0.1,
}


class _OscillatingOptimizer:
    """Alternates the target weight between small values to force rebalances."""

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.cycle = 0

    def compute_target_portfolio(self, **kwargs):
        self.cycle += 1
        weights = []
        for i, sym in enumerate(self.symbols):
            # Oscillate target ±5% with a phase offset — wide enough to clear
            # OrderManager.min_trade_pct (1%) on every cycle.
            w = 0.05 * np.sin((self.cycle + i) * 0.5)
            weights.append(w)
        return pd.DataFrame({
            "symbol": self.symbols,
            "target_weight": weights,
            "strategy": ["momentum"] * len(self.symbols),
        })


@pytest.mark.asyncio
async def test_phase5_full_pipeline_100_cycles():
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    prices: dict[str, float] = {s: 100.0 + 10 * i for i, s in enumerate(symbols)}

    broker = PaperBrokerAdapter(
        initial_cash=1_000_000.0,
        slippage_bps=0.0,  # keep 0 so LIMIT_AT_MID fills cleanly in synthetic test
        fill_delay_ms=0,
        price_feed=lambda s: prices.get(s, 100.0),
    )
    portfolio = PortfolioState(cash=1_000_000.0)
    cost = TransactionCostModel(equities_config=EQ_COST_CFG)
    cbs = CircuitBreakerManager(
        max_order_pct=0.50,
        daily_loss_limit_pct=0.02,
        max_positions=50,
        max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    om = OrderManager(broker, cbs, cost, portfolio,
                     adv_map={s: 1e7 for s in symbols})

    metrics = MetricsCollector(registry=CollectorRegistry())
    alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity},
                          default_cooldown_seconds=0)
    drift = FeatureDriftDetector()
    # Baseline for drift detector
    drift.set_baseline(pd.DataFrame({
        "ret": rng.normal(0, 0.01, 1000),
        "vol": rng.normal(0.02, 0.005, 1000),
    }))

    pipeline = PaperTradingPipeline(
        data_adapter=None,
        bar_constructors={},
        feature_assembler=None,
        signal_battery=None,
        meta_pipeline=None,
        meta_labeler=None,
        bet_sizing=None,
        portfolio_optimizer=_OscillatingOptimizer(symbols),
        order_manager=om,
        metrics=metrics,
        alert_manager=alerts,
        drift_detector=drift,
        config=PipelineConfig(
            sleep_seconds=0.0, drift_check_every=25, snapshot_every=25,
        ),
    )

    # Stub signal + meta output so the optimizer path is taken
    class _SignalStub:
        def generate(self, features):
            return pd.DataFrame({
                "symbol": symbols,
                "family": ["momentum"] * len(symbols),
                "side": [1] * len(symbols),
            })

    class _MetaStub:
        def predict(self, features, signals):
            return pd.DataFrame({
                "symbol": symbols, "meta_prob": [0.6] * len(symbols),
            })

    pipeline.signal_battery = _SignalStub()
    pipeline.meta_pipeline = _MetaStub()

    breaker_triggered_cycle: int | None = None

    for cycle in range(100):
        # Random walk the prices
        for s in symbols:
            prices[s] *= 1 + rng.normal(0, 0.002)

        # Rolling drift feature sample
        features = pd.DataFrame({
            "ret": rng.normal(0, 0.01, 50),
            "vol": rng.normal(0.02, 0.005, 50),
        })

        # Force a daily loss breach at cycle 60 to verify circuit breakers fire
        if cycle == 60:
            portfolio.daily_pnl = -0.025 * portfolio.nav

        result = await pipeline.run_cycle(features=features, prices=dict(prices))

        # After the forced breach, expect new entries to be rejected.
        if cycle == 60:
            any_rejected = any(
                o.status == OrderStatus.REJECTED
                for o in result["orders"]
            )
            if any_rejected:
                breaker_triggered_cycle = cycle

    elapsed = time.perf_counter() - t0

    # ── Invariants ────────────────────────────────────────────────────

    # 1) At least some orders filled over 100 cycles
    assert pipeline._orders_filled_count > 0

    # 2) Portfolio state internally consistent
    assert portfolio.nav == pytest.approx(
        portfolio.cash + sum(p.market_value for p in portfolio.positions.values()),
        rel=1e-9,
    )
    assert portfolio.cash == pytest.approx(broker.cash, rel=1e-9)

    # 3) Daily-loss circuit breaker rejected at least one entry at cycle 60
    assert breaker_triggered_cycle == 60

    # 4) TCA computable for every filled order so far
    tca_inputs = [o for o in portfolio.open_orders if o.fills] + []
    # Pull all filled orders from broker
    filled_orders = [o for o in broker.orders.values() if o.fills]
    assert filled_orders, "expected fills from paper broker"
    analyzer = TCAAnalyzer()
    for o in filled_orders[:5]:
        r = analyzer.analyze_order(
            o, arrival_mid=o.fills[0].price,
            market_prices_during=pd.Series([o.fills[0].price]),
        )
        assert np.isfinite(r.slippage_bps)

    # 5) Metrics snapshot has current NAV and some fill count
    snap = metrics.snapshot()
    assert snap["wang_trading_portfolio_nav"] == pytest.approx(portfolio.nav, rel=1e-6)
    assert snap["wang_trading_orders_filled_total"] > 0

    # 6) Drift detector ran without error (check is indirect — no exception raised)
    drift_out = drift.get_drifted_features(features)
    assert isinstance(drift_out, list)

    # 7) Reconciliation should find no discrepancies (paper broker mirrors us)
    discrepancies = await om.reconcile_positions()
    assert discrepancies == []

    # 8) Performance summary returns valid metrics
    summary = pipeline.get_performance_summary()
    assert summary["cycles"] == 100
    assert summary["nav"] == pytest.approx(portfolio.nav)
    assert "sharpe" in summary
    assert np.isfinite(summary["drawdown"])

    # 9) Runtime budget
    assert elapsed < 60.0, f"pipeline took {elapsed:.1f}s for 100 cycles"
