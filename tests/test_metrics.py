"""Tests for MetricsCollector (Phase 5 / P5.08)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from prometheus_client import CollectorRegistry

from src.execution.models import PortfolioState, Position
from src.monitoring.metrics import MetricsCollector


def _ts() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def collector() -> MetricsCollector:
    return MetricsCollector(registry=CollectorRegistry())


def _sample(collector: MetricsCollector, metric_name: str,
            labels: dict[str, str] | None = None) -> float:
    """Fetch a metric sample value from an isolated registry."""
    labels = labels or {}
    for metric in collector.registry.collect():
        for s in metric.samples:
            if s.name == metric_name and s.labels == labels:
                return s.value
    raise KeyError(f"metric {metric_name} labels={labels} not found")


class TestPortfolioMetrics:
    def test_update_portfolio_sets_gauges(self, collector):
        pos = Position(
            symbol="AAPL", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=110.0,
        )
        state = PortfolioState(cash=50_000.0, positions={"AAPL": pos})
        collector.update_portfolio(state)

        assert _sample(collector, "wang_trading_portfolio_nav") == pytest.approx(state.nav)
        assert _sample(collector, "wang_trading_positions_count") == 1.0
        assert _sample(collector, "wang_trading_portfolio_gross_exposure") == pytest.approx(
            state.gross_exposure
        )

    def test_drawdown_gauge(self, collector):
        state = PortfolioState(cash=100_000.0)
        state.peak_nav = 120_000.0
        state.drawdown = 0.1667
        collector.update_portfolio(state)
        assert _sample(collector, "wang_trading_portfolio_drawdown") == pytest.approx(0.1667)


class TestCounters:
    def test_order_counters_increment(self, collector):
        collector.record_order_submitted()
        collector.record_order_submitted(2)
        collector.record_order_filled()
        collector.record_order_rejected()
        assert _sample(collector, "wang_trading_orders_submitted_total") == 3.0
        assert _sample(collector, "wang_trading_orders_filled_total") == 1.0
        assert _sample(collector, "wang_trading_orders_rejected_total") == 1.0

    def test_signal_counter_by_family(self, collector):
        collector.record_signal("momentum", 3)
        collector.record_signal("mean_reversion", 1)
        collector.record_signal("momentum", 2)
        assert _sample(collector, "wang_trading_signal_count_total",
                       labels={"family": "momentum"}) == 5.0
        assert _sample(collector, "wang_trading_signal_count_total",
                       labels={"family": "mean_reversion"}) == 1.0

    def test_circuit_breaker_labels(self, collector):
        collector.record_circuit_breaker("fat_finger")
        collector.record_circuit_breaker("drawdown_halt", 2)
        assert _sample(collector, "wang_trading_circuit_breaker_triggers_total",
                       labels={"breaker_type": "fat_finger"}) == 1.0
        assert _sample(collector, "wang_trading_circuit_breaker_triggers_total",
                       labels={"breaker_type": "drawdown_halt"}) == 2.0


class TestHistograms:
    def test_slippage_observations(self, collector):
        for bps in (1.0, 2.0, 5.0, 10.0):
            collector.record_fill(bps)
        count = _sample(collector, "wang_trading_execution_slippage_bps_count")
        total = _sample(collector, "wang_trading_execution_slippage_bps_sum")
        assert count == 4.0
        assert total == pytest.approx(18.0)

    def test_meta_label_prob_observations(self, collector):
        for p in (0.1, 0.5, 0.9):
            collector.record_meta_label_prob(p)
        count = _sample(collector, "wang_trading_meta_label_prob_count")
        assert count == 3.0


class TestGaugesLabeled:
    def test_feature_drift_gauge(self, collector):
        collector.record_feature_drift("returns_ffd", 0.23)
        assert _sample(collector, "wang_trading_feature_drift_kl",
                       labels={"feature": "returns_ffd"}) == pytest.approx(0.23)
        # Overwriting updates, not accumulates
        collector.record_feature_drift("returns_ffd", 0.31)
        assert _sample(collector, "wang_trading_feature_drift_kl",
                       labels={"feature": "returns_ffd"}) == pytest.approx(0.31)

    def test_bar_rate_and_data_gap(self, collector):
        collector.record_bar_rate("AAPL", 120.0)
        collector.record_data_gap("AAPL", 3.2)
        assert _sample(collector, "wang_trading_bar_formation_rate",
                       labels={"symbol": "AAPL"}) == 120.0
        assert _sample(collector, "wang_trading_data_gap_seconds",
                       labels={"symbol": "AAPL"}) == pytest.approx(3.2)

    def test_model_retrain_age(self, collector):
        collector.update_model_age(_ts() - timedelta(hours=5))
        age = _sample(collector, "wang_trading_model_last_retrain_age_hours")
        assert age == pytest.approx(5.0, abs=0.1)


class TestSnapshot:
    def test_snapshot_returns_values(self, collector):
        state = PortfolioState(cash=10_000.0)
        collector.update_portfolio(state)
        collector.record_order_filled()
        snap = collector.snapshot()
        assert snap["wang_trading_portfolio_nav"] == pytest.approx(10_000.0)
        assert snap["wang_trading_orders_filled_total"] == 1.0
