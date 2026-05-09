"""Tests for per-stage latency and cost telemetry metrics."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from src.monitoring.metrics import MetricsCollector


def test_stage_latency_cost_and_item_metrics_snapshot():
    metrics = MetricsCollector(registry=CollectorRegistry())

    metrics.record_stage_latency("data_fetch", 0.012)
    metrics.record_stage_items("data_fetch", "bars", 3)
    metrics.record_stage_cost("order_routing", 1.25)
    snap = metrics.snapshot()

    assert snap["wang_trading_stage_latency_seconds_count{stage=data_fetch}"] == 1.0
    assert snap["wang_trading_stage_items_total{item_type=bars,stage=data_fetch}"] == 3.0
    assert snap["wang_trading_stage_cost_usd_total{stage=order_routing}"] == 1.25
