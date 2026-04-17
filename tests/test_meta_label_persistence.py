"""Tests for meta-label persistence + feature hashing (C4)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.models import PortfolioState
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PaperTradingPipeline, PipelineConfig
from src.feature_factory.assembler import compute_feature_hash
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector


EQ_COST = {
    "commission_per_share": 0.0, "min_commission": 0.0,
    "spread_bps": 1.0, "slippage_bps": 1.0, "impact_coefficient": 0.1,
}


# ── Feature hashing ────────────────────────────────────────────────────

class TestFeatureHash:
    def test_deterministic(self):
        row = pd.Series({"rsi": 55.5, "atr": 1.25, "mom": 0.03})
        assert compute_feature_hash(row) == compute_feature_hash(row)

    def test_order_independence(self):
        r1 = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        r2 = pd.Series({"c": 3.0, "a": 1.0, "b": 2.0})
        assert compute_feature_hash(r1) == compute_feature_hash(r2)

    def test_different_rows_differ(self):
        r1 = pd.Series({"a": 1.0, "b": 2.0})
        r2 = pd.Series({"a": 1.0, "b": 2.1})
        assert compute_feature_hash(r1) != compute_feature_hash(r2)

    def test_nan_handled(self):
        row = pd.Series({"a": 1.0, "b": float("nan")})
        h = compute_feature_hash(row)
        assert len(h) == 64  # SHA-256 hex
        # Repeatable with a fresh NaN
        row2 = pd.Series({"b": float("nan"), "a": 1.0})
        assert compute_feature_hash(row2) == h

    def test_empty_row(self):
        h = compute_feature_hash(pd.Series(dtype=float))
        assert len(h) == 64

    def test_precision_rounding_absorbs_tiny_noise(self):
        r1 = pd.Series({"a": 1.0})
        r2 = pd.Series({"a": 1.0 + 1e-15})  # below default 10-decimal precision
        assert compute_feature_hash(r1) == compute_feature_hash(r2)


# ── Pipeline persistence hook ──────────────────────────────────────────

def _make_pipeline(db_manager=None) -> PaperTradingPipeline:
    pf = PortfolioState(cash=100_000.0)
    broker = PaperBrokerAdapter(initial_cash=100_000.0, slippage_bps=0.0,
                                 fill_delay_ms=0, price_feed=lambda s: 100.0)
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, max_positions=50, max_single_position_pct=0.50,
        max_gross_exposure=3.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST)
    om = OrderManager(broker, cbs, cost, pf)
    return PaperTradingPipeline(
        data_adapter=None, bar_constructors={},
        feature_assembler=None, signal_battery=None, meta_pipeline=None,
        meta_labeler=None, bet_sizing=None, portfolio_optimizer=None,
        order_manager=om,
        metrics=MetricsCollector(registry=CollectorRegistry()),
        alert_manager=AlertManager(
            channel_map={s: [LogChannel()] for s in AlertSeverity}
        ),
        drift_detector=FeatureDriftDetector(),
        config=PipelineConfig(max_cycles=1, sleep_seconds=0.0),
        db_manager=db_manager,
    )


class TestMetaLabelPersistence:
    def test_rows_written_with_correct_columns(self):
        db = MagicMock()
        db.insert_meta_label = AsyncMock()

        pipeline = _make_pipeline(db_manager=db)
        meta = pd.DataFrame([
            {"symbol": "AAPL", "family": "trend", "meta_prob": 0.72,
             "calibrated_prob": 0.70,
             "timestamp": datetime.now(timezone.utc)},
            {"symbol": "MSFT", "family": "carry", "meta_prob": 0.55,
             "calibrated_prob": 0.54,
             "timestamp": datetime.now(timezone.utc)},
        ])
        features = pd.DataFrame({
            "f1": [1.0, 2.0], "f2": [3.0, 4.0],
        })

        asyncio.run(pipeline._persist_meta_labels(meta, features))

        assert db.insert_meta_label.await_count == 2
        call = db.insert_meta_label.await_args_list[0].kwargs
        for key in ("timestamp", "symbol", "signal_family", "meta_prob",
                    "calibrated_prob", "model_version", "feature_hash"):
            assert key in call
        assert call["symbol"] == "AAPL"
        assert call["signal_family"] == "trend"
        assert call["meta_prob"] == pytest.approx(0.72)
        assert call["calibrated_prob"] == pytest.approx(0.70)
        assert len(call["feature_hash"]) == 64

    def test_db_failure_does_not_break_cycle(self):
        db = MagicMock()
        db.insert_meta_label = AsyncMock(side_effect=RuntimeError("DB down"))

        pipeline = _make_pipeline(db_manager=db)
        meta = pd.DataFrame([{
            "symbol": "AAPL", "family": "trend", "meta_prob": 0.5,
        }])
        features = pd.DataFrame({"f": [1.0]})

        # Must NOT raise — persistence failure is best-effort
        asyncio.run(pipeline._persist_meta_labels(meta, features))

    def test_no_db_manager_is_noop(self):
        pipeline = _make_pipeline(db_manager=None)
        # Should return cleanly with no DB side effects
        asyncio.run(pipeline._persist_meta_labels(
            pd.DataFrame([{"symbol": "AAPL", "family": "x", "meta_prob": 0.5}]),
            pd.DataFrame({"f": [1.0]}),
        ))

    def test_empty_meta_is_noop(self):
        db = MagicMock()
        db.insert_meta_label = AsyncMock()
        pipeline = _make_pipeline(db_manager=db)
        asyncio.run(pipeline._persist_meta_labels(pd.DataFrame(), pd.DataFrame()))
        db.insert_meta_label.assert_not_called()

    def test_feature_hash_aligns_with_feature_row(self):
        db = MagicMock()
        db.insert_meta_label = AsyncMock()
        pipeline = _make_pipeline(db_manager=db)

        meta = pd.DataFrame([
            {"symbol": "AAPL", "family": "trend", "meta_prob": 0.6},
        ])
        features = pd.DataFrame({"f1": [1.23], "f2": [4.56]})
        asyncio.run(pipeline._persist_meta_labels(meta, features))

        persisted_hash = db.insert_meta_label.await_args.kwargs["feature_hash"]
        expected = compute_feature_hash(features.iloc[0])
        assert persisted_hash == expected
