"""Tests for the C1 schema extensions: labels / meta_labels / positions_history.

Every test is tagged ``@pytest.mark.db`` so it is excluded from the default
``make test`` (which runs with ``-m "not integration and not db"``). Run
explicitly with ``pytest -m db`` against a TimescaleDB instance, or rely on
the SQLAlchemy/DDL static assertions at the top of the file to confirm the
schema is well-formed without a server.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data_engine.storage.database import SCHEMA_SQL, DatabaseManager
from src.execution.models import Position


# ── DDL-level assertions (run in the default suite) ─────────────────────

class TestSchemaDDL:
    def test_labels_table_present(self):
        assert "CREATE TABLE IF NOT EXISTS labels" in SCHEMA_SQL
        for col in ("event_timestamp", "symbol", "side", "label",
                    "sample_weight", "barrier_touched"):
            assert col in SCHEMA_SQL

    def test_meta_labels_table_present(self):
        assert "CREATE TABLE IF NOT EXISTS meta_labels" in SCHEMA_SQL
        assert "model_version" in SCHEMA_SQL
        assert "idx_meta_labels_model_version" in SCHEMA_SQL

    def test_positions_history_table_present(self):
        assert "CREATE TABLE IF NOT EXISTS positions_history" in SCHEMA_SQL
        for col in ("avg_entry_price", "unrealized_pnl", "stop_loss",
                    "take_profit", "vertical_barrier"):
            assert col in SCHEMA_SQL

    def test_all_three_become_hypertables(self):
        assert SCHEMA_SQL.count("create_hypertable('labels'") == 1
        assert SCHEMA_SQL.count("create_hypertable('meta_labels'") == 1
        assert SCHEMA_SQL.count("create_hypertable('positions_history'") == 1


# ── DB round-trip (only runs under `pytest -m db`) ──────────────────────

_DB_URL = os.environ.get("WANG_TEST_DB_URL")
_DB_AVAILABLE = _DB_URL is not None


@pytest.fixture()
def db_manager() -> DatabaseManager:
    if not _DB_AVAILABLE:
        pytest.skip("WANG_TEST_DB_URL not set — skipping live DB test")
    mgr = DatabaseManager(_DB_URL)
    mgr.setup_schema()
    yield mgr
    mgr.close()


@pytest.mark.db
class TestLabelsRoundTrip:
    def test_insert_and_get(self, db_manager):
        symbol = "TEST_LABELS"
        now = datetime.now(timezone.utc)
        df = pd.DataFrame([
            {
                "event_timestamp": now,
                "symbol": symbol,
                "side": 1,
                "volatility": 0.02,
                "vertical_barrier": now + timedelta(hours=1),
                "upper_barrier": 101.0,
                "lower_barrier": 99.0,
                "exit_timestamp": now + timedelta(minutes=30),
                "barrier_touched": "upper",
                "return_pct": 0.01,
                "label": 1,
                "holding_period_bars": 10,
                "sample_weight": 1.0,
            }
        ])
        n = asyncio.run(db_manager.insert_labels(df))
        assert n == 1
        result = asyncio.run(db_manager.get_labels(symbol=symbol))
        assert len(result) >= 1


@pytest.mark.db
class TestMetaLabelsRoundTrip:
    def test_insert_and_get_with_model_filter(self, db_manager):
        symbol = "TEST_META"
        now = datetime.now(timezone.utc)
        asyncio.run(db_manager.insert_meta_label(
            timestamp=now, symbol=symbol, signal_family="trend",
            meta_prob=0.72, calibrated_prob=0.68,
            model_version="v1.0.0", feature_hash="abc123",
        ))
        out = asyncio.run(db_manager.get_meta_labels(
            symbol=symbol, model_version="v1.0.0",
        ))
        assert len(out) >= 1
        assert "meta_prob" in out.columns


@pytest.mark.db
class TestPositionsHistoryRoundTrip:
    def test_snapshot_and_query(self, db_manager):
        symbol = "TEST_POS"
        now = datetime.now(timezone.utc)
        pos = Position(
            symbol=symbol, side=1, quantity=100.0, avg_entry_price=150.0,
            entry_timestamp=now, signal_family="trend", current_price=152.0,
            unrealized_pnl=200.0,
        )
        n = asyncio.run(db_manager.insert_positions_snapshot(now, {symbol: pos}))
        assert n == 1
        out = asyncio.run(db_manager.get_positions_history(symbol=symbol))
        assert len(out) >= 1


# ── Persistence hooks (do not require a DB) ─────────────────────────────

class TestPipelineHooksNoDB:
    def test_meta_labeler_pipeline_accepts_db_manager_kwarg(self):
        from src.labeling.meta_labeler_pipeline import MetaLabelingPipeline
        import inspect
        sig = inspect.signature(MetaLabelingPipeline.prepare_training_data)
        assert "db_manager" in sig.parameters
        assert "symbol" in sig.parameters

    def test_meta_labeler_predict_proba_accepts_db_manager_kwarg(self):
        from src.ml_layer.meta_labeler import MetaLabeler
        import inspect
        sig = inspect.signature(MetaLabeler.predict_proba)
        for name in ("db_manager", "symbol", "signal_family", "model_version"):
            assert name in sig.parameters

    def test_pipeline_config_has_positions_history_every(self):
        from src.execution.paper_trading import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.positions_history_every == 0  # disabled by default

    def test_database_manager_has_new_methods(self):
        for name in ("insert_labels", "insert_meta_label",
                     "insert_positions_snapshot", "get_labels",
                     "get_meta_labels", "get_positions_history"):
            assert callable(getattr(DatabaseManager, name))
