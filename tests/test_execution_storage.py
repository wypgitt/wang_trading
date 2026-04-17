"""Tests for ExecutionStorage (Phase 5 / P5.07).

Unit tests run against a temp-file SQLite DB (always available). The
TimescaleDB-specific hypertable DDL is silently skipped on SQLite.

A real-backend smoke test is provided and marked ``@pytest.mark.db`` so it
only runs when `pytest -m db` is invoked against a running Postgres
(Timescale) instance pointed to by ``DATABASE_URL``.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.execution.models import (
    ExecutionAlgo,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)
from src.execution.storage import ExecutionStorage
from src.execution.tca import TCAResult


def _ts(offset_s: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_s)


@pytest.fixture
def storage(tmp_path: Path) -> ExecutionStorage:
    db_path = tmp_path / "exec_test.sqlite"
    s = ExecutionStorage(f"sqlite:///{db_path}")
    s.setup_execution_schema()
    yield s
    s.close()


def _make_order(symbol: str = "AAPL", side: int = 1, qty: float = 100.0) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=_ts(),
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=qty * side,
        limit_price=150.0,
        execution_algo=ExecutionAlgo.TWAP,
        signal_family="momentum",
        meta_label_prob=0.6,
    )


class TestOrderStorage:
    def test_insert_and_retrieve(self, storage):
        order = _make_order()
        storage.insert_order(order)
        df = storage.get_orders("AAPL", _ts(-60), _ts(60))
        assert len(df) == 1
        assert df.iloc[0]["order_id"] == order.order_id
        assert df.iloc[0]["status"] == OrderStatus.PENDING.value
        assert df.iloc[0]["algo"] == "twap"

    def test_update_order_status(self, storage):
        order = _make_order()
        storage.insert_order(order)
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.updated_at = _ts(1)
        storage.update_order(order)
        df = storage.get_orders("AAPL", _ts(-60), _ts(60))
        assert df.iloc[0]["status"] == "filled"
        assert df.iloc[0]["filled_qty"] == pytest.approx(order.quantity)


class TestFillStorage:
    def test_insert_and_retrieve_fills(self, storage):
        order = _make_order()
        storage.insert_order(order)
        for i in range(3):
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                timestamp=_ts(i),
                price=150.0 + i * 0.1,
                quantity=33.0,
                commission=0.01,
                exchange="PAPER",
                is_maker=True,
            )
            storage.insert_fill(fill)
        df = storage.get_fills(order.order_id)
        assert len(df) == 3
        assert df["price"].tolist() == [150.0, 150.1, 150.2]
        assert df["is_maker"].tolist() == [1, 1, 1]


class TestTCAStorage:
    def test_round_trip(self, storage):
        result = TCAResult(
            order_id=str(uuid.uuid4()), symbol="AAPL", side=1,
            arrival_price=100.0, execution_price=101.0,
            slippage_bps=100.0, market_impact_bps=50.0,
            timing_cost_bps=50.0, total_cost_bps=100.0,
            commission=1.0, algo_used="twap",
            execution_duration_seconds=30.0, fill_rate=1.0,
            benchmark_vs_twap_bps=5.0, benchmark_vs_vwap_bps=3.0,
        )
        storage.insert_tca(result, timestamp=_ts())
        df = storage.get_tca_history(_ts(-60), _ts(60))
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["slippage_bps"] == pytest.approx(100.0)
        assert df.iloc[0]["algo"] == "twap"


class TestPortfolioSnapshots:
    def test_snapshot_round_trip(self, storage):
        state = PortfolioState(
            cash=50_000.0,
            positions={
                "AAPL": Position(
                    symbol="AAPL", side=1, quantity=100,
                    avg_entry_price=100.0, entry_timestamp=_ts(),
                    signal_family="", current_price=105.0,
                ),
            },
        )
        storage.insert_portfolio_snapshot(state, timestamp=_ts())
        df = storage.get_portfolio_history(_ts(-60), _ts(60))
        assert len(df) == 1
        row = df.iloc[0]
        assert row["nav"] == pytest.approx(state.nav)
        assert row["cash"] == pytest.approx(50_000.0)
        assert row["n_positions"] == 1

    def test_multiple_snapshots_ordered(self, storage):
        for i in range(5):
            state = PortfolioState(cash=50_000.0 + i * 100)
            storage.insert_portfolio_snapshot(state, timestamp=_ts(i))
        df = storage.get_portfolio_history(_ts(-60), _ts(60))
        assert len(df) == 5
        assert df["cash"].tolist() == [50_000, 50_100, 50_200, 50_300, 50_400]


# ── Real-DB smoke (opt-in) ────────────────────────────────────────────

@pytest.mark.db
class TestAgainstTimescale:
    def test_setup_against_real_timescale(self):
        url = os.environ.get("DATABASE_URL")
        if not url:
            pytest.skip("DATABASE_URL not set")
        storage = ExecutionStorage(url)
        try:
            storage.setup_execution_schema()
            order = _make_order()
            storage.insert_order(order)
            df = storage.get_orders("AAPL", _ts(-60), _ts(60))
            assert len(df) >= 1
        finally:
            storage.close()
