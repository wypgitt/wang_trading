"""Execution-layer database storage (Phase 5 / P5.07).

Extends the Phase 1 DatabaseManager with tables for orders, fills, TCA
results, and portfolio snapshots. All time-series tables are declared as
TimescaleDB hypertables; the hypertable DDL is skipped silently on non-
Timescale backends (e.g. SQLite used by unit tests).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.data_engine.storage.database import DatabaseManager
from src.execution.models import Fill, Order, PortfolioState
from src.execution.tca import TCAResult


# ── Schema ─────────────────────────────────────────────────────────────

EXECUTION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT        PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    side            SMALLINT    NOT NULL,
    order_type      TEXT        NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    filled_qty      DOUBLE PRECISION DEFAULT 0,
    status          TEXT        NOT NULL,
    algo            TEXT        DEFAULT '',
    signal_family   TEXT        DEFAULT '',
    meta_prob       DOUBLE PRECISION DEFAULT 0,
    limit_price     DOUBLE PRECISION,
    parent_order_id TEXT,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol_time ON orders (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status      ON orders (status);

CREATE TABLE IF NOT EXISTS fills (
    fill_id     TEXT        PRIMARY KEY,
    order_id    TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    quantity    DOUBLE PRECISION NOT NULL,
    commission  DOUBLE PRECISION DEFAULT 0,
    exchange    TEXT        NOT NULL,
    is_maker    SMALLINT    DEFAULT 0
);

SELECT create_hypertable('fills', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_fills_order_id  ON fills (order_id);
CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills (timestamp DESC);

CREATE TABLE IF NOT EXISTS tca_results (
    order_id            TEXT        NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              TEXT        NOT NULL,
    side                SMALLINT    NOT NULL,
    arrival_price       DOUBLE PRECISION NOT NULL,
    exec_price          DOUBLE PRECISION NOT NULL,
    slippage_bps        DOUBLE PRECISION NOT NULL,
    impact_bps          DOUBLE PRECISION NOT NULL,
    timing_bps          DOUBLE PRECISION NOT NULL,
    total_bps           DOUBLE PRECISION NOT NULL,
    commission          DOUBLE PRECISION DEFAULT 0,
    algo                TEXT        NOT NULL,
    duration_seconds    DOUBLE PRECISION DEFAULT 0,
    fill_rate           DOUBLE PRECISION DEFAULT 0,
    twap_bench_bps      DOUBLE PRECISION DEFAULT 0,
    vwap_bench_bps      DOUBLE PRECISION DEFAULT 0
);

SELECT create_hypertable('tca_results', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_tca_symbol_time ON tca_results (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tca_algo        ON tca_results (algo);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    timestamp        TIMESTAMPTZ      NOT NULL,
    nav              DOUBLE PRECISION NOT NULL,
    cash             DOUBLE PRECISION NOT NULL,
    gross_exposure   DOUBLE PRECISION NOT NULL,
    net_exposure     DOUBLE PRECISION NOT NULL,
    drawdown         DOUBLE PRECISION DEFAULT 0,
    daily_pnl        DOUBLE PRECISION DEFAULT 0,
    peak_nav         DOUBLE PRECISION DEFAULT 0,
    n_positions      INTEGER          DEFAULT 0
);

SELECT create_hypertable('portfolio_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);
"""


# ── Storage ────────────────────────────────────────────────────────────

class ExecutionStorage(DatabaseManager):
    """Execution-layer persistence on top of the Phase 1 database manager."""

    def setup_execution_schema(self) -> None:
        """Run DDL, tolerating TimescaleDB-only statements on sqlite/postgres-plain."""
        logger.info("Setting up execution schema…")
        engine = self.engine
        for stmt in EXECUTION_SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            with engine.connect() as conn:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    msg = str(e).lower()
                    if "already a hypertable" in msg:
                        continue
                    if "create_hypertable" in msg or "no such function" in msg \
                       or "does not exist" in msg and "create_hypertable" in stmt.lower():
                        # Non-Timescale backend; skip hypertable DDL silently.
                        continue
                    if stmt.strip().lower().startswith("select create_hypertable"):
                        continue
                    logger.warning("Execution schema statement warning: %s", e)

    # ── Orders ─────────────────────────────────────────────────────────

    def insert_order(self, order: Order) -> None:
        row = self._order_row(order)
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO orders (
                    order_id, timestamp, symbol, side, order_type,
                    quantity, filled_qty, status, algo, signal_family,
                    meta_prob, limit_price, parent_order_id,
                    created_at, updated_at
                ) VALUES (
                    :order_id, :timestamp, :symbol, :side, :order_type,
                    :quantity, :filled_qty, :status, :algo, :signal_family,
                    :meta_prob, :limit_price, :parent_order_id,
                    :created_at, :updated_at
                )
            """), row)
            conn.commit()

    def update_order(self, order: Order) -> None:
        row = self._order_row(order)
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE orders SET
                    filled_qty   = :filled_qty,
                    status       = :status,
                    updated_at   = :updated_at
                WHERE order_id = :order_id
            """), row)
            conn.commit()

    @staticmethod
    def _order_row(order: Order) -> dict:
        return {
            "order_id": order.order_id,
            "timestamp": order.timestamp,
            "symbol": order.symbol,
            "side": order.side,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "filled_qty": order.filled_quantity,
            "status": order.status.value,
            "algo": order.execution_algo.value,
            "signal_family": order.signal_family,
            "meta_prob": order.meta_label_prob,
            "limit_price": order.limit_price,
            "parent_order_id": order.parent_order_id,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
        }

    def get_orders(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        query = text("""
            SELECT * FROM orders
            WHERE symbol = :symbol AND timestamp BETWEEN :start AND :end
            ORDER BY timestamp ASC
        """)
        return pd.read_sql(query, self.engine,
                           params={"symbol": symbol, "start": start, "end": end})

    # ── Fills ──────────────────────────────────────────────────────────

    def insert_fill(self, fill: Fill) -> None:
        row = {
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "timestamp": fill.timestamp,
            "price": fill.price,
            "quantity": fill.quantity,
            "commission": fill.commission,
            "exchange": fill.exchange,
            "is_maker": 1 if fill.is_maker else 0,
        }
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO fills (fill_id, order_id, timestamp, price,
                                   quantity, commission, exchange, is_maker)
                VALUES (:fill_id, :order_id, :timestamp, :price,
                        :quantity, :commission, :exchange, :is_maker)
            """), row)
            conn.commit()

    def get_fills(self, order_id: str) -> pd.DataFrame:
        query = text("SELECT * FROM fills WHERE order_id = :oid ORDER BY timestamp ASC")
        return pd.read_sql(query, self.engine, params={"oid": order_id})

    # ── TCA ────────────────────────────────────────────────────────────

    def insert_tca(self, result: TCAResult, *, timestamp: datetime | None = None) -> None:
        ts = timestamp or datetime.utcnow()
        row = {
            "order_id": result.order_id,
            "timestamp": ts,
            "symbol": result.symbol,
            "side": result.side,
            "arrival_price": result.arrival_price,
            "exec_price": result.execution_price,
            "slippage_bps": result.slippage_bps,
            "impact_bps": result.market_impact_bps,
            "timing_bps": result.timing_cost_bps,
            "total_bps": result.total_cost_bps,
            "commission": result.commission,
            "algo": result.algo_used,
            "duration_seconds": result.execution_duration_seconds,
            "fill_rate": result.fill_rate,
            "twap_bench_bps": result.benchmark_vs_twap_bps,
            "vwap_bench_bps": result.benchmark_vs_vwap_bps,
        }
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO tca_results (
                    order_id, timestamp, symbol, side,
                    arrival_price, exec_price,
                    slippage_bps, impact_bps, timing_bps, total_bps,
                    commission, algo, duration_seconds, fill_rate,
                    twap_bench_bps, vwap_bench_bps
                ) VALUES (
                    :order_id, :timestamp, :symbol, :side,
                    :arrival_price, :exec_price,
                    :slippage_bps, :impact_bps, :timing_bps, :total_bps,
                    :commission, :algo, :duration_seconds, :fill_rate,
                    :twap_bench_bps, :vwap_bench_bps
                )
            """), row)
            conn.commit()

    def get_tca_history(
        self, start: datetime, end: datetime, symbol: str | None = None
    ) -> pd.DataFrame:
        params: dict[str, Any]
        if symbol is None:
            query = text("""
                SELECT * FROM tca_results
                WHERE timestamp BETWEEN :start AND :end
                ORDER BY timestamp ASC
            """)
            params = {"start": start, "end": end}
        else:
            query = text("""
                SELECT * FROM tca_results
                WHERE timestamp BETWEEN :start AND :end AND symbol = :symbol
                ORDER BY timestamp ASC
            """)
            params = {"start": start, "end": end, "symbol": symbol}
        return pd.read_sql(query, self.engine, params=params)

    # ── Portfolio snapshots ────────────────────────────────────────────

    def insert_portfolio_snapshot(
        self, state: PortfolioState, *, timestamp: datetime | None = None
    ) -> None:
        ts = timestamp or datetime.utcnow()
        row = {
            "timestamp": ts,
            "nav": state.nav,
            "cash": state.cash,
            "gross_exposure": state.gross_exposure,
            "net_exposure": state.net_exposure,
            "drawdown": state.drawdown,
            "daily_pnl": state.daily_pnl,
            "peak_nav": state.peak_nav,
            "n_positions": state.position_count,
        }
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO portfolio_snapshots (
                    timestamp, nav, cash, gross_exposure, net_exposure,
                    drawdown, daily_pnl, peak_nav, n_positions
                ) VALUES (
                    :timestamp, :nav, :cash, :gross_exposure, :net_exposure,
                    :drawdown, :daily_pnl, :peak_nav, :n_positions
                )
            """), row)
            conn.commit()

    def get_portfolio_history(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = text("""
            SELECT * FROM portfolio_snapshots
            WHERE timestamp BETWEEN :start AND :end
            ORDER BY timestamp ASC
        """)
        return pd.read_sql(query, self.engine, params={"start": start, "end": end})
