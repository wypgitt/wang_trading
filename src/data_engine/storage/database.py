"""
Database storage layer for TimescaleDB.

Creates hypertables for ticks and bars with proper indexing.
Provides async insert methods for the ingestion pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import (
    create_engine, text, Column, Float, Integer, String, DateTime, Enum,
    MetaData, Table, Index,
)
from sqlalchemy.engine import Engine

from src.data_engine.models import Tick, Bar, BarType, Side


# ── Schema Definition ──

SCHEMA_SQL = """
-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Raw ticks table
CREATE TABLE IF NOT EXISTS raw_ticks (
    timestamp    TIMESTAMPTZ NOT NULL,
    symbol       TEXT        NOT NULL,
    price        DOUBLE PRECISION NOT NULL,
    volume       DOUBLE PRECISION NOT NULL,
    side         SMALLINT    DEFAULT 0,   -- 1=buy, -1=sell, 0=unknown
    exchange     TEXT        DEFAULT '',
    trade_id     TEXT        DEFAULT ''
);

-- Convert to hypertable (time-partitioned)
SELECT create_hypertable('raw_ticks', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time
    ON raw_ticks (symbol, timestamp DESC);

-- Bars table
CREATE TABLE IF NOT EXISTS bars (
    timestamp         TIMESTAMPTZ      NOT NULL,
    open_time         TIMESTAMPTZ      NOT NULL,
    symbol            TEXT             NOT NULL,
    bar_type          TEXT             NOT NULL,
    open              DOUBLE PRECISION NOT NULL,
    high              DOUBLE PRECISION NOT NULL,
    low               DOUBLE PRECISION NOT NULL,
    close             DOUBLE PRECISION NOT NULL,
    volume            DOUBLE PRECISION NOT NULL,
    dollar_volume     DOUBLE PRECISION NOT NULL,
    tick_count        INTEGER          NOT NULL,
    buy_volume        DOUBLE PRECISION DEFAULT 0,
    sell_volume       DOUBLE PRECISION DEFAULT 0,
    buy_ticks         INTEGER          DEFAULT 0,
    sell_ticks        INTEGER          DEFAULT 0,
    imbalance         DOUBLE PRECISION DEFAULT 0,
    threshold         DOUBLE PRECISION DEFAULT 0,
    vwap              DOUBLE PRECISION DEFAULT 0,
    bar_duration_secs DOUBLE PRECISION DEFAULT 0
);

SELECT create_hypertable('bars', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_type_time
    ON bars (symbol, bar_type, timestamp DESC);

-- CUSUM events table
CREATE TABLE IF NOT EXISTS cusum_events (
    timestamp    TIMESTAMPTZ NOT NULL,
    symbol       TEXT        NOT NULL,
    cusum_value  DOUBLE PRECISION NOT NULL,
    threshold    DOUBLE PRECISION NOT NULL,
    direction    SMALLINT    NOT NULL  -- 1=positive break, -1=negative break
);

SELECT create_hypertable('cusum_events', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Features table (Phase 2)
CREATE TABLE IF NOT EXISTS features (
    timestamp    TIMESTAMPTZ NOT NULL,
    symbol       TEXT        NOT NULL,
    feature_name TEXT        NOT NULL,
    value        DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('features', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Signals table (Phase 3)
CREATE TABLE IF NOT EXISTS signals (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    signal_family   TEXT        NOT NULL,
    side            SMALLINT    NOT NULL,
    confidence      DOUBLE PRECISION DEFAULT 0,
    meta_label_prob DOUBLE PRECISION DEFAULT 0,
    bet_size        DOUBLE PRECISION DEFAULT 0
);

SELECT create_hypertable('signals', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);
"""


class DatabaseManager:
    """
    Manages database connections and provides insert/query methods.
    """

    def __init__(self, db_url: str):
        self._url = db_url
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self._url, pool_size=5, max_overflow=10)
        return self._engine

    def setup_schema(self) -> None:
        """Create all tables and hypertables."""
        logger.info("Setting up database schema...")
        with self.engine.connect() as conn:
            # Execute each statement separately (TimescaleDB functions
            # don't work well in multi-statement batches)
            for statement in SCHEMA_SQL.split(";"):
                statement = statement.strip()
                if statement:
                    try:
                        conn.execute(text(statement))
                    except Exception as e:
                        # Ignore "already a hypertable" errors
                        if "already a hypertable" in str(e):
                            continue
                        logger.warning(f"Schema statement warning: {e}")
            conn.commit()
        logger.info("Database schema ready")

    def insert_ticks(self, ticks: list[Tick]) -> int:
        """Bulk insert ticks. Returns number inserted."""
        if not ticks:
            return 0

        rows = [
            {
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "price": t.price,
                "volume": t.volume,
                "side": t.side.value,
                "exchange": t.exchange,
                "trade_id": t.trade_id,
            }
            for t in ticks
        ]

        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO raw_ticks (timestamp, symbol, price, volume, side, exchange, trade_id)
                    VALUES (:timestamp, :symbol, :price, :volume, :side, :exchange, :trade_id)
                """),
                rows,
            )
            conn.commit()

        return len(rows)

    def insert_bars(self, bars: list[Bar]) -> int:
        """Bulk insert bars. Returns number inserted."""
        if not bars:
            return 0

        rows = [
            {
                "timestamp": b.timestamp,
                "open_time": b.open_time,
                "symbol": b.symbol,
                "bar_type": b.bar_type.value,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "dollar_volume": b.dollar_volume,
                "tick_count": b.tick_count,
                "buy_volume": b.buy_volume,
                "sell_volume": b.sell_volume,
                "buy_ticks": b.buy_ticks,
                "sell_ticks": b.sell_ticks,
                "imbalance": b.imbalance,
                "threshold": b.threshold,
                "vwap": b.vwap,
                "bar_duration_secs": b.bar_duration_seconds,
            }
            for b in bars
        ]

        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO bars (timestamp, open_time, symbol, bar_type,
                        open, high, low, close, volume, dollar_volume, tick_count,
                        buy_volume, sell_volume, buy_ticks, sell_ticks,
                        imbalance, threshold, vwap, bar_duration_secs)
                    VALUES (:timestamp, :open_time, :symbol, :bar_type,
                        :open, :high, :low, :close, :volume, :dollar_volume, :tick_count,
                        :buy_volume, :sell_volume, :buy_ticks, :sell_ticks,
                        :imbalance, :threshold, :vwap, :bar_duration_secs)
                """),
                rows,
            )
            conn.commit()

        return len(rows)

    def get_bars(
        self,
        symbol: str,
        bar_type: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10_000,
    ) -> pd.DataFrame:
        """Query bars as a DataFrame."""
        query = """
            SELECT * FROM bars
            WHERE symbol = :symbol AND bar_type = :bar_type
        """
        params: dict = {"symbol": symbol, "bar_type": bar_type}

        if start:
            query += " AND timestamp >= :start"
            params["start"] = start
        if end:
            query += " AND timestamp <= :end"
            params["end"] = end

        query += " ORDER BY timestamp ASC LIMIT :limit"
        params["limit"] = limit

        return pd.read_sql(text(query), self.engine, params=params)

    def get_latest_tick_time(self, symbol: str) -> Optional[datetime]:
        """Get the most recent tick timestamp for a symbol."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(timestamp) FROM raw_ticks WHERE symbol = :symbol"),
                {"symbol": symbol},
            ).scalar()
        return result

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None
