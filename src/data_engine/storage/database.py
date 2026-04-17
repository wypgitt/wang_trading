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
from sqlalchemy import create_engine, text
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
    bar_duration_seconds DOUBLE PRECISION DEFAULT 0
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

-- Triple-barrier labels (Phase 3 / design doc §11.2)
CREATE TABLE IF NOT EXISTS labels (
    event_timestamp     TIMESTAMPTZ NOT NULL,
    symbol              TEXT        NOT NULL,
    side                SMALLINT    NOT NULL,
    volatility          DOUBLE PRECISION,
    vertical_barrier    TIMESTAMPTZ,
    upper_barrier       DOUBLE PRECISION,
    lower_barrier       DOUBLE PRECISION,
    exit_timestamp      TIMESTAMPTZ,
    barrier_touched     TEXT,
    return_pct          DOUBLE PRECISION,
    label               SMALLINT,
    holding_period_bars INTEGER,
    sample_weight       DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('labels', 'event_timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_labels_symbol_time ON labels (symbol, event_timestamp DESC);

-- Persisted meta-labeler probabilities (Phase 3 / design doc §11.2)
CREATE TABLE IF NOT EXISTS meta_labels (
    event_timestamp  TIMESTAMPTZ NOT NULL,
    symbol           TEXT        NOT NULL,
    signal_family    TEXT        NOT NULL,
    meta_prob        DOUBLE PRECISION NOT NULL,
    calibrated_prob  DOUBLE PRECISION,
    model_version    TEXT        NOT NULL,
    feature_hash     TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('meta_labels', 'event_timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_meta_labels_symbol_time ON meta_labels (symbol, event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_meta_labels_model_version ON meta_labels (model_version);

-- Position-state snapshots (design doc §11.2)
CREATE TABLE IF NOT EXISTS positions_history (
    timestamp          TIMESTAMPTZ NOT NULL,
    symbol             TEXT        NOT NULL,
    side               SMALLINT    NOT NULL,
    quantity           DOUBLE PRECISION NOT NULL,
    avg_entry_price    DOUBLE PRECISION NOT NULL,
    current_price      DOUBLE PRECISION NOT NULL,
    unrealized_pnl     DOUBLE PRECISION,
    realized_pnl       DOUBLE PRECISION,
    signal_family      TEXT,
    entry_timestamp    TIMESTAMPTZ,
    stop_loss          DOUBLE PRECISION,
    take_profit        DOUBLE PRECISION,
    vertical_barrier   TIMESTAMPTZ
);
SELECT create_hypertable('positions_history', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_pos_hist_symbol_time ON positions_history (symbol, timestamp DESC);
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
                "bar_duration_seconds": b.bar_duration_seconds,
            }
            for b in bars
        ]

        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO bars (timestamp, open_time, symbol, bar_type,
                        open, high, low, close, volume, dollar_volume, tick_count,
                        buy_volume, sell_volume, buy_ticks, sell_ticks,
                        imbalance, threshold, vwap, bar_duration_seconds)
                    VALUES (:timestamp, :open_time, :symbol, :bar_type,
                        :open, :high, :low, :close, :volume, :dollar_volume, :tick_count,
                        :buy_volume, :sell_volume, :buy_ticks, :sell_ticks,
                        :imbalance, :threshold, :vwap, :bar_duration_seconds)
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

    # ── Labels / meta-labels / positions (C1) ──────────────────────────

    _LABEL_COLUMNS: tuple[str, ...] = (
        "event_timestamp", "symbol", "side", "volatility",
        "vertical_barrier", "upper_barrier", "lower_barrier",
        "exit_timestamp", "barrier_touched", "return_pct",
        "label", "holding_period_bars", "sample_weight",
    )

    async def insert_labels(self, labels_df: pd.DataFrame) -> int:
        """Bulk-insert a labels DataFrame. Returns rows inserted."""
        if labels_df is None or len(labels_df) == 0:
            return 0
        rows: list[dict] = []
        for _, row in labels_df.iterrows():
            payload = {col: row.get(col) for col in self._LABEL_COLUMNS}
            rows.append(payload)
        cols = ", ".join(self._LABEL_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in self._LABEL_COLUMNS)
        with self.engine.connect() as conn:
            conn.execute(
                text(f"INSERT INTO labels ({cols}) VALUES ({placeholders})"),
                rows,
            )
            conn.commit()
        return len(rows)

    async def insert_meta_label(
        self,
        timestamp: datetime,
        symbol: str,
        signal_family: str,
        meta_prob: float,
        calibrated_prob: Optional[float] = None,
        model_version: str = "",
        feature_hash: Optional[str] = None,
    ) -> None:
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO meta_labels (
                        event_timestamp, symbol, signal_family, meta_prob,
                        calibrated_prob, model_version, feature_hash
                    ) VALUES (
                        :ts, :symbol, :family, :prob, :cal_prob,
                        :model_version, :feature_hash
                    )
                """),
                {
                    "ts": timestamp, "symbol": symbol, "family": signal_family,
                    "prob": float(meta_prob),
                    "cal_prob": None if calibrated_prob is None else float(calibrated_prob),
                    "model_version": model_version,
                    "feature_hash": feature_hash,
                },
            )
            conn.commit()

    async def insert_positions_snapshot(
        self, timestamp: datetime, positions: dict,
    ) -> int:
        """Snapshot a ``{symbol: Position}`` mapping into positions_history."""
        if not positions:
            return 0
        rows = []
        for symbol, pos in positions.items():
            rows.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "side": getattr(pos, "side", 0),
                "quantity": float(getattr(pos, "quantity", 0.0)),
                "avg_entry_price": float(getattr(pos, "avg_entry_price", 0.0)),
                "current_price": float(getattr(pos, "current_price", 0.0)),
                "unrealized_pnl": float(getattr(pos, "unrealized_pnl", 0.0) or 0.0),
                "realized_pnl": float(getattr(pos, "realized_pnl", 0.0) or 0.0),
                "signal_family": getattr(pos, "signal_family", "") or "",
                "entry_timestamp": getattr(pos, "entry_timestamp", None),
                "stop_loss": getattr(pos, "stop_loss", None),
                "take_profit": getattr(pos, "take_profit", None),
                "vertical_barrier": getattr(pos, "vertical_barrier", None),
            })
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO positions_history (
                        timestamp, symbol, side, quantity, avg_entry_price,
                        current_price, unrealized_pnl, realized_pnl,
                        signal_family, entry_timestamp, stop_loss, take_profit,
                        vertical_barrier
                    ) VALUES (
                        :timestamp, :symbol, :side, :quantity, :avg_entry_price,
                        :current_price, :unrealized_pnl, :realized_pnl,
                        :signal_family, :entry_timestamp, :stop_loss, :take_profit,
                        :vertical_barrier
                    )
                """),
                rows,
            )
            conn.commit()
        return len(rows)

    async def get_labels(
        self, symbol: str,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM labels WHERE symbol = :symbol"
        params: dict = {"symbol": symbol}
        if start is not None:
            query += " AND event_timestamp >= :start"
            params["start"] = start
        if end is not None:
            query += " AND event_timestamp <= :end"
            params["end"] = end
        query += " ORDER BY event_timestamp ASC"
        return pd.read_sql(text(query), self.engine, params=params)

    async def get_meta_labels(
        self, symbol: str,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
        model_version: Optional[str] = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM meta_labels WHERE symbol = :symbol"
        params: dict = {"symbol": symbol}
        if start is not None:
            query += " AND event_timestamp >= :start"
            params["start"] = start
        if end is not None:
            query += " AND event_timestamp <= :end"
            params["end"] = end
        if model_version is not None:
            query += " AND model_version = :model_version"
            params["model_version"] = model_version
        query += " ORDER BY event_timestamp ASC"
        return pd.read_sql(text(query), self.engine, params=params)

    async def get_positions_history(
        self, symbol: str,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM positions_history WHERE symbol = :symbol"
        params: dict = {"symbol": symbol}
        if start is not None:
            query += " AND timestamp >= :start"
            params["start"] = start
        if end is not None:
            query += " AND timestamp <= :end"
            params["end"] = end
        query += " ORDER BY timestamp ASC"
        return pd.read_sql(text(query), self.engine, params=params)
