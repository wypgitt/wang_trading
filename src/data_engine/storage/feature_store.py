"""
Feature Store (Parquet-based)

Stores versioned feature matrices as Parquet files, organized by
symbol and date. Supports both local filesystem and GCS backends.

This is the system's source of truth for reproducible experiments:
every model training run references a specific version of the
feature store, tracked by DVC.

Phase 1 sets up the infrastructure; Phase 2 populates it with
actual features (FFD, entropy, microstructure, etc.).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


class FeatureStore:
    """
    Versioned feature storage backed by Parquet files.

    Directory structure:
        {base_path}/
            bars/
                {symbol}/
                    {bar_type}/
                        {YYYY-MM-DD}.parquet
            features/
                {symbol}/
                    {version}/
                        {YYYY-MM-DD}.parquet
    """

    def __init__(self, base_path: str = "./data/features"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_bars(
        self,
        df: pd.DataFrame,
        symbol: str,
        bar_type: str,
    ) -> Path:
        """
        Save bars DataFrame to partitioned Parquet files.

        Partitions by date for efficient time-range queries.
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}/{bar_type}, skipping")
            return self.base_path

        dir_path = self.base_path / "bars" / symbol / bar_type
        dir_path.mkdir(parents=True, exist_ok=True)

        # Partition by date
        if "timestamp" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        for date, group in df.groupby("date"):
            file_path = dir_path / f"{date}.parquet"
            group_clean = group.drop(columns=["date"], errors="ignore")

            if file_path.exists():
                # Append to existing file
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, group_clean]).drop_duplicates(
                    subset=["timestamp", "symbol"], keep="last"
                )
                combined.to_parquet(file_path, index=False)
            else:
                group_clean.to_parquet(file_path, index=False)

        logger.info(f"Saved {len(df)} bars for {symbol}/{bar_type}")
        return dir_path

    def load_bars(
        self,
        symbol: str,
        bar_type: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load bars from Parquet files, optionally filtered by date range."""
        dir_path = self.base_path / "bars" / symbol / bar_type

        if not dir_path.exists():
            logger.warning(f"No data for {symbol}/{bar_type}")
            return pd.DataFrame()

        files = sorted(dir_path.glob("*.parquet"))

        if start:
            files = [f for f in files if f.stem >= start.strftime("%Y-%m-%d")]
        if end:
            files = [f for f in files if f.stem <= end.strftime("%Y-%m-%d")]

        if not files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        result = pd.concat(dfs, ignore_index=True)

        if "timestamp" in result.columns:
            result["timestamp"] = pd.to_datetime(result["timestamp"])
            result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Loaded {len(result)} bars for {symbol}/{bar_type}")
        return result

    def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        version: str = "v1",
    ) -> Path:
        """Save a feature matrix (Phase 2+)."""
        dir_path = self.base_path / "features" / symbol / version
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / "features.parquet"
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(df)} feature rows for {symbol}/{version}")
        return file_path

    def load_features(
        self,
        symbol: str,
        version: str = "v1",
    ) -> pd.DataFrame:
        """Load a feature matrix (Phase 2+)."""
        file_path = self.base_path / "features" / symbol / version / "features.parquet"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def list_symbols(self, data_type: str = "bars") -> list[str]:
        """List all symbols with stored data."""
        dir_path = self.base_path / data_type
        if not dir_path.exists():
            return []
        return [d.name for d in dir_path.iterdir() if d.is_dir()]
