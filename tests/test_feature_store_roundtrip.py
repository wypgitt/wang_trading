"""Round-trip test: FeatureAssembler output survives parquet serialization.

Covers the Phase-2 audit checklist requirement that a full assembled
feature matrix can be persisted to Parquet and reloaded byte-identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_engine.storage.feature_store import FeatureStore
from src.feature_factory.assembler import FeatureAssembler


def _synthetic_bars(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n))), index=idx)
    volume = pd.Series(rng.integers(1000, 10_000, size=n).astype(float), index=idx)
    buy_frac = rng.uniform(0.2, 0.8, size=n)
    buy = volume * buy_frac
    sell = volume - buy
    return pd.DataFrame({
        "close": close,
        "volume": volume,
        "dollar_volume": close * volume,
        "buy_volume": buy,
        "sell_volume": sell,
        "tick_count": pd.Series(rng.integers(50, 500, size=n), index=idx),
        "bar_duration_seconds": pd.Series(np.full(n, 60.0), index=idx),
    })


def _tight_config() -> dict:
    return {
        "structural_breaks": {"window": 30, "min_window_sadf": 15, "min_period_chow": 20},
        "entropy": {"window": 50},
        "microstructure": {"window": 30},
        "volatility": {
            "window": 120, "refit_interval": 40,
            "short_window": 5, "long_window": 20, "vvol_window": 20,
        },
        "classical": {"rsi_window": 14, "bbw_window": 20, "ret_z_windows": [5, 10, 20]},
    }


def test_feature_matrix_parquet_roundtrip(tmp_path: Path):
    """Assemble → save to Parquet → reload → verify identical."""
    bars = _synthetic_bars(n=400, seed=42)
    assembler = FeatureAssembler(config=_tight_config())
    matrix = assembler.assemble(bars)
    assert not matrix.empty
    assert matrix.shape[1] > 10  # non-trivial matrix

    store = FeatureStore(base_path=tmp_path)
    matrix_to_save = matrix.reset_index().rename(columns={"index": "timestamp"})
    store.save_features(matrix_to_save, symbol="TEST", version="phase2_audit")

    reloaded = store.load_features(symbol="TEST", version="phase2_audit")
    assert not reloaded.empty
    assert list(reloaded.columns) == list(matrix_to_save.columns)
    assert len(reloaded) == len(matrix_to_save)

    # Numeric columns must round-trip to the same float64 values (Parquet
    # preserves precision). Compare the frame-wide equivalence rather than
    # a single column.
    numeric_cols = [c for c in matrix_to_save.columns if c != "timestamp"]
    np.testing.assert_allclose(
        reloaded[numeric_cols].to_numpy(dtype=float),
        matrix_to_save[numeric_cols].to_numpy(dtype=float),
        rtol=0, atol=0,
    )


def test_feature_matrix_parquet_preserves_index_values(tmp_path: Path):
    """After round-trip, the timestamp column values match exactly."""
    bars = _synthetic_bars(n=200, seed=1)
    assembler = FeatureAssembler(config=_tight_config())
    matrix = assembler.assemble(bars)
    if matrix.empty:
        pytest.skip("short series produced empty feature matrix")

    store = FeatureStore(base_path=tmp_path)
    df = matrix.reset_index().rename(columns={"index": "timestamp"})
    store.save_features(df, symbol="IDX", version="v1")

    back = store.load_features(symbol="IDX", version="v1")
    # Pandas parquet normalises tz to UTC; just check the instant equality.
    a = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC") if df["timestamp"].dt.tz else pd.to_datetime(df["timestamp"])
    b = pd.to_datetime(back["timestamp"])
    if b.dt.tz is not None:
        b = b.dt.tz_convert("UTC")
    if a.dt.tz is not None:
        a = a.dt.tz_convert("UTC")
    # Comparing Timestamps via their numeric view avoids tz-naiveté edge cases.
    np.testing.assert_array_equal(
        a.astype("int64").to_numpy(),
        b.astype("int64").to_numpy(),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
