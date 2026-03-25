"""Tests for config loading and feature store."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import Settings, load_settings, _interpolate_env_vars
from src.data_engine.storage.feature_store import FeatureStore


class TestConfig:
    def test_default_settings(self):
        settings = Settings()
        assert settings.system.name == "quant-system"
        assert settings.database.port == 5432
        assert settings.bars.equities.primary_type == "tib"

    def test_env_var_interpolation(self):
        os.environ["TEST_KEY_123"] = "my_secret"
        result = _interpolate_env_vars("key=${TEST_KEY_123}")
        assert result == "key=my_secret"
        del os.environ["TEST_KEY_123"]

    def test_nested_interpolation(self):
        os.environ["TEST_NESTED"] = "value"
        data = {"outer": {"inner": "${TEST_NESTED}"}}
        result = _interpolate_env_vars(data)
        assert result["outer"]["inner"] == "value"
        del os.environ["TEST_NESTED"]

    def test_missing_env_var_becomes_empty(self):
        result = _interpolate_env_vars("${DEFINITELY_NOT_SET_XYZ}")
        assert result == ""

    def test_load_settings_no_file_returns_defaults(self):
        settings = load_settings("/nonexistent/path.yaml")
        assert settings.system.name == "quant-system"

    def test_database_url_property(self):
        settings = Settings()
        settings.database.user = "testuser"
        settings.database.password = "testpass"
        settings.database.host = "localhost"
        settings.database.port = 5432
        settings.database.name = "testdb"
        assert "testuser:testpass@localhost:5432/testdb" in settings.database.url


class TestFeatureStore:
    def test_save_and_load_bars(self, tmp_path):
        store = FeatureStore(base_path=str(tmp_path))

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "symbol": "AAPL",
            "close": range(100, 110),
            "volume": [1000] * 10,
        })

        store.save_bars(df, "AAPL", "tib")
        loaded = store.load_bars("AAPL", "tib")

        assert len(loaded) == 10
        assert loaded["close"].iloc[0] == 100

    def test_empty_save(self, tmp_path):
        store = FeatureStore(base_path=str(tmp_path))
        store.save_bars(pd.DataFrame(), "AAPL", "tib")
        loaded = store.load_bars("AAPL", "tib")
        assert loaded.empty

    def test_load_nonexistent(self, tmp_path):
        store = FeatureStore(base_path=str(tmp_path))
        loaded = store.load_bars("NOSYMBOL", "nobar")
        assert loaded.empty

    def test_list_symbols(self, tmp_path):
        store = FeatureStore(base_path=str(tmp_path))

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "symbol": "AAPL",
            "close": range(5),
        })
        store.save_bars(df, "AAPL", "tib")
        store.save_bars(df, "MSFT", "tib")

        symbols = store.list_symbols("bars")
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_save_and_load_features(self, tmp_path):
        store = FeatureStore(base_path=str(tmp_path))

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "ffd_close": [0.1, 0.2, -0.1, 0.3, -0.2],
            "entropy": [0.5, 0.6, 0.4, 0.7, 0.5],
        })

        store.save_features(df, "AAPL", "v1")
        loaded = store.load_features("AAPL", "v1")
        assert len(loaded) == 5
        assert "ffd_close" in loaded.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
