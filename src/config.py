"""
Configuration management.

Loads settings from YAML config file with environment variable interpolation.
"""

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from loguru import logger


# ── Config Models ──

class AlpacaConfig(BaseModel):
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    feed: str = "iex"


class PolygonConfig(BaseModel):
    api_key: str = ""


class BinanceConfig(BaseModel):
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True


class IBKRConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


class DataSourcesConfig(BaseModel):
    alpaca: AlpacaConfig = AlpacaConfig()
    polygon: PolygonConfig = PolygonConfig()
    binance: BinanceConfig = BinanceConfig()
    ibkr: IBKRConfig = IBKRConfig()


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "quantsystem"
    user: str = "quant"
    password: str = ""
    tick_chunk_interval: str = "1 day"
    bar_chunk_interval: str = "7 days"

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class FeatureStoreConfig(BaseModel):
    backend: str = "parquet"
    local_path: str = "./data/features"
    gcs_bucket: str = ""


class AssetBarConfig(BaseModel):
    primary_type: str = "tib"
    secondary_type: str = "dollar"
    tick_bar_size: int = 500
    volume_bar_size: float = 50000
    dollar_bar_size: float = 1_000_000
    imbalance_ewma_span: int = 100


class BarsConfig(BaseModel):
    equities: AssetBarConfig = AssetBarConfig()
    crypto: AssetBarConfig = AssetBarConfig(
        tick_bar_size=200,
        volume_bar_size=10.0,
        dollar_bar_size=500_000,
        imbalance_ewma_span=50,
    )
    futures: AssetBarConfig = AssetBarConfig(
        primary_type="dollar",
        secondary_type="tib",
        dollar_bar_size=2_000_000,
    )


class InstrumentsConfig(BaseModel):
    equities: dict = Field(default_factory=lambda: {
        "universe": "sp500",
        "custom_symbols": [],
        "test_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "SPY"],
    })
    crypto: dict = Field(default_factory=lambda: {
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "exchanges": ["binance", "coinbase"],
    })
    futures: dict = Field(default_factory=lambda: {
        "symbols": ["ES", "NQ", "CL", "GC"],
    })


class ValidationConfig(BaseModel):
    min_ticks_per_bar: int = 10
    max_bar_duration_seconds: int = 86400
    outlier_std_threshold: float = 10.0
    gap_threshold_seconds: int = 300


class CUSUMConfig(BaseModel):
    threshold_multiplier: float = 1.5
    min_events_per_day: int = 2
    max_events_per_day: int = 50


class SystemConfig(BaseModel):
    name: str = "quant-system"
    environment: str = "development"
    log_level: str = "INFO"


class Settings(BaseModel):
    system: SystemConfig = SystemConfig()
    data_sources: DataSourcesConfig = DataSourcesConfig()
    database: DatabaseConfig = DatabaseConfig()
    feature_store: FeatureStoreConfig = FeatureStoreConfig()
    bars: BarsConfig = BarsConfig()
    instruments: InstrumentsConfig = InstrumentsConfig()
    validation: ValidationConfig = ValidationConfig()
    cusum: CUSUMConfig = CUSUMConfig()


# ── Loader ──

def _interpolate_env_vars(obj):
    """Recursively replace ${VAR_NAME} with environment variable values."""
    if isinstance(obj, str):
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, obj)
        for var_name in matches:
            env_value = os.environ.get(var_name, "")
            obj = obj.replace(f"${{{var_name}}}", env_value)
        return obj
    elif isinstance(obj, dict):
        return {k: _interpolate_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_interpolate_env_vars(item) for item in obj]
    return obj


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from YAML file with env var interpolation.

    Searches for config in this order:
    1. Explicit path argument
    2. QUANT_CONFIG environment variable
    3. config/settings.yaml (relative to project root)
    4. Default values
    """
    if config_path is None:
        config_path = os.environ.get("QUANT_CONFIG")

    if config_path is None:
        # Search relative to this file's location
        project_root = Path(__file__).parent.parent.parent
        candidate = project_root / "config" / "settings.yaml"
        if candidate.exists():
            config_path = str(candidate)

    if config_path and Path(config_path).exists():
        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if raw:
            raw = _interpolate_env_vars(raw)
            return Settings(**raw)
    else:
        logger.warning("No config file found, using defaults")

    return Settings()


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
