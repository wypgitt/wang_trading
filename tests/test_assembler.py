"""Tests for the feature assembler."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from src.feature_factory.assembler import FeatureAssembler


def _synthetic_bars(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Generate a bars DataFrame rich enough to exercise every feature block."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    # Moderate-volatility random walk so GARCH, entropy, etc. all see signal.
    returns = rng.normal(0.0, 0.01, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)
    volume = pd.Series(rng.integers(1000, 10000, size=n).astype(float), index=idx)
    # Buy fraction: uniform over [0.2, 0.8] per bar gives the VPIN/OFI features
    # enough variance to be stationary after differencing. A slow sinusoid
    # here would produce near-constant VPIN values that ADF cannot reject.
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
    """Smaller windows so the warmup period leaves >= 50 usable rows."""
    return {
        "structural_breaks": {
            "window": 30, "min_window_sadf": 15, "min_period_chow": 20,
        },
        "entropy": {"window": 50},
        "microstructure": {"window": 30},
        "volatility": {
            "window": 120, "refit_interval": 40,
            "short_window": 5, "long_window": 20, "vvol_window": 20,
        },
        "classical": {
            "rsi_window": 14, "bbw_window": 20, "ret_z_windows": [5, 10, 20],
        },
    }


class TestFeatureAssembler:
    def test_assemble_returns_clean_dataframe(self):
        bars = _synthetic_bars(n=500)
        assembler = FeatureAssembler(config=_tight_config())
        X = assembler.assemble(bars)

        assert not X.empty
        # No NaN, no inf.
        assert np.isfinite(X.to_numpy()).all()

    def test_feature_name_accessors(self):
        bars = _synthetic_bars(n=500)
        assembler = FeatureAssembler(config=_tight_config())
        X = assembler.assemble(bars)
        names = assembler.get_feature_names()
        assert names == list(X.columns)
        assert len(names) > 10  # multi-family block sanity check
        # At least one FFD column was emitted with an optimal d value.
        d_values = assembler.get_optimal_d_values()
        assert "close" in d_values

    def test_all_features_stationary(self):
        """
        After the assembler's post-hoc stationarisation pass, every feature
        should clear a moderate ADF threshold. We use p < 0.10 rather than
        strict 0.05 because the cleaned slice after warmup is only ~300
        rows, where ADF has limited power against slow-moving features
        (e.g. VPIN, which is already a 30-bar rolling mean).
        """
        bars = _synthetic_bars(n=500)
        assembler = FeatureAssembler(config=_tight_config())
        X = assembler.assemble(bars)

        for col in X.columns:
            vals = X[col].to_numpy(dtype=float)
            if np.std(vals) == 0:
                continue  # trivially stationary
            pval = adfuller(vals, autolag="AIC")[1]
            assert pval < 0.10, f"{col} is not stationary (p={pval:.3f})"

    def test_missing_bar_columns_raises(self):
        bars = _synthetic_bars(n=200).drop(columns=["buy_volume"])
        assembler = FeatureAssembler(config=_tight_config())
        with pytest.raises(KeyError):
            assembler.assemble(bars)

    def test_assemble_without_optional_inputs(self):
        bars = _synthetic_bars(n=500)
        assembler = FeatureAssembler(config=_tight_config())
        X = assembler.assemble(bars)  # no sentiment / no on-chain
        # No column should have those prefixes.
        assert not any(c.startswith("sentiment_") for c in X.columns)
        assert not any(c.startswith("net_flow") for c in X.columns)

    def test_assemble_with_sentiment_and_onchain(self):
        bars = _synthetic_bars(n=500)
        idx = bars.index

        # Fake sentiment features aligned to the bars.
        rng = np.random.default_rng(7)
        sentiment = pd.DataFrame(
            {
                "sentiment_score": rng.normal(0.0, 0.1, size=len(idx)),
                "sentiment_mom_1d": rng.normal(0.0, 0.05, size=len(idx)),
                "sentiment_mom_3d": rng.normal(0.0, 0.05, size=len(idx)),
                "article_count": rng.integers(0, 10, size=len(idx)),
            },
            index=idx,
        )
        # Fake on-chain features — sparse daily sampling.
        onchain = pd.DataFrame(
            {
                "net_flow": rng.normal(0.0, 100.0, size=len(idx)),
                "flow_ratio": np.abs(rng.normal(1.0, 0.2, size=len(idx))),
            },
            index=idx,
        )

        assembler = FeatureAssembler(config=_tight_config())
        X = assembler.assemble(
            bars,
            sentiment_scores=sentiment,
            onchain_features=onchain,
        )
        assert "sentiment_score" in X.columns
        assert "net_flow" in X.columns
        assert np.isfinite(X.to_numpy()).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
