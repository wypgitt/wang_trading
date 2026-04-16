"""Tests for entropy features (AFML Ch. 18)."""

import numpy as np
import pandas as pd
import pytest

from src.feature_factory.entropy import (
    _lz76_complexity,
    approx_entropy,
    compute_entropy_features,
    lempel_ziv_entropy,
    sample_entropy,
    shannon_entropy,
)


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_random_higher_than_trending(self):
        """Random returns should produce higher Shannon entropy than a trend."""
        rng = np.random.default_rng(0)
        # Pure random walk (i.i.d. returns) — the return distribution should
        # fill the quantile bins evenly.
        random_prices = pd.Series(100.0 + rng.normal(0.0, 1.0, size=500).cumsum())
        # Strong uniform uptrend with minimal noise — all returns cluster in
        # the top quantile bins, reducing entropy.
        trend = pd.Series(
            100.0 + np.arange(500, dtype=float) * 0.5
            + rng.normal(0.0, 0.001, size=500)
        )

        h_rand = shannon_entropy(random_prices, n_bins=10, window=100).dropna()
        h_trend = shannon_entropy(trend, n_bins=10, window=100).dropna()

        assert h_rand.mean() > h_trend.mean(), (
            f"rand={h_rand.mean():.3f}, trend={h_trend.mean():.3f}"
        )

    def test_output_range(self):
        """Shannon entropy with n_bins=10 must be in [0, log2(10)]."""
        rng = np.random.default_rng(1)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=300).cumsum())
        h = shannon_entropy(s, n_bins=10, window=100).dropna()
        assert (h >= 0).all()
        assert (h <= np.log2(10) + 1e-9).all()

    def test_output_shape(self):
        s = pd.Series(100.0 + np.arange(200.0))
        out = shannon_entropy(s, n_bins=5, window=50)
        assert len(out) == len(s)
        assert out.iloc[:50].isna().all()

    def test_window_larger_than_series(self):
        s = pd.Series(100.0 + np.arange(20.0))
        out = shannon_entropy(s, n_bins=5, window=50)
        assert out.isna().all()

    def test_invalid_params(self):
        s = pd.Series(np.arange(100.0))
        with pytest.raises(ValueError):
            shannon_entropy(s, n_bins=1)
        with pytest.raises(ValueError):
            shannon_entropy(s, window=1)

    def test_nan_tolerant(self):
        """NaN values in the series must not crash the rolling computation."""
        rng = np.random.default_rng(2)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=200).cumsum())
        s.iloc[50:55] = np.nan
        out = shannon_entropy(s, n_bins=10, window=50)
        # We don't demand a specific value, just that some entries are finite.
        assert out.dropna().size > 0


# ---------------------------------------------------------------------------
# Lempel-Ziv
# ---------------------------------------------------------------------------

class TestLempelZivEntropy:
    def test_periodic_pattern_has_low_complexity(self):
        """A perfectly periodic binary string should have very low LZ complexity."""
        c_periodic = _lz76_complexity("01" * 50)
        c_random = _lz76_complexity("1100101011011100101000111100110101")
        # The periodic c grows as ~sqrt(n); the random string's grows ~linearly.
        assert c_periodic < c_random

    def test_empty_string(self):
        assert _lz76_complexity("") == 0

    def test_single_char(self):
        assert _lz76_complexity("0") == 1

    def test_entropy_lower_for_trending_than_random(self):
        """A strong trend produces almost all-1 bits → low LZ entropy."""
        rng = np.random.default_rng(3)
        trend = pd.Series(100.0 + np.arange(400, dtype=float) * 0.5)  # all positive returns
        random_prices = pd.Series(100.0 + rng.normal(0.0, 1.0, size=400).cumsum())

        h_trend = lempel_ziv_entropy(trend, window=100).dropna()
        h_rand = lempel_ziv_entropy(random_prices, window=100).dropna()
        assert h_trend.mean() < h_rand.mean()

    def test_output_shape(self):
        s = pd.Series(100.0 + np.arange(200.0))
        out = lempel_ziv_entropy(s, window=50)
        assert len(out) == len(s)
        assert out.iloc[:50].isna().all()

    def test_window_larger_than_series(self):
        s = pd.Series(np.arange(10.0))
        out = lempel_ziv_entropy(s, window=50)
        assert out.isna().all()

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            lempel_ziv_entropy(pd.Series([1.0, 2.0]), window=1)


# ---------------------------------------------------------------------------
# Approximate / Sample Entropy
# ---------------------------------------------------------------------------

class TestApproxEntropy:
    def test_shape_and_warmup(self):
        rng = np.random.default_rng(4)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=200).cumsum())
        out = approx_entropy(s, m=2, window=50)
        assert len(out) == len(s)
        assert out.iloc[:50].isna().all()

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            approx_entropy(pd.Series(np.arange(100.0)), m=0)
        with pytest.raises(ValueError):
            approx_entropy(pd.Series(np.arange(100.0)), m=2, window=3)


class TestSampleEntropy:
    def test_sine_lower_than_noise(self):
        """SampEn on a deterministic sine should be (much) lower than on noise."""
        rng = np.random.default_rng(5)
        t = np.linspace(0.0, 20 * np.pi, 600)
        # Use a "price" that exponentiates a sine wave so log-returns are
        # genuinely sinusoidal (deterministic, periodic, highly regular).
        sine_prices = pd.Series(100.0 * np.exp(0.01 * np.sin(t)))
        noise_prices = pd.Series(100.0 + rng.normal(0.0, 1.0, size=600).cumsum())

        h_sine = sample_entropy(sine_prices, m=2, window=120).dropna()
        h_noise = sample_entropy(noise_prices, m=2, window=120).dropna()

        assert h_sine.mean() < h_noise.mean(), (
            f"sine={h_sine.mean():.3f}, noise={h_noise.mean():.3f}"
        )

    def test_output_finite_or_nan(self):
        rng = np.random.default_rng(6)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=200).cumsum())
        out = sample_entropy(s, m=2, window=80).dropna()
        # SampEn outputs are non-negative (it's -log of a probability ratio <=1).
        assert (out >= 0).all()

    def test_short_series_all_nan(self):
        s = pd.Series(np.arange(5.0))
        out = sample_entropy(s, m=2, window=50)
        assert out.isna().all()

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            sample_entropy(pd.Series(np.arange(100.0)), m=0)
        with pytest.raises(ValueError):
            sample_entropy(pd.Series(np.arange(100.0)), m=2, window=3)

    def test_nan_tolerant(self):
        rng = np.random.default_rng(7)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=200).cumsum())
        s.iloc[30:40] = np.nan
        out = sample_entropy(s, m=2, window=80)
        assert out.dropna().size >= 0  # must not raise


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestComputeEntropyFeatures:
    def test_default_columns(self):
        rng = np.random.default_rng(8)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=300).cumsum())
        df = compute_entropy_features(s, window=80)
        assert list(df.columns) == ["shannon_entropy", "lz_entropy", "sample_entropy"]
        assert len(df) == len(s)

    def test_include_apen(self):
        rng = np.random.default_rng(9)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=200).cumsum())
        df = compute_entropy_features(s, window=80, include_apen=True)
        assert "approx_entropy" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
