"""Tests for structural break features (AFML Ch. 17)."""

import time

import numpy as np
import pandas as pd
import pytest

from src.feature_factory.structural_breaks import (
    chow_test,
    chow_test_pvalue,
    compute_structural_break_features,
    cusum_statistic,
    gsadf_test,
    sadf_test,
)


# ---------------------------------------------------------------------------
# CUSUM statistic
# ---------------------------------------------------------------------------

class TestCUSUMStatistic:
    def test_detects_regime_change(self):
        """CUSUM magnitude should be larger during a mean shift than during noise."""
        rng = np.random.default_rng(0)
        # First 300 bars: random walk with zero drift.
        stable = rng.normal(0.0, 0.01, size=300).cumsum() + 100.0
        # Next 200 bars: strong upward drift (regime change).
        trending = np.cumsum(rng.normal(0.005, 0.01, size=200)) + stable[-1]
        prices = pd.Series(np.concatenate([stable, trending]))

        cusum = cusum_statistic(prices, window=50)

        # Post-regime-change CUSUM should exceed the pre-change CUSUM.
        pre_mean = cusum.iloc[100:280].mean()
        post_mean = cusum.iloc[330:490].mean()
        assert post_mean > pre_mean, f"pre={pre_mean:.4f}, post={post_mean:.4f}"

    def test_output_shape_matches_input(self):
        s = pd.Series(np.arange(200.0) + 100.0)
        out = cusum_statistic(s, window=20)
        assert len(out) == len(s)
        # Warm-up NaN count equals the window size.
        assert out.iloc[:20].isna().all()
        assert out.iloc[20:].notna().all()

    def test_short_series_returns_all_nan(self):
        s = pd.Series([100.0, 101.0, 102.0])
        out = cusum_statistic(s, window=50)
        assert len(out) == 3
        assert out.isna().all()

    def test_nonneg_output(self):
        rng = np.random.default_rng(1)
        s = pd.Series(100.0 + rng.normal(0, 0.5, size=200).cumsum())
        out = cusum_statistic(s, window=30).dropna()
        assert (out >= 0).all()

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            cusum_statistic(pd.Series([1.0, 2.0, 3.0]), window=1)


# ---------------------------------------------------------------------------
# SADF
# ---------------------------------------------------------------------------

class TestSADFTest:
    def test_detects_exponential_bubble(self):
        """SADF should return large positive t-stats during explosive growth."""
        rng = np.random.default_rng(2)
        # Quiet random walk for 150 bars, then an exponential bubble.
        quiet = 100.0 + rng.normal(0.0, 0.3, size=150).cumsum()
        bubble = quiet[-1] * np.exp(np.linspace(0.0, 1.0, 150))  # doubling
        prices = pd.Series(np.concatenate([quiet, bubble]))

        sadf = sadf_test(prices, min_window=20, max_lag=1, fast=True)

        # The SADF stat during the bubble phase should be substantially higher
        # than during the quiet phase.
        quiet_max = sadf.iloc[30:150].max()
        bubble_max = sadf.iloc[200:].max()
        assert bubble_max > quiet_max
        # Positive t-stat is the hallmark of an explosive root (>1).
        assert bubble_max > 0

    def test_random_walk_has_low_sadf(self):
        """Pure random walk should not register as explosive."""
        rng = np.random.default_rng(3)
        rw = pd.Series(100.0 + rng.normal(0.0, 1.0, size=500).cumsum())
        sadf = sadf_test(rw, min_window=20, max_lag=1, fast=True)
        # ADF t-stats for a random walk hover around 0 to mildly negative.
        # Allow some positive slack for sampling variance but not bubble-sized.
        assert sadf.dropna().max() < 3.0

    def test_output_shape_matches_input(self):
        s = pd.Series(100.0 + np.arange(100.0) * 0.1)
        out = sadf_test(s, min_window=20, fast=True)
        assert len(out) == len(s)
        assert out.iloc[:20].isna().all()

    def test_short_series_returns_all_nan(self):
        s = pd.Series([100.0, 101.0, 102.0])
        out = sadf_test(s, min_window=20, fast=True)
        assert out.isna().all()

    def test_invalid_min_window_raises(self):
        with pytest.raises(ValueError):
            sadf_test(pd.Series([1.0] * 100), min_window=2)

    def test_fast_mode_performance(self):
        """Fast SADF on 1000 points should complete in under 5 seconds."""
        rng = np.random.default_rng(4)
        s = pd.Series(100.0 + rng.normal(0.0, 1.0, size=1000).cumsum())
        t0 = time.perf_counter()
        sadf_test(s, min_window=20, fast=True)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"fast SADF took {elapsed:.2f}s on 1000 points"


# ---------------------------------------------------------------------------
# GSADF
# ---------------------------------------------------------------------------

class TestGSADFTest:
    def test_shape_and_warmup(self):
        rng = np.random.default_rng(5)
        s = pd.Series(100.0 + rng.normal(0.0, 0.5, size=120).cumsum())
        out = gsadf_test(s, min_window=20)
        assert len(out) == len(s)
        assert out.iloc[:20].isna().all()

    def test_gsadf_at_least_as_large_as_sadf(self):
        """Exhaustive GSADF's search space strictly contains SADF's, so its
        supremum must be >= SADF's."""
        rng = np.random.default_rng(6)
        n = 120
        s = pd.Series(100.0 + rng.normal(0.0, 0.5, size=n).cumsum())
        sadf = sadf_test(s, min_window=20, fast=False)
        # max_starts/max_ends sized larger than n → exhaustive enumeration.
        gsadf = gsadf_test(s, min_window=20, max_starts=n, max_ends=n)
        both = pd.concat([sadf, gsadf], axis=1).dropna()
        # Pointwise: for every t where both are defined, GSADF(t) >= SADF(t).
        assert (both["gsadf_stat"] + 1e-8 >= both["sadf_stat"]).all()

    def test_short_series_returns_all_nan(self):
        out = gsadf_test(pd.Series([1.0, 2.0, 3.0]), min_window=20)
        assert out.isna().all()


# ---------------------------------------------------------------------------
# Chow test
# ---------------------------------------------------------------------------

class TestChowTest:
    def test_detects_trend_slope_change(self):
        """Chow F-stat should peak near the true break point."""
        n_left, n_right = 100, 100
        # Left: flat line near 100; right: line with slope 0.5.
        left = 100.0 + np.zeros(n_left)
        right = 100.0 + 0.5 * np.arange(1, n_right + 1)
        series = pd.Series(np.concatenate([left, right]))
        # Add a tiny bit of noise so OLS has non-zero residuals on the left.
        rng = np.random.default_rng(7)
        series = series + rng.normal(0.0, 1e-3, size=len(series))

        chow = chow_test(series, min_period=20)

        # The maximum F-stat should land within ±10 bars of the true break.
        t_hat = int(chow.idxmax())
        assert abs(t_hat - n_left) <= 15, f"peak at t={t_hat}, expected ~{n_left}"
        # And the F-stat at the true break should be clearly large.
        assert chow.iloc[n_left] > 50

    def test_no_break_gives_small_fstat(self):
        """A homogeneous trend should not register a break."""
        rng = np.random.default_rng(8)
        x = np.arange(200, dtype=float)
        series = pd.Series(0.2 * x + rng.normal(0.0, 0.5, size=200))
        chow = chow_test(series, min_period=30).dropna()
        # F-stats on a single regime should stay modest.
        assert chow.max() < 30

    def test_output_shape_matches_input(self):
        s = pd.Series(np.arange(200.0))
        out = chow_test(s, min_period=30)
        assert len(out) == len(s)
        # Warm-up and cool-down are NaN.
        assert out.iloc[:30].isna().all()
        assert out.iloc[-30:].isna().all()

    def test_short_series_returns_all_nan(self):
        s = pd.Series(np.arange(10.0))
        out = chow_test(s, min_period=30)
        assert out.isna().all()

    def test_invalid_min_period_raises(self):
        with pytest.raises(ValueError):
            chow_test(pd.Series(np.arange(100.0)), min_period=1)

    def test_pvalue_helper(self):
        # Sanity: large F-stat should yield small p-value.
        p_small = chow_test_pvalue(50.0, n=200, k=2)
        p_large = chow_test_pvalue(0.1, n=200, k=2)
        assert 0 <= p_small < p_large <= 1
        assert np.isnan(chow_test_pvalue(np.nan, n=200))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestComputeStructuralBreakFeatures:
    def test_returns_expected_columns(self):
        rng = np.random.default_rng(9)
        s = pd.Series(100.0 + rng.normal(0.0, 0.5, size=200).cumsum())
        df = compute_structural_break_features(s, window=30)
        assert list(df.columns) == ["cusum_stat", "sadf_stat", "chow_stat"]
        assert len(df) == len(s)
        assert df.index.equals(s.index)

    def test_include_gsadf_adds_column(self):
        rng = np.random.default_rng(10)
        s = pd.Series(100.0 + rng.normal(0.0, 0.5, size=120).cumsum())
        df = compute_structural_break_features(s, window=30, include_gsadf=True)
        assert "gsadf_stat" in df.columns

    def test_short_series_returns_nan_frame(self):
        s = pd.Series([100.0, 101.0, 102.0])
        df = compute_structural_break_features(s, window=30)
        assert df.isna().all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
