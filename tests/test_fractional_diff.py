"""Tests for Fractional Differentiation (FFD) — AFML Ch. 5."""

import numpy as np
import pandas as pd
import pytest

from src.feature_factory.fractional_diff import (
    find_min_d,
    frac_diff_features,
    frac_diff_ffd,
    get_weights_ffd,
)


class TestGetWeightsFFD:
    def test_first_weight_is_one(self):
        w = get_weights_ffd(d=0.5)
        assert w[0] == 1.0

    def test_weights_negative_after_w0_for_d_in_0_1(self):
        """For 0 < d < 1, AFML's recurrence yields w_0=1 and w_k<0 for all k>=1."""
        for d in [0.1, 0.4, 0.5, 0.7, 0.9]:
            w = get_weights_ffd(d=d, threshold=1e-4)
            assert len(w) > 3
            assert w[0] == 1.0
            assert np.all(w[1:] < 0), f"expected all-negative tail for d={d}, got {w[:6]}"

    def test_w1_equals_negative_d(self):
        """Directly from the recurrence: w_1 = -w_0 * d / 1 = -d."""
        for d in [0.2, 0.55, 0.83]:
            w = get_weights_ffd(d=d, threshold=1e-4)
            np.testing.assert_allclose(w[1], -d)

    def test_weights_decrease_in_magnitude(self):
        """|w_k| decreases monotonically after the first step."""
        w = get_weights_ffd(d=0.3, threshold=1e-4)
        mag = np.abs(w)
        # After w_0, magnitudes should be monotonically non-increasing.
        assert np.all(np.diff(mag[1:]) <= 0)

    def test_threshold_respected(self):
        """All returned weights satisfy |w_k| >= threshold (except the last)."""
        threshold = 1e-4
        w = get_weights_ffd(d=0.6, threshold=threshold)
        assert np.all(np.abs(w) >= threshold)

    def test_d_zero_returns_single_weight(self):
        """d=0 means identity; weights recurrence produces only w_0=1 before truncation."""
        w = get_weights_ffd(d=0.0, threshold=1e-5)
        # w_1 = -1 * (0 - 0) / 1 = 0, so the loop terminates after w_0.
        assert len(w) == 1
        assert w[0] == 1.0

    def test_d_one_returns_two_weights(self):
        """d=1 produces weights [1, -1] before truncation (higher-order terms are 0)."""
        w = get_weights_ffd(d=1.0, threshold=1e-5)
        assert len(w) == 2
        assert w[0] == 1.0
        assert w[1] == -1.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            get_weights_ffd(d=0.5, threshold=0.0)
        with pytest.raises(ValueError):
            get_weights_ffd(d=0.5, threshold=-1.0)


class TestFracDiffFFD:
    def test_d_zero_is_identity(self):
        """d=0: the FFD output equals the input exactly."""
        s = pd.Series(np.arange(1, 51, dtype=float))
        out = frac_diff_ffd(s, d=0.0)
        # d=0 produces weights [1.0], so every point passes through unchanged.
        assert len(out) == len(s)
        np.testing.assert_allclose(out.values, s.values)

    def test_d_one_equals_first_difference(self):
        """d=1: the FFD output equals np.diff of the input."""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=100).cumsum())
        out = frac_diff_ffd(s, d=1.0)
        expected = np.diff(s.values)
        # With weights [1, -1], width=2, so the first value is dropped.
        assert len(out) == len(s) - 1
        np.testing.assert_allclose(out.values, expected)

    def test_output_index_aligned_to_input(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        s = pd.Series(np.arange(200, dtype=float), index=dates, name="price")
        out = frac_diff_ffd(s, d=0.4)
        # Output index must be a subset of input index, and name preserved.
        assert out.index.isin(dates).all()
        assert out.name == "price"

    def test_empty_series(self):
        out = frac_diff_ffd(pd.Series([], dtype=float), d=0.5)
        assert out.empty

    def test_short_series_returns_empty(self):
        """If the series is shorter than the FFD window, the output is empty."""
        w = get_weights_ffd(d=0.3, threshold=1e-5)
        s = pd.Series(np.arange(len(w) - 1, dtype=float))  # one short
        out = frac_diff_ffd(s, d=0.3, threshold=1e-5)
        assert out.empty

    def test_constant_series_produces_finite_output(self):
        """A constant input gives finite output equal to c * sum(weights)."""
        # Use threshold=1e-3 so the FFD window fits within the series length.
        s = pd.Series([5.0] * 200)
        threshold = 1e-3
        out = frac_diff_ffd(s, d=0.5, threshold=threshold)
        assert not out.empty
        assert np.isfinite(out.values).all()
        expected = 5.0 * get_weights_ffd(d=0.5, threshold=threshold).sum()
        np.testing.assert_allclose(out.values, expected, rtol=1e-9)

    def test_nan_in_window_drops_from_output(self):
        """NaNs in the input propagate into the window and are dropped from output."""
        s = pd.Series(np.arange(50, dtype=float))
        s.iloc[10] = np.nan
        out = frac_diff_ffd(s, d=0.4)
        # Output must be finite and must not include indices whose window
        # contained the NaN.
        assert np.isfinite(out.values).all()
        assert 10 not in out.index

    def test_increasing_d_reduces_unit_root_pvalue(self):
        """Higher d should move a random walk toward stationarity (lower ADF p-value)."""
        from statsmodels.tsa.stattools import adfuller

        rng = np.random.default_rng(42)
        rw = pd.Series(rng.normal(size=2000).cumsum())
        # Use a looser weight threshold so the FFD window fits inside the series
        # even for low d (where weights decay slowly as ~k^-(d+1)).
        threshold = 1e-3

        pvals = {}
        for d in [0.1, 0.5, 1.0]:
            diffed = frac_diff_ffd(rw, d=d, threshold=threshold)
            assert len(diffed) > 100
            pvals[d] = adfuller(diffed.values, autolag="AIC")[1]

        # p-value should be monotonically (weakly) non-increasing in d.
        assert pvals[0.1] >= pvals[0.5] >= pvals[1.0]
        # And d=1.0 should clearly reject (returns of a random walk are stationary).
        assert pvals[1.0] < 0.05


class TestFindMinD:
    def test_random_walk_needs_nonzero_d(self):
        """A random walk is non-stationary; min d must be strictly positive."""
        rng = np.random.default_rng(7)
        rw = pd.Series(rng.normal(size=1500).cumsum())
        d = find_min_d(rw, step=0.1, threshold=1e-3)
        # The exact d depends on sample size and seed, but it must be well
        # above zero. White noise (d=0) would not be found stationary here
        # because the series *is* integrated of order 1.
        assert d >= 0.2
        assert d <= 1.0

    def test_white_noise_needs_low_d(self):
        """White noise is already stationary; min d should be 0.0."""
        rng = np.random.default_rng(11)
        wn = pd.Series(rng.normal(size=1500))
        d = find_min_d(wn, step=0.1)
        assert d == 0.0

    def test_returns_upper_bound_on_failure(self):
        """A pathological series that never passes ADF returns 1.0 as fallback."""
        # Constant series: ADF is undefined; find_min_d must fall back to 1.0.
        s = pd.Series([1.0] * 200)
        d = find_min_d(s, step=0.25)
        assert d == 1.0

    def test_invalid_step_raises(self):
        s = pd.Series(np.arange(100, dtype=float))
        with pytest.raises(ValueError):
            find_min_d(s, step=0.0)

    def test_invalid_range_raises(self):
        s = pd.Series(np.arange(100, dtype=float))
        with pytest.raises(ValueError):
            find_min_d(s, d_range=(1.0, 0.5))


class TestFracDiffFeatures:
    def test_returns_dataframe_and_d_map(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "price": rng.normal(size=800).cumsum(),
            "vol": rng.normal(loc=10, scale=1, size=800),  # stationary
        })
        out, d_map = frac_diff_features(df, p_value=0.05)

        assert set(d_map.keys()) == {"price", "vol"}
        assert set(out.columns) == {"price", "vol"}
        # Stationary series should be left alone (d = 0).
        assert d_map["vol"] == 0.0
        # Random walk column should need a non-trivial d.
        assert d_map["price"] > 0.0
        # All rows in the output are fully populated.
        assert out.notna().all().all()

    def test_column_subset(self):
        df = pd.DataFrame({
            "a": np.arange(200, dtype=float),
            "b": np.arange(200, dtype=float),
        })
        out, d_map = frac_diff_features(df, columns=["a"])
        assert list(d_map.keys()) == ["a"]
        assert list(out.columns) == ["a"]

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(KeyError):
            frac_diff_features(df, columns=["missing"])

    def test_auto_selects_numeric_columns(self):
        df = pd.DataFrame({
            "x": np.arange(300, dtype=float),
            "label": ["a"] * 300,  # non-numeric, should be skipped
        })
        out, d_map = frac_diff_features(df)
        assert "label" not in d_map
        assert "x" in d_map


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
