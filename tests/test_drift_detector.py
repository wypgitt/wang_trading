"""Tests for FeatureDriftDetector (Phase 5 / P5.10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import FeatureDriftDetector


@pytest.fixture
def baseline():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "f1": rng.normal(0.0, 1.0, size=5000),
        "f2": rng.normal(0.0, 1.0, size=5000),
        "f3": rng.normal(0.0, 1.0, size=5000),
    })
    det = FeatureDriftDetector(n_bins=50)
    det.set_baseline(df)
    return det


class TestNoDrift:
    def test_identical_distribution_not_drifted(self, baseline):
        rng = np.random.default_rng(99)
        cur = pd.DataFrame({
            "f1": rng.normal(0.0, 1.0, size=2000),
            "f2": rng.normal(0.0, 1.0, size=2000),
            "f3": rng.normal(0.0, 1.0, size=2000),
        })
        result = baseline.check_drift(cur, window=1000)
        # KL should be tiny for sampled-from-same distribution
        assert (result["kl_divergence"] < 0.1).all()
        assert not result["drifted"].any()


class TestMeanShift:
    def test_five_sigma_mean_shift_flags_drift(self, baseline):
        rng = np.random.default_rng(0)
        cur = pd.DataFrame({
            "f1": rng.normal(5.0, 1.0, size=500),  # +5σ
            "f2": rng.normal(0.0, 1.0, size=500),
            "f3": rng.normal(0.0, 1.0, size=500),
        })
        drifted = baseline.get_drifted_features(cur)
        assert "f1" in drifted
        assert "f2" not in drifted


class TestVarianceShift:
    def test_doubled_variance_flags_drift(self, baseline):
        rng = np.random.default_rng(1)
        cur = pd.DataFrame({
            "f1": rng.normal(0.0, 1.0, size=500),
            "f2": rng.normal(0.0, np.sqrt(3.0), size=500),  # var ~= 3 (> 2)
            "f3": rng.normal(0.0, 1.0, size=500),
        })
        result = baseline.check_drift(cur, window=500)
        f2_row = result[result["feature"] == "f2"].iloc[0]
        assert f2_row["var_ratio"] > 2.0
        assert f2_row["drifted"]


class TestKL:
    def test_kl_near_zero_for_identical(self, baseline):
        rng = np.random.default_rng(7)
        cur = pd.DataFrame({
            "f1": rng.normal(0.0, 1.0, size=5000),
            "f2": rng.normal(0.0, 1.0, size=5000),
            "f3": rng.normal(0.0, 1.0, size=5000),
        })
        result = baseline.check_drift(cur, window=5000)
        assert (result["kl_divergence"] < 0.05).all()


class TestRecommendAction:
    def test_no_action(self):
        assert FeatureDriftDetector.recommend_action([], 10) == "No action needed"

    def test_below_20_pct_monitor(self):
        msg = FeatureDriftDetector.recommend_action(["a"], 10)
        assert "Monitor" in msg

    def test_between_20_and_50_warning(self):
        msg = FeatureDriftDetector.recommend_action(["a", "b", "c"], 10)
        assert "Warning" in msg

    def test_above_50_critical(self):
        msg = FeatureDriftDetector.recommend_action(["a"] * 6, 10)
        assert "Critical" in msg


class TestVolatileRegime:
    def test_volatile_regime_flags_multiple(self, baseline):
        """Baseline: calm N(0,1). Current: volatile regime — wider, shifted."""
        rng = np.random.default_rng(3)
        cur = pd.DataFrame({
            "f1": rng.normal(2.0, 2.5, size=500),
            "f2": rng.normal(-1.5, 2.0, size=500),
            "f3": rng.normal(0.0, 1.0, size=500),  # stable
        })
        drifted = baseline.get_drifted_features(cur)
        assert "f1" in drifted and "f2" in drifted
        assert "f3" not in drifted
        msg = FeatureDriftDetector.recommend_action(drifted, 3)
        assert "Warning" in msg or "Critical" in msg
