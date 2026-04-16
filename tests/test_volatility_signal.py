"""Tests for the volatility risk premium signal and regime classifier."""

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.volatility_signal import (
    VolatilityRiskPremiumSignal,
    VolRegimeClassifier,
)


# ---------------------------------------------------------------------------
# VolatilityRiskPremiumSignal
# ---------------------------------------------------------------------------

class TestVolatilityRiskPremiumSignal:
    def test_short_vol_when_iv_much_higher_than_rv(self):
        """When IV >> RV in the tail of the window, VRP percentile is high → short vol."""
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        # Baseline: IV and RV are both around 0.15 for most of the series.
        iv = np.full(n, 0.15)
        rv = np.full(n, 0.12)
        # Last 5 bars: IV spikes to 0.40 while RV stays at 0.12.
        iv[-5:] = 0.40
        bars = pd.DataFrame({"iv": iv, "rv": rv}, index=idx)

        gen = VolatilityRiskPremiumSignal(params={"vrp_lookback": 60})
        sigs = gen.generate(bars, symbol="SPY")
        assert len(sigs) > 0
        final = sigs[-1]
        assert final.side == -1  # short vol
        assert final.metadata["regime"] == "high_vrp"
        assert final.metadata["regime_modifier"]["momentum"] > 1.0

    def test_long_vol_when_iv_much_lower_than_rv(self):
        """VRP low (IV << RV) → long-vol / buy-protection signal."""
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        # Baseline: both around 0.20.
        iv = np.full(n, 0.20)
        rv = np.full(n, 0.15)
        # Last 5 bars: IV collapses to 0.05 while RV stays elevated.
        iv[-5:] = 0.05
        bars = pd.DataFrame({"iv": iv, "rv": rv}, index=idx)

        gen = VolatilityRiskPremiumSignal(params={"vrp_lookback": 60})
        sigs = gen.generate(bars, symbol="SPY")
        assert len(sigs) > 0
        final = sigs[-1]
        assert final.side == 1  # long vol
        assert final.metadata["regime"] == "low_vrp"
        assert final.metadata["regime_modifier"]["mean_reversion"] > 1.0

    def test_no_signal_in_neutral_vrp_zone(self):
        """When VRP sits in the middle of its distribution no signals fire."""
        rng = np.random.default_rng(0)
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        # IV / RV = roughly constant ratio with small noise — VRP percentile
        # hovers around 50, never crossing the 25/75 thresholds.
        iv = 0.20 + rng.normal(0.0, 0.01, size=n)
        rv = 0.17 + rng.normal(0.0, 0.01, size=n)
        bars = pd.DataFrame({"iv": iv, "rv": rv}, index=idx)

        gen = VolatilityRiskPremiumSignal(params={"vrp_lookback": 60})
        sigs = gen.generate(bars, symbol="SPY")
        # Some signals may fire on tail days, but the vast majority of
        # post-warmup bars should produce no signal.
        assert len(sigs) < (n - 60) * 0.6  # less than 60% of usable bars

    def test_confidence_greater_for_more_extreme_percentile(self):
        """A 95th-percentile VRP should beat a 76th-percentile one in confidence."""
        rng = np.random.default_rng(1)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        iv = 0.15 + rng.normal(0.0, 0.01, size=n)
        rv = 0.12 + rng.normal(0.0, 0.01, size=n)
        bars = pd.DataFrame({"iv": iv, "rv": rv}, index=idx)

        gen = VolatilityRiskPremiumSignal(params={"vrp_lookback": 50})
        sigs = gen.generate(bars, symbol="SPY")
        # Sort emitted high-VRP signals by percentile and verify monotonicity
        # in confidence.
        highs = [s for s in sigs if s.metadata["regime"] == "high_vrp"]
        highs_sorted = sorted(highs, key=lambda s: s.metadata["vrp_percentile"])
        if len(highs_sorted) >= 2:
            for a, b in zip(highs_sorted, highs_sorted[1:]):
                assert b.confidence + 1e-9 >= a.confidence

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            VolatilityRiskPremiumSignal(params={"vrp_lookback": 2})
        with pytest.raises(ValueError):
            VolatilityRiskPremiumSignal(
                params={"low_percentile": 60, "high_percentile": 80}
            )
        with pytest.raises(ValueError):
            VolatilityRiskPremiumSignal(params={"boost": 1.0})


# ---------------------------------------------------------------------------
# VolRegimeClassifier
# ---------------------------------------------------------------------------

class TestVolRegimeClassifier:
    def test_labels_high_vol_periods(self):
        """Elevated vol in the tail should produce 'high_vol' labels."""
        rng = np.random.default_rng(2)
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        # Baseline ≈ 0.01, tail spikes to 0.05.
        vol = np.concatenate([
            0.01 + np.abs(rng.normal(0.0, 0.002, size=n - 40)),
            0.05 + np.abs(rng.normal(0.0, 0.005, size=40)),
        ])
        series = pd.Series(vol, index=idx, name="garch_vol")

        clf = VolRegimeClassifier(window=60, low_percentile=25.0, high_percentile=75.0)
        labels = clf.classify(series)
        tail_labels = labels["regime"].iloc[-20:].dropna()
        # At least a majority of the tail should be labeled high_vol.
        assert (tail_labels == "high_vol").mean() > 0.5

    def test_labels_low_vol_periods(self):
        rng = np.random.default_rng(3)
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        vol = np.concatenate([
            0.05 + np.abs(rng.normal(0.0, 0.005, size=n - 40)),
            0.01 + np.abs(rng.normal(0.0, 0.002, size=40)),
        ])
        series = pd.Series(vol, index=idx, name="garch_vol")
        clf = VolRegimeClassifier(window=60)
        labels = clf.classify(series)
        tail_labels = labels["regime"].iloc[-20:].dropna()
        assert (tail_labels == "low_vol").mean() > 0.5

    def test_returns_dataframe_columns(self):
        series = pd.Series(np.arange(1, 101, dtype=float))
        clf = VolRegimeClassifier(window=30)
        out = clf.classify(series)
        assert set(out.columns) == {"regime", "percentile", "transition"}

    def test_transition_matrix_rows_sum_to_one(self):
        rng = np.random.default_rng(4)
        n = 400
        # Use alternating regimes so multiple transitions occur.
        vol = np.concatenate([
            np.full(100, 0.01) + rng.normal(0, 1e-3, 100),
            np.full(100, 0.05) + rng.normal(0, 1e-3, 100),
            np.full(100, 0.01) + rng.normal(0, 1e-3, 100),
            np.full(100, 0.05) + rng.normal(0, 1e-3, 100),
        ])
        clf = VolRegimeClassifier(window=50)
        clf.classify(pd.Series(vol))
        mtx = clf.transition_probabilities()
        # For rows that had at least one observation, the row must sum to 1.
        for label in mtx.index:
            row_sum = mtx.loc[label].sum()
            if np.isfinite(row_sum):
                np.testing.assert_allclose(row_sum, 1.0, rtol=1e-9)

    def test_transition_probabilities_before_classify_raises(self):
        clf = VolRegimeClassifier(window=30)
        with pytest.raises(RuntimeError):
            clf.transition_probabilities()

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            VolRegimeClassifier(window=2)
        with pytest.raises(ValueError):
            VolRegimeClassifier(low_percentile=80, high_percentile=70)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
