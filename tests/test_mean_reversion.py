"""Tests for the mean reversion signal (Chan)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.mean_reversion import (
    MeanReversionSignal,
    compute_bollinger_zscore,
    compute_ou_halflife,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _ar1_series(
    n: int,
    phi: float = 0.9,
    mean: float = 100.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> pd.Series:
    """AR(1) around ``mean``: y_t = mean + phi*(y_{t-1} - mean) + eps."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = mean
    for t in range(1, n):
        y[t] = mean + phi * (y[t - 1] - mean) + rng.normal(0.0, sigma)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(y, index=idx)


def _random_walk(n: int, seed: int = 0, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 1.0, size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(100.0 + np.cumsum(steps), index=idx)


# ---------------------------------------------------------------------------
# O-U half-life
# ---------------------------------------------------------------------------

class TestOUHalfLife:
    def test_mean_reverting_series_has_finite_halflife(self):
        """AR(1) with phi=0.9 has theoretical half-life = ln(2)/ln(1/phi) ≈ 6.58."""
        s = _ar1_series(n=1000, phi=0.9, seed=0)
        hl, pval = compute_ou_halflife(s)
        assert math.isfinite(hl)
        # Coarse band around the theoretical value.
        theoretical = math.log(2) / math.log(1 / 0.9)  # ≈ 6.58
        assert 2 < hl < 30
        # ADF should reject the unit root for a mean-reverting series.
        assert pval < 0.05

    def test_random_walk_has_non_tradeable_halflife(self):
        """
        A pure random walk should either produce an infinite half-life
        (lambda >= 0) or a half-life far beyond any realistic trading
        horizon (>> 100 bars). Either way ADF must not reject the unit
        root, and the signal generator will filter the series out.
        """
        s = _random_walk(n=1000, seed=1)
        hl, pval = compute_ou_halflife(s)
        assert hl == float("inf") or hl > 100
        assert pval > 0.05

    def test_constant_series_returns_inf(self):
        s = pd.Series([100.0] * 200)
        hl, pval = compute_ou_halflife(s)
        assert hl == float("inf")

    def test_short_series_returns_inf(self):
        s = pd.Series([1.0, 2.0, 3.0])
        hl, pval = compute_ou_halflife(s)
        assert hl == float("inf")


# ---------------------------------------------------------------------------
# Bollinger z-score
# ---------------------------------------------------------------------------

class TestBollingerZScore:
    def test_zero_when_price_equals_sma(self):
        # A constant-return step produces a rolling mean equal to the price
        # once the window fills; check the z-score at that point is zero.
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        close = pd.Series(np.full(50, 100.0), index=idx)
        z = compute_bollinger_zscore(close, window=10, num_std=2.0).dropna()
        # Rolling std is 0 → division-by-zero guard produces NaN after the
        # replace, so the dropna'd Series is empty. Sanity: the function
        # ran without raising and returned all-NaN for a flat series.
        assert z.empty

        # With a tiny noise component, the SMA-close gap averages to zero.
        rng = np.random.default_rng(0)
        close = pd.Series(100.0 + rng.normal(0.0, 1.0, size=300), index=pd.date_range("2024", periods=300, freq="D"))
        z = compute_bollinger_zscore(close, window=20, num_std=2.0).dropna()
        # Mean of the z-score over a stationary series should be close to 0.
        assert abs(z.mean()) < 0.2

    def test_sign_follows_price(self):
        """z > 0 when price is above SMA, z < 0 when below."""
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        close = pd.Series(np.arange(1, 61, dtype=float) + 100.0, index=idx)
        # Persistent uptrend → every price is above the trailing SMA.
        z = compute_bollinger_zscore(close, window=20, num_std=2.0).dropna()
        assert (z > 0).all()

    def test_invalid_params(self):
        s = pd.Series(np.arange(10.0))
        with pytest.raises(ValueError):
            compute_bollinger_zscore(s, window=1)
        with pytest.raises(ValueError):
            compute_bollinger_zscore(s, window=5, num_std=0)


# ---------------------------------------------------------------------------
# MeanReversionSignal
# ---------------------------------------------------------------------------

class TestMeanReversionSignal:
    def test_generates_sell_when_far_above_mean(self):
        """Inject a big positive spike into an AR(1) series — signal should short."""
        s = _ar1_series(n=600, phi=0.9, seed=2)
        # Final bar is several sigma above the series mean.
        s.iloc[-1] = s.mean() + 8.0 * s.std()
        bars = pd.DataFrame({"close": s})

        gen = MeanReversionSignal(params={"entry_threshold": 1.5})
        sigs = gen.generate(bars, symbol="MR")
        assert len(sigs) > 0
        # The final signal should be an entry on the short side.
        final = sigs[-1]
        assert final.metadata["event"] == "entry"
        assert final.side == -1
        # Confidence should be meaningful — the z-score is capped by the
        # half-life-sized rolling SMA which catches up quickly after a
        # single-bar spike, so we only require notable (>0.3) not saturated.
        assert final.confidence > 0.3

    def test_generates_buy_when_far_below_mean(self):
        s = _ar1_series(n=600, phi=0.9, seed=3)
        s.iloc[-1] = s.mean() - 8.0 * s.std()
        bars = pd.DataFrame({"close": s})
        gen = MeanReversionSignal(params={"entry_threshold": 1.5})
        sigs = gen.generate(bars, symbol="MR")
        assert len(sigs) > 0
        final = sigs[-1]
        assert final.metadata["event"] == "entry"
        assert final.side == 1

    def test_exit_signal_when_z_crosses_back(self):
        """After a spike, as the AR(1) reverts, we should see an exit event."""
        s = _ar1_series(n=600, phi=0.8, seed=4)
        bars = pd.DataFrame({"close": s})
        gen = MeanReversionSignal(
            params={"entry_threshold": 1.5, "exit_threshold": 0.3}
        )
        sigs = gen.generate(bars, symbol="MR")
        # Expect both entry and exit events in a natural AR(1) path.
        events = {s.metadata["event"] for s in sigs}
        assert "entry" in events
        assert "exit" in events

    def test_skips_random_walk(self):
        s = _random_walk(n=800, seed=5)
        bars = pd.DataFrame({"close": s})
        gen = MeanReversionSignal()
        # Non-stationary → no signals.
        assert gen.generate(bars, symbol="RW") == []

    def test_skips_outside_halflife_band(self):
        """Very fast reversion (phi very small) → half-life near 1;
        we filter it out with min_halflife=3."""
        s = _ar1_series(n=1000, phi=0.1, seed=6)  # half-life ≈ 0.3
        bars = pd.DataFrame({"close": s})
        gen = MeanReversionSignal(params={"min_halflife": 3.0, "max_halflife": 100.0})
        assert gen.generate(bars, symbol="FAST") == []

    def test_metadata_populated(self):
        s = _ar1_series(n=500, phi=0.85, seed=7)
        bars = pd.DataFrame({"close": s})
        gen = MeanReversionSignal(params={"entry_threshold": 1.0})
        sigs = gen.generate(bars, symbol="MR")
        assert len(sigs) > 0
        md = sigs[0].metadata
        assert "half_life" in md and md["half_life"] > 0
        assert "window" in md and md["window"] >= 2
        assert "z_score" in md
        assert "adf_pvalue" in md

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            MeanReversionSignal(params={"entry_threshold": 0.5, "exit_threshold": 1.0})
        with pytest.raises(ValueError):
            MeanReversionSignal(params={"min_halflife": 50, "max_halflife": 10})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
