"""Tests for microstructure features (AFML Ch. 19 + Johnson)."""

import numpy as np
import pandas as pd
import pytest

from src.feature_factory.microstructure import (
    amihud_lambda,
    compute_microstructure_features,
    hasbrouck_lambda,
    kyle_lambda,
    order_flow_imbalance,
    roll_spread,
    trade_intensity,
    vpin,
)


def _make_bars(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Small synthetic bars DataFrame using the project's Bar fields."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    close = pd.Series(100.0 + rng.normal(0.0, 0.5, size=n).cumsum(), index=ts)
    volume = pd.Series(rng.integers(100, 1000, size=n).astype(float), index=ts)
    # Split volume into buy/sell: start balanced, then skew long.
    buy = volume * rng.uniform(0.4, 0.6, size=n)
    sell = volume - buy
    dollar_volume = close * volume
    tick_count = pd.Series(rng.integers(10, 100, size=n), index=ts)
    duration = pd.Series(np.full(n, 60.0), index=ts)
    return pd.DataFrame({
        "close": close,
        "volume": volume,
        "dollar_volume": dollar_volume,
        "buy_volume": buy,
        "sell_volume": sell,
        "tick_count": tick_count,
        "bar_duration_seconds": duration,
    })


# ---------------------------------------------------------------------------
# Kyle's lambda
# ---------------------------------------------------------------------------

class TestKyleLambda:
    def test_positive_when_price_tracks_signed_volume(self):
        """If dp = 0.1 * signed_volume + noise, lambda should be ~0.1 > 0."""
        rng = np.random.default_rng(1)
        n = 400
        sv = rng.normal(0.0, 10.0, size=n)
        dp = 0.1 * sv + rng.normal(0.0, 0.05, size=n)
        close = pd.Series(100.0 + np.cumsum(dp))
        signed_volume = pd.Series(sv)
        volume = pd.Series(np.abs(sv) + 10.0)

        lam = kyle_lambda(close, volume, signed_volume, window=100).dropna()
        # Both the mean and median should be positive and close to 0.1.
        assert lam.mean() > 0
        assert abs(lam.mean() - 0.1) < 0.05

    def test_warmup_is_nan(self):
        bars = _make_bars(n=200)
        lam = kyle_lambda(
            bars["close"], bars["volume"], bars["buy_volume"] - bars["sell_volume"],
            window=50,
        )
        assert lam.iloc[:50].isna().all()

    def test_invalid_window(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            kyle_lambda(s, s, s, window=2)


# ---------------------------------------------------------------------------
# Amihud's lambda
# ---------------------------------------------------------------------------

class TestAmihudLambda:
    def test_higher_for_low_volume(self):
        """Low-dollar-volume bars should produce higher Amihud illiquidity."""
        rng = np.random.default_rng(2)
        n = 400
        returns = rng.normal(0.0, 0.01, size=n)
        close = pd.Series(100.0 * np.exp(np.cumsum(returns)))
        # First half: high dollar volume; second half: low dollar volume.
        dv = pd.Series(
            np.concatenate([np.full(n // 2, 1e7), np.full(n // 2, 1e4)])
        )
        amihud = amihud_lambda(close, dv, window=50).dropna()
        # Post-volume-drop Amihud should exceed pre-drop Amihud by a large margin.
        before = amihud.iloc[:150].mean()
        after = amihud.iloc[-100:].mean()
        assert after > before * 10

    def test_nan_on_zero_volume(self):
        """Zero dollar volume should not divide by zero."""
        close = pd.Series(np.linspace(100.0, 110.0, 100))
        dv = pd.Series(np.zeros(100))
        out = amihud_lambda(close, dv, window=20)
        # All entries should be NaN (divided by zero → NaN, then rolling mean NaN).
        assert out.isna().all()


# ---------------------------------------------------------------------------
# Roll spread
# ---------------------------------------------------------------------------

class TestRollSpread:
    def test_positive_for_bid_ask_bounce(self):
        """Simulate bid-ask bounce → negative autocov → positive Roll spread."""
        rng = np.random.default_rng(3)
        n = 500
        # True mid price: slow random walk
        mid = 100.0 + rng.normal(0.0, 0.01, size=n).cumsum()
        # Alternating side (bounce): +/- half-spread of 0.1
        side = np.where(rng.uniform(size=n) > 0.5, 1.0, -1.0)
        half_spread = 0.05
        obs = pd.Series(mid + side * half_spread)

        out = roll_spread(obs, window=100).dropna()
        # Estimated fractional spread should be >> 0 and roughly 2*half_spread/mid.
        assert out.mean() > 0
        # Expected magnitude ≈ 2 * half_spread / 100 = 0.001. Allow a wide band.
        assert 1e-4 < out.mean() < 5e-3

    def test_zero_when_autocov_positive(self):
        """Momentum-like series (positive autocov) gives spread = 0."""
        close = pd.Series(100.0 + np.arange(300.0) * 0.5)  # deterministic trend
        out = roll_spread(close, window=50).dropna()
        # delta_p is constant → cov = 0 → spread = 0.
        assert (out == 0).all()

    def test_warmup_nan(self):
        out = roll_spread(pd.Series(np.arange(100.0)), window=30)
        assert out.iloc[:30].isna().all()


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

class TestVPIN:
    def test_one_sided_flow_gives_vpin_near_one(self):
        n = 200
        buy = pd.Series(np.full(n, 100.0))
        sell = pd.Series(np.zeros(n))
        total = buy + sell
        out = vpin(buy, sell, total, window=50).dropna()
        np.testing.assert_allclose(out.values, 1.0)

    def test_balanced_flow_gives_vpin_near_zero(self):
        n = 200
        buy = pd.Series(np.full(n, 50.0))
        sell = pd.Series(np.full(n, 50.0))
        total = buy + sell
        out = vpin(buy, sell, total, window=50).dropna()
        np.testing.assert_allclose(out.values, 0.0)

    def test_range(self):
        rng = np.random.default_rng(4)
        n = 400
        buy = pd.Series(rng.uniform(0, 100, size=n))
        sell = pd.Series(rng.uniform(0, 100, size=n))
        total = buy + sell
        out = vpin(buy, sell, total, window=50).dropna()
        assert (out >= 0).all()
        assert (out <= 1).all()


# ---------------------------------------------------------------------------
# Order flow imbalance
# ---------------------------------------------------------------------------

class TestOrderFlowImbalance:
    def test_range_is_minus_one_to_one(self):
        rng = np.random.default_rng(5)
        n = 400
        buy = pd.Series(rng.uniform(0, 100, size=n))
        sell = pd.Series(rng.uniform(0, 100, size=n))
        out = order_flow_imbalance(buy, sell, window=20).dropna()
        assert (out >= -1).all()
        assert (out <= 1).all()

    def test_all_buys_gives_plus_one(self):
        n = 100
        buy = pd.Series(np.full(n, 100.0))
        sell = pd.Series(np.zeros(n))
        out = order_flow_imbalance(buy, sell, window=10).dropna()
        np.testing.assert_allclose(out.values, 1.0)

    def test_all_sells_gives_minus_one(self):
        n = 100
        buy = pd.Series(np.zeros(n))
        sell = pd.Series(np.full(n, 100.0))
        out = order_flow_imbalance(buy, sell, window=10).dropna()
        np.testing.assert_allclose(out.values, -1.0)


# ---------------------------------------------------------------------------
# Trade intensity
# ---------------------------------------------------------------------------

class TestTradeIntensity:
    def test_simple_division(self):
        tick_count = pd.Series([60.0, 120.0, 30.0])
        duration = pd.Series([60.0, 60.0, 60.0])
        out = trade_intensity(tick_count, duration)
        np.testing.assert_allclose(out.values, [1.0, 2.0, 0.5])

    def test_zero_duration_is_nan(self):
        tick_count = pd.Series([60.0, 0.0, 10.0])
        duration = pd.Series([60.0, 0.0, 60.0])
        out = trade_intensity(tick_count, duration)
        assert np.isnan(out.iloc[1])
        assert np.isfinite(out.iloc[0])
        assert np.isfinite(out.iloc[2])


# ---------------------------------------------------------------------------
# Hasbrouck's lambda
# ---------------------------------------------------------------------------

class TestHasbrouckLambda:
    def test_runs_on_small_series(self):
        rng = np.random.default_rng(6)
        n = 200
        sv = rng.normal(0.0, 10.0, size=n)
        dp = 0.05 * sv + rng.normal(0.0, 0.1, size=n)
        close = pd.Series(100.0 + np.cumsum(dp))
        signed_volume = pd.Series(sv)
        out = hasbrouck_lambda(close, signed_volume, window=80, lags=3)
        # At least some entries should be finite — the VAR will fit on this
        # well-behaved synthetic data.
        assert out.dropna().size > 0
        # And the mean should be positive (price moves with signed flow).
        assert out.dropna().mean() > 0

    def test_invalid_window(self):
        close = pd.Series(np.arange(100.0))
        sv = pd.Series(np.arange(100.0))
        with pytest.raises(ValueError):
            hasbrouck_lambda(close, sv, window=10, lags=5)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestComputeMicrostructureFeatures:
    def test_default_columns_present(self):
        bars = _make_bars(n=200)
        out = compute_microstructure_features(bars, window=50)
        expected = {
            "kyle_lambda",
            "amihud_lambda",
            "roll_spread",
            "vpin",
            "order_flow_imbalance",
            "trade_intensity",
        }
        assert set(out.columns) == expected
        assert len(out) == len(bars)
        assert out.index.equals(bars.index)

    def test_missing_column_raises(self):
        bars = _make_bars(n=100).drop(columns=["buy_volume"])
        with pytest.raises(KeyError):
            compute_microstructure_features(bars, window=30)

    def test_include_hasbrouck(self):
        bars = _make_bars(n=200)
        out = compute_microstructure_features(
            bars, window=60, include_hasbrouck=True, hasbrouck_lags=3
        )
        assert "hasbrouck_lambda" in out.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
