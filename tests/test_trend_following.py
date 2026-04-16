"""Tests for the trend-following signals (Clenow)."""

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.trend_following import (
    DonchianBreakoutSignal,
    MovingAverageCrossoverSignal,
    atr_position_size,
)


# ---------------------------------------------------------------------------
# Synthetic OHLC helpers
# ---------------------------------------------------------------------------

def _trend_bars(n: int, drift: float, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Generate OHLC bars from a GBM-like trend."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=vol, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)))
    # High / low around close with small intra-bar noise.
    high_noise = np.abs(rng.normal(0.0, vol * 0.7, size=n))
    low_noise = np.abs(rng.normal(0.0, vol * 0.7, size=n))
    high = close * (1.0 + high_noise)
    low = close * (1.0 - low_noise)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"close": close.values, "high": high.values, "low": low.values}, index=idx
    )


# ---------------------------------------------------------------------------
# MA crossover
# ---------------------------------------------------------------------------

class TestMovingAverageCrossoverSignal:
    def test_long_on_uptrend(self):
        # Strong drift, low vol → fast EMA stays above slow throughout.
        bars = _trend_bars(n=300, drift=0.003, vol=0.005, seed=0)
        gen = MovingAverageCrossoverSignal(
            params={"fast_period": 10, "slow_period": 30}
        )
        sigs = gen.generate(bars, symbol="UP")
        assert len(sigs) > 0
        long_frac = sum(1 for s in sigs if s.side == 1) / len(sigs)
        assert long_frac > 0.95
        assert sigs[-1].side == 1

    def test_short_on_downtrend(self):
        bars = _trend_bars(n=300, drift=-0.003, vol=0.005, seed=1)
        gen = MovingAverageCrossoverSignal(
            params={"fast_period": 10, "slow_period": 30}
        )
        sigs = gen.generate(bars, symbol="DOWN")
        assert len(sigs) > 0
        short_frac = sum(1 for s in sigs if s.side == -1) / len(sigs)
        assert short_frac > 0.95

    def test_triple_ma_requires_majority(self):
        """Triple-MA must agree when all three MAs clearly align."""
        bars = _trend_bars(n=400, drift=0.002, seed=2)
        gen = MovingAverageCrossoverSignal(
            params={"fast_period": 10, "medium_period": 20, "slow_period": 50}
        )
        sigs = gen.generate(bars, symbol="TRI")
        # With a clean uptrend, all three MAs line up fast > medium > slow,
        # so side should be long for (nearly) every emitted signal.
        tail = sigs[-50:]
        assert all(s.side == 1 for s in tail)

    def test_confidence_greater_for_stronger_trend(self):
        weak = _trend_bars(n=300, drift=0.0005, vol=0.005, seed=3)
        strong = _trend_bars(n=300, drift=0.004, vol=0.005, seed=3)
        gen = MovingAverageCrossoverSignal(
            params={"fast_period": 10, "slow_period": 30}
        )
        c_weak = np.mean([s.confidence for s in gen.generate(weak, symbol="W")])
        c_strong = np.mean([s.confidence for s in gen.generate(strong, symbol="S")])
        assert c_strong > c_weak

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            MovingAverageCrossoverSignal(
                params={"fast_period": 30, "slow_period": 20}
            )
        with pytest.raises(ValueError):
            MovingAverageCrossoverSignal(
                params={"fast_period": 10, "medium_period": 60, "slow_period": 50}
            )


# ---------------------------------------------------------------------------
# Donchian breakout
# ---------------------------------------------------------------------------

class TestDonchianBreakoutSignal:
    def test_long_entry_on_new_high(self):
        """A series that makes new highs should produce a long entry signal."""
        n = 200
        # Flat for the first 100 bars, then strict uptrend.
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        flat = np.full(100, 100.0)
        rally = 100.0 + np.arange(1, 101, dtype=float)  # monotonic up
        close = np.concatenate([flat, rally])
        high = close * 1.005
        low = close * 0.995
        bars = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)

        gen = DonchianBreakoutSignal(params={"entry_period": 20, "exit_period": 10})
        sigs = gen.generate(bars, symbol="RALLY")
        entry_sigs = [s for s in sigs if s.metadata["event"] == "entry" and s.side == 1]
        assert len(entry_sigs) >= 1
        # First entry should land shortly after the breakout begins.
        first_entry_t = entry_sigs[0].timestamp
        assert first_entry_t >= idx[100].to_pydatetime()

    def test_short_entry_on_new_low(self):
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        flat = np.full(100, 100.0)
        selloff = 100.0 - np.arange(1, 101, dtype=float)  # strict downtrend
        close = np.concatenate([flat, np.maximum(selloff, 1.0)])  # avoid non-positive
        high = close * 1.005
        low = close * 0.995
        bars = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)

        gen = DonchianBreakoutSignal(params={"entry_period": 20, "exit_period": 10})
        sigs = gen.generate(bars, symbol="SELL")
        entry_shorts = [s for s in sigs if s.metadata["event"] == "entry" and s.side == -1]
        assert len(entry_shorts) >= 1

    def test_long_exit_when_price_breaks_exit_channel(self):
        """After a long entry, a pullback below the exit-low should emit exit=0."""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        # 20 bars flat, 60 bars up to establish a long, then 60 bars of mean reversion.
        flat = np.full(40, 100.0)
        rally = 100.0 + np.arange(1, 61, dtype=float)  # long entry
        pullback = rally[-1] - np.arange(1, 101, dtype=float) * 1.5  # sharp drop
        close = np.concatenate([flat, rally, pullback])
        high = close * 1.005
        low = close * 0.995
        bars = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)

        gen = DonchianBreakoutSignal(params={"entry_period": 20, "exit_period": 10})
        sigs = gen.generate(bars, symbol="LE")
        events = [(s.metadata["event"], s.side) for s in sigs]
        # Need at least one long entry followed by an exit.
        assert ("entry", 1) in events
        assert ("exit", 0) in events
        # Exit comes after the last long entry in the event stream.
        last_entry_idx = max(i for i, e in enumerate(events) if e == ("entry", 1))
        assert ("exit", 0) in events[last_entry_idx + 1 :]

    def test_requires_ohlc_columns(self):
        gen = DonchianBreakoutSignal()
        bars = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="missing columns"):
            gen.validate_input(bars)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            DonchianBreakoutSignal(params={"entry_period": 1})
        with pytest.raises(ValueError):
            DonchianBreakoutSignal(params={"exit_period": 1})


# ---------------------------------------------------------------------------
# ATR position sizing
# ---------------------------------------------------------------------------

class TestATRPositionSize:
    def test_smaller_size_for_more_volatile_asset(self):
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        rng = np.random.default_rng(5)
        # Low-vol asset.
        low_vol_close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, size=n))), index=idx
        )
        # High-vol asset — 10x the bar-to-bar vol.
        high_vol_close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.05, size=n))), index=idx
        )

        def _hl(close):
            return close * 1.01, close * 0.99

        lv_h, lv_l = _hl(low_vol_close)
        hv_h, hv_l = _hl(high_vol_close)

        lv_units = atr_position_size(
            low_vol_close, lv_h, lv_l, risk_per_trade=0.02,
            account_value=100_000.0, atr_period=20,
        ).dropna()
        hv_units = atr_position_size(
            high_vol_close, hv_h, hv_l, risk_per_trade=0.02,
            account_value=100_000.0, atr_period=20,
        ).dropna()

        # Higher ATR → smaller position in units.
        assert hv_units.mean() < lv_units.mean()

    def test_size_scales_with_account(self):
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        rng = np.random.default_rng(6)
        close = pd.Series(100.0 + rng.normal(0.0, 1.0, size=n).cumsum(), index=idx)
        high = close * 1.01
        low = close * 0.99

        small = atr_position_size(
            close, high, low, account_value=50_000.0, atr_period=14,
        ).dropna()
        big = atr_position_size(
            close, high, low, account_value=500_000.0, atr_period=14,
        ).dropna()
        # Tenfold account → tenfold units.
        np.testing.assert_allclose(big.mean() / small.mean(), 10.0, rtol=1e-9)

    def test_warmup_is_nan(self):
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        close = pd.Series(np.linspace(100.0, 110.0, n), index=idx)
        high = close * 1.01
        low = close * 0.99
        units = atr_position_size(close, high, low, atr_period=20)
        # EMA with min_periods=20 is NaN for positions 0..18 (19 bars of warmup)
        # and produces its first value at position 19.
        assert units.iloc[:19].isna().all()
        assert np.isfinite(units.iloc[19])

    def test_invalid_params(self):
        s = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(ValueError):
            atr_position_size(s, s, s, atr_period=1)
        with pytest.raises(ValueError):
            atr_position_size(s, s, s, atr_multiplier=0)
        with pytest.raises(ValueError):
            atr_position_size(s, s, s, risk_per_trade=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
