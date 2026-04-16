"""Tests for Layer-2 Kelly criterion sizing (Chan, design-doc §8.2)."""

import numpy as np
import pandas as pd
import pytest

from src.bet_sizing.kelly import (
    fractional_kelly,
    kelly_fraction,
    kelly_from_meta_labeler,
    rolling_kelly,
)


# ---------------------------------------------------------------------------
# kelly_fraction — closed-form values
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def test_classical_60_40_at_1to1_payoff(self):
        # p=0.6, q=0.4, W=L=1 → f* = (0.6 - 0.4) = 0.2.
        assert np.isclose(kelly_fraction(0.6, 1.0, 1.0), 0.2)

    def test_zero_edge_at_fifty_fifty(self):
        # p=0.5, W=L=1 → f* = 0 (fair coin).
        assert kelly_fraction(0.5, 1.0, 1.0) == 0.0

    def test_negative_edge_clips_to_zero(self):
        # p=0.4, W=L=1 → f* = (0.4 − 0.6)/(1·1) = −0.2 → clipped to 0.
        assert kelly_fraction(0.4, 1.0, 1.0) == 0.0

    def test_asymmetric_payoff(self):
        # p=0.5, W=2, L=1 → f* = (0.5·2 − 0.5·1)/(2·1) = 0.5/2 = 0.25.
        assert np.isclose(kelly_fraction(0.5, 2.0, 1.0), 0.25)

    def test_huge_edge_clips_to_one(self):
        # p=0.99, W=10, L=0.1 → f* would be massive → clipped to 1.
        f = kelly_fraction(0.99, 10.0, 0.1)
        assert f == 1.0

    def test_rejects_bad_win_prob(self):
        with pytest.raises(ValueError):
            kelly_fraction(1.2, 1.0, 1.0)
        with pytest.raises(ValueError):
            kelly_fraction(-0.1, 1.0, 1.0)

    def test_rejects_nonpositive_magnitudes(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.6, 0.0, 1.0)
        with pytest.raises(ValueError):
            kelly_fraction(0.6, 1.0, -0.5)


# ---------------------------------------------------------------------------
# fractional_kelly
# ---------------------------------------------------------------------------

class TestFractionalKelly:
    def test_quarter_of_full(self):
        full = kelly_fraction(0.6, 1.0, 1.0)
        frac = fractional_kelly(0.6, 1.0, 1.0, fraction=0.25)
        assert np.isclose(frac, 0.25 * full)

    def test_half_of_full(self):
        full = kelly_fraction(0.7, 2.0, 1.0)
        frac = fractional_kelly(0.7, 2.0, 1.0, fraction=0.5)
        assert np.isclose(frac, 0.5 * full)

    def test_zero_fraction_gives_zero(self):
        assert fractional_kelly(0.6, 1.0, 1.0, fraction=0.0) == 0.0

    def test_rejects_out_of_range_fraction(self):
        with pytest.raises(ValueError):
            fractional_kelly(0.6, 1.0, 1.0, fraction=-0.1)
        with pytest.raises(ValueError):
            fractional_kelly(0.6, 1.0, 1.0, fraction=1.5)


# ---------------------------------------------------------------------------
# rolling_kelly
# ---------------------------------------------------------------------------

class TestRollingKelly:
    def _synthetic_returns(self, n: int = 400, win_rate: float = 0.6,
                           seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        wins = rng.uniform(0.005, 0.015, size=int(n * win_rate))
        losses = -rng.uniform(0.005, 0.015, size=n - int(n * win_rate))
        all_ret = np.concatenate([wins, losses])
        rng.shuffle(all_ret)
        return pd.Series(
            all_ret,
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
            name="ret",
        )

    def test_series_length_matches_input(self):
        r = self._synthetic_returns(n=300)
        kf = rolling_kelly(r, window=50)
        assert len(kf) == len(r)
        assert kf.index.equals(r.index)

    def test_warmup_is_nan(self):
        r = self._synthetic_returns(n=200)
        kf = rolling_kelly(r, window=50)
        assert kf.iloc[:49].isna().all()
        # After window warmup, we should have numeric values.
        assert kf.iloc[49:].notna().all()

    def test_positive_when_edge_exists(self):
        r = self._synthetic_returns(n=400, win_rate=0.65, seed=3)
        kf = rolling_kelly(r, window=100, fraction=0.5)
        # At least some windows should deliver a positive Kelly.
        assert (kf.dropna() > 0).any()

    def test_zero_when_no_wins(self):
        # All-loss returns: Kelly must return 0 for every window.
        r = pd.Series(
            -0.01 * np.ones(200),
            index=pd.date_range("2024-01-01", periods=200, freq="1h"),
        )
        kf = rolling_kelly(r, window=50)
        assert (kf.dropna() == 0.0).all()

    def test_zero_when_no_losses(self):
        r = pd.Series(
            0.01 * np.ones(200),
            index=pd.date_range("2024-01-01", periods=200, freq="1h"),
        )
        kf = rolling_kelly(r, window=50)
        assert (kf.dropna() == 0.0).all()

    def test_rejects_bad_inputs(self):
        r = pd.Series([0.01, -0.01])
        with pytest.raises(ValueError):
            rolling_kelly(r, window=1)
        with pytest.raises(ValueError):
            rolling_kelly(r, window=10, fraction=2.0)
        with pytest.raises(ValueError):
            rolling_kelly(np.array([0.01, -0.01]), window=10)


# ---------------------------------------------------------------------------
# kelly_from_meta_labeler
# ---------------------------------------------------------------------------

class TestKellyFromMetaLabeler:
    def test_positive_for_profitable_signal(self):
        # prob=0.7, W=2%, L=1% → positive fractional Kelly.
        f = kelly_from_meta_labeler(0.7, 0.02, 0.01, fraction=0.25)
        assert f > 0.0

    def test_matches_fractional_kelly(self):
        f_mel = kelly_from_meta_labeler(0.65, 0.015, 0.012, fraction=0.25)
        f_ref = fractional_kelly(0.65, 0.015, 0.012, fraction=0.25)
        assert np.isclose(f_mel, f_ref)

    def test_zero_for_unprofitable_signal(self):
        # Symmetric payoff + prob<0.5 → EV < 0 → raw Kelly negative →
        # clipped to 0 → fractional is 0.
        # (Note: prob=0.4 with asymmetric W=2×L is still positive-EV —
        # Kelly only goes to zero when the probability×payoff product is
        # below the loss-side product.)
        assert kelly_from_meta_labeler(0.4, 0.01, 0.01) == 0.0

    def test_capped_at_fraction_times_one(self):
        # Extreme edge: prob=0.99, W big, L tiny → full Kelly = 1 →
        # fractional Kelly = fraction.
        f = kelly_from_meta_labeler(0.99, 10.0, 0.1, fraction=0.25)
        assert np.isclose(f, 0.25)
