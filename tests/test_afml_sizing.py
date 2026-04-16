"""Tests for AFML Layer-1 probability→size conversion."""

import numpy as np
import pytest

from src.bet_sizing.afml_sizing import (
    bet_size_from_probability,
    bet_size_with_side,
    discretize_bet_size,
)


# ---------------------------------------------------------------------------
# bet_size_from_probability
# ---------------------------------------------------------------------------

class TestBetSizeFromProbability:
    def test_p_half_is_zero(self):
        assert bet_size_from_probability(0.5) == 0.0

    def test_p_one_is_max_size(self):
        # p=1 saturates: z → +∞ → Φ(z) → 1 → size = max_size.
        assert np.isclose(bet_size_from_probability(1.0), 1.0)
        assert np.isclose(
            bet_size_from_probability(1.0, max_size=2.5), 2.5,
        )

    def test_p_zero_is_zero(self):
        # p=0 saturates NEGATIVE in the raw AFML formula, but we clip at 0
        # per our convention: meta-label < 0.5 → "skip this trade", not reverse.
        assert bet_size_from_probability(0.0) == 0.0

    def test_p_below_half_clips_to_zero(self):
        # p < 0.5 should ALWAYS produce size ≈ 0 (our convention).
        for p in (0.0, 0.1, 0.3, 0.4, 0.49):
            assert bet_size_from_probability(p) == 0.0, (
                f"p={p} should clip to 0"
            )

    def test_p_above_half_produces_positive_size(self):
        # Concave ramp: strictly increasing in p above 0.5.
        prior = 0.0
        for p in (0.51, 0.6, 0.7, 0.8, 0.9, 0.99):
            size = bet_size_from_probability(p)
            assert size > prior, f"non-monotonic at p={p}"
            prior = size

    def test_vectorized_input_returns_ndarray(self):
        probs = np.array([0.3, 0.5, 0.6, 0.95, 1.0])
        sizes = bet_size_from_probability(probs)
        assert isinstance(sizes, np.ndarray)
        assert sizes.shape == probs.shape
        # Spot-check: p=0.3 clipped to 0, p=0.5 exactly 0, p=1.0 at cap.
        assert sizes[0] == 0.0
        assert sizes[1] == 0.0
        assert 0.0 < sizes[2] < 1.0
        assert np.isclose(sizes[4], 1.0)

    def test_vectorized_matches_scalar(self):
        probs = np.array([0.55, 0.7, 0.9])
        vec = bet_size_from_probability(probs)
        scalar = np.array([bet_size_from_probability(p) for p in probs])
        assert np.allclose(vec, scalar)

    def test_max_size_scales_linearly(self):
        p = 0.8
        base = bet_size_from_probability(p, max_size=1.0)
        scaled = bet_size_from_probability(p, max_size=3.0)
        assert np.isclose(scaled, 3.0 * base)

    def test_rejects_out_of_range_probability(self):
        with pytest.raises(ValueError):
            bet_size_from_probability(1.5)
        with pytest.raises(ValueError):
            bet_size_from_probability(np.array([0.5, -0.1]))

    def test_rejects_negative_max_size(self):
        with pytest.raises(ValueError):
            bet_size_from_probability(0.7, max_size=-1.0)


# ---------------------------------------------------------------------------
# discretize_bet_size
# ---------------------------------------------------------------------------

class TestDiscretize:
    def test_spec_examples(self):
        assert discretize_bet_size(0.35) == 0.4
        assert discretize_bet_size(0.51) == 0.6
        assert discretize_bet_size(0.01) == 0.0

    def test_rounds_to_nearest_tier(self):
        assert discretize_bet_size(0.09) == 0.0
        assert discretize_bet_size(0.11) == 0.2
        assert discretize_bet_size(0.29) == 0.2
        assert discretize_bet_size(0.31) == 0.4
        assert discretize_bet_size(0.79) == 0.8
        assert discretize_bet_size(1.0) == 1.0

    def test_custom_tiers(self):
        tiers = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert discretize_bet_size(0.33, tiers=tiers) == 0.25
        assert discretize_bet_size(0.4, tiers=tiers) == 0.5

    def test_tiebreak_goes_to_lower_tier(self):
        # 0.3 is equidistant from 0.2 and 0.4; argmin picks first occurrence → 0.2.
        assert discretize_bet_size(0.3) == 0.2

    def test_empty_tiers_raises(self):
        with pytest.raises(ValueError):
            discretize_bet_size(0.5, tiers=[])


# ---------------------------------------------------------------------------
# bet_size_with_side
# ---------------------------------------------------------------------------

class TestBetSizeWithSide:
    def test_long_direction(self):
        # side=+1 keeps sign positive.
        s = bet_size_with_side(prob=0.8, side=1)
        assert s > 0
        # Magnitude matches unsigned sizer.
        assert np.isclose(s, bet_size_from_probability(0.8))

    def test_short_direction(self):
        # side=-1 flips sign; magnitude unchanged.
        s = bet_size_with_side(prob=0.8, side=-1)
        assert s < 0
        assert np.isclose(abs(s), bet_size_from_probability(0.8))

    def test_p_below_half_yields_zero_regardless_of_side(self):
        # p < 0.5 → magnitude 0 → result is 0 even after *side.
        assert bet_size_with_side(prob=0.2, side=1) == 0.0
        assert bet_size_with_side(prob=0.2, side=-1) == 0.0

    def test_p_zero_with_short_side_is_zero_not_neg_max(self):
        # Guardrail the spec called out explicitly.
        assert bet_size_with_side(prob=0.0, side=-1) == 0.0

    def test_neutral_side_is_zero(self):
        assert bet_size_with_side(prob=0.9, side=0) == 0.0

    def test_rejects_invalid_side(self):
        with pytest.raises(ValueError):
            bet_size_with_side(prob=0.8, side=2)
        with pytest.raises(ValueError):
            bet_size_with_side(prob=0.8, side=0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Concavity / near-certainty saturation
# ---------------------------------------------------------------------------

class TestShape:
    """
    The AFML map is an S-curve — not strictly concave — but it *does* have
    the design-doc property that MARGINAL signals (p just above 0.5) get
    vanishingly small sizes while HIGHLY confident signals saturate near
    max. These two tests pin those boundary behaviours.
    """

    def test_marginal_signal_gets_small_size(self):
        # p=0.51: barely above chance → size should be < 5% of max.
        assert bet_size_from_probability(0.51) < 0.05

    def test_near_max_at_high_confidence(self):
        # Design-doc expectation: p=0.95 gets "near-max" bet.
        s = bet_size_from_probability(0.95)
        assert s > 0.9, f"p=0.95 should be near-max; got {s}"
