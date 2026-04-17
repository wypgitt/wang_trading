"""Tests for Probability of Backtest Overfitting (PBO / CSCV)."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from src.backtesting.pbo import compute_pbo, validate_pbo


def _random_matrix(T: int, N: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.01, size=(T, N))
    idx = pd.date_range("2024-01-01", periods=T, freq="B")
    return pd.DataFrame(data, index=idx, columns=[f"var_{i}" for i in range(N)])


def _matrix_with_one_winner(
    T: int,
    n_noise: int,
    winner_drift: float = 0.003,
    seed: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, size=(T, n_noise))
    winner = rng.normal(winner_drift, 0.01, size=(T, 1))
    data = np.concatenate([winner, noise], axis=1)
    idx = pd.date_range("2024-01-01", periods=T, freq="B")
    cols = ["winner"] + [f"noise_{i}" for i in range(n_noise)]
    return pd.DataFrame(data, index=idx, columns=cols)


class TestComputePBO:
    def test_random_matrix_near_half(self):
        M = _random_matrix(T=600, N=20, seed=7)
        pbo, details = compute_pbo(M, n_partitions=8)
        # Asymptotically PBO → 0.5 for iid noise; 600 bars × 8 partitions
        # fluctuates but should land in a broad window around the mean.
        assert 0.30 < pbo < 0.70
        assert len(details) == comb(8, 4)

    def test_genuine_winner_gives_low_pbo(self):
        M = _matrix_with_one_winner(T=600, n_noise=15, winner_drift=0.004)
        pbo, details = compute_pbo(M, n_partitions=8)
        assert pbo < 0.30
        # The champion picked IS should almost always be "winner"
        top_pick = details["is_best_strategy"].value_counts().idxmax()
        assert top_pick == "winner"

    def test_more_noise_strategies_raises_pbo(self):
        """Holding the signal fixed, an all-noise pool has more headroom for
        a lucky IS champion to fail OOS than a pool dominated by a single
        strong winner. Extreme contrast keeps the comparison stable."""
        strong_winner = _matrix_with_one_winner(
            T=600, n_noise=5, winner_drift=0.005, seed=42
        )
        all_noise = _random_matrix(T=600, N=30, seed=42)
        pbo_strong, _ = compute_pbo(strong_winner, n_partitions=8)
        pbo_noise, _ = compute_pbo(all_noise, n_partitions=8)
        # Signal-dominated pool → PBO ~ 0; pure noise → PBO ~ 0.5.
        assert pbo_noise > pbo_strong
        assert pbo_strong < 0.25
        assert pbo_noise > 0.25

    def test_combination_count_matches_binomial(self):
        M = _random_matrix(T=400, N=10, seed=0)
        for S in (4, 6, 8, 10):
            _, details = compute_pbo(M, n_partitions=S)
            assert len(details) == comb(S, S // 2)

    def test_details_columns(self):
        M = _random_matrix(T=400, N=10, seed=0)
        _, details = compute_pbo(M, n_partitions=6)
        assert set(details.columns) == {
            "combination_id",
            "is_best_strategy",
            "oos_rank",
            "logit",
        }
        # rank must land within [1, N] modulo mid-rank ties
        assert details["oos_rank"].between(1, 10).all()

    def test_pbo_in_unit_interval(self):
        M = _random_matrix(T=500, N=12, seed=99)
        pbo, _ = compute_pbo(M, n_partitions=10)
        assert 0.0 <= pbo <= 1.0


class TestValidation:
    def test_odd_partitions_rejected(self):
        M = _random_matrix(100, 5)
        with pytest.raises(ValueError, match="even"):
            compute_pbo(M, n_partitions=5)

    def test_too_few_partitions_rejected(self):
        M = _random_matrix(100, 5)
        with pytest.raises(ValueError):
            compute_pbo(M, n_partitions=0)

    def test_empty_matrix_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            compute_pbo(pd.DataFrame(), n_partitions=4)

    def test_single_strategy_rejected(self):
        M = _random_matrix(100, 1)
        with pytest.raises(ValueError, match="at least 2"):
            compute_pbo(M, n_partitions=4)

    def test_too_few_rows_rejected(self):
        M = _random_matrix(5, 5)
        with pytest.raises(ValueError, match="at least"):
            compute_pbo(M, n_partitions=10)


class TestValidatePBO:
    def test_below_threshold_passes(self):
        ok, msg = validate_pbo(0.25, max_pbo=0.40)
        assert ok
        assert "informative" in msg

    def test_between_threshold_and_half_fails_but_not_worse_than_random(self):
        ok, msg = validate_pbo(0.45, max_pbo=0.40)
        assert not ok
        assert "overfitting" in msg.lower()

    def test_above_half_flagged_worse_than_random(self):
        ok, msg = validate_pbo(0.65, max_pbo=0.40)
        assert not ok
        assert "worse than random" in msg

    def test_out_of_range_pbo_raises(self):
        with pytest.raises(ValueError):
            validate_pbo(1.5)
        with pytest.raises(ValueError):
            validate_pbo(-0.1)

    def test_custom_threshold(self):
        ok, _ = validate_pbo(0.35, max_pbo=0.30)
        assert not ok
        ok2, _ = validate_pbo(0.25, max_pbo=0.30)
        assert ok2
