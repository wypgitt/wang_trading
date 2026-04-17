"""Tests for CPCVEngine and the §9.1 validation gate."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from src.backtesting.cpcv import CPCVEngine, validate_strategy
from src.backtesting.walk_forward import BacktestResult


def _make_labels_df(n: int, horizon: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    event_start = idx
    event_end = idx.shift(horizon, freq="B")
    labels_df = pd.DataFrame(
        {"event_start": event_start, "event_end": event_end}, index=idx
    )
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 3)), index=idx)
    y = pd.Series(np.random.default_rng(1).integers(0, 2, n), index=idx)
    return X, labels_df, y


def _fake_result(total_return: float, sharpe: float = 0.5) -> BacktestResult:
    equity = pd.Series(
        [100_000, 100_000 * (1 + total_return)],
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    return BacktestResult(
        trades=[],
        equity_curve=equity,
        returns=equity.pct_change().fillna(0),
        drawdown_curve=pd.Series([0.0, min(0.0, total_return)], index=equity.index),
        metrics={
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": min(0.0, total_return),
            "win_rate": 0.55,
            "profit_factor": 1.5,
            "total_trades": 10,
        },
    )


class TestGeneratePaths:
    def test_45_paths_for_N10_k2(self):
        n = 300
        X, labels_df, y = _make_labels_df(n)
        engine = CPCVEngine(n_groups=10, n_test_groups=2, embargo_pct=0.0)
        paths = engine.generate_paths(X, y, labels_df)
        assert len(paths) == comb(10, 2) == 45

    def test_path_count_matches_binomial(self):
        n = 200
        X, labels_df, y = _make_labels_df(n)
        for N, k in [(6, 2), (8, 3), (5, 1)]:
            engine = CPCVEngine(n_groups=N, n_test_groups=k, embargo_pct=0.0)
            paths = engine.generate_paths(X, y, labels_df)
            assert len(paths) == comb(N, k)

    def test_train_and_test_dont_overlap_in_any_path(self):
        n = 300
        X, labels_df, y = _make_labels_df(n)
        engine = CPCVEngine(n_groups=10, n_test_groups=2, embargo_pct=0.01)
        paths = engine.generate_paths(X, y, labels_df)
        for path in paths:
            train_idx, test_idx = path[0]
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_union_of_all_test_folds_covers_every_sample(self):
        """Every sample position appears in at least one test fold across
        the 45 paths. Each group is part of C(N-1, k-1) = 9 combinations."""
        n = 300
        X, labels_df, y = _make_labels_df(n)
        engine = CPCVEngine(n_groups=10, n_test_groups=2, embargo_pct=0.0)
        paths = engine.generate_paths(X, y, labels_df)
        union = np.unique(np.concatenate([p[0][1] for p in paths]))
        assert union.tolist() == list(range(n))

    def test_each_sample_appears_in_expected_number_of_test_folds(self):
        """With N=10 groups and k=2, each group should appear in C(9,1)=9 combos."""
        n = 300
        X, labels_df, y = _make_labels_df(n)
        engine = CPCVEngine(n_groups=10, n_test_groups=2, embargo_pct=0.0)
        paths = engine.generate_paths(X, y, labels_df)
        all_test = np.concatenate([p[0][1] for p in paths])
        counts = np.bincount(all_test, minlength=n)
        # Every position should be in exactly 9 test folds
        assert (counts == 9).all()

    def test_purging_excludes_overlapping_training_samples(self):
        """A sample whose label period overlaps the test span must not appear
        in the training set."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        # Horizon = 10 bars → adjacent samples' label periods heavily overlap
        labels_df = pd.DataFrame(
            {
                "event_start": idx,
                "event_end": idx.shift(10, freq="B"),
            },
            index=idx,
        )
        X = pd.DataFrame(np.zeros((n, 2)), index=idx)
        y = pd.Series(np.zeros(n), index=idx)

        engine = CPCVEngine(n_groups=10, n_test_groups=2, embargo_pct=0.0)
        paths = engine.generate_paths(X, y, labels_df)

        # Confirm that on at least one path, samples adjacent to the test
        # block were purged (i.e., train+test < n).
        any_purged = False
        for path in paths:
            train_idx, test_idx = path[0]
            if len(train_idx) + len(test_idx) < n:
                any_purged = True
                break
        assert any_purged, "purging should remove some overlapping samples"


class TestAssembleAndStats:
    def test_assemble_equity_curves_shape(self):
        results = [_fake_result(0.01 * i) for i in range(-5, 40)]  # 45 results
        engine = CPCVEngine()
        df = engine.assemble_equity_curves(results)
        assert df.shape[1] == 45

    def test_get_path_statistics_includes_summary(self):
        results = [_fake_result(0.01 * i) for i in range(-5, 40)]
        engine = CPCVEngine()
        stats = engine.get_path_statistics(results)
        assert "path_0" in stats.index
        for key in ("mean", "std", "min", "max"):
            assert key in stats.index
        assert stats.loc["mean", "sharpe"] == pytest.approx(0.5)

    def test_empty_results_handled(self):
        engine = CPCVEngine()
        assert engine.assemble_equity_curves([]).empty
        assert engine.get_path_statistics([]).empty


class TestValidateStrategy:
    def test_profitable_strategy_passes(self):
        # 40/45 paths are profitable -> 88.9% > 60%
        results = [_fake_result(0.02) for _ in range(40)] + [
            _fake_result(-0.01) for _ in range(5)
        ]
        passed, stats = validate_strategy(results)
        assert passed
        assert stats["positive_count"] == 40
        assert stats["positive_pct"] == pytest.approx(40 / 45)
        assert stats["path_count"] == 45

    def test_random_strategy_fails_gate(self):
        rng = np.random.default_rng(3)
        # Symmetric noise centred at 0 → ~50% positive, fails 60% gate
        results = [_fake_result(rng.normal(0, 0.01)) for _ in range(45)]
        passed, stats = validate_strategy(results)
        assert not passed
        assert stats["positive_pct"] < 0.60

    def test_empty_results_returns_false(self):
        passed, stats = validate_strategy([])
        assert passed is False
        assert stats["path_count"] == 0

    def test_threshold_is_configurable(self):
        # 55% positive: fails default 60% but passes at 50%
        results = [_fake_result(0.01) for _ in range(25)] + [
            _fake_result(-0.01) for _ in range(20)
        ]
        passed_default, _ = validate_strategy(results)
        passed_loose, _ = validate_strategy(results, min_positive_paths_pct=0.50)
        assert not passed_default
        assert passed_loose


class TestValidation:
    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            CPCVEngine(n_groups=1)
        with pytest.raises(ValueError):
            CPCVEngine(n_groups=5, n_test_groups=5)
        with pytest.raises(ValueError):
            CPCVEngine(n_groups=5, n_test_groups=0)
        with pytest.raises(ValueError):
            CPCVEngine(embargo_pct=1.5)

    def test_rejects_missing_label_columns(self):
        X = pd.DataFrame(np.zeros((20, 2)))
        bad_labels = pd.DataFrame({"foo": range(20)})
        engine = CPCVEngine(n_groups=4, n_test_groups=1)
        with pytest.raises(ValueError, match="event_start"):
            engine.generate_paths(X, None, bad_labels)

    def test_rejects_misaligned_lengths(self):
        X, labels_df, y = _make_labels_df(100)
        labels_short = labels_df.iloc[:50]
        engine = CPCVEngine(n_groups=4, n_test_groups=1)
        with pytest.raises(ValueError, match="align"):
            engine.generate_paths(X, y, labels_short)
