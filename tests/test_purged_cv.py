"""Tests for purged k-fold CV with embargo (AFML Ch. 7)."""

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.purged_cv import (
    PurgedKFoldCV,
    cross_val_score_purged,
    purged_train_test_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int, label_duration: int = 1, freq: str = "1D"):
    """Build synthetic X/y/labels_df where each sample's label spans N bars."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    }, index=idx)
    y = pd.Series(rng.integers(0, 2, size=n), index=idx, name="y")

    event_start = idx
    # Label lifespan = label_duration bars, clipped to the series end.
    end_positions = np.clip(np.arange(n) + label_duration - 1, 0, n - 1)
    event_end = idx[end_positions]
    labels_df = pd.DataFrame(
        {"event_start": event_start, "event_end": event_end},
        index=idx,
    )
    return X, y, labels_df


# ---------------------------------------------------------------------------
# PurgedKFoldCV basics
# ---------------------------------------------------------------------------

class TestPurgedKFoldBasics:
    def test_yields_n_splits_folds(self):
        X, y, labels = _make_dataset(100)
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.0)
        splits = list(cv.split(X, y, labels))
        assert len(splits) == 5

    def test_test_folds_cover_all_positions(self):
        X, y, labels = _make_dataset(100)
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.0)
        all_test = np.concatenate([test for _, test in cv.split(X, y, labels)])
        assert len(all_test) == 100
        assert set(all_test.tolist()) == set(range(100))

    def test_train_and_test_dont_overlap(self):
        X, y, labels = _make_dataset(80)
        cv = PurgedKFoldCV(n_splits=4, embargo_pct=0.05)
        for train, test in cv.split(X, y, labels):
            assert len(np.intersect1d(train, test)) == 0

    def test_get_n_splits(self):
        cv = PurgedKFoldCV(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            PurgedKFoldCV(n_splits=1)
        with pytest.raises(ValueError):
            PurgedKFoldCV(embargo_pct=1.5)

    def test_rejects_misaligned_inputs(self):
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        labels = pd.DataFrame(
            {"event_start": [], "event_end": []},
            index=pd.DatetimeIndex([]),
        )
        cv = PurgedKFoldCV(n_splits=2)
        with pytest.raises(ValueError):
            list(cv.split(X, y, labels))


# ---------------------------------------------------------------------------
# Purging semantics
# ---------------------------------------------------------------------------

class TestPurging:
    def test_non_overlapping_labels_keep_all_training(self):
        # Each label spans a single bar → no overlap possible.
        X, y, labels = _make_dataset(100, label_duration=1)
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.0)
        for train, test in cv.split(X, y, labels):
            expected_train_count = 100 - len(test)
            assert len(train) == expected_train_count

    def test_overlapping_labels_are_purged(self):
        """
        Construct a case where labels near the fold boundary overlap into the
        test window. Those samples must be excluded from training.
        """
        n = 100
        X, y, labels = _make_dataset(n, label_duration=5)
        # Folds of size 20. Fold 1 = [0,20). The last couple of labels in
        # [0, 20) span into the test fold [20, 40), so they overlap the test
        # labels; but we want to check the OTHER direction too: labels in
        # [14, 20) span up to position 18, 19, 20, 21, 22, 23. Those that
        # span into [20, 40) overlap test.
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.0)
        splits = list(cv.split(X, y, labels))
        _, test_fold1 = splits[1]  # test is [20, 40)
        train_fold1, _ = splits[1]

        # Samples at positions 16..19 span label windows that reach into the
        # test fold — they must be purged.
        for leaky in [16, 17, 18, 19]:
            assert leaky not in train_fold1, (
                f"purging failed: position {leaky} leaks into test fold"
            )

        # Samples far from the test fold must be retained (e.g. position 5).
        assert 5 in train_fold1

    def test_purged_count_exceeds_test_size_when_labels_overlap(self):
        n = 100
        X, y, labels = _make_dataset(n, label_duration=5)
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.0)
        for train, test in cv.split(X, y, labels):
            removed = n - len(train) - len(test)
            # With label_duration=5 and fold_size=20, each non-edge fold
            # should purge ~4 adjacent training samples on each side.
            assert removed > 0, "expected some samples to be purged"


# ---------------------------------------------------------------------------
# Embargo semantics
# ---------------------------------------------------------------------------

class TestEmbargo:
    def test_embargo_removes_forward_samples(self):
        """With no label overlap, a 5% embargo on 100 samples removes the
        5 samples immediately after each test fold."""
        X, y, labels = _make_dataset(100, label_duration=1)
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.05)
        splits = list(cv.split(X, y, labels))
        train_fold0, test_fold0 = splits[0]  # test = [0, 20), embargo = [20, 25)
        # Samples 20..24 should be embargoed; samples 25..99 should be in train.
        for pos in range(20, 25):
            assert pos not in train_fold0
        for pos in range(25, 100):
            assert pos in train_fold0

    def test_zero_embargo_keeps_adjacent_samples(self):
        X, y, labels = _make_dataset(100, label_duration=1)
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.0)
        train_fold0, test_fold0 = next(cv.split(X, y, labels))
        # Position 20 (just after fold 0) must be in train.
        assert 20 in train_fold0


# ---------------------------------------------------------------------------
# purged_train_test_split
# ---------------------------------------------------------------------------

class TestPurgedTrainTestSplit:
    def test_basic_split_shapes(self):
        X, y, labels = _make_dataset(100, label_duration=1)
        X_tr, X_te, y_tr, y_te = purged_train_test_split(
            X, y, labels, test_size=0.2, embargo_pct=0.0,
        )
        assert len(X_te) == 20
        assert len(X_tr) == 80
        assert len(y_tr) == len(X_tr)
        assert len(y_te) == len(X_te)

    def test_test_is_most_recent(self):
        X, y, labels = _make_dataset(50, label_duration=1)
        X_tr, X_te, _, _ = purged_train_test_split(
            X, y, labels, test_size=0.3,
        )
        # Test timestamps should all be >= train timestamps.
        assert X_te.index.min() > X_tr.index.max()

    def test_purging_shrinks_train(self):
        # With a long label lifespan, some tail train samples must be purged.
        X, y, labels = _make_dataset(100, label_duration=10)
        X_tr, X_te, _, _ = purged_train_test_split(
            X, y, labels, test_size=0.2, embargo_pct=0.0,
        )
        # Without purging, train would have 80 samples. With label_duration=10
        # and test starting at position 80, samples 71..79 overlap into the
        # test window and must be removed.
        assert len(X_tr) < 80

    def test_rejects_bad_test_size(self):
        X, y, labels = _make_dataset(10)
        with pytest.raises(ValueError):
            purged_train_test_split(X, y, labels, test_size=0.0)
        with pytest.raises(ValueError):
            purged_train_test_split(X, y, labels, test_size=1.0)


# ---------------------------------------------------------------------------
# cross_val_score_purged
# ---------------------------------------------------------------------------

class TestCrossValScorePurged:
    def test_returns_n_scores(self):
        from sklearn.dummy import DummyClassifier

        X, y, labels = _make_dataset(100, label_duration=2)
        scores = cross_val_score_purged(
            DummyClassifier(strategy="most_frequent"),
            X, y, labels, n_splits=5, embargo_pct=0.01,
            scoring="accuracy",
        )
        assert scores.shape == (5,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_supports_sample_weight(self):
        from sklearn.linear_model import LogisticRegression

        X, y, labels = _make_dataset(120, label_duration=1)
        # Make y learnable from features so LogReg converges.
        y = (X["f1"] + X["f2"] > 0).astype(int)
        weights = pd.Series(np.ones(len(X)), index=X.index)
        scores = cross_val_score_purged(
            LogisticRegression(max_iter=200),
            X, y, labels, n_splits=4, embargo_pct=0.0,
            sample_weight=weights, scoring="accuracy",
        )
        assert scores.shape == (4,)
        assert (scores > 0.5).mean() >= 0.5  # usually beats chance

    def test_log_loss_vs_neg_log_loss(self):
        """log_loss returns the positive loss; neg_log_loss returns its negation."""
        from sklearn.dummy import DummyClassifier

        X, y, labels = _make_dataset(80, label_duration=1)
        est = DummyClassifier(strategy="prior")
        ll = cross_val_score_purged(
            est, X, y, labels, n_splits=4, scoring="log_loss",
        )
        nll = cross_val_score_purged(
            est, X, y, labels, n_splits=4, scoring="neg_log_loss",
        )
        # Scores are deterministic for DummyClassifier, so ll ≈ -nll.
        assert np.allclose(ll, -nll)
        assert (ll >= 0).all()

    def test_rejects_unknown_scoring(self):
        from sklearn.dummy import DummyClassifier
        X, y, labels = _make_dataset(40)
        with pytest.raises(ValueError):
            cross_val_score_purged(
                DummyClassifier(), X, y, labels, scoring="mystery",
            )

    def test_roc_auc_needs_proba(self):
        from sklearn.linear_model import LogisticRegression
        X, y, labels = _make_dataset(100, label_duration=1)
        y = (X["f1"] > 0).astype(int)
        scores = cross_val_score_purged(
            LogisticRegression(max_iter=200),
            X, y, labels, n_splits=5, scoring="roc_auc",
        )
        assert scores.shape == (5,)
        assert np.all((scores >= 0) & (scores <= 1))


# ---------------------------------------------------------------------------
# Standard K-Fold vs Purged K-Fold — demonstrate the difference
# ---------------------------------------------------------------------------

class TestComparisonWithStandardKFold:
    def test_standard_kfold_includes_leaky_samples_purged_excludes(self):
        """
        With overlapping labels, a sample at fold boundary leaks info.
        Standard KFold keeps it in the training set; Purged KFold removes it.
        """
        from sklearn.model_selection import KFold

        n = 100
        X, y, labels = _make_dataset(n, label_duration=5)
        standard = KFold(n_splits=5, shuffle=False)
        purged = PurgedKFoldCV(n_splits=5, embargo_pct=0.0)

        for (std_train, std_test), (prg_train, prg_test) in zip(
            standard.split(X), purged.split(X, y, labels),
        ):
            # Test folds are the same (both use contiguous slicing).
            assert np.array_equal(std_test, prg_test)
            # Purged train must be a strict subset of standard train.
            assert set(prg_train.tolist()).issubset(set(std_train.tolist()))
            # At least for non-edge folds, purged is strictly smaller.
            if std_test[0] not in (0,) and std_test[-1] != n - 1:
                assert len(prg_train) < len(std_train)
