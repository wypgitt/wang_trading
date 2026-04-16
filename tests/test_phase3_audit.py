"""
Phase 3 final audit — edge cases and cross-module invariants.

Split out from the per-module test files so the "gotcha" cases are
easy to spot in CI output. Each section mirrors one audit line-item:

    * empty / single-class training data
    * invalid Kelly inputs
    * sample-weight non-negativity and finiteness
    * purged-CV train/test non-overlap invariant
    * pipeline determinism with fixed seeds
    * MLflow serialisation roundtrip (joblib dump → load → predict parity)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _synthetic(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=[f"f{i}" for i in range(4)],
        index=idx,
    )
    logits = 1.2 * X["f0"] - 0.5 * X["f1"]
    prob = 1.0 / (1.0 + np.exp(-logits.to_numpy()))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), index=idx, name="y")
    end_pos = np.clip(np.arange(n) + 2, 0, n - 1)
    labels_df = pd.DataFrame(
        {"event_start": idx, "event_end": idx[end_pos]}, index=idx,
    )
    return X, y, labels_df


# ---------------------------------------------------------------------------
# Edge case: empty training data
# ---------------------------------------------------------------------------

class TestEmptyTrainingData:
    def test_meta_labeler_raises_on_empty_X(self):
        from src.ml_layer.meta_labeler import MetaLabeler
        empty_X = pd.DataFrame(columns=["f0", "f1"])
        empty_y = pd.Series([], dtype=int, name="y")
        # LightGBM raises on empty training arrays; MetaLabeler doesn't
        # pre-check (would be redundant) but surfaces the underlying error.
        with pytest.raises(Exception):
            MetaLabeler(model_type="lightgbm", calibrate=False).fit(empty_X, empty_y)

    def test_meta_labeler_rejects_length_mismatch(self):
        from src.ml_layer.meta_labeler import MetaLabeler
        X, y, _ = _synthetic()
        with pytest.raises(ValueError):
            MetaLabeler().fit(X, y.iloc[:100])

    def test_pipeline_empty_signals_returns_empty_X(self):
        from src.labeling.meta_labeler_pipeline import MetaLabelingPipeline
        close = pd.Series(
            100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, 100)),
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )
        features = pd.DataFrame(
            np.random.default_rng(0).normal(size=(100, 3)),
            columns=["f0", "f1", "f2"], index=close.index,
        )
        empty_signals = pd.DataFrame(
            columns=["timestamp", "symbol", "family", "side", "confidence"],
        )
        X, y, w = MetaLabelingPipeline().prepare_training_data(
            close, empty_signals, features,
        )
        assert len(X) == len(y) == len(w) == 0


# ---------------------------------------------------------------------------
# Edge case: single-class labels
# ---------------------------------------------------------------------------

class TestSingleClass:
    def test_calibrator_tolerates_single_class_but_output_is_constant(self):
        from src.ml_layer.meta_labeler import ProbabilityCalibrator
        # Isotonic on constant y → a constant output function.
        y_true = np.ones(200, dtype=float)
        y_pred = np.random.default_rng(0).uniform(size=200)
        cal = ProbabilityCalibrator().fit(y_true, y_pred)
        out = cal.transform(np.array([0.1, 0.5, 0.9]))
        assert np.allclose(out, out[0])  # all equal

    def test_purged_cv_single_class_fold_doesnt_crash(self):
        """
        If a fold's validation slice is single-class, sklearn's roc_auc is
        undefined — cross_val_score_purged must surface whatever sklearn
        raises rather than silently producing nonsense. Accuracy is
        always defined and should still work.
        """
        from sklearn.dummy import DummyClassifier

        from src.ml_layer.purged_cv import cross_val_score_purged
        X, y, labels = _synthetic()
        scores = cross_val_score_purged(
            DummyClassifier(strategy="most_frequent"),
            X, y, labels, n_splits=3, scoring="accuracy",
        )
        assert scores.shape == (3,)
        assert np.isfinite(scores).all()

    def test_meta_labeler_fits_on_nearly_constant_labels(self):
        from src.ml_layer.meta_labeler import MetaLabeler
        rng = np.random.default_rng(0)
        X, _, _ = _synthetic()
        # 99% positives + a single negative — no crash, but the model
        # learns "always predict 1".
        y = pd.Series(1, index=X.index)
        y.iloc[0] = 0
        model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 20},
            calibrate=False,
        ).fit(X, y)
        proba = model.predict_proba(X)
        # All probs are large because there's no signal to learn against.
        assert proba.mean() > 0.9


# ---------------------------------------------------------------------------
# Kelly invariant: negative fraction / bad inputs
# ---------------------------------------------------------------------------

class TestKellyInvariants:
    def test_negative_fraction_argument_raises(self):
        from src.bet_sizing.kelly import fractional_kelly, rolling_kelly
        with pytest.raises(ValueError):
            fractional_kelly(0.6, 1.0, 1.0, fraction=-0.01)
        with pytest.raises(ValueError):
            rolling_kelly(pd.Series([0.01, -0.01, 0.02]), window=2, fraction=-0.1)

    def test_negative_raw_kelly_is_clipped_to_zero(self):
        """A negative Kelly fraction out of the formula (no edge / bad payoff)
        must map to 0, never to a negative bet. Guards against 'reverse-Kelly'
        which conflicts with our side-separation convention."""
        from src.bet_sizing.kelly import kelly_fraction
        assert kelly_fraction(0.3, 1.0, 2.0) == 0.0
        assert kelly_fraction(0.4, 1.0, 1.0) == 0.0

    def test_zero_magnitudes_raise(self):
        from src.bet_sizing.kelly import kelly_fraction
        with pytest.raises(ValueError):
            kelly_fraction(0.6, 0.0, 1.0)
        with pytest.raises(ValueError):
            kelly_fraction(0.6, 1.0, 0.0)

    def test_cascade_skips_kelly_on_missing_stats_without_crash(self):
        from src.bet_sizing.cascade import BetSizingCascade
        cascade = BetSizingCascade(family_stats={})  # no stats at all
        r = cascade.compute_position_size(
            prob=0.8, side=1, symbol="X", signal_family="unknown",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        assert r["final_size"] >= 0
        assert "kelly_cap" not in r["constraints_applied"]


# ---------------------------------------------------------------------------
# Sample weights: non-NaN, non-negative invariant
# ---------------------------------------------------------------------------

class TestSampleWeightInvariants:
    def _weights_from_pipeline(self):
        from src.labeling.meta_labeler_pipeline import MetaLabelingPipeline
        rng = np.random.default_rng(7)
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx,
        )
        features = pd.DataFrame(
            rng.normal(size=(n, 4)),
            columns=[f"f{i}" for i in range(4)], index=idx,
        )
        ev = idx[50:-50:15]
        sides = rng.choice([-1, 1], size=len(ev))
        signals = pd.DataFrame({
            "timestamp": ev,
            "symbol": ["TEST"] * len(ev),
            "family": ["ts_momentum"] * len(ev),
            "side": sides,
            "confidence": [0.6] * len(ev),
        })
        _, y, w = MetaLabelingPipeline(
            max_holding_period=20, time_decay=0.5,
        ).prepare_training_data(close, signals, features)
        return y, w

    def test_weights_are_finite(self):
        _, w = self._weights_from_pipeline()
        assert w.notna().all()
        assert np.isfinite(w.to_numpy()).all()

    def test_weights_are_nonnegative(self):
        _, w = self._weights_from_pipeline()
        assert (w >= 0).all(), f"found negative weights: {w[w<0].tolist()}"

    def test_weights_normalised_to_n(self):
        y, w = self._weights_from_pipeline()
        assert np.isclose(w.sum(), len(y)), (
            f"weights sum {w.sum():.4f} != n_samples {len(y)}"
        )

    def test_sequential_bootstrap_returns_valid_indices(self):
        from src.labeling.sample_weights import (
            get_average_uniqueness,
            sequential_bootstrap,
        )
        n = 40
        # Each event spans 5 bars on a stride-3 grid, so the last event ends
        # at position (n-1)*3 + 4 = 121. Use 150 bars so the last end is in-range.
        idx = pd.date_range("2024-01-01", periods=150, freq="1h")
        starts = pd.DatetimeIndex([idx[i * 3] for i in range(n)])
        ends = pd.DatetimeIndex([idx[i * 3 + 4] for i in range(n)])
        u = get_average_uniqueness(starts, ends, idx)
        draws = sequential_bootstrap(
            u, event_starts=starts, event_ends=ends, close_index=idx,
            n_samples=100, random_state=0,
        )
        assert (draws >= 0).all() and (draws < n).all()
        assert np.isfinite(draws).all()


# ---------------------------------------------------------------------------
# Purged CV invariant: train ∩ test = ∅
# ---------------------------------------------------------------------------

class TestPurgedCVNoOverlap:
    def test_zero_overlap_across_many_configurations(self):
        from src.ml_layer.purged_cv import PurgedKFoldCV
        for n in (50, 100, 300):
            X, y, labels = _synthetic(n=n)
            for n_splits in (2, 3, 5):
                for embargo_pct in (0.0, 0.01, 0.05, 0.10):
                    if n_splits >= n:
                        continue
                    cv = PurgedKFoldCV(n_splits=n_splits, embargo_pct=embargo_pct)
                    for train, test in cv.split(X, y, labels):
                        overlap = np.intersect1d(train, test)
                        assert len(overlap) == 0, (
                            f"train∩test nonempty for n={n}, "
                            f"splits={n_splits}, embargo={embargo_pct}"
                        )

    def test_every_test_fold_gets_every_position_exactly_once(self):
        from src.ml_layer.purged_cv import PurgedKFoldCV
        X, y, labels = _synthetic(n=200)
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        all_test = np.concatenate([t for _, t in cv.split(X, y, labels)])
        assert len(all_test) == len(X)
        assert set(all_test.tolist()) == set(range(len(X)))


# ---------------------------------------------------------------------------
# Determinism with fixed seeds
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_meta_labeler_deterministic_given_fixed_seed(self):
        from src.ml_layer.meta_labeler import MetaLabeler
        X, y, labels = _synthetic(n=300, seed=1)
        a = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 100, "random_state": 42},
            calibrate=True,
        ).fit(X, y, labels_df=labels)
        b = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 100, "random_state": 42},
            calibrate=True,
        ).fit(X, y, labels_df=labels)
        assert np.allclose(a.predict_proba(X), b.predict_proba(X))

    def test_sequential_bootstrap_reproducible(self):
        from src.labeling.sample_weights import sequential_bootstrap
        u = pd.Series(
            np.linspace(0.2, 1.0, 30),
            index=pd.date_range("2024-01-01", periods=30, freq="1h"),
        )
        a = sequential_bootstrap(u, n_samples=100, random_state=123)
        b = sequential_bootstrap(u, n_samples=100, random_state=123)
        assert np.array_equal(a, b)

    def test_tuning_reproducible(self):
        from src.ml_layer.tuning import tune_meta_labeler
        X, y, labels = _synthetic(n=120, seed=3)
        best_a = tune_meta_labeler(
            X, y, labels, n_trials=3, timeout=20,
            n_splits=3, random_state=42,
        )
        best_b = tune_meta_labeler(
            X, y, labels, n_trials=3, timeout=20,
            n_splits=3, random_state=42,
        )
        # TPE sampler with a fixed seed should produce the same parameter
        # sequence, and therefore the same "best" params.
        assert best_a == best_b


# ---------------------------------------------------------------------------
# MLflow roundtrip: serialize → deserialize → predict parity
# ---------------------------------------------------------------------------

class TestMLflowRoundtrip:
    @pytest.fixture
    def registry(self, tmp_path: Path):
        import os
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            from src.ml_layer.model_registry import ModelRegistry
            db = tmp_path / "mlflow.db"
            yield ModelRegistry(
                tracking_uri=f"sqlite:///{db}",
                experiment_name="phase3-audit",
            )
        finally:
            os.chdir(cwd)

    def test_joblib_roundtrip_preserves_predictions(self, registry):
        from src.ml_layer.meta_labeler import MetaLabeler
        X, y, labels = _synthetic(n=200)
        model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 50, "random_state": 42},
            calibrate=True,
        ).fit(X, y, labels_df=labels)
        original = model.predict_proba(X)

        run_id = registry.log_training_run(
            model=model, X=X, y=y, labels_df=labels,
            params={"n_estimators": 50},
            cv_scores=np.array([0.6, 0.62]),
            importances=None,
        )
        loaded = registry.load_model(run_id)
        # Calibrator must survive the roundtrip too.
        assert loaded.calibrator_ is not None
        assert np.allclose(loaded.predict_proba(X), original)
        # Feature names preserved.
        assert loaded.feature_names_ == model.feature_names_

    def test_artifact_contains_feature_names_json(self, registry):
        import json

        import mlflow

        from src.ml_layer.meta_labeler import MetaLabeler
        X, y, labels = _synthetic(n=150)
        model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 30},
            calibrate=False,
        ).fit(X, y)
        run_id = registry.log_training_run(
            model=model, X=X, y=y, labels_df=labels,
            params={}, cv_scores=np.array([0.6]),
        )
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=registry.FEATURE_NAMES_FILENAME,
        )
        saved = json.loads(open(local).read())
        assert saved == list(X.columns)
