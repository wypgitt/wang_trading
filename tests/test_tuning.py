"""Tests for Optuna-based meta-labeler tuning (design doc §7.1)."""

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.meta_labeler import MetaLabeler
from src.ml_layer.tuning import (
    create_objective,
    retrain_with_best_params,
    tune_meta_labeler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=["f0", "f1", "f2", "f3"],
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    logits = 1.0 * X["f0"] - 0.8 * X["f1"] + 0.3 * X["f2"]
    prob = 1.0 / (1.0 + np.exp(-logits.to_numpy()))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int),
                  index=X.index, name="y")
    # Labels span 3 bars each (small overlap so purging has something to do).
    end_pos = np.clip(np.arange(n) + 2, 0, n - 1)
    labels_df = pd.DataFrame(
        {"event_start": X.index, "event_end": X.index[end_pos]},
        index=X.index,
    )
    return X, y, labels_df


_EXPECTED_KEYS = {
    "learning_rate", "n_estimators", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
}


# ---------------------------------------------------------------------------
# create_objective
# ---------------------------------------------------------------------------

class TestCreateObjective:
    def test_returns_callable(self):
        X, y, labels = _synthetic_dataset()
        obj = create_objective(X, y, labels)
        assert callable(obj)

    def test_rejects_unknown_model(self):
        X, y, labels = _synthetic_dataset()
        with pytest.raises(ValueError):
            create_objective(X, y, labels, model_type="catboost")

    def test_rejects_unknown_scoring(self):
        X, y, labels = _synthetic_dataset()
        with pytest.raises(ValueError):
            create_objective(X, y, labels, scoring="mystery")


# ---------------------------------------------------------------------------
# tune_meta_labeler
# ---------------------------------------------------------------------------

class TestTuneMetaLabeler:
    def test_runs_5_trials_under_30s(self):
        import time
        X, y, labels = _synthetic_dataset(n=150, seed=1)
        t0 = time.perf_counter()
        best = tune_meta_labeler(
            X, y, labels,
            n_trials=5, timeout=30,
            n_splits=3, model_type="lightgbm",
            scoring="neg_log_loss",
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"tuning took {elapsed:.2f}s"
        # Study must contain at least 5 trials (complete or pruned).
        study = tune_meta_labeler.last_study_
        assert len(study.trials) >= 5

    def test_best_params_has_expected_keys(self):
        X, y, labels = _synthetic_dataset(n=150, seed=2)
        best = tune_meta_labeler(
            X, y, labels,
            n_trials=5, timeout=30,
            n_splits=3, model_type="lightgbm",
            scoring="neg_log_loss",
        )
        assert _EXPECTED_KEYS.issubset(best.keys())
        # Value ranges match the sampler configuration.
        assert 0.005 <= best["learning_rate"] <= 0.3
        assert 100 <= best["n_estimators"] <= 2000
        assert 3 <= best["max_depth"] <= 10

    def test_random_forest_backend_returns_rf_keys(self):
        X, y, labels = _synthetic_dataset(n=120, seed=3)
        best = tune_meta_labeler(
            X, y, labels,
            n_trials=3, timeout=20,
            n_splits=3, model_type="random_forest",
            scoring="accuracy",
        )
        # RF still samples from the full search space (so we get the same
        # best_params keys regardless of backend), but the RF-only builder
        # uses n_estimators/max_depth/min_child_weight and ignores the rest.
        assert "n_estimators" in best
        assert "max_depth" in best
        assert "min_child_weight" in best

    def test_sample_weight_plumbed_through(self):
        # Smoke-test: passing a weight Series should not blow up the pipeline.
        X, y, labels = _synthetic_dataset(n=120, seed=4)
        weights = pd.Series(
            np.ones(len(X)) + np.linspace(0, 1, len(X)), index=X.index,
        )
        best = tune_meta_labeler(
            X, y, labels, sample_weight=weights,
            n_trials=3, timeout=20, n_splits=3,
            model_type="lightgbm",
        )
        assert _EXPECTED_KEYS.issubset(best.keys())

    def test_study_cached_on_function(self):
        X, y, labels = _synthetic_dataset(n=100, seed=5)
        tune_meta_labeler(X, y, labels, n_trials=3, timeout=15, n_splits=3)
        import optuna
        study = tune_meta_labeler.last_study_
        assert isinstance(study, optuna.study.Study)
        assert len(study.trials) >= 3


# ---------------------------------------------------------------------------
# retrain_with_best_params
# ---------------------------------------------------------------------------

class TestRetrainWithBestParams:
    def test_produces_fitted_model(self):
        X, y, labels = _synthetic_dataset(n=150, seed=6)
        best = tune_meta_labeler(
            X, y, labels,
            n_trials=3, timeout=20, n_splits=3,
            model_type="lightgbm",
        )
        model = retrain_with_best_params(
            best, X, y, labels, model_type="lightgbm",
        )
        assert isinstance(model, MetaLabeler)
        assert model.model_ is not None
        # Forwarding labels_df engaged the purged-CV path, so OOF preds exist.
        assert model.oof_predictions_ is not None
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_respects_calibrate_flag(self):
        X, y, labels = _synthetic_dataset(n=120, seed=7)
        best = tune_meta_labeler(
            X, y, labels, n_trials=3, timeout=20, n_splits=3,
        )
        raw = retrain_with_best_params(
            best, X, y, labels, calibrate=False,
        )
        assert raw.calibrator_ is None
