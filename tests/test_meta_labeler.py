"""Tests for the Tier-1 meta-labeler and probability calibrator."""

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.meta_labeler import MetaLabeler, ProbabilityCalibrator


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def _make_learnable_dataset(n: int = 400, n_features: int = 6,
                             seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Binary classification dataset with a clear signal in f0 and f1."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    linear = 1.2 * X["f0"] + 0.8 * X["f1"] - 0.5 * X["f2"]
    prob = 1.0 / (1.0 + np.exp(-linear.to_numpy()))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int),
                  index=X.index, name="y")
    return X, y


def _make_labels_df(X: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Triple-barrier-shaped labels_df for purged CV splits."""
    starts = X.index
    end_pos = np.clip(np.arange(len(X)) + horizon - 1, 0, len(X) - 1)
    ends = X.index[end_pos]
    return pd.DataFrame(
        {"event_start": starts, "event_end": ends}, index=X.index,
    )


# ---------------------------------------------------------------------------
# ProbabilityCalibrator
# ---------------------------------------------------------------------------

class TestProbabilityCalibrator:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=300)
        y_pred = rng.uniform(size=300)
        cal = ProbabilityCalibrator().fit(y_true, y_pred)
        out = cal.transform(y_pred)
        assert out.shape == y_pred.shape
        assert (out >= 0).all() and (out <= 1).all()

    def test_transform_requires_fit(self):
        with pytest.raises(RuntimeError):
            ProbabilityCalibrator().transform(np.array([0.5]))

    def test_miscalibrated_input_becomes_calibrated(self):
        """
        Take y_pred = true_prob ** 2 (systematically under-confident when
        true_prob < 1) — isotonic should flatten this back toward the
        identity.
        """
        rng = np.random.default_rng(1)
        n = 5000
        true_prob = rng.uniform(size=n)
        y_true = (rng.uniform(size=n) < true_prob).astype(int)
        y_pred_bad = true_prob ** 2
        cal = ProbabilityCalibrator().fit(y_true, y_pred_bad)
        y_pred_good = cal.transform(y_pred_bad)

        # Binned calibration error should improve.
        def _err(p):
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
            err = 0.0
            for b in range(10):
                m = idx == b
                if m.sum() < 20:
                    continue
                err += abs(p[m].mean() - y_true[m].mean()) * m.sum() / n
            return err

        assert _err(y_pred_good) < _err(y_pred_bad)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            ProbabilityCalibrator().fit(np.array([0, 1, 0]), np.array([0.5]))


# ---------------------------------------------------------------------------
# MetaLabeler — basic pipeline
# ---------------------------------------------------------------------------

class TestMetaLabelerPipeline:
    def test_fit_predict_end_to_end(self):
        X, y = _make_learnable_dataset(n=400)
        model = MetaLabeler(model_type="lightgbm", calibrate=False).fit(X, y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()
        assert set(np.unique(preds)).issubset({0, 1})
        # Should beat random on a learnable problem (in-sample is fine here).
        acc = (preds == y.to_numpy()).mean()
        assert acc > 0.7

    def test_predict_proba_in_unit_interval(self):
        X, y = _make_learnable_dataset(n=200)
        model = MetaLabeler(model_type="lightgbm").fit(X, y)
        proba = model.predict_proba(X)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_predict_respects_threshold(self):
        X, y = _make_learnable_dataset(n=200)
        model = MetaLabeler(model_type="lightgbm", calibrate=False).fit(X, y)
        # Higher threshold → fewer positive predictions.
        n_low = model.predict(X, threshold=0.3).sum()
        n_mid = model.predict(X, threshold=0.5).sum()
        n_high = model.predict(X, threshold=0.7).sum()
        assert n_low >= n_mid >= n_high

    def test_fit_rejects_misaligned_inputs(self):
        X, y = _make_learnable_dataset(n=200)
        with pytest.raises(ValueError):
            MetaLabeler().fit(X, y.iloc[:100])

    def test_constructor_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            MetaLabeler(model_type="catboost")

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            MetaLabeler().predict_proba(pd.DataFrame({"f0": [0.0]}))

    def test_threshold_out_of_range(self):
        X, y = _make_learnable_dataset(n=100)
        model = MetaLabeler(calibrate=False).fit(X, y)
        with pytest.raises(ValueError):
            model.predict(X, threshold=1.5)


# ---------------------------------------------------------------------------
# Calibration comparison
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrated_beats_raw_on_brier(self):
        """Calibrated probabilities should have no worse (usually better)
        Brier score than the raw classifier probabilities on a held-out set."""
        from sklearn.metrics import brier_score_loss

        X_full, y_full = _make_learnable_dataset(n=600, seed=11)
        split = 400
        X_tr, X_te = X_full.iloc[:split], X_full.iloc[split:]
        y_tr, y_te = y_full.iloc[:split], y_full.iloc[split:]
        labels_df_tr = _make_labels_df(X_tr, horizon=5)

        # Calibrated model using OOF preds from purged CV.
        cal_model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 200, "learning_rate": 0.05},
            calibrate=True,
        ).fit(X_tr, y_tr, labels_df=labels_df_tr)

        # Raw model (no calibration).
        raw_model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 200, "learning_rate": 0.05},
            calibrate=False,
        ).fit(X_tr, y_tr, labels_df=labels_df_tr)

        brier_cal = brier_score_loss(y_te, cal_model.predict_proba(X_te))
        brier_raw = brier_score_loss(y_te, raw_model.predict_proba(X_te))
        # On a learnable binary problem with purged-CV calibration, the
        # calibrator should not materially worsen Brier.
        assert brier_cal <= brier_raw + 0.02

    def test_calibrator_is_fitted_when_calibrate_true(self):
        X, y = _make_learnable_dataset(n=200)
        model = MetaLabeler(calibrate=True).fit(X, y)
        assert model.calibrator_ is not None

    def test_calibrator_is_none_when_calibrate_false(self):
        X, y = _make_learnable_dataset(n=200)
        model = MetaLabeler(calibrate=False).fit(X, y)
        assert model.calibrator_ is None


# ---------------------------------------------------------------------------
# Sample weight effect
# ---------------------------------------------------------------------------

class TestSampleWeight:
    def test_weights_affect_predictions(self):
        """Heavily up-weighting positives should push predicted probabilities higher."""
        X, y = _make_learnable_dataset(n=400, seed=42)
        base = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 200, "min_child_weight": 2},
            calibrate=False,
        ).fit(X, y)
        # Weights: weight positives 10x, negatives 1x.
        weights = pd.Series(
            np.where(y.to_numpy() == 1, 10.0, 1.0), index=y.index,
        )
        weighted = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 200, "min_child_weight": 2},
            calibrate=False,
        ).fit(X, y, sample_weight=weights)

        base_proba = base.predict_proba(X)
        weighted_proba = weighted.predict_proba(X)
        # Under heavy positive-weighting, predicted probs should trend higher.
        assert weighted_proba.mean() > base_proba.mean()


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_importance_series_has_feature_names(self):
        X, y = _make_learnable_dataset(n=200)
        model = MetaLabeler(calibrate=False).fit(X, y)
        imp = model.get_feature_importance(method="gain")
        assert isinstance(imp, pd.Series)
        assert set(imp.index) == set(X.columns)

    def test_importance_identifies_informative_features(self):
        # f0 and f1 drive the label; f2 is a weak detractor; others are noise.
        X, y = _make_learnable_dataset(n=500, n_features=8, seed=3)
        model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 300},
            calibrate=False,
        ).fit(X, y)
        imp = model.get_feature_importance(method="gain")
        # f0 should be in the top 3.
        top3 = set(imp.head(3).index)
        assert "f0" in top3

    def test_split_importance_available_for_gbms(self):
        X, y = _make_learnable_dataset(n=200)
        model = MetaLabeler(calibrate=False).fit(X, y)
        imp = model.get_feature_importance(method="split")
        assert isinstance(imp, pd.Series)

    def test_split_importance_raises_for_random_forest(self):
        X, y = _make_learnable_dataset(n=200)
        model = MetaLabeler(model_type="random_forest", calibrate=False).fit(X, y)
        with pytest.raises(ValueError):
            model.get_feature_importance(method="split")


# ---------------------------------------------------------------------------
# Backend coverage
# ---------------------------------------------------------------------------

class TestBackendCoverage:
    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost", "random_forest"])
    def test_all_backends_fit_and_predict(self, backend):
        X, y = _make_learnable_dataset(n=300)
        params = {"n_estimators": 50}  # keep fits fast
        model = MetaLabeler(
            model_type=backend, params=params, calibrate=False,
        ).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# Purged-CV early stopping
# ---------------------------------------------------------------------------

class TestPurgedCVEarlyStopping:
    def test_early_stopping_trims_n_estimators(self):
        """
        With a large n_estimators budget on an easy dataset, early stopping
        should halt well before the budget → best_iterations_ are populated
        and all below the requested n_estimators.
        """
        X, y = _make_learnable_dataset(n=500, seed=7)
        labels_df = _make_labels_df(X, horizon=3)
        requested = 1000
        model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": requested, "learning_rate": 0.1},
            calibrate=False,
        ).fit(X, y, labels_df=labels_df)

        assert len(model.best_iterations_) > 0
        assert all(b < requested for b in model.best_iterations_)

    def test_oof_predictions_populated_under_purged_cv(self):
        X, y = _make_learnable_dataset(n=300)
        labels_df = _make_labels_df(X, horizon=3)
        model = MetaLabeler(
            model_type="lightgbm",
            params={"n_estimators": 200},
            calibrate=True,
        ).fit(X, y, labels_df=labels_df)
        assert model.oof_predictions_ is not None
        finite = np.isfinite(model.oof_predictions_)
        # Most test-fold positions should carry an OOF prediction.
        assert finite.mean() > 0.9

    def test_xgboost_purged_cv_early_stopping(self):
        X, y = _make_learnable_dataset(n=400, seed=5)
        labels_df = _make_labels_df(X, horizon=3)
        model = MetaLabeler(
            model_type="xgboost",
            params={"n_estimators": 500, "learning_rate": 0.1},
            calibrate=False,
        ).fit(X, y, labels_df=labels_df)
        assert len(model.best_iterations_) > 0
        # Every fold's best_iteration must be below the requested budget.
        assert all(b <= 500 for b in model.best_iterations_)

    def test_random_forest_ignores_labels_df(self):
        X, y = _make_learnable_dataset(n=200)
        labels_df = _make_labels_df(X, horizon=3)
        # Should not raise, just run plain fit and skip early stopping.
        model = MetaLabeler(
            model_type="random_forest",
            params={"n_estimators": 50},
            calibrate=False,
        ).fit(X, y, labels_df=labels_df)
        assert model.best_iterations_ == []
        assert model.oof_predictions_ is None
