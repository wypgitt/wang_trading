"""Tests for feature-importance methods (AFML Ch. 8, design-doc §7.5)."""

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.feature_importance import (
    compute_all_importances,
    mda_importance,
    mdi_importance,
    select_features,
    sfi_importance,
    shap_importance,
)
from src.ml_layer.meta_labeler import MetaLabeler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _signal_plus_noise(
    n: int = 400, n_noise: int = 4, seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Dataset where ``signal`` carries all the predictive power and the rest
    of the columns are independent Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    signal = rng.normal(size=n)
    # Binary target is driven almost entirely by signal.
    prob = 1.0 / (1.0 + np.exp(-2.0 * signal))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int),
                  index=idx, name="y")

    cols: dict[str, np.ndarray] = {"signal": signal}
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.normal(size=n)
    X = pd.DataFrame(cols, index=idx)

    end_pos = np.clip(np.arange(n) + 2, 0, n - 1)
    labels_df = pd.DataFrame(
        {"event_start": idx, "event_end": idx[end_pos]}, index=idx,
    )
    return X, y, labels_df


def _fit_lgbm(X: pd.DataFrame, y: pd.Series, *, n_estimators: int = 100):
    return MetaLabeler(
        model_type="lightgbm",
        params={"n_estimators": n_estimators, "learning_rate": 0.1},
        calibrate=False,
    ).fit(X, y)


# ---------------------------------------------------------------------------
# MDI
# ---------------------------------------------------------------------------

class TestMDI:
    def test_sums_to_one(self):
        X, y, _ = _signal_plus_noise()
        model = _fit_lgbm(X, y)
        imp = mdi_importance(model, list(X.columns))
        assert np.isclose(imp.sum(), 1.0)
        assert (imp >= 0).all()

    def test_sorted_descending(self):
        X, y, _ = _signal_plus_noise()
        model = _fit_lgbm(X, y)
        imp = mdi_importance(model, list(X.columns))
        assert list(imp.values) == sorted(imp.values, reverse=True)

    def test_signal_ranks_first(self):
        X, y, _ = _signal_plus_noise(n=500)
        model = _fit_lgbm(X, y, n_estimators=200)
        imp = mdi_importance(model, list(X.columns))
        assert imp.index[0] == "signal"

    def test_length_mismatch_raises(self):
        X, y, _ = _signal_plus_noise()
        model = _fit_lgbm(X, y)
        with pytest.raises(ValueError):
            mdi_importance(model, ["only_one"])


# ---------------------------------------------------------------------------
# MDA
# ---------------------------------------------------------------------------

class TestMDA:
    def test_signal_is_significant(self):
        X, y, labels = _signal_plus_noise(n=500, seed=1)
        model = _fit_lgbm(X, y, n_estimators=200)
        mda = mda_importance(
            model, X, y, labels,
            n_splits=3, n_repeats=3, scoring="accuracy",
        )
        # signal has the largest MDA and a significant p-value.
        assert mda.index[0] == "signal"
        assert mda.loc["signal", "mean_importance"] > 0
        assert mda.loc["signal", "p_value"] < 0.05

    def test_noise_is_insignificant(self):
        X, y, labels = _signal_plus_noise(n=500, seed=2)
        model = _fit_lgbm(X, y, n_estimators=200)
        mda = mda_importance(
            model, X, y, labels,
            n_splits=3, n_repeats=3, scoring="accuracy",
        )
        # At least one noise feature should land non-significant.
        noise_rows = mda.loc[[c for c in mda.index if c.startswith("noise_")]]
        assert (noise_rows["p_value"] > 0.05).any(), (
            "expected at least one noise feature to be non-significant"
        )

    def test_output_shape_and_columns(self):
        X, y, labels = _signal_plus_noise(n=250)
        model = _fit_lgbm(X, y, n_estimators=50)
        mda = mda_importance(
            model, X, y, labels, n_splits=3, n_repeats=2,
        )
        assert set(mda.columns) == {
            "mean_importance", "std_importance", "p_value",
        }
        assert len(mda) == len(X.columns)

    def test_rejects_bad_scoring(self):
        X, y, labels = _signal_plus_noise(n=100)
        model = _fit_lgbm(X, y, n_estimators=10)
        with pytest.raises(ValueError):
            mda_importance(
                model, X, y, labels,
                n_splits=2, n_repeats=1, scoring="mystery",
            )


# ---------------------------------------------------------------------------
# SFI
# ---------------------------------------------------------------------------

class TestSFI:
    def test_signal_ranks_first(self):
        X, y, labels = _signal_plus_noise(n=400, seed=3)
        sfi = sfi_importance(
            X, y, labels, n_splits=3, scoring="accuracy",
        )
        assert sfi.index[0] == "signal"
        assert sfi.loc["signal"] > 0.55

    def test_noise_is_near_chance(self):
        X, y, labels = _signal_plus_noise(n=400, seed=4)
        sfi = sfi_importance(
            X, y, labels, n_splits=3, scoring="accuracy",
        )
        noise_scores = sfi.loc[[c for c in sfi.index if c.startswith("noise_")]]
        # Noise features should cluster around 0.5 (~chance for balanced binary).
        assert noise_scores.mean() < 0.58

    def test_returns_series(self):
        X, y, labels = _signal_plus_noise(n=200)
        sfi = sfi_importance(X, y, labels, n_splits=3)
        assert isinstance(sfi, pd.Series)
        assert set(sfi.index) == set(X.columns)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

class TestSHAP:
    def test_shape_matches_input(self):
        X, y, _ = _signal_plus_noise(n=300)
        model = _fit_lgbm(X, y)
        shap_df = shap_importance(model, X, max_samples=200)
        assert shap_df.shape == (200, len(X.columns))
        assert list(shap_df.columns) == list(X.columns)

    def test_summary_importance_attached(self):
        X, y, _ = _signal_plus_noise(n=200)
        model = _fit_lgbm(X, y)
        shap_df = shap_importance(model, X, max_samples=100)
        summary = shap_df.attrs["summary_importance"]
        assert isinstance(summary, pd.Series)
        assert set(summary.index) == set(X.columns)

    def test_signal_has_highest_mean_abs_shap(self):
        X, y, _ = _signal_plus_noise(n=400, seed=5)
        model = _fit_lgbm(X, y, n_estimators=200)
        shap_df = shap_importance(model, X, max_samples=300)
        summary = shap_df.attrs["summary_importance"]
        assert summary.index[0] == "signal"

    def test_no_subsample_when_small(self):
        X, y, _ = _signal_plus_noise(n=100)
        model = _fit_lgbm(X, y)
        shap_df = shap_importance(model, X, max_samples=500)
        assert len(shap_df) == 100  # no subsampling; full dataset


# ---------------------------------------------------------------------------
# compute_all_importances + select_features
# ---------------------------------------------------------------------------

class TestComposition:
    def test_compute_all_returns_four_frames(self):
        X, y, labels = _signal_plus_noise(n=300, seed=6)
        model = _fit_lgbm(X, y, n_estimators=100)
        imps = compute_all_importances(
            model, X, y, labels,
            mda_kwargs={"n_splits": 3, "n_repeats": 2, "scoring": "accuracy"},
            sfi_kwargs={"n_splits": 3, "scoring": "accuracy"},
            shap_kwargs={"max_samples": 200},
        )
        assert set(imps.keys()) == {"mdi", "mda", "sfi", "shap"}
        for key, df in imps.items():
            assert isinstance(df, pd.DataFrame), f"{key} must be a DataFrame"

    def test_select_features_keeps_signal_drops_noise(self):
        X, y, labels = _signal_plus_noise(n=500, seed=7)
        model = _fit_lgbm(X, y, n_estimators=200)
        imps = compute_all_importances(
            model, X, y, labels,
            mda_kwargs={"n_splits": 3, "n_repeats": 3, "scoring": "accuracy"},
            sfi_kwargs={"n_splits": 3, "scoring": "accuracy"},
            shap_kwargs={"max_samples": 200},
        )
        kept = select_features(
            imps, mda_pvalue_threshold=0.05, min_sfi_score=0.52,
        )
        assert "signal" in kept
        # Most noise features should be dropped.
        noise_kept = [k for k in kept if k.startswith("noise_")]
        assert len(noise_kept) < sum(1 for c in X.columns if c.startswith("noise_"))

    def test_select_features_validates_dict(self):
        with pytest.raises(ValueError):
            select_features({"mdi": pd.DataFrame()})
