"""Tests for the MLflow-backed meta-labeler registry."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.meta_labeler import MetaLabeler
from src.ml_layer.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_registry(tmp_path: Path):
    """Per-test MLflow tracking URI backed by a tempdir SQLite db."""
    db_path = tmp_path / "mlflow.db"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    tracking_uri = f"sqlite:///{db_path}"

    # MLflow uses a separate artifact location when the tracking store is SQLite.
    # Keep experiment creation tidy by running the factory inside the tempdir
    # so any fallback relative paths resolve into it.
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        registry = ModelRegistry(
            tracking_uri=tracking_uri,
            experiment_name="test-meta-labeler",
        )
        yield registry
    finally:
        os.chdir(cwd)


def _fitted_model_and_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=["f0", "f1", "f2", "f3"],
        index=idx,
    )
    logits = 1.2 * X["f0"] - 0.6 * X["f1"]
    prob = 1.0 / (1.0 + np.exp(-logits.to_numpy()))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int),
                  index=idx, name="y")
    end_pos = np.clip(np.arange(n) + 2, 0, n - 1)
    labels_df = pd.DataFrame(
        {"event_start": idx, "event_end": idx[end_pos]}, index=idx,
    )
    model = MetaLabeler(
        model_type="lightgbm",
        params={"n_estimators": 50, "learning_rate": 0.1},
        calibrate=False,
    ).fit(X, y)
    return model, X, y, labels_df


# ---------------------------------------------------------------------------
# log_training_run
# ---------------------------------------------------------------------------

class TestLogTrainingRun:
    def test_returns_valid_run_id(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"learning_rate": 0.1, "n_estimators": 50},
            cv_scores=np.array([0.62, 0.65, 0.63, 0.61, 0.64]),
            importances=None,
        )
        assert isinstance(run_id, str) and len(run_id) > 0

    def test_metrics_are_logged(self, tmp_registry):
        import mlflow
        model, X, y, labels = _fitted_model_and_data()
        scores = np.array([0.60, 0.65, 0.62])
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"learning_rate": 0.1},
            cv_scores=scores,
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert "mean_cv_score" in run.data.metrics
        assert np.isclose(run.data.metrics["mean_cv_score"], scores.mean())
        assert "std_cv_score" in run.data.metrics
        for i in range(len(scores)):
            assert f"cv_fold_{i}" in run.data.metrics
        assert "train_accuracy" in run.data.metrics

    def test_params_stringify_containers(self, tmp_registry):
        import mlflow
        model, X, y, labels = _fitted_model_and_data()
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"n_estimators": 50, "tree_method": ["hist", "auto"]},
            cv_scores=np.array([0.6, 0.62]),
        )
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        # List value gets JSON-serialized.
        assert run.data.params["tree_method"] == '["hist", "auto"]'
        assert run.data.params["n_estimators"] == "50"
        # Auto-logged params.
        assert run.data.params["model_type"] == "lightgbm"
        assert int(run.data.params["n_features"]) == X.shape[1]
        assert int(run.data.params["n_samples"]) == X.shape[0]

    def test_importances_artifact_logged(self, tmp_registry):
        import mlflow
        model, X, y, labels = _fitted_model_and_data()
        mdi = pd.Series(
            [0.4, 0.3, 0.2, 0.1], index=list(X.columns), name="mdi",
        ).to_frame("importance")
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"n_estimators": 50},
            cv_scores=np.array([0.6]),
            importances={"mdi": mdi},
        )
        client = mlflow.tracking.MlflowClient()
        artifacts = {
            a.path for a in client.list_artifacts(run_id, path="importances")
        }
        assert "importances/mdi.csv" in artifacts


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_roundtrip_predictions_match(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        original_preds = model.predict_proba(X)
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"n_estimators": 50},
            cv_scores=np.array([0.6]),
        )
        loaded = tmp_registry.load_model(run_id)
        loaded_preds = loaded.predict_proba(X)
        assert loaded_preds.shape == original_preds.shape
        assert np.allclose(original_preds, loaded_preds)

    def test_loaded_model_is_meta_labeler(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"n_estimators": 50},
            cv_scores=np.array([0.6]),
        )
        loaded = tmp_registry.load_model(run_id)
        assert isinstance(loaded, MetaLabeler)
        assert loaded.model_type == "lightgbm"


# ---------------------------------------------------------------------------
# get_best_model
# ---------------------------------------------------------------------------

class TestGetBestModel:
    def test_returns_highest_metric_run(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        run_a = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"tag": "low"}, cv_scores=np.array([0.55, 0.56]),
        )
        run_b = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"tag": "high"}, cv_scores=np.array([0.75, 0.77]),
        )
        run_c = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"tag": "mid"}, cv_scores=np.array([0.65, 0.66]),
        )

        top = tmp_registry.get_best_model(metric="mean_cv_score", n=1)
        assert len(top) == 1
        assert top[0]["run_id"] == run_b
        assert "mean_cv_score" in top[0]["metrics"]
        assert top[0]["params"]["tag"] == "high"

    def test_n_greater_than_one_returns_sorted(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        ids: list[str] = []
        for score in (0.50, 0.80, 0.60, 0.70):
            ids.append(tmp_registry.log_training_run(
                model, X, y, labels,
                params={"score_tag": str(score)},
                cv_scores=np.array([score]),
            ))
        top3 = tmp_registry.get_best_model(metric="mean_cv_score", n=3)
        assert len(top3) == 3
        scores_in_order = [r["metrics"]["mean_cv_score"] for r in top3]
        assert scores_in_order == sorted(scores_in_order, reverse=True)

    def test_empty_experiment_returns_empty_list(self, tmp_registry):
        assert tmp_registry.get_best_model() == []


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_dataframe_with_run_columns(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        run_a = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"learning_rate": 0.05}, cv_scores=np.array([0.60]),
        )
        run_b = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"learning_rate": 0.10}, cv_scores=np.array([0.70]),
        )
        cmp = tmp_registry.compare_models([run_a, run_b])
        assert isinstance(cmp, pd.DataFrame)
        assert set(cmp.index) == {run_a, run_b}
        assert "param.learning_rate" in cmp.columns
        assert "metric.mean_cv_score" in cmp.columns
        assert cmp.loc[run_b, "metric.mean_cv_score"] > cmp.loc[run_a, "metric.mean_cv_score"]

    def test_missing_run_ids_skipped(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"n_estimators": 50}, cv_scores=np.array([0.6]),
        )
        cmp = tmp_registry.compare_models([run_id, "nonexistent-id"])
        assert list(cmp.index) == [run_id]


# ---------------------------------------------------------------------------
# promote_model
# ---------------------------------------------------------------------------

class TestPromoteModel:
    def test_rejects_unknown_stage(self, tmp_registry):
        model, X, y, labels = _fitted_model_and_data()
        run_id = tmp_registry.log_training_run(
            model, X, y, labels,
            params={"n_estimators": 50}, cv_scores=np.array([0.6]),
        )
        with pytest.raises(ValueError):
            tmp_registry.promote_model(run_id, stage="canary")
