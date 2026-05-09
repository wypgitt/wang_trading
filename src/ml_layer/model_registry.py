"""
Model registry for the meta-labeler (design-doc §7.5 + §13 Phase 6 retraining).

Thin MLflow wrapper that keeps a uniform interface across the rest of the ML
layer: log a training run (params, metrics, importances, serialized model),
load it back, rank runs by metric, compare runs side-by-side, and promote a
chosen run to a registered-model alias.

Serialization: joblib, because :class:`MetaLabeler` is a custom class
wrapping a backend estimator, not an MLflow-native flavour. The model is
dumped to a temp file and logged as a run artifact; ``load_model`` reverses
the path.

Tracking URI defaults to ``sqlite:///mlflow.db`` — a single-file local
store suitable for solo-operator workflows and trivially swapped for a
remote server in production.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.ml_layer.meta_labeler import MetaLabeler


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Light MLflow façade for the meta-labeler training/serving lifecycle.

    Parameters
    ----------
    tracking_uri : str, default ``"sqlite:///mlflow.db"``
        Any URI MLflow accepts. Local SQLite by default.
    experiment_name : str, default ``"meta-labeler"``
        Experiment name (created on first use). Also used as the
        registered-model name in :meth:`promote_model`.
    """

    MODEL_FILENAME = "meta_labeler.joblib"
    FEATURE_NAMES_FILENAME = "feature_names.json"
    IMPORTANCES_SUBDIR = "importances"

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "meta-labeler",
    ) -> None:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        experiment = mlflow.set_experiment(experiment_name)
        self._experiment_id: str = experiment.experiment_id

    # ------------------------------------------------------------------ API --

    # pylint: disable=unused-argument
    #   labels_df: kept in the signature for API symmetry with fit(); Phase 4
    #     will start logging the full purged-CV split manifest.
    #   sample_weight: surfaced here so callers don't drop weights silently;
    #     we only log its presence/absence (as a param flag) for now.
    def log_training_run(
        self,
        model: "MetaLabeler",
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        labels_df: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
        cv_scores: np.ndarray | None = None,
        importances: dict[str, pd.DataFrame] | None = None,
        sample_weight: pd.Series | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        # pylint: enable=unused-argument
        """
        Create an MLflow run, log everything, return the ``run_id``.

        Logged:
            params   — the caller's params + model_type, n_features, n_samples,
                       calibrate flag, has_sample_weight flag.
            metrics  — mean_cv_score, std_cv_score, per-fold cv_fold_{i},
                       train_accuracy.
            artifacts — serialized model (joblib), feature_names.json,
                        importances/<name>.csv per entry in ``importances``.
        """
        import mlflow

        cv_arr = np.asarray(cv_scores if cv_scores is not None else [], dtype=float)
        finite_cv = cv_arr[np.isfinite(cv_arr)]

        with mlflow.start_run(experiment_id=self._experiment_id) as run:
            # ---- params -------------------------------------------------
            safe_params: dict[str, Any] = {
                str(k): self._stringify(v) for k, v in (params or {}).items()
            }
            safe_params.update({
                "model_type": getattr(model, "model_type", type(model).__name__),
                "calibrate": bool(getattr(model, "calibrate", False)),
                "has_sample_weight": sample_weight is not None,
            })
            if X is not None:
                safe_params["n_features"] = X.shape[1]
                safe_params["n_samples"] = X.shape[0]
            mlflow.log_params(safe_params)

            # ---- metrics ------------------------------------------------
            if finite_cv.size > 0:
                mlflow.log_metric("mean_cv_score", float(finite_cv.mean()))
                mlflow.log_metric(
                    "std_cv_score",
                    float(finite_cv.std(ddof=1)) if finite_cv.size > 1 else 0.0,
                )
                for i, score in enumerate(cv_arr):
                    if np.isfinite(score):
                        mlflow.log_metric(f"cv_fold_{i}", float(score))
            else:
                logger.warning(
                    "ModelRegistry: cv_scores had no finite values; "
                    "skipping CV metrics"
                )

            for key, value in (metrics or {}).items():
                if isinstance(value, (int, float, bool, np.integer, np.floating)):
                    mlflow.log_metric(str(key), float(value))

            if X is not None and y is not None:
                try:
                    train_preds = model.predict(X)
                    mlflow.log_metric(
                        "train_accuracy",
                        float((train_preds == y.to_numpy()).mean()),
                    )
                except Exception as exc:  # noqa: BLE001 — never fail the run on metric
                    logger.warning(f"ModelRegistry: train_accuracy failed: {exc}")

            # ---- artifacts ---------------------------------------------
            with tempfile.TemporaryDirectory() as tmp:
                # Model (joblib).
                model_path = os.path.join(tmp, self.MODEL_FILENAME)
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)

                # Feature names.
                feature_names = (
                    list(X.columns) if X is not None
                    else list(getattr(model, "feature_names_", []) or [])
                )
                if feature_names:
                    fn_path = os.path.join(tmp, self.FEATURE_NAMES_FILENAME)
                    with open(fn_path, "w", encoding="utf-8") as f:
                        json.dump(feature_names, f)
                    mlflow.log_artifact(fn_path)

                # Importances.
                if importances:
                    imp_dir = os.path.join(tmp, self.IMPORTANCES_SUBDIR)
                    os.makedirs(imp_dir, exist_ok=True)
                    for name, obj in importances.items():
                        if isinstance(obj, pd.Series):
                            obj = obj.to_frame("value")
                        if not isinstance(obj, pd.DataFrame):
                            logger.warning(
                                f"ModelRegistry: importances[{name!r}] is "
                                f"{type(obj).__name__}; skipping"
                            )
                            continue
                        csv_path = os.path.join(imp_dir, f"{name}.csv")
                        obj.to_csv(csv_path)
                    mlflow.log_artifacts(
                        imp_dir, artifact_path=self.IMPORTANCES_SUBDIR,
                    )

            return run.info.run_id

    def load_model(self, run_id: str) -> "MetaLabeler":
        """Load a joblib-serialized MetaLabeler logged under ``run_id``."""
        import mlflow

        local_dir = tempfile.mkdtemp(prefix="mlflow_load_")
        # ``download_artifacts`` handles both runs:/ URIs and plain run_id.
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=self.MODEL_FILENAME,
            dst_path=local_dir,
        )
        return joblib.load(local_path)

    def get_best_model(
        self, metric: str = "mean_cv_score", n: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Return the top ``n`` runs in this experiment ranked by ``metric``.

        Assumes "higher is better" (uses DESC order). For inverted metrics
        the caller should log the negated version.
        """
        import mlflow

        # Default output_format is "pandas" so we get a DataFrame; narrow
        # the return type for mypy (the overload also includes list[Run]).
        runs_obj = mlflow.search_runs(
            experiment_ids=[self._experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=max(n, 1),
        )
        runs = runs_obj if isinstance(runs_obj, pd.DataFrame) else pd.DataFrame()
        if runs.empty:
            return []

        results: list[dict[str, Any]] = []
        for _, row in runs.iterrows():
            params = {
                k[len("params."):]: row[k]
                for k in row.index
                if k.startswith("params.") and pd.notna(row[k])
            }
            metrics = {
                k[len("metrics."):]: float(row[k])
                for k in row.index
                if k.startswith("metrics.") and pd.notna(row[k])
            }
            results.append({
                "run_id": row["run_id"],
                "params": params,
                "metrics": metrics,
            })
        return results

    def get_production_model(self) -> dict[str, Any] | None:
        """Return metadata for the current production model, if one exists.

        MLflow model stages are deprecated. Prefer the ``production`` alias
        and fall back to the legacy ``Production`` stage for older registries.
        """
        import mlflow

        client = mlflow.tracking.MlflowClient()
        model_name = self.experiment_name

        version = self._get_version_by_alias(client, model_name, "production")
        source = "alias"
        if version is None:
            version = self._get_version_by_stage(client, model_name, "Production")
            source = "stage"
        if version is None:
            logger.warning(
                f"ModelRegistry: no production model for {model_name!r}"
            )
            return None

        run_id = getattr(version, "run_id", "")
        info: dict[str, Any] = {
            "run_id": run_id,
            "version": getattr(version, "version", None),
            "stage": getattr(version, "current_stage", None) or "Production",
            "alias": "production" if source == "alias" else None,
            "source": source,
            "name": model_name,
        }
        if run_id:
            try:
                run = client.get_run(run_id)
                params = dict(run.data.params)
                metrics = dict(run.data.metrics)
                info["params"] = params
                info["metrics"] = metrics
                info["gates"] = self._extract_gate_flags(params, metrics)
                n_events = (
                    params.get("n_training_events")
                    or params.get("n_events")
                    or params.get("n_samples")
                    or metrics.get("n_training_events")
                    or metrics.get("n_events")
                )
                try:
                    info["n_training_events"] = int(float(n_events))
                except (TypeError, ValueError):
                    pass
                trained_at = getattr(run.info, "start_time", None)
                if trained_at is not None:
                    info["trained_at"] = pd.to_datetime(
                        trained_at, unit="ms", utc=True,
                    ).to_pydatetime()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"ModelRegistry: could not load production run metadata: {exc}"
                )
        return info

    def compare_models(self, run_ids: list[str]) -> pd.DataFrame:
        """
        Side-by-side comparison of multiple runs.

        Columns: ``param.<name>`` for each hyperparameter, ``metric.<name>``
        for each tracked metric. Index: ``run_id``. Useful after tuning +
        a few retrains to pick between candidates manually.
        """
        import mlflow

        client = mlflow.tracking.MlflowClient()
        rows: list[dict[str, Any]] = []
        for rid in run_ids:
            try:
                run = client.get_run(rid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"ModelRegistry: compare_models: run {rid!r} not found ({exc})"
                )
                continue
            row: dict[str, Any] = {"run_id": rid}
            for k, v in run.data.params.items():
                row[f"param.{k}"] = v
            for k, v in run.data.metrics.items():
                row[f"metric.{k}"] = float(v)
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("run_id")

    def promote_model(self, run_id: str, stage: str = "production") -> None:
        """
        Register the model from ``run_id`` and promote it.

        ``production`` and ``staging`` use MLflow registered-model aliases
        first. ``archived`` and ``none`` still use legacy stages when the
        client supports them, because they do not map to serving aliases.
        Older MLflow clients that lack aliases fall back to stage transitions.
        """
        import mlflow
        from mlflow.exceptions import MlflowException, RestException

        client = mlflow.tracking.MlflowClient()
        model_name = self.experiment_name

        # Registered-model entry may not exist yet — create it.
        try:
            client.create_registered_model(name=model_name)
        except (MlflowException, RestException) as exc:
            if "already exist" not in str(exc).lower():
                raise

        model_uri = f"runs:/{run_id}/{self.MODEL_FILENAME}"
        version = client.create_model_version(
            name=model_name, source=model_uri, run_id=run_id,
        )

        mlflow_stage = stage.strip().capitalize()
        if mlflow_stage not in ("Production", "Staging", "Archived", "None"):
            raise ValueError(
                f"stage must be one of Production/Staging/Archived/None "
                f"(got {stage!r})"
            )

        alias = mlflow_stage.lower()
        if alias in {"production", "staging"} and self._set_alias(
            client, model_name, alias, version.version,
        ):
            logger.info(
                f"ModelRegistry: promoted run {run_id} to "
                f"{model_name} v{version.version} (alias={alias})"
            )
            return

        self._transition_stage(client, model_name, version.version, mlflow_stage)
        logger.info(
            f"ModelRegistry: promoted run {run_id} to "
            f"{model_name} v{version.version} ({mlflow_stage})"
        )

    # ------------------------------------------------------------------ registry helpers --

    @staticmethod
    def _get_version_by_alias(client: Any, model_name: str, alias: str) -> Any | None:
        fn = getattr(client, "get_model_version_by_alias", None)
        if not callable(fn):
            return None
        try:
            return fn(model_name, alias)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _get_version_by_stage(client: Any, model_name: str, stage: str) -> Any | None:
        fn = getattr(client, "get_latest_versions", None)
        if not callable(fn):
            return None
        try:
            versions = list(fn(model_name, stages=[stage]))
        except Exception:  # noqa: BLE001
            return None
        return versions[0] if versions else None

    @staticmethod
    def _set_alias(
        client: Any,
        model_name: str,
        alias: str,
        version: str | int,
    ) -> bool:
        fn = getattr(client, "set_registered_model_alias", None)
        if not callable(fn):
            return False
        try:
            fn(model_name, alias, str(version))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"ModelRegistry: set alias {alias!r} failed; "
                f"falling back to stage transition: {exc}"
            )
            return False

    def _transition_stage(
        self,
        client: Any,
        model_name: str,
        version: str | int,
        stage: str,
    ) -> None:
        if stage == "Production":
            current = self._get_stage_versions(client, model_name, stage)
            for v in current:
                if str(getattr(v, "version", "")) != str(version):
                    client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Archived",
                    )
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )

    @staticmethod
    def _get_stage_versions(client: Any, model_name: str, stage: str) -> list[Any]:
        fn = getattr(client, "get_latest_versions", None)
        if not callable(fn):
            return []
        try:
            return list(fn(model_name, stages=[stage]))
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ helpers --

    @staticmethod
    def _stringify(v: Any) -> Any:
        """Coerce values MLflow's param store can't accept (nested containers)."""
        if isinstance(v, (list, tuple, dict, set)):
            return json.dumps(v, default=str)
        return v

    @staticmethod
    def _extract_gate_flags(
        params: dict[str, Any], metrics: dict[str, Any],
    ) -> dict[str, bool]:
        gates: dict[str, bool] = {}
        raw = params.get("gates")
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = {}
            if isinstance(decoded, dict):
                for key in ("cpcv", "dsr", "pbo"):
                    if key in decoded:
                        gates[key] = ModelRegistry._coerce_bool(decoded[key])
        for key in ("cpcv", "dsr", "pbo"):
            for source in (metrics, params):
                metric_key = f"gate_{key}"
                if metric_key in source:
                    gates[key] = ModelRegistry._coerce_bool(source[metric_key])
                    break
        return gates

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value) and float(value) != 0.0)
        return bool(value)
