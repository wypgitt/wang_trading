"""
Hyperparameter tuning for the Tier-1 meta-labeler (design doc §7.1).

Bayesian (TPE) search over GBM / Random-Forest hyperparameters evaluated via
purged k-fold CV so the optimizer never peeks at information that leaks across
overlapping label periods. Pruning via Optuna's ``MedianPruner`` terminates
clearly unpromising trials after a few folds, so the same trial budget covers
a wider search space than exhaustive CV would.

Exported entry points:

    create_objective(...)          — factory for Optuna objective functions
    tune_meta_labeler(...)         — run the study, return best params
    retrain_with_best_params(...)  — fit a final MetaLabeler on the tuned params
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger

from src.ml_layer.meta_labeler import MetaLabeler
from src.ml_layer.purged_cv import PurgedKFoldCV


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Metric → (direction, predict_proba_required)
#
# "higher is better" metrics (maximize): accuracy, f1, precision, recall, roc_auc.
# "neg_log_loss" is also maximize (less-negative = better).
# "log_loss" is minimize.
_METRIC_SPECS: dict[str, tuple[str, bool]] = {
    "accuracy": ("maximize", False),
    "f1": ("maximize", False),
    "precision": ("maximize", False),
    "recall": ("maximize", False),
    "roc_auc": ("maximize", True),
    "neg_log_loss": ("maximize", True),
    "log_loss": ("minimize", True),
}


def _score_fold(  # pylint: disable=too-many-return-statements
    scoring: str,
    y_true: pd.Series,
    preds: np.ndarray,
    proba: np.ndarray,
) -> float:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if scoring == "accuracy":
        return float(accuracy_score(y_true, preds))
    if scoring == "f1":
        return float(f1_score(y_true, preds, zero_division=0))
    if scoring == "precision":
        return float(precision_score(y_true, preds, zero_division=0))
    if scoring == "recall":
        return float(recall_score(y_true, preds, zero_division=0))
    if scoring == "roc_auc":
        # Undefined when fold val is single-class.
        if y_true.nunique() < 2:
            return float("nan")
        return float(roc_auc_score(y_true, proba))
    if scoring == "log_loss":
        return float(log_loss(y_true, proba, labels=[0, 1]))
    if scoring == "neg_log_loss":
        return float(-log_loss(y_true, proba, labels=[0, 1]))
    raise ValueError(f"unknown scoring {scoring!r}")


# ---------------------------------------------------------------------------
# Parameter-space sampler
# ---------------------------------------------------------------------------

def _sample_params(trial: "optuna.Trial", model_type: str) -> dict[str, Any]:
    """
    Sample one candidate parameter vector from the design-doc ranges.

    The GBM backends share a common search space. RandomForest remaps:
        min_child_weight → min_samples_leaf
    and drops the GBM-only subsample / colsample / reg_alpha / reg_lambda
    which have no direct analogue in scikit-learn's RandomForest.
    """
    lr = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
    n_est = trial.suggest_int("n_estimators", 100, 2000)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    mcw = trial.suggest_int("min_child_weight", 1, 50)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample = trial.suggest_float("colsample_bytree", 0.3, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)

    if model_type == "random_forest":
        return {
            "n_estimators": n_est,
            "max_depth": max_depth,
            "min_samples_leaf": mcw,
            "n_jobs": -1,
        }

    return {
        "learning_rate": lr,
        "n_estimators": n_est,
        "max_depth": max_depth,
        "min_child_weight": mcw,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "n_jobs": -1,
    }


def _build_classifier(model_type: str, params: dict[str, Any]):
    """Instantiate an unfitted backend classifier (no MetaLabeler wrapper)."""
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        p = dict(params)
        p.setdefault("verbose", -1)
        return LGBMClassifier(**p)
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        p = dict(params)
        p.setdefault("verbosity", 0)
        return XGBClassifier(**p)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    raise ValueError(f"unsupported model_type {model_type!r}")


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def create_objective(
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    model_type: str = "lightgbm",
    n_splits: int = 5,
    scoring: str = "neg_log_loss",
    embargo_pct: float = 0.01,
) -> Callable[["optuna.Trial"], float]:
    """
    Build an Optuna objective function for purged-CV tuning.

    The returned callable:
        * samples hyperparameters from the design-doc ranges,
        * runs :class:`PurgedKFoldCV` with ``n_splits`` / ``embargo_pct``,
        * reports the running mean fold score to Optuna each fold so
          :class:`MedianPruner` can kill weak trials early,
        * returns the mean fold score (or raises ``TrialPruned``).

    Args:
        X, y:         Training features and target.
        labels_df:    DataFrame with ``event_start`` / ``event_end`` columns
                      (required by PurgedKFoldCV.split).
        sample_weight: Optional per-row weight Series aligned with X.
        model_type:   ``"lightgbm"``, ``"xgboost"``, or ``"random_forest"``.
        n_splits:     Purged-CV fold count.
        scoring:      One of the keys in ``_METRIC_SPECS``.
        embargo_pct:  Forward-embargo fraction (see PurgedKFoldCV).
    """
    if model_type not in ("lightgbm", "xgboost", "random_forest"):
        raise ValueError(f"unsupported model_type {model_type!r}")
    if scoring not in _METRIC_SPECS:
        raise ValueError(
            f"unknown scoring {scoring!r}; choose from {sorted(_METRIC_SPECS)}"
        )
    _, needs_proba = _METRIC_SPECS[scoring]

    def objective(trial: "optuna.Trial") -> float:
        """Optuna objective: sample params, run purged CV, return mean score."""
        params = _sample_params(trial, model_type)
        cv = PurgedKFoldCV(n_splits=n_splits, embargo_pct=embargo_pct)

        scores: list[float] = []
        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, labels_df)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
            sw_tr = (
                np.asarray(sample_weight.iloc[train_idx])
                if sample_weight is not None else None
            )

            model = _build_classifier(model_type, params)
            fit_kwargs: dict[str, Any] = {}
            if sw_tr is not None:
                fit_kwargs["sample_weight"] = sw_tr
            try:
                model.fit(X_tr, y_tr, **fit_kwargs)
            except Exception as exc:  # noqa: BLE001 — crashes on pathological params
                logger.debug(
                    f"tuning: fold {fold_i} fit failed ({exc!r}); pruning"
                )
                raise optuna.TrialPruned()

            preds = model.predict(X_val)
            proba = (
                model.predict_proba(X_val)[:, 1]
                if needs_proba else np.zeros(len(X_val))
            )
            fold_score = _score_fold(scoring, y_val, preds, proba)
            if not np.isfinite(fold_score):
                # Single-class val under roc_auc etc. → skip fold.
                continue
            scores.append(fold_score)

            running_mean = float(np.mean(scores))
            trial.report(running_mean, step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not scores:
            raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

def tune_meta_labeler(
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    model_type: str = "lightgbm",
    n_trials: int = 100,
    timeout: int | None = 600,
    n_splits: int = 5,
    scoring: str = "neg_log_loss",
    embargo_pct: float = 0.01,
    random_state: int = 42,
    study_name: str | None = None,
) -> dict[str, Any]:
    """
    Run Bayesian hyperparameter search with purged-CV evaluation.

    The returned dict maps hyperparameter name → value (using Optuna's
    ``study.best_params``). The underlying study is stashed on the function
    as ``tune_meta_labeler.last_study_`` so callers that want the full
    optimisation history (param importances, intermediate scores, etc.)
    can inspect it without plumbing an extra return value.

    Args:
        n_trials:   Maximum trial count. Pass a large number and rely on
                    ``timeout`` to stop early.
        timeout:    Seconds; ``None`` = run until ``n_trials`` done.
    """
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    # Suppress optuna's verbose per-trial info chatter — callers see the
    # study summary via loguru below.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    direction, _ = _METRIC_SPECS[scoring]
    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=1)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    objective = create_objective(
        X, y, labels_df,
        sample_weight=sample_weight,
        model_type=model_type,
        n_splits=n_splits,
        scoring=scoring,
        embargo_pct=embargo_pct,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    n_complete = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    )
    logger.info(
        f"tune_meta_labeler ({model_type}): {n_complete} complete, {n_pruned} pruned; "
        f"best {scoring}={study.best_value:.5f}"
    )

    # Cache the study on the function so callers can grab it.
    tune_meta_labeler.last_study_ = study  # type: ignore[attr-defined]
    return dict(study.best_params)


# ---------------------------------------------------------------------------
# Final refit helper
# ---------------------------------------------------------------------------

def retrain_with_best_params(
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    model_type: str = "lightgbm",
    calibrate: bool = True,
) -> MetaLabeler:
    """
    Create a :class:`MetaLabeler` from ``best_params``, fit, and return.

    ``labels_df`` is forwarded to :meth:`MetaLabeler.fit` so the final model
    benefits from the same purged-CV early stopping and OOF calibration
    used during training elsewhere. Defaults to ``model_type="lightgbm"`` —
    callers who tuned with a different backend must pass it explicitly.
    """
    model = MetaLabeler(
        model_type=model_type,
        params=best_params,
        calibrate=calibrate,
    )
    model.fit(X, y, sample_weight=sample_weight, labels_df=labels_df)
    return model
