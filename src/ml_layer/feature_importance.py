"""
Feature-importance methods for the meta-labeler (AFML Ch. 8, design-doc §7.5).

Four complementary importance measures, each answering a different question:

    MDI   Mean Decrease Impurity — fast in-sample screening from the tree
          ensemble's internal split gains.
    MDA   Mean Decrease Accuracy — OOF permutation importance under purged
          CV; the single most trustworthy measure because it reflects what
          the model loses when a feature's information is destroyed.
    SFI   Single Feature Importance — CV score of a model trained on ONE
          feature at a time. High MDA but low SFI ⇒ information comes from
          interactions, not the feature in isolation.
    SHAP  Per-row attribution; explains individual predictions (audit
          trail). Mean absolute SHAP per feature is the global summary.

Feature selection (:func:`select_features`) keeps any feature that clears
either the MDA-significance bar OR the SFI standalone bar — the union rule
retains both "interaction-powered" and "standalone" alphas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone

from src.ml_layer.purged_cv import PurgedKFoldCV, cross_val_score_purged


# ---------------------------------------------------------------------------
# Scorer dispatch (higher-is-better convention)
# ---------------------------------------------------------------------------

_HIGHER_IS_BETTER = {
    "accuracy", "f1", "precision", "recall", "roc_auc", "neg_log_loss",
}
_NEEDS_PROBA = {"roc_auc", "neg_log_loss", "log_loss"}


def _score_fold(  # pylint: disable=too-many-return-statements
    scoring: str,
    y_true: pd.Series,
    preds: np.ndarray,
    proba: np.ndarray,
) -> float:
    """Fold-level score under the higher-is-better convention."""
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
        if y_true.nunique() < 2:
            return float("nan")
        return float(roc_auc_score(y_true, proba))
    if scoring == "neg_log_loss":
        return float(-log_loss(y_true, proba, labels=[0, 1]))
    if scoring == "log_loss":
        # Flip so MDA baseline-permuted stays "positive = informative".
        return float(-log_loss(y_true, proba, labels=[0, 1]))
    raise ValueError(f"unknown scoring {scoring!r}")


# ---------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------

def _underlying_model(model: Any) -> Any:
    """Return the fitted sklearn-compatible backend inside a MetaLabeler wrapper."""
    inner = getattr(model, "model_", None)
    return inner if inner is not None else model


# ---------------------------------------------------------------------------
# 1. MDI — Mean Decrease Impurity
# ---------------------------------------------------------------------------

def mdi_importance(
    model: Any, feature_names: list[str],
) -> pd.Series:
    """
    Extract the tree-ensemble's internal feature importance.

    LightGBM and XGBoost expose gain-based importances; sklearn's
    RandomForest exposes gini-decrease. All three are normalised to sum to
    1 so they can be compared across backends.

    Args:
        model:          Fitted classifier (or MetaLabeler wrapping one).
        feature_names:  Column names in the order used at fit time.

    Returns:
        pd.Series indexed by feature, values summing to 1, sorted descending.
    """
    inner = _underlying_model(model)

    # LightGBM.
    if hasattr(inner, "booster_") and hasattr(inner.booster_, "feature_importance"):
        raw = np.asarray(
            inner.booster_.feature_importance(importance_type="gain"),
            dtype=float,
        )
    # XGBoost / sklearn RF both expose .feature_importances_.
    elif hasattr(inner, "feature_importances_"):
        raw = np.asarray(inner.feature_importances_, dtype=float)
    else:
        raise ValueError(
            "mdi_importance: model has no feature_importances_ or booster_"
        )

    if len(raw) != len(feature_names):
        raise ValueError(
            f"mdi_importance: importances length {len(raw)} "
            f"!= feature_names length {len(feature_names)}"
        )

    total = raw.sum()
    if total > 0:
        raw = raw / total
    return pd.Series(raw, index=feature_names, name="mdi").sort_values(
        ascending=False,
    )


# ---------------------------------------------------------------------------
# 2. MDA — Mean Decrease Accuracy (permutation importance, purged CV)
# ---------------------------------------------------------------------------

def mda_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    n_splits: int = 5,
    sample_weight: pd.Series | None = None,
    scoring: str = "neg_log_loss",
    n_repeats: int = 5,
    embargo_pct: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance under purged k-fold CV.

    For each fold, a fresh model is fit on the purged training portion and
    scored on the validation fold (baseline). For each feature + repeat,
    the feature column is permuted in the validation set and the score is
    recomputed. The drop ``baseline − permuted`` is the MDA sample. With
    ``n_splits`` folds and ``n_repeats`` shuffles per feature per fold, each
    feature accumulates ``n_splits × n_repeats`` MDA samples, giving enough
    observations for a one-sided t-test of ``H0: feature is not informative``.

    Important:
        The model is REFIT inside each fold (not just re-scored with the
        caller's fitted model) so the MDA reflects genuine out-of-fold
        degradation. The caller's ``model`` is used only as a spec — we
        extract its backend estimator and ``clone`` it per fold.

    Returns:
        DataFrame indexed by feature with columns
        ``mean_importance``, ``std_importance``, ``p_value``; sorted by
        mean descending. ``p_value > 0.05`` → candidate for removal.
    """
    from scipy import stats as sp_stats

    if scoring not in _HIGHER_IS_BETTER and scoring != "log_loss":
        raise ValueError(f"unknown scoring {scoring!r}")
    needs_proba = scoring in _NEEDS_PROBA
    inner = _underlying_model(model)

    cv = PurgedKFoldCV(n_splits=n_splits, embargo_pct=embargo_pct)
    rng = np.random.default_rng(random_state)

    per_feature_samples: dict[str, list[float]] = {c: [] for c in X.columns}

    for _, (train_idx, test_idx) in enumerate(cv.split(X, y, labels_df)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[test_idx]
        sw_tr = (
            np.asarray(sample_weight.iloc[train_idx])
            if sample_weight is not None else None
        )

        fold_model = clone(inner)
        fit_kwargs: dict[str, Any] = {}
        if sw_tr is not None:
            fit_kwargs["sample_weight"] = sw_tr
        fold_model.fit(X_tr, y_tr, **fit_kwargs)

        baseline_preds = fold_model.predict(X_val)
        baseline_proba = (
            fold_model.predict_proba(X_val)[:, 1]
            if needs_proba else np.zeros(len(X_val))
        )
        baseline = _score_fold(scoring, y_val, baseline_preds, baseline_proba)
        if not np.isfinite(baseline):
            continue

        X_val_arr = X_val.to_numpy(copy=True)
        col_index = {c: i for i, c in enumerate(X_val.columns)}

        for feat in X.columns:
            ci = col_index[feat]
            original_col = X_val_arr[:, ci].copy()
            for _ in range(n_repeats):
                X_val_arr[:, ci] = rng.permutation(original_col)
                X_val_perm = pd.DataFrame(
                    X_val_arr, index=X_val.index, columns=X_val.columns,
                )
                permuted_preds = fold_model.predict(X_val_perm)
                permuted_proba = (
                    fold_model.predict_proba(X_val_perm)[:, 1]
                    if needs_proba else np.zeros(len(X_val))
                )
                permuted = _score_fold(
                    scoring, y_val, permuted_preds, permuted_proba,
                )
                if not np.isfinite(permuted):
                    continue
                per_feature_samples[feat].append(baseline - permuted)
            # Restore the column for the next feature's permutations.
            X_val_arr[:, ci] = original_col

    rows: list[dict[str, Any]] = []
    for feat, samples in per_feature_samples.items():
        if not samples:
            rows.append({
                "feature": feat,
                "mean_importance": 0.0,
                "std_importance": 0.0,
                "p_value": 1.0,
            })
            continue
        arr = np.asarray(samples, dtype=float)
        mean = float(arr.mean())
        if len(arr) > 1:
            std = float(arr.std(ddof=1))
            if std > 0:
                t_stat = mean / (std / np.sqrt(len(arr)))
                p_value = float(1.0 - sp_stats.t.cdf(t_stat, df=len(arr) - 1))
            else:
                p_value = 0.0 if mean > 0 else 1.0
        else:
            std = 0.0
            p_value = 1.0
        rows.append({
            "feature": feat,
            "mean_importance": mean,
            "std_importance": std,
            "p_value": p_value,
        })

    return (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values("mean_importance", ascending=False)
    )


# ---------------------------------------------------------------------------
# 3. SFI — Single Feature Importance
# ---------------------------------------------------------------------------

def _sfi_new_model(model_type: str, params: dict[str, Any] | None = None) -> Any:
    base = dict(params) if params else {}
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        base.setdefault("n_estimators", 100)
        base.setdefault("verbose", -1)
        return LGBMClassifier(**base)
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        base.setdefault("n_estimators", 100)
        base.setdefault("verbosity", 0)
        return XGBClassifier(**base)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        base.setdefault("n_estimators", 100)
        return RandomForestClassifier(**base)
    raise ValueError(f"unsupported model_type {model_type!r}")


def sfi_importance(
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    model_type: str = "lightgbm",
    n_splits: int = 5,
    sample_weight: pd.Series | None = None,
    scoring: str = "accuracy",
    embargo_pct: float = 0.01,
) -> pd.Series:
    """
    Per-feature purged-CV score of a model fit on that feature alone.

    A feature with high MDI/MDA but low SFI derives its predictive power
    from interactions with other features — useful diagnostic when deciding
    whether to keep a nominally-important but standalone-weak feature. The
    default ``scoring="accuracy"`` aligns with the 0.51 "beats random
    guessing" threshold used by :func:`select_features`.

    Returns:
        pd.Series indexed by feature, values = mean CV score, sorted desc.
    """
    scores: dict[str, float] = {}
    for feat in X.columns:
        model = _sfi_new_model(model_type)
        try:
            fold_scores = cross_val_score_purged(
                model,
                X[[feat]],
                y,
                labels_df,
                n_splits=n_splits,
                embargo_pct=embargo_pct,
                sample_weight=sample_weight,
                scoring=scoring,
            )
        except Exception as exc:  # noqa: BLE001 — degenerate single-feature fits
            logger.debug(f"sfi: feature {feat!r} failed: {exc}")
            scores[feat] = float("nan")
            continue
        scores[feat] = float(np.nanmean(fold_scores))

    return pd.Series(scores, name="sfi").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 4. SHAP — per-row attribution
# ---------------------------------------------------------------------------

def shap_importance(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    TreeExplainer SHAP values for a fitted tree model.

    The returned DataFrame holds per-row, per-feature SHAP values for the
    positive class. A companion mean(|SHAP|) summary is attached via
    ``df.attrs["summary_importance"]`` so callers can treat it as either a
    detailed audit artefact or a scalar importance vector.

    Args:
        model:       Fitted LightGBM/XGBoost/RandomForest (or MetaLabeler).
        X:           Feature matrix to explain.
        max_samples: Subsample cap — SHAP scales linearly with rows.
    """
    import shap

    inner = _underlying_model(model)

    if len(X) > max_samples:
        X_sub = X.sample(n=max_samples, random_state=random_state)
    else:
        X_sub = X.copy()

    # TreeExplainer handles LightGBM, XGBoost, and sklearn RF uniformly.
    explainer = shap.TreeExplainer(inner)
    shap_values = explainer.shap_values(X_sub)

    # Normalize to a 2-D array of positive-class attributions:
    #   - old API (list of [class0, class1]) → pick class 1
    #   - new API (ndarray of shape (n, n_features, n_classes)) → pick axis -1
    #   - binary RF/XGB may already return (n, n_features) → use as-is
    arr = np.asarray(shap_values)
    if isinstance(shap_values, list):
        # Pick positive class if binary; else class-0 as a sensible default.
        arr = np.asarray(shap_values[1] if len(shap_values) >= 2 else shap_values[0])
    elif arr.ndim == 3:
        arr = arr[..., 1]

    df = pd.DataFrame(arr, columns=X_sub.columns, index=X_sub.index)
    df.attrs["summary_importance"] = (
        df.abs().mean().sort_values(ascending=False).rename("shap_mean_abs")
    )
    return df


# ---------------------------------------------------------------------------
# Composition + selection
# ---------------------------------------------------------------------------

def compute_all_importances(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    model_type: str = "lightgbm",
    mda_kwargs: dict[str, Any] | None = None,
    sfi_kwargs: dict[str, Any] | None = None,
    shap_kwargs: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run all four importance methods and wrap each as a DataFrame.

    MDI and SFI are naturally 1-D (Series) so they are promoted to
    single-column DataFrames (``importance`` and ``score`` respectively)
    for a consistent ``dict[str, pd.DataFrame]`` return type.
    """
    mda_kwargs = dict(mda_kwargs) if mda_kwargs else {}
    sfi_kwargs = dict(sfi_kwargs) if sfi_kwargs else {}
    shap_kwargs = dict(shap_kwargs) if shap_kwargs else {}

    mdi = mdi_importance(model, list(X.columns)).to_frame("importance")
    mda = mda_importance(
        model, X, y, labels_df, sample_weight=sample_weight, **mda_kwargs,
    )
    sfi = sfi_importance(
        X, y, labels_df, model_type=model_type,
        sample_weight=sample_weight, **sfi_kwargs,
    ).to_frame("score")
    shap_df = shap_importance(model, X, **shap_kwargs)
    return {"mdi": mdi, "mda": mda, "sfi": sfi, "shap": shap_df}


def select_features(
    importances: dict[str, pd.DataFrame],
    mda_pvalue_threshold: float = 0.05,
    min_sfi_score: float = 0.51,
) -> list[str]:
    """
    Keep features that clear either the MDA or the SFI bar.

    The union rule preserves features that are informative only through
    interactions (high MDA, low SFI) as well as features that are
    standalone-informative (low MDA can happen if the model relied on a
    correlated feature for most of the split count — SFI catches these).
    """
    if "mda" not in importances or "sfi" not in importances:
        raise ValueError(
            "importances dict must contain 'mda' and 'sfi' entries"
        )
    mda = importances["mda"]
    sfi = importances["sfi"]
    if "p_value" not in mda.columns:
        raise ValueError("mda DataFrame must have a 'p_value' column")

    sfi_col = "score" if "score" in sfi.columns else sfi.columns[0]

    all_features = set(mda.index).union(sfi.index)
    keep_by_mda = set(mda.index[mda["p_value"] < mda_pvalue_threshold])
    keep_by_sfi = set(sfi.index[sfi[sfi_col] > min_sfi_score])
    keep = keep_by_mda | keep_by_sfi

    dropped = all_features - keep
    if dropped:
        logger.info(
            f"select_features: dropping {len(dropped)} / {len(all_features)} "
            f"features (p-val >= {mda_pvalue_threshold} AND sfi <= "
            f"{min_sfi_score}): {sorted(dropped)}"
        )
    else:
        logger.info(
            f"select_features: keeping all {len(all_features)} features"
        )

    # Preserve MDA's sort order for determinism.
    ordered = [f for f in mda.index if f in keep]
    # Append any SFI-only features not present in MDA index (shouldn't happen
    # in practice but keep the function robust).
    for f in sfi.index:
        if f in keep and f not in ordered:
            ordered.append(f)
    return ordered
