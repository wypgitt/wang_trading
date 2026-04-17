"""
Tier 1 Meta-Labeler (design doc §7.1)

Gradient-boosted-tree meta-labeler that replaces AFML's Random-Forest
recommendation with the industry-consensus choice for tabular financial
data (LightGBM primary, XGBoost fallback). Takes the features produced
by :class:`src.labeling.meta_labeler_pipeline.MetaLabelingPipeline` and
outputs a calibrated probability ``p(profitable | signal, features)``
that feeds the bet-sizing cascade (§8).

Training hardening:
    * optional purged-k-fold early stopping for GBM backends
    * isotonic calibration of out-of-fold predictions
    * sample_weight plumbed end-to-end

Supported backends: ``lightgbm`` (default), ``xgboost``, ``random_forest``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "n_estimators": 500,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
}

_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "n_estimators": 500,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
}

# Random-Forest uses different knobs; subsample / colsample are GBM-only.
_DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "n_jobs": -1,
}

_MODEL_DEFAULTS = {
    "lightgbm": _DEFAULT_LGBM_PARAMS,
    "xgboost": _DEFAULT_XGB_PARAMS,
    "random_forest": _DEFAULT_RF_PARAMS,
}


# ---------------------------------------------------------------------------
# Probability calibrator
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """
    Isotonic regression calibrator for binary classifier outputs.

    Intended for post-hoc calibration: you already have raw probabilities
    from a fitted classifier and want ``P(calibrated = p)`` to match the
    empirical frequency of positives at predicted probability ``p``. Unlike
    scikit-learn's ``CalibratedClassifierCV`` (which wraps a classifier and
    re-runs CV internally), this operates directly on probability arrays —
    the right fit for calibrating on out-of-fold predictions from purged CV.

    The fitted isotonic regressor is monotone non-decreasing and clipped to
    ``[0, 1]``, so the calibrated output is guaranteed to be a valid
    probability and preserves rank order.
    """

    def __init__(self) -> None:
        self._isotonic: IsotonicRegression | None = None

    def fit(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """Fit the isotonic map from raw → calibrated probability."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred_proba, dtype=float)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred_proba must have the same shape")
        if len(y_true) == 0:
            raise ValueError("cannot fit calibrator on empty inputs")

        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if not finite.any():
            raise ValueError("no finite observations to fit calibrator on")
        y_true = y_true[finite]
        y_pred = y_pred[finite]

        self._isotonic = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True,
        )
        self._isotonic.fit(y_pred, y_true)
        return self

    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply the fitted calibration map."""
        if self._isotonic is None:
            raise RuntimeError("ProbabilityCalibrator is not fitted")
        out = self._isotonic.transform(np.asarray(y_pred_proba, dtype=float))
        return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Meta-labeler
# ---------------------------------------------------------------------------

class MetaLabeler:
    """
    Tier-1 meta-labeler: gradient-boosted binary classifier.

    Attributes
    ----------
    model_ : fitted classifier (LGBMClassifier / XGBClassifier / RandomForestClassifier)
    calibrator_ : optional :class:`ProbabilityCalibrator`
    feature_names_ : list[str] — column order used during fit
    best_iterations_ : list[int] — per-fold best iteration when purged CV was used
    oof_predictions_ : np.ndarray | None — OOF probabilities (length == len(X)) when
                       purged CV was used, else None. NaN at indices that were not
                       included in any training fold's test set.
    """

    _SUPPORTED_MODEL_TYPES = ("lightgbm", "xgboost", "random_forest")

    def __init__(
        self,
        model_type: str = "lightgbm",
        params: dict[str, Any] | None = None,
        calibrate: bool = True,
    ) -> None:
        if model_type not in self._SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self._SUPPORTED_MODEL_TYPES} "
                f"(got {model_type!r})"
            )
        self.model_type = model_type
        self.params = dict(params) if params else {}
        self.calibrate = calibrate

        self.model_: Any = None
        self.calibrator_: ProbabilityCalibrator | None = None
        self.feature_names_: list[str] = []
        self.best_iterations_: list[int] = []
        self.oof_predictions_: np.ndarray | None = None

    # ------------------------------------------------------------------ API --

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        labels_df: pd.DataFrame | None = None,
    ) -> "MetaLabeler":
        """
        Fit the meta-labeler.

        When ``labels_df`` is provided AND the backend supports early
        stopping (LightGBM / XGBoost), the fit proceeds via purged k-fold
        CV: each fold trains with early stopping on its purged validation
        set, out-of-fold predictions are collected, and the final model is
        refit on the full training set using the mean best-iteration as
        ``n_estimators``. The calibrator is then fit on the OOF preds —
        honest calibration that avoids using in-sample probabilities.

        When ``labels_df`` is None (or the backend is RandomForest), a
        plain fit runs and calibration (if enabled) falls back to
        in-sample probabilities with a warning.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X ({len(X)}) and y ({len(y)}) must have the same length"
            )
        self.feature_names_ = list(X.columns)
        self.best_iterations_ = []
        self.oof_predictions_ = None

        base_params = self._resolve_params()
        use_purged_cv = (
            labels_df is not None and self.model_type in ("lightgbm", "xgboost")
        )

        if use_purged_cv:
            assert labels_df is not None  # for type-narrowing
            oof = self._fit_with_purged_cv(X, y, sample_weight, labels_df, base_params)
            self.oof_predictions_ = oof
        else:
            if labels_df is not None and self.model_type == "random_forest":
                logger.info(
                    "MetaLabeler: RandomForest backend does not support early "
                    "stopping; ignoring labels_df and running plain fit"
                )
            self.model_ = self._build_model(base_params)
            fit_kwargs: dict[str, Any] = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = np.asarray(sample_weight)
            self.model_.fit(X, y, **fit_kwargs)

        if self.calibrate:
            self.calibrator_ = self._fit_calibrator(X, y)
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        *,
        db_manager: object | None = None,
        symbol: str | None = None,
        signal_family: str = "",
        model_version: str = "",
    ) -> np.ndarray:
        """Return P(class=1) per row, calibrated when a calibrator is fitted.

        When ``db_manager`` and ``symbol`` are supplied, every returned
        probability is persisted to the ``meta_labels`` hypertable.
        """
        if self.model_ is None:
            raise RuntimeError("MetaLabeler is not fitted")
        raw = self.model_.predict_proba(X)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise RuntimeError(
                f"expected predict_proba to return a 2-column matrix "
                f"(got shape {raw.shape})"
            )
        raw_p1 = raw[:, 1]
        p1 = raw_p1
        if self.calibrator_ is not None:
            p1 = self.calibrator_.transform(p1)

        if db_manager is not None and symbol is not None:
            self._persist_meta_labels(
                X, raw_p1, p1, db_manager, symbol, signal_family, model_version,
            )
        return p1

    @staticmethod
    def _persist_meta_labels(
        X: pd.DataFrame, raw: np.ndarray, calibrated: np.ndarray,
        db_manager: object, symbol: str, signal_family: str, model_version: str,
    ) -> None:
        import asyncio as _asyncio
        timestamps = X.index if hasattr(X, "index") else range(len(raw))
        loop = _asyncio.new_event_loop()
        try:
            for i, ts in enumerate(timestamps):
                try:
                    loop.run_until_complete(db_manager.insert_meta_label(  # type: ignore[attr-defined]
                        timestamp=ts,
                        symbol=symbol,
                        signal_family=signal_family,
                        meta_prob=float(raw[i]),
                        calibrated_prob=float(calibrated[i]),
                        model_version=model_version,
                    ))
                except Exception:  # pragma: no cover - best-effort
                    break
        finally:
            loop.close()

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction at the given probability threshold."""
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, method: str = "gain") -> pd.Series:
        """
        Return a feature-importance Series, sorted descending.

        ``method`` ∈ ``{"gain", "split"}``. "gain" maps to the backend's
        native gain-based importance (gini for RandomForest); "split" maps
        to split counts (LightGBM / XGBoost only — raises for RF).
        """
        if self.model_ is None:
            raise RuntimeError("MetaLabeler is not fitted")
        if method not in ("gain", "split"):
            raise ValueError("method must be 'gain' or 'split'")

        if self.model_type == "lightgbm":
            booster = self.model_.booster_
            raw = booster.feature_importance(importance_type=method)
            names = booster.feature_name()
        elif self.model_type == "xgboost":
            xgb_type = "gain" if method == "gain" else "weight"
            booster = self.model_.get_booster()
            scores = booster.get_score(importance_type=xgb_type)
            # booster.get_score omits features with 0 importance; pad them.
            names = self.feature_names_
            raw = np.asarray([scores.get(n, 0.0) for n in names], dtype=float)
        else:  # random_forest
            if method == "split":
                raise ValueError(
                    "RandomForest does not expose split-count importance; "
                    "use method='gain'"
                )
            raw = np.asarray(self.model_.feature_importances_, dtype=float)
            names = self.feature_names_

        return pd.Series(raw, index=names, name=f"importance_{method}").sort_values(
            ascending=False,
        )

    # ------------------------------------------------------------------ helpers --

    def _resolve_params(self) -> dict[str, Any]:
        defaults = dict(_MODEL_DEFAULTS[self.model_type])
        defaults.update(self.params)
        return defaults

    def _build_model(
        self,
        params: dict[str, Any],
        early_stopping_rounds: int | None = None,
    ) -> Any:
        if self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            p = dict(params)
            p.setdefault("verbose", -1)
            return LGBMClassifier(**p)
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            p = dict(params)
            p.setdefault("verbosity", 0)
            # Only set early_stopping_rounds on the constructor when
            # early-stopping fits are planned (an eval_set will be supplied).
            if early_stopping_rounds is not None:
                p["early_stopping_rounds"] = early_stopping_rounds
            return XGBClassifier(**p)
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        raise ValueError(f"unknown model_type: {self.model_type}")

    def _fit_fold(
        self,
        fold_params: dict[str, Any],
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sw_tr: np.ndarray | None,
        early_stopping_rounds: int,
    ) -> tuple[Any, int]:
        """Fit one fold with early stopping; return (model, best_iteration)."""
        fit_kwargs: dict[str, Any] = {"eval_set": [(X_val, y_val)]}
        if sw_tr is not None:
            fit_kwargs["sample_weight"] = sw_tr

        if self.model_type == "lightgbm":
            import lightgbm as lgb
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            ]
            model = self._build_model(fold_params)
            model.fit(X_tr, y_tr, **fit_kwargs)
            best = model.best_iteration_ or int(fold_params["n_estimators"])
            return model, int(best)

        if self.model_type == "xgboost":
            model = self._build_model(
                fold_params, early_stopping_rounds=early_stopping_rounds,
            )
            model.fit(X_tr, y_tr, **fit_kwargs)
            # XGBoost 2.x+: best_iteration set after early stopping (0-indexed).
            best = getattr(model, "best_iteration", None)
            if best is None:
                best = int(fold_params["n_estimators"])
            # best_iteration is 0-indexed; add 1 so it matches n_estimators semantics.
            return model, int(best) + 1

        raise RuntimeError(
            f"early stopping not supported for backend {self.model_type!r}"
        )

    def _fit_with_purged_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None,
        labels_df: pd.DataFrame,
        base_params: dict[str, Any],
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        early_stopping_rounds: int = 30,
    ) -> np.ndarray:
        """Purged-CV fit with per-fold early stopping; returns OOF probs."""
        from src.ml_layer.purged_cv import PurgedKFoldCV

        cv = PurgedKFoldCV(n_splits=n_splits, embargo_pct=embargo_pct)
        n = len(X)
        oof = np.full(n, np.nan, dtype=float)
        best_iters: list[int] = []

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, labels_df)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.warning(
                    f"MetaLabeler: fold {fold_i} has empty split; skipping"
                )
                continue

            X_tr = X.iloc[train_idx]
            X_val = X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[test_idx]
            sw_tr = None
            if sample_weight is not None:
                sw_tr = np.asarray(sample_weight.iloc[train_idx])

            # Skip degenerate folds (single class in val → can't early-stop on AUC-like metrics).
            if y_val.nunique() < 2:
                logger.debug(
                    f"MetaLabeler: fold {fold_i} val is single-class; "
                    "fitting without early stopping"
                )
                fallback = self._build_model(base_params)
                fit_kwargs: dict[str, Any] = {}
                if sw_tr is not None:
                    fit_kwargs["sample_weight"] = sw_tr
                fallback.fit(X_tr, y_tr, **fit_kwargs)
                oof[test_idx] = fallback.predict_proba(X_val)[:, 1]
                best_iters.append(int(base_params["n_estimators"]))
                continue

            fold_model, best = self._fit_fold(
                base_params, X_tr, y_tr, X_val, y_val, sw_tr,
                early_stopping_rounds=early_stopping_rounds,
            )
            oof[test_idx] = fold_model.predict_proba(X_val)[:, 1]
            best_iters.append(best)

        self.best_iterations_ = best_iters

        # Refit on the full dataset using the averaged best iteration.
        final_params = dict(base_params)
        if best_iters:
            final_params["n_estimators"] = max(
                int(round(float(np.mean(best_iters)))), 10,
            )
        self.model_ = self._build_model(final_params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight)
        self.model_.fit(X, y, **fit_kwargs)
        return oof

    def _fit_calibrator(
        self, X: pd.DataFrame, y: pd.Series,
    ) -> ProbabilityCalibrator | None:
        """Fit isotonic calibration on OOF preds when available, else in-sample."""
        y_arr = np.asarray(y, dtype=float)
        if self.oof_predictions_ is not None:
            mask = np.isfinite(self.oof_predictions_)
            if not mask.any():
                logger.warning(
                    "MetaLabeler: no finite OOF predictions; skipping calibration"
                )
                return None
            return ProbabilityCalibrator().fit(
                y_arr[mask], self.oof_predictions_[mask],
            )

        logger.warning(
            "MetaLabeler: calibrating on in-sample probabilities — pass "
            "labels_df to fit() for honest out-of-fold calibration"
        )
        in_sample = self.model_.predict_proba(X)[:, 1]
        return ProbabilityCalibrator().fit(y_arr, in_sample)
