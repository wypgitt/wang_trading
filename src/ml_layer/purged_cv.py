"""
Purged K-Fold Cross-Validation with Embargo (AFML Ch. 7)

Financial labels are **not** i.i.d. — triple-barrier labels have variable,
overlapping lifespans, and feature autocorrelation leaks information across
time. Standard k-fold CV trains on data that shares label periods with the
test fold, inflates measured accuracy, and produces strategies that collapse
out of sample.

This module implements AFML's two safeguards:

    * **Purge** — before training, drop any candidate sample whose label
      period ``[event_start, event_end]`` intersects the union of test-fold
      label periods. Eliminates *direct* leakage.
    * **Embargo** — additionally drop samples whose position falls within
      ``embargo_pct * len(X)`` positions forward of the test fold's end.
      Eliminates *indirect* leakage from autocorrelated features.

Standard CV is never used in the pipeline — this splitter replaces it.
See design doc §7.4.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_inputs(X, y, labels_df: pd.DataFrame) -> None:
    if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError("X must be a pandas DataFrame/Series or numpy array")
    if not isinstance(labels_df, pd.DataFrame):
        raise ValueError("labels_df must be a pandas DataFrame")
    missing = {"event_start", "event_end"} - set(labels_df.columns)
    if missing:
        raise ValueError(
            f"labels_df must have columns 'event_start' and 'event_end' "
            f"(missing: {sorted(missing)})"
        )
    if len(X) != len(labels_df):
        raise ValueError(
            f"X ({len(X)}) and labels_df ({len(labels_df)}) must align 1:1"
        )
    if y is not None and len(y) != len(X):
        raise ValueError(
            f"y ({len(y)}) and X ({len(X)}) must have the same length"
        )


def _purge_and_embargo(
    n: int,
    test_start_pos: int,
    test_end_pos: int,
    event_starts: np.ndarray,
    event_ends: np.ndarray,
    embargo_size: int,
) -> np.ndarray:
    """
    Return the training indices for one (test) fold after purge + embargo.

    Purging uses the merged test-label span ``[min(event_start), max(event_end)]``
    — any candidate whose label period overlaps that span is removed. This is
    conservative relative to per-test-sample purging (AFML Ch. 7), but
    equivalent when the test fold is temporally contiguous (which is always
    the case here since folds are position-slices of a time-ordered panel).
    Embargo is applied forward only — the backward direction is already
    covered by purging.
    """
    test_label_start = event_starts[test_start_pos:test_end_pos].min()
    test_label_end = event_ends[test_start_pos:test_end_pos].max()

    candidate = np.concatenate([
        np.arange(0, test_start_pos),
        np.arange(test_end_pos, n),
    ])

    # Purge: overlap with merged test span.
    cand_start = event_starts[candidate]
    cand_end = event_ends[candidate]
    overlap = (cand_end >= test_label_start) & (cand_start <= test_label_end)

    # Forward embargo.
    embargo_end = min(test_end_pos + embargo_size, n)
    in_embargo = (candidate >= test_end_pos) & (candidate < embargo_end)

    keep = ~overlap & ~in_embargo
    return candidate[keep]


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class PurgedKFoldCV:
    """
    AFML Ch. 7 purged k-fold with a forward embargo.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds. Folds are contiguous position-slices of the
        time-ordered sample array.
    embargo_pct : float, default 0.01
        Fraction of ``len(X)`` used as the forward embargo after each test
        fold. ``0.01`` on 10 000 samples = 100 samples embargoed.

    Notes
    -----
    * Callers must sort ``X``, ``y``, and ``labels_df`` consistently by time
      before calling ``split`` — the splitter treats positions as the time
      axis and does not reorder.
    * The splitter is compatible with scikit-learn's CV interface in spirit
      (``split`` / ``get_n_splits``) but accepts an extra ``labels_df`` arg
      which standard scikit-learn CV does not.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo_pct < 0.0 or embargo_pct >= 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    # X/y/groups kept in the signature to match scikit-learn's CV contract.
    # pylint: disable=unused-argument
    def get_n_splits(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray | None = None,
        y: pd.Series | np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return the configured fold count (scikit-learn CV interface)."""
        return self.n_splits
    # pylint: enable=unused-argument

    def split(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        y: pd.Series | np.ndarray | None,
        labels_df: pd.DataFrame,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield ``(train_indices, test_indices)`` pairs.

        Indices are positional (0-based) into the sample array; they may be
        used with ``X.iloc[...]`` / ``X[...]``.
        """
        _validate_inputs(X, y, labels_df)
        n = len(X)
        embargo_size = int(round(self.embargo_pct * n))

        # Contiguous, equal-size folds with the remainder absorbed by the
        # last fold.
        fold_size = n // self.n_splits
        fold_bounds: list[tuple[int, int]] = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            fold_bounds.append((start, end))

        # Cache as numpy for cheap slicing.
        event_starts = pd.to_datetime(labels_df["event_start"]).to_numpy()
        event_ends = pd.to_datetime(labels_df["event_end"]).to_numpy()

        for test_start_pos, test_end_pos in fold_bounds:
            test_indices = np.arange(test_start_pos, test_end_pos)
            train_indices = _purge_and_embargo(
                n, test_start_pos, test_end_pos,
                event_starts, event_ends, embargo_size,
            )
            logger.debug(
                f"PurgedKFold fold [{test_start_pos}:{test_end_pos}]: "
                f"{len(train_indices)} train / {len(test_indices)} test "
                f"({n - len(train_indices) - len(test_indices)} purged/embargoed)"
            )
            yield train_indices, test_indices


# ---------------------------------------------------------------------------
# Single train / test split (no shuffling)
# ---------------------------------------------------------------------------

def purged_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    test_size: float = 0.2,
    embargo_pct: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-ordered train/test split with purge + embargo.

    The test set is the **most recent** ``test_size`` fraction of the data
    (no shuffling — this is time-series ML). The training set is everything
    before, minus samples whose label periods overlap the test block, minus
    samples in the forward-embargo window.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    _validate_inputs(X, y, labels_df)
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must be in (0, 1)")

    n = len(X)
    n_test = max(1, int(round(test_size * n)))
    test_start_pos = n - n_test
    test_end_pos = n
    embargo_size = int(round(embargo_pct * n))

    event_starts = pd.to_datetime(labels_df["event_start"]).to_numpy()
    event_ends = pd.to_datetime(labels_df["event_end"]).to_numpy()

    train_indices = _purge_and_embargo(
        n, test_start_pos, test_end_pos,
        event_starts, event_ends, embargo_size,
    )
    test_indices = np.arange(test_start_pos, test_end_pos)

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Scoring dispatch
# ---------------------------------------------------------------------------

_SCORERS: dict[str, tuple[str, bool]] = {
    # key -> (sklearn scorer name, negate_result)
    "accuracy": ("accuracy", False),
    "f1": ("f1", False),
    "precision": ("precision", False),
    "recall": ("recall", False),
    "roc_auc": ("roc_auc", False),
    "neg_log_loss": ("neg_log_loss", False),
    "log_loss": ("neg_log_loss", True),  # return the positive loss
}


def cross_val_score_purged(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    sample_weight: pd.Series | None = None,
    scoring: str = "accuracy",
) -> np.ndarray:
    """
    Run purged k-fold CV and return per-fold scores.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Must implement ``fit`` and either ``predict`` or ``predict_proba``
        depending on the scoring metric. Cloned before each fit.
    sample_weight : optional per-sample weight Series (aligned with X).
        Passed to ``estimator.fit(..., sample_weight=...)`` using only the
        training indices for each fold.
    scoring : one of
        ``"accuracy"``, ``"f1"``, ``"precision"``, ``"recall"``,
        ``"roc_auc"``, ``"log_loss"``, ``"neg_log_loss"``.

    Returns
    -------
    np.ndarray of shape (n_splits,)
    """
    from sklearn.base import clone
    from sklearn.metrics import get_scorer

    if scoring not in _SCORERS:
        raise ValueError(
            f"unknown scoring {scoring!r}; choose from {sorted(_SCORERS)}"
        )

    sk_name, negate = _SCORERS[scoring]
    scorer = get_scorer(sk_name)

    cv = PurgedKFoldCV(n_splits=n_splits, embargo_pct=embargo_pct)
    scores: list[float] = []
    for train_idx, test_idx in cv.split(X, y, labels_df):
        if len(train_idx) == 0:
            logger.warning(
                "cross_val_score_purged: empty train set after purge/embargo; "
                "skipping fold"
            )
            scores.append(float("nan"))
            continue

        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        model = clone(estimator)
        fit_kwargs: dict = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(
                sample_weight.iloc[train_idx]
            )

        model.fit(X_tr, y_tr, **fit_kwargs)
        score = float(scorer(model, X_te, y_te))
        if negate:
            score = -score
        scores.append(score)

    return np.asarray(scores, dtype=float)
