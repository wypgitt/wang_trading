"""
Combinatorial Purged Cross-Validation (AFML Ch. 12–13, design-doc §9.1).

Given N time-ordered groups and a test-group size k, CPCV enumerates every
C(N, k) train/test split. For each split the training set is everything
outside the chosen k groups, further pruned by the purge + embargo rules
from :mod:`src.ml_layer.purged_cv` so overlapping labels never leak.

Each split is called a *path* here (matching the design-doc §9.1 "45 paths"
language for the default N=10, k=2 configuration). A path is returned as a
list containing a single ``(train_idx, test_idx)`` tuple; the outer list
shape keeps the API forward-compatible with AFML's multi-segment path
assembly, where one path can stitch multiple test folds into a full
out-of-sample trajectory.

Design-doc §9.1 requires ≥ 60% of paths to show positive net returns —
that's what :func:`validate_strategy` enforces.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.backtesting.walk_forward import BacktestResult, WalkForwardBacktester
from src.ml_layer.purged_cv import _purge_and_embargo  # noqa: PLC2701


Path = list[tuple[np.ndarray, np.ndarray]]


class CPCVEngine:
    """Enumerate all C(N, k) purged train/test paths."""

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not (1 <= n_test_groups < n_groups):
            raise ValueError("n_test_groups must satisfy 1 <= k < n_groups")
        if embargo_pct < 0.0 or embargo_pct >= 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    # ── path construction ──────────────────────────────────────────────

    def _group_bounds(self, n: int) -> list[tuple[int, int]]:
        size = n // self.n_groups
        bounds: list[tuple[int, int]] = []
        for i in range(self.n_groups):
            start = i * size
            end = (i + 1) * size if i < self.n_groups - 1 else n
            bounds.append((start, end))
        return bounds

    # pylint: disable=unused-argument
    def generate_paths(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        labels_df: pd.DataFrame,
    ) -> list[Path]:
        """Build all C(n_groups, n_test_groups) purged paths.

        Each returned path is ``[(train_idx, test_idx)]`` where ``test_idx``
        is the concatenation of the chosen k test groups and ``train_idx`` is
        everything else after purge + embargo against the test label span.
        """
        if {"event_start", "event_end"} - set(labels_df.columns):
            raise ValueError("labels_df requires 'event_start' and 'event_end'")
        if len(X) != len(labels_df):
            raise ValueError("X and labels_df must align 1:1")

        n = len(X)
        bounds = self._group_bounds(n)
        embargo_size = int(round(self.embargo_pct * n))

        event_starts = pd.to_datetime(labels_df["event_start"]).to_numpy()
        event_ends = pd.to_datetime(labels_df["event_end"]).to_numpy()

        paths: list[Path] = []
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_indices = np.concatenate(
                [np.arange(bounds[g][0], bounds[g][1]) for g in combo]
            )
            # Purge against the merged test-label span (min event_start over
            # chosen groups → max event_end over chosen groups), matching the
            # PurgedKFold convention.
            span_start = int(min(bounds[g][0] for g in combo))
            span_end = int(max(bounds[g][1] for g in combo))
            train_indices = _purge_and_embargo(
                n,
                span_start,
                span_end,
                event_starts,
                event_ends,
                embargo_size,
            )
            # Remove any training samples that slipped into the test groups
            # via the ``[span_start, span_end]`` range but don't belong to
            # the chosen groups (i.e., groups inside the span but not in
            # the combination).
            train_indices = np.setdiff1d(
                train_indices, test_indices, assume_unique=False
            )
            paths.append([(train_indices, test_indices)])

        logger.debug(
            f"CPCVEngine: generated {len(paths)} paths "
            f"(N={self.n_groups}, k={self.n_test_groups})"
        )
        return paths

    # ── backtest orchestration ─────────────────────────────────────────

    def run_backtest_paths(
        self,
        backtester: WalkForwardBacktester,
        paths: list[Path],
        close: pd.DataFrame,
        features_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        meta_labeling_pipeline: Callable,
        model_class: Any,
        **model_params: Any,
    ) -> list[BacktestResult]:
        """Fit a fresh model on each path's train slice, backtest the test slice.

        ``meta_labeling_pipeline`` is a callable ``(features, signals) -> (X, y, sample_weight)``
        that converts raw features + signal side into the supervised training
        tuple expected by ``model_class``. The fitted model is queried via
        ``predict_proba`` to produce the meta-label probabilities fed to the
        backtester.
        """
        results: list[BacktestResult] = []
        for path in paths:
            train_idx, test_idx = path[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            feats_train = features_df.iloc[train_idx]
            sigs_train = signals_df.iloc[train_idx]
            X_tr, y_tr, w_tr = meta_labeling_pipeline(feats_train, sigs_train)

            model = model_class(**model_params)
            fit_kwargs = {}
            if w_tr is not None:
                fit_kwargs["sample_weight"] = np.asarray(w_tr)
            model.fit(X_tr, y_tr, **fit_kwargs)

            feats_test = features_df.iloc[test_idx]
            sigs_test = signals_df.iloc[test_idx]
            # predict_proba's last column is P(class=1) for binary meta-labels
            proba = model.predict_proba(feats_test)
            probs_arr = proba[:, -1] if getattr(proba, "ndim", 1) > 1 else proba
            test_index = features_df.index[test_idx]
            probs = pd.DataFrame(
                np.broadcast_to(
                    probs_arr[:, None], (len(test_idx), close.shape[1])
                ).copy(),
                index=test_index,
                columns=close.columns,
            )

            close_test = close.loc[test_index]
            sizes_test = pd.DataFrame(
                0.05, index=test_index, columns=close.columns
            )
            result = backtester.run(
                close=close_test,
                signals_df=sigs_test.reindex(columns=close.columns).fillna(0),
                meta_probs=probs,
                bet_sizes=sizes_test,
            )
            results.append(result)
        return results

    # ── aggregation helpers ────────────────────────────────────────────

    @staticmethod
    def assemble_equity_curves(results: list[BacktestResult]) -> pd.DataFrame:
        """Stack each path's equity curve into a single DataFrame.

        Columns are ``path_0 … path_{N-1}``. Curves are aligned on the union
        of their indices; non-overlapping bars are filled with ``NaN``.
        """
        if not results:
            return pd.DataFrame()
        series = {
            f"path_{i}": r.equity_curve for i, r in enumerate(results)
        }
        return pd.concat(series, axis=1)

    @staticmethod
    def get_path_statistics(results: list[BacktestResult]) -> pd.DataFrame:
        """Per-path performance stats plus a summary footer."""
        if not results:
            return pd.DataFrame()

        rows: list[dict] = []
        for i, r in enumerate(results):
            m = r.metrics
            rows.append(
                {
                    "path": f"path_{i}",
                    "total_return": m.get("total_return", 0.0),
                    "sharpe": m.get("sharpe", 0.0),
                    "max_drawdown": m.get("max_drawdown", 0.0),
                    "win_rate": m.get("win_rate", 0.0),
                    "profit_factor": m.get("profit_factor", 0.0),
                    "total_trades": m.get("total_trades", 0),
                }
            )
        df = pd.DataFrame(rows).set_index("path")

        numeric_cols = df.select_dtypes(include="number").columns
        summary = pd.DataFrame(
            {
                "mean": df[numeric_cols].mean(),
                "std": df[numeric_cols].std(ddof=0),
                "min": df[numeric_cols].min(),
                "max": df[numeric_cols].max(),
            }
        ).T
        # fraction of paths with positive returns
        positive_pct = float((df["total_return"] > 0).mean())
        summary.loc["positive_pct", "total_return"] = positive_pct
        return pd.concat([df, summary])


# ── gate 1: ≥60% positive paths ────────────────────────────────────────


def validate_strategy(
    results: list[BacktestResult],
    min_positive_paths_pct: float = 0.60,
) -> tuple[bool, dict]:
    """Gate 1 (design-doc §9.1): fraction of paths with positive returns."""

    if not results:
        return False, {
            "path_count": 0,
            "positive_count": 0,
            "positive_pct": 0.0,
            "mean_sharpe": 0.0,
            "mean_return": 0.0,
            "threshold": min_positive_paths_pct,
        }

    returns = np.array([r.metrics.get("total_return", 0.0) for r in results])
    sharpes = np.array([r.metrics.get("sharpe", 0.0) for r in results])
    drawdowns = np.array([r.metrics.get("max_drawdown", 0.0) for r in results])

    positive = returns > 0
    positive_count = int(positive.sum())
    positive_pct = float(positive_count / len(results))

    stats = {
        "path_count": len(results),
        "positive_count": positive_count,
        "positive_pct": positive_pct,
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "mean_sharpe": float(sharpes.mean()),
        "median_sharpe": float(np.median(sharpes)),
        "mean_max_drawdown": float(drawdowns.mean()),
        "worst_drawdown": float(drawdowns.min()),
        "threshold": min_positive_paths_pct,
    }
    passed = positive_pct >= min_positive_paths_pct
    return passed, stats
