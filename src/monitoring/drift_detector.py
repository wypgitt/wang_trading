"""Feature drift detection (Phase 5 / P5.10).

Design doc §12.4: every feature is monitored for distribution drift versus
its training baseline. Four statistics are computed per feature — KL
divergence, KS test, mean shift (z-score), variance ratio — and a feature
is flagged if any exceed its threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ── Thresholds ─────────────────────────────────────────────────────────
KL_THRESHOLD = 0.5
KS_PVALUE_THRESHOLD = 0.01
MEAN_SHIFT_Z_THRESHOLD = 3.0
VAR_RATIO_UPPER = 2.0
VAR_RATIO_LOWER = 0.5


@dataclass
class _FeatureBaseline:
    mean: float
    std: float
    var: float
    hist: np.ndarray  # density values per bin
    edges: np.ndarray  # bin edges
    samples: np.ndarray  # retained for KS test


class FeatureDriftDetector:
    """Monitors features for drift vs training baselines."""

    def __init__(self, n_bins: int = 50) -> None:
        self.n_bins = n_bins
        self._baselines: dict[str, _FeatureBaseline] = {}

    # ── Baseline ───────────────────────────────────────────────────────

    def set_baseline(self, training_features: pd.DataFrame) -> None:
        self._baselines = {}
        for col in training_features.columns:
            vals = training_features[col].dropna().to_numpy(dtype=float)
            if len(vals) < 2:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1))
            var = float(vals.var(ddof=1))
            lo, hi = float(vals.min()), float(vals.max())
            if hi <= lo:
                hi = lo + 1e-9
            edges = np.linspace(lo, hi, self.n_bins + 1)
            hist, _ = np.histogram(vals, bins=edges, density=True)
            self._baselines[col] = _FeatureBaseline(
                mean=mean, std=std, var=var, hist=hist, edges=edges, samples=vals,
            )

    # ── Core check ─────────────────────────────────────────────────────

    def check_drift(
        self, current_features: pd.DataFrame, window: int = 100
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for col, base in self._baselines.items():
            if col not in current_features.columns:
                continue
            recent = current_features[col].dropna().to_numpy(dtype=float)
            if len(recent) == 0:
                continue
            if len(recent) > window:
                recent = recent[-window:]
            row = self._metrics_for(col, recent, base)
            rows.append(row)
        return pd.DataFrame(rows)

    def _metrics_for(
        self, name: str, recent: np.ndarray, base: _FeatureBaseline,
    ) -> dict:
        # Histogram over baseline edges (allow out-of-range via clipping)
        clipped = np.clip(recent, base.edges[0], base.edges[-1])
        cur_hist, _ = np.histogram(clipped, bins=base.edges, density=True)
        # Widths: convert density → probability
        widths = np.diff(base.edges)
        p = cur_hist * widths
        q = base.hist * widths
        # Smooth zeros
        eps = 1e-9
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        kl = float(np.sum(p * np.log(p / q)))

        # KS test against baseline samples
        if len(recent) >= 2 and len(base.samples) >= 2:
            ks_stat, ks_p = stats.ks_2samp(recent, base.samples)
            ks_stat = float(ks_stat)
            ks_p = float(ks_p)
        else:
            ks_stat, ks_p = 0.0, 1.0

        mean_shift = (
            (recent.mean() - base.mean) / base.std if base.std > 0 else 0.0
        )
        cur_var = float(recent.var(ddof=1)) if len(recent) > 1 else 0.0
        var_ratio = (cur_var / base.var) if base.var > 0 else 0.0

        drifted = (
            kl > KL_THRESHOLD
            or ks_p < KS_PVALUE_THRESHOLD
            or abs(mean_shift) > MEAN_SHIFT_Z_THRESHOLD
            or var_ratio > VAR_RATIO_UPPER
            or (var_ratio < VAR_RATIO_LOWER and cur_var > 0)
        )

        return {
            "feature": name,
            "kl_divergence": kl,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "mean_shift": float(mean_shift),
            "var_ratio": var_ratio,
            "drifted": bool(drifted),
        }

    # ── Convenience ────────────────────────────────────────────────────

    def get_drifted_features(self, current_features: pd.DataFrame) -> list[str]:
        df = self.check_drift(current_features)
        if df.empty:
            return []
        return df.loc[df["drifted"], "feature"].tolist()

    @staticmethod
    def recommend_action(drifted: list[str], total_features: int) -> str:
        if total_features <= 0 or not drifted:
            return "No action needed"
        pct = len(drifted) / total_features
        if pct < 0.20:
            return "Monitor — consider mask drifted features to training median"
        if pct < 0.50:
            return "Warning — regime change likely. Retrain model soon."
        return "Critical — major regime shift. Reduce position sizes. Immediate retrain."
