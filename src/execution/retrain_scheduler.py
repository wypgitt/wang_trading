"""Model retraining scheduler (Phase 5 / P5.14).

Runs alongside the paper-trading loop and retrains the meta-labeler on a
cadence. Only promotes a new model if purged-CV performance improves over
the incumbent. Alerts on every outcome (promoted / kept / failed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.monitoring.alerting import Alert, AlertManager, AlertSeverity

log = logging.getLogger(__name__)


# Trainer callable signature:
#   trainer(close, features, signals, meta_pipeline) -> (new_model, cv_score)
TrainerFn = Callable[
    [pd.Series, pd.DataFrame, pd.DataFrame, Any],
    tuple[Any, float],
]


@dataclass
class RetrainStatus:
    last_retrain: datetime | None = None
    next_scheduled: datetime | None = None
    new_bars_since_retrain: int = 0
    promotions: int = 0
    rejected: int = 0


class RetrainScheduler:
    """Decide when to retrain, run it, and promote only if improved."""

    def __init__(
        self,
        *,
        retrain_interval_days: int = 7,
        min_new_bars: int = 100,
        model_registry: Any | None = None,
        alert_manager: AlertManager,
        trainer: TrainerFn | None = None,
        evaluator: Callable[[Any, pd.DataFrame, pd.Series], float] | None = None,
    ) -> None:
        self.retrain_interval_days = retrain_interval_days
        self.min_new_bars = min_new_bars
        self.model_registry = model_registry
        self.alert_manager = alert_manager
        self.trainer = trainer
        self.evaluator = evaluator
        self.status = RetrainStatus()

    # ── Scheduling ─────────────────────────────────────────────────────

    async def should_retrain(
        self,
        current_model_age_hours: float,
        new_bars_since_retrain: int,
    ) -> bool:
        age_threshold = self.retrain_interval_days * 24.0
        return (
            current_model_age_hours >= age_threshold
            and new_bars_since_retrain >= self.min_new_bars
        )

    # ── Retrain ────────────────────────────────────────────────────────

    async def retrain(
        self,
        close: pd.Series,
        features: pd.DataFrame,
        signals: pd.DataFrame,
        meta_pipeline: Any,
        current_model: Any,
        *,
        eval_X: pd.DataFrame | None = None,
        eval_y: pd.Series | None = None,
    ) -> Any | None:
        """Train a new model and promote if it improves on the incumbent.

        Returns the promoted model or None if the incumbent was retained.
        Requires either `self.trainer` (returns (model, cv_score)) or
        `self.evaluator` + an `eval_X/eval_y` pair to score both models.
        """
        if self.trainer is None:
            await self._alert(
                severity=AlertSeverity.WARNING,
                title="Retrain skipped",
                message="No trainer configured; retrain is a no-op",
            )
            return None

        try:
            new_model, new_score = self.trainer(close, features, signals, meta_pipeline)
        except Exception as exc:
            log.exception("retrain failed")
            await self._alert(
                severity=AlertSeverity.CRITICAL,
                title="Retrain failed",
                message=f"Exception during training: {exc}",
            )
            return None

        current_score = self._score(current_model, eval_X, eval_y)
        promoted = self._should_promote(new_score, current_score)

        self.status.last_retrain = datetime.now(timezone.utc)
        self.status.next_scheduled = (
            self.status.last_retrain + timedelta(days=self.retrain_interval_days)
        )
        self.status.new_bars_since_retrain = 0

        if promoted:
            self.status.promotions += 1
            if self.model_registry is not None:
                try:
                    run_id = self.model_registry.log_training_run(
                        model=new_model, params={}, metrics={"cv_score": new_score},
                    ) if hasattr(self.model_registry, "log_training_run") else None
                    if run_id is not None and hasattr(self.model_registry, "promote_model"):
                        self.model_registry.promote_model(run_id)
                except Exception as exc:
                    log.warning("model registry logging failed: %s", exc)
            await self._alert(
                severity=AlertSeverity.INFO,
                title="Model updated",
                message=(
                    f"New meta-labeler promoted (CV {new_score:.4f} "
                    f"vs incumbent {current_score:.4f})"
                ),
            )
            return new_model

        self.status.rejected += 1
        await self._alert(
            severity=AlertSeverity.WARNING,
            title="Retrain did not improve",
            message=(
                f"Keeping incumbent: new CV {new_score:.4f} "
                f"≤ incumbent {current_score:.4f}"
            ),
        )
        return None

    # ── Status ─────────────────────────────────────────────────────────

    def get_retrain_status(self) -> dict:
        last = self.status.last_retrain
        age_hours = (
            (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
            if last else None
        )
        return {
            "last_retrain": last,
            "age_hours": age_hours,
            "new_bars_since_retrain": self.status.new_bars_since_retrain,
            "next_scheduled": self.status.next_scheduled,
            "promotions": self.status.promotions,
            "rejected": self.status.rejected,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _score(
        self, model: Any | None, X: pd.DataFrame | None, y: pd.Series | None,
    ) -> float:
        if model is None or X is None or y is None or self.evaluator is None:
            return float("-inf")
        try:
            return float(self.evaluator(model, X, y))
        except Exception as exc:
            log.warning("evaluator failed: %s", exc)
            return float("-inf")

    @staticmethod
    def _should_promote(new_score: float, current_score: float) -> bool:
        if not np.isfinite(new_score):
            return False
        if not np.isfinite(current_score):
            return True
        return new_score > current_score

    async def _alert(self, *, severity: AlertSeverity, title: str, message: str) -> None:
        await self.alert_manager.send_alert(
            Alert(severity=severity, title=title, message=message, source="retrain"),
        )
