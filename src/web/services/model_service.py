"""Production meta-labeler card service for ``GET /api/v1/model``.

The model card is sourced from the MLflow registry (the second of the two
real engine outputs). :class:`ModelRegistry.get_production_model` returns the
production-aliased run's params, metrics, gate verdicts, and training metadata
— all of which are REAL and mapped straight through here. Everything the
engine does not log yet (AUC/Brier/ECE, calibration buckets, feature
importance, drift, RL-shadow) stays null/empty and is named in a warning;
it is never synthesised.

Every external call is injectable so tests need no MLflow/DB infra:

* ``registry_factory`` builds the :class:`ModelRegistry` (default lazily
  constructs one; wrapped in try/except so a missing/broken MLflow store
  degrades to an all-null card rather than a 500).
* ``meta_probs_provider`` returns the recent meta-probabilities for the
  histogram. The live cycle does not persist ``meta_labels`` yet, so the
  honest default is an EMPTY list — no fabricated distribution.
* ``now_fn`` supplies "now" for the ``last_retrain_hours`` delta.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..cache import LRUTTLCache
from ..dtos import (
    MetaProbBucket,
    ModelGates,
    ModelResponse,
)

log = logging.getLogger(__name__)


# The single warning that names every COMING / not-logged field, so the
# client can render the honest "not produced yet" affordances. The retrain
# timeline is included because the registry exposes only a CV-score ranking
# (get_best_model orders by metrics.mean_cv_score DESC and carries no run
# dates) — a chronological promote/reject history needs a time-ordered MLflow
# query that does not exist yet, so serving the ranking as a "timeline" would
# be misleading. Held empty until that query lands.
_NOT_PRODUCED = (
    "auc/brier/ece, calibration, feature importance, drift, retrain timeline "
    "and RL-shadow not produced by the engine yet"
)

# Histogram resolution for the meta-probability distribution: ten [0,1) bins.
_HIST_BINS = 10


def _default_registry_factory():
    # Imported lazily so the BFF stays importable where MLflow is absent.
    from src.ml_layer.model_registry import ModelRegistry

    return ModelRegistry()


class ModelService:
    """Build the production meta-labeler card, degrading to all-null."""

    # Cache keys (maxsize-1 cache, so the key is constant).
    _KEY = "model_card"

    def __init__(
        self,
        *,
        registry_factory: Optional[Callable[[], Any]] = None,
        meta_probs_provider: Optional[Callable[[], list[float]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        cache_ttl_seconds: float = 60.0,
        error_ttl_seconds: float = 5.0,
    ) -> None:
        self._registry_factory = registry_factory or _default_registry_factory
        self._meta_probs_provider = meta_probs_provider or (lambda: [])
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        # The MLflow registry is built ONCE (the import + set_experiment cost
        # ~1s) and reused; the result is memoised so the hot read is sub-ms.
        # The retrain loop promotes at most hourly, so a 60s TTL still feels
        # live; transient errors get a short TTL so they self-heal quickly.
        self._registry: Any = None
        self._ttl = float(cache_ttl_seconds)
        self._error_ttl = float(error_ttl_seconds)
        self._cache = LRUTTLCache(maxsize=1, ttl=self._ttl)

    def _get_registry(self) -> Any:
        """The single, reused MLflow registry (built lazily on first success)."""

        if self._registry is None:
            self._registry = self._registry_factory()
        return self._registry

    def get_model(self) -> tuple[ModelResponse, list[str]]:
        """Return ``(response, warnings)`` for the model card (TTL-cached).

        Degrades rather than raises:
        * registry construction / ``get_production_model`` failure
          -> all-null card + ``'model registry unavailable'`` (cached briefly).
        * no production model registered
          -> all-null card + ``'no production model registered'`` (the
          client's MODEL_REQUIRED state).
        """

        cached = self._cache.get(self._KEY, default=None)
        if cached is not None:
            return cached
        result, ttl = self._compute()
        self._cache.set(self._KEY, result, ttl_seconds=ttl)
        return result

    def _compute(self) -> tuple[tuple[ModelResponse, list[str]], float]:
        try:
            registry = self._get_registry()
            info = registry.get_production_model()
        except Exception as exc:  # noqa: BLE001 — degrade, never 500
            log.warning("model registry unavailable: %s", exc)
            # Reset so a transient construction failure is retried next time.
            self._registry = None
            return (ModelResponse(), ["model registry unavailable"]), self._error_ttl

        if info is None:
            return (ModelResponse(), ["no production model registered"]), self._ttl

        warnings: list[str] = [_NOT_PRODUCED]

        params = info.get("params") or {}
        metrics = info.get("metrics") or {}
        gates_raw = info.get("gates") or {}

        trained_at = info.get("trained_at")
        last_retrain_hours = self._hours_since(trained_at)

        response = ModelResponse(
            version=info.get("version") or info.get("name"),
            run_id=info.get("run_id"),
            type=params.get("model_type"),
            trained_at=trained_at,
            last_retrain_hours=last_retrain_hours,
            training_events=info.get("n_training_events"),
            cv_score=_optional_float(metrics.get("mean_cv_score")),
            train_acc=_optional_float(metrics.get("train_accuracy")),
            gates=ModelGates(
                cpcv=gates_raw.get("cpcv"),
                dsr=gates_raw.get("dsr"),
                pbo=gates_raw.get("pbo"),
            ),
            # COMING: a chronological retrain timeline needs a time-ordered
            # registry query (see _NOT_PRODUCED) — held empty, never the
            # CV-ranked, dateless get_best_model output mislabeled as a timeline.
            retrain_timeline=[],
            meta_prob_hist=self._meta_prob_hist(),
            # COMING / not-logged — held null/empty, never synthesised:
            auc=None,
            brier=None,
            ece=None,
            calibration=[],
            feature_importance=[],
            drift=[],
            rl_shadow=None,
        )
        return (response, warnings), self._ttl

    # ── internals ─────────────────────────────────────────────────────

    def _hours_since(self, trained_at: Optional[datetime]) -> Optional[float]:
        if not isinstance(trained_at, datetime):
            return None
        now = self._now_fn()
        ref = trained_at
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - ref).total_seconds() / 3600.0

    def _meta_prob_hist(self) -> list[MetaProbBucket]:
        """Bucket recent meta-probabilities into ten [0,1) bins.

        Returns ``[]`` when the provider yields nothing (the honest default:
        the live cycle does not persist meta_labels yet).
        """

        try:
            probs = self._meta_probs_provider()
        except Exception as exc:  # noqa: BLE001
            log.warning("meta-prob provider unavailable: %s", exc)
            return []

        if not probs:
            return []

        counts = [0] * _HIST_BINS
        for raw in probs:
            value = _optional_float(raw)
            if value is None:
                continue
            # Clamp into [0, 1) then map to a bin index 0..9.
            clamped = min(max(value, 0.0), 0.999999)
            idx = int(clamped * _HIST_BINS)
            if idx >= _HIST_BINS:
                idx = _HIST_BINS - 1
            counts[idx] += 1

        return [
            MetaProbBucket(bucket=f"{i / _HIST_BINS:.1f}", count=counts[i])
            for i in range(_HIST_BINS)
        ]


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


__all__ = ["ModelService"]
