"""Freshness SLO + readiness for the engine->BFF data flow.

Liveness (``/livez``) answers "is the BFF process up" — trivially yes if it can
respond. Readiness (``/readyz``) answers "is fresh data actually flowing" — it
gates on the trade-ideas snapshot being present and within the staleness
threshold, because that snapshot is the BFF's primary read and the thing the
engine bridge must keep fresh. The freshness vector (snapshot age, last-bar age,
model-loaded) is also exported as Prometheus gauges for alerting.

Every probe degrades to ``None``/``False`` and never raises — a health check
must never 500.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..metrics import (
    bff_last_bar_age_seconds,
    bff_model_loaded,
    bff_snapshot_age_seconds,
)

log = logging.getLogger(__name__)

# Default staleness threshold (mirrors the snapshot cache + tokens.json config).
_STALE_THRESHOLD_SECONDS = 90.0


class HealthService:
    def __init__(
        self,
        *,
        ideas: Any = None,
        gateway: Any = None,
        model: Any = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        stale_threshold_seconds: float = _STALE_THRESHOLD_SECONDS,
    ) -> None:
        self._ideas = ideas
        self._gateway = gateway
        self._model = model
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._threshold = float(stale_threshold_seconds)

    def freshness(self) -> dict[str, Any]:
        """The freshness vector + update the Prometheus gauges (never raises)."""

        snapshot_age = _safe(lambda: self._ideas.snapshot_age_seconds()) if self._ideas else None
        last_bar_age = _safe(lambda: self._gateway.latest_bar_age_seconds()) if self._gateway else None
        model_loaded = False
        if self._model is not None:
            try:
                response, _ = self._model.get_model()
                model_loaded = getattr(response, "version", None) is not None
            except Exception as exc:  # noqa: BLE001
                log.warning("health: model probe failed: %s", exc)

        # Export gauges (-1 encodes "unavailable" so a dead feed flat-lines).
        bff_snapshot_age_seconds.set(snapshot_age if snapshot_age is not None else -1.0)
        bff_last_bar_age_seconds.set(last_bar_age if last_bar_age is not None else -1.0)
        bff_model_loaded.set(1.0 if model_loaded else 0.0)

        snapshot_present = snapshot_age is not None
        return {
            "now": self._now().isoformat(),
            "snapshot_present": snapshot_present,
            "snapshot_age_seconds": snapshot_age,
            "snapshot_stale": snapshot_present and snapshot_age > self._threshold,
            "stale_threshold_seconds": self._threshold,
            "last_bar_age_seconds": last_bar_age,
            "model_loaded": model_loaded,
        }

    def ready(self) -> tuple[bool, dict[str, Any]]:
        """Readiness: a fresh trade-ideas snapshot is present (data is flowing).

        Returns ``(ok, freshness)``. Not ready ⇒ the engine bridge isn't
        delivering (publisher down, or stale past threshold) — the orchestrator
        should hold traffic / alert, but the BFF itself stays alive.
        """

        f = self.freshness()
        ok = bool(f["snapshot_present"]) and not bool(f["snapshot_stale"])
        return ok, f


def _safe(fn: Callable[[], Optional[float]]) -> Optional[float]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — a probe must never break health
        log.warning("health probe failed: %s", exc)
        return None


__all__ = ["HealthService"]
