"""Graduated capital deployment controller (P6.06).

Live trading ramps through phases: a small pilot with reduced size, widening
as the system earns its keep. This module owns the deployment plan, phase
promotion logic, halt state, and paper-vs-live divergence detection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Phase definition ──────────────────────────────────────────────────────

@dataclass
class DeploymentPhase:
    phase_id: int
    name: str
    target_capital: float
    position_size_multiplier: float
    min_duration_days: int
    promotion_criteria: dict[str, Any] = field(default_factory=dict)
    entry_date: datetime | None = None
    status: str = "pending"  # "pending" | "active" | "completed" | "halted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "target_capital": self.target_capital,
            "position_size_multiplier": self.position_size_multiplier,
            "min_duration_days": self.min_duration_days,
            "promotion_criteria": dict(self.promotion_criteria),
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "status": self.status,
        }


def default_phases() -> list[DeploymentPhase]:
    """The canonical four-phase ramp."""
    base_criteria = {"min_sharpe": 1.0, "max_drawdown": 0.10}
    return [
        DeploymentPhase(1, "pilot", 5_000.0, 0.25, 14, dict(base_criteria)),
        DeploymentPhase(2, "beta", 15_000.0, 0.50, 28, dict(base_criteria)),
        DeploymentPhase(3, "scale", 50_000.0, 0.75, 42, dict(base_criteria)),
        DeploymentPhase(4, "full", math.inf, 1.0, 0, {}),
    ]


# ── Controller ────────────────────────────────────────────────────────────

class CapitalDeploymentController:
    """Owns the current deployment phase, promotion, and halt state."""

    def __init__(
        self,
        *,
        portfolio: Any,
        metrics: Any | None = None,
        phases: list[DeploymentPhase] | None = None,
        asset_class: str = "equities",
        alert_manager: Any | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.metrics = metrics
        self.asset_class = asset_class
        self.alert_manager = alert_manager
        self.phases = phases if phases is not None else default_phases()
        if not self.phases:
            raise ValueError("CapitalDeploymentController requires at least one phase")
        self._current_index = 0
        self.phases[0].status = "active"
        self.phases[0].entry_date = _utcnow()
        self._halted: bool = False
        self._halt_reason: str = ""

    # ── Queries ───────────────────────────────────────────────────────

    def get_current_phase(self) -> DeploymentPhase:
        return self.phases[self._current_index]

    def get_size_multiplier(self) -> float:
        if self._halted:
            return 0.0
        return self.get_current_phase().position_size_multiplier

    def get_deployment_status(self) -> dict[str, Any]:
        return {
            "asset_class": self.asset_class,
            "current_phase_id": self.get_current_phase().phase_id,
            "current_phase_name": self.get_current_phase().name,
            "size_multiplier": self.get_size_multiplier(),
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "phases": [p.to_dict() for p in self.phases],
        }

    # ── Promotion ─────────────────────────────────────────────────────

    async def check_promotion(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Evaluate whether the current phase has earned promotion.

        Returns a dict with ``eligible`` and a per-criterion breakdown.
        """
        now = now or _utcnow()
        phase = self.get_current_phase()
        result: dict[str, Any] = {
            "phase_id": phase.phase_id,
            "eligible": False,
            "reasons": [],
            "criteria": {},
        }

        if self._halted:
            result["reasons"].append("halted")
            return result

        if phase.phase_id == self.phases[-1].phase_id:
            result["reasons"].append("already_at_final_phase")
            return result

        # Duration
        entry = phase.entry_date or now
        days_in_phase = (now - entry).days
        duration_ok = days_in_phase >= phase.min_duration_days
        result["criteria"]["duration"] = {
            "days_in_phase": days_in_phase,
            "required": phase.min_duration_days,
            "passed": duration_ok,
        }
        if not duration_ok:
            result["reasons"].append("min_duration_not_met")

        # Performance criteria
        stats = await self._fetch_live_stats()
        for key, threshold in phase.promotion_criteria.items():
            value = stats.get(key)
            passed = self._eval_criterion(key, value, threshold)
            result["criteria"][key] = {
                "value": value, "threshold": threshold, "passed": passed,
            }
            if not passed:
                result["reasons"].append(f"criterion_failed:{key}")

        result["eligible"] = duration_ok and all(
            c.get("passed", False) for c in result["criteria"].values()
        )
        return result

    async def _fetch_live_stats(self) -> dict[str, Any]:
        if self.metrics is None:
            return {}
        # Duck-typed: metrics collector may expose get_performance_stats()
        for name in ("get_performance_stats", "get_live_stats", "get_stats"):
            fn = getattr(self.metrics, name, None)
            if callable(fn):
                try:
                    res = fn()
                    if hasattr(res, "__await__"):
                        res = await res  # type: ignore[misc]
                    return dict(res or {})
                except Exception as exc:
                    log.warning("metrics.%s failed: %s", name, exc)
        return {}

    @staticmethod
    def _eval_criterion(key: str, value: Any, threshold: Any) -> bool:
        if value is None:
            return False
        try:
            v = float(value)
            t = float(threshold)
        except (TypeError, ValueError):
            return False
        # Conventions: "max_*" → value must be <=; everything else >=.
        if key.startswith("max_"):
            return v <= t
        return v >= t

    def promote(self, *, now: datetime | None = None) -> DeploymentPhase:
        """Move to the next phase. Caller is responsible for gating on
        ``check_promotion()``; this method does not re-evaluate criteria."""
        if self._halted:
            raise RuntimeError(f"Cannot promote while halted: {self._halt_reason}")
        if self._current_index >= len(self.phases) - 1:
            raise RuntimeError("Already at final phase")
        now = now or _utcnow()
        self.phases[self._current_index].status = "completed"
        self._current_index += 1
        new_phase = self.phases[self._current_index]
        new_phase.status = "active"
        new_phase.entry_date = now
        log.info(
            "capital deployment promoted to phase %s (%s) @ %s",
            new_phase.phase_id, new_phase.name, now.isoformat(),
        )
        return new_phase

    # ── Halt ──────────────────────────────────────────────────────────

    def halt(self, reason: str) -> None:
        if self._halted:
            return
        self._halted = True
        self._halt_reason = reason
        self.get_current_phase().status = "halted"
        log.error("capital deployment halted: %s", reason)
        mgr = self.alert_manager
        if mgr is not None and hasattr(mgr, "send"):
            try:
                mgr.send(
                    subject="Capital deployment halted",
                    body=f"asset_class={self.asset_class} reason={reason}",
                    severity="critical",
                )
            except Exception:  # pragma: no cover
                log.exception("halt alert failed")

    def resume(self) -> None:
        if not self._halted:
            return
        self._halted = False
        self._halt_reason = ""
        self.get_current_phase().status = "active"

    # ── Divergence detection ──────────────────────────────────────────

    @staticmethod
    def detect_divergence(
        paper_history: Iterable[float],
        live_history: Iterable[float],
        *,
        window_days: int = 7,
        threshold: float = 1.0,
    ) -> dict[str, Any]:
        """Compare paper vs. live Sharpe over the trailing ``window_days``
        and flag when the gap exceeds ``threshold``.

        Inputs are sequences of daily returns (decimal form).
        """
        paper = [float(x) for x in paper_history][-window_days:]
        live = [float(x) for x in live_history][-window_days:]
        paper_sharpe = _daily_sharpe(paper)
        live_sharpe = _daily_sharpe(live)
        gap = abs(paper_sharpe - live_sharpe)
        return {
            "paper_sharpe": paper_sharpe,
            "live_sharpe": live_sharpe,
            "gap": gap,
            "threshold": threshold,
            "diverged": gap > threshold,
            "window_days": window_days,
        }


# ── helpers ───────────────────────────────────────────────────────────────

def _daily_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std <= 0:
        return 0.0
    # Annualized, 252 trading days.
    return (mean / std) * math.sqrt(252)
