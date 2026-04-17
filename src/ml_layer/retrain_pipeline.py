"""End-to-end retrain pipeline (C2 / design doc §12.3).

Wires the retraining loop from raw bars → features → signals → labels →
train → compare-vs-incumbent → strategy gate (CPCV / DSR / PBO) → registry
promotion. Wraps ``RetrainScheduler`` and the ``StrategyGate`` so operators
can invoke a full retrain with one call.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import pandas as pd

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── RetrainRun record ─────────────────────────────────────────────────────

@dataclass
class RetrainRun:
    run_id: str
    started_at: datetime
    symbol: str
    trigger: str  # "scheduled" | "manual" | "drift" | "emergency"
    completed_at: datetime | None = None
    training_rows: int = 0
    cv_score: float | None = None
    gate_results: dict[str, Any] = field(default_factory=dict)
    promoted: bool = False
    incumbent_version: str | None = None
    new_version: str | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "symbol": self.symbol,
            "trigger": self.trigger,
            "training_rows": self.training_rows,
            "cv_score": self.cv_score,
            "gate_results": self.gate_results,
            "promoted": self.promoted,
            "incumbent_version": self.incumbent_version,
            "new_version": self.new_version,
            "rejection_reason": self.rejection_reason,
        }


# ── Pipeline ──────────────────────────────────────────────────────────────

TrainerFn = Callable[[pd.DataFrame, pd.Series, pd.Series], tuple[Any, float]]
ScoreFn = Callable[[Any, pd.DataFrame, pd.Series], float]


class RetrainPipeline:
    """Full retrain orchestration. Everything is duck-typed so tests can
    inject minimal mocks; production binds real factories."""

    def __init__(
        self,
        *,
        meta_labeling_pipeline: Any,
        feature_assembler: Any,
        signal_battery: Any,
        gate: Any,
        registry: Any,
        cost_model: Any | None = None,
        alert_manager: Any | None = None,
        trainer: TrainerFn | None = None,
        scorer: ScoreFn | None = None,
        min_improvement_pct: float = 0.05,
    ) -> None:
        self.meta_labeling_pipeline = meta_labeling_pipeline
        self.feature_assembler = feature_assembler
        self.signal_battery = signal_battery
        self.gate = gate
        self.registry = registry
        self.cost_model = cost_model
        self.alert_manager = alert_manager
        self.trainer = trainer
        self.scorer = scorer
        self.min_improvement_pct = float(min_improvement_pct)

    # ── Public API ────────────────────────────────────────────────────

    async def run(
        self,
        symbol: str,
        close: pd.Series,
        bars: pd.DataFrame,
        trigger: str = "scheduled",
    ) -> RetrainRun:
        run = RetrainRun(
            run_id=str(uuid.uuid4()),
            started_at=_utcnow(),
            symbol=symbol,
            trigger=trigger,
        )
        try:
            await self._execute(run, close, bars, bypass_threshold=(trigger == "emergency"))
        except Exception as exc:
            log.exception("retrain pipeline failed for %s", symbol)
            run.rejection_reason = f"exception: {exc}"
            run.gate_results["trace"] = traceback.format_exc()
        finally:
            run.completed_at = _utcnow()
            await self._alert_outcome(run)
        return run

    async def emergency_retrain(self, symbol: str, reason: str,
                                 close: pd.Series, bars: pd.DataFrame) -> RetrainRun:
        log.warning("emergency retrain for %s: %s", symbol, reason)
        run = await self.run(symbol, close, bars, trigger="emergency")
        run.gate_results.setdefault("emergency_reason", reason)
        return run

    async def run_all_symbols(
        self,
        universe: Sequence[str],
        data_loader: Callable[[str], tuple[pd.Series, pd.DataFrame]],
    ) -> list[RetrainRun]:
        runs: list[RetrainRun] = []
        for symbol in universe:
            try:
                close, bars = data_loader(symbol)
            except Exception as exc:
                log.warning("data_loader failed for %s: %s", symbol, exc)
                run = RetrainRun(
                    run_id=str(uuid.uuid4()), started_at=_utcnow(),
                    symbol=symbol, trigger="scheduled",
                    completed_at=_utcnow(),
                    rejection_reason=f"data_loader_failed: {exc}",
                )
                runs.append(run)
                continue
            runs.append(await self.run(symbol, close, bars))
        return runs

    # ── Core steps ────────────────────────────────────────────────────

    async def _execute(
        self, run: RetrainRun, close: pd.Series, bars: pd.DataFrame,
        *, bypass_threshold: bool,
    ) -> None:
        # 1-3. Features → signals → labels
        features = await _maybe_await(self.feature_assembler.compute(bars))
        signals = await _maybe_await(self.signal_battery.generate(features))
        X, y, sw = self.meta_labeling_pipeline.prepare_training_data(
            close, signals, features,
        )
        run.training_rows = len(X)

        if run.training_rows == 0:
            run.rejection_reason = "no_training_rows"
            return

        # 4. Load incumbent
        incumbent = self._get_incumbent()
        run.incumbent_version = _version_of(incumbent)

        # 5. Train new
        if self.trainer is None:
            run.rejection_reason = "no_trainer_configured"
            return
        new_model, cv_score = self.trainer(X, y, sw)
        run.cv_score = float(cv_score)

        # 6. Compare to incumbent
        if not bypass_threshold and incumbent is not None and self.scorer is not None:
            incumbent_score = self.scorer(self._incumbent_model(incumbent), X, y)
            if incumbent_score and incumbent_score > 0:
                improvement = (run.cv_score - incumbent_score) / abs(incumbent_score)
            else:
                improvement = float("inf") if run.cv_score > 0 else 0.0
            run.gate_results["incumbent_score"] = float(incumbent_score)
            run.gate_results["improvement_pct"] = float(improvement)
            if improvement < self.min_improvement_pct:
                run.rejection_reason = (
                    f"insufficient_improvement: {improvement:.4f} "
                    f"< {self.min_improvement_pct}"
                )
                return

        # 7. Strategy gate
        gate_result = await self._run_gate(new_model, X, y)
        run.gate_results.update(gate_result)
        if not gate_result.get("passed", False):
            run.rejection_reason = (
                f"gate_failed: {gate_result.get('failing_gates', [])}"
            )
            return

        # 8-10. Register + promote
        new_run_id = self._register_and_promote(new_model, run.cv_score, gate_result)
        run.new_version = new_run_id
        run.promoted = True

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_incumbent(self) -> Any | None:
        if self.registry is None:
            return None
        for name in ("get_production_model", "get_best_model"):
            fn = getattr(self.registry, name, None)
            if callable(fn):
                try:
                    return fn()
                except TypeError:
                    try:
                        return fn(stage="production")
                    except Exception:
                        continue
                except Exception:
                    continue
        return None

    @staticmethod
    def _incumbent_model(incumbent: Any) -> Any:
        if isinstance(incumbent, dict):
            return incumbent.get("model") or incumbent
        return getattr(incumbent, "model", incumbent)

    async def _run_gate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        for name in ("validate", "quick_validate"):
            fn = getattr(self.gate, name, None)
            if not callable(fn):
                continue
            try:
                result = fn(model=model, X=X, y=y, cost_model=self.cost_model)
                if asyncio.iscoroutine(result):
                    result = await result
            except TypeError:
                try:
                    result = fn(model, X, y)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as exc:
                    log.warning("gate %s failed: %s", name, exc)
                    continue
            except Exception as exc:
                log.warning("gate %s failed: %s", name, exc)
                continue
            return _coerce_gate_result(result)
        return {"passed": False, "failing_gates": ["gate_unavailable"]}

    def _register_and_promote(
        self, model: Any, cv_score: float, gate_result: dict[str, Any],
    ) -> str | None:
        reg = self.registry
        if reg is None:
            return None
        run_id = None
        log_fn = getattr(reg, "log_training_run", None)
        if callable(log_fn):
            try:
                run_id = log_fn(
                    model=model, params={}, metrics={
                        "cv_score": cv_score,
                        **{f"gate_{k}": v for k, v in gate_result.items()
                           if isinstance(v, (int, float, bool))},
                    },
                )
            except Exception as exc:
                log.warning("registry.log_training_run failed: %s", exc)
        promote = getattr(reg, "promote_model", None)
        if callable(promote) and run_id is not None:
            try:
                promote(run_id)
            except Exception as exc:
                log.warning("registry.promote_model failed: %s", exc)
        return run_id

    async def _alert_outcome(self, run: RetrainRun) -> None:
        if self.alert_manager is None:
            return
        title = (
            f"Retrain promoted ({run.symbol})" if run.promoted
            else f"Retrain rejected ({run.symbol})"
        )
        body = (
            f"trigger={run.trigger} rows={run.training_rows} "
            f"cv={run.cv_score} reason={run.rejection_reason}"
        )
        for name in ("send_alert", "send"):
            fn = getattr(self.alert_manager, name, None)
            if not callable(fn):
                continue
            try:
                from src.monitoring.alerting import Alert, AlertSeverity
                sev = AlertSeverity.INFO if run.promoted else AlertSeverity.WARNING
                alert = Alert(severity=sev, title=title, message=body,
                              source="retrain_pipeline",
                              metadata={"run": run.to_dict()})
                res = fn(alert)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # pragma: no cover
                log.exception("alert failed")
            break


# ── Module-level helpers ─────────────────────────────────────────────────

async def _maybe_await(x: Any) -> Any:
    if asyncio.iscoroutine(x):
        return await x
    return x


def _version_of(incumbent: Any) -> str | None:
    if incumbent is None:
        return None
    if isinstance(incumbent, dict):
        return str(incumbent.get("run_id") or incumbent.get("version") or "")
    return str(getattr(incumbent, "run_id", None) or getattr(incumbent, "version", None) or "")


def _coerce_gate_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        out = dict(result)
    else:
        out = {
            "passed": bool(getattr(result, "passed", False)),
            "failing_gates": list(getattr(result, "failing_gates", []) or []),
            "cpcv": getattr(result, "cpcv", None),
            "dsr": getattr(result, "dsr", None),
            "pbo": getattr(result, "pbo", None),
        }
    # Normalize: ensure "passed" present.
    if "passed" not in out:
        failing = out.get("failing_gates") or []
        out["passed"] = not failing
    return out
