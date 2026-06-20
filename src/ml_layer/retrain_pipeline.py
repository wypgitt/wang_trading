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

import numpy as np
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
        features = await _compute_features(self.feature_assembler, bars)
        signals = await _generate_signals(self.signal_battery, bars, features, run.symbol)
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
        gate_result = await self._run_gate(
            new_model, X, y,
            close=close, features=features, signals=signals, symbol=run.symbol,
        )
        run.gate_results.update(gate_result)
        if not gate_result.get("passed", False):
            failing = gate_result.get("failing_gates") or _failing_gate_names(gate_result)
            run.rejection_reason = f"gate_failed: {failing}"
            return

        # 8-10. Register + promote
        new_run_id = self._register_and_promote(
            new_model,
            run.cv_score,
            gate_result,
            X,
            y,
            sw,
        )
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

    async def _run_gate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        close: pd.Series | None = None,
        features: pd.DataFrame | None = None,
        signals: pd.DataFrame | None = None,
        symbol: str = "",
    ) -> dict[str, Any]:
        """Run the CPCV / DSR / PBO promotion gate against the candidate.

        The gate (:class:`StrategyGate`) owns the §9 statistics; this method's
        job is to thread it the data it needs — a wide ``close``/``signals``
        panel, per-bar ``bet_sizes``, and the raw ``features`` from which the
        gate builds its PBO variant grid — and surface the *real* verdict.

        Two deliberate behaviours, both fixing the historical
        ``gate_unavailable`` bug where every retrain was silently rejected:

        * The gate is invoked through its actual ``evaluate_candidate``
          signature (not a guessed ``(model, X, y)`` call that always raised).
        * A gate that *errors while computing* is NOT swallowed — the
          exception propagates so the run fails loudly with a traceback,
          rather than degrading every candidate to a uniform reject.
        """
        fn = getattr(self.gate, "evaluate_candidate", None)
        if not callable(fn):
            # No usable gate wired. This is a configuration error, not a model
            # verdict — surface it loudly and flag it distinctly so it can
            # never be mistaken for a real CPCV/DSR/PBO failure.
            log.error(
                "gate %s exposes no evaluate_candidate(); cannot validate %s",
                type(self.gate).__name__, symbol,
            )
            return {"passed": False, "failing_gates": ["gate_not_configured"]}

        inputs = self._build_gate_inputs(close, features, signals, symbol)
        result = fn(**inputs)
        if asyncio.iscoroutine(result):
            result = await result
        return _coerce_gate_result(result)

    def _build_gate_inputs(
        self,
        close: pd.Series | None,
        features: pd.DataFrame | None,
        signals: pd.DataFrame | None,
        symbol: str,
    ) -> dict[str, Any]:
        """Assemble the keyword arguments ``StrategyGate.evaluate_candidate``
        expects from the per-symbol data the pipeline computed."""
        from src.backtesting.gate_orchestrator import (  # lazy: avoid import cycle
            _bets_to_wide,
            _signals_to_wide,
        )

        close_w = close.to_frame(symbol) if isinstance(close, pd.Series) else close
        if close_w is None or getattr(close_w, "empty", True):
            raise ValueError(f"gate requires a non-empty close panel for {symbol!r}")

        index = close_w.index
        flat = _flat_signals_or_none(signals)
        signals_w = _signals_to_wide(flat, index, symbol)
        bets_w = _bets_to_wide(flat, index, symbol)

        horizon = _coerce_horizon(
            getattr(self.meta_labeling_pipeline, "max_holding_period", None)
        )

        return {
            "close": close_w,
            "signals": signals_w,
            "bet_sizes": bets_w,
            "features": features,
            "cost_model": self.cost_model,
            "cpcv_horizon": horizon,
        }

    def _register_and_promote(
        self,
        model: Any,
        cv_score: float,
        gate_result: dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series,
    ) -> str | None:
        reg = self.registry
        if reg is None:
            return None
        run_id = None
        log_fn = getattr(reg, "log_training_run", None)
        if callable(log_fn):
            try:
                gates = _gate_flags(gate_result)
                run_id = log_fn(
                    model=model,
                    X=X,
                    y=y,
                    labels_df=pd.DataFrame({"label": y}, index=y.index),
                    params={
                        "source": "retrain_pipeline",
                        "gates": gates,
                    },
                    cv_scores=np.array([cv_score], dtype=float),
                    sample_weight=sample_weight,
                    metrics={
                        "cv_score": cv_score,
                        **{f"gate_{k}": float(v) for k, v in gates.items()},
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


async def _compute_features(feature_assembler: Any, bars: pd.DataFrame) -> pd.DataFrame:
    for name in ("compute", "assemble"):
        fn = getattr(feature_assembler, name, None)
        if callable(fn):
            return await _maybe_await(fn(bars))
    raise AttributeError("feature_assembler must expose compute() or assemble()")


async def _generate_signals(
    signal_battery: Any,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    for name, args in (
        ("generate", (bars,)),
        ("generate_all", (bars,)),
        ("generate", (features,)),
        ("generate_all", (features,)),
    ):
        fn = getattr(signal_battery, name, None)
        if not callable(fn):
            continue
        try:
            return await _maybe_await(fn(*args, symbol=symbol))
        except TypeError:
            try:
                return await _maybe_await(fn(*args))
            except TypeError:
                continue
    raise AttributeError("signal_battery must expose generate() or generate_all()")


def _coerce_horizon(value: Any, default: int = 10) -> int:
    """Best-effort positive-int label horizon for CPCV purging; ``default``
    when the pipeline's meta-labeler doesn't expose a usable
    ``max_holding_period`` (e.g. a duck-typed mock)."""
    try:
        horizon = int(value)
    except (TypeError, ValueError):
        return default
    return horizon if horizon >= 1 else default


def _failing_gate_names(gate_result: dict[str, Any]) -> list[str]:
    """Names of the §9 gates that did not pass (real ``False`` *or* not-run
    ``None``), so a rejected retrain says *which* gate blocked it instead of an
    empty list. Falls back to the cpcv/dsr/pbo aliases when present."""
    names: list[str] = []
    for full, short in (
        ("gate_1_cpcv", "cpcv"),
        ("gate_2_dsr", "dsr"),
        ("gate_3_pbo", "pbo"),
    ):
        if full in gate_result:
            value = gate_result[full]
            passed = value.get("passed") if isinstance(value, dict) else value
        elif short in gate_result:
            passed = gate_result[short]
        else:
            continue
        if passed is not True:
            names.append(full)
    return names


def _flat_signals_or_none(signals: Any) -> pd.DataFrame | None:
    """Return ``signals`` only if it is a flat signal-battery frame the wide
    helpers can pivot (``timestamp``/``symbol``/``side`` columns); otherwise
    ``None`` so the gate sees an explicit no-signal panel rather than crashing
    on a malformed frame. This normalises inputs — it does NOT swallow gate
    computation errors, which are surfaced loudly."""
    required = {"timestamp", "symbol", "side"}
    if not isinstance(signals, pd.DataFrame) or signals.empty:
        return None
    if required - set(signals.columns):
        return None
    return signals


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


def _gate_flags(gate_result: dict[str, Any]) -> dict[str, bool]:
    """Normalize gate orchestrator output to preflight's cpcv/dsr/pbo flags."""

    def _passed(value: Any) -> bool:
        if isinstance(value, dict):
            return bool(value.get("passed", False))
        return bool(value)

    return {
        "cpcv": _passed(gate_result.get("gate_1_cpcv", gate_result.get("cpcv"))),
        "dsr": _passed(gate_result.get("gate_2_dsr", gate_result.get("dsr"))),
        "pbo": _passed(gate_result.get("gate_3_pbo", gate_result.get("pbo"))),
    }
