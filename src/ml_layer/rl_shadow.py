"""Shadow-mode comparison engine for the RL portfolio optimizer (P6.10).

The engine logs every decision made by HRP (production) and the RL shadow
agent side by side, simulates P&L for both, and reports whether the RL policy
has earned promotion. The ``RLPromotionController`` is the lever operators
pull to swap optimizers or roll back.
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


# ── Decision record ───────────────────────────────────────────────────────

@dataclass
class ShadowDecisionRecord:
    timestamp: datetime
    hrp_target: dict[str, float]
    rl_target: dict[str, float]
    executed_target: dict[str, float]
    market_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.timestamp.isoformat(),
            "hrp_target": self.hrp_target,
            "rl_target": self.rl_target,
            "executed_target": self.executed_target,
            "market_state": self.market_state,
        }


# ── Engine ────────────────────────────────────────────────────────────────

class ShadowComparisonEngine:
    """Accumulates paired HRP/RL decisions and compares realised P&L."""

    def __init__(
        self,
        *,
        execution_storage: Any | None = None,
        alert_manager: Any | None = None,
    ) -> None:
        self.execution_storage = execution_storage
        self.alert_manager = alert_manager
        self.records: list[ShadowDecisionRecord] = []

    # ── Recording ─────────────────────────────────────────────────────

    def record_decision(
        self,
        timestamp: datetime | str,
        hrp_target: dict[str, float],
        rl_target: dict[str, float],
        executed_target: dict[str, float],
        market_state: dict[str, Any] | None = None,
    ) -> ShadowDecisionRecord:
        ts = (
            datetime.fromisoformat(timestamp)
            if isinstance(timestamp, str) else timestamp
        )
        record = ShadowDecisionRecord(
            timestamp=ts,
            hrp_target=dict(hrp_target),
            rl_target=dict(rl_target),
            executed_target=dict(executed_target),
            market_state=dict(market_state or {}),
        )
        self.records.append(record)
        if self.execution_storage is not None and hasattr(self.execution_storage, "save_shadow_decision"):
            try:
                self.execution_storage.save_shadow_decision(record.to_dict())
            except Exception:  # pragma: no cover
                log.exception("execution_storage.save_shadow_decision failed")
        return record

    # ── Comparison ────────────────────────────────────────────────────

    def _window(self, start: datetime, end: datetime) -> list[ShadowDecisionRecord]:
        return [r for r in self.records if start <= r.timestamp <= end]

    def compute_comparison(
        self,
        start: datetime,
        end: datetime,
        *,
        returns_provider: Any | None = None,
    ) -> dict[str, Any]:
        """Simulate per-period P&L for both policies over ``[start, end]``.

        Each ``record.market_state`` must carry ``symbol_returns``: a
        ``{symbol: return_over_next_period}`` mapping. Optionally, a
        ``returns_provider(record)`` callable can be passed to compute those
        returns on the fly.
        """
        window = self._window(start, end)
        hrp_returns: list[float] = []
        rl_returns: list[float] = []

        for rec in window:
            sym_rets: dict[str, float] = (
                (returns_provider(rec) if returns_provider else rec.market_state.get("symbol_returns")) or {}
            )
            hrp_returns.append(_weighted_return(rec.hrp_target, sym_rets))
            rl_returns.append(_weighted_return(rec.rl_target, sym_rets))

        hrp_metrics = _return_metrics(hrp_returns)
        rl_metrics = _return_metrics(rl_returns)
        t_stat, p_value = _paired_t_test(hrp_returns, rl_returns)
        rl_is_better = rl_metrics["sharpe"] > hrp_metrics["sharpe"]
        significance = (
            "significant" if p_value < 0.05
            else ("trending" if p_value < 0.20 else "insignificant")
        )
        return {
            "n_observations": len(window),
            "hrp_metrics": hrp_metrics,
            "rl_metrics": rl_metrics,
            "paired_t_stat": t_stat,
            "p_value": p_value,
            "rl_is_better": bool(rl_is_better),
            "significance": significance,
        }

    # ── Reporting ─────────────────────────────────────────────────────

    def generate_shadow_report(self, *, lookback_days: int = 30,
                               now: datetime | None = None) -> str:
        now = now or _utcnow()
        start = now - timedelta(days=lookback_days)
        comp = self.compute_comparison(start, now)
        lines = [
            f"Shadow comparison report · last {lookback_days}d · "
            f"n_obs={comp['n_observations']}",
            f"HRP Sharpe = {comp['hrp_metrics']['sharpe']:+.3f}  "
            f"max_dd = {comp['hrp_metrics']['max_drawdown']:.3f}",
            f"RL  Sharpe = {comp['rl_metrics']['sharpe']:+.3f}  "
            f"max_dd = {comp['rl_metrics']['max_drawdown']:.3f}",
            f"paired t-stat = {comp['paired_t_stat']:+.3f}  "
            f"p = {comp['p_value']:.4f}  ({comp['significance']})",
            f"RL is better: {comp['rl_is_better']}",
        ]
        return "\n".join(lines)

    # ── Promotion eligibility ─────────────────────────────────────────

    def check_promotion_eligibility(
        self,
        *,
        now: datetime | None = None,
        required_months: int = 6,
        min_sharpe_gap: float = 0.3,
        max_drawdown_delta: float = 0.02,
        max_p_value: float = 0.05,
        gates: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        now = now or _utcnow()
        reasons: list[str] = []
        if not self.records:
            return {
                "eligible": False,
                "reasons": ["no_shadow_history"],
                "comparison": None,
            }

        start_record = min(self.records, key=lambda r: r.timestamp)
        months_span = (now - start_record.timestamp).days / 30.4375
        if months_span < required_months:
            reasons.append(
                f"insufficient_history:{months_span:.1f}mo<{required_months}"
            )

        comp = self.compute_comparison(start_record.timestamp, now)
        hrp, rl = comp["hrp_metrics"], comp["rl_metrics"]
        if (rl["sharpe"] - hrp["sharpe"]) < min_sharpe_gap:
            reasons.append("sharpe_gap_insufficient")
        if (rl["max_drawdown"] - hrp["max_drawdown"]) > max_drawdown_delta:
            reasons.append("drawdown_regression")
        if comp["p_value"] >= max_p_value:
            reasons.append("t_test_not_significant")
        if gates:
            for g in ("cpcv", "dsr", "pbo"):
                if not gates.get(g, False):
                    reasons.append(f"gate_failed:{g}")
        else:
            reasons.append("gates_not_supplied")

        return {
            "eligible": not reasons,
            "reasons": reasons,
            "comparison": comp,
            "months_span": months_span,
        }


# ── Metric helpers ────────────────────────────────────────────────────────

def _weighted_return(weights: dict[str, float], returns: dict[str, float]) -> float:
    total = 0.0
    for sym, w in weights.items():
        total += float(w) * float(returns.get(sym, 0.0))
    return total


def _return_metrics(returns: Iterable[float]) -> dict[str, float]:
    r = [float(x) for x in returns]
    if not r:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "n_trades": 0}
    n = len(r)
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r) / max(n - 1, 1)
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0
    # Drawdown over cumulative P&L curve
    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for x in r:
        cum *= 1.0 + x
        peak = max(peak, cum)
        dd = (peak - cum) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    n_trades = sum(1 for x in r if x != 0.0)
    return {
        "total_return": float(cum - 1.0),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_trades": int(n_trades),
    }


def _paired_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test on ``b - a``. Returns ``(t_stat, p_value)`` with a
    scipy-backed p-value when available; falls back to a normal approximation."""
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    if std <= 0:
        return 0.0, 1.0
    t_stat = mean / (std / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore
        p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2.0)
    except Exception:  # pragma: no cover
        # Two-sided normal approximation — fine for n ≥ ~30.
        p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return float(t_stat), float(p_value)


# ── Promotion controller ──────────────────────────────────────────────────

class RLPromotionController:
    """Governs the HRP → RL → HRP switch."""

    def __init__(
        self,
        *,
        shadow_engine: ShadowComparisonEngine,
        optimizer_holder: Any,
        hrp_optimizer: Any,
        rl_optimizer: Any,
        alert_manager: Any | None = None,
        mlflow_logger: Any | None = None,
    ) -> None:
        """``optimizer_holder`` is any object with a settable ``optimizer``
        attribute — e.g. the pipeline or order manager wrapper. The controller
        flips that attribute and emits an alert."""
        self.shadow_engine = shadow_engine
        self.optimizer_holder = optimizer_holder
        self.hrp_optimizer = hrp_optimizer
        self.rl_optimizer = rl_optimizer
        self.alert_manager = alert_manager
        self.mlflow_logger = mlflow_logger
        self.current: str = "hrp"

    async def promote_rl_to_production(
        self, *, gates: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        eligibility = self.shadow_engine.check_promotion_eligibility(gates=gates)
        if not eligibility["eligible"]:
            raise RuntimeError(
                f"RL promotion blocked: {eligibility['reasons']}"
            )
        if self.mlflow_logger is not None and hasattr(self.mlflow_logger, "log_run"):
            try:
                self.mlflow_logger.log_run({
                    "event": "rl_promotion",
                    "comparison": eligibility["comparison"],
                })
            except Exception:  # pragma: no cover
                log.exception("mlflow log_run failed")
        self.optimizer_holder.optimizer = self.rl_optimizer
        self.current = "rl"
        await self._alert(
            "RL optimizer promoted to production",
            f"rl_sharpe={eligibility['comparison']['rl_metrics']['sharpe']:+.3f}",
            severity="critical",
        )
        return {"current": self.current, "eligibility": eligibility}

    async def revert_to_hrp(
        self, reason: str, *, flatten: bool = False,
    ) -> dict[str, Any]:
        self.optimizer_holder.optimizer = self.hrp_optimizer
        prior = self.current
        self.current = "hrp"
        await self._alert(
            "RL optimizer reverted to HRP",
            f"reason={reason}",
            severity="critical",
        )
        if flatten:
            flatten_fn = getattr(self.optimizer_holder, "flatten_all", None)
            if callable(flatten_fn):
                try:
                    res = flatten_fn()
                    if hasattr(res, "__await__"):
                        await res  # type: ignore[misc]
                except Exception:  # pragma: no cover
                    log.exception("flatten_all failed")
        return {"prior": prior, "current": self.current, "reason": reason}

    async def _alert(self, title: str, message: str, *, severity: str) -> None:
        if self.alert_manager is None:
            return
        send = getattr(self.alert_manager, "send", None)
        if callable(send):
            try:
                res = send(subject=title, body=message, severity=severity)
                if hasattr(res, "__await__"):
                    await res  # type: ignore[misc]
            except Exception:  # pragma: no cover
                log.exception("alert send failed")
