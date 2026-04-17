"""Live trading pipeline (P6.07).

Thin subclass of :class:`PaperTradingPipeline` that layers on the safeguards
specific to real capital: a preflight gate, a HALT file, a deployment-phase
size multiplier, daily paper/live divergence checks, weekly promotion
evaluation, and a forced-flatten emergency shutdown path.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.execution.paper_trading import PaperTradingPipeline, _shutdown_alert
from src.monitoring.alerting import Alert, AlertSeverity

log = logging.getLogger(__name__)


HALT_FILE = Path(".live_halt")
COMPLIANCE_LOG_PATH = Path("logs/live_trading_compliance.log")
OPERATOR_CHECKIN_PATH = Path(".operator_checkin")


# ── Helpers ───────────────────────────────────────────────────────────────

class _ShadowAwareOptimizer:
    """Wraps the production HRP optimizer so that every ``compute_target_portfolio``
    call also records the RL shadow decision and — when RL is promoted —
    returns the RL target instead of HRP's."""

    def __init__(self, pipeline: "LiveTradingPipeline", inner: Any) -> None:
        self._pipeline = pipeline
        self._inner = inner

    def compute_target_portfolio(self, **kwargs) -> pd.DataFrame:
        hrp_target = self._inner.compute_target_portfolio(**kwargs)

        p = self._pipeline
        rl_target: pd.DataFrame | None = None
        if p.shadow_rl_agent is not None:
            try:
                rl_target = p.shadow_rl_agent.get_shadow_target_portfolio(kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("shadow RL predict failed: %s", exc)

        active = rl_target if (p.rl_is_production and rl_target is not None) else hrp_target

        if p.shadow_comparison is not None and rl_target is not None:
            try:
                p.shadow_comparison.record_decision(
                    timestamp=datetime.now(timezone.utc),
                    hrp_target=_weights(hrp_target),
                    rl_target=_weights(rl_target),
                    executed_target=_weights(active),
                    market_state={"nav": kwargs.get("nav")},
                )
            except Exception:  # pragma: no cover
                log.exception("shadow_comparison.record_decision failed")
        return active


def _weights(frame: Any) -> dict[str, float]:
    if frame is None or not hasattr(frame, "iterrows"):
        return {}
    if "symbol" not in frame.columns or "target_weight" not in frame.columns:
        return {}
    return dict(zip(frame["symbol"], frame["target_weight"].astype(float)))


class _ScaledBetSizing:
    """Wraps a bet-sizing component so the deployment multiplier is applied
    to the ``final_size`` column on every compute call."""

    def __init__(self, inner: Any, controller: Any) -> None:
        self._inner = inner
        self._controller = controller

    def compute(self, meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        out = self._inner.compute(meta, features) if self._inner is not None else None
        if out is None or not hasattr(out, "copy"):
            return out
        mult = float(self._controller.get_size_multiplier())
        if mult == 1.0 or "final_size" not in out.columns:
            return out
        out = out.copy()
        out["final_size"] = out["final_size"] * mult
        return out


# ── Pipeline ──────────────────────────────────────────────────────────────

class LiveTradingPipeline(PaperTradingPipeline):
    """Production pipeline driving real capital."""

    def __init__(
        self,
        *,
        broker_factory: Any,
        preflight_checker: Any,
        deployment_controller: Any,
        halt_file: Path | str | None = None,
        operator_checkin_path: Path | str | None = None,
        operator_checkin_max_age_h: float = 1.0,
        compliance_log_path: Path | str | None = None,
        shadow_rl_agent: Any | None = None,
        shadow_comparison: Any | None = None,
        rl_is_production: bool = False,
        promotion_controller: Any | None = None,
        rl_revert_dd_threshold: float = 0.05,
        rl_revert_window_days: int = 3,
        recovery_manager: Any | None = None,
        **kwargs,
    ) -> None:
        # Wrap bet_sizing with the deployment multiplier *before* calling
        # super().__init__ so the parent stores the wrapped version.
        inner_sizing = kwargs.get("bet_sizing")
        if inner_sizing is not None:
            kwargs["bet_sizing"] = _ScaledBetSizing(inner_sizing, deployment_controller)
        super().__init__(**kwargs)

        self.broker_factory = broker_factory
        self.preflight_checker = preflight_checker
        self.deployment_controller = deployment_controller
        self.halt_file = Path(halt_file) if halt_file else HALT_FILE
        self.operator_checkin_path = (
            Path(operator_checkin_path) if operator_checkin_path else OPERATOR_CHECKIN_PATH
        )
        self.operator_checkin_max_age_h = float(operator_checkin_max_age_h)
        self.compliance_log_path = (
            Path(compliance_log_path) if compliance_log_path else COMPLIANCE_LOG_PATH
        )
        self._last_divergence_check: datetime | None = None
        self._last_promotion_check: datetime | None = None
        self._last_eligibility_check: datetime | None = None
        self._emergency_flatten_requested = False

        # RL integration
        self.shadow_rl_agent = shadow_rl_agent
        self.shadow_comparison = shadow_comparison
        self.rl_is_production = bool(rl_is_production)
        self.promotion_controller = promotion_controller
        self.rl_revert_dd_threshold = float(rl_revert_dd_threshold)
        self.rl_revert_window_days = int(rl_revert_window_days)
        self._rl_promoted_at: datetime | None = (
            datetime.now(timezone.utc) if rl_is_production else None
        )
        self._rl_nav_at_promotion: float | None = (
            self.order_manager.portfolio.nav if rl_is_production else None
        )

        # Wrap the portfolio optimizer to log shadow decisions + swap on promotion.
        if self.portfolio_optimizer is not None and shadow_rl_agent is not None:
            self.portfolio_optimizer = _ShadowAwareOptimizer(self, self.portfolio_optimizer)

        self.recovery_manager = recovery_manager
        if recovery_manager is not None:
            try:
                recovery_manager.install_crash_handler(self)
            except Exception:  # pragma: no cover
                log.exception("install_crash_handler failed")

    # ── Startup ───────────────────────────────────────────────────────

    async def startup_sequence(self) -> dict[str, Any]:
        # Recovery from a prior run runs *before* the HALT-file refusal so
        # the recovery path is the only way to clear the HALT sentinel cleanly.
        recovery_summary: dict[str, Any] | None = None
        if self.recovery_manager is not None:
            prior = await self.recovery_manager.detect_previous_run()
            if prior and prior.get("snapshot") is not None:
                recovery_summary = await self.recovery_manager.recover(
                    prior["snapshot"], self.order_manager.broker,
                )

        if self.halt_file.exists():
            raise RuntimeError(
                f"Live trading HALT file present at {self.halt_file} — "
                "remove it and re-run preflight before restarting."
            )

        # Preflight
        checks = await self.preflight_checker.run_all_checks()
        summary = self.preflight_checker.summary(checks)
        if not summary["all_passed"]:
            msg = f"preflight failed: {summary['blockers_failed']} blockers"
            await self._send_alert(
                AlertSeverity.CRITICAL, "Preflight failed — live trading blocked", msg,
            )
            raise RuntimeError(msg)

        # Operator check-in
        if not self._operator_recently_checked_in():
            raise RuntimeError(
                f"Operator check-in older than {self.operator_checkin_max_age_h}h "
                f"(expected file: {self.operator_checkin_path})"
            )

        # Starting announcement
        phase = self.deployment_controller.get_current_phase()
        await self._send_alert(
            AlertSeverity.CRITICAL,
            "Live trading starting",
            f"phase={phase.name} multiplier={phase.position_size_multiplier}",
            metadata={"phase": phase.to_dict()},
        )
        self._log_compliance("startup", {
            "phase": phase.to_dict(),
            "preflight": summary,
        })
        self._save_snapshot()
        return {
            "phase": phase.to_dict(),
            "preflight": summary,
            "recovery": recovery_summary,
        }

    def _operator_recently_checked_in(self) -> bool:
        if not self.operator_checkin_path.exists():
            return False
        age_h = (
            datetime.now(timezone.utc).timestamp()
            - self.operator_checkin_path.stat().st_mtime
        ) / 3600
        return age_h < self.operator_checkin_max_age_h

    # ── Cycle ─────────────────────────────────────────────────────────

    async def run_cycle(self, **kwargs) -> dict[str, Any]:
        if self.halt_file.exists() or self.deployment_controller._halted:
            log.warning("Live cycle skipped: halted")
            return {"cycle": self.cycle_count, "halted": True}

        result = await super().run_cycle(**kwargs)
        result["deployment_multiplier"] = self.deployment_controller.get_size_multiplier()

        self._log_compliance("cycle", {
            "cycle": self.cycle_count,
            "nav": result.get("nav"),
            "multiplier": result["deployment_multiplier"],
            "orders": len(result.get("orders", [])),
            "exits": len(result.get("exits", [])),
        })

        now = datetime.now(timezone.utc)
        if self._should_check_divergence(now):
            await self._check_divergence()
            self._last_divergence_check = now
        if self._should_check_promotion(now):
            await self._check_promotion()
            self._last_promotion_check = now
        # RL safeguards
        await self._maybe_auto_revert_rl(now)
        if self._should_check_rl_eligibility(now):
            await self._check_rl_eligibility()
            self._last_eligibility_check = now
        return result

    # ── RL safeguards ─────────────────────────────────────────────────

    async def _maybe_auto_revert_rl(self, now: datetime) -> None:
        if not self.rl_is_production or self._rl_promoted_at is None:
            return
        since_promotion = now - self._rl_promoted_at
        if since_promotion > timedelta(days=self.rl_revert_window_days):
            return
        pf = self.order_manager.portfolio
        base = self._rl_nav_at_promotion or pf.nav
        if base <= 0:
            return
        dd_since_promotion = (base - pf.nav) / base
        if dd_since_promotion <= self.rl_revert_dd_threshold:
            return
        reason = (
            f"auto-revert: {dd_since_promotion:.2%} drawdown within "
            f"{since_promotion.days}d of promotion"
        )
        if self.promotion_controller is not None:
            try:
                await self.promotion_controller.revert_to_hrp(reason)
            except Exception:  # pragma: no cover
                log.exception("auto-revert failed")
        self.rl_is_production = False
        await self._send_alert(
            AlertSeverity.CRITICAL,
            "RL auto-reverted to HRP",
            reason,
            metadata={"drawdown": dd_since_promotion},
        )

    def _should_check_rl_eligibility(self, now: datetime) -> bool:
        if self.promotion_controller is None or self.shadow_comparison is None:
            return False
        if self.rl_is_production:
            return False
        return (
            self._last_eligibility_check is None
            or (now - self._last_eligibility_check) >= timedelta(days=7)
        )

    async def _check_rl_eligibility(self) -> None:
        engine = self.shadow_comparison
        if engine is None:
            return
        try:
            eligibility = engine.check_promotion_eligibility()
        except Exception:  # pragma: no cover
            log.exception("eligibility check failed")
            return
        self._log_compliance("rl_eligibility_check", eligibility)
        if eligibility.get("eligible"):
            await self._send_alert(
                AlertSeverity.WARNING,
                "RL promotion eligible — operator approval required",
                "Run with --approve-rl-promotion to promote.",
                metadata=eligibility,
            )

    def _should_check_divergence(self, now: datetime) -> bool:
        return (
            self._last_divergence_check is None
            or (now - self._last_divergence_check) >= timedelta(days=1)
        )

    def _should_check_promotion(self, now: datetime) -> bool:
        return (
            self._last_promotion_check is None
            or (now - self._last_promotion_check) >= timedelta(days=7)
        )

    async def _check_divergence(self) -> None:
        paper = getattr(self.metrics, "paper_returns", None) or []
        live = getattr(self.metrics, "live_returns", None) or []
        if not paper or not live:
            return
        result = self.deployment_controller.detect_divergence(paper, live)
        if result.get("diverged"):
            self.deployment_controller.halt(
                f"paper/live divergence gap={result['gap']:.2f}"
            )
            await self._send_alert(
                AlertSeverity.CRITICAL,
                "Paper/live divergence detected",
                f"gap={result['gap']:.2f} threshold={result['threshold']:.2f}",
                metadata=result,
            )

    async def _check_promotion(self) -> None:
        result = await self.deployment_controller.check_promotion()
        self._log_compliance("promotion_check", result)
        if result.get("eligible"):
            new_phase = self.deployment_controller.promote()
            await self._send_alert(
                AlertSeverity.INFO,
                "Deployment phase promoted",
                f"now in phase {new_phase.name}",
                metadata={"phase": new_phase.to_dict()},
            )

    # ── Shutdown ──────────────────────────────────────────────────────

    async def shutdown(self, *, emergency: bool = False) -> None:
        self.running = False
        pf = self.order_manager.portfolio

        # Cancel all open orders
        for order in list(pf.open_orders):
            try:
                await self.order_manager.broker.cancel_order(order.order_id)
            except Exception as exc:
                log.warning("cancel failed for %s: %s", order.order_id, exc)

        if emergency or self._emergency_flatten_requested:
            await self._flatten_all(pf)

        self._save_snapshot()

        await self._send_alert(
            AlertSeverity.CRITICAL if emergency else AlertSeverity.INFO,
            "Live trading shutdown" + (" (EMERGENCY)" if emergency else ""),
            f"nav={pf.nav:.2f} cycles={self.cycle_count} "
            f"orders_filled={self._orders_filled_count}",
            metadata={"emergency": emergency},
        )
        self._log_compliance("shutdown", {
            "emergency": emergency,
            "nav": pf.nav,
            "cycles": self.cycle_count,
        })

        # Write HALT file — operator must delete before restart.
        try:
            self.halt_file.parent.mkdir(parents=True, exist_ok=True)
            self.halt_file.write_text(
                f"halted_at={datetime.now(timezone.utc).isoformat()}\n"
                f"emergency={emergency}\n"
            )
        except Exception as exc:  # pragma: no cover
            log.warning("failed to write HALT file: %s", exc)

    async def _flatten_all(self, pf: Any) -> None:
        """Submit market exit orders for every open position."""
        from src.execution.models import Order, OrderType
        import uuid

        for symbol, pos in list(pf.positions.items()):
            order = Order(
                order_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                side=-pos.side,
                order_type=OrderType.MARKET,
                quantity=-pos.side * pos.quantity,
                signal_family=pos.signal_family,
            )
            try:
                await self.order_manager.broker.submit_order(order)
            except Exception as exc:
                log.warning("flatten failed for %s: %s", symbol, exc)

    def request_emergency_flatten(self) -> None:
        self._emergency_flatten_requested = True

    # ── Operator-driven RL switches ───────────────────────────────────

    async def approve_rl_promotion(self, *, gates: dict | None = None) -> dict:
        if self.promotion_controller is None:
            raise RuntimeError("promotion_controller not configured")
        result = await self.promotion_controller.promote_rl_to_production(gates=gates)
        self.rl_is_production = True
        self._rl_promoted_at = datetime.now(timezone.utc)
        self._rl_nav_at_promotion = self.order_manager.portfolio.nav
        self._log_compliance("rl_promoted", result)
        return result

    async def revert_to_hrp(self, reason: str, *, flatten: bool = False) -> dict:
        if self.promotion_controller is None:
            raise RuntimeError("promotion_controller not configured")
        result = await self.promotion_controller.revert_to_hrp(reason, flatten=flatten)
        self.rl_is_production = False
        self._log_compliance("rl_reverted", result)
        return result

    # ── Utility ───────────────────────────────────────────────────────

    def _log_compliance(self, event: str, payload: dict[str, Any]) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        try:
            self.compliance_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.compliance_log_path.open("a") as fh:
                fh.write(str(entry) + "\n")
        except Exception as exc:  # pragma: no cover
            log.debug("compliance log write failed: %s", exc)

    async def _send_alert(
        self, severity: AlertSeverity, title: str, message: str,
        *, metadata: dict | None = None,
    ) -> None:
        try:
            await self.alert_manager.send_alert(Alert(
                severity=severity,
                title=title,
                message=message,
                source="live_trading",
                metadata=metadata or {},
            ))
        except Exception as exc:  # pragma: no cover
            log.warning("alert send failed: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="live_trading")
    p.add_argument("--asset-class", choices=["equities", "crypto", "futures"],
                   default="equities")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--emergency-flatten", action="store_true",
                   help="Flatten all positions and exit")
    p.add_argument("--enable-rl-shadow", action="store_true",
                   help="Run the RL agent in shadow mode alongside HRP")
    p.add_argument("--check-rl-promotion", action="store_true",
                   help="Print RL shadow eligibility and exit")
    p.add_argument("--approve-rl-promotion", action="store_true",
                   help="Operator approval to promote RL to production")
    p.add_argument("--revert-to-hrp", type=str, default=None,
                   metavar="REASON",
                   help="Revert the live optimizer to HRP with the given reason")
    return p.parse_args()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - glue only
    args = _parse_args() if argv is None else _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if HALT_FILE.exists():
        print(f"REFUSED: HALT file {HALT_FILE} exists; remove it after preflight.",
              file=sys.stderr)
        return 1
    log.info(
        "Live trading CLI — asset_class=%s emergency_flatten=%s",
        args.asset_class, args.emergency_flatten,
    )
    log.warning(
        "Live trading requires a project-level bootstrap that wires the real "
        "BrokerFactory, PreflightChecker, and CapitalDeploymentController.",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
