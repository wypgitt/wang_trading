"""Pre-flight checks that must pass before live capital is deployed (P6.05).

Each check is a standalone async method that returns a ``PreflightCheck`` so
CI/humans can see a detailed go/no-go breakdown. A *blocker* failing means
trading must not start; a *warning* surfaces an issue without blocking.

Dependencies (broker factory, portfolio, model registry, metrics collector)
are duck-typed — only the attributes and methods actually called here are
required. This keeps the preflight loosely coupled and easy to mock in tests.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Check record ──────────────────────────────────────────────────────────

@dataclass
class PreflightCheck:
    name: str
    description: str
    severity: str  # "blocker" | "warning"
    passed: bool = False
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


# ── Default config ────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "broker": {"heartbeat_timeout_s": 5.0, "min_buying_power_per_asset": 1000.0},
    "model": {
        "max_age_days": 30,
        "min_training_events": 500,
        "required_gates": ("cpcv", "dsr", "pbo"),
    },
    "paper": {
        "min_weeks_history": 8,
        "min_sharpe": 1.0,
        "max_drawdown": 0.15,
        "min_win_rate": 0.5,
        "min_completed_trades": 50,
    },
    "infra": {
        "max_db_disk_pct": 80,
        "feature_freshness_max_h": 24,
    },
    "risk": {
        "max_single_position": 0.10,
        "max_daily_loss": 0.02,
        "dead_mans_switch_path": ".dead_mans_switch",
        "dead_mans_switch_max_age_h": 24,
    },
}


# ── Checker ───────────────────────────────────────────────────────────────

class PreflightChecker:
    """Runs a suite of go/no-go checks before live trading is enabled."""

    CATEGORIES: tuple[str, ...] = (
        "broker_connectivity",
        "model_readiness",
        "paper_validation",
        "infrastructure",
        "risk_limits",
        "operator",
    )

    def __init__(
        self,
        *,
        broker_factory: Any = None,
        portfolio: Any = None,
        model_registry: Any = None,
        metrics: Any = None,
        paper_stats: dict[str, Any] | None = None,
        infra: dict[str, Any] | None = None,
        risk_config: dict[str, Any] | None = None,
        operator_config: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.broker_factory = broker_factory
        self.portfolio = portfolio
        self.model_registry = model_registry
        self.metrics = metrics
        self.paper_stats = paper_stats or {}
        self.infra = infra or {}
        self.risk_config = risk_config or {}
        self.operator_config = operator_config or {}
        self.config = {**DEFAULTS, **(config or {})}

    # ── Public API ────────────────────────────────────────────────────

    async def run_all_checks(self) -> list[PreflightCheck]:
        category_fns: dict[str, Callable[[], Awaitable[list[PreflightCheck]]]] = {
            "broker_connectivity": self.check_broker_connectivity,
            "model_readiness": self.check_model_readiness,
            "paper_validation": self.check_paper_validation,
            "infrastructure": self.check_infrastructure,
            "risk_limits": self.check_risk_limits,
            "operator": self.check_operator,
        }
        out: list[PreflightCheck] = []
        for fn in category_fns.values():
            out.extend(await fn())
        return out

    async def run_category(self, category: str) -> list[PreflightCheck]:
        fn = getattr(self, f"check_{category}", None)
        if fn is None:
            raise ValueError(f"Unknown check category: {category!r}")
        return await fn()

    @staticmethod
    def summary(checks: list[PreflightCheck]) -> dict[str, Any]:
        blockers_failed = sum(
            1 for c in checks if c.severity == "blocker" and not c.passed
        )
        warnings_failed = sum(
            1 for c in checks if c.severity == "warning" and not c.passed
        )
        return {
            "all_passed": blockers_failed == 0,
            "blockers_failed": blockers_failed,
            "warnings_failed": warnings_failed,
            "checks": [c.to_dict() for c in checks],
        }

    # ── Broker connectivity ───────────────────────────────────────────

    async def check_broker_connectivity(self) -> list[PreflightCheck]:
        out: list[PreflightCheck] = []
        timeout = float(self.config["broker"]["heartbeat_timeout_s"])
        min_bp = float(self.config["broker"]["min_buying_power_per_asset"])

        if self.broker_factory is None:
            out.append(PreflightCheck(
                "broker.factory_present", "Broker factory configured",
                severity="blocker", passed=False, message="no broker_factory",
            ))
            return out

        # Heartbeat
        hb_check = PreflightCheck(
            "broker.heartbeat",
            "All configured brokers respond to heartbeat",
            severity="blocker",
        )
        try:
            hb = await asyncio.wait_for(self.broker_factory.heartbeat_all(), timeout)
            hb_check.details = {"results": hb}
            hb_check.passed = bool(hb) and all(hb.values())
            hb_check.message = "all brokers healthy" if hb_check.passed else f"failing: {[k for k, v in hb.items() if not v]}"
        except asyncio.TimeoutError:
            hb_check.passed = False
            hb_check.message = f"heartbeat timed out after {timeout}s"
        out.append(hb_check)

        # Buying power & permissions
        brokers = self.broker_factory.get_all_brokers() or {}
        bp_check = PreflightCheck(
            "broker.buying_power",
            f"Each broker has buying power > ${min_bp:.0f}",
            severity="blocker",
        )
        details: dict[str, Any] = {}
        all_ok = bool(brokers)
        for name, broker in brokers.items():
            try:
                acct = await broker.get_account()
            except Exception as exc:
                details[name] = f"error: {exc}"
                all_ok = False
                continue
            bp = float(acct.get("buying_power") or acct.get("cash") or acct.get("nav") or 0.0)
            details[name] = bp
            if bp < min_bp:
                all_ok = False
        bp_check.details = details
        bp_check.passed = all_ok
        bp_check.message = "buying power sufficient" if all_ok else "one or more brokers under minimum"
        out.append(bp_check)

        # Trading permissions (duck-typed flag)
        perm_check = PreflightCheck(
            "broker.trading_permissions",
            "Broker API keys have trading permissions",
            severity="blocker",
        )
        perm_details: dict[str, Any] = {}
        perm_ok = True
        for name, broker in brokers.items():
            can_trade = getattr(broker, "can_trade", True)
            perm_details[name] = bool(can_trade)
            if not can_trade:
                perm_ok = False
        perm_check.details = perm_details
        perm_check.passed = perm_ok and bool(brokers)
        perm_check.message = "all keys have trading permissions" if perm_check.passed else "missing trading scope"
        out.append(perm_check)

        # Market open
        market_check = PreflightCheck(
            "broker.market_open",
            "At least one asset class market is currently open",
            severity="blocker",
        )
        market_open = False
        for broker in brokers.values():
            try:
                if hasattr(broker, "is_market_open"):
                    if await broker.is_market_open():  # type: ignore[misc]
                        market_open = True
                        break
            except Exception:
                continue
        # Crypto never closes; if any crypto broker is configured, assume open.
        if not market_open and any(
            type(b).__name__ == "CCXTBrokerAdapter" for b in brokers.values()
        ):
            market_open = True
        market_check.passed = market_open
        market_check.message = "at least one market is open" if market_open else "all markets closed"
        out.append(market_check)

        return out

    # ── Model readiness ───────────────────────────────────────────────

    async def check_model_readiness(self) -> list[PreflightCheck]:
        out: list[PreflightCheck] = []
        cfg = self.config["model"]
        reg = self.model_registry

        prod_check = PreflightCheck(
            "model.production_exists",
            "Production meta-labeler exists in the model registry",
            severity="blocker",
        )
        model_info: dict[str, Any] | None = None
        if reg is None:
            prod_check.passed = False
            prod_check.message = "model_registry not configured"
            out.append(prod_check)
            return out
        try:
            model_info = reg.get_production_model()  # duck-typed
        except AttributeError:
            try:
                model_info = reg.get_best_model(stage="production")  # type: ignore[call-arg]
            except Exception as exc:
                prod_check.message = f"error: {exc}"
        except Exception as exc:
            prod_check.message = f"error: {exc}"
        prod_check.passed = bool(model_info)
        prod_check.details = {"model": str(model_info) if model_info else None}
        if prod_check.passed and not prod_check.message:
            prod_check.message = "production model present"
        elif not prod_check.passed and not prod_check.message:
            prod_check.message = "no production model found"
        out.append(prod_check)

        # Downstream checks only meaningful if we have model info
        info = model_info or {}
        if not isinstance(info, dict):
            info = getattr(info, "__dict__", {}) or {}

        # Age
        age_check = PreflightCheck(
            "model.age",
            f"Production model trained within last {cfg['max_age_days']} days",
            severity="blocker",
        )
        trained_at = info.get("trained_at") or info.get("training_time")
        if isinstance(trained_at, str):
            try:
                trained_at = datetime.fromisoformat(trained_at)
            except ValueError:
                trained_at = None
        if isinstance(trained_at, datetime):
            if trained_at.tzinfo is None:
                trained_at = trained_at.replace(tzinfo=timezone.utc)
            age = (_utcnow() - trained_at).days
            age_check.details = {"age_days": age}
            age_check.passed = age <= int(cfg["max_age_days"])
            age_check.message = f"model is {age} days old"
        else:
            age_check.passed = False
            age_check.message = "trained_at not available"
        out.append(age_check)

        # Training events
        evt_check = PreflightCheck(
            "model.training_events",
            f"Model trained on at least {cfg['min_training_events']} events",
            severity="blocker",
        )
        n_events = int(info.get("n_training_events") or info.get("n_events") or 0)
        evt_check.details = {"n_events": n_events}
        evt_check.passed = n_events >= int(cfg["min_training_events"])
        evt_check.message = f"{n_events} training events"
        out.append(evt_check)

        # Gates
        gate_check = PreflightCheck(
            "model.gates",
            "Model passed CPCV, DSR, and PBO gates",
            severity="blocker",
        )
        gates = info.get("gates") or {}
        missing = [g for g in cfg["required_gates"] if not gates.get(g)]
        gate_check.details = {"gates": gates, "missing": missing}
        gate_check.passed = not missing and bool(gates)
        gate_check.message = "all gates passed" if gate_check.passed else f"failing gates: {missing}"
        out.append(gate_check)

        # Regime detector
        regime_check = PreflightCheck(
            "model.regime_detector",
            "LSTM regime detector loaded and producing valid predictions",
            severity="blocker",
        )
        regime = getattr(reg, "regime_detector", None)
        try:
            ok = bool(regime and regime.is_ready())  # duck-typed
        except Exception:
            ok = False
        regime_check.passed = ok
        regime_check.message = "regime detector ready" if ok else "regime detector not ready"
        out.append(regime_check)

        return out

    # ── Paper validation ──────────────────────────────────────────────

    async def check_paper_validation(self) -> list[PreflightCheck]:
        cfg = self.config["paper"]
        s = self.paper_stats or {}
        out: list[PreflightCheck] = []

        def _make(name: str, desc: str, passed: bool, value: Any, threshold: Any) -> PreflightCheck:
            return PreflightCheck(
                name, desc, severity="blocker", passed=passed,
                message=f"value={value} threshold={threshold}",
                details={"value": value, "threshold": threshold},
            )

        weeks = float(s.get("weeks_history", 0))
        out.append(_make(
            "paper.weeks_history", "At least 8 weeks of paper trading history",
            weeks >= cfg["min_weeks_history"], weeks, cfg["min_weeks_history"],
        ))

        sharpe = float(s.get("sharpe", 0))
        out.append(_make(
            "paper.sharpe", "Paper trading Sharpe > 1.0",
            sharpe > cfg["min_sharpe"], sharpe, cfg["min_sharpe"],
        ))

        dd = float(s.get("max_drawdown", 1.0))
        out.append(_make(
            "paper.drawdown", "Paper trading max drawdown < 15%",
            dd < cfg["max_drawdown"], dd, cfg["max_drawdown"],
        ))

        wr = float(s.get("win_rate", 0))
        out.append(_make(
            "paper.win_rate", "Paper trading win rate > 50%",
            wr > cfg["min_win_rate"], wr, cfg["min_win_rate"],
        ))

        n = int(s.get("n_trades", 0))
        out.append(_make(
            "paper.n_trades", "At least 50 completed paper trades",
            n >= cfg["min_completed_trades"], n, cfg["min_completed_trades"],
        ))
        return out

    # ── Infrastructure ────────────────────────────────────────────────

    async def check_infrastructure(self) -> list[PreflightCheck]:
        cfg = self.config["infra"]
        infra = self.infra or {}
        out: list[PreflightCheck] = []

        def _add(name: str, desc: str, passed: bool, msg: str, details: dict | None = None) -> None:
            out.append(PreflightCheck(
                name, desc, severity="blocker",
                passed=passed, message=msg, details=details or {},
            ))

        db_reachable = bool(infra.get("db_reachable", False))
        db_pct = float(infra.get("db_disk_pct", 100.0))
        _add(
            "infra.timescaledb",
            f"TimescaleDB reachable and < {cfg['max_db_disk_pct']}% disk",
            db_reachable and db_pct < cfg["max_db_disk_pct"],
            f"reachable={db_reachable} disk_pct={db_pct}",
            {"reachable": db_reachable, "disk_pct": db_pct},
        )

        _add(
            "infra.prometheus",
            "Prometheus scrape target healthy",
            bool(infra.get("prometheus_up", False)), "",
        )
        _add(
            "infra.grafana",
            "Grafana server reachable",
            bool(infra.get("grafana_up", False)), "",
        )
        _add(
            "infra.alerts",
            "Alert channels (Telegram) working",
            bool(infra.get("alerts_ok", False)), "",
        )
        _add(
            "infra.mlflow",
            "MLflow tracking server reachable",
            bool(infra.get("mlflow_up", False)), "",
        )

        freshness_h = float(infra.get("feature_freshness_h", 9e9))
        _add(
            "infra.feature_freshness",
            f"Feature store updated within last {cfg['feature_freshness_max_h']}h",
            freshness_h <= cfg["feature_freshness_max_h"],
            f"freshness_h={freshness_h}",
            {"freshness_h": freshness_h},
        )
        return out

    # ── Risk limits ──────────────────────────────────────────────────

    async def check_risk_limits(self) -> list[PreflightCheck]:
        cfg = self.config["risk"]
        risk = self.risk_config or {}
        out: list[PreflightCheck] = []

        max_single = float(risk.get("max_single_position", 1.0))
        out.append(PreflightCheck(
            "risk.max_single_position",
            f"max_single_position <= {cfg['max_single_position']}",
            severity="blocker",
            passed=max_single <= cfg["max_single_position"],
            message=f"value={max_single}",
            details={"value": max_single, "threshold": cfg["max_single_position"]},
        ))

        daily_loss = float(risk.get("max_daily_loss", 1.0))
        out.append(PreflightCheck(
            "risk.max_daily_loss",
            f"max_daily_loss <= {cfg['max_daily_loss']}",
            severity="blocker",
            passed=daily_loss <= cfg["max_daily_loss"],
            message=f"value={daily_loss}",
            details={"value": daily_loss, "threshold": cfg["max_daily_loss"]},
        ))

        out.append(PreflightCheck(
            "risk.circuit_breakers",
            "Circuit breaker thresholds configured",
            severity="blocker",
            passed=bool(risk.get("circuit_breakers")),
            message="" if risk.get("circuit_breakers") else "not configured",
        ))

        # Dead man's switch
        path = Path(risk.get("dead_mans_switch_path") or cfg["dead_mans_switch_path"])
        max_age_h = float(cfg["dead_mans_switch_max_age_h"])
        dms_check = PreflightCheck(
            "risk.dead_mans_switch",
            f"Dead man's switch file exists and is < {max_age_h}h old",
            severity="blocker",
        )
        if path.exists():
            age_h = (_utcnow().timestamp() - path.stat().st_mtime) / 3600
            dms_check.passed = age_h < max_age_h
            dms_check.details = {"path": str(path), "age_h": age_h}
            dms_check.message = f"age_h={age_h:.2f}"
        else:
            dms_check.passed = False
            dms_check.message = f"missing: {path}"
        out.append(dms_check)

        return out

    # ── Operator ──────────────────────────────────────────────────────

    async def check_operator(self) -> list[PreflightCheck]:
        op = self.operator_config or {}
        out: list[PreflightCheck] = []
        out.append(PreflightCheck(
            "operator.risk_ack",
            "Operator has acknowledged live trading risks",
            severity="warning",
            passed=bool(op.get("risk_acknowledged")),
        ))
        out.append(PreflightCheck(
            "operator.emergency_contact",
            "Emergency contact information configured",
            severity="warning",
            passed=bool(op.get("emergency_contact")),
        ))
        out.append(PreflightCheck(
            "operator.working_hours",
            "Deployment during operator's working hours",
            severity="warning",
            passed=bool(op.get("within_working_hours", False)),
        ))
        return out


# ── CLI ───────────────────────────────────────────────────────────────────

def _print_summary(summary: dict[str, Any]) -> None:
    print(f"all_passed     = {summary['all_passed']}")
    print(f"blockers_failed= {summary['blockers_failed']}")
    print(f"warnings_failed= {summary['warnings_failed']}")
    print("-" * 60)
    for c in summary["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['severity']:7s} {c['name']:40s} {c['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("preflight")
    parser.add_argument("--full-check", action="store_true", help="Run the full suite")
    parser.add_argument("--check", type=str, default=None,
                        help="Run a single category")
    args = parser.parse_args(argv)

    checker = PreflightChecker()  # no deps → every blocker fails — exit code 1
    if args.check:
        checks = asyncio.run(checker.run_category(args.check))
    else:
        checks = asyncio.run(checker.run_all_checks())
    summary = PreflightChecker.summary(checks)
    _print_summary(summary)
    if summary["blockers_failed"] > 0:
        return 1
    if summary["warnings_failed"] > 0:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
