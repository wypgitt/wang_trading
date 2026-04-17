"""Daily operations — reconciliation + summary report (Phase 5 / P5.13).

Runs once per trading day (or every 24h for 24x7 crypto) and emits a
human-readable report plus persists a portfolio snapshot. Any discrepancy
raises an alert through the configured AlertManager.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.execution.models import Order, OrderStatus, PortfolioState
from src.execution.order_manager import OrderManager
from src.execution.storage import ExecutionStorage
from src.execution.tca import TCAAnalyzer, TCAResult
from src.monitoring.alerting import AlertManager

log = logging.getLogger(__name__)


# ── Reconciliation ────────────────────────────────────────────────────

class DailyReconciliation:
    """End-of-day / 24h reconciliation + report generation."""

    async def run(
        self,
        order_manager: OrderManager,
        execution_storage: ExecutionStorage | None,
        alert_manager: AlertManager,
        *,
        now: datetime | None = None,
        fills_today: list[Order] | None = None,
        tca_results: list[TCAResult] | None = None,
        drift_report: pd.DataFrame | None = None,
    ) -> dict:
        now = now or datetime.now(timezone.utc)
        pf = order_manager.portfolio

        # 1. Reconcile positions
        discrepancies = await order_manager.reconcile_positions()

        # 2. Daily P&L (current NAV minus start-of-day peak snapshot)
        daily_pnl = pf.daily_pnl
        daily_return = daily_pnl / pf.nav if pf.nav > 0 else 0.0

        # 3. TCA roll-up
        tca_results = list(tca_results or [])
        tca_summary = TCAAnalyzer().get_tca_summary(tca_results) if tca_results else {}

        # 4. Drift
        drift_report = drift_report if drift_report is not None else pd.DataFrame()
        drift_flags = (
            drift_report.loc[drift_report.get("drifted", False), "feature"].tolist()
            if not drift_report.empty and "drifted" in drift_report.columns
            else []
        )

        # 5. Persist snapshot
        if execution_storage is not None:
            try:
                execution_storage.insert_portfolio_snapshot(pf, timestamp=now)
            except Exception as exc:
                log.warning("snapshot persistence failed: %s", exc)

        # 6. Build report
        report = generate_daily_report(
            portfolio=pf,
            trades=list(fills_today or []),
            tca_results=tca_results,
            drift_report=drift_report,
            as_of=now,
        )

        # 7. Alert on issues
        if discrepancies:
            await alert_manager.send_alert(
                alert_manager.alert_position_reconciliation(discrepancies),
            )
        for feat in drift_flags:
            await alert_manager.send_alert(
                alert_manager.alert_feature_drift(feat, kl=1.0),
            )

        return {
            "as_of": now,
            "nav": pf.nav,
            "daily_pnl": daily_pnl,
            "daily_return": daily_return,
            "discrepancies": discrepancies,
            "tca_summary": tca_summary,
            "drift_flags": drift_flags,
            "report": report,
        }


# ── Report ─────────────────────────────────────────────────────────────

def _returns(portfolio: PortfolioState, as_of: datetime) -> dict[str, float]:
    """Placeholder returns — real implementation would query historical snapshots."""
    start_nav = portfolio.peak_nav if portfolio.peak_nav else portfolio.nav
    daily_return = portfolio.daily_pnl / portfolio.nav if portfolio.nav > 0 else 0.0
    total_return = (
        (portfolio.nav - start_nav) / start_nav if start_nav > 0 else 0.0
    )
    return {
        "daily": daily_return,
        "mtd": total_return,  # approximation; real impl diff’s month-start NAV
        "ytd": total_return,  # approximation
    }


def generate_daily_report(
    portfolio: PortfolioState,
    trades: list[Order],
    tca_results: list[TCAResult],
    drift_report: pd.DataFrame | None = None,
    *,
    as_of: datetime | None = None,
    breakers_triggered: list[str] | None = None,
    last_model_retrain: datetime | None = None,
    next_retrain: datetime | None = None,
) -> str:
    as_of = as_of or datetime.now(timezone.utc)
    drift_report = drift_report if drift_report is not None else pd.DataFrame()
    breakers_triggered = breakers_triggered or []

    rets = _returns(portfolio, as_of)
    filled_today = [t for t in trades if t.status == OrderStatus.FILLED]
    pnl_today = sum(
        (t.avg_fill_price - (t.limit_price or t.avg_fill_price)) * t.quantity
        for t in filled_today
    )
    avg_slippage = (
        sum(r.slippage_bps for r in tca_results) / len(tca_results)
        if tca_results else 0.0
    )

    # Top/bottom by unrealized P&L
    by_pnl = sorted(
        portfolio.positions.values(), key=lambda p: p.unrealized_pnl, reverse=True,
    )
    top = by_pnl[:3]
    bottom = list(reversed(by_pnl[-3:])) if len(by_pnl) > 3 else []

    # Exposure by signal_family
    family_exposure: dict[str, float] = {}
    for p in portfolio.positions.values():
        family = p.signal_family or "unassigned"
        family_exposure[family] = family_exposure.get(family, 0.0) + abs(p.market_value)

    drift_lines: list[str]
    if not drift_report.empty and "drifted" in drift_report.columns:
        drifted = drift_report.loc[drift_report["drifted"], "feature"].tolist()
        drift_lines = drifted if drifted else ["No drift detected"]
    else:
        drift_lines = ["No drift report"]

    model_age = (
        f"{(as_of - last_model_retrain).total_seconds() / 3600:.1f}h"
        if last_model_retrain else "unknown"
    )
    next_retrain_str = next_retrain.isoformat() if next_retrain else "unscheduled"

    lines: list[str] = []
    lines.append(f"=== Daily Report — {as_of.date().isoformat()} ===")
    lines.append("")
    lines.append("## Portfolio")
    lines.append(f"  NAV:          ${portfolio.nav:,.2f}")
    lines.append(f"  Cash:         ${portfolio.cash:,.2f}")
    lines.append(f"  Daily P&L:    ${portfolio.daily_pnl:,.2f}")
    lines.append(f"  Daily return: {rets['daily']:.2%}")
    lines.append(f"  MTD return:   {rets['mtd']:.2%}")
    lines.append(f"  YTD return:   {rets['ytd']:.2%}")
    lines.append(f"  Drawdown:     {portfolio.drawdown:.2%}")
    lines.append(f"  Peak NAV:     ${portfolio.peak_nav:,.2f}")
    lines.append(f"  Positions:    {portfolio.position_count}")
    lines.append(f"  Gross:        ${portfolio.gross_exposure:,.2f}")
    lines.append(f"  Net:          ${portfolio.net_exposure:,.2f}")
    lines.append("")
    lines.append("## Trades Today")
    lines.append(f"  Count (filled): {len(filled_today)}")
    lines.append(f"  Net P&L:        ${pnl_today:,.2f}")
    lines.append(f"  Avg slippage:   {avg_slippage:.2f} bps")
    lines.append("")
    lines.append("## Top Positions (by unrealized P&L)")
    if top:
        for p in top:
            lines.append(
                f"  {p.symbol:<10} side={p.side:+d} qty={p.quantity:>6.0f} "
                f"upnl=${p.unrealized_pnl:,.0f} "
                f"ret={p.return_pct:.2%}"
            )
    else:
        lines.append("  (no positions)")
    if bottom:
        lines.append("## Bottom Positions")
        for p in bottom:
            lines.append(
                f"  {p.symbol:<10} side={p.side:+d} qty={p.quantity:>6.0f} "
                f"upnl=${p.unrealized_pnl:,.0f} "
                f"ret={p.return_pct:.2%}"
            )
    lines.append("")
    lines.append("## Exposure by Strategy")
    if family_exposure:
        for fam, exp in sorted(family_exposure.items(), key=lambda x: -x[1]):
            lines.append(f"  {fam:<20} ${exp:,.0f}")
    else:
        lines.append("  (no exposure)")
    lines.append("")
    lines.append("## Feature Drift Warnings")
    for ln in drift_lines:
        lines.append(f"  - {ln}")
    lines.append("")
    lines.append("## Circuit Breaker Activations")
    if breakers_triggered:
        for b in breakers_triggered:
            lines.append(f"  - {b}")
    else:
        lines.append("  None")
    lines.append("")
    lines.append("## Model")
    lines.append(f"  Last retrain: {model_age} ago")
    lines.append(f"  Next retrain: {next_retrain_str}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="daily_ops")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-reconciliation", action="store_true")
    g.add_argument("--send-daily-report", action="store_true")
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


def main() -> int:  # pragma: no cover — CLI glue
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    log.info("daily_ops: run_reconciliation=%s send_daily_report=%s",
             args.run_reconciliation, args.send_daily_report)
    log.warning(
        "CLI requires a bootstrapped OrderManager/ExecutionStorage/AlertManager; "
        "wire this up from your project bootstrap script."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
