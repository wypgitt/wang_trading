"""Grafana dashboard + alert-rule JSON generation (Phase 5 / P5.11).

Panels are declared in a compact schema that is rendered into Grafana's
JSON model. Dashboards can be imported via the Grafana HTTP API (see
`scripts/setup_grafana.py`).
"""

from __future__ import annotations

from typing import Any

_METRIC_PREFIX = "wang_trading"


def _panel(
    title: str,
    panel_type: str,
    expr: str,
    *,
    panel_id: int,
    grid: tuple[int, int, int, int],
    datasource: str = "Prometheus",
    extra: dict[str, Any] | None = None,
) -> dict:
    x, y, w, h = grid
    panel: dict[str, Any] = {
        "id": panel_id,
        "title": title,
        "type": panel_type,
        "datasource": datasource,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"expr": expr, "refId": "A"}],
    }
    if extra:
        panel.update(extra)
    return panel


def _row(title: str, panel_id: int, y: int) -> dict:
    return {
        "id": panel_id,
        "type": "row",
        "title": title,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "collapsed": False,
    }


# ── Main dashboard ─────────────────────────────────────────────────────

def generate_main_dashboard() -> dict:
    m = _METRIC_PREFIX
    panels: list[dict] = []
    next_id = 1

    def add_row(title: str, y: int) -> None:
        nonlocal next_id
        panels.append(_row(title, next_id, y))
        next_id += 1

    def add_panel(title: str, ptype: str, expr: str, grid, *,
                  datasource: str = "Prometheus", extra=None) -> None:
        nonlocal next_id
        panels.append(_panel(title, ptype, expr, panel_id=next_id,
                             grid=grid, datasource=datasource, extra=extra))
        next_id += 1

    # Row 1 — Portfolio Overview (y=0)
    add_row("Portfolio Overview", 0)
    add_panel("NAV", "timeseries", f"{m}_portfolio_nav", (0, 1, 12, 8))
    add_panel("Daily P&L", "barchart", f"{m}_portfolio_daily_pnl", (12, 1, 6, 8))
    add_panel("Drawdown", "timeseries", f"{m}_portfolio_drawdown", (18, 1, 6, 8))
    add_panel("Gross Exposure", "gauge", f"{m}_portfolio_gross_exposure", (0, 9, 6, 4))
    add_panel("Net Exposure", "gauge", f"{m}_portfolio_net_exposure", (6, 9, 6, 4))

    # Row 2 — Positions (y=13)
    add_row("Positions", 13)
    add_panel(
        "Positions", "table",
        "SELECT symbol, side, quantity, avg_entry_price, current_price, "
        "unrealized_pnl FROM positions ORDER BY abs(unrealized_pnl) DESC",
        (0, 14, 18, 8), datasource="TimescaleDB",
    )
    add_panel("Position Count", "gauge", f"{m}_positions_count", (18, 14, 6, 8))

    # Row 3 — Signals & Model (y=22)
    add_row("Signals & Model", 22)
    add_panel("Signals per Family", "barchart",
              f"sum by (family) ({m}_signal_count)", (0, 23, 10, 8))
    add_panel("Meta-Labeler Probability",
              "histogram", f"{m}_meta_label_prob_bucket", (10, 23, 8, 8))
    add_panel("Regime Detector State", "piechart",
              f"{m}_signal_count{{family=~\"regime_.*\"}}", (18, 23, 6, 8))
    add_panel("Model Age (hours)", "gauge",
              f"{m}_model_last_retrain_age_hours", (0, 31, 6, 4))

    # Row 4 — Execution (y=35)
    add_row("Execution", 35)
    add_panel("Slippage Distribution", "histogram",
              f"{m}_execution_slippage_bps_bucket", (0, 36, 10, 8))
    add_panel(
        "TCA Summary (24h)", "table",
        "SELECT symbol, algo, AVG(slippage_bps) AS avg_slippage, "
        "AVG(impact_bps) AS avg_impact, COUNT(*) AS n "
        "FROM tca_results WHERE timestamp > now() - INTERVAL '24 hours' "
        "GROUP BY symbol, algo ORDER BY n DESC",
        (10, 36, 10, 8), datasource="TimescaleDB",
    )
    add_panel("Orders Submitted", "stat",
              f"{m}_orders_submitted_total", (20, 36, 4, 4))
    add_panel("Orders Filled", "stat",
              f"{m}_orders_filled_total", (20, 40, 2, 4))
    add_panel("Orders Rejected", "stat",
              f"{m}_orders_rejected_total", (22, 40, 2, 4))

    # Row 5 — Data Health (y=44)
    add_row("Data Health", 44)
    add_panel("Bar Formation Rate", "timeseries",
              f"{m}_bar_formation_rate", (0, 45, 12, 8))
    add_panel("Feature Drift (KL)", "heatmap",
              f"{m}_feature_drift_kl", (12, 45, 8, 8))
    add_panel("Data Gaps", "timeseries",
              f"{m}_data_gap_seconds", (20, 45, 4, 8))

    # Row 6 — Risk (y=53)
    add_row("Risk", 53)
    add_panel(
        "Strategy Allocation", "piechart",
        f"sum by (family) ({m}_signal_count)",
        (0, 54, 8, 8),
    )
    add_panel(
        "Correlation Matrix", "heatmap",
        "SELECT timestamp, symbol_pair, correlation "
        "FROM correlation_matrix WHERE timestamp > now() - INTERVAL '1 hour'",
        (8, 54, 8, 8), datasource="TimescaleDB",
    )
    add_panel("Circuit Breaker Triggers", "barchart",
              f"sum by (breaker_type) ({m}_circuit_breaker_triggers)",
              (16, 54, 8, 8))

    return {
        "uid": "wang-trading-main",
        "title": "Wang Trading — Main Dashboard",
        "tags": ["wang-trading", "quant"],
        "timezone": "browser",
        "refresh": "30s",
        "schemaVersion": 38,
        "version": 1,
        "time": {"from": "now-24h", "to": "now"},
        "panels": panels,
    }


# ── Alerting rules ─────────────────────────────────────────────────────

def generate_alerting_rules() -> dict:
    m = _METRIC_PREFIX
    rules = [
        {
            "uid": "dd-warning",
            "title": "Drawdown Warning",
            "condition": "A",
            "data": [{"refId": "A", "model": {"expr": f"{m}_portfolio_drawdown"}}],
            "for": "1m",
            "annotations": {"summary": "Drawdown > 5%"},
            "labels": {"severity": "warning"},
            "threshold": 0.05,
        },
        {
            "uid": "dd-critical",
            "title": "Drawdown Critical",
            "condition": "A",
            "data": [{"refId": "A", "model": {"expr": f"{m}_portfolio_drawdown"}}],
            "for": "0s",
            "annotations": {"summary": "Drawdown > 10%"},
            "labels": {"severity": "critical"},
            "threshold": 0.10,
        },
        {
            "uid": "daily-loss",
            "title": "Daily Loss Breach",
            "condition": "A",
            "data": [{"refId": "A", "model": {"expr": f"{m}_portfolio_daily_pnl"}}],
            "for": "0s",
            "annotations": {"summary": "Daily P&L below -2%"},
            "labels": {"severity": "critical"},
            "threshold": -0.02,
        },
        {
            "uid": "model-stale",
            "title": "Model Stale",
            "condition": "A",
            "data": [
                {"refId": "A",
                 "model": {"expr": f"{m}_model_last_retrain_age_hours"}},
            ],
            "for": "10m",
            "annotations": {"summary": "No retrain in 720h"},
            "labels": {"severity": "warning"},
            "threshold": 720,
        },
        {
            "uid": "data-gap",
            "title": "Data Gap",
            "condition": "A",
            "data": [
                {"refId": "A", "model": {"expr": f"{m}_data_gap_seconds"}},
            ],
            "for": "1m",
            "annotations": {"summary": "Data gap > 300s"},
            "labels": {"severity": "warning"},
            "threshold": 300,
        },
        {
            "uid": "feature-drift",
            "title": "Feature Drift",
            "condition": "A",
            "data": [
                {"refId": "A", "model": {"expr": f"{m}_feature_drift_kl"}},
            ],
            "for": "5m",
            "annotations": {"summary": "KL divergence > 0.5 on a feature"},
            "labels": {"severity": "warning"},
            "threshold": 0.5,
        },
    ]
    return {
        "apiVersion": 1,
        "groups": [
            {
                "name": "wang-trading-core",
                "folder": "wang-trading",
                "interval": "1m",
                "rules": rules,
            }
        ],
    }
