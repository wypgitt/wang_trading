"""``GET /api/v1/overview`` — Command Center top-row aggregates.

Action counts, the top-N actionable ideas, and per-stage latency are
aggregated from the trade-ideas snapshot — the same source the
``/trade-ideas`` route serves. Portfolio metrics (NAV, P&L, drawdown,
exposures, positions) require a persisted portfolio, which the engine
does not write yet (see docs/data_readiness.md), so they stay null until
that persistence lands.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..envelope import envelope
from ..services.trade_ideas_service import TradeIdeasService

router = APIRouter(prefix="/overview", tags=["overview"])

# Highest-conviction ideas surfaced on the Command Center top row.
_TOP_ACTIONABLE = 5

# Portfolio fields with no producer today; kept in the payload as null so
# the client contract is stable once a portfolio store lands.
_PORTFOLIO_UNAVAILABLE = (
    "portfolio metrics (nav/pnl/drawdown/exposure) unavailable: no persisted portfolio"
)


@router.get("")
def get_overview() -> dict:
    action_counts = {
        "BUY": 0, "SELL": 0, "WATCH": 0,
        "MODEL_REQUIRED": 0, "NO_DATA": 0, "ERROR": 0,
    }
    top_actionable: list[dict] = []
    stage_latency_seconds: dict[str, float] = {}
    warnings = [_PORTFOLIO_UNAVAILABLE]

    # Degrade rather than 500: a missing/stale snapshot must not take the
    # dashboard down. list_ideas() can fall through to a sync pipeline
    # regenerate, so any failure there is caught here.
    try:
        response = TradeIdeasService().list_ideas()
    except Exception:  # noqa: BLE001
        response = None
        warnings.append("overview trade-ideas aggregation unavailable")

    if response is not None:
        totals = response.totals
        action_counts = {
            "BUY": totals.buy,
            "SELL": totals.sell,
            "WATCH": totals.watch,
            "MODEL_REQUIRED": totals.model_required,
            "NO_DATA": totals.no_data,
            "ERROR": totals.error,
        }
        actionable = sorted(
            (idea for idea in response.ideas if idea.action in ("BUY", "SELL")),
            key=lambda idea: abs(idea.target_weight or 0.0),
            reverse=True,
        )
        top_actionable = [idea.model_dump(mode="json") for idea in actionable[:_TOP_ACTIONABLE]]
        for idea in response.ideas:
            for stage, secs in (idea.stage_latency_seconds or {}).items():
                stage_latency_seconds[stage] = stage_latency_seconds.get(stage, 0.0) + float(secs)

    data = {
        "nav": None,
        "daily_pnl": None,
        "drawdown": None,
        "gross_exposure": None,
        "net_exposure": None,
        "positions_count": None,
        "action_counts": action_counts,
        "top_actionable": top_actionable,
        "stage_latency_seconds": stage_latency_seconds,
    }
    return envelope(data, source="overview_service", warnings=warnings)
