"""``GET /api/v1/overview`` — Command Center top-row aggregates.

Pulls NAV, P&L, drawdown, exposures, action counts, top-N actionable
ideas, and stage latency. The Command Center loads this single payload
on first paint and then uses ``/stream/ops`` for live updates.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..envelope import envelope

router = APIRouter(prefix="/overview", tags=["overview"])


@router.get("")
def get_overview() -> dict:
    # TODO: aggregate from portfolio_service, trade_ideas_service,
    # monitoring_service, model_service.
    data = {
        "nav": None,
        "daily_pnl": None,
        "drawdown": None,
        "gross_exposure": None,
        "net_exposure": None,
        "positions_count": None,
        "action_counts": {
            "BUY": 0, "SELL": 0, "WATCH": 0,
            "MODEL_REQUIRED": 0, "NO_DATA": 0, "ERROR": 0,
        },
        "top_actionable": [],
        "stage_latency_seconds": {},
        "warnings": [],
    }
    return envelope(data, source="overview_service", warnings=["overview aggregator stub"])
