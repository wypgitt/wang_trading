"""``GET /api/v1/trade-ideas`` and ``GET /api/v1/trade-ideas/{symbol}``.

The list endpoint is the highest-traffic surface in the app. It must
remain compatible with the v1 ``src.ui.trade_ideas`` report payload so
the existing report machinery does not need to be rewritten before the
React table can render.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from ..envelope import envelope
from ..services.trade_ideas_service import TradeIdeasService

router = APIRouter(prefix="/trade-ideas", tags=["trade-ideas"])


@router.get("")
def list_trade_ideas(
    symbols: Annotated[str | None, Query(description="comma-separated symbols")] = None,
    bar_limit: int = 500,
    min_abs_weight: float = 0.0025,
    allow_confidence_fallback: bool = False,
) -> dict:
    parsed = [s.strip().upper() for s in (symbols or "").split(",") if s.strip()] or None
    service = TradeIdeasService()
    try:
        response = service.list_ideas(
            symbols=parsed,
            bar_limit=bar_limit,
            min_abs_weight=min_abs_weight,
            allow_confidence_fallback=allow_confidence_fallback,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return envelope(
        response.model_dump(mode="json"),
        source="trade_ideas_service",
    )


@router.get("/{symbol}")
def get_trade_idea_detail(symbol: str) -> dict:
    # TODO: wire to TradeIdeasService.get_detail once it can return the
    # full TradeIdeaDetail (signals + signal_metadata + shap + sizing
    # waterfall + microstructure + cost forecast + track record).
    raise HTTPException(
        status_code=501,
        detail="TradeIdeaDetail not yet implemented; see docs/api_contracts_v2.md §1.2",
    )
