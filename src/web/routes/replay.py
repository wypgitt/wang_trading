"""``GET /api/v1/replay`` — reconstruct state at a past timestamp."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from ..envelope import envelope
from ..services.replay_service import ReplayService

router = APIRouter(prefix="/replay", tags=["replay"])
_service = ReplayService()


@router.get("")
def replay(ts: datetime = Query(...), symbol: str | None = None) -> dict:
    try:
        snapshot = _service.snapshot_at(ts, symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return envelope(
        snapshot.model_dump(mode="json"),
        source="replay_service",
        warnings=snapshot.warnings,
    )
