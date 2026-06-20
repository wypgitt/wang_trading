"""``GET /api/v1/replay`` — reconstruct state at a past timestamp.

NOTE: a MOCK scaffold, not part of the v1 8-endpoint contract (the client has
no accessor). Kept mounted for the v2 roadmap; it must still obey the BFF
contract — read-only, enveloped, no leaked exception text.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from ..deps import get_replay_service
from ..envelope import envelope
from ..services.replay_service import ReplayService

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("")
def replay(
    ts: datetime = Query(...),
    symbol: str | None = None,
    service: ReplayService = Depends(get_replay_service),
) -> dict:
    # No raw HTTPException: any failure propagates to the generic handler,
    # which logs the trace by request_id and returns an enveloped INTERNAL
    # error with no leaked exception text. Pass the DTO so data is camelCased.
    snapshot = service.snapshot_at(ts, symbol=symbol)
    return envelope(snapshot, source="replay_service", warnings=snapshot.warnings)
