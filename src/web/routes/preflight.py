"""``GET /api/v1/preflight`` — go-live checklist status."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_preflight_service
from ..envelope import envelope
from ..services.preflight_service import PreflightService

router = APIRouter(prefix="/preflight", tags=["preflight"])


@router.get("")
async def get_preflight(
    service: PreflightService = Depends(get_preflight_service),
) -> dict:
    status = await service.status()
    return envelope(status, source="preflight_service")
