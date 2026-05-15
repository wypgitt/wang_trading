"""``GET /api/v1/preflight`` — go-live checklist status."""

from __future__ import annotations

from fastapi import APIRouter

from ..envelope import envelope
from ..services.preflight_service import PreflightService

router = APIRouter(prefix="/preflight", tags=["preflight"])
_service = PreflightService()


@router.get("")
def get_preflight() -> dict:
    status = _service.status()
    return envelope(
        status.model_dump(mode="json"),
        source="preflight_service",
    )
