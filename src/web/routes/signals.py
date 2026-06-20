"""``/signals/families`` and ``/signals/family-{id}`` — the strategy roster.

Two read-only views:

  * ``GET /signals/families`` — the full family roster as a bare array of
    :class:`SignalFamilyCard` (camelCased element-wise by the envelope).
  * ``GET /signals/family-{family_id}`` — one family's card plus the
    snapshot ideas attributed to it. An unknown id raises :class:`NotFound`
    (HTTP 404 via the registered handler).

Note the literal path segment is ``family-{family_id}`` (the client calls
``/signals/family-ts_momentum``), not a nested ``/family/{id}``.

The service degrades internally (a down snapshot yields empty activity, not
a 500); the only hard error here is a genuinely unknown family id.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_signals_service
from ..dtos import SignalFamilyCard, SignalFamilyDetailResponse
from ..envelope import envelope
from ..services.signals_service import SignalsService

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/families")
def get_families(
    service: SignalsService = Depends(get_signals_service),
) -> dict:
    families: list[SignalFamilyCard] = service.list_families()
    return envelope(families, source="signals_service")


@router.get("/family-{family_id}")
def get_family(
    family_id: str,
    service: SignalsService = Depends(get_signals_service),
) -> dict:
    detail: SignalFamilyDetailResponse = service.get_family(family_id)
    return envelope(detail, source="signals_service")
