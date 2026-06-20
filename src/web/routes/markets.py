"""``GET /api/v1/markets`` — the markets grid (bare array of rows).

One :class:`MarketRow` per instrument in the static universe. Each row is
bar-derived where the bars hypertable has data, and reference-only (price/
spark/bar null, ``data_available=False``) where it does not — never
synthesised. The trade-ideas snapshot is joined to flag live ideas.

Degrade-don't-500: a down bars database produces all-``data_available=False``
rows plus a single envelope warning, not a 5xx.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_markets_service
from ..envelope import envelope
from ..services.markets_service import MarketsService

router = APIRouter(prefix="/markets", tags=["markets"])

_BARS_DB_UNAVAILABLE = "bars database unavailable"


@router.get("")
def get_markets(
    service: MarketsService = Depends(get_markets_service),
) -> dict:
    warnings: list[str] = []

    # Degrade rather than 500: the service already swallows per-symbol
    # gateway/snapshot failures, but guard the aggregate call too.
    try:
        rows = service.list_markets()
    except Exception:  # noqa: BLE001
        rows = []
        warnings.append("markets grid unavailable")

    # When EVERY row is bars-unreachable, the bars DB is down as a whole.
    if rows and all(not row.data_available for row in rows):
        warnings.append(_BARS_DB_UNAVAILABLE)

    return envelope(rows, source="markets_service", warnings=warnings)
