"""``GET /api/v1/symbols/{symbol}`` — single-instrument detail.

Returns the full :class:`SymbolView` (real OHLCV candle series + latest-bar
microstructure from the bars hypertable) plus the joined trade idea from the
snapshot, when one exists.

Unknown symbols raise :class:`NotFound`, which the registered handler
envelopes as a 404 with ``errors[].code == "NOT_FOUND"``. A bars/snapshot
source failure does not 500: the service already degrades to
``data_available=False`` / ``idea=None``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_symbols_service
from ..envelope import envelope
from ..errors import NotFound
from ..services.symbols_service import SymbolsService

router = APIRouter(prefix="/symbols", tags=["symbols"])


@router.get("/{symbol}")
def get_symbol(
    symbol: str,
    service: SymbolsService = Depends(get_symbols_service),
) -> dict:
    # NotFound (unknown symbol) propagates → enveloped 404. Any *other*
    # unexpected failure degrades to a reference-only view rather than 500
    # (defence-in-depth; the service already degrades known source failures).
    try:
        detail = service.get_symbol(symbol)
    except NotFound:
        raise
    except Exception:  # noqa: BLE001
        from ..dtos import SymbolDetailResponse, SymbolView

        view = SymbolView(symbol=symbol.upper(), name=symbol.upper(), type="equity",
                          data_available=False)
        return envelope(
            SymbolDetailResponse(sym=view, idea=None),
            source="symbols_service",
            warnings=["symbol detail unavailable"],
        )

    # Honest freshness signal, consistent with /markets' DB-down warning.
    warnings = (
        ["bars database unavailable"] if not detail.sym.data_available else []
    )
    return envelope(detail, source="symbols_service", warnings=warnings)
