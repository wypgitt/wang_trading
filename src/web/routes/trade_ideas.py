"""``GET /api/v1/trade-ideas`` and ``GET /api/v1/trade-ideas/{symbol}``.

The list endpoint is the highest-traffic surface in the app. It reads the
tmpfs snapshot the engine publishes (``src.execution.trade_idea_publisher``)
and never fabricates or regenerates: a missing/unreadable snapshot surfaces
as a typed ``SnapshotUnavailable`` (enveloped, no leaked exception text), and
the snapshot's measured staleness rides in the envelope so the client can
flag degraded freshness.

``/{symbol}`` resolves the single idea from the same snapshot — the detail
drawer hydrates from this row; there is no separate engine round-trip in v1.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ..deps import get_trade_ideas_service
from ..envelope import envelope
from ..errors import SnapshotUnavailable
from ..services.trade_ideas_service import TradeIdeasService

router = APIRouter(prefix="/trade-ideas", tags=["trade-ideas"])

# Snapshots older than this are flagged degraded in the envelope (still shipped).
_STALE_THRESHOLD_SECONDS = 90.0


def _parse_symbols(symbols: str | None) -> list[str] | None:
    return [s.strip().upper() for s in (symbols or "").split(",") if s.strip()] or None


@router.get("")
def list_trade_ideas(
    symbols: Annotated[str | None, Query(description="comma-separated symbols")] = None,
    service: TradeIdeasService = Depends(get_trade_ideas_service),
) -> dict:
    try:
        response, staleness = service.read_snapshot(symbols=_parse_symbols(symbols))
    except Exception as exc:  # noqa: BLE001
        raise SnapshotUnavailable("trade-ideas snapshot unavailable") from exc

    # Read-only invariant: the BFF never regenerates on the request path. No
    # readable snapshot ⇒ an honest 503 (enveloped by the handler), never a
    # raw 500 and never a synchronous engine run.
    if response is None:
        raise SnapshotUnavailable("no readable trade-ideas snapshot")

    warnings: list[str] = []
    if staleness is not None and staleness > _STALE_THRESHOLD_SECONDS:
        warnings.append(f"trade-ideas snapshot is {staleness:.0f}s old (stale)")

    return envelope(
        response,
        source="trade_ideas_service",
        staleness_seconds=staleness,
        warnings=warnings,
    )


@router.get("/{symbol}")
def get_trade_idea_detail(
    symbol: str,
    service: TradeIdeasService = Depends(get_trade_ideas_service),
) -> dict:
    # Resolve the single idea from the snapshot (the drawer hydrates from the
    # list row). Detail-only fields (full SHAP / cascade / signal rows) remain
    # COMING until a persisted detail store lands; this never 500s the client.
    try:
        response, staleness = service.read_snapshot(symbols=[symbol.upper()])
    except Exception as exc:  # noqa: BLE001
        raise SnapshotUnavailable("trade-ideas snapshot unavailable") from exc

    warnings: list[str] = []
    if response is None:
        # Distinguish "no snapshot at all" from "snapshot present, no match"
        # so the client can tell COMING/empty apart from a real null idea.
        warnings.append("trade-ideas snapshot unavailable")
        idea = None
    else:
        idea = next(
            (i for i in response.ideas if i.symbol.upper() == symbol.upper()), None
        )

    return envelope(
        idea,
        source="trade_ideas_service",
        staleness_seconds=staleness,
        warnings=warnings,
    )
