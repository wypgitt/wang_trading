"""``GET /symbols/{symbol}`` — single-instrument detail.

Builds a full :class:`SymbolView` from the two real sources: the bars
hypertable (via :class:`BarsGateway`) for the OHLCV series + latest-bar
microstructure, and the trade-ideas snapshot (via
:class:`TradeIdeasService`) for the joined idea, when one exists.

The read-only invariant holds: every field is either derived from a
persisted bars row / snapshot idea, or held null/empty because its
producer does not exist yet (``market_cap`` has no source). The service
never fabricates and never raises on a source failure — when the bars
table is unreachable it returns ``data_available=False`` and the route
ships the row anyway.
"""

from __future__ import annotations

import logging
from typing import Any

from ..dtos import (
    BarMicro,
    Candle,
    PricePoint,
    SymbolDetailResponse,
    SymbolView,
    TradeIdea,
)
from ..errors import NotFound
from ..reference import instrument
from .bars_gateway import BarsGateway, closes_of, window_change_pct
from .trade_ideas_service import TradeIdeasService

log = logging.getLogger(__name__)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bar_micro(row: dict[str, Any], bar_type: str) -> BarMicro:
    """Map the latest persisted bars row to the microstructure DTO.

    Mirrors the ``/markets`` mapping exactly: only persisted bars columns
    are surfaced; absent values stay null (``exclude_none`` drops them).
    """

    buy_volume = _optional_float(row.get("buy_volume"))
    sell_volume = _optional_float(row.get("sell_volume"))
    buy_ticks = _optional_float(row.get("buy_ticks"))
    sell_ticks = _optional_float(row.get("sell_ticks"))
    tick_count = _optional_int(row.get("tick_count"))

    # Both are DERIVED from persisted columns (not separate columns), so they
    # are honest computations, not fabrications — mirror the /markets mapping.
    volume_imbalance = None
    if buy_volume is not None and sell_volume is not None:
        volume_imbalance = buy_volume - sell_volume
    tick_imbalance_ratio = None
    if buy_ticks is not None and sell_ticks is not None and tick_count:
        tick_imbalance_ratio = (buy_ticks - sell_ticks) / tick_count

    return BarMicro(
        bar_type=bar_type,
        vwap=_optional_float(row.get("vwap")),
        dollar_volume=_optional_float(row.get("dollar_volume")),
        tick_count=tick_count,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        volume_imbalance=volume_imbalance,
        tick_imbalance_ratio=tick_imbalance_ratio,
        imbalance=_optional_float(row.get("imbalance")),
        threshold=_optional_float(row.get("threshold")),
        bar_duration_seconds=_optional_float(row.get("bar_duration_seconds")),
    )


class SymbolsService:
    """Assembles the single-instrument detail view, degradation-safe."""

    def __init__(
        self,
        *,
        gateway: BarsGateway | None = None,
        ideas: TradeIdeasService | None = None,
        candle_limit: int = 130,
    ) -> None:
        self._gateway = gateway or BarsGateway()
        self._ideas = ideas or TradeIdeasService()
        self._candle_limit = int(candle_limit)

    def get_symbol(self, symbol: str) -> SymbolDetailResponse:
        ref = instrument(symbol)
        if ref is None:
            raise NotFound(f"unknown symbol {symbol}")

        bar_type = ref["bar_type"]
        rows = self._gateway.recent_bars(symbol, bar_type, self._candle_limit)

        view = self._build_view(symbol, ref, bar_type, rows)
        idea = self._join_idea(symbol)
        return SymbolDetailResponse(sym=view, idea=idea)

    # ── internals ─────────────────────────────────────────────────────

    def _build_view(
        self,
        symbol: str,
        ref: dict[str, Any],
        bar_type: str,
        rows: list[dict[str, Any]] | None,
    ) -> SymbolView:
        base = dict(
            symbol=symbol.upper(),
            name=ref["name"],
            type=ref["asset_class"],
            bar_type=bar_type,
            market_cap=None,  # COMING: no source in bars or the universe map
        )

        # DB unreachable: render the static card, everything bar-derived null.
        if rows is None:
            return SymbolView(
                **base,
                price=None,
                spark=[],
                change_window_pct=None,
                volume=None,
                bars_loaded=0,
                latest_bar_at=None,
                candles=[],
                line=[],
                bar=None,
                data_available=False,
            )

        # Table reachable but empty for this symbol.
        if not rows:
            return SymbolView(
                **base,
                price=None,
                spark=[],
                change_window_pct=None,
                volume=None,
                bars_loaded=0,
                latest_bar_at=None,
                candles=[],
                line=[],
                bar=None,
                data_available=True,
            )

        candles = [
            Candle(
                t=i,
                o=_optional_float(r.get("open")),
                h=_optional_float(r.get("high")),
                l=_optional_float(r.get("low")),
                c=_optional_float(r.get("close")),
                v=_optional_float(r.get("volume")),
            )
            for i, r in enumerate(rows)
        ]
        line = [
            PricePoint(t=i, v=_optional_float(r.get("close")))
            for i, r in enumerate(rows)
        ]
        closes = closes_of(rows)  # shared with /markets — single source of truth
        last = rows[-1]

        return SymbolView(
            **base,
            price=_optional_float(last.get("close")),
            spark=closes,
            change_window_pct=window_change_pct(closes),
            volume=_optional_float(last.get("volume")),
            bars_loaded=len(rows),
            latest_bar_at=last.get("timestamp"),
            candles=candles,
            line=line,
            bar=_bar_micro(last, bar_type),
            data_available=True,
        )

    def _join_idea(self, symbol: str) -> TradeIdea | None:
        """Matching idea from the snapshot, or ``None`` (never raises)."""

        try:
            response, _ = self._ideas.read_snapshot(symbols=[symbol])  # read-only
        except Exception:  # noqa: BLE001 — degrade: the view still ships
            log.warning("symbols: idea join failed symbol=%s", symbol, exc_info=True)
            return None
        if response is None:
            return None
        wanted = symbol.upper()
        for idea in response.ideas:
            if idea.symbol.upper() == wanted:
                return idea
        return None


__all__ = ["SymbolsService"]
