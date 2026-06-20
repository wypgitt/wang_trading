"""``/markets`` grid assembly — bar-derived rows, honestly partial.

For every instrument in the static universe (:func:`reference.all_symbols`),
this service reads the most-recent bars via :class:`BarsGateway` and shapes a
:class:`MarketRow`. The gateway already encodes the three states the grid must
render honestly:

* ``None``  → the bars table was unreachable for this symbol; the row renders
  from static reference metadata only (``data_available=False``, price/spark/
  bar all null).
* ``[]``    → the table is reachable but holds no rows for the symbol
  (``data_available=True``, ``bars_loaded=0``, price null).
* rows      → real bars; price/spark/volume/microstructure derived from them.

The trade-ideas snapshot is joined once (a single read-only
:meth:`read_snapshot` call — never regenerates) to flag which symbols carry a
live idea. ``market_cap`` is *always* null — no producer exists in the bars
table or the universe map, so it is never synthesised.

Every external call (bars DB, snapshot) is injected through the constructor so
tests need no live infrastructure.
"""

from __future__ import annotations

import logging
from typing import Any

from ..dtos import BarMicro, MarketRow
from ..reference import all_symbols, instrument
from .bars_gateway import BarsGateway, closes_of, window_change_pct
from .trade_ideas_service import TradeIdeasService

log = logging.getLogger(__name__)


def _f(value: Any) -> float | None:
    """Coerce a bar-row cell to ``float`` or ``None`` (never raise)."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    """Coerce a bar-row cell to ``int`` or ``None`` (never raise)."""

    f = _f(value)
    if f is None:
        return None
    return int(f)


class MarketsService:
    """Assemble the markets grid from bars + the trade-ideas snapshot."""

    def __init__(
        self,
        *,
        gateway: BarsGateway | None = None,
        ideas: TradeIdeasService | None = None,
        spark_limit: int = 60,
    ) -> None:
        # Lazy defaults: real reads only happen when the route calls
        # ``list_markets``; tests inject fakes through these args.
        self._gateway = gateway if gateway is not None else BarsGateway()
        self._ideas = ideas if ideas is not None else TradeIdeasService()
        self._spark_limit = int(spark_limit)

    # ── public API ─────────────────────────────────────────────────────

    def list_markets(self) -> list[MarketRow]:
        """One :class:`MarketRow` per instrument in declaration order."""

        idea_by_symbol = self._load_idea_index()

        rows: list[MarketRow] = []
        for symbol in all_symbols():
            ref = instrument(symbol)
            if ref is None:  # defensive — all_symbols() draws from INSTRUMENTS
                continue
            bar_type = ref["bar_type"]
            bars = self._gateway.recent_bars(symbol, bar_type, self._spark_limit)
            row = self._row_from_bars(symbol, ref, bar_type, bars)
            self._apply_idea(row, idea_by_symbol.get(symbol.upper()))
            rows.append(row)
        return rows

    # ── internals ──────────────────────────────────────────────────────

    def _load_idea_index(self) -> dict[str, Any]:
        """Join the snapshot once; degrade to empty on any failure."""

        try:
            response, _ = self._ideas.read_snapshot()  # read-only: never regenerates
        except Exception as exc:  # noqa: BLE001 — snapshot optional; degrade
            log.warning("markets: trade-ideas join unavailable: %s", exc)
            return {}
        if response is None:
            return {}
        return {idea.symbol.upper(): idea for idea in response.ideas}

    def _row_from_bars(
        self,
        symbol: str,
        ref: dict[str, Any],
        bar_type: str,
        bars: list[dict[str, Any]] | None,
    ) -> MarketRow:
        base = dict(
            symbol=symbol,
            name=ref["name"],
            type=ref["asset_class"],
            market_cap=None,  # COMING: no source in bars or the universe map
        )

        # DB unreachable for this symbol — render from reference only.
        if bars is None:
            return MarketRow(**base, data_available=False, bars_loaded=0)

        # Table reachable but empty — row renders, price stays null.
        if not bars:
            return MarketRow(**base, data_available=True, bars_loaded=0)

        last = bars[-1]
        closes = closes_of(bars)  # shared with /symbols — single source of truth

        return MarketRow(
            **base,
            data_available=True,
            bars_loaded=len(bars),
            price=_f(last.get("close")),
            spark=closes,
            change_window_pct=window_change_pct(closes),
            volume=_f(last.get("volume")),
            latest_bar_at=last.get("timestamp"),
            bar=self._bar_micro(bar_type, last),
        )

    @staticmethod
    def _bar_micro(bar_type: str, last: dict[str, Any]) -> BarMicro:
        buy_volume = _f(last.get("buy_volume"))
        sell_volume = _f(last.get("sell_volume"))
        buy_ticks = _f(last.get("buy_ticks"))
        sell_ticks = _f(last.get("sell_ticks"))
        tick_count = _i(last.get("tick_count"))

        volume_imbalance = None
        if buy_volume is not None and sell_volume is not None:
            volume_imbalance = buy_volume - sell_volume

        tick_imbalance_ratio = None
        if (
            buy_ticks is not None
            and sell_ticks is not None
            and tick_count
        ):
            tick_imbalance_ratio = (buy_ticks - sell_ticks) / tick_count

        return BarMicro(
            bar_type=bar_type,
            vwap=_f(last.get("vwap")),
            dollar_volume=_f(last.get("dollar_volume")),
            tick_count=tick_count,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_imbalance=volume_imbalance,
            tick_imbalance_ratio=tick_imbalance_ratio,
            imbalance=_f(last.get("imbalance")),
            threshold=_f(last.get("threshold")),
            bar_duration_seconds=_f(last.get("bar_duration_seconds")),
        )

    @staticmethod
    def _apply_idea(row: MarketRow, idea: Any) -> None:
        if idea is None:
            return
        row.has_idea = True
        row.action = idea.action
        row.target_weight = idea.target_weight


__all__ = ["MarketsService"]
