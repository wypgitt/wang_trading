"""``/signals`` services — the strategy roster.

Two read-only views over the signal battery's *static* roster plus the
*live* trade-ideas snapshot:

  * ``list_families()`` — one :class:`SignalFamilyCard` per registered
    family, in ``reference.FAMILY_META`` declaration order.
  * ``get_family(id)``  — that card plus the snapshot ideas attributed to
    the family.

Honest by construction (the read-only invariant):

  * **Metadata** (name / category / source / thesis / asset_classes /
    params) is *reference* data — it describes the strategy, it never
    reports a measured value.
  * **Status** is derived from what the engine can actually *run*: the
    battery's dispatch ``kind``. ``kind == 'bars'`` means the family runs
    from the bars a caller already supplies → ``live``; every other kind
    (panel / pair / exchange / futures-curve context that no caller wires)
    is dormant → ``shadow``.
  * **Activity** (``active_signals`` / per-family ``ideas``) is counted
    from the live trade-ideas snapshot — real engine output.
  * **Every performance field** (sharpe, win-rate, trades, contribution,
    P&L, allocation, avg-hold, regime-fit, equity-curve) is COMING: the
    engine writes no backtest-run or track-record persistence, so these
    stay ``None`` / empty. They are NEVER synthesised.

The battery is injectable so tests need no engine import: pass a
``battery_factory`` (default ``create_default_battery``). If the factory
raises, status falls back to a static bars-runnable allow-list.
"""

from __future__ import annotations

import logging

from .. import reference
from ..dtos import (
    FamilyParam,
    SignalFamilyCard,
    SignalFamilyDetailResponse,
    TradeIdea,
)
from ..errors import NotFound
from .trade_ideas_service import TradeIdeasService

log = logging.getLogger(__name__)


def create_default_battery():
    """Lazy proxy to the engine factory.

    Imported lazily inside the call so the BFF stays importable where the
    full signal-battery stack is unavailable (unit tests, docs builds).
    """

    from src.signal_battery.orchestrator import (
        create_default_battery as _factory,
    )

    return _factory()


# Families that run directly from bars when the battery cannot be probed
# (factory import/raise). These mirror ``kind == 'bars'`` registrations so
# the fallback degrades to the same live/shadow split, never a guess.
_STATIC_LIVE_FAMILIES: frozenset[str] = frozenset(
    {
        "ts_momentum",
        "mean_reversion",
        "ma_crossover",
        "donchian_breakout",
    }
)


class SignalsService:
    """Read-only roster of signal families (metadata + live activity)."""

    def __init__(
        self,
        *,
        ideas: TradeIdeasService | None = None,
        battery_factory=create_default_battery,
    ) -> None:
        self._ideas = ideas if ideas is not None else TradeIdeasService()
        self._battery_factory = battery_factory

    # ── status resolution ────────────────────────────────────────────

    def _status_map(self) -> dict[str, str]:
        """``family_id -> 'live' | 'shadow'`` from the battery dispatch kind.

        ``kind == 'bars'`` ⇒ runnable from bars a caller already supplies
        ⇒ ``live``; every other kind needs panel/pair/exchange/futures
        context no caller wires ⇒ ``shadow``. On any factory failure, fall
        back to the static bars-runnable allow-list so the surface still
        reflects runnability rather than going dark.
        """

        try:
            battery = self._battery_factory()
            kinds = {r.name: r.kind for r in battery._registry}
        except Exception as exc:  # noqa: BLE001
            log.warning("signal battery probe failed, using static status: %s", exc)
            return {
                fid: ("live" if fid in _STATIC_LIVE_FAMILIES else "shadow")
                for fid in reference.FAMILY_META
            }

        status: dict[str, str] = {}
        for fid in reference.FAMILY_META:
            kind = kinds.get(fid)
            status[fid] = "live" if kind == "bars" else "shadow"
        return status

    # ── snapshot activity ────────────────────────────────────────────

    def _snapshot_ideas(self) -> list[TradeIdea]:
        """All ideas in the live snapshot, or ``[]`` if the source is down."""

        try:
            response, _ = self._ideas.read_snapshot()  # read-only: never regenerates
        except Exception as exc:  # noqa: BLE001
            log.warning("signals: trade-ideas snapshot unavailable: %s", exc)
            return []
        return list(response.ideas) if response is not None else []

    @staticmethod
    def _attributed_to(idea: TradeIdea, family_id: str) -> bool:
        """Is ``idea`` attributed to ``family_id`` (strategy / top family)?"""

        return idea.strategy == family_id or idea.top_signal_family == family_id

    # ── card builder ─────────────────────────────────────────────────

    def _build_card(
        self,
        family_id: str,
        meta: reference.FamilyMeta,
        status: str,
        ideas: list[TradeIdea],
    ) -> SignalFamilyCard:
        active = sum(1 for idea in ideas if self._attributed_to(idea, family_id))
        return SignalFamilyCard(
            id=family_id,
            name=meta["name"],
            category=meta["category"],
            source=meta["source"],
            thesis=meta["thesis"],
            asset_classes=list(meta["asset_classes"]),
            params=[FamilyParam(**p) for p in meta["params"]],
            status=status,  # type: ignore[arg-type]
            active_signals=active,
            # COMING — no backtest-run / track-record persistence:
            sharpe=None,
            win_rate=None,
            trades=None,
            contribution_pct=None,
            pnl_ytd=None,
            allocation=None,
            avg_hold_bars=None,
            regime_fit={},
            equity_curve=[],
        )

    # ── public API ───────────────────────────────────────────────────

    def list_families(self) -> list[SignalFamilyCard]:
        """One card per family, in ``FAMILY_META`` declaration order."""

        status = self._status_map()
        ideas = self._snapshot_ideas()
        return [
            self._build_card(
                fid,
                meta,
                status.get(fid, "shadow"),
                ideas,
            )
            for fid, meta in reference.FAMILY_META.items()
        ]

    def get_family(self, family_id: str) -> SignalFamilyDetailResponse:
        """The family's card plus its attributed snapshot ideas."""

        meta = reference.FAMILY_META.get(family_id)
        if meta is None:
            raise NotFound(f"unknown signal family: {family_id}", field="family_id")

        status = self._status_map()
        ideas = self._snapshot_ideas()
        card = self._build_card(
            family_id,
            meta,
            status.get(family_id, "shadow"),
            ideas,
        )
        family_ideas = [
            idea for idea in ideas if self._attributed_to(idea, family_id)
        ]
        return SignalFamilyDetailResponse(strategy=card, ideas=family_ideas)


__all__ = ["SignalsService", "create_default_battery"]
