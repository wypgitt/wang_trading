"""``GET /api/v1/overview`` — Command Center top-row aggregates.

Action counts, the top-N actionable ideas, and per-stage engine latency are
aggregated from the trade-ideas snapshot — the same source ``/trade-ideas``
serves. Portfolio metrics (NAV, history) require a persisted portfolio, which
the engine does not write yet (docs/data_readiness.md), so they stay null
until that persistence lands.

This is the reference route: real where data exists, ``null`` + an envelope
``warning`` where it does not, and degrade-don't-500 on any source failure.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_trade_ideas_service
from ..dtos import (
    ActionCounts,
    EnginePulse,
    EnginePulseStage,
    OverviewResponse,
)
from ..envelope import envelope
from ..services.trade_ideas_service import TradeIdeasService

router = APIRouter(prefix="/overview", tags=["overview"])

# Highest-conviction ideas surfaced on the Command Center top row.
_TOP_ACTIONABLE = 5

# Canonical engine pipeline order for the latency strip; unknown stages are
# appended in first-seen order so a new stage never silently disappears.
_STAGE_ORDER = (
    "data_fetch",
    "feature_compute",
    "signal_generation",
    "meta_inference",
    "sizing",
    "target_generation",
)

# Portfolio fields with no producer today; kept in the payload as null so the
# client contract is stable once a portfolio store lands.
_PORTFOLIO_UNAVAILABLE = (
    "portfolio metrics (nav/pnl/drawdown/exposure) unavailable: no persisted portfolio"
)


def _engine_pulse(ideas) -> EnginePulse:
    acc: dict[str, float] = {}
    for idea in ideas:
        for stage, secs in (idea.stage_latency_seconds or {}).items():
            acc[stage] = acc.get(stage, 0.0) + float(secs)
    ordered = [s for s in _STAGE_ORDER if s in acc]
    ordered += [s for s in acc if s not in _STAGE_ORDER]
    stages = [EnginePulseStage(stage=s, seconds=round(acc[s], 3)) for s in ordered]
    total = round(sum(s.seconds for s in stages), 3)
    return EnginePulse(stages=stages, total_seconds=total)


@router.get("")
def get_overview(
    service: TradeIdeasService = Depends(get_trade_ideas_service),
) -> dict:
    warnings = [_PORTFOLIO_UNAVAILABLE]
    counts = ActionCounts()
    top_actionable: list = []
    pulse = EnginePulse()
    staleness: float | None = None

    # Degrade rather than 500: a missing/stale snapshot must not take the
    # dashboard down — and the read-only path never regenerates it.
    try:
        response, staleness = service.read_snapshot()
    except Exception:  # noqa: BLE001
        response = None

    if response is None:
        warnings.append("overview trade-ideas aggregation unavailable")
    else:
        totals = response.totals
        counts = ActionCounts(
            buy=totals.buy,
            sell=totals.sell,
            watch=totals.watch,
            model_required=totals.model_required,
            no_data=totals.no_data,
        )
        actionable = sorted(
            (idea for idea in response.ideas if idea.action in ("BUY", "SELL")),
            key=lambda idea: abs(idea.target_weight or 0.0),
            reverse=True,
        )
        top_actionable = actionable[:_TOP_ACTIONABLE]
        pulse = _engine_pulse(response.ideas)

    data = OverviewResponse(
        action_counts=counts,
        top_actionable=top_actionable,
        engine_pulse=pulse,
        nav=None,
        nav_history=None,
        regime=None,
    )
    return envelope(
        data,
        source="overview_service",
        staleness_seconds=staleness,
        warnings=warnings,
    )
