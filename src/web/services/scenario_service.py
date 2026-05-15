"""Parametric / historical scenario engine.

Re-runs portfolio impact under user-supplied shocks against
:mod:`src.portfolio.factor_risk` for factor exposures and
:mod:`src.bet_sizing.cascade` for per-idea size impact. Strictly
read-only: no state mutation, no broker calls.

The initial scaffold returns deterministic mock results so the frontend
can wire the page. Replace the body of :meth:`run` with the real engine.
"""

from __future__ import annotations

from ..dtos import (
    AffectedIdea,
    ScenarioRequest,
    ScenarioResult,
    SuggestedHedge,
)


class ScenarioService:
    def __init__(self) -> None:
        pass

    def library(self) -> list[dict[str, str]]:
        return [
            {"id": "spy_down_3",  "label": "SPY -3%",   "type": "parametric"},
            {"id": "btc_down_10", "label": "BTC -10%",  "type": "parametric"},
            {"id": "vix_x15",     "label": "VIX x1.5",  "type": "vol_shock"},
            {"id": "corr_break",  "label": "Cross-asset corr -> 1", "type": "correlation"},
            {"id": "hist_2020_03_16", "label": "Replay 2020-03-16 shock", "type": "historical"},
        ]

    def run(self, request: ScenarioRequest) -> ScenarioResult:
        # TODO: integrate src.portfolio.factor_risk + bet_sizing.cascade.
        # For now return a deterministic mock so the page can render.
        shocks = request.shocks
        symbol_shock = sum(abs(v) for v in (shocks.symbol_pct or {}).values())
        approx_pnl = -41200.0 * (1.0 + symbol_shock)
        return ScenarioResult(
            pnl_impact_usd=approx_pnl,
            drawdown_impact=-0.012,
            factor_exposure_deltas={"F1": -0.04, "F2": 0.01},
            breaker_headroom={
                "max_daily_loss_pct_remaining": 0.008,
                "max_gross_exposure_remaining": 0.21,
            },
            affected_ideas=[
                AffectedIdea(symbol="AAPL", old_target=0.012, new_target=0.008, flipped=False),
                AffectedIdea(symbol="TSLA", old_target=0.011, new_target=-0.002, flipped=True),
            ],
            suggested_hedges=[SuggestedHedge(long="TLT", short="SPY", ratio=0.5)],
            warnings=["scenario engine stub — wire factor_risk + bet_sizing for live results"],
        )
