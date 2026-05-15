"""Regime snapshot service.

Wraps :mod:`src.ml_layer.regime_detector` (LSTM + HMM labeler). The
detector emits per-bar probabilities for ``trending_up``,
``trending_down``, ``mean_reverting``, and ``high_volatility``; this
service consumes the latest row and returns a :class:`RegimeSnapshot`.

The current implementation is a stub that returns ``None`` until the
runtime regime detector is wired into the BFF context. Routes can call
``RegimeService.latest_snapshot()`` defensively and fall back to ``None``
in the envelope.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..envelope import RegimeProbabilities, RegimeSnapshot


class RegimeService:
    def __init__(self) -> None:
        # TODO: receive a handle to the runtime regime detector via DI.
        self._detector = None

    def latest_snapshot(self, symbol: str | None = None) -> RegimeSnapshot | None:
        if self._detector is None:
            return None
        # TODO: pull the latest detector output for the universe (or the
        # specified symbol when per-symbol regimes are supported).
        return None

    def fit_score(self, family: str, regime: RegimeSnapshot | None) -> float | None:
        """Return [0, 1] alignment between a signal family and a regime.

        Used to populate ``regime_fit_score`` on :class:`TradeIdea`. The
        mapping is heuristic for now and should be replaced by the
        regime-conditioned family attribution table from
        ``/api/v1/signals/family-regime-attribution``.
        """

        if regime is None:
            return None
        prefer = _FAMILY_REGIME_AFFINITY.get(family, {})
        return prefer.get(regime.label)


_FAMILY_REGIME_AFFINITY: dict[str, dict[str, float]] = {
    "ts_momentum":    {"trending_up": 0.85, "trending_down": 0.55, "mean_reverting": 0.25, "high_volatility": 0.40},
    "cs_momentum":    {"trending_up": 0.80, "trending_down": 0.55, "mean_reverting": 0.30, "high_volatility": 0.45},
    "ma_crossover":   {"trending_up": 0.80, "trending_down": 0.55, "mean_reverting": 0.20, "high_volatility": 0.35},
    "donchian_breakout": {"trending_up": 0.82, "trending_down": 0.58, "mean_reverting": 0.20, "high_volatility": 0.40},
    "mean_reversion": {"trending_up": 0.30, "trending_down": 0.30, "mean_reverting": 0.85, "high_volatility": 0.45},
    "stat_arb":       {"trending_up": 0.35, "trending_down": 0.35, "mean_reverting": 0.80, "high_volatility": 0.40},
    "futures_carry":  {"trending_up": 0.65, "trending_down": 0.45, "mean_reverting": 0.55, "high_volatility": 0.35},
    "funding_rate_arb": {"trending_up": 0.60, "trending_down": 0.55, "mean_reverting": 0.60, "high_volatility": 0.40},
    "cross_exchange_arb": {"trending_up": 0.60, "trending_down": 0.60, "mean_reverting": 0.60, "high_volatility": 0.65},
    "vrp":            {"trending_up": 0.55, "trending_down": 0.55, "mean_reverting": 0.50, "high_volatility": 0.80},
}
