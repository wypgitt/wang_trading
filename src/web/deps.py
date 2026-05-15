"""FastAPI ``Depends()`` providers for the BFF service singletons.

Each service is a plain Python class living in ``src/web/services/``.
The BFF wires them in via :func:`functools.lru_cache(maxsize=1)` so that
every request reuses the same instance — equivalent to a process-wide
singleton, but lazy-initialised on first use.

Routes write::

    @router.get("")
    def get_trade_ideas(svc: TradeIdeasService = Depends(get_trade_ideas_service)):
        ...

Tests can override any provider via ``app.dependency_overrides`` without
touching the lru_cache (the override layer wraps it).
"""

from __future__ import annotations

from functools import lru_cache

from .services.preflight_service import PreflightService
from .services.regime_service import RegimeService
from .services.replay_service import ReplayService
from .services.scenario_service import ScenarioService
from .services.trade_ideas_service import TradeIdeasService


@lru_cache(maxsize=1)
def get_trade_ideas_service() -> TradeIdeasService:
    return TradeIdeasService()


@lru_cache(maxsize=1)
def get_regime_service() -> RegimeService:
    return RegimeService()


@lru_cache(maxsize=1)
def get_scenario_service() -> ScenarioService:
    return ScenarioService()


@lru_cache(maxsize=1)
def get_replay_service() -> ReplayService:
    return ReplayService()


@lru_cache(maxsize=1)
def get_preflight_service() -> PreflightService:
    return PreflightService()


def reset_service_singletons() -> None:
    """Clear all cached singletons. Tests call this between cases."""

    get_trade_ideas_service.cache_clear()
    get_regime_service.cache_clear()
    get_scenario_service.cache_clear()
    get_replay_service.cache_clear()
    get_preflight_service.cache_clear()


__all__ = [
    "get_preflight_service",
    "get_regime_service",
    "get_replay_service",
    "get_scenario_service",
    "get_trade_ideas_service",
    "reset_service_singletons",
]
