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

from .services.bars_gateway import BarsGateway
from .services.health_service import HealthService
from .services.markets_service import MarketsService
from .services.model_service import ModelService
from .services.preflight_service import PreflightService
from .services.regime_service import RegimeService
from .services.replay_service import ReplayService
from .services.scenario_service import ScenarioService
from .services.signals_service import SignalsService
from .services.symbols_service import SymbolsService
from .services.trade_ideas_service import TradeIdeasService


@lru_cache(maxsize=1)
def get_trade_ideas_service() -> TradeIdeasService:
    return TradeIdeasService()


@lru_cache(maxsize=1)
def get_markets_service() -> MarketsService:
    return MarketsService()


@lru_cache(maxsize=1)
def get_symbols_service() -> SymbolsService:
    return SymbolsService()


@lru_cache(maxsize=1)
def get_signals_service() -> SignalsService:
    return SignalsService()


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()


@lru_cache(maxsize=1)
def get_health_service() -> HealthService:
    # Aggregates the freshness SLO from the singleton readers + a bars probe.
    return HealthService(
        ideas=get_trade_ideas_service(),
        gateway=BarsGateway(),
        model=get_model_service(),
    )


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
    get_markets_service.cache_clear()
    get_symbols_service.cache_clear()
    get_signals_service.cache_clear()
    get_model_service.cache_clear()
    get_health_service.cache_clear()
    get_regime_service.cache_clear()
    get_scenario_service.cache_clear()
    get_replay_service.cache_clear()
    get_preflight_service.cache_clear()


__all__ = [
    "get_health_service",
    "get_markets_service",
    "get_model_service",
    "get_preflight_service",
    "get_regime_service",
    "get_replay_service",
    "get_scenario_service",
    "get_signals_service",
    "get_symbols_service",
    "get_trade_ideas_service",
    "reset_service_singletons",
]
