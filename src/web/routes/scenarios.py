"""``GET /api/v1/scenarios/library`` and ``POST /api/v1/scenarios/run``.

The scenario engine is strictly read-only. It rebalances against a
shocked factor / vol / correlation surface but never mutates portfolio
state.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_scenario_service
from ..dtos import ScenarioRequest
from ..envelope import envelope
from ..services.scenario_service import ScenarioService

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.get("/library")
def scenario_library(
    service: ScenarioService = Depends(get_scenario_service),
) -> dict:
    return envelope(service.library(), source="scenario_service")


@router.post("/run")
def run_scenario(
    request: ScenarioRequest,
    service: ScenarioService = Depends(get_scenario_service),
) -> dict:
    result = service.run(request)
    return envelope(result, source="scenario_service", warnings=result.warnings)
