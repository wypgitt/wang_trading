"""``GET /api/v1/scenarios/library`` and ``POST /api/v1/scenarios/run``.

The scenario engine is strictly read-only. It rebalances against a
shocked factor / vol / correlation surface but never mutates portfolio
state.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..dtos import ScenarioRequest
from ..envelope import envelope
from ..services.scenario_service import ScenarioService

router = APIRouter(prefix="/scenarios", tags=["scenarios"])
_service = ScenarioService()


@router.get("/library")
def scenario_library() -> dict:
    return envelope(_service.library(), source="scenario_service")


@router.post("/run")
def run_scenario(request: ScenarioRequest) -> dict:
    result = _service.run(request)
    return envelope(
        result.model_dump(mode="json"),
        source="scenario_service",
        warnings=result.warnings,
    )
