"""``GET /api/v1/model`` — the production meta-labeler card.

Sourced from the MLflow registry: version, run id, CV score, train accuracy,
training-event count, promotion-gate verdicts, and the retrain timeline are
all REAL engine outputs. AUC/Brier/ECE, calibration, feature importance,
drift, and the RL shadow are held null/empty (named in a warning) because the
engine does not log them yet — never synthesised.

Returned even with no model registered (all-null card + a warning) so the
client can render its MODEL_REQUIRED state. Degrade-don't-500 on any registry
failure.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_model_service
from ..envelope import envelope
from ..services.model_service import ModelService

router = APIRouter(prefix="/model", tags=["model"])


@router.get("")
def get_model(
    service: ModelService = Depends(get_model_service),
) -> dict:
    response, warnings = service.get_model()
    return envelope(
        response,
        source="model_service",
        model_version=response.version,
        warnings=warnings,
    )
