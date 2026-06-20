"""Standard API response envelope.

Every BFF response is wrapped in :class:`ApiEnvelope` (see
docs/api_contracts_v2.md §0.1). The envelope carries freshness, model, and
regime metadata in addition to the endpoint payload, plus separate
``warnings`` and ``errors`` lists.

Routes use the :func:`envelope` helper or build the model directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


def _new_request_id() -> str:
    """Short per-response id so the UI's Error state can render a copyable ref."""
    return "req_" + uuid.uuid4().hex[:16]

T = TypeVar("T")

RegimeLabel = Literal[
    "trending_up", "trending_down", "mean_reverting", "high_volatility", "unknown"
]


class RegimeProbabilities(BaseModel):
    trending_up: float
    trending_down: float
    mean_reverting: float
    high_volatility: float


class RegimeSnapshot(BaseModel):
    label: RegimeLabel
    probabilities: RegimeProbabilities
    as_of: datetime


class ApiError(BaseModel):
    code: str
    message: str
    field: str | None = None


class ApiEnvelope(BaseModel, Generic[T]):
    as_of: datetime
    source: str
    staleness_seconds: float | None = None
    source_freshness: dict[str, float] | None = None
    model_version: str | None = None
    regime: RegimeSnapshot | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[ApiError] = Field(default_factory=list)
    request_id: str = Field(default_factory=_new_request_id)
    data: T | None = None


def envelope(
    data: Any,
    *,
    source: str,
    staleness_seconds: float | None = None,
    source_freshness: dict[str, float] | None = None,
    model_version: str | None = None,
    regime: RegimeSnapshot | None = None,
    warnings: list[str] | None = None,
    errors: list[ApiError] | None = None,
) -> dict[str, Any]:
    """Build a fully serialised envelope dict.

    Routes can ``return envelope(...)`` directly; FastAPI will serialise
    the dict. Use :class:`ApiEnvelope` directly when typing is preferred.
    """

    env = ApiEnvelope[Any](
        as_of=datetime.now(timezone.utc),
        source=source,
        staleness_seconds=staleness_seconds,
        source_freshness=source_freshness,
        model_version=model_version,
        regime=regime,
        warnings=list(warnings or []),
        errors=list(errors or []),
        data=data,
    )
    return env.model_dump(mode="json", exclude_none=True)
