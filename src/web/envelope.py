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
        data=None,
    )
    # Envelope metadata is ALWAYS present (no exclude_none): the clients bind
    # ``staleness_seconds``/``source_freshness``/``request_id`` as load-bearing
    # freshness/trust signals — dropping them when null would break the UI.
    dumped = env.model_dump(mode="json")
    # The honesty mechanism (absent inner field -> omitted -> null on the
    # client) and the camelCase contract apply to the DATA payload only.
    dumped["data"] = _dump_data(data)
    return dumped


def _dump_data(data: Any) -> Any:
    """Serialise the data payload: camelCase out, omit absent inner fields.

    Recurses through lists so bare-array responses (``/markets``,
    ``/signals/families``) camelCase each element. Plain dicts/scalars pass
    through untouched (their keys are data, not field names).
    """

    if isinstance(data, BaseModel):
        return data.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(data, list):
        return [_dump_data(item) for item in data]
    return data
