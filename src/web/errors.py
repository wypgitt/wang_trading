"""Error taxonomy and exception handlers for the BFF.

Implements docs/backend_design.md §9.1–9.2. The frontend switches on
``ErrorCode``, never on ``message``. ``STALE_*`` codes are intentional
200s with a non-empty ``errors`` array — the response payload is still
shipped, just flagged degraded.

Routes raise the typed subclasses (``BrokerUnavailable``, ``ModelUnavailable``,
``DbUnavailable``, ``TimeoutError``, ``DegradedResponse``); the registered
handlers translate them into enveloped JSON with the correct HTTP status.
The generic ``Exception`` handler never leaks exception text or stack
traces to the response — it logs them keyed by ``X-Request-Id`` instead.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .envelope import ApiError, envelope


logger = logging.getLogger("web.errors")


class ErrorCode(str, Enum):
    """Stable, frontend-facing error identifiers (backend_design §9.1)."""

    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    STALE_FACTOR_MODEL = "STALE_FACTOR_MODEL"
    STALE_MODEL = "STALE_MODEL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    BROKER_UNAVAILABLE = "BROKER_UNAVAILABLE"
    DB_UNAVAILABLE = "DB_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL = "INTERNAL"


# HTTP status mapping per §9.1 table.
_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.BAD_REQUEST: 400,
    ErrorCode.VALIDATION_FAILED: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.CONFLICT: 409,
    ErrorCode.STALE_FACTOR_MODEL: 200,
    ErrorCode.STALE_MODEL: 200,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.BROKER_UNAVAILABLE: 503,
    ErrorCode.DB_UNAVAILABLE: 503,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.INTERNAL: 500,
}


def http_status_for(code: ErrorCode) -> int:
    """Map an :class:`ErrorCode` to its HTTP status code."""

    return _HTTP_STATUS.get(code, 500)


class ApiException(Exception):
    """Base class for all BFF-raised, envelope-friendly exceptions.

    Subclasses set ``code`` and (optionally) override ``http_status``.
    Routes raise these; the registered handlers serialise them into
    the standard envelope with an ``errors`` entry.
    """

    code: ErrorCode = ErrorCode.INTERNAL
    http_status: int | None = None  # falls back to _HTTP_STATUS

    def __init__(
        self,
        message: str = "",
        *,
        field: str | None = None,
        source: str = "bff",
        data: Any = None,
    ) -> None:
        super().__init__(message or self.code.value)
        self.message = message or self.code.value
        self.field = field
        self.source = source
        self.data = data

    @property
    def status_code(self) -> int:
        if self.http_status is not None:
            return self.http_status
        return http_status_for(self.code)

    def to_envelope(self) -> dict[str, Any]:
        return envelope(
            self.data,
            source=self.source,
            errors=[ApiError(code=self.code.value, message=self.message, field=self.field)],
        )


class DegradedResponse(ApiException):
    """Success with degraded data — HTTP 200, ``STALE_*`` code in errors.

    Routes raise this when the payload is shippable but a fallback /
    stale source was used. Defaults to ``STALE_MODEL``; pass ``code=``
    to specialise (e.g. ``ErrorCode.STALE_FACTOR_MODEL``).
    """

    code = ErrorCode.STALE_MODEL

    def __init__(
        self,
        message: str = "",
        *,
        code: ErrorCode | None = None,
        field: str | None = None,
        source: str = "bff",
        data: Any = None,
    ) -> None:
        if code is not None:
            # Per-instance override; class attribute remains the default.
            self.code = code
        super().__init__(message, field=field, source=source, data=data)


class BrokerUnavailable(ApiException):
    code = ErrorCode.BROKER_UNAVAILABLE


class ModelUnavailable(ApiException):
    code = ErrorCode.MODEL_UNAVAILABLE


class DbUnavailable(ApiException):
    code = ErrorCode.DB_UNAVAILABLE


class TimeoutError(ApiException):  # noqa: A001 — domain exception, intentional shadowing
    """Upstream timeout — surfaces as HTTP 504 per §9.3."""

    code = ErrorCode.TIMEOUT


class BadRequest(ApiException):
    code = ErrorCode.BAD_REQUEST


class NotFound(ApiException):
    code = ErrorCode.NOT_FOUND


class Unauthenticated(ApiException):
    code = ErrorCode.UNAUTHENTICATED


class Forbidden(ApiException):
    code = ErrorCode.FORBIDDEN


class Conflict(ApiException):
    code = ErrorCode.CONFLICT


class RateLimited(ApiException):
    code = ErrorCode.RATE_LIMITED


# ----------------------------------------------------------------------
# Exception handlers
# ----------------------------------------------------------------------


def _json_envelope_response(status_code: int, body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body))


async def _api_exception_handler(request: Request, exc: ApiException) -> JSONResponse:
    # DegradedResponse is logged at INFO; everything else at WARNING+.
    log_level = logging.INFO if isinstance(exc, DegradedResponse) else logging.WARNING
    logger.log(
        log_level,
        "api_exception",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "route": f"{request.method} {request.url.path}",
            "code": exc.code.value,
        },
    )
    return _json_envelope_response(exc.status_code, exc.to_envelope())


async def _validation_handler(
    request: Request, exc: RequestValidationError | ValidationError
) -> JSONResponse:
    # Surface field paths but never the offending value.
    errors: list[ApiError] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        errors.append(
            ApiError(
                code=ErrorCode.VALIDATION_FAILED.value,
                message=err.get("msg", "validation failed"),
                field=loc or None,
            )
        )
    if not errors:
        errors.append(ApiError(code=ErrorCode.VALIDATION_FAILED.value, message="validation failed"))
    body = envelope(None, source="bff", errors=errors)
    logger.warning(
        "validation_failed",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "route": f"{request.method} {request.url.path}",
            "error_count": len(errors),
        },
    )
    return _json_envelope_response(http_status_for(ErrorCode.VALIDATION_FAILED), body)


async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all. NEVER includes exception text or stack trace in response."""

    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "route": f"{request.method} {request.url.path}",
        },
    )
    body = envelope(
        None,
        source="bff",
        errors=[
            ApiError(
                code=ErrorCode.INTERNAL.value,
                message=(
                    f"internal error (request_id={request_id})"
                    if request_id
                    else "internal error"
                ),
            )
        ],
    )
    return _json_envelope_response(http_status_for(ErrorCode.INTERNAL), body)


def register_exception_handlers(app: FastAPI) -> None:
    """Wire all BFF exception handlers onto ``app`` (backend_design §9.2)."""

    app.add_exception_handler(ApiException, _api_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(ValidationError, _validation_handler)
    app.add_exception_handler(Exception, _generic_exception_handler)


__all__ = [
    "ApiException",
    "BadRequest",
    "BrokerUnavailable",
    "Conflict",
    "DbUnavailable",
    "DegradedResponse",
    "ErrorCode",
    "Forbidden",
    "ModelUnavailable",
    "NotFound",
    "RateLimited",
    "TimeoutError",
    "Unauthenticated",
    "http_status_for",
    "register_exception_handlers",
]
