"""JSON structured logging for the BFF (backend_design §10.1).

Every log line is a single JSON object on stdout. journald captures it
in prod; ``docker logs`` captures it in dev. The middleware in
``src/web/app.py`` enriches each request's log records with
``request_id``, ``user``, ``role``, ``route``, ``status``, and
``duration_ms`` via a :class:`contextvars.ContextVar`.

Usage::

    from src.web.logging import setup_logging
    setup_logging(level="INFO")

then ``logging.getLogger("web.routes.trade_ideas").info("...")`` emits
JSON with the request context already attached.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# ---- Per-request context -----------------------------------------------

_request_id_var: ContextVar[str | None] = ContextVar("bff_request_id", default=None)
_user_var: ContextVar[str | None] = ContextVar("bff_user", default=None)
_role_var: ContextVar[str | None] = ContextVar("bff_role", default=None)
_route_var: ContextVar[str | None] = ContextVar("bff_route", default=None)
_status_var: ContextVar[int | None] = ContextVar("bff_status", default=None)
_duration_var: ContextVar[float | None] = ContextVar("bff_duration_ms", default=None)


def bind_request_context(
    *,
    request_id: str | None = None,
    user: str | None = None,
    role: str | None = None,
    route: str | None = None,
) -> dict[str, ContextVar[Any]]:
    """Bind request fields onto the contextvars; returns a token-style map.

    The middleware calls this on request entry and ``reset_request_context``
    on exit. Callers don't need to touch the contextvars directly.
    """

    tokens: dict[str, Any] = {}
    if request_id is not None:
        tokens["request_id"] = _request_id_var.set(request_id)
    if user is not None:
        tokens["user"] = _user_var.set(user)
    if role is not None:
        tokens["role"] = _role_var.set(role)
    if route is not None:
        tokens["route"] = _route_var.set(route)
    return tokens


def set_response_context(*, status: int | None = None, duration_ms: float | None = None) -> None:
    """Late-bind status and duration once the response is built."""

    if status is not None:
        _status_var.set(status)
    if duration_ms is not None:
        _duration_var.set(duration_ms)


def reset_request_context(tokens: dict[str, Any]) -> None:
    """Reset contextvars to whatever they were before the request started."""

    if "request_id" in tokens:
        _request_id_var.reset(tokens["request_id"])
    if "user" in tokens:
        _user_var.reset(tokens["user"])
    if "role" in tokens:
        _role_var.reset(tokens["role"])
    if "route" in tokens:
        _route_var.reset(tokens["route"])
    # status/duration are short-lived per-request, just clear them
    _status_var.set(None)
    _duration_var.set(None)


def current_request_id() -> str | None:
    return _request_id_var.get()


# ---- Formatter ---------------------------------------------------------

# Reserved LogRecord attribute names — anything else attached via ``extra``
# is surfaced as a top-level JSON field.
_RESERVED_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


class JSONFormatter(logging.Formatter):
    """Render a :class:`logging.LogRecord` as the docs §10.1 JSON shape."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        # Base fields per §10.1.
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "ts": ts.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Pull request context from contextvars (always shown when set).
        req_id = _request_id_var.get()
        if req_id is not None:
            payload["request_id"] = req_id
        user = _user_var.get()
        if user is not None:
            payload["user"] = user
        role = _role_var.get()
        if role is not None:
            payload["role"] = role
        route = _route_var.get()
        if route is not None:
            payload["route"] = route
        status = _status_var.get()
        if status is not None:
            payload["status"] = status
        duration = _duration_var.get()
        if duration is not None:
            payload["duration_ms"] = round(duration, 3)

        # Promote any ``extra`` fields the caller attached.
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            # Explicit per-record overrides win over contextvars.
            payload[key] = value

        # Exception info, if any, goes under a single "exc" field as a string
        # — never inside the response body, only in the log.
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


# ---- Setup helper ------------------------------------------------------


def setup_logging(*, level: str | int = "INFO", stream: Any = None) -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: re-installs the same handler shape on each call. Safe to
    call from app startup and from test fixtures.
    """

    root = logging.getLogger()
    # Strip any pre-existing handlers (e.g. uvicorn's default) so we don't
    # emit double-formatted lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    if isinstance(level, str):
        root.setLevel(level.upper())
    else:
        root.setLevel(level)

    # Make uvicorn / FastAPI loggers propagate to the root JSON handler
    # instead of writing their own access lines.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


__all__ = [
    "JSONFormatter",
    "bind_request_context",
    "current_request_id",
    "reset_request_context",
    "set_response_context",
    "setup_logging",
]
