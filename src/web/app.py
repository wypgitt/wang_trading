"""FastAPI app entrypoint for the Wang Trading BFF.

Run locally::

    pip install -r requirements.txt
    uvicorn src.web.app:app --reload --port 8080

The app mounts all v2 routes under ``/api/v1`` and serves:

- ``/healthz`` for orchestration probes
- ``/metrics`` for Prometheus scraping (text/plain)

Observability:

- Every request gets an ``X-Request-Id`` (passed through or generated).
- Latency/status are recorded in ``bff_request_duration_seconds`` and
  ``bff_requests_total`` by the metrics middleware.
- Structured JSON logs are emitted on stdout (per ``backend_design`` §10.1).
- OpenTelemetry auto-instrumentation activates only when
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set in the environment (decision §24.4).

CORS is permissive in dev; tighten for production via ``config/web.yaml``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from .envelope import envelope
from .errors import register_exception_handlers
from .logging import (
    bind_request_context,
    reset_request_context,
    set_response_context,
    setup_logging,
)
from .metrics import instrument, metrics_endpoint
from .routes import overview, preflight, replay, scenarios, trade_ideas


_REQUEST_ID_HEADER = "X-Request-Id"
_logger = logging.getLogger("web.app")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Stub lifespan: log placeholders for DB pool / model registry init.

    Real wiring lands in a later sprint. For now we just emit a startup
    line so operators can confirm the BFF booted cleanly.
    """

    setup_logging(level=os.environ.get("WANG_LOG_LEVEL", "INFO"))
    _logger.info(
        "bff_startup",
        extra={
            "stage": "lifespan",
            "db_pool": "stub (deferred to later sprint)",
            "model_registry": "stub (deferred to later sprint)",
            "otel_enabled": bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")),
        },
    )
    try:
        yield
    finally:
        _logger.info("bff_shutdown", extra={"stage": "lifespan"})


app = FastAPI(
    title="Wang Trading BFF",
    version="2.0",
    description="Read-only operator cockpit API. See docs/web_app_design_v2.md.",
    lifespan=_lifespan,
)

# CORS — permissive in dev; production deploys override via
# config/web.yaml (handled by reverse proxy + nginx in §17).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[_REQUEST_ID_HEADER],
)


# ----------------------------------------------------------------------
# Middlewares
# ----------------------------------------------------------------------


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """Read or generate X-Request-Id and attach to logging context + headers."""

    incoming = request.headers.get(_REQUEST_ID_HEADER)
    request_id = incoming or uuid.uuid4().hex
    request.state.request_id = request_id

    route = f"{request.method} {request.url.path}"
    tokens = bind_request_context(request_id=request_id, route=route)

    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        status = getattr(response, "status_code", None)
        set_response_context(status=status, duration_ms=duration_ms)
        _logger.info(
            "http_request",
            extra={"status": status, "duration_ms": round(duration_ms, 3)},
        )
        if response is not None:
            response.headers.setdefault(_REQUEST_ID_HEADER, request_id)
        reset_request_context(tokens)


# Metrics middleware (counter + histogram). Mounted after request-id so
# the request_id is already bound on the contextvar when metrics fires.
instrument(app)


# ----------------------------------------------------------------------
# Exception handlers (errors.py owns the full taxonomy)
# ----------------------------------------------------------------------

register_exception_handlers(app)


# ----------------------------------------------------------------------
# Health & metrics
# ----------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    return envelope(
        {"ok": True, "now": datetime.now(timezone.utc).isoformat()},
        source="health",
    )


# Prometheus exposition. We expose the metrics two ways:
#
# 1. A direct `@app.get("/metrics")` route that serves the Prometheus
#    text body. This avoids the 307 redirect FastAPI emits when a
#    `mount()` is hit on its bare prefix without a trailing slash.
# 2. An ASGI `mount()` of `metrics_endpoint()` at `/metrics`, mirroring
#    the design doc's wording (§10.3) and giving sub-route flexibility
#    for future expansion.
#
# The `instrument()` middleware skips any path starting with `/metrics`
# so scrapes don't appear in `bff_requests_total`.
def _prometheus_response() -> Response:
    body = generate_latest(REGISTRY)
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics", include_in_schema=False)
def _metrics_get() -> Response:
    return _prometheus_response()


app.mount("/metrics", metrics_endpoint())


# ----------------------------------------------------------------------
# Conditional OpenTelemetry (decision §24.4)
# ----------------------------------------------------------------------

if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        _logger.info(
            "otel_enabled",
            extra={"endpoint": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")},
        )
    except Exception as exc:  # pragma: no cover - depends on optional pkg
        _logger.warning(
            "otel_setup_failed",
            extra={"error": str(exc)},
        )


# ----------------------------------------------------------------------
# v1 routes (Phase 1)
# ----------------------------------------------------------------------

app.include_router(overview.router, prefix="/api/v1")
app.include_router(trade_ideas.router, prefix="/api/v1")

# v2 routes (new pages)
app.include_router(scenarios.router, prefix="/api/v1")
app.include_router(replay.router, prefix="/api/v1")
app.include_router(preflight.router, prefix="/api/v1")
