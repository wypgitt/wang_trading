"""Prometheus self-metrics for the BFF (backend_design §10.3).

Exposes a ``/metrics`` ASGI endpoint and an ``instrument(app)`` helper
that wires a per-request middleware to record ``bff_requests_total`` and
``bff_request_duration_seconds``. SSE-related counters and the cache /
upstream-error counters are exported for direct use by the services
that own them.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Bucket layout from backend_design §10.3 row 2.
_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def _get_or_create_counter(
    name: str,
    documentation: str,
    labelnames: tuple[str, ...],
    *,
    registry: CollectorRegistry = REGISTRY,
) -> Counter:
    # Re-import safety: if the module is reloaded (pytest does this),
    # prometheus_client raises ValueError on duplicate registration. We
    # look up the existing collector and reuse it instead.
    existing = getattr(registry, "_names_to_collectors", {}).get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return Counter(name, documentation, labelnames, registry=registry)


def _get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: tuple[str, ...],
    buckets: tuple[float, ...],
    *,
    registry: CollectorRegistry = REGISTRY,
) -> Histogram:
    existing = getattr(registry, "_names_to_collectors", {}).get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return Histogram(name, documentation, labelnames, buckets=buckets, registry=registry)


def _get_or_create_gauge(
    name: str,
    documentation: str,
    labelnames: tuple[str, ...],
    *,
    registry: CollectorRegistry = REGISTRY,
) -> Gauge:
    existing = getattr(registry, "_names_to_collectors", {}).get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return Gauge(name, documentation, labelnames, registry=registry)


# ---- Public metric handles ---------------------------------------------

bff_requests_total: Counter = _get_or_create_counter(
    "bff_requests_total",
    "Total HTTP requests served by the BFF.",
    ("route", "status"),
)

bff_request_duration_seconds: Histogram = _get_or_create_histogram(
    "bff_request_duration_seconds",
    "Latency of BFF HTTP requests in seconds.",
    ("route",),
    _DURATION_BUCKETS,
)

bff_sse_active_connections: Gauge = _get_or_create_gauge(
    "bff_sse_active_connections",
    "Currently-open SSE connections, by stream.",
    ("stream",),
)

bff_sse_events_published_total: Counter = _get_or_create_counter(
    "bff_sse_events_published_total",
    "SSE events emitted, by stream and event type.",
    ("stream", "event"),
)

bff_cache_hits_total: Counter = _get_or_create_counter(
    "bff_cache_hits_total",
    "BFF in-process cache hits.",
    ("cache",),
)

bff_cache_misses_total: Counter = _get_or_create_counter(
    "bff_cache_misses_total",
    "BFF in-process cache misses.",
    ("cache",),
)

bff_upstream_errors_total: Counter = _get_or_create_counter(
    "bff_upstream_errors_total",
    "Errors raised by upstream services (DB, broker, mlflow, ...).",
    ("service", "code"),
)

# Freshness SLO gauges (the engine->BFF data-flow health). Set by the health
# check; -1 encodes "unavailable" so a flat-line at -1 alerts on a dead feed.
bff_snapshot_age_seconds: Gauge = _get_or_create_gauge(
    "bff_snapshot_age_seconds",
    "Age of the trade-ideas snapshot in seconds (engine->BFF freshness). -1 = no snapshot.",
    (),
)

bff_last_bar_age_seconds: Gauge = _get_or_create_gauge(
    "bff_last_bar_age_seconds",
    "Age of the most recent bar in seconds. -1 = unavailable (ingestion down / empty).",
    (),
)

bff_model_loaded: Gauge = _get_or_create_gauge(
    "bff_model_loaded",
    "1 if a production model is registered/loaded, else 0.",
    (),
)


# ---- Route normalisation ----------------------------------------------


def _normalise_route(scope: dict[str, Any]) -> str:
    """Prefer the matched route template; fall back to raw path.

    Using the template (``/api/v1/trade-ideas/{symbol}``) keeps label
    cardinality bounded — otherwise every unique symbol blows up the
    metrics surface.
    """

    route = scope.get("route")
    if route is not None:
        path = getattr(route, "path", None)
        if path:
            return f"{scope.get('method', 'GET')} {path}"
    return f"{scope.get('method', 'GET')} {scope.get('path', '?')}"


# ---- ASGI metrics endpoint --------------------------------------------


def metrics_endpoint() -> Callable[..., Awaitable[None]]:
    """Return an ASGI-compatible callable serving Prometheus text format.

    Mount via ``app.mount('/metrics', metrics_endpoint())``.
    """

    async def _asgi(scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            # Defensive: only accept HTTP. lifespan/websocket get an empty 404.
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return

        body = generate_latest(REGISTRY)
        headers = [
            (b"content-type", CONTENT_TYPE_LATEST.encode("latin-1")),
            (b"content-length", str(len(body)).encode("latin-1")),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    return _asgi


# ---- App instrumentation ----------------------------------------------


def instrument(app: Any) -> None:
    """Attach a request-counter + latency-histogram middleware to ``app``.

    Skips the ``/metrics`` mount itself so scrapers don't show up in the
    request count.
    """

    @app.middleware("http")
    async def _metrics_middleware(request: Any, call_next: Callable) -> Any:
        # Skip the metrics endpoint to keep scrapes out of the histogram.
        if request.url.path.startswith("/metrics"):
            return await call_next(request)

        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            route_label = _normalise_route(request.scope)
            bff_request_duration_seconds.labels(route=route_label).observe(duration)
            bff_requests_total.labels(route=route_label, status=str(status)).inc()


__all__ = [
    "bff_cache_hits_total",
    "bff_cache_misses_total",
    "bff_last_bar_age_seconds",
    "bff_model_loaded",
    "bff_request_duration_seconds",
    "bff_requests_total",
    "bff_snapshot_age_seconds",
    "bff_sse_active_connections",
    "bff_sse_events_published_total",
    "bff_upstream_errors_total",
    "instrument",
    "metrics_endpoint",
]
