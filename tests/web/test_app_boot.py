"""Boot-time smoke tests for the FastAPI app."""

from __future__ import annotations


def test_healthz_returns_envelope(client) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "health"
    assert "as_of" in body
    assert body["data"]["ok"] is True
    # Request-id header round-trips/generates.
    assert r.headers.get("x-request-id") is not None
    assert len(r.headers["x-request-id"]) >= 8


def test_request_id_passthrough(client) -> None:
    r = client.get("/healthz", headers={"X-Request-Id": "test-fixed-123"})
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == "test-fixed-123"


def test_metrics_endpoint_prometheus_format(client) -> None:
    # Trigger one request so the counter has a sample.
    client.get("/healthz")

    r = client.get("/metrics")
    assert r.status_code == 200
    content_type = r.headers["content-type"]
    # prometheus_client emits "text/plain; version=0.0.4; charset=utf-8"
    assert content_type.startswith("text/plain")
    body = r.text
    # Each declared metric (or its _total / _bucket variant) appears in
    # the exposition format. We don't assert on specific labels because
    # the histogram emits multiple lines per route.
    assert "bff_requests_total" in body
    assert "bff_request_duration_seconds" in body
    # Comment lines per OpenMetrics conventions.
    assert "# HELP" in body
    assert "# TYPE" in body


def test_unknown_route_404_uses_envelope_or_default(client) -> None:
    # FastAPI's default 404 isn't enveloped (route never matched), which
    # is fine — the goal here is to assert the app doesn't 500 on miss.
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404


def test_v1_routes_mounted(client) -> None:
    # Routes exist (200 if stub returns; the point is they're mounted).
    for path in [
        "/api/v1/overview",
        "/api/v1/trade-ideas",
        "/api/v1/preflight",
    ]:
        r = client.get(path)
        # Stubs may return 200 with degraded data or 503; only structural
        # routing failures (404 from unmounted prefix) would be a bug.
        assert r.status_code != 404, f"{path} returned 404 — route not mounted"
