"""Exception handler tests for the BFF error taxonomy.

For each ``ApiException`` subclass, register a one-off route that raises
the exception, then assert the registered handler converts it into the
expected (status, envelope-error-code) pair.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .conftest import _ASGIClient, _make_client  # type: ignore[attr-defined]

from fastapi import Body  # noqa: E402

from src.web.errors import (
    BadRequest,
    BrokerUnavailable,
    Conflict,
    DbUnavailable,
    DegradedResponse,
    ErrorCode,
    Forbidden,
    ModelUnavailable,
    NotFound,
    RateLimited,
    TimeoutError as ApiTimeoutError,
    Unauthenticated,
    register_exception_handlers,
)


class _Echo(BaseModel):
    n: int = Field(ge=0)


def _build_app() -> FastAPI:
    """Minimal app with just the error handlers wired up."""

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/bad-request")
    def _bad_request():
        raise BadRequest("payload malformed", field="symbol")

    @app.get("/not-found")
    def _not_found():
        raise NotFound("symbol unknown")

    @app.get("/unauth")
    def _unauth():
        raise Unauthenticated("missing session")

    @app.get("/forbidden")
    def _forbidden():
        raise Forbidden("insufficient role")

    @app.get("/conflict")
    def _conflict():
        raise Conflict("idempotency-key mismatch")

    @app.get("/broker-down")
    def _broker():
        raise BrokerUnavailable("alpaca down")

    @app.get("/model-down")
    def _model():
        raise ModelUnavailable("no production model")

    @app.get("/db-down")
    def _db():
        raise DbUnavailable("timescale unreachable")

    @app.get("/timeout")
    def _timeout():
        raise ApiTimeoutError("mlflow timed out")

    @app.get("/rate-limited")
    def _rl():
        raise RateLimited("per-user cap exceeded")

    @app.get("/degraded")
    def _degraded():
        raise DegradedResponse(
            "factor model older than 24h",
            code=ErrorCode.STALE_FACTOR_MODEL,
            data={"placeholder": True},
        )

    @app.get("/stale-model")
    def _stale_model():
        raise DegradedResponse("model fallback used", code=ErrorCode.STALE_MODEL)

    @app.get("/boom")
    def _boom():
        raise RuntimeError("totally unhandled (should be redacted)")

    @app.post("/echo")
    def _echo(body: _Echo = Body(...)):
        return {"n": body.n}

    return app


@pytest.fixture()
def err_client():
    app = _build_app()
    try:
        from fastapi.testclient import TestClient  # noqa: WPS433

        with TestClient(app) as c:
            yield c
    except (ImportError, RuntimeError):
        yield _ASGIClient(app)


def _assert_error(resp, *, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    codes = [e["code"] for e in body.get("errors", [])]
    assert code in codes, f"expected {code} in errors, got {codes}"
    assert "as_of" in body
    assert "source" in body
    return body


def test_bad_request(err_client) -> None:
    body = _assert_error(err_client.get("/bad-request"), status=400, code="BAD_REQUEST")
    assert body["errors"][0]["field"] == "symbol"


def test_not_found(err_client) -> None:
    _assert_error(err_client.get("/not-found"), status=404, code="NOT_FOUND")


def test_unauthenticated(err_client) -> None:
    _assert_error(err_client.get("/unauth"), status=401, code="UNAUTHENTICATED")


def test_forbidden(err_client) -> None:
    _assert_error(err_client.get("/forbidden"), status=403, code="FORBIDDEN")


def test_conflict(err_client) -> None:
    _assert_error(err_client.get("/conflict"), status=409, code="CONFLICT")


def test_broker_unavailable(err_client) -> None:
    _assert_error(err_client.get("/broker-down"), status=503, code="BROKER_UNAVAILABLE")


def test_model_unavailable(err_client) -> None:
    _assert_error(err_client.get("/model-down"), status=503, code="MODEL_UNAVAILABLE")


def test_db_unavailable(err_client) -> None:
    _assert_error(err_client.get("/db-down"), status=503, code="DB_UNAVAILABLE")


def test_timeout(err_client) -> None:
    _assert_error(err_client.get("/timeout"), status=504, code="TIMEOUT")


def test_rate_limited(err_client) -> None:
    _assert_error(err_client.get("/rate-limited"), status=429, code="RATE_LIMITED")


def test_degraded_factor_model_returns_200(err_client) -> None:
    body = _assert_error(
        err_client.get("/degraded"),
        status=200,
        code="STALE_FACTOR_MODEL",
    )
    # Data is still shipped — that's the whole point of a degraded response.
    assert body["data"] == {"placeholder": True}


def test_degraded_stale_model_returns_200(err_client) -> None:
    _assert_error(err_client.get("/stale-model"), status=200, code="STALE_MODEL")


def test_unhandled_exception_redacts_message(err_client) -> None:
    r = err_client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    codes = [e["code"] for e in body["errors"]]
    assert "INTERNAL" in codes
    # Exception text must not leak.
    flat = repr(body).lower()
    assert "totally unhandled" not in flat
    assert "runtimeerror" not in flat
    assert "traceback" not in flat


def test_request_validation_error_returns_422(err_client) -> None:
    r = err_client.post("/echo", json={"n": -1})
    assert r.status_code == 422
    body = r.json()
    codes = [e["code"] for e in body["errors"]]
    assert "VALIDATION_FAILED" in codes
    # Field path is included.
    fields = [e.get("field") for e in body["errors"]]
    assert any(f and "n" in f for f in fields), f"fields={fields}"
