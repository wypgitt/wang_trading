"""Shared fixtures for the BFF test suite.

The starlette ``TestClient`` requires ``httpx`` which is in
``requirements.txt`` but may not be installed in the local dev shell. To
keep the suite runnable everywhere, we expose a ``client`` fixture that
uses :class:`fastapi.testclient.TestClient` when httpx is available and
falls back to an in-process ASGI driver otherwise. Both expose the same
``.get/.post/.headers/.status_code/.json/.text`` surface used by the
tests.
"""

from __future__ import annotations

import asyncio
import json as _json
from typing import Any
from urllib.parse import urlencode

import pytest

from src.web.app import app
from src.web.deps import reset_service_singletons


class _ASGIResponse:
    __slots__ = ("status_code", "headers", "_body")

    def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return _json.loads(self._body.decode("utf-8"))


class _ASGIClient:
    """Tiny ASGI driver that fakes just enough to test our routes.

    Does *not* drive the lifespan protocol — pytest fixture scope handles
    setup/teardown by importing the app (lifespan only runs under uvicorn
    or TestClient). This is fine for our tests; they don't depend on
    lifespan-initialised state.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    def get(self, path: str, *, headers: dict[str, str] | None = None) -> _ASGIResponse:
        return self._request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        *,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> _ASGIResponse:
        body = b""
        merged_headers = dict(headers or {})
        if json is not None:
            body = _json.dumps(json).encode("utf-8")
            merged_headers.setdefault("content-type", "application/json")
            merged_headers["content-length"] = str(len(body))
        return self._request("POST", path, headers=merged_headers, body=body)

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> _ASGIResponse:
        if "?" in path:
            raw_path, query = path.split("?", 1)
        else:
            raw_path, query = path, ""

        request_headers = []
        for k, v in (headers or {}).items():
            request_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": raw_path,
            "raw_path": raw_path.encode("utf-8"),
            "query_string": query.encode("latin-1"),
            "root_path": "",
            "headers": request_headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "extensions": {},
        }

        sent: dict[str, Any] = {"status": 500, "headers": {}, "body": bytearray()}
        body_sent = {"sent": False}

        async def receive() -> dict[str, Any]:
            if body_sent["sent"]:
                return {"type": "http.disconnect"}
            body_sent["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                sent["status"] = message["status"]
                sent["headers"] = {
                    k.decode("latin-1").lower(): v.decode("latin-1")
                    for k, v in message.get("headers", [])
                }
            elif message["type"] == "http.response.body":
                sent["body"].extend(message.get("body", b""))

        async def _drive() -> None:
            # starlette.ServerErrorMiddleware emits the 500 response and
            # then re-raises so test clients can see the original error.
            # Mimic Starlette's TestClient(raise_server_exceptions=False)
            # behaviour: swallow the re-raise once the response was sent.
            try:
                await self._app(scope, receive, send)
            except Exception:
                if not sent["body"] and sent["status"] == 500:
                    raise
                # else: the handler already wrote the response, ignore.

        asyncio.run(_drive())
        return _ASGIResponse(sent["status"], sent["headers"], bytes(sent["body"]))


def _make_client():
    try:
        from fastapi.testclient import TestClient  # noqa: WPS433

        return TestClient(app)
    except (ImportError, RuntimeError):
        return _ASGIClient(app)


@pytest.fixture()
def client():
    """Yield a TestClient (preferred) or an in-process ASGI fallback."""

    reset_service_singletons()
    c = _make_client()
    # Real TestClient supports context-manager semantics; ours doesn't.
    if hasattr(c, "__enter__"):
        with c as ctx:
            yield ctx
    else:
        yield c
    reset_service_singletons()
