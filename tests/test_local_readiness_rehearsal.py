"""Unit checks for the local readiness rehearsal helpers."""

from __future__ import annotations

import urllib.request

from scripts.local_readiness_rehearsal import _local_http_probe_server


def test_local_http_probe_server_returns_200():
    with _local_http_probe_server() as url:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            assert response.status == 200
            assert response.read().decode("utf-8").strip() == "ok"
