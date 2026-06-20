"""Tests for ``/signals/families`` and ``/signals/family-{id}``.

Verifies the family roster renders all 10 families (camelCase, metadata
present, performance fields absent/null), derives ``live``/``shadow`` status
from the battery dispatch kind, counts/joins activity from a seeded
trade-ideas snapshot, and 404s an unknown family. No engine/DB/MLflow
needed: a fake ``battery_factory`` supplies the status kinds and a tmpfs
snapshot supplies the ideas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.web.app import app
from src.web.deps import get_signals_service
from src.web.reference import FAMILY_META
from src.web.services.signals_service import SignalsService
from src.web.services.trade_ideas_service import TradeIdeasService


@pytest.fixture(autouse=True)
def _clear_overrides():
    # Guarantee override cleanup even if an assertion fails mid-test, so a
    # leaked get_signals_service override cannot corrupt later tests.
    yield
    app.dependency_overrides.clear()


def test_real_battery_family_names_match_family_meta():
    # Pin the load-bearing contract: status resolution looks up by family id,
    # so every real Signal's .name must equal its FAMILY_META key. No infra.
    from src.web.services.signals_service import create_default_battery

    battery = create_default_battery()
    assert {r.name for r in battery._registry} == set(FAMILY_META.keys())


# ── fakes / fixtures ──────────────────────────────────────────────────


class _FakeRegistration:
    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind


class _FakeBattery:
    """Mimics ``SignalBattery._registry`` — only ``.name``/``.kind`` read."""

    def __init__(self, kinds: dict[str, str]) -> None:
        self._registry = [_FakeRegistration(n, k) for n, k in kinds.items()]


# kind == 'bars' -> live; everything else -> shadow. Mirrors the real
# create_default_battery split: the four bars-runnable families are live,
# the context-only families (carry/arb/panel/vol) are shadow.
_FAKE_KINDS = {
    "ts_momentum": "bars",
    "cs_momentum": "panel",
    "mean_reversion": "bars",
    "ma_crossover": "bars",
    "donchian_breakout": "bars",
    "vrp": "bars_extra",
    "futures_carry": "bars_extra",
    "funding_rate_arb": "bars_extra",
    "stat_arb": "pair",
    "cross_exchange_arb": "exchange_prices",
}


def _fake_battery_factory():
    return _FakeBattery(_FAKE_KINDS)


def _idea(symbol, *, strategy=None, top_family=None, action="BUY", weight=0.05):
    return {
        "symbol": symbol,
        "action": action,
        "target_weight": weight,
        "target_notional": weight * 1_000_000,
        "latest_price": 100.0,
        "bars_loaded": 500,
        "feature_rows": 480,
        "signal_count": 2,
        "top_signal_family": top_family,
        "top_signal_side": 1,
        "strategy": strategy,
        "reason": "test idea",
        "stage_latency_seconds": {},
        "errors": [],
    }


def _seed_snapshot(tmp_path, ideas):
    path = tmp_path / "trade_ideas.json"
    report = {"idea_count": len(ideas), "totals": {}, "ideas": ideas}
    payload = {"as_of": datetime.now(timezone.utc).isoformat(), "report": report}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _override_service(snapshot_path, *, battery_factory=_fake_battery_factory):
    """Point the provider at a SignalsService reading ``snapshot_path``."""

    ideas_service = TradeIdeasService(tmpfs_path=str(snapshot_path))
    service = SignalsService(ideas=ideas_service, battery_factory=battery_factory)
    app.dependency_overrides[get_signals_service] = lambda: service


# ── /signals/families ─────────────────────────────────────────────────


def test_families_lists_all_ten(client, tmp_path):
    snapshot = _seed_snapshot(tmp_path, [])
    _override_service(snapshot)

    resp = client.get("/api/v1/signals/families")
    assert resp.status_code == 200
    body = resp.json()
    cards = body["data"]

    assert len(cards) == len(FAMILY_META) == 10
    # Declaration order preserved.
    assert [c["id"] for c in cards] == list(FAMILY_META.keys())

    app.dependency_overrides.clear()


def test_families_camelcase_and_metadata_present(client, tmp_path):
    snapshot = _seed_snapshot(tmp_path, [])
    _override_service(snapshot)

    resp = client.get("/api/v1/signals/families")
    cards = {c["id"]: c for c in resp.json()["data"]}
    ts = cards["ts_momentum"]

    # Reference metadata is real and present.
    assert ts["name"] == "Time-Series Momentum"
    assert ts["category"] == "Momentum"
    assert ts["source"] == "Clenow · Chan"
    assert "momentum" in ts["thesis"].lower()
    assert ts["assetClasses"] == ["equity", "crypto", "future"]

    # Params carried through as {key, value} pairs.
    assert ts["params"]
    assert ts["params"][0]["key"] == "lookbacks"
    assert all({"key", "value"} <= set(p) for p in ts["params"])

    app.dependency_overrides.clear()


def test_families_status_live_vs_shadow(client, tmp_path):
    snapshot = _seed_snapshot(tmp_path, [])
    _override_service(snapshot)

    cards = {c["id"]: c for c in client.get("/api/v1/signals/families").json()["data"]}

    # bars-runnable -> live.
    assert cards["ts_momentum"]["status"] == "live"
    assert cards["mean_reversion"]["status"] == "live"
    # context-only families -> shadow (no panel/pair/exchange caller).
    assert cards["stat_arb"]["status"] == "shadow"
    assert cards["cross_exchange_arb"]["status"] == "shadow"
    assert cards["cs_momentum"]["status"] == "shadow"

    app.dependency_overrides.clear()


def test_families_performance_fields_absent(client, tmp_path):
    snapshot = _seed_snapshot(tmp_path, [])
    _override_service(snapshot)

    ts = {c["id"]: c for c in client.get("/api/v1/signals/families").json()["data"]}[
        "ts_momentum"
    ]

    # COMING fields are omitted (exclude_none) or empty — never synthesised.
    for field in (
        "sharpe",
        "winRate",
        "trades",
        "contributionPct",
        "pnlYtd",
        "allocation",
        "avgHoldBars",
    ):
        assert ts.get(field) is None, field
    assert ts.get("regimeFit", {}) == {}
    assert ts.get("equityCurve", []) == []

    app.dependency_overrides.clear()


def test_families_active_signals_counted_from_snapshot(client, tmp_path):
    ideas = [
        _idea("NVDA", strategy="ts_momentum"),
        _idea("AAPL", top_family="ts_momentum"),
        _idea("SPY", strategy="mean_reversion"),
    ]
    snapshot = _seed_snapshot(tmp_path, ideas)
    _override_service(snapshot)

    cards = {c["id"]: c for c in client.get("/api/v1/signals/families").json()["data"]}
    assert cards["ts_momentum"]["activeSignals"] == 2
    assert cards["mean_reversion"]["activeSignals"] == 1
    assert cards["stat_arb"]["activeSignals"] == 0

    app.dependency_overrides.clear()


# ── /signals/family-{id} ──────────────────────────────────────────────


def test_family_detail_filters_ideas(client, tmp_path):
    ideas = [
        _idea("NVDA", strategy="ts_momentum"),
        _idea("AAPL", top_family="ts_momentum"),
        _idea("SPY", strategy="mean_reversion"),
    ]
    snapshot = _seed_snapshot(tmp_path, ideas)
    _override_service(snapshot)

    resp = client.get("/api/v1/signals/family-ts_momentum")
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["strategy"]["id"] == "ts_momentum"
    assert data["strategy"]["status"] == "live"
    symbols = sorted(i["symbol"] for i in data["ideas"])
    assert symbols == ["AAPL", "NVDA"]

    app.dependency_overrides.clear()


def test_family_detail_unknown_404(client, tmp_path):
    snapshot = _seed_snapshot(tmp_path, [])
    _override_service(snapshot)

    resp = client.get("/api/v1/signals/family-does_not_exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["errors"]
    assert body["errors"][0]["code"] == "NOT_FOUND"

    app.dependency_overrides.clear()


def test_families_status_falls_back_when_battery_unavailable(client, tmp_path):
    snapshot = _seed_snapshot(tmp_path, [])

    def _boom():
        raise RuntimeError("battery import failed")

    _override_service(snapshot, battery_factory=_boom)

    cards = {c["id"]: c for c in client.get("/api/v1/signals/families").json()["data"]}
    # Static fallback allow-list: these four stay live, the rest shadow.
    assert cards["ts_momentum"]["status"] == "live"
    assert cards["ma_crossover"]["status"] == "live"
    assert cards["donchian_breakout"]["status"] == "live"
    assert cards["mean_reversion"]["status"] == "live"
    assert cards["stat_arb"]["status"] == "shadow"
    assert cards["vrp"]["status"] == "shadow"

    app.dependency_overrides.clear()
