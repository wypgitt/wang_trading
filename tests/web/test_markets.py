"""Tests for ``GET /api/v1/markets`` — the markets grid (bare array).

Verifies the route shapes one honestly-partial row per instrument from the
bars hypertable, joins the trade-ideas snapshot, holds ``marketCap`` null
(no producer), and degrades — never 500s — when the bars DB is unreachable.
No live infra: the bars gateway is injected with a ``fetch`` fake and the
snapshot is seeded via ``WANG_TRADE_IDEAS_PATH``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.web.app import app
from src.web.deps import get_markets_service
from src.web.services.bars_gateway import BarsGateway
from src.web.services.markets_service import MarketsService
from src.web.services.trade_ideas_service import TradeIdeasService


# Symbols with real bars; one symbol returns [] (no rows); one returns None
# (DB unreachable). Everything else falls through to None as well.
_WITH_BARS = {"NVDA", "BTC"}
_EMPTY = "AAPL"
_DOWN = "MSFT"


def _bar_row(close, *, ts, volume=1000.0):
    return {
        "timestamp": ts,
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 2.0,
        "close": close,
        "volume": volume,
        "dollar_volume": close * volume,
        "tick_count": 50,
        "buy_volume": 600.0,
        "sell_volume": 400.0,
        "buy_ticks": 30,
        "sell_ticks": 20,
        "imbalance": 0.2,
        "threshold": 100.0,
        "vwap": close,
        "bar_duration_seconds": 12.0,
    }


def _fake_fetch(symbol, bar_type, limit):
    """Return a few rows for some symbols, [] for one, None for one."""

    if symbol == _DOWN:
        return None
    if symbol == _EMPTY:
        return []
    if symbol in _WITH_BARS:
        base = datetime(2026, 6, 19, 14, 0, tzinfo=timezone.utc)
        # Rising closes so change_window_pct is positive and well-defined.
        return [
            _bar_row(100.0, ts=base.isoformat()),
            _bar_row(102.0, ts=base.isoformat()),
            _bar_row(110.0, ts=base.isoformat()),
        ]
    return None


def _seed_snapshot(tmp_path, monkeypatch, *, symbols_with_ideas):
    ideas = []
    for sym in symbols_with_ideas:
        ideas.append(
            {
                "symbol": sym,
                "action": "BUY",
                "target_weight": 0.05,
                "target_notional": 50_000.0,
                "reason": "test idea",
                "stage_latency_seconds": {},
                "errors": [],
            }
        )
    report = {
        "idea_count": len(ideas),
        "totals": {
            "buy": len(ideas), "sell": 0, "watch": 0,
            "model_required": 0, "no_data": 0, "error": 0,
            "gross_target_weight": 0.0, "net_target_weight": 0.0,
        },
        "ideas": ideas,
    }
    path = tmp_path / "trade_ideas.json"
    payload = {"as_of": datetime.now(timezone.utc).isoformat(), "report": report}
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(path))


def _install_service(tmp_path, monkeypatch, *, symbols_with_ideas=("NVDA",)):
    _seed_snapshot(tmp_path, monkeypatch, symbols_with_ideas=symbols_with_ideas)
    service = MarketsService(
        gateway=BarsGateway(fetch=_fake_fetch),
        ideas=TradeIdeasService(tmpfs_path=str(tmp_path / "trade_ideas.json")),
        spark_limit=60,
    )
    app.dependency_overrides[get_markets_service] = lambda: service
    return service


def _rows_by_symbol(data):
    return {row["symbol"]: row for row in data}


def test_markets_returns_enveloped_array(client, tmp_path, monkeypatch):
    _install_service(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/v1/markets")
    finally:
        app.dependency_overrides.pop(get_markets_service, None)

    assert resp.status_code == 200
    body = resp.json()
    # Envelope shape.
    assert "data" in body and "warnings" in body and "as_of" in body
    assert body["source"] == "markets_service"
    # Bare array, one row per instrument in the universe.
    data = body["data"]
    assert isinstance(data, list)
    from src.web.reference import all_symbols

    assert len(data) == len(all_symbols())


def test_markets_row_with_bars_is_camelcase_and_omits_market_cap(
    client, tmp_path, monkeypatch
):
    _install_service(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/v1/markets")
    finally:
        app.dependency_overrides.pop(get_markets_service, None)

    rows = _rows_by_symbol(resp.json()["data"])
    nvda = rows["NVDA"]

    # Bar-derived, camelCased.
    assert nvda["dataAvailable"] is True
    assert nvda["price"] == 110.0
    assert nvda["spark"] == [100.0, 102.0, 110.0]
    assert nvda["barsLoaded"] == 3
    assert nvda["volume"] == 1000.0
    assert "latestBarAt" in nvda
    # change_window_pct = (110 - 100) / 100
    assert round(nvda["changeWindowPct"], 6) == 0.1

    # Nested microstructure, camelCase keys.
    bar = nvda["bar"]
    assert bar["barType"] == "tib"  # NVDA is an equity -> tib
    assert bar["volumeImbalance"] == 200.0  # 600 - 400
    assert round(bar["tickImbalanceRatio"], 6) == round((30 - 20) / 50, 6)

    # market_cap has no producer -> omitted (exclude_none) on the wire.
    assert "marketCap" not in nvda


def test_markets_crypto_row_uses_dollar_bar_type(client, tmp_path, monkeypatch):
    _install_service(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/v1/markets")
    finally:
        app.dependency_overrides.pop(get_markets_service, None)

    btc = _rows_by_symbol(resp.json()["data"])["BTC"]
    assert btc["dataAvailable"] is True
    assert btc["bar"]["barType"] == "dollar"


def test_markets_empty_table_row_is_available_but_priceless(
    client, tmp_path, monkeypatch
):
    _install_service(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/v1/markets")
    finally:
        app.dependency_overrides.pop(get_markets_service, None)

    aapl = _rows_by_symbol(resp.json()["data"])[_EMPTY]
    # Reachable table, no rows: row renders but price/spark/bar are null/empty.
    assert aapl["dataAvailable"] is True
    assert aapl["barsLoaded"] == 0
    assert aapl.get("price") is None
    assert aapl.get("spark", []) == []
    assert "bar" not in aapl  # null -> omitted


def test_markets_db_down_row_has_data_available_false(client, tmp_path, monkeypatch):
    _install_service(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/v1/markets")
    finally:
        app.dependency_overrides.pop(get_markets_service, None)

    msft = _rows_by_symbol(resp.json()["data"])[_DOWN]
    assert msft["dataAvailable"] is False
    assert msft["barsLoaded"] == 0
    assert msft.get("price") is None


def test_markets_joins_idea_from_snapshot(client, tmp_path, monkeypatch):
    _install_service(tmp_path, monkeypatch, symbols_with_ideas=("NVDA",))
    try:
        resp = client.get("/api/v1/markets")
    finally:
        app.dependency_overrides.pop(get_markets_service, None)

    rows = _rows_by_symbol(resp.json()["data"])
    # NVDA is in the snapshot -> hasIdea true with action/targetWeight.
    assert rows["NVDA"]["hasIdea"] is True
    assert rows["NVDA"]["action"] == "BUY"
    assert rows["NVDA"]["targetWeight"] == 0.05
    # BTC is not in the snapshot -> hasIdea false (default), action omitted.
    assert rows["BTC"]["hasIdea"] is False
    assert "action" not in rows["BTC"]


def test_markets_warns_when_every_row_db_unreachable(client, tmp_path, monkeypatch):
    # A gateway that returns None for every symbol => whole bars DB is down.
    _seed_snapshot(tmp_path, monkeypatch, symbols_with_ideas=())
    service = MarketsService(
        gateway=BarsGateway(fetch=lambda s, bt, n: None),
        ideas=TradeIdeasService(tmpfs_path=str(tmp_path / "trade_ideas.json")),
    )
    app.dependency_overrides[get_markets_service] = lambda: service
    try:
        resp = client.get("/api/v1/markets")
    finally:
        app.dependency_overrides.pop(get_markets_service, None)

    assert resp.status_code == 200
    body = resp.json()
    assert all(row["dataAvailable"] is False for row in body["data"])
    assert any("bars database unavailable" in w for w in body["warnings"])
