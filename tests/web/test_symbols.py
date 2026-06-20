"""Tests for ``GET /api/v1/symbols/{symbol}`` — single-instrument detail.

Verifies the route builds a real OHLCV candle series + latest-bar
microstructure from injected bars rows, joins a trade idea from a seeded
snapshot, degrades (``dataAvailable=False``) when the bars table is
unreachable, and 404s an unknown symbol with an enveloped ``NOT_FOUND``
error. No DB/MLflow needed: the gateway ``fetch`` is injected and the
snapshot is seeded via ``WANG_TRADE_IDEAS_PATH``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.web.app import app
from src.web.deps import get_symbols_service
from src.web.services.bars_gateway import BarsGateway
from src.web.services.symbols_service import SymbolsService
from src.web.services.trade_ideas_service import TradeIdeasService


def _bar_row(i, close, *, bar_type="tib"):
    return {
        "timestamp": datetime(2026, 6, 19, 14, i % 60, tzinfo=timezone.utc).isoformat(),
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": 1000.0 + i,
        "dollar_volume": (1000.0 + i) * close,
        "tick_count": 50 + i,
        "buy_volume": 600.0 + i,
        "sell_volume": 400.0,
        "buy_ticks": 30,
        "sell_ticks": 20,
        "imbalance": 0.2,
        "threshold": 0.5,
        "vwap": close + 0.1,
        "bar_duration_seconds": 12.0,
    }


def _candle_rows(n=130, *, bar_type="tib"):
    # oldest-first, as the gateway returns.
    return [_bar_row(i, 100.0 + i, bar_type=bar_type) for i in range(n)]


def _override(service: SymbolsService):
    app.dependency_overrides[get_symbols_service] = lambda: service


def _clear_override():
    app.dependency_overrides.pop(get_symbols_service, None)


def _idea(symbol):
    return {
        "symbol": symbol,
        "action": "BUY",
        "target_weight": 0.075,
        "target_notional": 75_000.0,
        "estimated_quantity": None,
        "latest_price": 229.0,
        "latest_bar_at": None,
        "bars_loaded": 130,
        "feature_rows": 120,
        "signal_count": 2,
        "top_signal_family": "ts_momentum",
        "top_signal_side": 1,
        "top_signal_confidence": 0.7,
        "avg_signal_confidence": 0.6,
        "meta_probability": 0.66,
        "calibrated_probability": 0.61,
        "bet_size": 0.075,
        "strategy": "ts_momentum",
        "reason": "test idea",
        "stage_latency_seconds": {},
        "errors": [],
    }


def _seed_snapshot(tmp_path, monkeypatch, ideas):
    path = tmp_path / "trade_ideas.json"
    totals = {
        "buy": 1, "sell": 0, "watch": 0,
        "model_required": 0, "no_data": 0, "error": 0,
        "gross_target_weight": 0.075, "net_target_weight": 0.075,
    }
    report = {"idea_count": len(ideas), "totals": totals, "ideas": ideas}
    payload = {"as_of": datetime.now(timezone.utc).isoformat(), "report": report}
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("WANG_TRADE_IDEAS_PATH", str(path))
    return path


def test_symbol_detail_real_ohlc_and_idea(client, tmp_path, monkeypatch):
    rows = _candle_rows(130)

    def _fetch(symbol, bar_type, limit):
        assert symbol == "NVDA"
        assert bar_type == "tib"
        return rows[-limit:]

    seed_path = _seed_snapshot(tmp_path, monkeypatch, [_idea("NVDA")])
    gateway = BarsGateway(fetch=_fetch)
    ideas = TradeIdeasService(tmpfs_path=str(seed_path))
    _override(SymbolsService(gateway=gateway, ideas=ideas, candle_limit=130))
    try:
        resp = client.get("/api/v1/symbols/NVDA")
    finally:
        _clear_override()

    assert resp.status_code == 200
    body = resp.json()
    assert body["errors"] == []
    sym = body["data"]["sym"]

    # Static reference fields.
    assert sym["symbol"] == "NVDA"
    assert sym["name"] == "NVIDIA Corp."
    assert sym["type"] == "equity"
    assert sym["barType"] == "tib"
    assert sym["dataAvailable"] is True
    assert sym["barsLoaded"] == 130

    # Real OHLC candles in camelCase (Candle uses t/o/h/l/c/v keys).
    candles = sym["candles"]
    assert len(candles) == 130
    first = candles[0]
    assert set(first) == {"t", "o", "h", "l", "c", "v"}
    assert first["t"] == 0
    assert first["c"] == 100.0
    last = candles[-1]
    assert last["c"] == 100.0 + 129
    assert sym["price"] == 100.0 + 129

    # Close-line + spark mirror the closes.
    assert sym["line"][0] == {"t": 0, "v": 100.0}
    assert sym["spark"][-1] == 100.0 + 129

    # Latest-bar microstructure carries barType.
    assert sym["bar"]["barType"] == "tib"
    assert sym["bar"]["vwap"] == (100.0 + 129) + 0.1

    # market_cap has no source -> omitted (exclude_none) -> null on client.
    assert "marketCap" not in sym

    # Joined idea from the seeded snapshot.
    assert body["data"]["idea"]["symbol"] == "NVDA"
    assert body["data"]["idea"]["action"] == "BUY"


def test_symbol_detail_db_down_degrades(client, tmp_path, monkeypatch):
    # Gateway returns None (DB unreachable) -> dataAvailable False, no 500.
    gateway = BarsGateway(fetch=lambda s, bt, n: None)
    _override(SymbolsService(gateway=gateway, ideas=TradeIdeasService(), candle_limit=130))
    try:
        resp = client.get("/api/v1/symbols/AAPL")
    finally:
        _clear_override()

    assert resp.status_code == 200
    body = resp.json()
    sym = body["data"]["sym"]
    assert sym["dataAvailable"] is False
    assert sym["barsLoaded"] == 0
    assert sym["candles"] == []
    assert sym["spark"] == []
    assert "price" not in sym  # null -> omitted
    # Honest freshness signal, consistent with /markets' DB-down warning.
    assert any("bars database unavailable" in w for w in body["warnings"])


def test_symbol_detail_no_rows(client, tmp_path, monkeypatch):
    # Table reachable but empty for this symbol -> dataAvailable True, 0 bars.
    gateway = BarsGateway(fetch=lambda s, bt, n: [])
    _override(SymbolsService(gateway=gateway, ideas=TradeIdeasService(), candle_limit=130))
    try:
        resp = client.get("/api/v1/symbols/MSFT")
    finally:
        _clear_override()

    assert resp.status_code == 200
    sym = resp.json()["data"]["sym"]
    assert sym["dataAvailable"] is True
    assert sym["barsLoaded"] == 0
    assert sym["candles"] == []
    assert "price" not in sym


def test_symbol_detail_unknown_symbol_404(client):
    resp = client.get("/api/v1/symbols/ZZZZ")
    assert resp.status_code == 404
    body = resp.json()
    assert body["data"] is None
    assert any(e["code"] == "NOT_FOUND" for e in body["errors"])
