"""Tests for the live AlpacaBrokerAdapter (P6.01).

These tests mock the Alpaca SDK entirely — no network calls are made. They
are tagged ``@pytest.mark.live_broker`` so CI can skip them if desired.
"""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.broker_adapter import AlpacaBrokerAdapter, BrokerNetworkError
from src.execution.models import ExecutionAlgo, Order, OrderStatus, OrderType

pytestmark = pytest.mark.live_broker


# ── Shim the alpaca-py SDK so `_build_order_request` can import it ─────────

def _install_alpaca_shim() -> None:
    if "alpaca" in sys.modules and hasattr(sys.modules["alpaca"], "_wang_shim"):
        return

    alpaca_pkg = types.ModuleType("alpaca")
    alpaca_pkg._wang_shim = True  # type: ignore[attr-defined]
    alpaca_pkg.__path__ = []  # mark as package

    trading_pkg = types.ModuleType("alpaca.trading")
    trading_pkg.__path__ = []
    data_pkg = types.ModuleType("alpaca.data")
    data_pkg.__path__ = []

    # ── enums ─────────────────────────────────────────────────────────
    enums_mod = types.ModuleType("alpaca.trading.enums")

    class _Enum:
        def __init__(self, val: str) -> None:
            self.value = val

        def __repr__(self) -> str:  # pragma: no cover
            return f"_Enum({self.value!r})"

    class OrderSide:
        BUY = _Enum("buy")
        SELL = _Enum("sell")

    class TimeInForce:
        DAY = _Enum("day")
        GTC = _Enum("gtc")
        IOC = _Enum("ioc")
        FOK = _Enum("fok")

    enums_mod.OrderSide = OrderSide
    enums_mod.TimeInForce = TimeInForce

    # ── requests ──────────────────────────────────────────────────────
    requests_mod = types.ModuleType("alpaca.trading.requests")

    class _Req:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self._kwargs = kwargs

    class MarketOrderRequest(_Req):
        pass

    class LimitOrderRequest(_Req):
        pass

    class StopLimitOrderRequest(_Req):
        pass

    class TakeProfitRequest(_Req):
        pass

    class StopLossRequest(_Req):
        pass

    requests_mod.MarketOrderRequest = MarketOrderRequest
    requests_mod.LimitOrderRequest = LimitOrderRequest
    requests_mod.StopLimitOrderRequest = StopLimitOrderRequest
    requests_mod.TakeProfitRequest = TakeProfitRequest
    requests_mod.StopLossRequest = StopLossRequest

    # ── trading client ────────────────────────────────────────────────
    client_mod = types.ModuleType("alpaca.trading.client")

    class TradingClient:  # pragma: no cover - replaced by mocks in tests
        def __init__(self, *a, **kw):
            pass

    client_mod.TradingClient = TradingClient

    # ── data ──────────────────────────────────────────────────────────
    data_hist_mod = types.ModuleType("alpaca.data.historical")

    class StockHistoricalDataClient:  # pragma: no cover
        def __init__(self, *a, **kw):
            pass

    data_hist_mod.StockHistoricalDataClient = StockHistoricalDataClient

    data_req_mod = types.ModuleType("alpaca.data.requests")

    class StockLatestQuoteRequest(_Req):
        pass

    data_req_mod.StockLatestQuoteRequest = StockLatestQuoteRequest

    sys.modules["alpaca"] = alpaca_pkg
    sys.modules["alpaca.trading"] = trading_pkg
    sys.modules["alpaca.trading.enums"] = enums_mod
    sys.modules["alpaca.trading.requests"] = requests_mod
    sys.modules["alpaca.trading.client"] = client_mod
    sys.modules["alpaca.data"] = data_pkg
    sys.modules["alpaca.data.historical"] = data_hist_mod
    sys.modules["alpaca.data.requests"] = data_req_mod


_install_alpaca_shim()


# ── Helpers ───────────────────────────────────────────────────────────────

def _order(
    order_type: OrderType = OrderType.MARKET,
    symbol: str = "AAPL",
    side: int = 1,
    qty: float = 10.0,
    limit_price: float | None = None,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    execution_algo: ExecutionAlgo = ExecutionAlgo.IMMEDIATE,
) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=qty * side,
        limit_price=limit_price,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        execution_algo=execution_algo,
    )


def _make_adapter(**kwargs) -> AlpacaBrokerAdapter:
    adapter = AlpacaBrokerAdapter(api_key="k", secret_key="s", paper=True, **kwargs)
    adapter._client = MagicMock()
    adapter._data_client = MagicMock()
    return adapter


# ── Constructor / live gate ───────────────────────────────────────────────

class TestConstructor:
    def test_paper_default_url(self):
        adapter = AlpacaBrokerAdapter(api_key="k", secret_key="s")
        assert adapter.paper is True
        assert adapter.base_url == AlpacaBrokerAdapter.PAPER_URL

    def test_live_blocked_without_env(self, monkeypatch):
        monkeypatch.delenv("WANG_ALLOW_LIVE_TRADING", raising=False)
        with pytest.raises(ValueError, match="Live trading blocked"):
            AlpacaBrokerAdapter(api_key="k", secret_key="s", paper=False)

    def test_live_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("WANG_ALLOW_LIVE_TRADING", "yes")
        adapter = AlpacaBrokerAdapter(api_key="k", secret_key="s", paper=False)
        assert adapter.paper is False
        assert adapter.base_url == AlpacaBrokerAdapter.LIVE_URL

    def test_backwards_compat_api_secret_kwarg(self):
        adapter = AlpacaBrokerAdapter(api_key="k", api_secret="s", paper=True)
        assert adapter.secret_key == "s"


# ── submit_order ──────────────────────────────────────────────────────────

class TestSubmitOrder:
    def test_market_order_populates_response(self):
        adapter = _make_adapter()
        resp = MagicMock()
        resp.id = "alpaca-123"
        resp.status = "new"
        resp.filled_qty = 0
        resp.legs = None
        adapter._client.submit_order.return_value = resp

        order = _order(order_type=OrderType.MARKET)
        result = asyncio.run(adapter.submit_order(order))

        assert result.order_id == "alpaca-123"
        assert result.status == OrderStatus.SUBMITTED
        assert adapter._client.submit_order.call_count == 1
        req = adapter._client.submit_order.call_args.args[0]
        assert req.symbol == "AAPL"
        assert req.qty == 10.0
        assert req.side.value == "buy"

    def test_bracket_order_includes_tp_and_stop(self):
        adapter = _make_adapter()
        resp = MagicMock()
        resp.id = "bracket-1"
        resp.status = "accepted"
        resp.filled_qty = 0
        leg1 = MagicMock(id="tp-leg")
        leg2 = MagicMock(id="sl-leg")
        resp.legs = [leg1, leg2]
        adapter._client.submit_order.return_value = resp

        order = _order(
            order_type=OrderType.BRACKET,
            limit_price=150.0,
            stop_price=140.0,
            take_profit_price=160.0,
        )
        result = asyncio.run(adapter.submit_order(order))

        req = adapter._client.submit_order.call_args.args[0]
        assert req.order_class == "bracket"
        assert req.limit_price == 150.0
        assert req.take_profit.limit_price == 160.0
        assert req.stop_loss.stop_price == 140.0
        assert adapter._bracket_children[result.order_id] == ["tp-leg", "sl-leg"]

    def test_submit_wraps_network_error(self):
        adapter = _make_adapter()
        adapter._client.submit_order.side_effect = RuntimeError("timeout")

        order = _order()
        with pytest.raises(BrokerNetworkError, match="submit_order failed"):
            asyncio.run(adapter.submit_order(order))

    def test_tif_mapping_from_execution_algo(self):
        adapter = _make_adapter()
        resp = MagicMock(id="x", status="new", filled_qty=0, legs=None)
        adapter._client.submit_order.return_value = resp

        order = _order(execution_algo=ExecutionAlgo.TWAP)
        asyncio.run(adapter.submit_order(order))
        req = adapter._client.submit_order.call_args.args[0]
        assert req.time_in_force.value == "day"


# ── cancel / status / account / quote ─────────────────────────────────────

class TestCancelAndStatus:
    def test_cancel_order_calls_sdk(self):
        adapter = _make_adapter()
        adapter._client.cancel_order_by_id.return_value = None

        ok = asyncio.run(adapter.cancel_order("order-1"))

        assert ok is True
        adapter._client.cancel_order_by_id.assert_called_once_with("order-1")

    def test_cancel_wraps_network_error(self):
        adapter = _make_adapter()
        adapter._client.cancel_order_by_id.side_effect = ConnectionError("boom")
        with pytest.raises(BrokerNetworkError):
            asyncio.run(adapter.cancel_order("order-1"))

    def test_get_account(self):
        adapter = _make_adapter()
        acct = MagicMock()
        acct.cash = "1000"
        acct.equity = "2000"
        acct.buying_power = "4000"
        acct.long_market_value = "1500"
        acct.short_market_value = "0"
        acct.pattern_day_trader = False
        adapter._client.get_account.return_value = acct

        out = asyncio.run(adapter.get_account())
        assert out["cash"] == 1000.0
        assert out["nav"] == 2000.0
        assert out["buying_power"] == 4000.0

    def test_heartbeat_true_and_false(self):
        adapter = _make_adapter()
        adapter._client.get_clock.return_value = MagicMock()
        assert asyncio.run(adapter.heartbeat()) is True

        adapter._client.get_clock.side_effect = RuntimeError("down")
        assert asyncio.run(adapter.heartbeat()) is False

    def test_get_quote(self):
        adapter = _make_adapter()
        q = MagicMock(bid_price=99.0, ask_price=101.0)
        adapter._data_client.get_stock_latest_quote.return_value = {"AAPL": q}

        out = asyncio.run(adapter.get_quote("AAPL"))
        assert out["bid"] == 99.0
        assert out["ask"] == 101.0
        assert out["mid"] == 100.0
        assert out["spread"] == pytest.approx(2.0)


# ── poll_fills ────────────────────────────────────────────────────────────

class TestPollFills:
    def test_poll_fills_returns_incremental(self):
        adapter = _make_adapter()
        resp_a = MagicMock(id="o1", symbol="AAPL", qty="10", side="buy",
                           status="partially_filled", filled_qty="5",
                           filled_avg_price="100", legs=None)
        resp_b = MagicMock(id="o1", symbol="AAPL", qty="10", side="buy",
                           status="filled", filled_qty="10",
                           filled_avg_price="100", legs=None)
        adapter._client.get_order_by_id.side_effect = [resp_a, resp_b]

        first = asyncio.run(adapter.poll_fills("o1"))
        second = asyncio.run(adapter.poll_fills("o1"))

        assert len(first) == 1
        assert len(second) == 1
