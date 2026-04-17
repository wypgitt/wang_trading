"""Tests for the live CCXTBrokerAdapter (P6.02).

All CCXT calls are mocked — no network is touched. Tests are tagged
``@pytest.mark.live_broker`` so CI can opt out if desired.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.broker_adapter import BrokerNetworkError, CCXTBrokerAdapter
from src.execution.models import Order, OrderStatus, OrderType

pytestmark = pytest.mark.live_broker


# ── Fake CCXT client ──────────────────────────────────────────────────────

class _FakeCCXT:
    """Minimal CCXT-like async client used across tests."""

    def __init__(self, markets: dict | None = None, *, precision_qty: float = 3,
                 precision_price: float = 2) -> None:
        self.markets = markets or {}
        self._precision_qty = precision_qty
        self._precision_price = precision_price
        self.has = {"fetchTicker": True, "fetchStatus": True, "fetchPositions": False}
        # Calls recorded on AsyncMocks
        self.create_order = AsyncMock()
        self.cancel_order = AsyncMock()
        self.fetch_order = AsyncMock()
        self.fetch_balance = AsyncMock()
        self.fetch_ticker = AsyncMock()
        self.fetch_order_book = AsyncMock()
        self.fetch_funding_rate = AsyncMock()
        self.fetch_status = AsyncMock(return_value={"status": "ok"})
        self.fetch_time = AsyncMock(return_value=1_700_000_000_000)
        self.fetch_positions = AsyncMock(return_value=[])
        self.load_markets = AsyncMock(return_value=self.markets)
        self.close = AsyncMock()

    # Precision helpers behave like CCXT's (truncate to N decimals).
    def amount_to_precision(self, symbol, amt):
        return f"{float(amt):.{int(self._precision_qty)}f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.{int(self._precision_price)}f}"


def _adapter(client: _FakeCCXT | None = None, *, exchange_name: str = "binance",
             **kwargs) -> CCXTBrokerAdapter:
    a = CCXTBrokerAdapter(
        exchange_name=exchange_name,
        api_key="k",
        secret_key="s",
        sandbox=True,
        **kwargs,
    )
    a._client = client if client is not None else _FakeCCXT()
    a._markets_loaded = True
    return a


def _order(qty: float = 1.0, side: int = 1,
           order_type: OrderType = OrderType.LIMIT,
           symbol: str = "BTC/USDT", limit_price: float | None = 100.0) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=qty * side,
        limit_price=limit_price,
    )


# ── Constructor / live gate ───────────────────────────────────────────────

class TestConstructor:
    def test_sandbox_default(self):
        a = CCXTBrokerAdapter(exchange_name="binance", api_key="k", secret_key="s")
        assert a.sandbox is True
        assert a.exchange_name == "binance"

    def test_live_blocked_without_env(self, monkeypatch):
        monkeypatch.delenv("WANG_ALLOW_LIVE_CRYPTO", raising=False)
        with pytest.raises(ValueError, match="Live crypto trading blocked"):
            CCXTBrokerAdapter(
                exchange_name="binance", api_key="k", secret_key="s", sandbox=False,
            )

    def test_live_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("WANG_ALLOW_LIVE_CRYPTO", "yes")
        a = CCXTBrokerAdapter(
            exchange_name="binance", api_key="k", secret_key="s", sandbox=False,
        )
        assert a.sandbox is False

    def test_passphrase_stored(self):
        a = CCXTBrokerAdapter(
            exchange_name="coinbase", api_key="k", secret_key="s",
            passphrase="p", sandbox=True,
        )
        assert a.passphrase == "p"

    def test_backwards_compat_kwargs(self):
        a = CCXTBrokerAdapter(
            exchange="binance", api_secret="s",
            lot_size_map={"BTC/USDT": 0.001},
            min_order_size_map={"BTC/USDT": 0.01},
        )
        assert a.exchange_name == "binance"
        assert a.round_to_lot("BTC/USDT", 0.12345) == pytest.approx(0.123)


# ── Symbol normalization ──────────────────────────────────────────────────

class TestSymbolNormalization:
    def test_kraken_xbt_to_btc(self):
        a = CCXTBrokerAdapter(exchange_name="kraken", api_key="k", secret_key="s")
        assert a.normalize_symbol("XBT/USD") == "BTC/USD"
        assert a.normalize_symbol("XBTUSD") == "BTC/USD"

    def test_concat_to_slash(self):
        a = CCXTBrokerAdapter(exchange_name="binance", api_key="k", secret_key="s")
        assert a.normalize_symbol("BTCUSDT") == "BTC/USDT"

    def test_already_unified_passthrough(self):
        a = CCXTBrokerAdapter(exchange_name="binance", api_key="k", secret_key="s")
        assert a.normalize_symbol("ETH/USDT") == "ETH/USDT"


# ── submit_order ──────────────────────────────────────────────────────────

class TestSubmitOrder:
    def test_limit_order_roundtrips(self):
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.001}, "cost": {"min": 10}}}}
        client = _FakeCCXT(markets=markets)
        client.create_order.return_value = {
            "id": "abc", "status": "closed", "filled": 1.0,
            "average": 100.0, "fee": {"cost": 0.1},
        }
        adapter = _adapter(client)

        order = _order(qty=1.0, limit_price=100.0)
        result = asyncio.run(adapter.submit_order(order))

        assert result.order_id == "abc"
        assert result.status == OrderStatus.FILLED
        assert len(result.fills) == 1
        assert result.fills[0].price == 100.0
        args, _ = client.create_order.call_args[:2]
        sym, typ, side, qty, price, _params = args
        assert sym == "BTC/USDT"
        assert typ == "limit"
        assert side == "buy"
        assert qty == pytest.approx(1.0)
        assert price == pytest.approx(100.0)

    def test_reject_below_min_amount(self):
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.01}, "cost": {"min": 0}}}}
        client = _FakeCCXT(markets=markets)
        adapter = _adapter(client)

        order = _order(qty=0.001, limit_price=100.0)
        result = asyncio.run(adapter.submit_order(order))

        assert result.status == OrderStatus.REJECTED
        client.create_order.assert_not_called()

    def test_reject_below_min_cost(self):
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0}, "cost": {"min": 100}}}}
        client = _FakeCCXT(markets=markets)
        adapter = _adapter(client)

        order = _order(qty=0.1, limit_price=50.0)  # cost = 5
        result = asyncio.run(adapter.submit_order(order))

        assert result.status == OrderStatus.REJECTED
        client.create_order.assert_not_called()

    def test_precision_rounding_applied(self):
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0}, "cost": {"min": 0}}}}
        client = _FakeCCXT(markets=markets, precision_qty=3, precision_price=2)
        client.create_order.return_value = {"id": "x", "status": "open", "filled": 0}
        adapter = _adapter(client)

        order = _order(qty=1.23456789, limit_price=100.987654)
        asyncio.run(adapter.submit_order(order))
        args = client.create_order.call_args.args
        _, _, _, qty, price, _ = args
        assert qty == pytest.approx(1.235)  # truncated / rounded to 3 decimals
        assert price == pytest.approx(100.99)

    def test_limit_at_mid_uses_order_book(self):
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0}, "cost": {"min": 0}}}}
        client = _FakeCCXT(markets=markets)
        client.fetch_order_book.return_value = {
            "bids": [[99.0, 1]], "asks": [[101.0, 1]],
        }
        client.create_order.return_value = {"id": "y", "status": "open", "filled": 0}
        adapter = _adapter(client)

        order = _order(order_type=OrderType.LIMIT_AT_MID, qty=0.5, limit_price=None)
        asyncio.run(adapter.submit_order(order))
        _, _, _, _, price, _ = client.create_order.call_args.args
        assert price == pytest.approx(100.00)

    def test_network_error_wrapped(self):
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0}, "cost": {"min": 0}}}}
        client = _FakeCCXT(markets=markets)
        client.create_order.side_effect = RuntimeError("timeout")
        adapter = _adapter(client)

        with pytest.raises(BrokerNetworkError):
            asyncio.run(adapter.submit_order(_order()))


# ── Quote / funding / status ──────────────────────────────────────────────

class TestMarketData:
    def test_get_quote(self):
        client = _FakeCCXT()
        client.fetch_ticker.return_value = {"bid": 99.0, "ask": 101.0, "last": 100.0}
        adapter = _adapter(client)

        out = asyncio.run(adapter.get_quote("BTC/USDT"))
        assert out["mid"] == pytest.approx(100.0)
        assert out["spread"] == pytest.approx(2.0)

    def test_funding_rate_annualized(self):
        client = _FakeCCXT()
        client.fetch_funding_rate.return_value = {
            "fundingRate": 0.0001, "fundingInterval": 8.0,
        }
        adapter = _adapter(client)
        rate = asyncio.run(adapter.get_funding_rate("BTC/USDT"))
        # 0.0001 * (24/8) * 365 = 0.1095
        assert rate == pytest.approx(0.0001 * 3 * 365)
        assert isinstance(rate, float)

    def test_heartbeat_ok(self):
        client = _FakeCCXT()
        adapter = _adapter(client)
        assert asyncio.run(adapter.heartbeat()) is True

    def test_heartbeat_failure(self):
        client = _FakeCCXT()
        client.fetch_status.side_effect = RuntimeError("down")
        adapter = _adapter(client)
        assert asyncio.run(adapter.heartbeat()) is False


# ── Positions / account / symbols ─────────────────────────────────────────

class TestPositionsAndMeta:
    def test_get_positions_from_spot_balances(self):
        client = _FakeCCXT()
        client.fetch_balance.return_value = {
            "total": {"BTC": 0.5, "ETH": 0.0, "USDT": 1000.0},
            "free": {"USDT": 500.0},
        }
        adapter = _adapter(client)
        positions = asyncio.run(adapter.get_positions())
        assert "BTC" in positions
        assert positions["BTC"].quantity == pytest.approx(0.5)
        assert "USDT" not in positions  # quote currencies skipped

    def test_get_supported_symbols(self):
        markets = {"BTC/USDT": {}, "ETH/USDT": {}}
        client = _FakeCCXT(markets=markets)
        adapter = _adapter(client)
        syms = asyncio.run(adapter.get_supported_symbols())
        assert syms == ["BTC/USDT", "ETH/USDT"]

    def test_cancel_order_calls_sdk(self):
        client = _FakeCCXT()
        adapter = _adapter(client)
        ok = asyncio.run(adapter.cancel_order("abc", "BTC/USDT"))
        assert ok is True
        client.cancel_order.assert_called_once_with("abc", "BTC/USDT")
