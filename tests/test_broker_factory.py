"""Tests for broker_factory.BrokerFactory and SmartOrderRouter (P6.04)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.execution.broker_adapter import (
    AlpacaBrokerAdapter,
    CCXTBrokerAdapter,
)
from src.execution.broker_factory import BrokerFactory, SmartOrderRouter
from src.execution.ibkr_adapter import IBKRBrokerAdapter
from src.execution.models import Order, OrderStatus, OrderType


CONFIG = {
    "alpaca": {"api_key": "k", "secret_key": "s", "paper": True},
    "binance": {"api_key": "k", "secret_key": "s", "sandbox": True},
    "ibkr": {"host": "127.0.0.1", "port": 7497, "client_id": 1, "live": False},
}


def _order(symbol: str = "BTC/USDT", qty: float = 1.0, side: int = 1) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=qty * side,
        limit_price=100.0,
    )


# ── Factory classification + resolution ──────────────────────────────────

class TestClassification:
    def test_equity(self):
        f = BrokerFactory(CONFIG)
        assert f.classify("AAPL") == BrokerFactory.ASSET_EQUITIES

    def test_crypto_slash(self):
        f = BrokerFactory(CONFIG)
        assert f.classify("BTC/USDT") == BrokerFactory.ASSET_CRYPTO

    def test_futures_root(self):
        f = BrokerFactory(CONFIG)
        assert f.classify("ES") == BrokerFactory.ASSET_FUTURES

    def test_futures_with_contract(self):
        f = BrokerFactory(CONFIG)
        assert f.classify("ESZ25") == BrokerFactory.ASSET_FUTURES


class TestGetBroker:
    def test_returns_alpaca_for_equity(self):
        f = BrokerFactory(CONFIG)
        b = f.get_broker("AAPL")
        assert isinstance(b, AlpacaBrokerAdapter)

    def test_returns_ccxt_for_crypto(self):
        f = BrokerFactory(CONFIG)
        b = f.get_broker("BTC/USDT")
        assert isinstance(b, CCXTBrokerAdapter)

    def test_returns_ibkr_for_futures(self):
        f = BrokerFactory(CONFIG)
        b = f.get_broker("ES")
        assert isinstance(b, IBKRBrokerAdapter)

    def test_caching_same_instance(self):
        f = BrokerFactory(CONFIG)
        a = f.get_broker("AAPL")
        b = f.get_broker("MSFT")
        assert a is b

    def test_override_asset_class(self):
        f = BrokerFactory(CONFIG)
        # AAPL forced to crypto via explicit asset_class
        b = f.get_broker("AAPL", asset_class=BrokerFactory.ASSET_CRYPTO)
        assert isinstance(b, CCXTBrokerAdapter)


# ── Heartbeat + shutdown ─────────────────────────────────────────────────

class TestHeartbeatAll:
    def test_aggregates(self):
        f = BrokerFactory(CONFIG)
        # Pre-populate cache with mocks
        eq = MagicMock()
        eq.heartbeat = AsyncMock(return_value=True)
        cr = MagicMock()
        cr.heartbeat = AsyncMock(return_value=False)
        f._cache = {"equities": eq, "crypto": cr}

        out = asyncio.run(f.heartbeat_all())
        assert out == {"equities": True, "crypto": False}

    def test_empty(self):
        f = BrokerFactory(CONFIG)
        assert asyncio.run(f.heartbeat_all()) == {}

    def test_exception_becomes_false(self):
        f = BrokerFactory(CONFIG)
        eq = MagicMock()
        eq.heartbeat = AsyncMock(side_effect=RuntimeError("x"))
        f._cache = {"equities": eq}
        out = asyncio.run(f.heartbeat_all())
        assert out == {"equities": False}


# ── Smart order router ───────────────────────────────────────────────────

def _fake_ccxt_broker(name: str, bid: float, ask: float, depth: float = 100.0):
    broker = MagicMock(spec=CCXTBrokerAdapter)
    broker.exchange_name = name
    broker.normalize_symbol = lambda s: s
    broker.get_quote = AsyncMock(
        return_value={"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spread": ask - bid}
    )
    broker.client = MagicMock()
    broker.client.fetch_order_book = AsyncMock(return_value={
        "bids": [[bid, depth]],
        "asks": [[ask, depth]],
    })
    broker.submit_order = AsyncMock(side_effect=lambda o: _filled(o))
    return broker


def _filled(order: Order) -> Order:
    order.status = OrderStatus.FILLED
    return order


class TestSmartOrderRouter:
    def test_best_quote_buy_picks_lowest_ask(self):
        a = _fake_ccxt_broker("binance", bid=99, ask=101)
        b = _fake_ccxt_broker("coinbase", bid=98.5, ask=100.5)
        router = SmartOrderRouter([a, b])
        name, price = asyncio.run(router.get_best_quote("BTC/USDT", side=1))
        assert name == "coinbase"
        assert price == 100.5

    def test_best_quote_sell_picks_highest_bid(self):
        a = _fake_ccxt_broker("binance", bid=99, ask=101)
        b = _fake_ccxt_broker("coinbase", bid=99.5, ask=100.5)
        router = SmartOrderRouter([a, b])
        name, price = asyncio.run(router.get_best_quote("BTC/USDT", side=-1))
        assert name == "coinbase"
        assert price == 99.5

    def test_route_small_order_goes_to_best(self):
        a = _fake_ccxt_broker("binance", bid=99, ask=101, depth=10)
        b = _fake_ccxt_broker("coinbase", bid=99.5, ask=100.5, depth=10)
        router = SmartOrderRouter([a, b])

        order = _order(qty=2.0)  # small: 2 < 10 depth on best venue
        children = asyncio.run(router.route_order(order))
        assert len(children) == 1
        # Child went to coinbase (best ask)
        b.submit_order.assert_called_once()
        a.submit_order.assert_not_called()
        assert abs(children[0].quantity) == 2.0

    def test_route_large_order_splits(self):
        a = _fake_ccxt_broker("binance", bid=99, ask=101, depth=3)
        b = _fake_ccxt_broker("coinbase", bid=99.5, ask=100.5, depth=3)
        router = SmartOrderRouter([a, b])

        order = _order(qty=5.0)  # larger than any single venue's depth
        children = asyncio.run(router.route_order(order))
        assert len(children) == 2
        qtys = sorted(abs(c.quantity) for c in children)
        assert qtys == [2.0, 3.0]  # 3 filled at coinbase, remaining 2 at binance
        a.submit_order.assert_called_once()
        b.submit_order.assert_called_once()

    def test_aggregated_depth_merges(self):
        a = _fake_ccxt_broker("binance", bid=99, ask=101, depth=5)
        b = _fake_ccxt_broker("coinbase", bid=99.5, ask=100.5, depth=5)
        router = SmartOrderRouter([a, b])
        df = asyncio.run(router.get_aggregated_depth("BTC/USDT", levels=1))
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"exchange", "side", "price", "size"}
        # bids first (sorted desc), asks (sorted asc)
        assert df.iloc[0]["price"] == 99.5
        assert df[df.side == "ask"].iloc[0]["price"] == 100.5
