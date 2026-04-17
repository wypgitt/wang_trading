"""Tests for broker adapters (Phase 5 / P5.04)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from src.execution.algorithms import IcebergAlgo, ImmediateAlgo, TWAPAlgo
from src.execution.broker_adapter import (
    AlpacaBrokerAdapter,
    BaseBrokerAdapter,
    CCXTBrokerAdapter,
    PaperBrokerAdapter,
    reconcile_positions,
)
from src.execution.models import (
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _order(symbol: str = "AAPL", side: int = 1, qty: float = 100.0,
           order_type: OrderType = OrderType.MARKET,
           limit: float | None = None) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=_ts(),
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=qty * side,
        limit_price=limit,
    )


# ── PaperBrokerAdapter ────────────────────────────────────────────────

class TestPaperBroker:
    @pytest.mark.asyncio
    async def test_fills_buy_and_updates_positions(self):
        broker = PaperBrokerAdapter(
            initial_cash=10_000, price_feed=lambda s: 100.0,
            slippage_bps=0, fill_delay_ms=0,
        )
        order = _order(qty=10, side=1)
        result = await broker.submit_order(order)
        assert result.status == OrderStatus.FILLED
        positions = await broker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 10
        assert positions["AAPL"].side == 1

    @pytest.mark.asyncio
    async def test_applies_slippage(self):
        broker = PaperBrokerAdapter(
            initial_cash=10_000, price_feed=lambda s: 100.0,
            slippage_bps=20, fill_delay_ms=0,  # 20 bps = 0.2%
        )
        buy = _order(qty=10, side=1)
        await broker.submit_order(buy)
        # Buy fills above mid
        assert buy.fills[0].price == pytest.approx(100.0 * 1.002)

        sell = _order(symbol="TSLA", qty=10, side=-1)
        await broker.submit_order(sell)
        # Sell fills below mid
        assert sell.fills[0].price == pytest.approx(100.0 * 0.998)

    @pytest.mark.asyncio
    async def test_cash_decreases_on_buy_increases_on_sell(self):
        broker = PaperBrokerAdapter(
            initial_cash=10_000, price_feed=lambda s: 100.0,
            slippage_bps=0, fill_delay_ms=0,
        )
        await broker.submit_order(_order(qty=10, side=1))
        assert broker.cash == pytest.approx(10_000 - 1000)

        # Sell the position
        sell = _order(side=-1, qty=10)
        await broker.submit_order(sell)
        assert broker.cash == pytest.approx(10_000)  # back to start
        assert (await broker.get_positions()) == {}

    @pytest.mark.asyncio
    async def test_short_position(self):
        broker = PaperBrokerAdapter(
            initial_cash=10_000, price_feed=lambda s: 50.0,
            slippage_bps=0, fill_delay_ms=0,
        )
        await broker.submit_order(_order(qty=20, side=-1))
        positions = await broker.get_positions()
        assert positions["AAPL"].side == -1
        assert positions["AAPL"].quantity == 20
        # Cash went UP (short sale credits cash)
        assert broker.cash == pytest.approx(10_000 + 1000)

    @pytest.mark.asyncio
    async def test_cancel_filled_order_returns_false(self):
        broker = PaperBrokerAdapter(price_feed=lambda s: 100.0, fill_delay_ms=0)
        order = _order(qty=1)
        await broker.submit_order(order)
        assert order.status == OrderStatus.FILLED
        assert await broker.cancel_order(order.order_id) is False

    @pytest.mark.asyncio
    async def test_cancel_pending_order_returns_true(self):
        # Limit way below market so it won't fill
        broker = PaperBrokerAdapter(price_feed=lambda s: 100.0, fill_delay_ms=0,
                                    slippage_bps=0)
        order = _order(qty=10, side=1, order_type=OrderType.LIMIT, limit=50.0)
        await broker.submit_order(order)
        assert order.status != OrderStatus.FILLED
        assert await broker.cancel_order(order.order_id) is True
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_account_reports_nav(self):
        broker = PaperBrokerAdapter(
            initial_cash=10_000, price_feed=lambda s: 100.0,
            slippage_bps=0, fill_delay_ms=0,
        )
        await broker.submit_order(_order(qty=10, side=1))
        acct = await broker.get_account()
        assert acct["cash"] == pytest.approx(9_000)
        assert acct["nav"] == pytest.approx(10_000)  # 9k cash + 1k position
        assert acct["position_count"] == 1

    @pytest.mark.asyncio
    async def test_quote_and_mid(self):
        broker = PaperBrokerAdapter(price_feed=lambda s: 200.0, spread_bps=10)
        q = await broker.get_quote("AAPL")
        assert q["mid"] == pytest.approx(200.0)
        assert q["bid"] < q["mid"] < q["ask"]
        mid = await broker.get_mid_price("AAPL")
        assert mid == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_heartbeat_true(self):
        broker = PaperBrokerAdapter()
        assert await broker.heartbeat() is True


# ── Position reconciliation ────────────────────────────────────────────

class TestReconciliation:
    def test_detects_quantity_discrepancy(self):
        internal = PortfolioState(cash=0.0, positions={
            "AAPL": Position(
                symbol="AAPL", side=1, quantity=100, avg_entry_price=150,
                entry_timestamp=_ts(), signal_family="",
            ),
        })
        broker_pos = {
            "AAPL": Position(
                symbol="AAPL", side=1, quantity=95, avg_entry_price=150,
                entry_timestamp=_ts(), signal_family="",
            ),
        }
        diffs = reconcile_positions(internal, broker_pos)
        assert len(diffs) == 1
        assert diffs[0].symbol == "AAPL"
        assert diffs[0].delta == pytest.approx(-5)

    def test_detects_missing_symbol(self):
        internal = PortfolioState(cash=0.0)
        broker_pos = {
            "TSLA": Position(
                symbol="TSLA", side=-1, quantity=50, avg_entry_price=200,
                entry_timestamp=_ts(), signal_family="",
            ),
        }
        diffs = reconcile_positions(internal, broker_pos)
        assert len(diffs) == 1
        assert diffs[0].symbol == "TSLA"

    def test_no_diff_when_matched(self):
        pos_args = dict(avg_entry_price=150.0, entry_timestamp=_ts(), signal_family="")
        internal = PortfolioState(cash=0.0, positions={
            "AAPL": Position(symbol="AAPL", side=1, quantity=100, **pos_args),
        })
        broker = {
            "AAPL": Position(symbol="AAPL", side=1, quantity=100, **pos_args),
        }
        assert reconcile_positions(internal, broker) == []


# ── PaperBroker wired into execution algos ────────────────────────────

class TestPaperBrokerWithAlgos:
    @pytest.mark.asyncio
    async def test_immediate_algo_on_paper_broker(self):
        broker = PaperBrokerAdapter(
            price_feed=lambda s: 100.0, slippage_bps=0, fill_delay_ms=0,
        )
        order = _order(qty=10, side=1, order_type=OrderType.LIMIT)
        algo = ImmediateAlgo(order, broker, timeout_seconds=0.0, use_market_fallback=False)
        await algo.execute()
        assert algo.is_complete
        assert order.filled_quantity == pytest.approx(10)

    @pytest.mark.asyncio
    async def test_twap_algo_on_paper_broker(self):
        broker = PaperBrokerAdapter(
            price_feed=lambda s: 100.0, slippage_bps=0, fill_delay_ms=0,
        )
        order = _order(qty=100, side=1, order_type=OrderType.TWAP)
        algo = TWAPAlgo(order, broker, n_slices=4, duration_minutes=0.0,
                       child_timeout_seconds=0.0, time_scale=0.0)
        await algo.execute()
        assert order.filled_quantity == pytest.approx(100)
        assert len(algo.child_orders) == 4

    @pytest.mark.asyncio
    async def test_iceberg_algo_on_paper_broker(self):
        broker = PaperBrokerAdapter(
            price_feed=lambda s: 50.0, slippage_bps=0, fill_delay_ms=0,
        )
        order = _order(symbol="BTC-USD", qty=100, side=1,
                       order_type=OrderType.ICEBERG)
        algo = IcebergAlgo(order, broker, visible_size=25,
                          child_timeout_seconds=0.0, time_scale=0.0)
        await algo.execute()
        assert order.filled_quantity == pytest.approx(100)
        assert len(algo.child_orders) == 4


# ── Alpaca / CCXT skeletons ────────────────────────────────────────────

class TestAdapterSkeletons:
    def test_alpaca_adapter_instantiates(self):
        adapter = AlpacaBrokerAdapter(api_key="x", api_secret="y", paper=True)
        assert adapter.paper is True
        assert isinstance(adapter, BaseBrokerAdapter)

    def test_ccxt_adapter_lot_rounding(self):
        adapter = CCXTBrokerAdapter(
            exchange="binance",
            lot_size_map={"BTC/USDT": 0.001},
            min_order_size_map={"BTC/USDT": 0.01},
        )
        assert adapter.round_to_lot("BTC/USDT", 0.12345) == pytest.approx(0.123)
        assert adapter.meets_min_size("BTC/USDT", 0.02) is True
        assert adapter.meets_min_size("BTC/USDT", 0.005) is False
