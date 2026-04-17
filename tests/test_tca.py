"""Tests for post-trade TCA (Phase 5 / P5.06)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.execution.models import ExecutionAlgo, Fill, Order, OrderStatus, OrderType
from src.execution.tca import TCAAnalyzer, TCAResult


def _ts(offset_s: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_s)


def _make_filled_order(
    symbol: str = "AAPL",
    side: int = 1,
    fills: list[tuple[float, float]] | None = None,
    algo: ExecutionAlgo = ExecutionAlgo.IMMEDIATE,
    commission_per_fill: float = 0.0,
) -> Order:
    """fills = list of (price, quantity)."""
    fills = fills or [(100.0, 100)]
    order_id = str(uuid.uuid4())
    order = Order(
        order_id=order_id,
        timestamp=_ts(),
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=side * sum(q for _, q in fills),
        execution_algo=algo,
    )
    for i, (p, q) in enumerate(fills):
        order.add_fill(Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            timestamp=_ts(i),
            price=p,
            quantity=side * q,
            commission=commission_per_fill,
            exchange="TEST",
        ))
    return order


# ── Slippage ──────────────────────────────────────────────────────────

class TestSlippage:
    def test_buy_fills_above_arrival_is_positive_slippage(self):
        order = _make_filled_order(side=1, fills=[(101.0, 100)])
        result = TCAAnalyzer().analyze_order(
            order, arrival_mid=100.0,
            market_prices_during=pd.Series([100.0, 100.5, 101.0]),
        )
        assert result.slippage_bps > 0
        assert result.slippage_bps == pytest.approx(100.0, abs=1e-6)  # 1%

    def test_sell_fills_below_arrival_is_positive_slippage(self):
        order = _make_filled_order(side=-1, fills=[(99.0, 100)])
        result = TCAAnalyzer().analyze_order(
            order, arrival_mid=100.0,
            market_prices_during=pd.Series([100.0, 99.5, 99.0]),
        )
        # Cost is always reported as positive for slippage losses
        assert result.slippage_bps > 0

    def test_fill_at_arrival_is_zero_slippage(self):
        order = _make_filled_order(side=1, fills=[(100.0, 100)])
        result = TCAAnalyzer().analyze_order(
            order, arrival_mid=100.0, market_prices_during=pd.Series([100.0]),
        )
        assert result.slippage_bps == pytest.approx(0.0)


# ── Benchmarks ────────────────────────────────────────────────────────

class TestBenchmarks:
    def test_execution_at_twap_zero_bps(self):
        # Prices during window: 99, 100, 101 → TWAP = 100
        order = _make_filled_order(side=1, fills=[(100.0, 100)])
        result = TCAAnalyzer().analyze_order(
            order, arrival_mid=99.0,
            market_prices_during=pd.Series([99.0, 100.0, 101.0]),
        )
        assert result.benchmark_vs_twap_bps == pytest.approx(0.0, abs=1e-6)

    def test_vwap_with_volumes(self):
        order = _make_filled_order(side=1, fills=[(101.0, 100)])
        prices = pd.Series([100.0, 101.0, 102.0])
        volumes = pd.Series([1000.0, 1000.0, 1000.0])  # equal → vwap == mean
        result = TCAAnalyzer().analyze_order(
            order, arrival_mid=100.0,
            market_prices_during=prices,
            market_volumes_during=volumes,
        )
        # VWAP = 101, exec = 101 → 0 bps
        assert result.benchmark_vs_vwap_bps == pytest.approx(0.0, abs=1e-6)


# ── Degradation detection ─────────────────────────────────────────────

def _result(slippage_bps: float, fill_rate: float = 1.0, algo: str = "immediate") -> TCAResult:
    return TCAResult(
        order_id=str(uuid.uuid4()), symbol="AAPL", side=1,
        arrival_price=100.0, execution_price=100.0,
        slippage_bps=slippage_bps, market_impact_bps=0.0, timing_cost_bps=0.0,
        total_cost_bps=slippage_bps, commission=0.0, algo_used=algo,
        execution_duration_seconds=10.0, fill_rate=fill_rate,
        benchmark_vs_twap_bps=0.0, benchmark_vs_vwap_bps=0.0,
    )


class TestDegradation:
    def test_triggers_when_recent_slippage_doubles(self):
        hist = [_result(5.0 + i * 0.1) for i in range(30)]  # ~5 bps with small variance
        recent = [_result(20.0) for _ in range(5)]  # 4x historical
        warnings = TCAAnalyzer().detect_execution_degradation(recent, hist)
        assert any("degradation" in w.lower() or "doubled" in w.lower() for w in warnings)

    def test_no_warning_within_tolerance(self):
        hist = [_result(5.0 + (i % 3)) for i in range(30)]
        recent = [_result(5.5) for _ in range(5)]
        assert TCAAnalyzer().detect_execution_degradation(recent, hist) == []

    def test_fill_rate_degradation(self):
        hist = [_result(5.0, fill_rate=0.99) for _ in range(30)]
        recent = [_result(5.0, fill_rate=0.60) for _ in range(5)]
        warnings = TCAAnalyzer().detect_execution_degradation(recent, hist)
        assert any("Fill rate" in w for w in warnings)


# ── Batch + summary ───────────────────────────────────────────────────

class TestBatchAnalysis:
    def test_batch_returns_dataframe(self):
        orders = [
            _make_filled_order(symbol="AAPL", fills=[(101.0, 100)]),
            _make_filled_order(symbol="TSLA", fills=[(201.0, 50)]),
        ]
        arrival = {"AAPL": 100.0, "TSLA": 200.0}
        mkt = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "TSLA", "TSLA"],
            "price":  [100.0, 101.0, 200.0, 201.0],
        })
        df = TCAAnalyzer().analyze_batch(orders, arrival, mkt)
        assert len(df) == 2
        assert set(df["symbol"]) == {"AAPL", "TSLA"}
        assert "slippage_bps" in df.columns

    def test_summary_aggregates_by_algo(self):
        results = [
            _result(10.0, algo="twap"),
            _result(12.0, algo="twap"),
            _result(20.0, algo="immediate"),
        ]
        summary = TCAAnalyzer().get_tca_summary(results)
        assert summary["n_orders"] == 3
        assert summary["mean_slippage_bps"] == pytest.approx(14.0)
        assert "twap" in summary["cost_by_algo"]
        assert summary["cost_by_algo"]["twap"]["n"] == 2
        assert summary["cost_by_algo"]["twap"]["mean_slippage_bps"] == pytest.approx(11.0)
