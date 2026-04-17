"""Tests for daily operations (Phase 5 / P5.13)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.backtesting.transaction_costs import TransactionCostModel
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.daily_ops import DailyReconciliation, generate_daily_report
from src.execution.models import (
    ExecutionAlgo,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)
from src.execution.order_manager import OrderManager
from src.execution.storage import ExecutionStorage
from src.execution.tca import TCAResult
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel


EQ_COST_CFG = {
    "commission_per_share": 0.0,
    "min_commission": 0.0,
    "spread_bps": 1.0,
    "slippage_bps": 1.0,
    "impact_coefficient": 0.1,
}


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _make_position(symbol: str, side: int = 1, qty: float = 100,
                   entry: float = 100.0, current: float | None = None,
                   family: str = "momentum") -> Position:
    return Position(
        symbol=symbol, side=side, quantity=qty, avg_entry_price=entry,
        entry_timestamp=_ts(), signal_family=family,
        current_price=current if current is not None else entry,
        unrealized_pnl=side * qty * ((current or entry) - entry),
    )


def _make_filled_order(symbol: str, side: int = 1, qty: float = 10,
                       price: float = 100.0) -> Order:
    oid = str(uuid.uuid4())
    o = Order(
        order_id=oid, timestamp=_ts(),
        symbol=symbol, side=side, order_type=OrderType.LIMIT,
        quantity=qty * side, limit_price=price,
    )
    o.add_fill(Fill(
        fill_id=str(uuid.uuid4()), order_id=oid, timestamp=_ts(),
        price=price, quantity=qty * side, commission=0.0, exchange="T",
    ))
    return o


# ── Reconciliation ────────────────────────────────────────────────────

def _build_om(portfolio: PortfolioState) -> tuple[OrderManager, PaperBrokerAdapter]:
    broker = PaperBrokerAdapter(
        initial_cash=portfolio.cash, slippage_bps=0.0, fill_delay_ms=0,
        price_feed=lambda s: 100.0,
    )
    cost = TransactionCostModel(equities_config=EQ_COST_CFG)
    cbs = CircuitBreakerManager()
    return OrderManager(broker, cbs, cost, portfolio), broker


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_detects_synthetic_discrepancy(self, tmp_path: Path):
        pf = PortfolioState(cash=50_000.0, positions={
            "AAPL": _make_position("AAPL", qty=100, entry=100.0, current=100.0),
        })
        om, broker = _build_om(pf)
        # Introduce broker-side rogue position
        broker.positions["TSLA"] = Position(
            symbol="TSLA", side=1, quantity=10, avg_entry_price=200.0,
            entry_timestamp=_ts(), signal_family="",
        )
        alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity})
        storage = ExecutionStorage(f"sqlite:///{tmp_path / 'x.db'}")
        storage.setup_execution_schema()

        result = await DailyReconciliation().run(om, storage, alerts)
        assert any(d["symbol"] == "TSLA" for d in result["discrepancies"])


class TestDailyReport:
    def _sample(self) -> tuple[PortfolioState, list[Order], list[TCAResult], pd.DataFrame]:
        pf = PortfolioState(cash=50_000.0, positions={
            "AAPL": _make_position("AAPL", qty=100, entry=100.0, current=105.0,
                                   family="momentum"),
            "TSLA": _make_position("TSLA", side=-1, qty=50, entry=200.0,
                                   current=190.0, family="mean_rev"),
        })
        pf.daily_pnl = 700.0
        trades = [_make_filled_order("AAPL", qty=10, price=100.0)]
        tca = [TCAResult(
            order_id=trades[0].order_id, symbol="AAPL", side=1,
            arrival_price=100.0, execution_price=100.1,
            slippage_bps=10.0, market_impact_bps=5.0, timing_cost_bps=5.0,
            total_cost_bps=10.0, commission=0.0,
            algo_used=ExecutionAlgo.TWAP.value,
            execution_duration_seconds=30.0, fill_rate=1.0,
            benchmark_vs_twap_bps=0.0, benchmark_vs_vwap_bps=0.0,
        )]
        drift = pd.DataFrame([
            {"feature": "ret_ffd", "drifted": True},
            {"feature": "vol_garch", "drifted": False},
        ])
        return pf, trades, tca, drift

    def test_report_contains_all_sections(self):
        pf, trades, tca, drift = self._sample()
        report = generate_daily_report(
            pf, trades, tca, drift,
            breakers_triggered=["REDUCE_SIZE_50"],
            last_model_retrain=_ts() - timedelta(hours=48),
            next_retrain=_ts() + timedelta(hours=72),
        )
        for section in (
            "Daily Report",
            "Portfolio",
            "Trades Today",
            "Top Positions",
            "Exposure by Strategy",
            "Feature Drift Warnings",
            "Circuit Breaker Activations",
            "Model",
        ):
            assert section in report, f"missing section: {section}"

    def test_report_surfaces_drifted_features(self):
        pf, trades, tca, drift = self._sample()
        report = generate_daily_report(pf, trades, tca, drift)
        assert "ret_ffd" in report
        assert "REDUCE_SIZE" not in report  # not passed in this variant

    def test_report_lists_top_positions_by_pnl(self):
        pf, trades, tca, drift = self._sample()
        report = generate_daily_report(pf, trades, tca, drift)
        # AAPL: +$500, TSLA: +$500. Both should appear
        assert "AAPL" in report
        assert "TSLA" in report

    def test_pnl_and_slippage_match_manual_calc(self):
        pf, trades, tca, drift = self._sample()
        report = generate_daily_report(pf, trades, tca, drift)
        # Avg slippage = 10.0 bps (one result)
        assert "10.00 bps" in report
        # Count filled trades = 1
        assert "Count (filled): 1" in report

    def test_empty_portfolio_handled(self):
        pf = PortfolioState(cash=10_000.0)
        report = generate_daily_report(pf, [], [], pd.DataFrame())
        assert "(no positions)" in report
        assert "Count (filled): 0" in report


class TestReconciliationPnL:
    @pytest.mark.asyncio
    async def test_daily_pnl_reflects_portfolio_state(self, tmp_path: Path):
        pf = PortfolioState(cash=100_000.0, positions={
            "AAPL": _make_position("AAPL", qty=100, entry=100.0, current=110.0),
        })
        pf.daily_pnl = 1000.0  # $10/share * 100 shares
        om, _ = _build_om(pf)
        alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity})
        storage = ExecutionStorage(f"sqlite:///{tmp_path / 'y.db'}")
        storage.setup_execution_schema()
        result = await DailyReconciliation().run(om, storage, alerts)
        assert result["daily_pnl"] == pytest.approx(1000.0)
        assert result["daily_return"] == pytest.approx(1000.0 / pf.nav)
