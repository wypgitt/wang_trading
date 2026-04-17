"""Tests for circuit breakers (Phase 5 / P5.02)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.execution.circuit_breakers import (
    Action,
    CircuitBreakerAction,
    CircuitBreakerManager,
    OperatorCheckin,
    Severity,
)
from src.execution.models import (
    Order,
    OrderType,
    PortfolioState,
    Position,
)


NAV = 100_000.0


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _order(symbol: str, side: int, qty: float, price: float) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=_ts(),
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=qty * side,
        limit_price=price,
    )


def _empty_portfolio(cash: float = NAV) -> PortfolioState:
    return PortfolioState(cash=cash)


def _portfolio_with_positions(positions: dict[str, Position], cash: float = 0.0) -> PortfolioState:
    return PortfolioState(cash=cash, positions=positions)


class TestPreTradeChecks:
    def test_normal_small_order_passes(self):
        cb = CircuitBreakerManager()
        pf = _empty_portfolio()
        order = _order("AAPL", side=1, qty=10, price=150.0)  # $1500 = 1.5% NAV
        ok, reason = cb.check_pre_trade(order, pf)
        assert ok
        assert reason is None

    def test_fat_finger_rejects_6pct_order(self):
        cb = CircuitBreakerManager(max_order_pct=0.05)
        pf = _empty_portfolio()
        order = _order("AAPL", side=1, qty=60, price=100.0)  # $6000 = 6% NAV
        ok, reason = cb.check_pre_trade(order, pf)
        assert not ok
        assert "Fat finger" in reason

    def test_daily_loss_limit_blocks_new_entry(self):
        cb = CircuitBreakerManager(daily_loss_limit_pct=0.02)
        pf = _empty_portfolio()
        pf.daily_pnl = -0.025 * NAV  # -2.5%
        order = _order("AAPL", side=1, qty=10, price=100.0)
        ok, reason = cb.check_pre_trade(order, pf)
        assert not ok
        assert "Daily loss limit" in reason

    def test_daily_loss_limit_allows_exit(self):
        cb = CircuitBreakerManager(daily_loss_limit_pct=0.02)
        pos = Position(
            symbol="AAPL", side=1, quantity=50, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = _portfolio_with_positions({"AAPL": pos}, cash=NAV - 5000)
        pf.daily_pnl = -0.025 * pf.nav
        # Exit = sell-to-close
        order = _order("AAPL", side=-1, qty=50, price=100.0)
        ok, reason = cb.check_pre_trade(order, pf)
        assert ok, f"Exit should be allowed, got: {reason}"

    def test_max_positions_blocks_21st(self):
        cb = CircuitBreakerManager(max_positions=20, max_single_position_pct=1.0,
                                    max_order_pct=1.0, max_gross_exposure=100.0)
        positions = {
            f"SYM{i}": Position(
                symbol=f"SYM{i}", side=1, quantity=1, avg_entry_price=1.0,
                entry_timestamp=_ts(), signal_family="", current_price=1.0,
            )
            for i in range(20)
        }
        pf = _portfolio_with_positions(positions, cash=NAV)
        order = _order("NEW", side=1, qty=1, price=100.0)
        ok, reason = cb.check_pre_trade(order, pf)
        assert not ok
        assert "Max positions" in reason

    def test_max_gross_exposure_rejects(self):
        cb = CircuitBreakerManager(max_gross_exposure=1.50, max_single_position_pct=1.0,
                                    max_order_pct=1.0)
        # NAV = 100k. Existing gross = 140k (140%)
        pos = Position(
            symbol="EXISTING", side=1, quantity=140_000, avg_entry_price=1.0,
            entry_timestamp=_ts(), signal_family="", current_price=1.0,
        )
        # cash = -40k so NAV = 140k - 40k = 100k, gross = 140k = 140% of NAV
        pf = _portfolio_with_positions({"EXISTING": pos}, cash=-40_000.0)
        # Add 20k more gross → 160% > 150% limit
        order = _order("NEW", side=1, qty=20_000, price=1.0)
        ok, reason = cb.check_pre_trade(order, pf)
        assert not ok
        assert "Gross exposure" in reason

    def test_max_single_position_pct(self):
        cb = CircuitBreakerManager(max_single_position_pct=0.10, max_order_pct=1.0,
                                    max_gross_exposure=100.0)
        pf = _empty_portfolio()
        # 11% position
        order = _order("AAPL", side=1, qty=110, price=100.0)
        ok, reason = cb.check_pre_trade(order, pf)
        assert not ok
        assert "NAV" in reason or "Position" in reason

    def test_crypto_allocation_limit(self):
        cb = CircuitBreakerManager(max_crypto_pct=0.30, max_single_position_pct=1.0,
                                    max_order_pct=1.0, max_gross_exposure=100.0)
        # NAV = 100k. BTC = 25k (25%). Adding 10k ETH → 35% > 30%.
        pos = Position(
            symbol="BTC-USD", side=1, quantity=250, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = _portfolio_with_positions({"BTC-USD": pos}, cash=75_000.0)
        order = _order("ETH-USD", side=1, qty=100, price=100.0)
        ok, reason = cb.check_pre_trade(order, pf)
        assert not ok
        assert "Crypto" in reason


class TestPortfolioHealth:
    def _pf_with_dd(self, dd_pct: float) -> PortfolioState:
        # Construct so that peak_nav > nav by dd_pct
        pos = Position(
            symbol="X", side=1, quantity=100, avg_entry_price=100.0,
            entry_timestamp=_ts(), signal_family="", current_price=100.0,
        )
        pf = PortfolioState(cash=0.0, positions={"X": pos})
        # Push peak up, then drop price
        pf.update_prices({"X": 100.0})
        pf.peak_nav = pf.nav / (1.0 - dd_pct)
        pf.drawdown = dd_pct
        return pf

    def test_drawdown_10_reduces_50(self):
        cb = CircuitBreakerManager()
        pf = self._pf_with_dd(0.10)
        actions = cb.check_portfolio_health(pf)
        assert any(a.action == Action.REDUCE_SIZE_50 for a in actions)

    def test_drawdown_15_reduces_75(self):
        cb = CircuitBreakerManager()
        pf = self._pf_with_dd(0.16)
        actions = cb.check_portfolio_health(pf)
        assert any(a.action == Action.REDUCE_SIZE_75 for a in actions)

    def test_drawdown_20_halt_and_flatten(self):
        cb = CircuitBreakerManager()
        pf = self._pf_with_dd(0.20)
        actions = cb.check_portfolio_health(pf)
        assert any(a.action == Action.HALT_AND_FLATTEN for a in actions)
        assert any(a.severity == Severity.EMERGENCY for a in actions)

    def test_model_staleness(self):
        cb = CircuitBreakerManager()
        pf = _empty_portfolio()
        now = _ts()
        actions = cb.check_portfolio_health(
            pf, last_model_retrain=now - timedelta(days=70), now=now,
        )
        assert any(a.action == Action.HALT for a in actions)

    def test_connectivity_timeout(self):
        cb = CircuitBreakerManager()
        pf = _empty_portfolio()
        now = _ts()
        actions = cb.check_portfolio_health(
            pf, last_broker_heartbeat=now - timedelta(seconds=120), now=now,
        )
        assert any(a.action == Action.FLATTEN_ALL for a in actions)

    def test_multiple_breakers_fire_simultaneously(self):
        cb = CircuitBreakerManager()
        pf = self._pf_with_dd(0.20)
        now = _ts()
        actions = cb.check_portfolio_health(
            pf,
            last_model_retrain=now - timedelta(days=70),
            last_broker_heartbeat=now - timedelta(seconds=120),
            portfolio_correlation=0.90,
            now=now,
        )
        action_set = {a.action for a in actions}
        assert Action.HALT_AND_FLATTEN in action_set
        assert Action.HALT in action_set
        assert Action.FLATTEN_ALL in action_set
        assert Action.REDUCE_GROSS_50 in action_set


class TestDeadManSwitch:
    def test_flatten_and_halt_after_24h(self):
        cb = CircuitBreakerManager(dead_man_hours=24.0)
        now = _ts()
        last = now - timedelta(hours=25)
        action = cb.check_dead_man_switch(last, now=now)
        assert action is not None
        assert action.action == Action.FLATTEN_ALL_AND_HALT
        assert action.severity == Severity.EMERGENCY

    def test_no_action_within_24h(self):
        cb = CircuitBreakerManager(dead_man_hours=24.0)
        now = _ts()
        last = now - timedelta(hours=12)
        assert cb.check_dead_man_switch(last, now=now) is None


class TestOperatorCheckin:
    def test_checkin_persists(self, tmp_path):
        path = tmp_path / "checkin.txt"
        a = OperatorCheckin(path=path)
        t = a.checkin()
        assert path.exists()
        # Fresh instance reads same file
        b = OperatorCheckin(path=path)
        last = b.get_last_checkin()
        assert last is not None
        assert abs((last - t).total_seconds()) < 1.0

    def test_get_last_checkin_none_if_missing(self, tmp_path):
        path = tmp_path / "nope.txt"
        a = OperatorCheckin(path=path)
        assert a.get_last_checkin() is None
