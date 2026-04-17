"""Circuit Breakers (Phase 5 / P5.02).

Implements the 8 circuit breakers from design doc §10.5:
daily loss limit, drawdown throttle, model staleness, data quality,
connectivity, dead-man switch, fat-finger, correlation spike.

Pre-trade checks run synchronously before order submission.
Portfolio-health checks run on a timer and emit CircuitBreakerAction events.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.execution.models import Order, PortfolioState


# ── Action + severity constants ────────────────────────────────────────

class Action:
    REDUCE_SIZE_50 = "REDUCE_SIZE_50"
    REDUCE_SIZE_75 = "REDUCE_SIZE_75"
    REDUCE_GROSS_50 = "REDUCE_GROSS_50"
    HALT = "HALT"
    FLATTEN_ALL = "FLATTEN_ALL"
    HALT_AND_FLATTEN = "HALT_AND_FLATTEN"
    FLATTEN_ALL_AND_HALT = "FLATTEN_ALL_AND_HALT"


class Severity:
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class CircuitBreakerAction:
    action: str
    reason: str
    severity: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Operator heartbeat ─────────────────────────────────────────────────

class OperatorCheckin:
    """Persistent operator heartbeat for the dead-man switch."""

    def __init__(self, path: str | os.PathLike[str] = "logs/operator_checkin.txt") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def checkin(self) -> datetime:
        now = datetime.now(timezone.utc)
        self.path.write_text(now.isoformat(), encoding="utf-8")
        return now

    def get_last_checkin(self) -> datetime | None:
        if not self.path.exists():
            return None
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw)


# ── Circuit breaker manager ────────────────────────────────────────────

def _is_crypto(symbol: str) -> bool:
    s = symbol.upper()
    return (
        "/" in s
        or s.endswith("-USD")
        or s.endswith("-USDT")
        or s.endswith("USDT")
        or s.endswith("USDC")
        or s in {"BTC", "ETH", "SOL"}
    )


class CircuitBreakerManager:
    """Enforces pre-trade and portfolio-health circuit breakers."""

    def __init__(
        self,
        *,
        max_order_pct: float = 0.05,
        daily_loss_limit_pct: float = 0.02,
        max_positions: int = 20,
        max_gross_exposure: float = 1.50,
        max_single_position_pct: float = 0.10,
        max_crypto_pct: float = 0.30,
        dd_throttle_50: float = 0.10,
        dd_throttle_75: float = 0.15,
        dd_halt: float = 0.20,
        model_stale_warn_days: int = 30,
        model_stale_halt_days: int = 60,
        connectivity_timeout_s: float = 60.0,
        bar_rate_sigma: float = 3.0,
        correlation_spike: float = 0.80,
        dead_man_hours: float = 24.0,
        crypto_detector=None,
    ) -> None:
        self.max_order_pct = max_order_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_positions = max_positions
        self.max_gross_exposure = max_gross_exposure
        self.max_single_position_pct = max_single_position_pct
        self.max_crypto_pct = max_crypto_pct
        self.dd_throttle_50 = dd_throttle_50
        self.dd_throttle_75 = dd_throttle_75
        self.dd_halt = dd_halt
        self.model_stale_warn_days = model_stale_warn_days
        self.model_stale_halt_days = model_stale_halt_days
        self.connectivity_timeout_s = connectivity_timeout_s
        self.bar_rate_sigma = bar_rate_sigma
        self.correlation_spike = correlation_spike
        self.dead_man_hours = dead_man_hours
        self._is_crypto = crypto_detector or _is_crypto

    # ── Pre-trade checks ───────────────────────────────────────────────

    def _is_exit(self, order: Order, portfolio: PortfolioState) -> bool:
        """An order is an 'exit' if it reduces an existing position."""
        pos = portfolio.positions.get(order.symbol)
        if pos is None:
            return False
        # Exit: order side opposite to position side, quantity ≤ position quantity
        return order.side == -pos.side and abs(order.quantity) <= pos.quantity + 1e-9

    def check_pre_trade(
        self, order: Order, portfolio: PortfolioState
    ) -> tuple[bool, str | None]:
        nav = portfolio.nav
        if nav <= 0:
            return False, "Non-positive NAV"

        price = order.limit_price or 0.0
        if price <= 0:
            # Fall back to current mark from position if any; else we can't size-check.
            pos = portfolio.positions.get(order.symbol)
            price = pos.current_price if pos else 0.0
        order_notional = abs(order.quantity) * price
        is_exit = self._is_exit(order, portfolio)

        # Fat finger
        if order_notional > self.max_order_pct * nav:
            return (
                False,
                f"Fat finger: order notional {order_notional:.0f} "
                f"exceeds {self.max_order_pct:.1%} of NAV",
            )

        # Daily loss limit — only blocks new entries
        if not is_exit and portfolio.daily_pnl < -self.daily_loss_limit_pct * nav:
            return (
                False,
                f"Daily loss limit breached "
                f"({portfolio.daily_pnl / nav:.2%}); new entries halted",
            )

        # Max positions — only for genuine new entries (no existing symbol)
        if (
            order.symbol not in portfolio.positions
            and portfolio.position_count >= self.max_positions
        ):
            return (
                False,
                f"Max positions reached ({self.max_positions})",
            )

        # Max gross exposure — estimate incremental
        if not is_exit:
            new_gross = portfolio.gross_exposure + order_notional
            if new_gross > self.max_gross_exposure * nav:
                return (
                    False,
                    f"Gross exposure {new_gross / nav:.1%} exceeds "
                    f"{self.max_gross_exposure:.1%} limit",
                )

        # Max per-instrument
        existing_mv = 0.0
        pos = portfolio.positions.get(order.symbol)
        if pos is not None:
            existing_mv = abs(pos.market_value)
        projected = existing_mv + (0 if is_exit else order_notional)
        if projected > self.max_single_position_pct * nav:
            return (
                False,
                f"Position in {order.symbol} would be "
                f"{projected / nav:.1%} of NAV (> {self.max_single_position_pct:.1%})",
            )

        # Max crypto allocation
        if self._is_crypto(order.symbol) and not is_exit:
            crypto_exposure = sum(
                abs(p.market_value)
                for sym, p in portfolio.positions.items()
                if self._is_crypto(sym)
            )
            projected_crypto = crypto_exposure + order_notional
            if projected_crypto > self.max_crypto_pct * nav:
                return (
                    False,
                    f"Crypto allocation {projected_crypto / nav:.1%} "
                    f"exceeds {self.max_crypto_pct:.1%} limit",
                )

        return True, None

    # ── Portfolio health checks ────────────────────────────────────────

    def check_portfolio_health(
        self,
        portfolio: PortfolioState,
        *,
        last_model_retrain: datetime | None = None,
        last_broker_heartbeat: datetime | None = None,
        bar_rate_zscore: float | None = None,
        portfolio_correlation: float | None = None,
        now: datetime | None = None,
    ) -> list[CircuitBreakerAction]:
        now = now or datetime.now(timezone.utc)
        actions: list[CircuitBreakerAction] = []

        # Drawdown throttle (evaluate most severe first)
        dd = portfolio.drawdown
        if dd >= self.dd_halt:
            actions.append(
                CircuitBreakerAction(
                    action=Action.HALT_AND_FLATTEN,
                    reason=f"Drawdown {dd:.1%} at/above halt threshold {self.dd_halt:.1%}",
                    severity=Severity.EMERGENCY,
                    timestamp=now,
                )
            )
        elif dd >= self.dd_throttle_75:
            actions.append(
                CircuitBreakerAction(
                    action=Action.REDUCE_SIZE_75,
                    reason=f"Drawdown {dd:.1%} above {self.dd_throttle_75:.1%}",
                    severity=Severity.CRITICAL,
                    timestamp=now,
                )
            )
        elif dd >= self.dd_throttle_50:
            actions.append(
                CircuitBreakerAction(
                    action=Action.REDUCE_SIZE_50,
                    reason=f"Drawdown {dd:.1%} above {self.dd_throttle_50:.1%}",
                    severity=Severity.WARNING,
                    timestamp=now,
                )
            )

        # Model staleness
        if last_model_retrain is not None:
            age_days = (now - last_model_retrain).total_seconds() / 86400.0
            if age_days > self.model_stale_halt_days:
                actions.append(
                    CircuitBreakerAction(
                        action=Action.HALT,
                        reason=f"Model stale {age_days:.0f}d > {self.model_stale_halt_days}d",
                        severity=Severity.CRITICAL,
                        timestamp=now,
                    )
                )
            elif age_days > self.model_stale_warn_days:
                actions.append(
                    CircuitBreakerAction(
                        action=Action.REDUCE_SIZE_50,
                        reason=f"Model stale {age_days:.0f}d > {self.model_stale_warn_days}d",
                        severity=Severity.WARNING,
                        timestamp=now,
                    )
                )

        # Connectivity
        if last_broker_heartbeat is not None:
            gap = (now - last_broker_heartbeat).total_seconds()
            if gap > self.connectivity_timeout_s:
                actions.append(
                    CircuitBreakerAction(
                        action=Action.FLATTEN_ALL,
                        reason=f"Broker heartbeat stale {gap:.0f}s > {self.connectivity_timeout_s:.0f}s",
                        severity=Severity.CRITICAL,
                        timestamp=now,
                    )
                )

        # Data quality (bar formation rate)
        if bar_rate_zscore is not None and abs(bar_rate_zscore) > self.bar_rate_sigma:
            actions.append(
                CircuitBreakerAction(
                    action=Action.REDUCE_SIZE_50,
                    reason=f"Bar rate deviation {bar_rate_zscore:.1f}σ > {self.bar_rate_sigma}σ",
                    severity=Severity.WARNING,
                    timestamp=now,
                )
            )

        # Correlation spike
        if (
            portfolio_correlation is not None
            and portfolio_correlation > self.correlation_spike
        ):
            actions.append(
                CircuitBreakerAction(
                    action=Action.REDUCE_GROSS_50,
                    reason=f"Portfolio correlation {portfolio_correlation:.2f} > {self.correlation_spike:.2f}",
                    severity=Severity.WARNING,
                    timestamp=now,
                )
            )

        return actions

    # ── Dead-man switch ────────────────────────────────────────────────

    def check_dead_man_switch(
        self,
        last_operator_checkin: datetime | None,
        *,
        now: datetime | None = None,
    ) -> CircuitBreakerAction | None:
        if last_operator_checkin is None:
            return None
        now = now or datetime.now(timezone.utc)
        gap = now - last_operator_checkin
        if gap > timedelta(hours=self.dead_man_hours):
            hours = gap.total_seconds() / 3600.0
            return CircuitBreakerAction(
                action=Action.FLATTEN_ALL_AND_HALT,
                reason=f"No operator check-in for {hours:.1f}h > {self.dead_man_hours}h",
                severity=Severity.EMERGENCY,
                timestamp=now,
            )
        return None
