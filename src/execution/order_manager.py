"""Order Manager (Phase 5 / P5.05).

Top-level orchestration layer that connects:
    Phase 4 target portfolio  →  pre-trade checks  →  cost estimate  →
    algo selection  →  broker execution  →  portfolio state updates.

One cycle: update prices → health check → exits → rebalance → reconcile.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.backtesting.transaction_costs import CostEstimate, TransactionCostModel
from src.execution.algorithms import (
    BaseExecutionAlgo,
    IcebergAlgo,
    ImmediateAlgo,
    TWAPAlgo,
    VWAPAlgo,
    select_execution_algo,
)
from src.execution.broker_adapter import (
    BaseBrokerAdapter,
    reconcile_positions,
)
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.models import (
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)

log = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    order: Order
    accepted: bool
    rejection_reason: str | None = None
    cost_estimate: CostEstimate | None = None
    algo_class: type[BaseExecutionAlgo] | None = None


class OrderManager:
    """Orchestrates the execution pipeline from target portfolio to fills."""

    def __init__(
        self,
        broker: BaseBrokerAdapter,
        circuit_breakers: CircuitBreakerManager,
        cost_model: TransactionCostModel,
        portfolio: PortfolioState,
        *,
        asset_class_map: dict[str, str] | None = None,
        adv_map: dict[str, float] | None = None,
        volatility_map: dict[str, float] | None = None,
        expected_alpha_bps: dict[str, float] | None = None,
        min_trade_pct: float = 0.01,
        alpha_cost_ratio: float = 0.10,
    ) -> None:
        self.broker = broker
        self.circuit_breakers = circuit_breakers
        self.cost_model = cost_model
        self.portfolio = portfolio
        self.asset_class_map = asset_class_map or {}
        self.adv_map = adv_map or {}
        self.volatility_map = volatility_map or {}
        self.expected_alpha_bps = expected_alpha_bps or {}
        self.min_trade_pct = min_trade_pct
        self.alpha_cost_ratio = alpha_cost_ratio

    # ── Rebalance to target ────────────────────────────────────────────

    async def execute_target_portfolio(
        self,
        target: pd.DataFrame,
        prices: dict[str, float],
    ) -> list[Order]:
        """Execute trades needed to move from current positions to target weights."""
        trades = self._compute_required_trades(target, prices)
        # Exits first, entries second (free capital before deploying)
        trades.sort(key=lambda t: 0 if t["is_exit"] else 1)

        orders: list[Order] = []
        for trade in trades:
            outcome = await self._execute_single_trade(trade, prices)
            orders.append(outcome.order)
        return orders

    def _compute_required_trades(
        self, target: pd.DataFrame, prices: dict[str, float]
    ) -> list[dict]:
        """Diff target weights against current positions → list of trades."""
        nav = self.portfolio.nav
        target_weight_map = {
            row["symbol"]: float(row["target_weight"]) for _, row in target.iterrows()
        }
        current_weight_map: dict[str, float] = {}
        for sym, pos in self.portfolio.positions.items():
            mv = pos.market_value
            current_weight_map[sym] = mv / nav if nav > 0 else 0.0

        trades: list[dict] = []
        symbols = set(target_weight_map) | set(current_weight_map)
        for sym in symbols:
            tgt_w = target_weight_map.get(sym, 0.0)
            cur_w = current_weight_map.get(sym, 0.0)
            delta_w = tgt_w - cur_w
            if abs(delta_w) < self.min_trade_pct:
                continue
            price = float(prices.get(sym, 0.0))
            if price <= 0:
                continue
            delta_notional = delta_w * nav
            delta_shares = delta_notional / price
            side = 1 if delta_shares > 0 else -1
            is_exit = (
                sym in self.portfolio.positions
                and self.portfolio.positions[sym].side == -side
            )
            trades.append(
                {
                    "symbol": sym,
                    "side": side,
                    "quantity": abs(delta_shares),
                    "price": price,
                    "is_exit": is_exit,
                    "current_weight": cur_w,
                    "target_weight": tgt_w,
                }
            )
        return trades

    async def _execute_single_trade(
        self, trade: dict, prices: dict[str, float], urgency: str = "normal",
    ) -> TradeOutcome:
        order = self._build_order(trade)

        # Pre-trade circuit breakers
        allowed, reason = self.circuit_breakers.check_pre_trade(order, self.portfolio)
        if not allowed:
            order.status = OrderStatus.REJECTED
            log.warning("Order rejected: %s (%s)", order.symbol, reason)
            return TradeOutcome(order=order, accepted=False, rejection_reason=reason)

        # Cost estimate
        cost_est = self._estimate_cost(order, trade["price"])
        if cost_est is not None:
            alpha_bps = self.expected_alpha_bps.get(order.symbol)
            if alpha_bps is not None and alpha_bps > 0:
                ratio = cost_est.cost_bps / alpha_bps
                if ratio > self.alpha_cost_ratio:
                    log.warning(
                        "Cost %.1fbps > %.0f%% of alpha %.1fbps on %s",
                        cost_est.cost_bps, self.alpha_cost_ratio * 100,
                        alpha_bps, order.symbol,
                    )

        # Algo selection + execution
        adv = self.adv_map.get(order.symbol, float("inf"))
        asset_class = self.asset_class_map.get(order.symbol, "equities")
        algo_cls = select_execution_algo(
            order, adv=adv, urgency=urgency, asset_class=asset_class,
        )
        algo = self._build_algo(algo_cls, order, urgency)

        self.portfolio.open_orders.append(order)
        try:
            await algo.execute()
        finally:
            for fill in algo.fills:
                self._record_parent_fill(order, fill)
            if order in self.portfolio.open_orders and order.is_complete:
                self.portfolio.open_orders.remove(order)

        return TradeOutcome(
            order=order, accepted=True, cost_estimate=cost_est, algo_class=algo_cls,
        )

    def _build_order(self, trade: dict, order_type: OrderType = OrderType.LIMIT) -> Order:
        return Order(
            order_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol=trade["symbol"],
            side=trade["side"],
            order_type=order_type,
            quantity=trade["quantity"] * trade["side"],
            limit_price=trade["price"],
        )

    def _build_algo(
        self, algo_cls: type[BaseExecutionAlgo], order: Order, urgency: str,
    ) -> BaseExecutionAlgo:
        if algo_cls is ImmediateAlgo:
            return ImmediateAlgo(
                order, self.broker, timeout_seconds=0.0,
                use_market_fallback=(urgency == "high"),
            )
        if algo_cls is TWAPAlgo:
            return TWAPAlgo(
                order, self.broker, n_slices=5, duration_minutes=0.0,
                child_timeout_seconds=0.0, time_scale=0.0,
            )
        if algo_cls is VWAPAlgo:
            return VWAPAlgo(
                order, self.broker, duration_minutes=0.0,
                child_timeout_seconds=0.0, time_scale=0.0,
            )
        if algo_cls is IcebergAlgo:
            return IcebergAlgo(
                order, self.broker, visible_pct=0.10,
                child_timeout_seconds=0.0, time_scale=0.0,
            )
        raise ValueError(f"Unsupported algo class: {algo_cls}")

    def _record_parent_fill(self, parent: Order, fill) -> None:
        """Apply a child fill to portfolio ledger and roll it up to parent.

        PortfolioState.record_fill looks up the symbol via open_orders and
        the child fill's order_id doesn't match the parent, so we do the
        ledger update directly here. Also attach the fill to the parent so
        its status/fill metrics stay consistent with what the broker did.
        """
        parent.fills.append(fill)
        parent.filled_quantity += fill.quantity
        parent.updated_at = fill.timestamp
        if abs(parent.filled_quantity) >= abs(parent.quantity) - 1e-9:
            parent.status = OrderStatus.FILLED
        elif parent.filled_quantity != 0:
            parent.status = OrderStatus.PARTIAL_FILL

        pf = self.portfolio
        pf.cash -= fill.quantity * fill.price + fill.commission
        pos = pf.positions.get(parent.symbol)
        if pos is None:
            side = 1 if fill.quantity > 0 else -1
            pf.positions[parent.symbol] = Position(
                symbol=parent.symbol,
                side=side,
                quantity=abs(fill.quantity),
                avg_entry_price=fill.price,
                entry_timestamp=fill.timestamp,
                signal_family=parent.signal_family,
                current_price=fill.price,
                stop_loss=parent.stop_price,
                take_profit=parent.take_profit_price,
            )
        else:
            pos.apply_fill(fill)
            if pos.quantity <= 1e-12:
                del pf.positions[parent.symbol]
        pf._recompute_exposures()
        pf._recompute_nav()

    def _estimate_cost(self, order: Order, price: float) -> CostEstimate | None:
        try:
            asset_class = self.asset_class_map.get(order.symbol, "equities")
            adv = self.adv_map.get(order.symbol, 1_000_000.0)
            vol = self.volatility_map.get(order.symbol, 0.02)
            return self.cost_model.estimate(
                order_size=abs(order.quantity),
                price=price,
                adv=adv,
                volatility=vol,
                asset_class=asset_class,
            )
        except Exception as exc:
            log.debug("Cost estimate failed for %s: %s", order.symbol, exc)
            return None

    # ── Exit checks (triple-barrier) ───────────────────────────────────

    async def check_position_exits(
        self, prices: dict[str, float]
    ) -> list[Order]:
        orders: list[Order] = []
        now = datetime.now(timezone.utc)
        # Iterate over a copy since record_fill may delete positions
        for symbol, pos in list(self.portfolio.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            pos.update_price(price)

            exit_reason: str | None = None
            urgency = "normal"
            order_type = OrderType.LIMIT

            # Stop loss — urgent market exit
            if pos.stop_loss is not None:
                if (pos.side > 0 and price <= pos.stop_loss) or (
                    pos.side < 0 and price >= pos.stop_loss
                ):
                    exit_reason = "stop_loss"
                    urgency = "high"
                    order_type = OrderType.MARKET
            # Take profit — limit exit
            if exit_reason is None and pos.take_profit is not None:
                if (pos.side > 0 and price >= pos.take_profit) or (
                    pos.side < 0 and price <= pos.take_profit
                ):
                    exit_reason = "take_profit"
                    order_type = OrderType.LIMIT
            # Vertical barrier — time expiry
            if exit_reason is None and pos.vertical_barrier is not None:
                if now >= pos.vertical_barrier:
                    exit_reason = "time_expiry"
                    order_type = OrderType.LIMIT

            if exit_reason is None:
                continue

            trade = {
                "symbol": symbol,
                "side": -pos.side,
                "quantity": pos.quantity,
                "price": price,
                "is_exit": True,
                "current_weight": 0.0,
                "target_weight": 0.0,
            }
            order = self._build_order(trade, order_type=order_type)
            order.signal_family = f"exit:{exit_reason}"

            allowed, reason = self.circuit_breakers.check_pre_trade(order, self.portfolio)
            if not allowed:
                order.status = OrderStatus.REJECTED
                orders.append(order)
                log.warning("Exit rejected for %s: %s", symbol, reason)
                continue

            adv = self.adv_map.get(symbol, float("inf"))
            asset_class = self.asset_class_map.get(symbol, "equities")
            algo_cls = select_execution_algo(
                order, adv=adv, urgency=urgency, asset_class=asset_class,
            )
            algo = self._build_algo(algo_cls, order, urgency)

            self.portfolio.open_orders.append(order)
            try:
                await algo.execute()
            finally:
                for fill in algo.fills:
                    self._record_parent_fill(order, fill)
                if order in self.portfolio.open_orders and order.is_complete:
                    self.portfolio.open_orders.remove(order)
            orders.append(order)

        return orders

    # ── Reconciliation ─────────────────────────────────────────────────

    async def reconcile_positions(self) -> list[dict]:
        broker_positions = await self.broker.get_positions()
        diffs = reconcile_positions(self.portfolio, broker_positions)
        return [
            {
                "symbol": d.symbol,
                "internal_signed_qty": d.internal_side * d.internal_qty,
                "broker_signed_qty": d.broker_side * d.broker_qty,
                "delta": d.delta,
            }
            for d in diffs
        ]

    # ── Full cycle ─────────────────────────────────────────────────────

    async def run_cycle(
        self,
        prices: dict[str, float],
        *,
        signals: pd.DataFrame | None = None,
        target: pd.DataFrame | None = None,
    ) -> dict:
        summary: dict[str, Any] = {
            "exits": [],
            "rebalance_orders": [],
            "breakers": [],
            "reconciliation": [],
        }

        # a. update prices
        self.portfolio.update_prices(prices)

        # b. portfolio health (circuit breakers)
        breakers = self.circuit_breakers.check_portfolio_health(self.portfolio)
        summary["breakers"] = breakers
        halt = any(
            b.action in ("HALT", "HALT_AND_FLATTEN", "FLATTEN_ALL_AND_HALT")
            for b in breakers
        )

        # c. position exits
        summary["exits"] = await self.check_position_exits(prices)

        # d. rebalance
        if target is not None and not halt:
            summary["rebalance_orders"] = await self.execute_target_portfolio(
                target, prices
            )

        # e. reconcile
        summary["reconciliation"] = await self.reconcile_positions()
        return summary
