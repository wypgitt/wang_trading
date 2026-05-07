#!/usr/bin/env python3
"""Paper broker lifecycle probe.

Exercises the broker surface that operators care about before a paper
production rehearsal: heartbeat, account, quote, submit/fill, cancel,
flatten, and final flat verification. The probe always forces the paper
adapter, even when a live config is supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.bootstrap import load_runtime_config
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.broker_factory import BrokerFactory
from src.execution.models import Order, OrderStatus, OrderType

log = logging.getLogger(__name__)


def _paper_runtime_config(
    runtime: dict[str, Any],
    *,
    symbol: str,
    price: float,
) -> dict[str, Any]:
    config = dict(runtime or {})
    broker = dict(config.get("broker") or {})
    broker["adapter"] = "paper"
    broker.setdefault("fill_delay_ms", 0)
    config["broker"] = broker
    config["dry_run"] = True

    prices = dict(config.get("paper_prices") or {})
    prices.setdefault(symbol, float(price))
    config["paper_prices"] = prices
    return config


def _order(
    symbol: str,
    *,
    side: int,
    quantity: float,
    order_type: OrderType,
    limit_price: float | None = None,
) -> Order:
    qty = abs(float(quantity)) * (1 if side > 0 else -1)
    return Order(
        order_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        side=1 if qty > 0 else -1,
        order_type=order_type,
        quantity=qty,
        limit_price=limit_price,
        signal_family="lifecycle_probe",
    )


def _position_rows(positions: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, pos in sorted(positions.items()):
        rows.append({
            "symbol": symbol,
            "side": int(pos.side),
            "quantity": float(pos.quantity),
            "signed_qty": float(pos.side * pos.quantity),
            "market_value": float(pos.market_value),
        })
    return rows


def _order_row(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "status": order.status.value,
        "quantity": float(order.quantity),
        "filled_quantity": float(order.filled_quantity),
        "fills": len(order.fills),
        "avg_fill_price": float(order.avg_fill_price),
    }


async def run_lifecycle_check(
    *,
    config_path: str | None = None,
    symbol: str = "AAPL",
    asset_class: str = "equities",
    quantity: float = 1.0,
    price: float = 100.0,
) -> dict[str, Any]:
    """Run a paper-only broker lifecycle check and return a JSON-safe summary."""

    runtime = (
        load_runtime_config(config_path, default_name="live_trading")
        if config_path else {}
    )
    config = _paper_runtime_config(runtime, symbol=symbol, price=price)
    factory = BrokerFactory(config)
    broker = factory.get_broker(symbol, asset_class=asset_class)
    if not isinstance(broker, PaperBrokerAdapter):
        raise RuntimeError(
            f"broker lifecycle check must use PaperBrokerAdapter, got {type(broker).__name__}"
        )

    steps: list[dict[str, Any]] = []

    heartbeat = await broker.heartbeat()
    steps.append({"name": "heartbeat", "ok": bool(heartbeat)})

    account_before = await broker.get_account()
    steps.append({
        "name": "account",
        "ok": "cash" in account_before and "nav" in account_before,
        "account": account_before,
    })

    quote = await broker.get_quote(symbol)
    steps.append({
        "name": "quote",
        "ok": float(quote.get("mid", 0.0) or 0.0) > 0.0,
        "quote": quote,
    })

    market_order = await broker.submit_order(
        _order(symbol, side=1, quantity=quantity, order_type=OrderType.MARKET),
    )
    market_fills = await broker.poll_fills(market_order.order_id)
    steps.append({
        "name": "submit_market_fill",
        "ok": (
            market_order.status == OrderStatus.FILLED
            and abs(market_order.filled_quantity - abs(float(quantity))) < 1e-9
            and len(market_fills) == 1
        ),
        "order": _order_row(market_order),
    })

    positions_after_submit = await broker.get_positions()
    steps.append({
        "name": "positions_after_submit",
        "ok": bool(positions_after_submit),
        "positions": _position_rows(positions_after_submit),
    })

    stale_limit = await broker.submit_order(
        _order(
            symbol,
            side=1,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=float(quote["mid"]) * 0.5,
        ),
    )
    cancelled = await broker.cancel_order(stale_limit.order_id)
    steps.append({
        "name": "cancel_open_order",
        "ok": cancelled and stale_limit.status == OrderStatus.CANCELLED,
        "order": _order_row(stale_limit),
    })

    flatten_orders: list[dict[str, Any]] = []
    for pos in list((await broker.get_positions()).values()):
        flatten = await broker.submit_order(
            _order(
                pos.symbol,
                side=-int(pos.side),
                quantity=float(pos.quantity),
                order_type=OrderType.MARKET,
            ),
        )
        flatten_orders.append(_order_row(flatten))

    final_positions = await broker.get_positions()
    active_orders = [
        _order_row(order)
        for order in broker.orders.values()
        if order.is_active
    ]
    steps.append({
        "name": "flatten_and_verify_flat",
        "ok": not final_positions and not active_orders,
        "flatten_orders": flatten_orders,
        "positions": _position_rows(final_positions),
        "active_orders": active_orders,
    })

    account_after = await broker.get_account()
    passed = all(bool(step["ok"]) for step in steps)
    return {
        "status": "passed" if passed else "failed",
        "paper_only": True,
        "live_orders_sent": 0,
        "broker": type(broker).__name__,
        "symbol": symbol,
        "asset_class": asset_class,
        "steps": steps,
        "account_before": account_before,
        "account_after": account_after,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="broker_lifecycle_check")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument(
        "--asset-class",
        choices=["equities", "crypto", "futures"],
        default="equities",
    )
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--price", type=float, default=100.0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    try:
        result = asyncio.run(run_lifecycle_check(
            config_path=args.config,
            symbol=args.symbol,
            asset_class=args.asset_class,
            quantity=args.quantity,
            price=args.price,
        ))
    except Exception as exc:  # noqa: BLE001
        log.exception("broker lifecycle check failed")
        result = {
            "status": "failed",
            "paper_only": True,
            "live_orders_sent": 0,
            "error": str(exc),
        }
    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
