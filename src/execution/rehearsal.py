"""Paper-production rehearsal support.

Rehearsal mode runs the live pipeline with paper brokers and records every
order that would have crossed the broker boundary. It is intentionally a
broker wrapper so the rest of the live bootstrap stays unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.execution.broker_adapter import BaseBrokerAdapter
from src.execution.models import Order, Position


class OrderRehearsalRecorder:
    """Append-only JSONL recorder for would-be live broker actions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")


class RecordingBrokerAdapter(BaseBrokerAdapter):
    """Broker wrapper that records submit/cancel/order-status/fill outcomes."""

    def __init__(self, inner: BaseBrokerAdapter, recorder: OrderRehearsalRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    async def submit_order(self, order: Order) -> Order:
        self.recorder.record("would_be_order", _order_payload(order))
        result = await self.inner.submit_order(order)
        self.recorder.record("paper_order_result", _order_payload(result))
        for fill in getattr(result, "fills", []) or []:
            self.recorder.record("paper_fill_result", _fill_payload(fill, result.symbol))
        return result

    async def cancel_order(self, order_id: str) -> bool:
        self.recorder.record("would_be_cancel", {"order_id": order_id})
        result = await self.inner.cancel_order(order_id)
        self.recorder.record("paper_cancel_result", {
            "order_id": order_id,
            "cancelled": bool(result),
        })
        return result

    async def get_order_status(self, order_id: str) -> Order:
        return await self.inner.get_order_status(order_id)

    async def get_open_orders(self, symbols: list[str] | None = None) -> list[Order]:
        return await self.inner.get_open_orders(symbols)

    async def get_positions(self) -> dict[str, Position]:
        return await self.inner.get_positions()

    async def get_account(self) -> dict[str, Any]:
        return await self.inner.get_account()

    async def get_quote(self, symbol: str) -> dict[str, float]:
        return await self.inner.get_quote(symbol)

    async def heartbeat(self) -> bool:
        return await self.inner.heartbeat()

    async def poll_fills(self, order_id: str):
        return await self.inner.poll_fills(order_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _order_payload(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "timestamp": order.timestamp,
        "symbol": order.symbol,
        "side": order.side,
        "order_type": getattr(order.order_type, "value", order.order_type),
        "quantity": order.quantity,
        "filled_quantity": order.filled_quantity,
        "status": getattr(order.status, "value", order.status),
        "limit_price": order.limit_price,
        "signal_family": order.signal_family,
        "meta_label_prob": order.meta_label_prob,
        "fills": [_fill_payload(fill, order.symbol) for fill in order.fills],
    }


def _fill_payload(fill: Any, symbol: str = "") -> dict[str, Any]:
    return {
        "fill_id": getattr(fill, "fill_id", ""),
        "order_id": getattr(fill, "order_id", ""),
        "timestamp": getattr(fill, "timestamp", None),
        "symbol": symbol,
        "price": getattr(fill, "price", None),
        "quantity": getattr(fill, "quantity", None),
        "commission": getattr(fill, "commission", None),
        "exchange": getattr(fill, "exchange", ""),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)
