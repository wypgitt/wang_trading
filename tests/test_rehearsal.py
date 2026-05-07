"""Tests for paper-production rehearsal recording."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.models import Order, OrderType
from src.execution.rehearsal import OrderRehearsalRecorder, RecordingBrokerAdapter


def test_recording_broker_records_would_be_orders(tmp_path):
    record_path = tmp_path / "rehearsal.jsonl"
    broker = RecordingBrokerAdapter(
        PaperBrokerAdapter(fill_delay_ms=0, price_feed=lambda _: 100.0),
        OrderRehearsalRecorder(record_path),
    )
    order = Order(
        order_id="order-1",
        timestamp=datetime.now(timezone.utc),
        symbol="AAPL",
        side=1,
        order_type=OrderType.MARKET,
        quantity=10.0,
    )

    filled = asyncio.run(broker.submit_order(order))

    assert filled.filled_quantity == 10.0
    rows = [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "would_be_order",
        "paper_order_result",
        "paper_fill_result",
    ]
    assert rows[0]["payload"]["symbol"] == "AAPL"
    assert rows[0]["payload"]["quantity"] == 10.0
