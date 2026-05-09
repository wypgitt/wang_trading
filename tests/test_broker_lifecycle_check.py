"""Tests for the paper broker lifecycle probe."""

from __future__ import annotations

import asyncio

from scripts.broker_lifecycle_check import run_lifecycle_check


def test_paper_broker_lifecycle_check_passes_without_live_credentials():
    result = asyncio.run(run_lifecycle_check(symbol="AAPL", quantity=2.0))

    assert result["status"] == "passed"
    assert result["paper_only"] is True
    assert result["live_orders_sent"] == 0
    assert result["broker"] == "PaperBrokerAdapter"
    assert {step["name"] for step in result["steps"]} == {
        "heartbeat",
        "account",
        "quote",
        "submit_market_fill",
        "positions_after_submit",
        "cancel_open_order",
        "flatten_and_verify_flat",
    }
    assert result["steps"][-1]["ok"] is True
    assert result["steps"][-1]["positions"] == []
