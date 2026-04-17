"""Tests for the compliance audit logger (P6.14)."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.audit_log import (
    AUDIT_LOG_DDL,
    AuditEntry,
    ComplianceAuditLogger,
    EventType,
)
from src.execution.models import Fill, Order, OrderStatus, OrderType


KEY = "very-secret-signing-key"


def _order(symbol: str = "AAPL") -> Order:
    return Order(
        order_id="o1", timestamp=datetime.now(timezone.utc),
        symbol=symbol, side=1, order_type=OrderType.MARKET, quantity=10,
    )


# ── Signature / verification ───────────────────────────────────────────

class TestSignature:
    def test_entry_has_signature(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        entry = logger.log_signal("AAPL", "trend", 1)
        assert entry.signature
        assert len(entry.signature) == 64  # hex-encoded SHA256

    def test_verify_untampered(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        entry = logger.log_signal("AAPL", "trend", 1)
        assert logger.verify_entry(entry) is True

    def test_tampered_entry_fails(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        entry = logger.log_signal("AAPL", "trend", 1)
        entry.decision_output["side"] = -1  # tamper
        assert logger.verify_entry(entry) is False

    def test_wrong_key_fails(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        entry = logger.log_signal("AAPL", "trend", 1)
        other = ComplianceAuditLogger(signing_key="other-key")
        assert other.verify_entry(entry) is False


# ── Chain verification ─────────────────────────────────────────────────

class TestChain:
    def test_clean_chain_verifies(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        logger.log_signal("AAPL", "trend", 1)
        logger.log_meta_label("AAPL", 0.72, model_version="v1")
        logger.log_order(_order())
        result = logger.verify_chain()
        assert result["ok"] is True
        assert result["total"] == 3

    def test_broken_chain_detected(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        logger.log_signal("AAPL", "trend", 1)
        middle = logger.log_meta_label("AAPL", 0.72)
        logger.log_order(_order())
        middle.decision_output["meta_prob"] = 0.99  # tamper mid-chain
        result = logger.verify_chain()
        assert result["ok"] is False
        assert any(middle.entry_id in b for b in result["broken"])


# ── Event sugar ────────────────────────────────────────────────────────

class TestEventLogging:
    def test_log_fill_captures_fields(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        fill = Fill(
            fill_id="f1", order_id="o1", timestamp=datetime.now(timezone.utc),
            price=100.0, quantity=10, commission=0.1, exchange="PAPER",
        )
        entry = logger.log_fill(fill, symbol="AAPL")
        assert entry.event_type == EventType.FILL_RECEIVED.value
        assert entry.decision_output["price"] == 100.0
        assert entry.decision_context["fill_id"] == "f1"

    def test_log_operator_action(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        entry = logger.log_operator_action(
            "approve_rl_promotion", operator="alice",
            details={"ticket": "OPS-123"},
        )
        assert entry.symbol == "SYSTEM"
        assert entry.decision_output == {"action": "approve_rl_promotion"}

    def test_log_breaker_duck_typed(self):
        breaker = MagicMock(
            reason="drawdown", action="HALT_AND_FLATTEN",
            breaker_type="NAV_DRAWDOWN", symbol=None,
        )
        logger = ComplianceAuditLogger(signing_key=KEY)
        entry = logger.log_breaker(breaker)
        assert entry.decision_output["action"] == "HALT_AND_FLATTEN"
        assert entry.symbol == "SYSTEM"


# ── Query + export ─────────────────────────────────────────────────────

class TestQueryExport:
    def test_query_filters_by_symbol(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        logger.log_signal("AAPL", "trend", 1)
        logger.log_signal("MSFT", "mean_rev", -1)
        logger.log_signal("AAPL", "carry", 1)
        df = logger.query({"symbol": "AAPL"})
        assert len(df) == 2
        assert set(df["symbol"]) == {"AAPL"}

    def test_query_filters_by_event_type(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        logger.log_signal("AAPL", "trend", 1)
        logger.log_order(_order())
        df = logger.query({"event_type": EventType.ORDER_SUBMITTED.value})
        assert len(df) == 1

    def test_export_csv(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        logger.log_signal("AAPL", "trend", 1)
        logger.log_meta_label("MSFT", 0.8)
        csv_text = logger.export_report(format="csv")
        # Parse back and confirm row count
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 entries
        assert "signature" in rows[0]

    def test_export_json(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        logger.log_signal("AAPL", "trend", 1)
        js = logger.export_report(format="json")
        assert "\"event_type\"" in js

    def test_empty_export_has_header(self):
        logger = ComplianceAuditLogger(signing_key=KEY)
        text = logger.export_report(format="csv")
        assert "entry_id" in text


# ── Storage integration ───────────────────────────────────────────────

class TestStorageDuckTyping:
    def test_calls_save_audit_entry(self):
        storage = MagicMock()
        storage.save_audit_entry = MagicMock()
        logger = ComplianceAuditLogger(storage=storage, signing_key=KEY)
        logger.log_signal("AAPL", "trend", 1)
        storage.save_audit_entry.assert_called_once()

    def test_ddl_has_hypertable(self):
        assert "create_hypertable" in AUDIT_LOG_DDL
        assert "audit_log" in AUDIT_LOG_DDL
