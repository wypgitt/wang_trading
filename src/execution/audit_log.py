"""Compliance audit log (P6.14).

Every trading decision is logged with an HMAC-SHA256 signature so the log
is tamper-evident. Entries chain via ``prev_signature`` — modifying any
entry's contents breaks the chain downstream.

Storage is duck-typed; the project's ``ExecutionStorage`` can drop in, but
an in-memory fallback is used in tests and for CLI lookups.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Event types ──────────────────────────────────────────────────────────

class EventType(str, Enum):
    SIGNAL_GENERATED = "signal_generated"
    META_LABEL_PREDICTED = "meta_label_predicted"
    BET_SIZED = "bet_sized"
    ORDER_SUBMITTED = "order_submitted"
    FILL_RECEIVED = "fill_received"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    BREAKER_TRIGGERED = "breaker_triggered"
    PHASE_PROMOTED = "phase_promoted"
    RL_SHADOW_DECISION = "rl_shadow_decision"
    OPERATOR_ACTION = "operator_action"


# ── Entry ────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    entry_id: str
    timestamp: datetime
    event_type: str
    symbol: str
    decision_context: dict[str, Any] = field(default_factory=dict)
    decision_output: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None
    prev_signature: str = ""
    signature: str = ""

    # ── Canonical payload ────────────────────────────────────────────

    def canonical_payload(self) -> bytes:
        """Stable, signable representation. Excludes the signature itself."""
        d = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "symbol": self.symbol,
            "decision_context": self.decision_context,
            "decision_output": self.decision_output,
            "model_version": self.model_version,
            "prev_signature": self.prev_signature,
        }
        return json.dumps(d, sort_keys=True, default=str).encode("utf-8")

    def compute_signature(self, signing_key: str) -> str:
        mac = hmac.new(
            signing_key.encode("utf-8"),
            self.canonical_payload(),
            hashlib.sha256,
        )
        return mac.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "symbol": self.symbol,
            "decision_context": self.decision_context,
            "decision_output": self.decision_output,
            "model_version": self.model_version,
            "prev_signature": self.prev_signature,
            "signature": self.signature,
        }


# ── In-memory fallback storage ───────────────────────────────────────────

class _InMemoryAuditStore:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def save_audit_entry(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    def fetch_audit_entries(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        event_type: str | None = None,
        symbol: str | None = None,
    ) -> list[AuditEntry]:
        out = []
        for e in self._entries:
            if start is not None and e.timestamp < start:
                continue
            if end is not None and e.timestamp > end:
                continue
            if event_type is not None and e.event_type != event_type:
                continue
            if symbol is not None and e.symbol != symbol:
                continue
            out.append(e)
        return out


# ── Logger ───────────────────────────────────────────────────────────────

class ComplianceAuditLogger:
    """Append-only, HMAC-signed audit trail over ``storage``."""

    def __init__(
        self,
        *,
        storage: Any | None = None,
        signing_key: str,
    ) -> None:
        if not signing_key:
            raise ValueError("signing_key must be non-empty")
        self.signing_key = signing_key
        self.storage = storage or _InMemoryAuditStore()
        self._last_signature: str = ""
        self._chain: list[AuditEntry] = []  # in-memory chain for verify_chain

    # ── Generic append ───────────────────────────────────────────────

    def _append(
        self,
        event_type: str,
        symbol: str,
        context: dict[str, Any],
        output: dict[str, Any],
        *,
        model_version: str | None = None,
        timestamp: datetime | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=timestamp or _utcnow(),
            event_type=event_type,
            symbol=symbol,
            decision_context=dict(context or {}),
            decision_output=dict(output or {}),
            model_version=model_version,
            prev_signature=self._last_signature,
        )
        entry.signature = entry.compute_signature(self.signing_key)
        self._last_signature = entry.signature
        self._chain.append(entry)
        save = getattr(self.storage, "save_audit_entry", None)
        if callable(save):
            try:
                save(entry)
            except Exception:  # pragma: no cover
                log.exception("audit storage save failed")
        return entry

    # ── Event-specific sugar ─────────────────────────────────────────

    def log_signal(self, symbol: str, family: str, side: int, *, context: dict | None = None) -> AuditEntry:
        return self._append(EventType.SIGNAL_GENERATED.value, symbol,
                            context or {}, {"family": family, "side": side})

    def log_meta_label(self, symbol: str, prob: float, *, context: dict | None = None,
                       model_version: str | None = None) -> AuditEntry:
        return self._append(EventType.META_LABEL_PREDICTED.value, symbol,
                            context or {}, {"meta_prob": float(prob)},
                            model_version=model_version)

    def log_bet_size(self, symbol: str, size: float, *, breakdown: dict | None = None) -> AuditEntry:
        return self._append(EventType.BET_SIZED.value, symbol,
                            breakdown or {}, {"final_size": float(size)})

    def log_order(self, order: Any) -> AuditEntry:
        ctx = {
            "order_id": getattr(order, "order_id", None),
            "order_type": getattr(getattr(order, "order_type", None), "value", None),
            "quantity": getattr(order, "quantity", None),
            "limit_price": getattr(order, "limit_price", None),
        }
        out = {"status": getattr(getattr(order, "status", None), "value", None)}
        return self._append(EventType.ORDER_SUBMITTED.value,
                            getattr(order, "symbol", ""), ctx, out)

    def log_fill(self, fill: Any, *, symbol: str = "") -> AuditEntry:
        ctx = {
            "order_id": getattr(fill, "order_id", None),
            "fill_id": getattr(fill, "fill_id", None),
        }
        out = {
            "price": getattr(fill, "price", None),
            "quantity": getattr(fill, "quantity", None),
            "commission": getattr(fill, "commission", None),
        }
        return self._append(EventType.FILL_RECEIVED.value, symbol, ctx, out)

    def log_breaker(self, breaker: Any) -> AuditEntry:
        ctx = {"reason": getattr(breaker, "reason", None)}
        out = {"action": getattr(breaker, "action", None),
               "breaker_type": getattr(breaker, "breaker_type", None)}
        return self._append(EventType.BREAKER_TRIGGERED.value,
                            getattr(breaker, "symbol", "SYSTEM") or "SYSTEM",
                            ctx, out)

    def log_operator_action(self, action: str, *, operator: str = "",
                            details: dict | None = None) -> AuditEntry:
        return self._append(EventType.OPERATOR_ACTION.value, "SYSTEM",
                            {"operator": operator, **(details or {})},
                            {"action": action})

    def log_phase_promotion(self, phase: dict[str, Any]) -> AuditEntry:
        return self._append(EventType.PHASE_PROMOTED.value, "SYSTEM",
                            {}, phase)

    def log_rl_shadow(self, hrp_target: dict, rl_target: dict,
                      executed_target: dict) -> AuditEntry:
        return self._append(
            EventType.RL_SHADOW_DECISION.value, "SYSTEM",
            {"hrp_target": hrp_target, "rl_target": rl_target},
            {"executed_target": executed_target},
        )

    # ── Verification ─────────────────────────────────────────────────

    def verify_entry(self, entry: AuditEntry) -> bool:
        expected = entry.compute_signature(self.signing_key)
        return hmac.compare_digest(expected, entry.signature)

    def verify_chain(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        entries = self._filter(self._chain, start=start, end=end)
        broken: list[str] = []
        prev = self._chain[0].prev_signature if self._chain else ""
        # If start is mid-chain, walk from the earliest chain entry to
        # pick up the correct prev_signature before the requested window.
        if start is not None and self._chain:
            prev = ""
            for e in self._chain:
                if e.timestamp >= start:
                    break
                prev = e.signature
        for e in entries:
            if not self.verify_entry(e):
                broken.append(f"{e.entry_id}: bad signature")
                continue
            if e.prev_signature != prev:
                broken.append(f"{e.entry_id}: prev_signature mismatch")
            prev = e.signature
        return {"total": len(entries), "broken": broken, "ok": not broken}

    # ── Query / export ───────────────────────────────────────────────

    def query(self, filters: dict[str, Any] | None = None) -> pd.DataFrame:
        filters = filters or {}
        fetch = getattr(self.storage, "fetch_audit_entries", None)
        if callable(fetch):
            entries = fetch(**filters)
        else:
            entries = self._filter(self._chain, **filters)
        return pd.DataFrame([e.to_dict() for e in entries])

    def export_report(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        format: str = "csv",
    ) -> str:
        df = self.query({"start": start, "end": end})
        if format == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            if df.empty:
                writer.writerow([
                    "entry_id", "timestamp", "event_type", "symbol",
                    "model_version", "signature",
                ])
            else:
                df.to_csv(buf, index=False)
            return buf.getvalue()
        if format == "json":
            return df.to_json(orient="records", date_format="iso")
        raise ValueError(f"unsupported format: {format!r}")

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _filter(
        entries: Iterable[AuditEntry],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        event_type: str | None = None,
        symbol: str | None = None,
    ) -> list[AuditEntry]:
        out = []
        for e in entries:
            if start is not None and e.timestamp < start:
                continue
            if end is not None and e.timestamp > end:
                continue
            if event_type is not None and e.event_type != event_type:
                continue
            if symbol is not None and e.symbol != symbol:
                continue
            out.append(e)
        return out


# ── Database DDL ─────────────────────────────────────────────────────────

AUDIT_LOG_DDL = """
-- TimescaleDB audit_log hypertable (P6.14).
-- Append-only: enforce via row-level trigger or RLS; here we leave it to
-- application-level discipline plus HMAC chaining.

CREATE TABLE IF NOT EXISTS audit_log (
    entry_id           UUID        PRIMARY KEY,
    ts                 TIMESTAMPTZ NOT NULL,
    event_type         TEXT        NOT NULL,
    symbol             TEXT        NOT NULL,
    decision_context   JSONB       NOT NULL DEFAULT '{}',
    decision_output    JSONB       NOT NULL DEFAULT '{}',
    model_version      TEXT        NULL,
    prev_signature     TEXT        NOT NULL DEFAULT '',
    signature          TEXT        NOT NULL
);

-- Hypertable partitioned by timestamp for long-range retention queries.
SELECT create_hypertable('audit_log', 'ts', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day');

CREATE INDEX IF NOT EXISTS audit_log_event_ts_idx
    ON audit_log (event_type, ts DESC);
CREATE INDEX IF NOT EXISTS audit_log_symbol_ts_idx
    ON audit_log (symbol, ts DESC);
"""
