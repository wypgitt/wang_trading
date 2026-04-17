"""Tests for disaster recovery + state persistence (P6.13)."""

from __future__ import annotations

import asyncio
import pickle
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.disaster_recovery import (
    RecoveryManager,
    SnapshotManager,
    StateSnapshot,
)
from src.execution.models import Order, OrderStatus, OrderType, PortfolioState, Position


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _pos(sym: str = "AAPL", qty: float = 100, side: int = 1) -> Position:
    return Position(
        symbol=sym, side=side, quantity=qty, avg_entry_price=100.0,
        entry_timestamp=_ts(), signal_family="",
    )


def _order(sym: str = "AAPL") -> Order:
    return Order(
        order_id="o1", timestamp=_ts(), symbol=sym, side=1,
        order_type=OrderType.LIMIT, quantity=10, limit_price=100.0,
        status=OrderStatus.SUBMITTED,
    )


# ── StateSnapshot ───────────────────────────────────────────────────────

class TestStateSnapshot:
    def test_roundtrip(self, tmp_path):
        pf = PortfolioState(cash=100_000.0, positions={"AAPL": _pos()})
        snap = StateSnapshot(
            timestamp=_ts(), portfolio=pf, open_orders=[_order()],
            active_positions=pf.positions, deployment_phase={"phase_id": 1},
            model_version="v1", last_successful_cycle=42,
        )
        path = tmp_path / "snap.pkl"
        snap.save(path)
        assert path.exists()
        loaded = StateSnapshot.load(path)
        assert loaded.last_successful_cycle == 42
        assert loaded.deployment_phase == {"phase_id": 1}
        assert "AAPL" in loaded.active_positions

    def test_checksum_detects_tampering(self, tmp_path):
        pf = PortfolioState(cash=50_000.0)
        snap = StateSnapshot(timestamp=_ts(), portfolio=pf)
        path = tmp_path / "snap.pkl"
        snap.save(path)

        # Corrupt the payload bytes while keeping the checksum.
        with path.open("rb") as fh:
            container = pickle.load(fh)
        container["payload"] = container["payload"] + b"\x00"  # tamper
        with path.open("wb") as fh:
            pickle.dump(container, fh)

        with pytest.raises(ValueError, match="checksum mismatch"):
            StateSnapshot.load(path)


# ── SnapshotManager ────────────────────────────────────────────────────

class TestSnapshotManager:
    def test_take_and_latest(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path, interval_seconds=60)
        pipeline = _fake_pipeline()
        snap = asyncio.run(mgr.take_snapshot(pipeline))
        assert isinstance(snap, StateSnapshot)
        latest = mgr.get_latest_snapshot()
        assert latest is not None
        assert latest.last_successful_cycle == pipeline.cycle_count

    def test_cleanup_old_snapshots(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path, retention_days=1)
        pipeline = _fake_pipeline()
        snap = asyncio.run(mgr.take_snapshot(pipeline))
        path = mgr.list_snapshots()[0]
        # Age it out
        import os, time
        old = time.time() - 48 * 3600
        os.utime(path, (old, old))
        removed = mgr.cleanup_old_snapshots()
        assert removed == 1
        assert mgr.list_snapshots() == []

    def test_verify_chain_clean(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path)
        asyncio.run(mgr.take_snapshot(_fake_pipeline()))
        asyncio.run(mgr.take_snapshot(_fake_pipeline()))
        result = mgr.verify_snapshot_chain()
        assert result["ok"] is True
        assert result["total"] == 2


# ── RecoveryManager ────────────────────────────────────────────────────

class TestRecoveryManager:
    def test_detect_clean_halt(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path)
        asyncio.run(mgr.take_snapshot(_fake_pipeline()))
        halt = tmp_path / "halt.lock"
        halt.write_text("x")
        rec = RecoveryManager(
            snapshot_manager=mgr, halt_file=halt, crash_file=tmp_path / "crash.lock",
        )
        prior = asyncio.run(rec.detect_previous_run())
        assert prior is not None
        assert prior["mode"] == "clean"
        assert prior["snapshot"] is not None

    def test_detect_crash(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path)
        asyncio.run(mgr.take_snapshot(_fake_pipeline()))
        crash = tmp_path / "crash.lock"
        crash.write_text("x")
        rec = RecoveryManager(
            snapshot_manager=mgr, halt_file=tmp_path / "halt.lock",
            crash_file=crash,
        )
        prior = asyncio.run(rec.detect_previous_run())
        assert prior is not None
        assert prior["mode"] == "crash"

    def test_detect_no_prior(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path)
        rec = RecoveryManager(
            snapshot_manager=mgr,
            halt_file=tmp_path / "halt.lock",
            crash_file=tmp_path / "crash.lock",
        )
        assert asyncio.run(rec.detect_previous_run()) is None

    def test_recover_reconciles_positions_and_cancels_orphans(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path)
        rec = RecoveryManager(
            snapshot_manager=mgr,
            halt_file=tmp_path / "halt.lock",
            crash_file=tmp_path / "crash.lock",
        )
        snap = StateSnapshot(
            timestamp=_ts(),
            portfolio=None,
            open_orders=[_order()],
            active_positions={
                "AAPL": _pos("AAPL", qty=100),     # matches broker
                "MSFT": _pos("MSFT", qty=50),      # missing at broker
            },
        )
        broker = MagicMock()
        broker.get_positions = AsyncMock(return_value={
            "AAPL": _pos("AAPL", qty=100),
            "TSLA": _pos("TSLA", qty=10),         # extra, not in snapshot
        })
        broker.cancel_order = AsyncMock()

        summary = asyncio.run(rec.recover(snap, broker))
        assert "MSFT" in summary["positions_missing"]
        assert "TSLA" in summary["positions_extra"]
        assert summary["positions_mismatched"] == []
        assert summary["orders_cancelled"] == ["o1"]
        broker.cancel_order.assert_awaited_once_with("o1")

    def test_recover_detects_mismatched_quantity(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path)
        rec = RecoveryManager(
            snapshot_manager=mgr,
            halt_file=tmp_path / "halt.lock",
            crash_file=tmp_path / "crash.lock",
        )
        snap = StateSnapshot(
            timestamp=_ts(), portfolio=None, open_orders=[],
            active_positions={"AAPL": _pos(qty=100)},
        )
        broker = MagicMock()
        broker.get_positions = AsyncMock(return_value={"AAPL": _pos(qty=150)})
        broker.cancel_order = AsyncMock()

        summary = asyncio.run(rec.recover(snap, broker))
        assert summary["positions_mismatched"] == ["AAPL"]

    def test_install_crash_handler_writes_crash_file(self, tmp_path):
        mgr = SnapshotManager(directory=tmp_path)
        crash = tmp_path / "crash.lock"
        rec = RecoveryManager(
            snapshot_manager=mgr,
            halt_file=tmp_path / "halt.lock",
            crash_file=crash,
        )
        pipeline = _fake_pipeline()
        rec.install_crash_handler(pipeline)
        # Invoke handler directly (avoid actually raising the signal).
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert crash.exists()
        assert "crashed_at" in crash.read_text()


# ── Helpers ────────────────────────────────────────────────────────────

def _fake_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pf = PortfolioState(cash=100_000.0, positions={"AAPL": _pos()})
    pipeline.order_manager.portfolio = pf
    pipeline.order_manager.broker = MagicMock()
    pipeline.order_manager.broker.cancel_order = AsyncMock()
    pipeline.cycle_count = 7
    pipeline.model_version = "test"
    phase = MagicMock()
    phase.to_dict.return_value = {"phase_id": 1}
    pipeline.deployment_controller.get_current_phase.return_value = phase
    pipeline.metrics.snapshot.return_value = {"nav": 100_000.0}
    return pipeline
