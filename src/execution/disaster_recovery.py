"""Disaster recovery + state persistence for the live pipeline (P6.13).

Snapshots the full pipeline state every few minutes so that a crash or
unexpected restart can be reconciled with the broker rather than lost.
The snapshot is checksummed; any byte-level tampering fails verification.

Three sentinel files distinguish restart modes:

* ``.live_halt``  — operator-requested shutdown (clean).
* ``.live_crash`` — process died with state still open (unclean).
* neither        — first-ever start.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


CRASH_FILE = Path(".live_crash")
HALT_FILE = Path(".live_halt")
SNAPSHOT_DIR = Path("logs/snapshots")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Snapshot ──────────────────────────────────────────────────────────────

@dataclass
class StateSnapshot:
    timestamp: datetime
    portfolio: Any
    open_orders: list[Any] = field(default_factory=list)
    active_positions: dict[str, Any] = field(default_factory=dict)
    deployment_phase: dict[str, Any] | None = None
    model_version: str | None = None
    last_successful_cycle: int = 0
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    # ── Serialisation ────────────────────────────────────────────────

    def _payload_bytes(self) -> bytes:
        """Pickle everything *except* the checksum itself."""
        payload = {
            "timestamp": self.timestamp,
            "portfolio": self.portfolio,
            "open_orders": self.open_orders,
            "active_positions": self.active_positions,
            "deployment_phase": self.deployment_phase,
            "model_version": self.model_version,
            "last_successful_cycle": self.last_successful_cycle,
            "metrics_snapshot": self.metrics_snapshot,
        }
        return pickle.dumps(payload)

    def compute_checksum(self) -> str:
        return hashlib.sha256(self._payload_bytes()).hexdigest()

    def save(self, path: str | Path) -> Path:
        self.checksum = self.compute_checksum()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump({
                "payload": self._payload_bytes(),
                "checksum": self.checksum,
                "version": 1,
            }, fh)
        return p

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> "StateSnapshot":
        p = Path(path)
        with p.open("rb") as fh:
            container = pickle.load(fh)
        payload_bytes = container["payload"]
        stored_checksum = container["checksum"]
        computed = hashlib.sha256(payload_bytes).hexdigest()
        if verify and computed != stored_checksum:
            raise ValueError(
                f"Snapshot checksum mismatch at {p} "
                f"(expected {stored_checksum[:12]}, got {computed[:12]})"
            )
        payload = pickle.loads(payload_bytes)
        snap = cls(checksum=stored_checksum, **payload)
        return snap


# ── Snapshot manager ──────────────────────────────────────────────────────

class SnapshotManager:
    """Periodic snapshots with retention + chain verification."""

    def __init__(
        self,
        *,
        directory: Path | str = SNAPSHOT_DIR,
        interval_seconds: float = 300.0,
        retention_days: int = 30,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = float(interval_seconds)
        self.retention_days = int(retention_days)

    # ── Capture ──────────────────────────────────────────────────────

    async def take_snapshot(self, pipeline: Any) -> StateSnapshot:
        snap = self._snapshot_from_pipeline(pipeline)
        path = self._new_snapshot_path()
        await asyncio.to_thread(snap.save, path)
        log.info("state snapshot written: %s", path)
        return snap

    def _snapshot_from_pipeline(self, pipeline: Any) -> StateSnapshot:
        pf = getattr(pipeline.order_manager.portfolio, "__dict__", {}) and pipeline.order_manager.portfolio
        open_orders = list(getattr(pf, "open_orders", []) or [])
        positions = dict(getattr(pf, "positions", {}) or {})
        phase = None
        ctrl = getattr(pipeline, "deployment_controller", None)
        if ctrl is not None and hasattr(ctrl, "get_current_phase"):
            try:
                phase = ctrl.get_current_phase().to_dict()
            except Exception:
                phase = None
        metrics_snap = {}
        metrics = getattr(pipeline, "metrics", None)
        if metrics is not None and hasattr(metrics, "snapshot"):
            try:
                metrics_snap = dict(metrics.snapshot() or {})
            except Exception:
                metrics_snap = {}
        return StateSnapshot(
            timestamp=_utcnow(),
            portfolio=pf,
            open_orders=open_orders,
            active_positions=positions,
            deployment_phase=phase,
            model_version=getattr(pipeline, "model_version", None),
            last_successful_cycle=int(getattr(pipeline, "cycle_count", 0)),
            metrics_snapshot=metrics_snap,
        )

    def _new_snapshot_path(self) -> Path:
        ts = _utcnow().strftime("%Y%m%dT%H%M%S_%f")
        return self.directory / f"snapshot_{ts}.pkl"

    # ── Retrieval ────────────────────────────────────────────────────

    def list_snapshots(self) -> list[Path]:
        return sorted(self.directory.glob("snapshot_*.pkl"))

    def get_latest_snapshot(self) -> StateSnapshot | None:
        paths = self.list_snapshots()
        if not paths:
            return None
        return StateSnapshot.load(paths[-1])

    # ── Retention / verification ─────────────────────────────────────

    def cleanup_old_snapshots(self, *, now: datetime | None = None) -> int:
        now = now or _utcnow()
        cutoff = now - timedelta(days=self.retention_days)
        removed = 0
        for p in self.list_snapshots():
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except FileNotFoundError:
                continue
            if mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        return removed

    def verify_snapshot_chain(self) -> dict[str, Any]:
        """Load every snapshot in chronological order, verifying checksums."""
        paths = self.list_snapshots()
        broken: list[str] = []
        for p in paths:
            try:
                StateSnapshot.load(p)
            except Exception as exc:
                broken.append(f"{p.name}: {exc}")
        return {
            "total": len(paths),
            "broken": broken,
            "ok": not broken,
        }


# ── Recovery ──────────────────────────────────────────────────────────────

class RecoveryManager:
    """Detects how the previous run ended and reconciles state with a broker."""

    def __init__(
        self,
        *,
        snapshot_manager: SnapshotManager,
        halt_file: Path | str = HALT_FILE,
        crash_file: Path | str = CRASH_FILE,
        alert_manager: Any | None = None,
    ) -> None:
        self.snapshot_manager = snapshot_manager
        self.halt_file = Path(halt_file)
        self.crash_file = Path(crash_file)
        self.alert_manager = alert_manager

    # ── Detection ────────────────────────────────────────────────────

    async def detect_previous_run(self) -> dict[str, Any] | None:
        clean = self.halt_file.exists()
        crashed = self.crash_file.exists()
        if not clean and not crashed:
            return None
        snapshot = self.snapshot_manager.get_latest_snapshot()
        mode = "clean" if clean and not crashed else "crash"
        return {
            "mode": mode,
            "snapshot": snapshot,
            "in_flight_orders": list(
                getattr(snapshot, "open_orders", []) if snapshot else []
            ),
            "halt_file": str(self.halt_file) if clean else None,
            "crash_file": str(self.crash_file) if crashed else None,
        }

    # ── Reconciliation ───────────────────────────────────────────────

    async def recover(self, snapshot: StateSnapshot, broker: Any) -> dict[str, Any]:
        """Align the saved state with what the broker says is currently open."""
        broker_positions = await broker.get_positions()
        saved_positions = snapshot.active_positions or {}

        missing: list[str] = []   # in snapshot, not at broker
        extra: list[str] = []     # at broker, not in snapshot
        mismatched: list[str] = []  # same symbol but different qty

        all_symbols = set(saved_positions) | set(broker_positions)
        for sym in all_symbols:
            s = saved_positions.get(sym)
            b = broker_positions.get(sym)
            if s and not b:
                missing.append(sym)
            elif b and not s:
                extra.append(sym)
            elif s and b:
                s_qty = getattr(s, "quantity", 0) * getattr(s, "side", 1)
                b_qty = getattr(b, "quantity", 0) * getattr(b, "side", 1)
                if abs(s_qty - b_qty) > 1e-6:
                    mismatched.append(sym)

        cancelled = await self._cancel_orphans(snapshot.open_orders, broker)

        summary = {
            "recovered_at": _utcnow().isoformat(),
            "snapshot_ts": snapshot.timestamp.isoformat(),
            "positions_missing": missing,
            "positions_extra": extra,
            "positions_mismatched": mismatched,
            "orders_cancelled": cancelled,
        }
        await self._send_alert(summary)
        # Clear sentinels — operator has acknowledged by running recovery.
        self.halt_file.unlink(missing_ok=True)
        self.crash_file.unlink(missing_ok=True)
        return summary

    async def _cancel_orphans(self, orders: list[Any], broker: Any) -> list[str]:
        cancelled: list[str] = []
        for order in orders or []:
            oid = getattr(order, "order_id", None)
            if not oid:
                continue
            try:
                await broker.cancel_order(oid)
                cancelled.append(oid)
            except Exception as exc:
                log.warning("orphan cancel failed for %s: %s", oid, exc)
        return cancelled

    async def _send_alert(self, summary: dict[str, Any]) -> None:
        mgr = self.alert_manager
        if mgr is None:
            return
        send = getattr(mgr, "send", None)
        if not callable(send):
            return
        try:
            res = send(
                subject="Recovery complete",
                body=str(summary),
                severity="warning",
            )
            if hasattr(res, "__await__"):
                await res  # type: ignore[misc]
        except Exception:  # pragma: no cover
            log.exception("recovery alert failed")

    # ── Crash handler ────────────────────────────────────────────────

    def install_crash_handler(self, pipeline: Any) -> None:
        """Register SIGTERM/SIGINT handlers that snapshot + mark CRASH."""
        def _handler(signum, _frame):
            log.error("caught signal %s — writing crash snapshot", signum)
            try:
                snap = self.snapshot_manager._snapshot_from_pipeline(pipeline)
                snap.save(self.snapshot_manager._new_snapshot_path())
            except Exception:
                log.exception("crash snapshot failed")
            try:
                self.crash_file.parent.mkdir(parents=True, exist_ok=True)
                self.crash_file.write_text(
                    f"crashed_at={_utcnow().isoformat()}\nsignal={signum}\n"
                )
            except Exception:  # pragma: no cover
                log.exception("crash file write failed")
            try:
                pf = pipeline.order_manager.portfolio
                broker = pipeline.order_manager.broker
                for order in list(getattr(pf, "open_orders", [])):
                    try:
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(
                            broker.cancel_order(order.order_id)
                        )
                    except Exception:
                        pass
            except Exception:  # pragma: no cover
                log.exception("crash cancel failed")

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # pragma: no cover - not in main thread
                log.warning("could not install handler for %s", sig)
