"""Time-travel replay service.

Reconstructs system state at a past timestamp ``T`` from the audit chain
(:mod:`src.execution.audit_log`) and persisted bars / features / signals
(:mod:`src.data_engine.storage.database`). All recomputation is
server-side; the client receives an immutable :class:`ReplaySnapshot`.

The initial scaffold returns an empty snapshot. Wire to the audit log
and the data engine in Phase 3.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..dtos import ReplaySnapshot


class ReplayService:
    def __init__(self) -> None:
        pass

    def snapshot_at(self, ts: datetime, symbol: str | None = None) -> ReplaySnapshot:
        # TODO: pull audit-chain events <= ts, reconstruct ideas, verify
        # chain, optionally re-run model from persisted features.
        return ReplaySnapshot(
            ts=ts,
            model_version=None,
            regime=None,
            ideas=[],
            audit_chain_verified_to=None,
            warnings=["replay engine stub — wire src.execution.audit_log + src.data_engine.storage"],
        )
