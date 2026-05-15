"""Preflight & go-live status service.

Wraps :mod:`src.execution.preflight`, the operator check-in sentinel,
the HALT sentinel, and :mod:`src.execution.infra_probe`. Returns a
:class:`PreflightStatus` that the Preflight & Go-Live page renders as a
blocker checklist with runbook links.

Mutation endpoints (check-in, clear-halt, request-retrain) are owned by
separate routes and require role-gated headers — they do not live in
this read service.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..dtos import PreflightCheck, PreflightStatus


class PreflightService:
    def __init__(self) -> None:
        pass

    def status(self) -> PreflightStatus:
        # TODO: pull the real check vector from src.execution.preflight.
        now = datetime.now(timezone.utc)
        checks = [
            PreflightCheck(name="operator_checkin", status="UNKNOWN", evaluated_at=now,
                           reason="preflight service stub — wire src.execution.preflight",
                           runbook="docs/go_live_checklist.md"),
        ]
        return PreflightStatus(
            overall="UNKNOWN",
            checks=checks,
            operator_sentinel=None,
            capital_deployment=None,
            infra_probes={},
            reconciliation=None,
        )
