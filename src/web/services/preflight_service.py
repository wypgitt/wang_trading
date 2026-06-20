"""Preflight & go-live status service.

Wraps :class:`src.execution.preflight.PreflightChecker` (and, when wired, the
:class:`src.execution.infra_probe.InfrastructureProbe`). The checker is fully
constructible with no broker/model deps — unconfigured checks report
``passed=False`` with a "not configured" message, which is the honest preflight
answer (you cannot go live without a broker), so this route runs the *real*
check vector rather than a stub.

Read-only: this service never performs the operator check-in or clears a halt —
those are role-gated mutations owned by separate routes (not built in v1).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from ..dtos import PreflightCheck, PreflightStatus

log = logging.getLogger(__name__)

CheckerFactory = Callable[[], Any]
ProbeFactory = Callable[[], Any]
NowFn = Callable[[], datetime]


def _default_checker() -> Any:
    from src.execution.preflight import PreflightChecker

    return PreflightChecker()


class PreflightService:
    def __init__(
        self,
        *,
        checker_factory: CheckerFactory | None = None,
        probe_factory: ProbeFactory | None = None,
        now_fn: NowFn | None = None,
    ) -> None:
        self._checker_factory = checker_factory or _default_checker
        self._probe_factory = probe_factory  # opt-in; None => infra not probed
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

    async def status(self) -> PreflightStatus:
        now = self._now()
        try:
            checker = self._checker_factory()
            raw_checks = await _maybe_await(checker.run_all_checks())
        except Exception as exc:  # noqa: BLE001
            log.warning("preflight checker unavailable: %s", exc)
            return PreflightStatus(
                overall="UNKNOWN",
                checks=[
                    PreflightCheck(
                        name="preflight",
                        status="UNKNOWN",
                        evaluated_at=now,
                        reason="preflight checker unavailable",
                    )
                ],
                infra_probes={},
            )

        dto_checks: list[PreflightCheck] = []
        blocker_failed = False
        for c in raw_checks:
            passed = bool(getattr(c, "passed", False))
            severity = getattr(c, "severity", "warning")
            if not passed and severity == "blocker":
                blocker_failed = True
            dto_checks.append(
                PreflightCheck(
                    name=getattr(c, "name", "check"),
                    status="PASS" if passed else "FAIL",
                    evaluated_at=now,
                    reason=getattr(c, "message", None) or None,
                )
            )

        if not dto_checks:
            overall = "UNKNOWN"
        elif blocker_failed:
            overall = "BLOCKED"
        else:
            overall = "READY"

        infra = await self._collect_infra()
        return PreflightStatus(
            overall=overall,
            checks=dto_checks,
            operator_sentinel=None,
            capital_deployment=None,
            infra_probes=infra,
            reconciliation=None,
        )

    async def _collect_infra(self) -> dict[str, str]:
        if self._probe_factory is None:
            return {}
        try:
            probe = self._probe_factory()
            result = await _maybe_await(probe.collect())
            return {k: str(v) for k, v in dict(result).items()}
        except Exception as exc:  # noqa: BLE001
            log.warning("preflight infra probe failed: %s", exc)
            return {}


async def _maybe_await(value: Any | Awaitable[Any]) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
