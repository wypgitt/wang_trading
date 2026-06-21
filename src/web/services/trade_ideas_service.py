"""Bridge to the existing ``src.ui.trade_ideas`` report.

This service is READ-ONLY: it reads the tmpfs snapshot written by
``src.execution.trade_idea_publisher`` and never regenerates it. A
missing / stale / unparsable snapshot degrades to ``(None, None)`` so
the BFF never runs the engine pipeline on the request path. Fields the
engine already emits map directly; v2-only fields (regime,
regime_fit_score, expected cost, top_shap_feature, track_record_*) are
left null until their producer is wired (the honesty contract).

The cache contract is documented in docs/backend_design.md §24 #5 and
mirrored by ``src.execution.trade_idea_publisher``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..dtos import (
    TradeIdea,
    TradeIdeasResponse,
    TradeIdeasTotals,
)

log = logging.getLogger(__name__)


_ENV_VAR = "WANG_TRADE_IDEAS_PATH"

# Mirrors ``trade_idea_publisher.SNAPSHOT_SCHEMA_VERSION`` (kept local to avoid
# an import cycle between the BFF and the execution package). A mismatch is
# logged but still parsed — forward/backward tolerant across the boundary.
SNAPSHOT_SCHEMA_VERSION = 1


def _default_tmpfs_path() -> Path:
    """Mirror ``trade_idea_publisher.default_output_path`` without import.

    Kept in sync manually to avoid creating an import cycle between the
    BFF and the execution package; the resolution rules come from
    docs/backend_design.md §24a.
    """

    env_override = os.environ.get(_ENV_VAR)
    if env_override:
        return Path(env_override)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "wang" / "trade_ideas.json"
    import sys

    if sys.platform.startswith("linux"):
        return Path("/run/wang") / "trade_ideas.json"
    return Path("/tmp/wang") / "trade_ideas.json"


# ── Tmpfs cache reader ────────────────────────────────────────────────


class TmpfsTradeIdeasCache:
    """Read-only consumer of the publisher's snapshot file.

    ``read()`` returns ``(response, staleness_seconds)`` whenever the file
    exists and parses, and ``None`` only when it is missing / unreadable /
    unparsable. Staleness is measured against the snapshot's embedded
    ``as_of`` (not file mtime, so publisher↔BFF clock drift stays visible).

    Crucially it does NOT reject a stale snapshot: with the regenerate path
    removed, rejecting on age would turn "stale" into a false "unavailable".
    A stale snapshot is shipped with its staleness so the route can flag it
    (the read-only "degrade but ship" contract). ``max_age_seconds`` is kept
    as an advisory reference for callers; the read no longer gates on it.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        max_age_seconds: float = 90.0,
    ) -> None:
        self.path = Path(path) if path else _default_tmpfs_path()
        self.max_age_seconds = float(max_age_seconds)  # advisory; read() never rejects on age

    def read(self) -> tuple[TradeIdeasResponse, float] | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            log.warning("trade idea cache read failed path=%s error=%s", self.path, exc)
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning("trade idea cache unparsable path=%s error=%s", self.path, exc)
            return None

        if not isinstance(payload, dict):
            log.warning("trade idea cache wrong shape path=%s", self.path)
            return None
        as_of_raw = payload.get("as_of")
        report = payload.get("report")
        if not as_of_raw or not isinstance(report, dict):
            log.warning("trade idea cache missing as_of/report path=%s", self.path)
            return None

        version = payload.get("schema_version")
        if version is not None and version != SNAPSHOT_SCHEMA_VERSION:
            # Tolerant: parse anyway, but make the boundary mismatch loud.
            log.warning(
                "trade idea snapshot schema_version=%s expected=%s path=%s "
                "(coordinate the engine->BFF format change)",
                version,
                SNAPSHOT_SCHEMA_VERSION,
                self.path,
            )

        try:
            as_of = datetime.fromisoformat(str(as_of_raw).replace("Z", "+00:00"))
        except ValueError:
            log.warning("trade idea cache bad as_of=%r path=%s", as_of_raw, self.path)
            return None
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        staleness = max(0.0, (datetime.now(timezone.utc) - as_of).total_seconds())
        # "Degrade but ship": a stale snapshot is STILL returned with its
        # measured staleness so the route can flag it (warning) — never
        # rejected (which, with no regenerate path, would falsely 503 a merely
        # stale snapshot). Only a missing/unparsable file yields None above.

        try:
            response = _response_from_report_dict(report)
        except Exception as exc:  # noqa: BLE001
            log.warning("trade idea cache adapter failed path=%s error=%s", self.path, exc)
            return None
        return response, staleness

    def age_seconds(self) -> float | None:
        """Lightweight freshness probe for health checks.

        Reads + parses only the envelope's ``as_of`` (skips the full report
        adaptation), so ``/healthz`` / ``/readyz`` stay cheap. Returns the
        snapshot's age in seconds, or ``None`` when there is no readable
        snapshot at all (missing / unparsable / no ``as_of``).
        """

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        as_of_raw = payload.get("as_of")
        if not as_of_raw:
            return None
        try:
            as_of = datetime.fromisoformat(str(as_of_raw).replace("Z", "+00:00"))
        except ValueError:
            return None
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - as_of).total_seconds())


# ── Service ──────────────────────────────────────────


class TradeIdeasService:
    """Read-only reader of the engine's tmpfs trade-idea snapshot.

    The BFF request path calls :meth:`read_snapshot`, which reads the
    published snapshot and NEVER regenerates it: a missing / stale /
    unparsable snapshot degrades to ``(None, None)`` so the BFF can never
    run the engine pipeline inside a web worker (the read-only invariant,
    docs/aperture_backend_design.md §5). Regenerating the snapshot is the
    engine publisher's job (``src.execution.trade_idea_publisher``), never
    the read layer's.
    """

    def __init__(
        self,
        *,
        tmpfs_path: str | Path | None = None,
        tmpfs_max_age_seconds: float = 90.0,
    ) -> None:
        self._cache = TmpfsTradeIdeasCache(
            path=tmpfs_path,
            max_age_seconds=tmpfs_max_age_seconds,
        )

    def read_snapshot(
        self, *, symbols: list[str] | None = None
    ) -> tuple[TradeIdeasResponse | None, float | None]:
        """Read the published snapshot WITHOUT regenerating (read-only path).

        Returns ``(response, staleness_seconds)`` when a fresh snapshot is
        present, or ``(None, None)`` when there is no readable/fresh snapshot
        (callers degrade). Staleness is measured against the snapshot's
        embedded ``as_of`` (not file mtime). Never raises on a missing file;
        never runs the engine pipeline. The staleness rides back as a return
        value (not a mutable attribute) so concurrent requests on the shared
        singleton cannot clobber each other's reading.
        """

        cached = self._cache.read()
        if cached is None:
            return None, None
        response, staleness = cached
        if symbols:
            normalised = (
                tuple(s for s in (str(x).strip().upper() for x in symbols) if s)
                or None
            )
            response = _filter_response_by_symbols(response, normalised)
        return response, staleness

    def snapshot_age_seconds(self) -> float | None:
        """Snapshot freshness (seconds since ``as_of``) for health checks."""

        return self._cache.age_seconds()


# ── Adapters ──────────────────────────────────────────────────────────


def _response_from_report_dict(ideas_dict: dict[str, Any]) -> TradeIdeasResponse:
    """Adapt the legacy ``TradeIdeaReport.to_dict()`` shape to the v2 DTO.

    Shared between the tmpfs reader (which sees the dict as JSON) and
    the sync regenerate path so the two code paths cannot diverge on
    field handling.
    """

    ideas: list[TradeIdea] = []
    for raw in ideas_dict.get("ideas", []) or []:
        ideas.append(_idea_from_legacy(raw))

    totals_raw = ideas_dict.get("totals", {}) or {}
    totals = TradeIdeasTotals(
        buy=int(totals_raw.get("buy", 0)),
        sell=int(totals_raw.get("sell", 0)),
        watch=int(totals_raw.get("watch", 0)),
        model_required=int(totals_raw.get("model_required", 0)),
        no_data=int(totals_raw.get("no_data", 0)),
        error=int(totals_raw.get("error", 0)),
        gross_target_weight=float(totals_raw.get("gross_target_weight", 0.0)),
        net_target_weight=float(totals_raw.get("net_target_weight", 0.0)),
    )
    return TradeIdeasResponse(
        idea_count=len(ideas),
        totals=totals,
        ideas=ideas,
    )


def _filter_response_by_symbols(
    response: TradeIdeasResponse,
    symbols: tuple[str, ...] | None,
) -> TradeIdeasResponse:
    """Return a copy of ``response`` filtered to ``symbols`` (if any)."""

    if not symbols:
        return response
    wanted = set(symbols)
    filtered_ideas = [idea for idea in response.ideas if idea.symbol.upper() in wanted]
    return TradeIdeasResponse(
        idea_count=len(filtered_ideas),
        totals=response.totals,
        ideas=filtered_ideas,
    )


def _idea_from_legacy(raw: dict[str, Any]) -> TradeIdea:
    """Adapt the legacy ``TradeIdea`` dict shape to the v2 Pydantic DTO."""

    latest_bar_at_raw = raw.get("latest_bar_at")
    latest_bar_at = None
    if latest_bar_at_raw:
        try:
            latest_bar_at = datetime.fromisoformat(str(latest_bar_at_raw).replace("Z", "+00:00"))
        except ValueError:
            latest_bar_at = None

    return TradeIdea(
        symbol=str(raw.get("symbol", "")),
        action=str(raw.get("action", "WATCH")),  # type: ignore[arg-type]
        target_weight=float(raw.get("target_weight", 0.0)),
        target_notional=float(raw.get("target_notional", 0.0)),
        estimated_quantity=_optional_float(raw.get("estimated_quantity")),
        latest_price=_optional_float(raw.get("latest_price")),
        latest_bar_at=latest_bar_at,
        bar_type=None,  # not in legacy idea row; live cycle puts it on the report
        bars_loaded=int(raw.get("bars_loaded", 0) or 0),
        feature_rows=int(raw.get("feature_rows", 0) or 0),
        signal_count=int(raw.get("signal_count", 0) or 0),
        top_signal_family=raw.get("top_signal_family"),
        top_signal_side=raw.get("top_signal_side"),
        top_signal_confidence=_optional_float(raw.get("top_signal_confidence")),
        avg_signal_confidence=_optional_float(raw.get("avg_signal_confidence")),
        meta_probability=_optional_float(raw.get("meta_probability")),
        calibrated_probability=_optional_float(raw.get("calibrated_probability")),
        regime=None,                          # populated by RegimeService in §6 wiring
        regime_fit_score=None,                # populated by SignalsService attribution
        bet_size=_optional_float(raw.get("bet_size")),
        sizing_constraints_applied=[],        # populated when bet sizing service lands
        strategy=raw.get("strategy"),
        reason=str(raw.get("reason", "")),
        expected_cost_bps=None,               # populated by CostForecastService
        top_shap_feature=None,                # populated by ShapService
        track_record_win_rate=None,           # populated by TrackRecordService
        track_record_n=None,
        stage_latency_seconds=dict(raw.get("stage_latency_seconds", {}) or {}),
        errors=list(raw.get("errors", []) or []),
    )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
