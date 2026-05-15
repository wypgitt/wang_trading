"""Bridge to the existing ``src.ui.trade_ideas`` report.

For Phase 1, this service prefers the tmpfs snapshot written by
``src.execution.trade_idea_publisher`` and falls back to a synchronous
regenerate (with a 30 s in-process LRU) when the snapshot is missing,
stale, or unparsable. The fallback re-uses the live-bootstrap-in-paper-
rehearsal report and adapts it to the v2 DTOs. Fields the existing
module already emits map directly; fields v2 adds (regime,
regime_fit_score, expected cost, top_shap_feature, track_record_*) are
populated by later services or left null when not yet wired.

The cache contract is documented in docs/backend_design.md §24 #5 and
mirrored by ``src.execution.trade_idea_publisher``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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

    ``read()`` returns ``(response, staleness_seconds)`` when the file
    exists, parses, and is younger than ``max_age_seconds`` (measured
    against the snapshot's embedded ``as_of`` field — not the file
    mtime, so clock drift between the publisher and BFF host stays
    visible in metrics). Returns ``None`` otherwise; callers should
    treat ``None`` as "fall back to sync regenerate".
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        max_age_seconds: float = 90.0,
    ) -> None:
        self.path = Path(path) if path else _default_tmpfs_path()
        self.max_age_seconds = float(max_age_seconds)

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

        try:
            as_of = datetime.fromisoformat(str(as_of_raw).replace("Z", "+00:00"))
        except ValueError:
            log.warning("trade idea cache bad as_of=%r path=%s", as_of_raw, self.path)
            return None
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        staleness = max(0.0, (datetime.now(timezone.utc) - as_of).total_seconds())
        if staleness > self.max_age_seconds:
            log.info(
                "trade idea cache stale path=%s staleness=%.1fs max=%.1fs",
                self.path,
                staleness,
                self.max_age_seconds,
            )
            return None

        try:
            response = _response_from_report_dict(report)
        except Exception as exc:  # noqa: BLE001
            log.warning("trade idea cache adapter failed path=%s error=%s", self.path, exc)
            return None
        return response, staleness


# ── Service ───────────────────────────────────────────────────────────


_LRU_MISS = object()


class _TtlLru:
    """Tiny TTL'd LRU keyed by an arbitrary hashable tuple.

    Used to debounce concurrent BFF requests that all miss the tmpfs
    cache — without this, every request would bootstrap the live
    pipeline. Keep it small (max 8 entries) since the keyspace is
    derived from filter arguments.
    """

    def __init__(self, *, ttl_seconds: float, maxsize: int = 8) -> None:
        self._ttl = float(ttl_seconds)
        self._maxsize = int(maxsize)
        self._lock = threading.Lock()
        self._store: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return _LRU_MISS
            stored_at, value = entry
            if (time.monotonic() - stored_at) > self._ttl:
                self._store.pop(key, None)
                return _LRU_MISS
            # Refresh recency for LRU eviction.
            self._store.pop(key, None)
            self._store[key] = entry
            return value

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._store.pop(key, None)
            self._store[key] = (time.monotonic(), value)
            while len(self._store) > self._maxsize:
                # ``dict`` preserves insertion order, so the first key
                # is the LRU candidate.
                oldest = next(iter(self._store))
                self._store.pop(oldest, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class TradeIdeasService:
    def __init__(
        self,
        *,
        config_path: str | None = None,
        tmpfs_path: str | Path | None = None,
        tmpfs_max_age_seconds: float = 90.0,
        regenerate_ttl_seconds: float = 30.0,
    ) -> None:
        self.config_path = config_path
        self._cache = TmpfsTradeIdeasCache(
            path=tmpfs_path,
            max_age_seconds=tmpfs_max_age_seconds,
        )
        self._regenerate_lru = _TtlLru(ttl_seconds=regenerate_ttl_seconds)
        # The route layer can inspect this after each call to surface
        # staleness in the response envelope; ``None`` means the
        # response came from the sync regenerate fallback.
        self._last_staleness_seconds: float | None = None

    def list_ideas(
        self,
        *,
        symbols: list[str] | None = None,
        bar_limit: int = 500,
        min_abs_weight: float = 0.0025,
        allow_confidence_fallback: bool = False,
    ) -> TradeIdeasResponse:
        normalised_symbols: tuple[str, ...] | None
        if symbols:
            normalised_symbols = tuple(
                s for s in (str(x).strip().upper() for x in symbols) if s
            ) or None
        else:
            normalised_symbols = None

        # Step 1: try the tmpfs snapshot. The publisher always writes
        # the *full* symbol universe, so we filter at the service layer
        # rather than re-running the pipeline for a sub-symbol request.
        cached = self._cache.read()
        if cached is not None:
            response, staleness = cached
            filtered = _filter_response_by_symbols(response, normalised_symbols)
            self._last_staleness_seconds = staleness
            return filtered

        # Step 2: sync regenerate, debounced by the LRU.
        self._last_staleness_seconds = None
        cache_key = (
            normalised_symbols or (),
            int(bar_limit),
            float(min_abs_weight),
            bool(allow_confidence_fallback),
        )
        cached_response = self._regenerate_lru.get(cache_key)
        if cached_response is not _LRU_MISS:
            return cached_response

        response = self._sync_regenerate(
            symbols=list(normalised_symbols) if normalised_symbols else None,
            bar_limit=bar_limit,
            min_abs_weight=min_abs_weight,
            allow_confidence_fallback=allow_confidence_fallback,
        )
        self._regenerate_lru.put(cache_key, response)
        return response

    # ── internals ─────────────────────────────────────────────────────

    def _sync_regenerate(
        self,
        *,
        symbols: list[str] | None,
        bar_limit: int,
        min_abs_weight: float,
        allow_confidence_fallback: bool,
    ) -> TradeIdeasResponse:
        # Lazy import keeps the BFF importable in environments where the
        # full bootstrap stack is unavailable (e.g. unit tests, docs builds).
        from src.ui.trade_ideas import generate_trade_idea_report_sync

        log.info(
            "trade idea sync regenerate symbols=%s bar_limit=%d",
            symbols,
            bar_limit,
        )
        report = generate_trade_idea_report_sync(
            config_path=self.config_path,
            symbols=symbols,
            bar_limit=bar_limit,
            min_abs_weight=min_abs_weight,
            allow_confidence_fallback=allow_confidence_fallback,
        )
        return _response_from_report_dict(report.to_dict())


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
