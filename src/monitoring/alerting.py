"""Tiered alerting (Phase 5 / P5.09).

Design doc §12.2 alert rules: tier by severity and route to a Log fallback
plus optional Telegram. Built-in templates cover the common breach types so
callers (circuit breakers, execution, monitoring) do not format messages
themselves.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


_EMOJI = {
    AlertSeverity.INFO: "ℹ️",
    AlertSeverity.WARNING: "⚠️",
    AlertSeverity.CRITICAL: "🚨",
    AlertSeverity.EMERGENCY: "🆘",
}


@dataclass
class Alert:
    severity: AlertSeverity
    title: str
    message: str
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        return _EMOJI[self.severity]

    def format(self) -> str:
        return (
            f"{self.emoji} *{self.severity.value}* — {self.title}\n"
            f"{self.message}\n"
            f"_{self.timestamp.isoformat()} | source: {self.source or 'unknown'}_"
        )


# ── Channels ───────────────────────────────────────────────────────────

class AlertChannel(ABC):
    @abstractmethod
    async def send(self, alert: Alert) -> bool: ...


class LogChannel(AlertChannel):
    """Always-on fallback: write alerts through the standard logger."""

    _LEVEL = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.CRITICAL: logging.ERROR,
        AlertSeverity.EMERGENCY: logging.CRITICAL,
    }

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or log

    async def send(self, alert: Alert) -> bool:
        self.logger.log(
            self._LEVEL[alert.severity],
            "[%s] %s — %s", alert.source or "alert", alert.title, alert.message,
        )
        return True


class TelegramChannel(AlertChannel):
    """Send alerts to a Telegram chat via bot API.

    The HTTP client is injected so tests can verify calls without hitting the
    real API. In production, pass a callable that wraps `httpx`/`aiohttp`.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        send_fn=None,
        rate_limit_seconds: float = 5.0,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.rate_limit_seconds = rate_limit_seconds
        self._last_send_ts: float = 0.0
        self._send_fn = send_fn  # async callable(url, params) → bool

    async def _default_send(self, url: str, params: dict) -> bool:  # pragma: no cover
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            log.warning("httpx not installed; Telegram send skipped: %s", exc)
            return False
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, data=params)
            return r.status_code == 200

    async def send(self, alert: Alert) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_send_ts
        if elapsed < self.rate_limit_seconds:
            wait = self.rate_limit_seconds - elapsed
            await asyncio.sleep(wait)
        self._last_send_ts = time.monotonic()

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        params = {
            "chat_id": self.chat_id,
            "text": alert.format(),
            "parse_mode": "Markdown",
        }
        sender = self._send_fn or self._default_send
        return bool(await sender(url, params))


# ── Manager ────────────────────────────────────────────────────────────

_DEFAULT_COOLDOWN_S = 300


class AlertManager:
    """Routes alerts by severity and suppresses duplicates within a cooldown."""

    def __init__(
        self,
        channel_map: dict[AlertSeverity, list[AlertChannel]] | None = None,
        *,
        default_cooldown_seconds: int = _DEFAULT_COOLDOWN_S,
        dedupe_path: str | Path | None = None,
    ) -> None:
        if channel_map is None:
            log_ch: list[AlertChannel] = [LogChannel()]
            channel_map = {s: list(log_ch) for s in AlertSeverity}
        self.channel_map = channel_map
        self.default_cooldown_seconds = default_cooldown_seconds
        # Dedup cache maps (source, title) → wall-clock epoch seconds.
        # Wall-clock (vs time.monotonic) is required so the cache survives
        # process restarts.
        self._dedupe_cache: dict[tuple[str, str], float] = {}
        self._dedupe_path = Path(dedupe_path) if dedupe_path else None
        self._load_dedupe()

    # ── Routing / dedupe ──
    def suppress_duplicate(self, alert: Alert, cooldown_seconds: int | None = None) -> bool:
        """Return True if the alert should be suppressed (duplicate within cooldown)."""
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.default_cooldown_seconds
        key = (alert.source, alert.title)
        now = time.time()  # wall-clock: cross-process-safe
        last = self._dedupe_cache.get(key)
        if last is not None and now - last < cooldown:
            return True
        self._dedupe_cache[key] = now
        self._persist_dedupe()
        return False

    def _load_dedupe(self) -> None:
        if self._dedupe_path is None or not self._dedupe_path.exists():
            return
        try:
            raw = json.loads(self._dedupe_path.read_text(encoding="utf-8"))
            self._dedupe_cache = {
                tuple(k.split("||", 1)): float(v) for k, v in raw.items()
            }
        except Exception as exc:
            log.warning("dedupe cache load failed: %s", exc)

    def _persist_dedupe(self) -> None:
        if self._dedupe_path is None:
            return
        try:
            self._dedupe_path.parent.mkdir(parents=True, exist_ok=True)
            blob = {f"{k[0]}||{k[1]}": v for k, v in self._dedupe_cache.items()}
            self._dedupe_path.write_text(json.dumps(blob), encoding="utf-8")
        except Exception as exc:
            log.warning("dedupe cache persist failed: %s", exc)

    async def send_alert(
        self, alert: Alert, *, cooldown_seconds: int | None = None,
    ) -> list[bool]:
        if self.suppress_duplicate(alert, cooldown_seconds):
            return []
        channels = self.channel_map.get(alert.severity, [])
        results = await asyncio.gather(
            *(ch.send(alert) for ch in channels), return_exceptions=False,
        ) if channels else []
        return list(results)

    # ── Templates ──
    def alert_drawdown(self, pct: float, nav: float) -> Alert:
        severity = AlertSeverity.CRITICAL if pct >= 0.10 else AlertSeverity.WARNING
        return Alert(
            severity=severity,
            title="Portfolio Drawdown",
            message=f"Drawdown {pct:.2%} at NAV ${nav:,.0f}",
            source="risk",
            metadata={"drawdown_pct": pct, "nav": nav},
        )

    def alert_daily_loss(self, pct: float) -> Alert:
        return Alert(
            severity=AlertSeverity.CRITICAL,
            title="Daily Loss Limit Breach",
            message=f"Daily P&L {pct:.2%} — auto-flatten and halt",
            source="risk",
            metadata={"daily_pnl_pct": pct},
        )

    def alert_circuit_breaker(self, action) -> Alert:
        sev_map = {
            "WARNING": AlertSeverity.WARNING,
            "CRITICAL": AlertSeverity.CRITICAL,
            "EMERGENCY": AlertSeverity.EMERGENCY,
        }
        severity = sev_map.get(getattr(action, "severity", "WARNING"), AlertSeverity.WARNING)
        return Alert(
            severity=severity,
            title=f"Circuit Breaker: {action.action}",
            message=action.reason,
            source="circuit_breaker",
            metadata={"action": action.action, "severity": action.severity},
        )

    def alert_model_stale(self, hours_since_retrain: float) -> Alert:
        severity = (
            AlertSeverity.CRITICAL if hours_since_retrain > 24 * 60
            else AlertSeverity.WARNING
        )
        return Alert(
            severity=severity,
            title="Model Staleness",
            message=f"Last retrain {hours_since_retrain:.1f}h ago",
            source="ml",
            metadata={"age_hours": hours_since_retrain},
        )

    def alert_data_gap(self, symbol: str, gap_seconds: float) -> Alert:
        return Alert(
            severity=AlertSeverity.WARNING,
            title=f"Data Gap: {symbol}",
            message=f"No ticks for {gap_seconds:.0f}s on {symbol}",
            source="data_engine",
            metadata={"symbol": symbol, "gap_s": gap_seconds},
        )

    def alert_execution_failure(self, order, error: str) -> Alert:
        return Alert(
            severity=AlertSeverity.CRITICAL,
            title="Execution Failure",
            message=(
                f"Order {order.order_id[:8]} on {order.symbol} failed: {error}"
            ),
            source="execution",
            metadata={"order_id": order.order_id, "symbol": order.symbol,
                      "error": error},
        )

    def alert_position_reconciliation(self, discrepancies: list) -> Alert:
        return Alert(
            severity=AlertSeverity.CRITICAL,
            title="Position Reconciliation Drift",
            message=f"{len(discrepancies)} position(s) out of sync with broker",
            source="execution",
            metadata={"discrepancies": discrepancies},
        )

    def alert_feature_drift(self, feature: str, kl: float) -> Alert:
        severity = (
            AlertSeverity.CRITICAL if kl > 1.0
            else AlertSeverity.WARNING if kl > 0.5
            else AlertSeverity.INFO
        )
        return Alert(
            severity=severity,
            title=f"Feature Drift: {feature}",
            message=f"KL divergence {kl:.3f} vs training distribution",
            source="feature_factory",
            metadata={"feature": feature, "kl": kl},
        )
