"""Tests for alerting (Phase 5 / P5.09)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

import pytest

from src.execution.circuit_breakers import CircuitBreakerAction
from src.execution.models import Order, OrderType
from src.monitoring.alerting import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertSeverity,
    LogChannel,
    TelegramChannel,
)


class RecorderChannel(AlertChannel):
    def __init__(self):
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return True


@pytest.fixture
def alert_mgr():
    rec = RecorderChannel()
    tg = RecorderChannel()
    mgr = AlertManager(
        channel_map={
            AlertSeverity.INFO: [rec],
            AlertSeverity.WARNING: [rec, tg],
            AlertSeverity.CRITICAL: [rec, tg],
            AlertSeverity.EMERGENCY: [rec, tg],
        },
        default_cooldown_seconds=300,
    )
    return mgr, rec, tg


class TestRouting:
    @pytest.mark.asyncio
    async def test_critical_goes_to_all_configured_channels(self, alert_mgr):
        mgr, rec, tg = alert_mgr
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            title="test",
            message="hello",
            source="test",
        )
        results = await mgr.send_alert(alert)
        assert results == [True, True]
        assert len(rec.sent) == 1 and len(tg.sent) == 1

    @pytest.mark.asyncio
    async def test_info_only_goes_to_info_channels(self, alert_mgr):
        mgr, rec, tg = alert_mgr
        await mgr.send_alert(Alert(
            severity=AlertSeverity.INFO, title="a", message="b", source="s",
        ))
        assert len(rec.sent) == 1
        assert len(tg.sent) == 0


class TestDuplicateSuppression:
    @pytest.mark.asyncio
    async def test_duplicate_within_cooldown_suppressed(self, alert_mgr):
        mgr, rec, tg = alert_mgr
        a = Alert(AlertSeverity.WARNING, title="dup", message="x", source="s")
        await mgr.send_alert(a)
        b = Alert(AlertSeverity.WARNING, title="dup", message="y", source="s")
        results = await mgr.send_alert(b)
        assert results == []
        assert len(rec.sent) == 1

    @pytest.mark.asyncio
    async def test_same_alert_after_cooldown_sent_again(self, alert_mgr):
        mgr, rec, tg = alert_mgr
        mgr.default_cooldown_seconds = 0  # immediate expiry
        a = Alert(AlertSeverity.WARNING, title="dup", message="x", source="s")
        await mgr.send_alert(a)
        # Nudge cache age
        key = ("s", "dup")
        mgr._dedupe_cache[key] -= 10
        await mgr.send_alert(a, cooldown_seconds=5)
        assert len(rec.sent) == 2

    @pytest.mark.asyncio
    async def test_different_titles_not_deduped(self, alert_mgr):
        mgr, rec, tg = alert_mgr
        await mgr.send_alert(Alert(AlertSeverity.WARNING, "a", "x", source="s"))
        await mgr.send_alert(Alert(AlertSeverity.WARNING, "b", "y", source="s"))
        assert len(rec.sent) == 2


class TestTemplates:
    def test_drawdown_template(self, alert_mgr):
        mgr, *_ = alert_mgr
        a = mgr.alert_drawdown(pct=0.12, nav=98_000)
        assert a.severity == AlertSeverity.CRITICAL
        assert "12" in a.message and "98,000" in a.message
        assert a.metadata["drawdown_pct"] == 0.12

    def test_daily_loss_template(self, alert_mgr):
        mgr, *_ = alert_mgr
        a = mgr.alert_daily_loss(pct=-0.025)
        assert a.severity == AlertSeverity.CRITICAL
        assert "halt" in a.message.lower()

    def test_circuit_breaker_template_maps_severity(self, alert_mgr):
        mgr, *_ = alert_mgr
        cba = CircuitBreakerAction(
            action="HALT_AND_FLATTEN",
            reason="Drawdown 20%",
            severity="EMERGENCY",
        )
        a = mgr.alert_circuit_breaker(cba)
        assert a.severity == AlertSeverity.EMERGENCY
        assert "HALT_AND_FLATTEN" in a.title

    def test_model_stale_template(self, alert_mgr):
        mgr, *_ = alert_mgr
        a1 = mgr.alert_model_stale(200)  # < 24*60=1440 → WARNING
        a2 = mgr.alert_model_stale(2000)  # > 1440 → CRITICAL
        assert a1.severity == AlertSeverity.WARNING
        assert a2.severity == AlertSeverity.CRITICAL

    def test_feature_drift_severity_by_kl(self, alert_mgr):
        mgr, *_ = alert_mgr
        assert mgr.alert_feature_drift("f", 0.2).severity == AlertSeverity.INFO
        assert mgr.alert_feature_drift("f", 0.7).severity == AlertSeverity.WARNING
        assert mgr.alert_feature_drift("f", 1.5).severity == AlertSeverity.CRITICAL

    def test_execution_failure_template(self, alert_mgr):
        mgr, *_ = alert_mgr
        order = Order(
            order_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL", side=1, order_type=OrderType.LIMIT, quantity=100,
        )
        a = mgr.alert_execution_failure(order, "broker rejected")
        assert a.severity == AlertSeverity.CRITICAL
        assert "AAPL" in a.message
        assert a.metadata["error"] == "broker rejected"


class TestDedupePersistence:
    @pytest.mark.asyncio
    async def test_dedupe_survives_process_restart(self, tmp_path):
        path = tmp_path / "dedupe.json"
        rec = RecorderChannel()
        mgr1 = AlertManager(
            channel_map={s: [rec] for s in AlertSeverity},
            default_cooldown_seconds=300, dedupe_path=path,
        )
        alert = Alert(AlertSeverity.WARNING, "X", "y", source="s")
        await mgr1.send_alert(alert)
        assert path.exists()

        # Fresh manager (simulated restart) — should still dedupe
        rec2 = RecorderChannel()
        mgr2 = AlertManager(
            channel_map={s: [rec2] for s in AlertSeverity},
            default_cooldown_seconds=300, dedupe_path=path,
        )
        result = await mgr2.send_alert(alert)
        assert result == []
        assert len(rec2.sent) == 0


class TestLogChannel:
    @pytest.mark.asyncio
    async def test_log_channel_writes(self, caplog):
        ch = LogChannel()
        caplog.set_level(logging.WARNING, logger=ch.logger.name)
        alert = Alert(AlertSeverity.WARNING, title="T", message="M", source="src")
        assert await ch.send(alert) is True
        assert any("T" in r.message and "M" in r.message for r in caplog.records)


class TestTelegramChannel:
    @pytest.mark.asyncio
    async def test_sends_via_injected_client(self):
        calls: list[tuple[str, dict]] = []

        async def fake_send(url, params):
            calls.append((url, params))
            return True

        tg = TelegramChannel(
            bot_token="TOKEN", chat_id="123",
            send_fn=fake_send, rate_limit_seconds=0.0,
        )
        alert = Alert(AlertSeverity.CRITICAL, title="T", message="M", source="s")
        assert await tg.send(alert) is True
        assert len(calls) == 1
        url, params = calls[0]
        assert "TOKEN" in url
        assert params["chat_id"] == "123"
        assert "T" in params["text"] and "M" in params["text"]
