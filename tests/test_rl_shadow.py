"""Tests for the shadow comparison engine and promotion controller (P6.10)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ml_layer.rl_shadow import (
    RLPromotionController,
    ShadowComparisonEngine,
)


def _ts(day: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)


def _seed(engine: ShadowComparisonEngine, *, months: float = 7.0,
          rl_edge: float = 0.002, n: int = 200, seed: int = 0) -> None:
    """Seed ``n`` noisy-but-correlated decisions spanning ``months`` where
    RL outperforms HRP by ``rl_edge`` per period on average."""
    import random
    rng = random.Random(seed)
    total_days = int(months * 30.4375)
    start = datetime.now(timezone.utc) - timedelta(days=total_days)
    for i in range(n):
        ts = start + timedelta(days=total_days * i / n)
        shock = rng.gauss(0.0, 0.01)
        engine.record_decision(
            timestamp=ts,
            hrp_target={"AAPL": 1.0},
            rl_target={"MSFT": 1.0},
            executed_target={"AAPL": 1.0},
            market_state={"symbol_returns": {
                "AAPL": 0.0005 + shock,
                "MSFT": 0.0005 + shock + rl_edge,
            }},
        )


# ── Recording ───────────────────────────────────────────────────────────

class TestRecord:
    def test_record_stores_entries(self):
        engine = ShadowComparisonEngine()
        engine.record_decision(
            _ts(0), {"AAPL": 0.5}, {"AAPL": 0.7},
            {"AAPL": 0.5}, {"symbol_returns": {"AAPL": 0.01}},
        )
        assert len(engine.records) == 1
        assert engine.records[0].hrp_target == {"AAPL": 0.5}

    def test_record_writes_to_storage_if_available(self):
        storage = MagicMock()
        engine = ShadowComparisonEngine(execution_storage=storage)
        engine.record_decision(
            _ts(0), {"A": 0.5}, {"A": 0.6}, {"A": 0.5}, {},
        )
        storage.save_shadow_decision.assert_called_once()


# ── Comparison metrics ─────────────────────────────────────────────────

class TestComparison:
    def test_returns_expected_keys(self):
        engine = ShadowComparisonEngine()
        _seed(engine, months=0.3, n=20, rl_edge=0.001)
        start = engine.records[0].timestamp
        end = engine.records[-1].timestamp
        out = engine.compute_comparison(start, end)
        for k in ("hrp_metrics", "rl_metrics", "paired_t_stat",
                  "p_value", "rl_is_better", "significance"):
            assert k in out
        assert out["n_observations"] == 20

    def test_rl_wins_when_edge_positive(self):
        engine = ShadowComparisonEngine()
        _seed(engine, months=0.3, n=60, rl_edge=0.002)
        start = engine.records[0].timestamp
        end = engine.records[-1].timestamp
        out = engine.compute_comparison(start, end)
        assert out["rl_is_better"] is True
        assert out["rl_metrics"]["sharpe"] > out["hrp_metrics"]["sharpe"]


# ── Promotion eligibility ──────────────────────────────────────────────

class TestEligibility:
    def test_no_history_not_eligible(self):
        engine = ShadowComparisonEngine()
        out = engine.check_promotion_eligibility()
        assert out["eligible"] is False
        assert "no_shadow_history" in out["reasons"]

    def test_insufficient_months(self):
        engine = ShadowComparisonEngine()
        _seed(engine, months=1.0, n=40)
        out = engine.check_promotion_eligibility(
            gates={"cpcv": True, "dsr": True, "pbo": True},
        )
        assert out["eligible"] is False
        assert any("insufficient_history" in r for r in out["reasons"])

    def test_all_criteria_met_eligible(self):
        engine = ShadowComparisonEngine()
        _seed(engine, months=7.0, n=250, rl_edge=0.003)
        out = engine.check_promotion_eligibility(
            gates={"cpcv": True, "dsr": True, "pbo": True},
        )
        assert out["eligible"] is True, out["reasons"]

    def test_failed_gate_blocks(self):
        engine = ShadowComparisonEngine()
        _seed(engine, months=7.0, n=250, rl_edge=0.003)
        out = engine.check_promotion_eligibility(
            gates={"cpcv": True, "dsr": False, "pbo": True},
        )
        assert out["eligible"] is False
        assert "gate_failed:dsr" in out["reasons"]


# ── Report ─────────────────────────────────────────────────────────────

class TestReport:
    def test_report_contains_sharpe(self):
        engine = ShadowComparisonEngine()
        _seed(engine, months=0.5, n=40, rl_edge=0.002)
        report = engine.generate_shadow_report(lookback_days=30)
        assert "Sharpe" in report
        assert "paired t-stat" in report


# ── Promotion controller ───────────────────────────────────────────────

class TestPromotionController:
    def _setup(self, eligible: bool):
        engine = ShadowComparisonEngine()
        if eligible:
            _seed(engine, months=7.0, n=250, rl_edge=0.003)
        holder = MagicMock()
        holder.optimizer = "hrp_instance"
        alert = MagicMock()
        alert.send = AsyncMock()
        return RLPromotionController(
            shadow_engine=engine, optimizer_holder=holder,
            hrp_optimizer="hrp_instance", rl_optimizer="rl_instance",
            alert_manager=alert,
        ), holder, alert

    def test_promote_fails_when_not_eligible(self):
        controller, holder, _ = self._setup(eligible=False)
        with pytest.raises(RuntimeError, match="RL promotion blocked"):
            asyncio.run(controller.promote_rl_to_production(
                gates={"cpcv": True, "dsr": True, "pbo": True},
            ))
        assert holder.optimizer == "hrp_instance"

    def test_promote_succeeds_when_eligible(self):
        controller, holder, alert = self._setup(eligible=True)
        result = asyncio.run(controller.promote_rl_to_production(
            gates={"cpcv": True, "dsr": True, "pbo": True},
        ))
        assert holder.optimizer == "rl_instance"
        assert result["current"] == "rl"
        alert.send.assert_awaited_once()

    def test_revert_swaps_back(self):
        controller, holder, alert = self._setup(eligible=True)
        asyncio.run(controller.promote_rl_to_production(
            gates={"cpcv": True, "dsr": True, "pbo": True},
        ))
        assert holder.optimizer == "rl_instance"
        asyncio.run(controller.revert_to_hrp("regression detected"))
        assert holder.optimizer == "hrp_instance"
        assert controller.current == "hrp"

    def test_revert_flatten_invoked(self):
        controller, holder, _ = self._setup(eligible=True)
        holder.flatten_all = AsyncMock()
        asyncio.run(controller.revert_to_hrp("catastrophe", flatten=True))
        holder.flatten_all.assert_awaited_once()
