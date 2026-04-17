"""Tests for capital deployment controller (P6.06)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.capital_deployment import (
    CapitalDeploymentController,
    DeploymentPhase,
    default_phases,
)


def _controller(**kwargs) -> CapitalDeploymentController:
    kwargs.setdefault("portfolio", MagicMock())
    return CapitalDeploymentController(**kwargs)


# ── Plan shape ────────────────────────────────────────────────────────────

class TestDefaultPlan:
    def test_four_phases(self):
        plan = default_phases()
        assert len(plan) == 4
        multipliers = [p.position_size_multiplier for p in plan]
        assert multipliers == [0.25, 0.50, 0.75, 1.0]
        caps = [p.target_capital for p in plan[:3]]
        assert caps == [5_000.0, 15_000.0, 50_000.0]

    def test_starts_at_phase_1(self):
        c = _controller()
        phase = c.get_current_phase()
        assert phase.phase_id == 1
        assert phase.status == "active"
        assert c.get_size_multiplier() == 0.25


# ── Promotion ────────────────────────────────────────────────────────────

class TestPromotion:
    def test_duration_not_met(self):
        c = _controller()
        # entry_date already set to now; check 5 days in → below 14-day minimum
        result = asyncio.run(c.check_promotion(
            now=c.get_current_phase().entry_date + timedelta(days=5)
        ))
        assert result["eligible"] is False
        assert "min_duration_not_met" in result["reasons"]

    def test_eligible_after_duration_and_metrics(self):
        metrics = MagicMock()
        metrics.get_performance_stats = MagicMock(
            return_value={"min_sharpe": 1.5, "max_drawdown": 0.05}
        )
        c = _controller(metrics=metrics)
        entry = c.get_current_phase().entry_date
        result = asyncio.run(c.check_promotion(now=entry + timedelta(days=20)))
        assert result["eligible"] is True

    def test_failing_criterion_blocks(self):
        metrics = MagicMock()
        metrics.get_performance_stats = MagicMock(
            return_value={"min_sharpe": 0.3, "max_drawdown": 0.05}
        )
        c = _controller(metrics=metrics)
        entry = c.get_current_phase().entry_date
        result = asyncio.run(c.check_promotion(now=entry + timedelta(days=20)))
        assert result["eligible"] is False
        assert any(r.startswith("criterion_failed") for r in result["reasons"])

    def test_promote_advances_phase(self):
        c = _controller()
        before = c.get_current_phase().phase_id
        new_phase = c.promote()
        assert new_phase.phase_id == before + 1
        assert new_phase.status == "active"
        assert c.phases[0].status == "completed"
        assert c.get_size_multiplier() == 0.50

    def test_promote_at_final_raises(self):
        c = _controller()
        for _ in range(len(c.phases) - 1):
            c.promote()
        with pytest.raises(RuntimeError, match="final phase"):
            c.promote()


# ── Halt ─────────────────────────────────────────────────────────────────

class TestHalt:
    def test_halt_zeroes_size_and_alerts(self):
        alert = MagicMock()
        c = _controller(alert_manager=alert)
        c.halt("drawdown breach")
        assert c.get_size_multiplier() == 0.0
        assert c._halted is True
        alert.send.assert_called_once()
        _, kwargs = alert.send.call_args
        assert kwargs["severity"] == "critical"

    def test_halt_prevents_promotion(self):
        c = _controller()
        c.halt("risk")
        with pytest.raises(RuntimeError):
            c.promote()

    def test_resume_restores(self):
        c = _controller()
        c.halt("test")
        c.resume()
        assert c.get_size_multiplier() == 0.25
        assert not c._halted


# ── Divergence detection ─────────────────────────────────────────────────

class TestDivergence:
    def test_diverges_when_live_much_worse(self):
        paper = [0.02, 0.01, 0.015, 0.018, 0.012, 0.02, 0.014]   # strong, positive
        live = [-0.02, -0.01, -0.015, -0.018, -0.012, -0.02, -0.014]  # mirrored, negative
        result = CapitalDeploymentController.detect_divergence(
            paper, live, window_days=7, threshold=1.0,
        )
        assert result["diverged"] is True

    def test_close_returns_no_divergence(self):
        import random
        random.seed(0)
        paper = [random.gauss(0.001, 0.005) for _ in range(14)]
        live = paper[:]  # identical history
        result = CapitalDeploymentController.detect_divergence(
            paper, live, window_days=7, threshold=1.0,
        )
        assert result["diverged"] is False
        assert result["gap"] == pytest.approx(0.0)


# ── Cascade integration ─────────────────────────────────────────────────

class TestCascadeMultiplier:
    def test_deployment_multiplier_scales_final_size(self):
        from src.bet_sizing.cascade import BetSizingCascade

        cascade = BetSizingCascade()
        kwargs = dict(
            prob=0.80, side=1, symbol="AAPL", signal_family="trend",
            current_vol=0.01, avg_vol=0.01, portfolio_nav=100_000,
        )
        base = cascade.compute_position_size(**kwargs)
        scaled = cascade.compute_position_size(**kwargs, deployment_multiplier=0.25)
        assert scaled["final_size"] == pytest.approx(base["final_size"] * 0.25)
        assert "deployment_multiplier" in scaled["constraints_applied"]
