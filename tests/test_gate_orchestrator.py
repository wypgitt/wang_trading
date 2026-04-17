"""Tests for the StrategyGate orchestrator (§9 promotion gate)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.backtesting.gate_orchestrator import StrategyGate, _cli
from src.backtesting.walk_forward import BacktestResult


def _fake_backtest_result(
    sharpe: float,
    n_bars: int = 500,
    seed: int = 0,
) -> BacktestResult:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    r = rng.normal(sharpe / 252, 1 / np.sqrt(252), n_bars)
    equity = pd.Series(100_000 * np.exp(np.cumsum(r)), index=idx)
    return BacktestResult(
        trades=[],
        equity_curve=equity,
        returns=equity.pct_change().fillna(0),
        drawdown_curve=pd.Series(0.0, index=idx),
        metrics={
            "sharpe": sharpe,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "max_drawdown": 0.0,
            "total_trades": 0,
        },
    )


class TestQuickValidate:
    def test_high_sharpe_single_trial_passes(self):
        gate = StrategyGate()
        result = gate.quick_validate(
            _fake_backtest_result(sharpe=2.5), n_trials=1
        )
        assert result["passed"] is True
        assert result["gate_2_dsr"]["passed"] is True
        assert result["gate_2_dsr"]["p_value"] < 0.05
        assert "PROCEED" in result["recommendation"]

    def test_low_sharpe_many_trials_fails(self):
        gate = StrategyGate()
        result = gate.quick_validate(
            _fake_backtest_result(sharpe=0.2), n_trials=5000
        )
        assert result["passed"] is False
        assert "ITERATE" in result["recommendation"]
        # DSR should appear as the failure reason
        assert "DSR" in result["recommendation"]

    def test_output_dict_keys(self):
        gate = StrategyGate()
        result = gate.quick_validate(
            _fake_backtest_result(sharpe=1.0), n_trials=1
        )
        expected = {
            "passed",
            "gate_1_cpcv",
            "gate_2_dsr",
            "gate_3_pbo",
            "backtest_result",
            "report",
            "recommendation",
        }
        assert expected.issubset(result.keys())
        for gkey in ("gate_1_cpcv", "gate_2_dsr", "gate_3_pbo"):
            assert "passed" in result[gkey]

    def test_recommendation_includes_specific_reasons(self):
        gate = StrategyGate()
        # Force DSR to fail (low sharpe + many trials)
        result = gate.quick_validate(
            _fake_backtest_result(sharpe=0.1), n_trials=1000
        )
        msg = result["recommendation"]
        assert "ITERATE" in msg
        # CPCV and PBO weren't run in quick path
        assert "CPCV not run" in msg
        assert "PBO not run" in msg


class TestQuickVsFull:
    def test_quick_validate_is_faster_than_loading_full(self):
        """quick_validate skips CPCV + PBO, so it completes in well under a
        second on a synthetic 500-bar result."""
        gate = StrategyGate()
        start = time.perf_counter()
        gate.quick_validate(_fake_backtest_result(sharpe=1.0), n_trials=1)
        duration = time.perf_counter() - start
        assert duration < 2.0


class TestRecommendationText:
    def test_proceed_message_cites_all_gates(self):
        gate = StrategyGate()
        g1 = {"passed": True, "positive_paths": 30, "total_paths": 45, "pct": 30/45}
        g2 = {"passed": True, "statistic": 3.0, "p_value": 0.001}
        g3 = {"passed": True, "pbo_value": 0.15}
        text = StrategyGate._recommend(True, g1, g2, g3)
        assert "PROCEED" in text
        assert "30/45" in text
        assert "0.001" in text
        assert "0.150" in text

    def test_iterate_message_lists_only_failing_gates(self):
        g1 = {"passed": True, "positive_paths": 30, "total_paths": 45, "pct": 30/45}
        g2 = {"passed": False, "statistic": -1.0, "p_value": 0.4}
        g3 = {"passed": True, "pbo_value": 0.15}
        text = StrategyGate._recommend(False, g1, g2, g3)
        assert "ITERATE" in text
        assert "DSR" in text
        assert "PBO" not in text.split("ITERATE:")[1]


class TestCLI:
    def test_cli_dry_run_with_symbol(self, capsys):
        rc = _cli(["--symbol", "AAPL", "--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "AAPL" in captured.out

    def test_cli_requires_symbol_or_all(self):
        with pytest.raises(SystemExit):
            _cli([])

    def test_cli_stub_message_printed(self, capsys):
        rc = _cli(["--symbol", "AAPL"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Gate orchestrator CLI" in captured.out
