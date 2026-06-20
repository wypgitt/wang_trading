"""Tests for the StrategyGate orchestrator (§9 promotion gate)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.backtesting.gate_orchestrator import StrategyGate, _cli
from src.backtesting.transaction_costs import EQUITIES_COSTS, TransactionCostModel
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


class TestEvaluateCandidate:
    """The retrain-pipeline gate path: real CPCV / DSR / PBO verdicts on a
    candidate strategy — never the historical ``gate_unavailable`` stub.

    Uses a tz-aware (UTC) bar index to mirror the real ``bars`` hypertable;
    the wide-signal helpers normalise to UTC, so a naive index would silently
    produce an all-zero signal panel.
    """

    @staticmethod
    def _panel(drift: float, vol: float, n: int = 500, seed: int = 0):
        idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
        rng = np.random.default_rng(seed)
        price = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        close = pd.DataFrame({"AAA": price}, index=idx)
        signals = pd.DataFrame(1, index=idx, columns=["AAA"])  # always long
        bets = pd.DataFrame(0.5, index=idx, columns=["AAA"])
        features = pd.DataFrame(
            {"f1": price, "f2": np.r_[0.0, np.diff(price)]}, index=idx
        )
        return close, signals, bets, features

    def test_strong_trend_passes_all_three_gates(self):
        cm = TransactionCostModel(equities_config=EQUITIES_COSTS)
        close, signals, bets, features = self._panel(drift=0.004, vol=0.004)
        res = StrategyGate().evaluate_candidate(
            close=close, signals=signals, bet_sizes=bets, features=features,
            cost_model=cm, cpcv_horizon=25,
        )
        assert res["passed"] is True
        # All three gates produced a *real* boolean verdict.
        assert res["gate_1_cpcv"]["passed"] is True
        assert res["gate_2_dsr"]["passed"] is True
        assert res["gate_3_pbo"]["passed"] is True
        # ...backed by real numbers, not a placeholder.
        assert res["gate_1_cpcv"]["total_paths"] == 45
        assert res["gate_1_cpcv"]["pct"] >= 0.60
        assert 0.0 <= res["gate_2_dsr"]["p_value"] < 0.05
        assert 0.0 <= res["gate_3_pbo"]["pbo_value"] < 0.40
        assert "PROCEED" in res["recommendation"]
        assert "gate_unavailable" not in str(res)

    def test_no_drift_noise_fails_with_real_verdict(self):
        cm = TransactionCostModel(equities_config=EQUITIES_COSTS)
        close, signals, bets, features = self._panel(drift=0.0, vol=0.01, seed=3)
        res = StrategyGate().evaluate_candidate(
            close=close, signals=signals, bet_sizes=bets, features=features,
            cost_model=cm, cpcv_horizon=25,
        )
        assert res["passed"] is False
        # DSR is a genuine *False* (a computed p-value ≥ 0.05), not None/stub.
        assert res["gate_2_dsr"]["passed"] is False
        assert res["gate_2_dsr"]["p_value"] >= 0.05
        assert "ITERATE" in res["recommendation"]
        assert "gate_unavailable" not in str(res)

    def test_too_few_bars_degrades_cpcv_to_not_run_not_a_fake_pass(self):
        """With too few bars to form CPCV paths, gate 1 reports "not run"
        (passed=None) and DSR falls back to the single backtest — never a
        fabricated positive-paths verdict."""
        cm = TransactionCostModel(equities_config=EQUITIES_COSTS)
        close, signals, bets, features = self._panel(drift=0.004, vol=0.004, n=12)
        res = StrategyGate().evaluate_candidate(
            close=close, signals=signals, bet_sizes=bets, features=features,
            cost_model=cm, cpcv_horizon=2, n_cpcv_groups=10,
        )
        assert res["gate_1_cpcv"]["passed"] is None  # not run, not False
        assert res["gate_1_cpcv"]["total_paths"] == 0
        # DSR still computes a real verdict from the single backtest.
        assert res["gate_2_dsr"]["passed"] in (True, False)
        assert "gate_unavailable" not in str(res)


class TestCLI:
    def test_cli_dry_run_with_symbol(self, capsys):
        rc = _cli(["--symbol", "AAPL", "--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "AAPL" in captured.out

    def test_cli_requires_symbol_or_all(self):
        with pytest.raises(SystemExit):
            _cli([])

    def test_cli_loader_failure_is_nonzero(self, capsys):
        rc = _cli(["--symbol", "AAPL"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "Gate orchestrator CLI failed" in captured.err
