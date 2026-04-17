"""Tests for the BacktestReport (§9.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.report import BacktestReport
from src.backtesting.transaction_costs import CostEstimate
from src.backtesting.walk_forward import BacktestResult, BacktestTrade


def _cost(total: float = 5.0) -> CostEstimate:
    return CostEstimate(
        commission=1.0,
        spread_cost=2.0,
        slippage=1.0,
        market_impact=total - 4.0,
        total_cost=total,
        cost_bps=5.0,
    )


def _fake_trade(
    ts: pd.Timestamp,
    net: float,
    family: str = "momentum",
) -> BacktestTrade:
    return BacktestTrade(
        entry_timestamp=ts,
        exit_timestamp=ts + pd.Timedelta(days=3),
        symbol="AAA",
        side=1,
        entry_price=100.0,
        exit_price=100.0 + net / 10.0,
        size=0.05,
        signal_family=family,
        gross_pnl=net + 5.0,
        costs=_cost(5.0),
        net_pnl=net,
        holding_period_bars=3,
        meta_label_prob=0.7,
        return_pct=net / 1_000.0,
    )


def _fake_result(
    n_days: int = 250,
    drift: float = 0.0008,
    seed: int = 0,
) -> BacktestResult:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    ret = rng.normal(drift, 0.01, n_days)
    equity = pd.Series(100_000 * np.exp(np.cumsum(ret)), index=idx)
    dd = equity / equity.cummax() - 1.0
    trades = [
        _fake_trade(
            idx[i],
            net=rng.normal(10, 20),
            family=("momentum" if j % 2 == 0 else "mean_reversion"),
        )
        for j, i in enumerate(range(5, n_days, 20))
    ]
    metrics = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annualized_return": 0.15,
        "annualized_vol": 0.12,
        "sharpe": 1.2,
        "sortino": 1.6,
        "calmar": 2.0,
        "max_drawdown": float(dd.min()),
        "max_drawdown_duration_bars": 30,
        "recovery_factor": 2.5,
        "win_rate": 0.56,
        "profit_factor": 1.7,
        "avg_trade": 8.0,
        "avg_holding_period_bars": 3.0,
        "total_trades": len(trades),
        "trades_per_month": 4.0,
        "turnover": 2.5,
        "cost_drag_bps": 20.0,
        "skewness": 0.1,
        "kurtosis": 1.0,
        "tail_ratio": 1.1,
    }
    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        returns=equity.pct_change().fillna(0),
        drawdown_curve=dd,
        metrics=metrics,
    )


class TestTextReport:
    def test_report_has_expected_sections(self):
        report = BacktestReport(result=_fake_result())
        text = report.generate_text_report()

        for section in (
            "BACKTEST REPORT",
            "SUMMARY",
            "PROMOTION GATES",
            "TRADE STATISTICS",
            "COST ANALYSIS",
            "MONTHLY RETURNS",
            "STRATEGY-FAMILY BREAKDOWN",
            "TOP DRAWDOWNS",
        ):
            assert section in text

    def test_report_is_nonempty(self):
        text = BacktestReport(result=_fake_result()).generate_text_report()
        assert len(text) > 500

    def test_gate_incomplete_when_nothing_supplied(self):
        text = BacktestReport(result=_fake_result()).generate_text_report()
        assert "INCOMPLETE" in text


class TestDataframes:
    def test_dataframes_keys(self):
        dfs = BacktestReport(result=_fake_result()).generate_dataframes()
        assert set(dfs.keys()) == {
            "monthly_returns",
            "drawdown_table",
            "trade_log",
            "strategy_breakdown",
            "cpcv_path_stats",
            "regime_breakdown",
        }

    def test_trade_log_row_count_matches_trades(self):
        r = _fake_result()
        dfs = BacktestReport(result=r).generate_dataframes()
        assert len(dfs["trade_log"]) == len(r.trades)

    def test_strategy_breakdown_aggregates_by_family(self):
        dfs = BacktestReport(result=_fake_result()).generate_dataframes()
        sb = dfs["strategy_breakdown"]
        assert {"momentum", "mean_reversion"}.issubset(sb.index)
        assert "n_trades" in sb.columns

    def test_monthly_returns_nonempty(self):
        dfs = BacktestReport(result=_fake_result()).generate_dataframes()
        assert not dfs["monthly_returns"].empty

    def test_regime_breakdown_requires_vol_series(self):
        r = _fake_result()
        no_vol = BacktestReport(result=r).generate_dataframes()
        assert no_vol["regime_breakdown"].empty

        vol = pd.Series(
            np.abs(np.random.default_rng(0).normal(0.02, 0.01, len(r.returns))),
            index=r.returns.index,
        )
        with_vol = BacktestReport(result=r, vol_series=vol).generate_dataframes()
        assert not with_vol["regime_breakdown"].empty
        assert set(with_vol["regime_breakdown"].index) == {"high_vol", "low_vol"}


class TestVerdict:
    def _result_with_gates(
        self, cpcv_passing: bool, dsr_passing: bool, pbo_passing: bool
    ) -> BacktestReport:
        # cpcv_results as mini fake results
        cpcv = [
            _fake_result(n_days=50, drift=0.001 if cpcv_passing else -0.001, seed=i)
            for i in range(45)
        ]
        for r in cpcv:
            r.metrics["total_return"] = 0.02 if cpcv_passing else -0.02

        dsr = {
            "dsr_statistic": 3.0 if dsr_passing else -0.5,
            "p_value": 0.001 if dsr_passing else 0.7,
            "observed_sharpe": 1.5,
            "expected_max_sharpe": 0.8,
            "n_trials": 5,
            "passed": dsr_passing,
        }
        pbo = {"pbo": 0.20 if pbo_passing else 0.55, "max_pbo": 0.40}

        return BacktestReport(
            result=_fake_result(),
            cpcv_results=cpcv,
            dsr_result=dsr,
            pbo_result=pbo,
        )

    def test_all_three_pass_yields_pass(self):
        rpt = self._result_with_gates(True, True, True)
        verdict, fails = rpt.overall_verdict()
        assert verdict == "PASS"
        assert fails == []
        text = rpt.generate_text_report()
        assert "OVERALL VERDICT: PASS" in text

    def test_cpcv_failure_fails_overall(self):
        rpt = self._result_with_gates(False, True, True)
        verdict, fails = rpt.overall_verdict()
        assert verdict == "FAIL"
        assert "CPCV" in fails

    def test_dsr_failure_fails_overall(self):
        rpt = self._result_with_gates(True, False, True)
        verdict, fails = rpt.overall_verdict()
        assert verdict == "FAIL"
        assert "DSR" in fails

    def test_pbo_failure_fails_overall(self):
        rpt = self._result_with_gates(True, True, False)
        verdict, fails = rpt.overall_verdict()
        assert verdict == "FAIL"
        assert "PBO" in fails


class TestSaveReport:
    def test_save_creates_expected_files(self, tmp_path):
        rpt = BacktestReport(result=_fake_result())
        target = tmp_path / "report_out"
        rpt.save_report(str(target))

        assert (target / "report.txt").exists()
        assert (target / "trade_log.csv").exists()
        assert (target / "monthly_returns.csv").exists()
        assert (target / "strategy_breakdown.csv").exists()
        assert (target / "equity_curve.csv").exists()
        assert (target / "drawdown_curve.csv").exists()

    def test_save_roundtrips_trade_log(self, tmp_path):
        r = _fake_result()
        target = tmp_path / "out"
        BacktestReport(result=r).save_report(str(target))
        df = pd.read_csv(target / "trade_log.csv")
        assert len(df) == len(r.trades)
