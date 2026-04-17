"""Tests for the Deflated Sharpe Ratio (Bailey & López de Prado 2014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.deflated_sharpe import (
    EULER_MASCHERONI,
    compute_dsr_from_backtest,
    compute_dsr_from_cpcv,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from src.backtesting.walk_forward import BacktestResult


def _fake_result(
    sharpe: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 0.0,
    n_bars: int = 252,
) -> BacktestResult:
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    # Build a return series with the requested Sharpe-like properties.
    rng = np.random.default_rng(42)
    r = rng.normal(sharpe / 252, 1 / np.sqrt(252), n_bars)
    equity = pd.Series(100_000 * np.exp(np.cumsum(r)), index=idx)
    return BacktestResult(
        trades=[],
        equity_curve=equity,
        returns=equity.pct_change().fillna(0),
        drawdown_curve=pd.Series(0.0, index=idx),
        metrics={
            "sharpe": sharpe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "max_drawdown": 0.0,
        },
    )


class TestExpectedMaxSharpe:
    def test_n_trials_one_returns_zero(self):
        assert expected_max_sharpe(1, 1.0) == 0.0

    def test_monotone_increasing_in_n_trials(self):
        sigma = 1.0
        values = [expected_max_sharpe(n, sigma) for n in (2, 5, 10, 100, 1000, 10000)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_scales_linearly_with_sharpe_std(self):
        a = expected_max_sharpe(100, 1.0)
        b = expected_max_sharpe(100, 2.0)
        assert b == pytest.approx(2 * a, rel=1e-12)

    def test_matches_closed_form_euler_weighting(self):
        from scipy.stats import norm
        import math

        n = 50
        sigma = 1.0
        expected = sigma * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n)
            + EULER_MASCHERONI * norm.ppf(1 - 1 / (n * math.e))
        )
        assert expected_max_sharpe(n, sigma) == pytest.approx(expected)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            expected_max_sharpe(0, 1.0)
        with pytest.raises(ValueError):
            expected_max_sharpe(10, -0.1)


class TestDeflatedSharpe:
    def test_high_sharpe_single_trial_is_significant(self):
        dsr, p = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            sharpe_std=1.0,
            n_trials=1,
            n_observations=252,
        )
        assert dsr > 0
        assert p < 0.01

    def test_low_sharpe_many_trials_not_significant(self):
        dsr, p = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            sharpe_std=1.0,
            n_trials=1000,
            n_observations=252,
        )
        assert dsr < 0
        assert p > 0.95

    def test_fat_tails_inflate_pvalue(self):
        # Use a modest Sharpe + small sample so the p-value has headroom —
        # a huge Sharpe saturates both p-values at machine zero.
        dsr_normal, p_normal = deflated_sharpe_ratio(
            observed_sharpe=0.15,
            sharpe_std=1.0,
            n_trials=1,
            n_observations=50,
            skewness=0.0,
            kurtosis=3.0,
        )
        dsr_fat, p_fat = deflated_sharpe_ratio(
            observed_sharpe=0.15,
            sharpe_std=1.0,
            n_trials=1,
            n_observations=50,
            skewness=0.0,
            kurtosis=10.0,  # excess kurt = 7
        )
        assert p_fat > p_normal
        assert dsr_fat < dsr_normal

    def test_pvalue_monotone_increasing_with_n_trials(self):
        """DSR statistic drops monotonically as n_trials grows (p → 1)."""
        prev_dsr = float("inf")
        for n in (1, 5, 25, 100, 500, 2000):
            dsr, _ = deflated_sharpe_ratio(
                observed_sharpe=1.0,
                sharpe_std=1.0,
                n_trials=n,
                n_observations=252,
            )
            assert dsr < prev_dsr, f"dsr not monotone: {dsr} vs {prev_dsr} at n={n}"
            prev_dsr = dsr

    def test_dsr_formula_arithmetic(self):
        """Hand-check one case: SR=1, σ_SR=0, n_trials=1, T=252, Gaussian."""
        # E[max]=0, std_corrected = sqrt((1 + 1/2)/252) = sqrt(1.5/252)
        dsr, _ = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            sharpe_std=1.0,  # irrelevant at n=1 since E[max]=0
            n_trials=1,
            n_observations=252,
        )
        import math
        expected = 1.0 / math.sqrt(1.5 / 252)
        assert dsr == pytest.approx(expected, rel=1e-9)

    def test_negative_variance_term_clamped(self):
        """When the Mertens correction would go negative, the fn mustn't crash."""
        dsr, p = deflated_sharpe_ratio(
            observed_sharpe=5.0,
            sharpe_std=1.0,
            n_trials=1,
            n_observations=252,
            skewness=2.0,  # pushes var_term toward/below 0
            kurtosis=3.0,
        )
        assert np.isfinite(dsr)
        assert 0.0 <= p <= 1.0


class TestComputeFromBacktest:
    def test_from_backtest_produces_expected_keys(self):
        br = _fake_result(sharpe=1.5, skewness=0.1, kurtosis=1.0)
        out = compute_dsr_from_backtest(br, n_trials=10)

        required = {
            "observed_sharpe",
            "expected_max_sharpe",
            "dsr_statistic",
            "p_value",
            "n_trials",
            "n_observations",
            "skewness",
            "kurtosis",
            "passed",
        }
        assert required.issubset(out.keys())
        assert out["observed_sharpe"] == 1.5
        # metrics["kurtosis"] is excess; DSR uses Pearson → +3
        assert out["kurtosis"] == pytest.approx(4.0)
        assert isinstance(out["passed"], bool)

    def test_high_sharpe_single_trial_passes_gate(self):
        br = _fake_result(sharpe=2.5, n_bars=500)
        out = compute_dsr_from_backtest(br, n_trials=1)
        assert out["passed"] is True
        assert out["p_value"] < 0.05

    def test_low_sharpe_many_trials_fails_gate(self):
        br = _fake_result(sharpe=0.3)
        out = compute_dsr_from_backtest(br, n_trials=5000)
        assert out["passed"] is False
        assert out["p_value"] > 0.05


class TestComputeFromCPCV:
    def test_cpcv_uses_path_sharpes_for_std(self):
        rng = np.random.default_rng(0)
        sharpes = rng.normal(1.0, 0.3, 20)
        results = [_fake_result(sharpe=float(s)) for s in sharpes]
        out = compute_dsr_from_cpcv(results, n_total_trials=100)
        assert out["observed_sharpe"] == pytest.approx(sharpes.mean(), rel=1e-9)
        assert out["sharpe_std"] == pytest.approx(sharpes.std(ddof=1), rel=1e-9)
        assert out["n_observations"] == 20
        assert len(out["path_sharpes"]) == 20

    def test_cpcv_empty_raises(self):
        with pytest.raises(ValueError):
            compute_dsr_from_cpcv([], n_total_trials=10)

    def test_cpcv_constant_sharpes_falls_back_to_unit_std(self):
        results = [_fake_result(sharpe=1.0) for _ in range(5)]
        out = compute_dsr_from_cpcv(results, n_total_trials=10)
        assert out["sharpe_std"] == 1.0  # degenerate fallback
