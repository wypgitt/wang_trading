"""Tests for the risk-parity optimiser (§8.4.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.risk_parity import (
    RiskParityOptimizer,
    compute_risk_parity_weights,
    marginal_risk_contribution,
    risk_contribution,
)


def _random_cov(n: int, seed: int = 0, loading_scale: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0, loading_scale, size=(n, max(n // 2, 1)))
    cov = loadings @ loadings.T + np.eye(n) * 0.01
    return pd.DataFrame(
        cov, index=[f"A{i}" for i in range(n)], columns=[f"A{i}" for i in range(n)]
    )


class TestComputeWeights:
    def test_weights_sum_to_one_and_nonnegative(self):
        cov = _random_cov(6, seed=1)
        w = compute_risk_parity_weights(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all()

    def test_equal_risk_contributions(self):
        cov = _random_cov(5, seed=2)
        w = compute_risk_parity_weights(cov, tol=1e-12)
        rc = risk_contribution(w, cov)
        # All RC_i should equal 1/n within a reasonable tolerance
        target = 1.0 / len(rc)
        assert np.allclose(rc.values, target, atol=1e-3)

    def test_diagonal_covariance_gives_inverse_volatility(self):
        """With Σ diagonal, equal-RC weights are ∝ 1/σ (inverse volatility)."""
        diag = np.array([0.01, 0.04, 0.09, 0.16])
        cov = pd.DataFrame(
            np.diag(diag),
            index=list("abcd"),
            columns=list("abcd"),
        )
        w = compute_risk_parity_weights(cov, tol=1e-12)

        inv_vol = 1.0 / np.sqrt(diag)
        expected = inv_vol / inv_vol.sum()
        assert np.allclose(w.values, expected, atol=1e-4)

    def test_identity_covariance_gives_equal_weights(self):
        n = 5
        cov = pd.DataFrame(
            np.eye(n), index=[f"A{i}" for i in range(n)], columns=[f"A{i}" for i in range(n)]
        )
        w = compute_risk_parity_weights(cov, tol=1e-12)
        assert np.allclose(w.values, 1.0 / n, atol=1e-6)

    def test_budget_controls_risk_contributions(self):
        cov = _random_cov(3, seed=3)
        budget = pd.Series([0.5, 0.3, 0.2], index=cov.columns)
        w = compute_risk_parity_weights(cov, budget=budget, tol=1e-12)
        rc = risk_contribution(w, cov)
        # Each RC should track its budget within a reasonable tolerance
        assert np.allclose(rc.values, budget.values, atol=5e-3)

    def test_convergence_matches_direct_scipy(self):
        """A second solve with tighter tolerance lands on the same point."""
        cov = _random_cov(4, seed=4)
        w1 = compute_risk_parity_weights(cov, tol=1e-10, max_iter=5000)
        w2 = compute_risk_parity_weights(cov, tol=1e-12, max_iter=10000)
        assert np.allclose(w1.values, w2.values, atol=1e-4)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_risk_parity_weights(pd.DataFrame([[0.01]], columns=["a"], index=["a"]))
        cov = _random_cov(3, seed=5)
        bad_budget = pd.Series([-1.0, 0.5, 0.5], index=cov.columns)
        with pytest.raises(ValueError, match="non-negative"):
            compute_risk_parity_weights(cov, budget=bad_budget)


class TestContributions:
    def test_risk_contributions_sum_to_one(self):
        cov = _random_cov(5, seed=6)
        w = pd.Series([0.1, 0.2, 0.3, 0.25, 0.15], index=cov.columns)
        rc = risk_contribution(w, cov)
        assert rc.sum() == pytest.approx(1.0, abs=1e-9)

    def test_marginal_risk_contribution_units(self):
        cov = _random_cov(4, seed=7)
        w = pd.Series([0.25] * 4, index=cov.columns)
        mrc = marginal_risk_contribution(w, cov)
        # Sanity: w . MRC = σ_p
        portfolio_vol = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
        assert (w * mrc).sum() == pytest.approx(portfolio_vol, rel=1e-9)


class TestOptimizer:
    def _sample_returns(self, T: int = 500, N: int = 5, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        loadings = rng.normal(0, 0.3, size=(N, 2))
        factors = rng.normal(0, 0.01, size=(T, 2))
        idio = rng.normal(0, 0.003, size=(T, N))
        data = factors @ loadings.T + idio
        return pd.DataFrame(
            data,
            index=pd.date_range("2024-01-01", periods=T, freq="B"),
            columns=[f"A{i}" for i in range(N)],
        )

    def test_get_weights_returns_valid_weights(self):
        opt = RiskParityOptimizer(lookback=250)
        w = opt.get_weights(self._sample_returns())
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all()

    def test_compare_with_hrp_returns_expected_columns(self):
        opt = RiskParityOptimizer(lookback=250)
        cmp_df = opt.compare_with_hrp(self._sample_returns())
        assert set(cmp_df.columns) == {
            "hrp_weight",
            "risk_parity_weight",
            "hrp_rc",
            "risk_parity_rc",
        }
        assert cmp_df["risk_parity_weight"].sum() == pytest.approx(1.0, abs=1e-6)
        assert cmp_df["hrp_weight"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_bad_construction_rejected(self):
        with pytest.raises(ValueError):
            RiskParityOptimizer(lookback=1)
        with pytest.raises(ValueError):
            RiskParityOptimizer(rebalance_frequency=0)

    def test_rejects_single_asset(self):
        opt = RiskParityOptimizer()
        one = pd.DataFrame({"A": np.random.default_rng(0).normal(size=30)})
        with pytest.raises(ValueError):
            opt.get_weights(one)
