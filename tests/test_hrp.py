"""Tests for Hierarchical Risk Parity (AFML Ch. 20)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.portfolio.hrp import (
    HRPPortfolioOptimizer,
    compute_hrp_weights,
    correlation_to_distance,
    get_recursive_bisection_weights,
    quasi_diagonalize,
)


def _synthetic_returns(
    n_bars: int,
    cov: np.ndarray,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_assets = cov.shape[0]
    data = rng.multivariate_normal(
        mean=np.zeros(n_assets), cov=cov, size=n_bars
    )
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=idx, columns=cols)


class TestDistance:
    def test_distance_range(self):
        corr = pd.DataFrame(
            [[1.0, 0.5, -1.0], [0.5, 1.0, 0.0], [-1.0, 0.0, 1.0]],
            index=list("abc"),
            columns=list("abc"),
        )
        d = correlation_to_distance(corr)
        assert d.loc["a", "a"] == pytest.approx(0.0)
        assert d.loc["a", "b"] == pytest.approx(np.sqrt(0.25))
        assert d.loc["a", "c"] == pytest.approx(1.0)
        assert (d.values >= 0).all()

    def test_distance_symmetry(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 4))
        df = pd.DataFrame(x).corr()
        d = correlation_to_distance(df)
        assert np.allclose(d.to_numpy(), d.to_numpy().T)


class TestQuasiDiagonalize:
    def test_produces_valid_permutation(self):
        rng = np.random.default_rng(1)
        # simulate a 8x8 correlation matrix by averaging random loadings
        loadings = rng.normal(size=(8, 3))
        cov = loadings @ loadings.T + np.eye(8)
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(squareform(dist, checks=False), method="single")
        order = quasi_diagonalize(link)
        assert sorted(order) == list(range(8))


class TestRecursiveBisection:
    def test_two_assets_weights_inverse_to_variance(self):
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.01]], index=["x", "y"], columns=["x", "y"]
        )
        w = get_recursive_bisection_weights(cov, ["x", "y"])
        # For two uncorrelated assets HRP ≡ inverse-variance: w ∝ (1/var)
        ivp = pd.Series({"x": 1 / 0.04, "y": 1 / 0.01})
        ivp /= ivp.sum()
        assert w["x"] == pytest.approx(ivp["x"])
        assert w["y"] == pytest.approx(ivp["y"])

    def test_identity_covariance_gives_equal_weights(self):
        n = 4
        cov = pd.DataFrame(
            np.eye(n), index=[f"A{i}" for i in range(n)], columns=[f"A{i}" for i in range(n)]
        )
        w = get_recursive_bisection_weights(cov, list(cov.columns))
        assert np.allclose(w.values, 1.0 / n)

    def test_accepts_integer_indices(self):
        cov = pd.DataFrame(
            np.diag([1.0, 2.0, 3.0, 4.0]),
            index=list("abcd"),
            columns=list("abcd"),
        )
        w = get_recursive_bisection_weights(cov, [0, 1, 2, 3])
        assert set(w.index) == set("abcd")
        assert w.sum() == pytest.approx(1.0)


class TestComputeHRP:
    def test_weights_sum_to_one_and_nonnegative(self):
        cov = np.eye(5) * 0.01 + 0.001
        returns = _synthetic_returns(500, cov=cov, seed=0)
        w = compute_hrp_weights(returns)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= 0).all()
        assert set(w.index) == set(returns.columns)

    def test_identity_correlation_gives_equal_weights(self):
        # Truly independent returns with equal variance
        n, T = 4, 5000
        rng = np.random.default_rng(12)
        data = rng.normal(0, 0.01, size=(T, n))
        df = pd.DataFrame(
            data, columns=[f"S{i}" for i in range(n)],
            index=pd.date_range("2024-01-01", periods=T, freq="B"),
        )
        w = compute_hrp_weights(df)
        # Empirical correlation isn't exactly zero so allow a small tolerance
        assert np.allclose(w.values, 0.25, atol=0.05)

    def test_three_clusters_allocated_across_clusters(self):
        """With 3 tight clusters of 3 correlated assets each, total weight
        within each cluster should be roughly 1/3, not concentrated in one."""
        rng = np.random.default_rng(3)
        T = 1000
        # 3 cluster factors
        f = rng.normal(0, 0.01, size=(T, 3))
        idiosync = rng.normal(0, 0.002, size=(T, 9))
        cols = []
        data = np.zeros((T, 9))
        for c in range(3):
            for k in range(3):
                idx = c * 3 + k
                data[:, idx] = f[:, c] + idiosync[:, idx]
                cols.append(f"c{c}_a{k}")
        returns = pd.DataFrame(
            data, columns=cols,
            index=pd.date_range("2024-01-01", periods=T, freq="B"),
        )
        w = compute_hrp_weights(returns)
        # Sum weights by cluster prefix
        cluster_sums = w.groupby([c.split("_")[0] for c in w.index]).sum()
        # Each cluster should hold roughly 1/3 of the book (±10%)
        assert (cluster_sums > 0.20).all()
        assert (cluster_sums < 0.45).all()
        assert cluster_sums.sum() == pytest.approx(1.0, abs=1e-9)

    def test_highly_correlated_assets_share_weight(self):
        """If two assets are near-perfect copies, they together should not
        dwarf a third uncorrelated asset with similar variance."""
        rng = np.random.default_rng(7)
        T = 2000
        factor = rng.normal(0, 0.01, T)
        a = factor + rng.normal(0, 0.0005, T)
        b = factor + rng.normal(0, 0.0005, T)
        c = rng.normal(0, 0.01, T)  # independent
        df = pd.DataFrame(
            {"a": a, "b": b, "c": c},
            index=pd.date_range("2024-01-01", periods=T, freq="B"),
        )
        w = compute_hrp_weights(df)
        # c stands alone → should get a noticeably larger slice than either
        # a or b individually. Their sum should still be comparable to c.
        assert w["c"] > w["a"]
        assert w["c"] > w["b"]

    def test_rejects_single_asset(self):
        df = pd.DataFrame({"A": [0.01, 0.02, 0.015]})
        with pytest.raises(ValueError, match="2 assets"):
            compute_hrp_weights(df)


class TestOptimizer:
    def test_rebalances_at_configured_frequency(self):
        rng = np.random.default_rng(0)
        cols = [f"S{i}" for i in range(4)]
        opt = HRPPortfolioOptimizer(rebalance_frequency=3, lookback=60)

        # feed 10 returns; track how often weights actually change
        prev_w = None
        change_count = 0
        for _ in range(12):
            opt.update(pd.Series(rng.normal(0, 0.01, len(cols)), index=cols))
            w = opt.get_weights()
            if prev_w is not None and not np.allclose(w.values, prev_w.values):
                change_count += 1
            prev_w = w.copy()
        # After the first rebalance every third call should change the cache,
        # so expect ≈ 4 changes (calls 3, 6, 9, 12).
        assert 2 <= change_count <= 5

    def test_get_target_positions(self):
        opt = HRPPortfolioOptimizer()
        weights = pd.Series([0.5, 0.5], index=["X", "Y"])
        prices = pd.Series([100.0, 50.0], index=["X", "Y"])
        targets = opt.get_target_positions(weights, portfolio_nav=10_000, prices=prices)
        assert targets["X"] == pytest.approx(50.0)
        assert targets["Y"] == pytest.approx(100.0)

    def test_bad_construction_rejected(self):
        with pytest.raises(ValueError):
            HRPPortfolioOptimizer(rebalance_frequency=0)
        with pytest.raises(ValueError):
            HRPPortfolioOptimizer(lookback=1)

    def test_positions_reject_bad_prices(self):
        opt = HRPPortfolioOptimizer()
        with pytest.raises(ValueError):
            opt.get_target_positions(
                pd.Series([0.5, 0.5], index=["a", "b"]),
                portfolio_nav=1000,
                prices=pd.Series([100.0, 0.0], index=["a", "b"]),
            )
        with pytest.raises(ValueError):
            opt.get_target_positions(
                pd.Series([1.0], index=["a"]),
                portfolio_nav=-1,
                prices=pd.Series([100.0], index=["a"]),
            )
