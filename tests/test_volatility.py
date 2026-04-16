"""Tests for GARCH volatility features (Sinclair)."""

import numpy as np
import pandas as pd
import pytest

from src.feature_factory.volatility import (
    compute_volatility_features,
    fit_garch,
    garch_volatility,
    realized_volatility,
    realized_vs_implied_spread,
    vol_of_vol,
    vol_term_structure,
)


def _simulate_garch11(
    omega: float, alpha: float, beta: float, n: int, seed: int
) -> np.ndarray:
    """Simulate a zero-mean GARCH(1,1) return path."""
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / max(1e-9, 1.0 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * rng.normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.normal()
    return eps


def _prices_from_returns(returns: np.ndarray, p0: float = 100.0) -> pd.Series:
    """Reconstruct a price series from simple/log returns treated as log."""
    return pd.Series(p0 * np.exp(np.cumsum(returns)))


# ---------------------------------------------------------------------------
# fit_garch
# ---------------------------------------------------------------------------

class TestFitGarch:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0, 0.01, size=500))
        fit = fit_garch(r)
        assert fit is not None
        assert set(fit.keys()) == {
            "omega", "alpha", "beta", "conditional_volatility",
            "log_likelihood", "aic", "bic",
        }
        assert isinstance(fit["alpha"], list) and isinstance(fit["beta"], list)
        assert len(fit["conditional_volatility"]) == len(r)

    def test_recovers_synthetic_params(self):
        """Fitted (omega, alpha, beta) should approximate the simulation."""
        omega_true, alpha_true, beta_true = 0.02, 0.1, 0.85
        r = _simulate_garch11(omega_true, alpha_true, beta_true, n=3000, seed=42)
        fit = fit_garch(pd.Series(r))
        assert fit is not None
        # Parameter recovery is noisy; allow generous bands.
        assert abs(fit["alpha"][0] - alpha_true) < 0.05
        assert abs(fit["beta"][0] - beta_true) < 0.05
        assert abs(fit["omega"] - omega_true) < 0.05

    def test_constant_returns_returns_none(self):
        r = pd.Series(np.zeros(200))
        assert fit_garch(r) is None

    def test_too_short_returns_none(self):
        r = pd.Series(np.random.default_rng(1).normal(0.0, 0.01, size=10))
        assert fit_garch(r) is None


# ---------------------------------------------------------------------------
# Rolling GARCH volatility
# ---------------------------------------------------------------------------

class TestGarchVolatility:
    def test_clustering_after_large_return(self):
        """A synthetic series with a burst of vol should show elevated GARCH sigma."""
        rng = np.random.default_rng(2)
        quiet = rng.normal(0.0, 0.005, size=400)
        # Two consecutive large shocks followed by decay.
        shock = np.concatenate([[0.06, -0.05], rng.normal(0.0, 0.01, size=198)])
        r = np.concatenate([quiet, shock])
        prices = _prices_from_returns(r)

        vol = garch_volatility(prices, window=200, refit_interval=25)
        pre = vol.iloc[300:399].mean()
        post = vol.iloc[405:430].mean()
        assert post > pre, f"pre={pre:.5f}, post={post:.5f}"

    def test_output_shape_and_warmup(self):
        rng = np.random.default_rng(3)
        prices = _prices_from_returns(rng.normal(0.0, 0.01, size=400))
        out = garch_volatility(prices, window=250, refit_interval=50)
        assert len(out) == len(prices)
        assert out.iloc[:250].isna().all()

    def test_nonneg_output(self):
        rng = np.random.default_rng(4)
        prices = _prices_from_returns(rng.normal(0.0, 0.01, size=400))
        out = garch_volatility(prices, window=250, refit_interval=50).dropna()
        assert (out > 0).all()

    def test_refit_interval_controls_refit_count(self):
        """Smaller refit_interval shouldn't explode the output magnitude."""
        rng = np.random.default_rng(5)
        prices = _prices_from_returns(rng.normal(0.0, 0.01, size=400))
        slow = garch_volatility(prices, window=250, refit_interval=1).dropna()
        fast = garch_volatility(prices, window=250, refit_interval=50).dropna()
        # Both should be in the same ballpark (same data, same fit window).
        assert abs(slow.mean() - fast.mean()) / slow.mean() < 0.5

    def test_invalid_params_raise(self):
        s = pd.Series(np.ones(100))
        with pytest.raises(ValueError):
            garch_volatility(s, window=10)
        with pytest.raises(ValueError):
            garch_volatility(s, window=50, refit_interval=0)


# ---------------------------------------------------------------------------
# Vol term structure
# ---------------------------------------------------------------------------

class TestVolTermStructure:
    def test_ratio_above_one_during_spike(self):
        """A vol spike should push short/long > 1."""
        rng = np.random.default_rng(6)
        quiet = rng.normal(0.0, 0.003, size=200)
        spike = rng.normal(0.0, 0.05, size=10)  # 10-bar volatility burst
        calm_tail = rng.normal(0.0, 0.003, size=50)
        r = np.concatenate([quiet, spike, calm_tail])
        prices = _prices_from_returns(r)

        vts = vol_term_structure(prices, short_window=5, long_window=30)
        # Peak during or just after the spike must exceed 1.
        spike_slice = vts.iloc[205:215].dropna()
        assert (spike_slice > 1.0).any()

    def test_warmup_nan(self):
        s = pd.Series(100.0 + np.arange(100.0))
        out = vol_term_structure(s, short_window=5, long_window=30)
        assert out.iloc[:30].isna().all()

    def test_invalid_windows(self):
        s = pd.Series(np.arange(100.0))
        with pytest.raises(ValueError):
            vol_term_structure(s, short_window=10, long_window=5)
        with pytest.raises(ValueError):
            vol_term_structure(s, short_window=1, long_window=30)


# ---------------------------------------------------------------------------
# Vol of vol
# ---------------------------------------------------------------------------

class TestVolOfVol:
    def test_higher_during_regime_transitions(self):
        """Vol-of-vol should spike when the conditional vol level changes."""
        # Synthesize a conditional vol series: flat low → jump → flat high.
        low = np.full(200, 0.01)
        transition = np.linspace(0.01, 0.05, 50)
        high = np.full(200, 0.05)
        series = pd.Series(np.concatenate([low, transition, high]))

        vv = vol_of_vol(series, window=30)
        stable_before = vv.iloc[180:199].mean()
        during_transition = vv.iloc[200:250].mean()
        assert during_transition > stable_before

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            vol_of_vol(pd.Series([1.0, 2.0]), window=1)


# ---------------------------------------------------------------------------
# RV / IV spread
# ---------------------------------------------------------------------------

class TestRVIVSpread:
    def test_simple_subtraction(self):
        rv = pd.Series([0.1, 0.2, 0.15])
        iv = pd.Series([0.18, 0.19, 0.22])
        out = realized_vs_implied_spread(rv, iv)
        np.testing.assert_allclose(out.values, [0.08, -0.01, 0.07])

    def test_realized_volatility_shape(self):
        s = pd.Series(100.0 + np.arange(200.0))
        rv = realized_volatility(s, window=20)
        assert len(rv) == len(s)
        assert rv.iloc[:20].isna().all()


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestComputeVolatilityFeatures:
    def test_default_columns(self):
        rng = np.random.default_rng(7)
        prices = _prices_from_returns(rng.normal(0.0, 0.01, size=400))
        df = compute_volatility_features(prices, window=250, refit_interval=50)
        assert set(df.columns) == {"garch_vol", "vol_term_structure", "vol_of_vol"}
        assert len(df) == len(prices)

    def test_with_implied_vol(self):
        rng = np.random.default_rng(8)
        prices = _prices_from_returns(rng.normal(0.0, 0.01, size=400))
        iv = pd.Series(rng.uniform(0.15, 0.30, size=400), index=prices.index)
        df = compute_volatility_features(
            prices, implied_vol=iv, window=250, refit_interval=50,
        )
        assert "rv_iv_spread" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
