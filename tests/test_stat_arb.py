"""Tests for the statistical arbitrage / pairs signal (Chan)."""

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.stat_arb import (
    KalmanFilterHedgeRatio,
    StatArbSignal,
    find_cointegrated_pairs,
    johansen_cointegration,
    scan_for_pairs,
)


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

def _cointegrated(
    n: int,
    true_hedge: float,
    true_intercept: float,
    noise: float = 1.0,
    phi: float = 0.85,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """
    y_t = true_hedge * x_t + intercept + AR(1) stationary residual.

    The AR(1) residual with phi=0.85 gives a spread half-life around
    ln(2)/ln(1/0.85) ≈ 4.3 bars — within the default tradeable band.
    """
    rng = np.random.default_rng(seed)
    x = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, size=n)))
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = phi * resid[t - 1] + rng.normal(0.0, noise)
    y = pd.Series(true_hedge * x.values + true_intercept + resid)
    return y, x


def _independent_walks(n: int, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    a = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, size=n)))
    b = pd.Series(50.0 + np.cumsum(rng.normal(0.0, 1.0, size=n)))
    return a, b


# ---------------------------------------------------------------------------
# find_cointegrated_pairs
# ---------------------------------------------------------------------------

class TestFindCointegratedPairs:
    def test_detects_cointegrated_pair(self):
        y, x = _cointegrated(n=500, true_hedge=2.0, true_intercept=5.0, seed=1)
        # Add a third unrelated random walk.
        rw, _ = _independent_walks(n=500, seed=99)
        df = pd.DataFrame({"Y": y, "X": x, "Z": rw})
        pairs = find_cointegrated_pairs(df, p_value_threshold=0.05)
        found = {(a, b) for a, b, _ in pairs}
        assert ("X", "Y") in found or ("Y", "X") in found
        # Z (random walk) should not be cointegrated with either.
        for a, b, _ in pairs:
            assert "Z" not in (a, b)

    def test_independent_random_walks_rejected(self):
        # Two independent random walks → should not pass the 0.05 filter.
        a, b = _independent_walks(n=500, seed=2)
        df = pd.DataFrame({"A": a, "B": b})
        pairs = find_cointegrated_pairs(df, p_value_threshold=0.05)
        assert pairs == []

    def test_returns_sorted_by_pvalue(self):
        y1, x = _cointegrated(n=400, true_hedge=2.0, true_intercept=0.0, noise=0.5, seed=3)
        y2 = y1 + np.random.default_rng(4).normal(0, 2.0, size=400)
        df = pd.DataFrame({"X": x, "Y1": y1, "Y2": y2})
        pairs = find_cointegrated_pairs(df)
        pvals = [p for _, _, p in pairs]
        assert pvals == sorted(pvals)


# ---------------------------------------------------------------------------
# johansen
# ---------------------------------------------------------------------------

class TestJohansen:
    def test_detects_single_cointegrating_vector(self):
        y, x = _cointegrated(n=500, true_hedge=2.0, true_intercept=0.0, seed=5)
        df = pd.DataFrame({"Y": y, "X": x})
        result = johansen_cointegration(df)
        assert "eigenvalues" in result
        assert "eigenvectors" in result
        assert result["n_cointegrating"] >= 1  # expect exactly 1 in theory
        # Trace stat for r=0 must clear the 5% critical value.
        assert result["trace_stats"][0] > result["critical_values"][0, 1]

    def test_too_small_input_raises(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            johansen_cointegration(df)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_converges_to_true_hedge_ratio(self):
        """After N bars, the filter's hedge ratio should be close to the true value."""
        true_hedge = 2.0
        y, x = _cointegrated(n=800, true_hedge=true_hedge, true_intercept=0.0, noise=0.3, seed=6)
        kf = KalmanFilterHedgeRatio(delta=1e-4, obs_cov=0.5)
        final_hr = None
        for yi, xi in zip(y.values, x.values):
            final_hr = kf.update(float(yi), float(xi))
        assert abs(final_hr - true_hedge) < 0.1

    def test_adapts_to_regime_change(self):
        """If the true hedge ratio changes mid-series, the filter should track it."""
        rng = np.random.default_rng(7)
        n_half = 400
        x_series = 100.0 + np.cumsum(rng.normal(0, 1, size=2 * n_half))
        # First half: hedge = 1.5; second half: hedge = 3.0.
        y1 = 1.5 * x_series[:n_half] + rng.normal(0, 0.2, size=n_half)
        y2 = 3.0 * x_series[n_half:] + rng.normal(0, 0.2, size=n_half)
        y = np.concatenate([y1, y2])
        # Larger delta for faster adaptation.
        kf = KalmanFilterHedgeRatio(delta=1e-3, obs_cov=0.1)
        hrs = [kf.update(float(yi), float(xi)) for yi, xi in zip(y, x_series)]
        # Late-first-half estimate should be near 1.5; end-of-series near 3.0.
        mid_first = np.mean(hrs[n_half - 50 : n_half])
        end = np.mean(hrs[-50:])
        assert abs(mid_first - 1.5) < 0.3
        assert abs(end - 3.0) < 0.4

    def test_get_spread_returns_series(self):
        y, x = _cointegrated(n=200, true_hedge=2.0, true_intercept=1.0, noise=0.2, seed=8)
        kf = KalmanFilterHedgeRatio()
        spread = kf.get_spread(y, x)
        assert len(spread) == 200
        # After the burn-in the spread should be small relative to y.
        assert abs(spread.iloc[-50:].mean()) < abs(y.iloc[-50:].mean())

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            KalmanFilterHedgeRatio(delta=0)
        with pytest.raises(ValueError):
            KalmanFilterHedgeRatio(obs_cov=-1)


# ---------------------------------------------------------------------------
# StatArbSignal
# ---------------------------------------------------------------------------

class TestStatArbSignal:
    def test_generates_entry_when_spread_is_wide(self):
        """Inject a big spread deviation → expect an entry signal."""
        # phi=0.95, noise=1.0 → spread half-life in the tradeable band (>2).
        y, x = _cointegrated(
            n=800, true_hedge=2.0, true_intercept=0.0,
            noise=1.0, phi=0.95, seed=9,
        )
        # Manually push the last y well above the fair value.
        y_shocked = y.copy()
        y_shocked.iloc[-1] = y_shocked.iloc[-1] + 30.0
        gen = StatArbSignal(params={"entry_threshold": 1.5})
        sigs = gen.generate(y_series=y_shocked, x_series=x, y_symbol="Y", x_symbol="X")
        assert len(sigs) > 0
        # The final signal should be an entry event and should short Y
        # (spread positive, so y is over-priced → short y).
        final = sigs[-1]
        assert final.metadata["event"] == "entry"
        assert final.side == -1
        assert final.metadata["pair"] == ("Y", "X")
        assert "hedge_ratio" in final.metadata

    def test_long_signal_when_y_below_fair(self):
        y, x = _cointegrated(
            n=800, true_hedge=2.0, true_intercept=0.0,
            noise=1.0, phi=0.95, seed=10,
        )
        y_shocked = y.copy()
        y_shocked.iloc[-1] = y_shocked.iloc[-1] - 30.0
        gen = StatArbSignal(params={"entry_threshold": 1.5})
        sigs = gen.generate(y_series=y_shocked, x_series=x, y_symbol="Y", x_symbol="X")
        assert len(sigs) > 0
        assert sigs[-1].side == 1

    def test_no_signals_on_independent_walks(self):
        """Non-cointegrated pair → spread's half-life is non-tradeable → no signals."""
        a, b = _independent_walks(n=600, seed=11)
        gen = StatArbSignal()
        sigs = gen.generate(y_series=a, x_series=b)
        assert sigs == []

    def test_requires_both_series(self):
        gen = StatArbSignal()
        with pytest.raises(ValueError):
            gen.generate(y_series=pd.Series([1.0, 2.0]))

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            StatArbSignal(params={"entry_threshold": 0.5, "exit_threshold": 1.0})


# ---------------------------------------------------------------------------
# scan_for_pairs
# ---------------------------------------------------------------------------

class TestScanForPairs:
    def test_returns_only_tradeable_pairs(self):
        # Persistent AR(1) residual → half-life well inside (1, 200).
        y, x = _cointegrated(
            n=600, true_hedge=2.0, true_intercept=0.0,
            noise=1.0, phi=0.95, seed=12,
        )
        rw_a, rw_b = _independent_walks(n=600, seed=13)
        prices = {"Y": y, "X": x, "A": rw_a, "B": rw_b}
        pairs = scan_for_pairs(
            prices,
            max_pairs=5,
            lookback=600,
            min_halflife=1.0,
            max_halflife=200.0,
        )
        # The cointegrated pair should be present; random-walk pair should not.
        assert any({a, b} == {"Y", "X"} for a, b in pairs)
        assert not any({a, b} == {"A", "B"} for a, b in pairs)

    def test_empty_input_returns_empty(self):
        assert scan_for_pairs({}, max_pairs=5) == []

    def test_filters_by_halflife(self):
        """With a very tight half-life band, even the true pair is rejected."""
        y, x = _cointegrated(n=400, true_hedge=2.0, true_intercept=0.0, noise=0.3, seed=14)
        prices = {"Y": y, "X": x}
        # Require half-life < 0.1 bars — impossible → empty result.
        pairs = scan_for_pairs(
            prices, max_pairs=5, lookback=400,
            min_halflife=0.001, max_halflife=0.01,
        )
        assert pairs == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
