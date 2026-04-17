"""Tests for the PCA factor risk model (§8.4.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.factor_risk import FactorRiskModel, detect_unintended_tilts


def _three_factor_returns(T: int = 500, N: int = 10, seed: int = 0) -> pd.DataFrame:
    """Build N assets driven by 3 latent factors plus idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    # Market-like factor with all-positive loadings + two secondary factors
    factors = rng.normal(0, 0.01, size=(T, 3))
    loadings = np.column_stack(
        [
            np.abs(rng.normal(0.8, 0.1, N)),  # market — all positive
            rng.normal(0, 0.5, N),  # balanced
            rng.normal(0, 0.5, N),  # balanced
        ]
    )
    noise = rng.normal(0, 0.003, size=(T, N))
    data = factors @ loadings.T + noise
    cols = [f"A{i}" for i in range(N)]
    return pd.DataFrame(
        data, columns=cols, index=pd.date_range("2024-01-01", periods=T, freq="B")
    )


class TestFit:
    def test_explained_variance_is_decreasing_and_partial(self):
        returns = _three_factor_returns()
        model = FactorRiskModel(n_factors=5).fit(returns)
        evr = model.explained_variance_ratio
        assert all(a >= b - 1e-12 for a, b in zip(evr.values, evr.values[1:]))
        assert evr.sum() < 1.0

    def test_shapes(self):
        returns = _three_factor_returns(N=10)
        model = FactorRiskModel(n_factors=3).fit(returns)
        assert model.factor_loadings.shape == (10, 3)
        assert model.factor_covariance.shape == (3, 3)
        assert len(model.idiosyncratic_variance) == 10
        assert model.factor_returns.shape[1] == 3

    def test_fit_requires_enough_assets(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
        with pytest.raises(ValueError, match="at least"):
            FactorRiskModel(n_factors=5).fit(df)

    def test_bad_construction_rejected(self):
        with pytest.raises(ValueError):
            FactorRiskModel(n_factors=0)
        with pytest.raises(ValueError):
            FactorRiskModel(lookback=1)


class TestFactorExposures:
    def test_equal_weights_dominated_by_market_factor(self):
        returns = _three_factor_returns(N=10, seed=2)
        model = FactorRiskModel(n_factors=3).fit(returns)
        equal = pd.Series(1.0 / 10, index=returns.columns)
        exposures = model.get_factor_exposures(equal)
        # Factor_1 should dominate since its loadings are all positive
        # (market-like) whereas Factors 2 and 3 have mean ≈ 0
        assert abs(exposures["Factor_1"]) > abs(exposures["Factor_2"])
        assert abs(exposures["Factor_1"]) > abs(exposures["Factor_3"])

    def test_exposures_indexed_by_factor_names(self):
        returns = _three_factor_returns(seed=3)
        model = FactorRiskModel(n_factors=3).fit(returns)
        exposures = model.get_factor_exposures(
            pd.Series(1.0 / 10, index=returns.columns)
        )
        assert list(exposures.index) == ["Factor_1", "Factor_2", "Factor_3"]


class TestRiskDecomposition:
    def test_systematic_plus_idiosyncratic_matches_total(self):
        returns = _three_factor_returns(seed=4)
        model = FactorRiskModel(n_factors=5).fit(returns)
        rng = np.random.default_rng(9)
        raw = rng.normal(size=len(returns.columns))
        w = pd.Series(raw / raw.sum(), index=returns.columns)
        d = model.get_risk_decomposition(w)
        assert d["total_risk"] == pytest.approx(
            d["systematic_risk"] + d["idiosyncratic_risk"], rel=1e-9
        )
        # factor_contributions add up to systematic
        assert d["factor_contributions"].sum() == pytest.approx(
            d["systematic_risk"], rel=1e-9
        )
        assert 0.0 <= d["pct_systematic"] <= 1.0

    def test_decomposition_requires_fit(self):
        model = FactorRiskModel()
        with pytest.raises(RuntimeError):
            model.get_risk_decomposition(pd.Series([1.0], index=["a"]))


class TestNeutralize:
    def test_neutralize_single_factor_zeros_exposure(self):
        returns = _three_factor_returns(seed=5)
        model = FactorRiskModel(n_factors=3).fit(returns)
        w = pd.Series(1.0 / 10, index=returns.columns)
        w_neutral = model.neutralize_factors(w, factors_to_neutralize=[0])
        new_exp = model.get_factor_exposures(w_neutral)
        assert abs(new_exp["Factor_1"]) < 1e-9

    def test_neutralize_multiple_factors(self):
        returns = _three_factor_returns(seed=6)
        model = FactorRiskModel(n_factors=3).fit(returns)
        rng = np.random.default_rng(11)
        w = pd.Series(rng.normal(size=10), index=returns.columns)
        w_neutral = model.neutralize_factors(w, factors_to_neutralize=[0, 1])
        new_exp = model.get_factor_exposures(w_neutral)
        assert abs(new_exp["Factor_1"]) < 1e-8
        assert abs(new_exp["Factor_2"]) < 1e-8

    def test_neutralize_is_closest_to_original(self):
        returns = _three_factor_returns(seed=7)
        model = FactorRiskModel(n_factors=3).fit(returns)
        w = pd.Series(1.0 / 10, index=returns.columns)
        w_neutral = model.neutralize_factors(w, factors_to_neutralize=[0])
        # Projection property: (w - w_neutral) is parallel to loadings_0
        residual = (w - w_neutral).to_numpy()
        loadings0 = model.factor_loadings["Factor_1"].to_numpy()
        # residual should be proportional to loadings0
        if np.linalg.norm(residual) > 1e-12:
            cos = (residual @ loadings0) / (
                np.linalg.norm(residual) * np.linalg.norm(loadings0)
            )
            assert abs(abs(cos) - 1.0) < 1e-6


class TestDetectTilts:
    def test_concentrated_portfolio_flagged(self):
        returns = _three_factor_returns(seed=8)
        model = FactorRiskModel(n_factors=3).fit(returns)
        # All weight in one asset — exposure = that asset's loadings
        concentrated = pd.Series(0.0, index=returns.columns)
        concentrated.iloc[0] = 1.0
        warnings = detect_unintended_tilts(
            concentrated, model, threshold_std=1.5
        )
        assert len(warnings) >= 1
        for w in warnings:
            assert {"factor", "exposure", "threshold", "z_score", "warning"} <= w.keys()

    def test_neutral_portfolio_has_no_tilts(self):
        returns = _three_factor_returns(seed=9)
        model = FactorRiskModel(n_factors=3).fit(returns)
        w = pd.Series(1.0 / 10, index=returns.columns)
        # Neutralise everything so no factor has material exposure
        w_neutral = model.neutralize_factors(w, factors_to_neutralize=[0, 1, 2])
        warnings = detect_unintended_tilts(
            w_neutral, model, threshold_std=1.5
        )
        assert warnings == []
