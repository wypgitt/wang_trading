"""Tests for the transaction cost model (Phase 4 — P4.01)."""

from __future__ import annotations

import math

import pytest

from src.backtesting.transaction_costs import (
    CRYPTO_COSTS,
    DEFAULT_MODEL,
    EQUITIES_COSTS,
    FUTURES_COSTS,
    CostEstimate,
    TransactionCostModel,
    estimate_round_trip,
)


@pytest.fixture
def model() -> TransactionCostModel:
    return TransactionCostModel(
        equities_config=EQUITIES_COSTS,
        crypto_config=CRYPTO_COSTS,
        futures_config=FUTURES_COSTS,
    )


class TestCostComposition:
    def test_total_equals_sum_of_components(self, model):
        est = model.estimate(
            order_size=1_000,
            price=200.0,
            adv=1_000_000,
            volatility=0.02,
            asset_class="equities",
        )
        parts = est.commission + est.spread_cost + est.slippage + est.market_impact
        assert est.total_cost == pytest.approx(parts, rel=1e-12)

    def test_cost_bps_is_correct_arithmetic(self, model):
        est = model.estimate(
            order_size=500,
            price=100.0,
            adv=2_000_000,
            volatility=0.015,
            asset_class="equities",
        )
        notional = 500 * 100.0
        expected_bps = est.total_cost / notional * 10_000.0
        assert est.cost_bps == pytest.approx(expected_bps, rel=1e-12)


class TestMarketImpact:
    def test_impact_scales_with_sqrt_of_order_size(self, model):
        """Doubling qty (×4) should multiply dollar impact by ~4·sqrt(4)/2 = 4."""
        small = model.estimate(1_000, 100.0, 1_000_000, 0.02, "equities")
        big = model.estimate(4_000, 100.0, 1_000_000, 0.02, "equities")
        # impact ∝ qty · sqrt(qty) = qty^1.5; ratio = 4^1.5 = 8
        ratio = big.market_impact / small.market_impact
        assert ratio == pytest.approx(8.0, rel=1e-9)

    def test_large_order_has_much_higher_impact_than_small(self, model):
        adv = 10_000_000
        small_qty = int(0.001 * adv)  # 0.1% of ADV
        large_qty = int(0.10 * adv)  # 10% of ADV

        small = model.estimate(small_qty, 50.0, adv, 0.02, "equities")
        large = model.estimate(large_qty, 50.0, adv, 0.02, "equities")

        # per-share impact ratio is sqrt(100) = 10, dollar impact ratio is
        # 10 · (large_qty/small_qty) = 10 · 100 = 1000
        per_share_small = small.market_impact / small_qty
        per_share_large = large.market_impact / large_qty
        assert per_share_large / per_share_small == pytest.approx(10.0, rel=1e-6)
        assert large.market_impact > 100 * small.market_impact

    def test_impact_zero_when_volatility_zero(self, model):
        est = model.estimate(1_000, 100.0, 1_000_000, 0.0, "equities")
        assert est.market_impact == 0.0


class TestSanityValues:
    def test_100_share_aapl_order_is_cheap(self, model):
        """A retail-sized AAPL order (100 shares @ $150) should cost <$10 all-in."""
        est = model.estimate(
            order_size=100,
            price=150.0,
            adv=50_000_000,
            volatility=0.02,
            asset_class="equities",
        )
        # Expected ≈ commission $1 + spread $3 + slippage $1.50 + tiny impact
        assert est.total_cost < 10.0
        assert est.total_cost > 0.0
        assert est.commission == pytest.approx(1.0)  # $0.005·100=$0.50 floored
        assert est.cost_bps < 50.0  # <50 bps on a small trade

    def test_crypto_costs_are_reasonable(self, model):
        """A $50k BTC order should cost a few tens of bps."""
        est = model.estimate(
            order_size=1.0,  # 1 BTC
            price=50_000.0,
            adv=10_000.0,  # 10k BTC daily volume
            volatility=0.03,
            asset_class="crypto",
        )
        assert est.total_cost > 0
        assert 0 < est.cost_bps < 200  # reasonable upper bound

    def test_futures_costs_are_reasonable(self, model):
        """A 5-contract ES order (notional ~$1M) should have per-contract commission."""
        est = model.estimate(
            order_size=5,
            price=200_000.0,  # ES contract value
            adv=2_000_000,
            volatility=0.012,
            asset_class="futures",
        )
        assert est.commission == pytest.approx(5 * 1.25)
        assert est.total_cost > est.commission
        assert est.cost_bps > 0


class TestRoundTrip:
    def test_round_trip_is_approximately_twice_single_leg(self, model):
        single = model.estimate(
            order_size=1_000,
            price=100.0,
            adv=1_000_000,
            volatility=0.02,
            asset_class="equities",
        )
        rt = estimate_round_trip(
            entry_size=1_000,
            entry_price=100.0,
            exit_price=100.0,  # same price => exactly 2x
            adv=1_000_000,
            volatility=0.02,
            asset_class="equities",
            model=model,
        )
        assert rt.total_cost == pytest.approx(2 * single.total_cost, rel=1e-9)
        assert rt.commission == pytest.approx(2 * single.commission, rel=1e-9)
        assert rt.market_impact == pytest.approx(2 * single.market_impact, rel=1e-9)

    def test_round_trip_with_price_move(self, model):
        rt = estimate_round_trip(
            entry_size=500,
            entry_price=100.0,
            exit_price=110.0,
            adv=1_000_000,
            volatility=0.02,
            asset_class="equities",
            model=model,
        )
        entry = model.estimate(500, 100.0, 1_000_000, 0.02, "equities")
        exit_ = model.estimate(500, 110.0, 1_000_000, 0.02, "equities")
        assert rt.total_cost == pytest.approx(entry.total_cost + exit_.total_cost)

    def test_round_trip_uses_default_model_when_none_passed(self):
        rt = estimate_round_trip(
            entry_size=100,
            entry_price=100.0,
            exit_price=100.0,
            adv=1_000_000,
            volatility=0.02,
            asset_class="equities",
        )
        assert isinstance(rt, CostEstimate)
        assert rt.total_cost > 0


class TestValidation:
    def test_rejects_unknown_asset_class(self, model):
        with pytest.raises(ValueError, match="no cost config"):
            model.estimate(100, 100.0, 1e6, 0.02, asset_class="fx")

    def test_rejects_nonpositive_price(self, model):
        with pytest.raises(ValueError):
            model.estimate(100, 0.0, 1e6, 0.02, "equities")

    def test_rejects_nonpositive_adv(self, model):
        with pytest.raises(ValueError):
            model.estimate(100, 100.0, 0.0, 0.02, "equities")

    def test_rejects_negative_volatility(self, model):
        with pytest.raises(ValueError):
            model.estimate(100, 100.0, 1e6, -0.01, "equities")

    def test_config_missing_keys_rejected(self):
        with pytest.raises(ValueError, match="missing required keys"):
            TransactionCostModel(equities_config={"commission_per_share": 0.005})


class TestDefaultModel:
    def test_default_model_preconfigured_for_all_asset_classes(self):
        for ac in ("equities", "crypto", "futures"):
            est = DEFAULT_MODEL.estimate(100, 100.0, 1e6, 0.02, asset_class=ac)
            assert est.total_cost > 0

    def test_signed_order_size_handled_via_abs(self, model):
        long_est = model.estimate(1000, 100.0, 1e6, 0.02, "equities")
        short_est = model.estimate(-1000, 100.0, 1e6, 0.02, "equities")
        assert long_est.total_cost == pytest.approx(short_est.total_cost)
