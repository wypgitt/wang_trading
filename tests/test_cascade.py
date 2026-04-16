"""Tests for the 5-layer bet-sizing cascade (design-doc §8)."""

import numpy as np
import pandas as pd
import pytest

from src.bet_sizing.cascade import (
    BetSizingCascade,
    CascadeConfig,
    FamilyStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cascade(**override) -> BetSizingCascade:
    cfg = CascadeConfig(**override) if override else CascadeConfig()
    return BetSizingCascade(
        config=cfg,
        family_stats={
            "ts_momentum": FamilyStats(avg_win=0.02, avg_loss=0.01),
            "mean_reversion": FamilyStats(avg_win=0.01, avg_loss=0.012),
        },
    )


# ---------------------------------------------------------------------------
# End-to-end behaviour
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_high_confidence_low_vol_produces_large_position(self):
        cascade = _default_cascade()
        result = cascade.compute_position_size(
            prob=0.9, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,  # no vol scaling
            portfolio_nav=1_000_000.0,
        )
        # Final size is capped at max_single_position = 0.10.
        assert result["final_size"] == 0.10
        # But AFML size and kelly-capped should still be > 0.
        assert result["afml_size"] > 0
        assert result["kelly_capped"] > 0
        assert "max_single_position" in result["constraints_applied"]

    def test_low_confidence_produces_zero(self):
        cascade = _default_cascade()
        result = cascade.compute_position_size(
            prob=0.4, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        assert result["afml_size"] == 0.0
        assert result["final_size"] == 0.0

    def test_neutral_side_produces_zero(self):
        cascade = _default_cascade()
        result = cascade.compute_position_size(
            prob=0.9, side=0, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        assert result["final_size"] == 0.0
        assert result["side"] == 0

    def test_short_side_preserves_sign(self):
        cascade = _default_cascade()
        result = cascade.compute_position_size(
            prob=0.85, side=-1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        assert result["final_size"] < 0
        assert result["side"] == -1


# ---------------------------------------------------------------------------
# Layer 2 — Kelly cap
# ---------------------------------------------------------------------------

class TestKellyCap:
    def test_kelly_caps_high_afml(self):
        """
        Kelly should bind the AFML size when the meta-labeler is confident
        but historical payoffs give a thin Kelly fraction. We use
        options-sized unit payoffs (W=L=0.5) where full Kelly at p=0.55 is
        0.2 → fractional Kelly 0.25×0.2 = 0.05. AFML at p=0.55 is ~0.08,
        so Kelly binds.
        """
        cascade = BetSizingCascade(
            family_stats={
                "sketchy_family": FamilyStats(avg_win=0.5, avg_loss=0.5),
            },
        )
        result = cascade.compute_position_size(
            prob=0.55, side=1, symbol="XYZ",
            signal_family="sketchy_family",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        # Kelly binds strictly below AFML on this setup.
        assert result["kelly_capped"] < result["afml_size"]
        assert result["kelly_capped"] <= 0.05 + 1e-9
        assert "kelly_cap" in result["constraints_applied"]

    def test_kelly_zeros_out_negative_edge_despite_high_prob(self):
        """
        When avg_loss >> avg_win, the Kelly fraction can be negative (no-bet)
        even for a meta-labeler probability > 0.5. Demonstrates that Kelly
        doesn't blindly trust the probability — it also requires the payoff
        profile to be healthy.
        """
        cascade = BetSizingCascade(
            family_stats={
                "lossy": FamilyStats(avg_win=0.5, avg_loss=2.0),
            },
        )
        result = cascade.compute_position_size(
            prob=0.7, side=1, symbol="XYZ", signal_family="lossy",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        assert result["kelly_capped"] == 0.0
        assert result["final_size"] == 0.0
        assert "kelly_cap" in result["constraints_applied"]

    def test_no_stats_skips_kelly(self):
        cascade = BetSizingCascade(family_stats={})
        result = cascade.compute_position_size(
            prob=0.8, side=1, symbol="XYZ",
            signal_family="unknown_family",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        # Kelly skipped → kelly_capped equals afml_size.
        assert np.isclose(result["kelly_capped"], result["afml_size"])
        assert "kelly_cap" not in result["constraints_applied"]


# ---------------------------------------------------------------------------
# Layer 3 — Vol adjustment
# ---------------------------------------------------------------------------

class TestVolAdjustment:
    def test_high_vol_reduces_size(self):
        cascade = _default_cascade()
        calm = cascade.compute_position_size(
            prob=0.85, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.02,  # calm: current < avg → boost, but clipped to max
            portfolio_nav=1_000_000.0,
        )
        stormy = cascade.compute_position_size(
            prob=0.85, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.05, avg_vol=0.02,  # stormy
            portfolio_nav=1_000_000.0,
        )
        assert stormy["vol_adjusted"] < calm["vol_adjusted"]
        assert "vol_scaling" in stormy["constraints_applied"]

    def test_vrp_top_quartile_haircut(self):
        cascade = _default_cascade()
        no_vrp = cascade.compute_position_size(
            prob=0.85, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.02, avg_vol=0.02,
            portfolio_nav=1_000_000.0,
            vrp_quartile=1,  # middle — no haircut
        )
        top_vrp = cascade.compute_position_size(
            prob=0.85, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.02, avg_vol=0.02,
            portfolio_nav=1_000_000.0,
            vrp_quartile=3,  # top → 25% cut
        )
        assert top_vrp["vol_adjusted"] < no_vrp["vol_adjusted"]
        assert "vrp_haircut" in top_vrp["constraints_applied"]


# ---------------------------------------------------------------------------
# Layer 4 — ATR
# ---------------------------------------------------------------------------

class TestATRCap:
    def test_atr_cap_applies_to_trend_signals(self):
        cascade = _default_cascade(
            max_single_position=1.0,  # disable Layer 5 single-cap to isolate Layer 4
        )
        # ATR 5% of price, atr_multiplier=2 → per-unit risk 10% of price.
        # risk_per_trade 1% → max_fraction = 0.01 / 0.10 = 0.10.
        result = cascade.compute_position_size(
            prob=0.99, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
            atr=5.0, price=100.0,
        )
        assert result["atr_capped"] <= 0.10 + 1e-9
        assert "atr_cap" in result["constraints_applied"]

    def test_atr_skipped_for_non_trend_without_futures(self):
        cascade = _default_cascade(max_single_position=1.0)
        result = cascade.compute_position_size(
            prob=0.99, side=1, symbol="AAPL",
            signal_family="mean_reversion",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
            atr=5.0, price=100.0,
            asset_class="equity",
        )
        # Mean-rev equity → ATR not applied.
        assert "atr_cap" not in result["constraints_applied"]

    def test_atr_applies_to_futures_regardless_of_family(self):
        cascade = _default_cascade(max_single_position=1.0)
        result = cascade.compute_position_size(
            prob=0.99, side=1, symbol="ES",
            signal_family="mean_reversion",   # not trend, but asset=futures
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
            atr=5.0, price=100.0,
            asset_class="futures",
        )
        assert "atr_cap" in result["constraints_applied"]


# ---------------------------------------------------------------------------
# Layer 5 — Risk budget
# ---------------------------------------------------------------------------

class TestRiskBudget:
    def test_15pct_raw_caps_to_10pct_single(self):
        # Use a 3x raw-size cap so Layer 1 can propose >10%.
        cascade = _default_cascade(max_raw_size=3.0)
        result = cascade.compute_position_size(
            prob=0.99, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        assert result["final_size"] == 0.10
        assert "max_single_position" in result["constraints_applied"]

    def test_family_cap_with_existing_positions(self):
        cascade = _default_cascade()
        # 25% of NAV already allocated to ts_momentum via existing positions.
        # Family cap = 30% → only 5% headroom left for a new ts_momentum bet.
        positions = {
            "MSFT": {"size": 0.10, "family": "ts_momentum",
                     "asset_class": "equity"},
            "GOOG": {"size": 0.15, "family": "ts_momentum",
                     "asset_class": "equity"},
        }
        result = cascade.compute_position_size(
            prob=0.95, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
            current_positions=positions,
        )
        assert result["final_size"] <= 0.05 + 1e-9
        assert "max_family_allocation" in result["constraints_applied"]

    def test_crypto_cap(self):
        cascade = _default_cascade()
        positions = {
            "BTC": {"size": 0.25, "family": "ts_momentum",
                    "asset_class": "crypto"},
        }
        result = cascade.compute_position_size(
            prob=0.95, side=1, symbol="ETH",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
            current_positions=positions,
            asset_class="crypto",
        )
        # Crypto cap is 30%; existing 25% leaves 5% headroom.
        # Family cap is also in play: ts_momentum already 25% used.
        assert result["final_size"] <= 0.05 + 1e-9


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class TestAuditTrail:
    def test_all_intermediate_sizes_returned(self):
        cascade = _default_cascade()
        result = cascade.compute_position_size(
            prob=0.85, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.01, avg_vol=0.01,
            portfolio_nav=1_000_000.0,
        )
        required = {
            "afml_size", "kelly_capped", "vol_adjusted", "atr_capped",
            "final_size", "side", "constraints_applied",
        }
        assert required.issubset(result.keys())

    def test_constraints_applied_lists_triggered_rules(self):
        # Force multiple constraints: wide-margin raw + family cap triggered.
        cascade = _default_cascade(max_raw_size=3.0)
        positions = {
            "MSFT": {"size": 0.25, "family": "ts_momentum",
                     "asset_class": "equity"},
        }
        result = cascade.compute_position_size(
            prob=0.99, side=1, symbol="AAPL",
            signal_family="ts_momentum",
            current_vol=0.02, avg_vol=0.01,  # storm: vol_scaling
            portfolio_nav=1_000_000.0,
            current_positions=positions,
            vrp_quartile=3,  # vrp_haircut
        )
        applied = result["constraints_applied"]
        assert "vol_scaling" in applied
        assert "vrp_haircut" in applied
        # Family cap triggers since 25% used + anything more > 30%.
        assert "max_family_allocation" in applied


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_side_raises(self):
        cascade = _default_cascade()
        with pytest.raises(ValueError):
            cascade.compute_position_size(
                prob=0.8, side=2, symbol="X", signal_family="ts_momentum",
                current_vol=0.01, avg_vol=0.01,
                portfolio_nav=1e6,
            )

    def test_nonpositive_nav_raises(self):
        cascade = _default_cascade()
        with pytest.raises(ValueError):
            cascade.compute_position_size(
                prob=0.8, side=1, symbol="X", signal_family="ts_momentum",
                current_vol=0.01, avg_vol=0.01,
                portfolio_nav=0.0,
            )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_shape_and_columns(self):
        cascade = _default_cascade()
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        features = pd.DataFrame(
            {"current_vol": np.full(10, 0.02), "avg_vol": np.full(10, 0.02)},
            index=idx,
        )
        signals = pd.DataFrame({
            "timestamp": idx[[2, 4, 6, 8]],
            "symbol": ["AAPL"] * 4,
            "family": ["ts_momentum"] * 4,
            "side": [1, -1, 1, 1],
            "prob": [0.8, 0.7, 0.4, 0.9],
        })
        out = cascade.compute_position_sizes_batch(
            signals, features, portfolio_state={"nav": 1_000_000.0},
        )
        assert len(out) == 4
        expected = {
            "timestamp", "symbol", "family", "prob",
            "afml_size", "kelly_capped", "vol_adjusted", "atr_capped",
            "final_size", "side", "constraints_applied",
        }
        assert expected.issubset(out.columns)

    def test_batch_signs_match_side(self):
        cascade = _default_cascade()
        idx = pd.date_range("2024-01-01", periods=5, freq="1h")
        features = pd.DataFrame(
            {"current_vol": np.full(5, 0.02), "avg_vol": np.full(5, 0.02)},
            index=idx,
        )
        signals = pd.DataFrame({
            "timestamp": [idx[2], idx[3]],
            "symbol": ["AAPL", "AAPL"],
            "family": ["ts_momentum", "ts_momentum"],
            "side": [1, -1],
            "prob": [0.85, 0.85],
        })
        out = cascade.compute_position_sizes_batch(
            signals, features, portfolio_state={"nav": 1_000_000.0},
        )
        assert out.loc[0, "final_size"] > 0
        assert out.loc[1, "final_size"] < 0

    def test_batch_missing_required_column_raises(self):
        cascade = _default_cascade()
        idx = pd.date_range("2024-01-01", periods=5, freq="1h")
        features = pd.DataFrame(
            {"current_vol": np.full(5, 0.02), "avg_vol": np.full(5, 0.02)},
            index=idx,
        )
        bad = pd.DataFrame({"timestamp": [idx[0]], "symbol": ["A"]})
        with pytest.raises(ValueError):
            cascade.compute_position_sizes_batch(
                bad, features, portfolio_state={"nav": 1e6},
            )

    def test_batch_no_feature_history_yields_zero(self):
        cascade = _default_cascade()
        # Features start AFTER the signal timestamp.
        features = pd.DataFrame(
            {"current_vol": [0.02, 0.02], "avg_vol": [0.02, 0.02]},
            index=pd.date_range("2024-01-10", periods=2, freq="1h"),
        )
        signals = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01")],
            "symbol": ["AAPL"], "family": ["ts_momentum"],
            "side": [1], "prob": [0.85],
        })
        out = cascade.compute_position_sizes_batch(
            signals, features, portfolio_state={"nav": 1e6},
        )
        assert out.loc[0, "final_size"] == 0.0
        assert "no_feature_history" in out.loc[0, "constraints_applied"]
