"""Tests for the meta-labeling pipeline (AFML §6.2 + design-doc §6)."""

import numpy as np
import pandas as pd
import pytest

from src.labeling.meta_labeler_pipeline import (
    MetaLabelingPipeline,
    configure_barrier_asymmetry,
)
from src.signal_battery.base_signal import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_index(n: int, freq: str = "1min") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=freq)


def _features(close: pd.Series) -> pd.DataFrame:
    """Minimal feature matrix over the same clock as ``close``."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f1": rng.normal(size=len(close)),
            "f2": rng.normal(size=len(close)),
            "ret1": close.pct_change().fillna(0.0).to_numpy(),
        },
        index=close.index,
    )


def _signals(event_positions, close, family="ts_momentum",
             side=1, confidence=0.7) -> pd.DataFrame:
    ts = close.index[event_positions]
    n = len(event_positions)
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": ["TEST"] * n,
        "family": [family] * n,
        "side": [side] * n,
        "confidence": [confidence] * n,
    })


# ---------------------------------------------------------------------------
# configure_barrier_asymmetry
# ---------------------------------------------------------------------------

class TestBarrierAsymmetry:
    def test_mean_reversion_has_tight_upper(self):
        upper, lower = configure_barrier_asymmetry("mean_reversion")
        assert upper < lower
        assert (upper, lower) == (1.0, 1.5)

    def test_momentum_has_tight_lower(self):
        for fam in ("ts_momentum", "cs_momentum", "ma_crossover",
                    "donchian", "breakout_signal"):
            upper, lower = configure_barrier_asymmetry(fam)
            assert upper > lower, f"{fam} should have wide TP / tight SL"
            assert (upper, lower) == (2.5, 1.0)

    def test_stat_arb_is_symmetric(self):
        assert configure_barrier_asymmetry("stat_arb") == (1.5, 1.5)
        assert configure_barrier_asymmetry("cross_exchange_arb") == (1.5, 1.5)

    def test_carry_profile(self):
        # Both futures carry and funding-rate arb use the carry profile.
        assert configure_barrier_asymmetry("futures_carry") == (3.0, 1.0)
        assert configure_barrier_asymmetry("funding_arb") == (3.0, 1.0)

    def test_default_is_symmetric(self):
        assert configure_barrier_asymmetry("some_unknown_family") == (2.0, 2.0)


# ---------------------------------------------------------------------------
# MetaLabelingPipeline — constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_rejects_bad_multipliers(self):
        with pytest.raises(ValueError):
            MetaLabelingPipeline(upper_barrier_mult=0.0)
        with pytest.raises(ValueError):
            MetaLabelingPipeline(lower_barrier_mult=-1.0)

    def test_rejects_bad_horizon(self):
        with pytest.raises(ValueError):
            MetaLabelingPipeline(max_holding_period=0)

    def test_rejects_bad_vol_span(self):
        with pytest.raises(ValueError):
            MetaLabelingPipeline(vol_span=1)

    def test_rejects_bad_time_decay(self):
        with pytest.raises(ValueError):
            MetaLabelingPipeline(time_decay=-0.1)
        with pytest.raises(ValueError):
            MetaLabelingPipeline(time_decay=1.1)


# ---------------------------------------------------------------------------
# MetaLabelingPipeline — main flow
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:
    def _setup(self, n=500, drift=0.0, sigma=0.01, seed=7):
        rng = np.random.default_rng(seed)
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(drift, sigma, n))),
            index=_bar_index(n),
        )
        features = _features(close)
        return close, features

    def test_returns_aligned_xywithmatching_lengths(self):
        close, features = self._setup()
        signals = _signals(range(50, 450, 20), close)

        pipe = MetaLabelingPipeline(max_holding_period=30, vol_span=50)
        X, y, w = pipe.prepare_training_data(close, signals, features)

        assert len(X) == len(y) == len(w)
        assert len(X) > 0
        assert X.index.equals(y.index)
        assert X.index.equals(w.index)
        assert X.index.name == "event_timestamp"

    def test_family_one_hot_encoded(self):
        close, features = self._setup()
        # Mix of two families on non-overlapping event times.
        ts_sigs = _signals(range(50, 250, 20), close, family="ts_momentum")
        mr_sigs = _signals(range(260, 450, 20), close, family="mean_reversion")
        signals = pd.concat([ts_sigs, mr_sigs], ignore_index=True)

        pipe = MetaLabelingPipeline(max_holding_period=30, vol_span=50)
        X, _, _ = pipe.prepare_training_data(close, signals, features)

        assert "signal_family_ts_momentum" in X.columns
        assert "signal_family_mean_reversion" in X.columns
        # One-hot: each row sums to 1 across family columns.
        fam_cols = [c for c in X.columns if c.startswith("signal_family_")]
        assert (X[fam_cols].sum(axis=1) == 1.0).all()
        # Pipeline tracked the family vocabulary.
        assert set(pipe.family_columns_) == {
            "signal_family_ts_momentum", "signal_family_mean_reversion"
        }

    def test_meta_labels_are_binary(self):
        close, features = self._setup()
        signals = _signals(range(50, 450, 20), close)
        pipe = MetaLabelingPipeline(max_holding_period=30, vol_span=50)
        _, y, _ = pipe.prepare_training_data(close, signals, features)
        assert y.dtype.kind in ("i", "u")
        assert set(y.unique()).issubset({0, 1})

    def test_sample_weights_positive_and_sum_to_n(self):
        close, features = self._setup()
        signals = _signals(range(50, 450, 20), close)
        pipe = MetaLabelingPipeline(max_holding_period=30, vol_span=50,
                                    time_decay=0.5)
        _, y, w = pipe.prepare_training_data(close, signals, features)
        assert (w > 0).all()
        assert np.isclose(w.sum(), len(y))

    def test_signal_side_and_confidence_preserved(self):
        close, features = self._setup()
        signals = _signals(range(50, 450, 20), close,
                           side=-1, confidence=0.82)
        pipe = MetaLabelingPipeline(max_holding_period=30, vol_span=50)
        X, _, _ = pipe.prepare_training_data(close, signals, features)
        assert (X["signal_side"] == -1).all()
        assert np.allclose(X["signal_confidence"], 0.82)

    def test_neutral_signals_are_dropped(self):
        close, features = self._setup()
        good = _signals(range(100, 200, 20), close, side=1)
        neutral = _signals(range(250, 350, 20), close, side=0)
        signals = pd.concat([good, neutral], ignore_index=True)
        pipe = MetaLabelingPipeline(max_holding_period=30, vol_span=50)
        X, _, _ = pipe.prepare_training_data(close, signals, features)
        # All kept rows have non-zero side.
        assert (X["signal_side"] != 0).all()
        # Count matches the directional subset.
        assert len(X) == len(good)

    def test_empty_signals_returns_empty_output(self):
        close, features = self._setup()
        empty = pd.DataFrame(columns=["timestamp", "symbol", "family",
                                      "side", "confidence"])
        pipe = MetaLabelingPipeline()
        X, y, w = pipe.prepare_training_data(close, empty, features)
        assert len(X) == len(y) == len(w) == 0

    def test_missing_signal_columns_raise(self):
        close, features = self._setup()
        bad = pd.DataFrame({"timestamp": [close.index[10]], "side": [1]})
        pipe = MetaLabelingPipeline()
        with pytest.raises(ValueError):
            pipe.prepare_training_data(close, bad, features)

    def test_multiple_families_same_timestamp_kept_separate(self):
        close, features = self._setup()
        ts = close.index[100]
        signals = pd.DataFrame({
            "timestamp": [ts, ts],
            "symbol": ["A", "A"],
            "family": ["ts_momentum", "mean_reversion"],
            "side": [1, -1],
            "confidence": [0.7, 0.6],
        })
        pipe = MetaLabelingPipeline(max_holding_period=30, vol_span=50)
        X, y, _ = pipe.prepare_training_data(close, signals, features)
        # Both signals become separate events.
        assert len(X) == 2
        assert set(X["signal_side"].tolist()) == {-1, 1}


# ---------------------------------------------------------------------------
# End-to-end label-quality sanity checks
# ---------------------------------------------------------------------------

class TestLabelQuality:
    def test_trending_plus_momentum_labels_mostly_one(self):
        """With a strong deterministic uptrend, momentum LONGs almost always win."""
        n = 500
        # Gentle deterministic drift + tiny noise — side=+1 momentum bets
        # should hit the upper barrier nearly every time.
        idx = _bar_index(n)
        prices = 100.0 + 0.2 * np.arange(n)
        close = pd.Series(prices, index=idx)
        features = _features(close)

        signals = _signals(range(150, 450, 10), close,
                           family="ts_momentum", side=1)
        pipe = MetaLabelingPipeline(
            max_holding_period=30, vol_span=50,
        )
        _, y, _ = pipe.prepare_training_data(close, signals, features)
        assert y.mean() > 0.9, (
            f"expected >90% profitable in a strong uptrend, got {y.mean():.2%}"
        )

    def test_random_walk_random_signals_near_half(self):
        """On a driftless random walk, label=1 hit rate should sit near 50%."""
        rng = np.random.default_rng(42)
        n = 2000
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))),
            index=_bar_index(n),
        )
        features = _features(close)
        event_positions = range(200, 1800, 10)
        sides = rng.choice([-1, 1], size=len(list(event_positions)))
        signals = pd.DataFrame({
            "timestamp": close.index[list(event_positions)],
            "symbol": ["TEST"] * len(sides),
            "family": ["ts_momentum"] * len(sides),
            "side": sides,
            "confidence": [0.5] * len(sides),
        })
        pipe = MetaLabelingPipeline(
            max_holding_period=30, vol_span=50,
        )
        _, y, _ = pipe.prepare_training_data(close, signals, features)
        # Symmetry: random side on a driftless walk → ~50% win rate.
        assert 0.35 < y.mean() < 0.65, (
            f"expected ~50% on random walk, got {y.mean():.2%}"
        )


# ---------------------------------------------------------------------------
# prepare_live_features
# ---------------------------------------------------------------------------

class TestPrepareLiveFeatures:
    def test_live_features_align_with_trained_columns(self):
        # Train first so the pipeline records the expected column order.
        rng = np.random.default_rng(0)
        close = pd.Series(
            100.0 + np.cumsum(rng.normal(0, 0.1, 400)),
            index=_bar_index(400),
        )
        features = _features(close)
        ts_sigs = _signals(range(100, 350, 15), close, family="ts_momentum")
        mr_sigs = _signals(range(120, 330, 15), close, family="mean_reversion")
        signals = pd.concat([ts_sigs, mr_sigs], ignore_index=True)

        pipe = MetaLabelingPipeline(max_holding_period=20, vol_span=50)
        X, _, _ = pipe.prepare_training_data(close, signals, features)
        trained_cols = list(X.columns)

        # Now build a live feature row for a ts_momentum signal.
        sig = Signal(
            timestamp=close.index[380].to_pydatetime(),
            symbol="TEST", family="ts_momentum", side=1, confidence=0.8,
        )
        live_row = pipe.prepare_live_features(features.iloc[380], sig)
        assert list(live_row.columns) == trained_cols
        assert len(live_row) == 1
        # Correct one-hot: only ts_momentum column is 1.
        assert live_row["signal_family_ts_momentum"].iloc[0] == 1.0
        assert live_row["signal_family_mean_reversion"].iloc[0] == 0.0
        assert live_row["signal_side"].iloc[0] == 1
        assert live_row["signal_confidence"].iloc[0] == 0.8

    def test_unseen_family_produces_all_zero_one_hot(self):
        rng = np.random.default_rng(0)
        close = pd.Series(
            100.0 + np.cumsum(rng.normal(0, 0.1, 300)),
            index=_bar_index(300),
        )
        features = _features(close)
        signals = _signals(range(80, 250, 15), close, family="ts_momentum")

        pipe = MetaLabelingPipeline(max_holding_period=20, vol_span=50)
        pipe.prepare_training_data(close, signals, features)

        unseen = Signal(
            timestamp=close.index[200].to_pydatetime(),
            symbol="TEST", family="new_alpha", side=-1, confidence=0.6,
        )
        row = pipe.prepare_live_features(features.iloc[200], unseen)
        # All family one-hots should be zero since "new_alpha" wasn't trained on.
        fam_cols = [c for c in row.columns if c.startswith("signal_family_")]
        assert row[fam_cols].to_numpy().sum() == 0.0
