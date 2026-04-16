"""Tests for the LSTM regime detector (design-doc §7.2)."""

import os

# Mirror tests/conftest.py — set before torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.ml_layer.regime_detector import (
    REGIME_NAMES,
    RegimeDetector,
    label_regimes,
    predict_regime,
    train_regime_detector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_regime_series(n: int = 1200, seed: int = 0):
    """
    Concatenate three regime-like sub-series so label_regimes has something
    obviously non-iid to fit on.

    Layout:
        segment A (0 : n/3)      low vol, positive drift (trending_up)
        segment B (n/3 : 2n/3)   low vol, negative drift (trending_down)
        segment C (2n/3 : n)     high vol, zero drift    (high_volatility)
    """
    rng = np.random.default_rng(seed)
    k = n // 3
    seg_a = rng.normal(0.002, 0.005, k)   # up
    seg_b = rng.normal(-0.002, 0.005, k)  # down
    seg_c = rng.normal(0.0, 0.02, n - 2 * k)  # high vol
    rets = np.concatenate([seg_a, seg_b, seg_c])

    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = pd.Series(rets, index=idx, name="ret")
    # Rolling std (ex-post) to serve as volatility feature.
    vol = returns.rolling(20, min_periods=5).std().bfill().fillna(
        returns.std()
    )
    return returns, vol


def _synthetic_features_and_labels(n: int = 600, n_features: int = 5, seed: int = 0):
    """Features whose label is a deterministic function of the first column."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    data = rng.normal(size=(n, n_features))
    X = pd.DataFrame(
        data, index=idx, columns=[f"f{i}" for i in range(n_features)],
    )
    # Label = rolling-mean bucket of f0 — a time-local signal the LSTM can learn.
    rolling = X["f0"].rolling(30, min_periods=1).mean()
    labels = pd.Series(
        pd.qcut(rolling, q=4, labels=False, duplicates="drop"),
        index=idx, name="regime",
    ).astype("float")  # allow NaN initially
    return X, labels.astype("Int64").astype(int)


# ---------------------------------------------------------------------------
# RegimeDetector forward pass
# ---------------------------------------------------------------------------

class TestRegimeDetectorForward:
    def test_output_shape(self):
        model = RegimeDetector(input_dim=6, hidden_dim=32, n_layers=1)
        x = torch.randn(4, 60, 6)  # [batch, seq, features]
        out = model(x)
        assert out.shape == (4, 4)

    def test_softmax_probabilities_sum_to_one(self):
        model = RegimeDetector(input_dim=6, hidden_dim=16, n_layers=1)
        x = torch.randn(8, 60, 6)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_return_attention_shape(self):
        model = RegimeDetector(input_dim=3, hidden_dim=8, n_layers=1,
                               sequence_length=10)
        x = torch.randn(2, 10, 3)
        logits, attn = model(x, return_attention=True)
        assert logits.shape == (2, 4)
        assert attn.shape == (2, 10)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_rejects_non_3d(self):
        model = RegimeDetector(input_dim=3)
        with pytest.raises(ValueError):
            model(torch.randn(4, 3))

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            RegimeDetector(input_dim=0)
        with pytest.raises(ValueError):
            RegimeDetector(input_dim=3, dropout=1.0)
        with pytest.raises(ValueError):
            RegimeDetector(input_dim=3, sequence_length=0)


# ---------------------------------------------------------------------------
# label_regimes
# ---------------------------------------------------------------------------

class TestLabelRegimes:
    def test_produces_four_labels(self):
        returns, vol = _synthetic_regime_series(n=900)
        labels = label_regimes(returns, vol, n_regimes=4)
        unique = set(labels.dropna().unique().astype(int).tolist())
        # Canonical mapping produces labels in {0, 1, 2, 3}.
        assert unique.issubset({0, 1, 2, 3})
        assert len(unique) >= 2  # HMM found at least two distinct regimes.

    def test_high_vol_segment_gets_high_vol_label(self):
        returns, vol = _synthetic_regime_series(n=1200, seed=1)
        labels = label_regimes(returns, vol, n_regimes=4).dropna().astype(int)
        # The last third of the series is the high-vol segment.
        n = len(labels)
        tail = labels.iloc[2 * n // 3 :]
        # The dominant label in the tail should be 3 (high_volatility).
        top = tail.value_counts().idxmax()
        assert top == 3, f"expected tail regime = 3 (high_vol); got {top}"

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            # Length mismatch.
            label_regimes(pd.Series([1, 2]), pd.Series([0.1]), n_regimes=2)
        with pytest.raises(ValueError):
            # Too short: 20 < 10 * n_regimes(4) = 40 → must raise.
            label_regimes(
                pd.Series(np.zeros(20)), pd.Series(np.ones(20)),
                n_regimes=4,
            )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTrainRegimeDetector:
    def test_loss_decreases(self):
        X, labels = _synthetic_features_and_labels(n=600, seed=42)
        # Keep it small and fast.
        model, metrics = train_regime_detector(
            X, labels,
            sequence_length=30, epochs=20, batch_size=16, lr=5e-3,
            validation_split=0.2, hidden_dim=16, n_layers=1,
            verbose=False, random_state=0,
        )
        losses = metrics["train_loss"]
        assert len(losses) == 20
        # Loss should decrease overall (noisy but monotonic in trend).
        assert losses[-1] < losses[0] - 0.05, (
            f"loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
        )

    def test_returns_trained_model(self):
        X, labels = _synthetic_features_and_labels(n=400, seed=1)
        model, metrics = train_regime_detector(
            X, labels,
            sequence_length=20, epochs=5, batch_size=16,
            hidden_dim=8, n_layers=1,
        )
        assert isinstance(model, RegimeDetector)
        assert model.input_dim == X.shape[1]
        assert "best_epoch" in metrics and "val_loss" in metrics

    def test_purge_gap_prevents_overlap(self):
        X, labels = _synthetic_features_and_labels(n=300, seed=2)
        # With sequence_length=50 and validation_split=0.2, we need enough
        # sequences for the purge gap.
        model, metrics = train_regime_detector(
            X, labels,
            sequence_length=50, epochs=2, batch_size=16,
            hidden_dim=8, n_layers=1,
        )
        # Just asserts the function didn't error — if the purge gap maths
        # is wrong it raises "not enough sequences" instead.
        assert metrics["n_train"] > 0
        assert metrics["n_val"] > 0


# ---------------------------------------------------------------------------
# predict_regime
# ---------------------------------------------------------------------------

class TestPredictRegime:
    def test_columns(self):
        X, _ = _synthetic_features_and_labels(n=200, seed=3)
        model = RegimeDetector(
            input_dim=X.shape[1], hidden_dim=8, n_layers=1,
            sequence_length=30,
        )
        out = predict_regime(model, X, sequence_length=30)
        expected = {
            "regime", "prob_trending_up", "prob_trending_down",
            "prob_mean_reverting", "prob_high_vol",
        }
        assert expected == set(out.columns)
        assert len(out) == len(X)

    def test_probabilities_sum_to_one_where_defined(self):
        X, _ = _synthetic_features_and_labels(n=200, seed=4)
        model = RegimeDetector(
            input_dim=X.shape[1], hidden_dim=8, n_layers=1,
            sequence_length=30,
        )
        out = predict_regime(model, X, sequence_length=30)
        prob_cols = [c for c in out.columns if c.startswith("prob_")]
        defined = out[prob_cols].dropna()
        sums = defined.sum(axis=1)
        assert np.allclose(sums.to_numpy(), 1.0, atol=1e-4)

    def test_warmup_rows_are_nan(self):
        X, _ = _synthetic_features_and_labels(n=150, seed=5)
        model = RegimeDetector(
            input_dim=X.shape[1], hidden_dim=8, n_layers=1,
            sequence_length=30,
        )
        out = predict_regime(model, X, sequence_length=30)
        # First 30 rows: no history yet.
        head = out.iloc[:30]
        assert head["regime"].isna().all()
        assert head["prob_trending_up"].isna().all()

    def test_sequence_length_exceeds_data_raises(self):
        X, _ = _synthetic_features_and_labels(n=40, seed=6)
        model = RegimeDetector(
            input_dim=X.shape[1], hidden_dim=8, n_layers=1,
            sequence_length=60,
        )
        with pytest.raises(ValueError):
            predict_regime(model, X, sequence_length=100)

    def test_feature_width_mismatch_raises(self):
        X, _ = _synthetic_features_and_labels(n=200, seed=7, n_features=5)
        model = RegimeDetector(
            input_dim=3, hidden_dim=8, n_layers=1, sequence_length=30,
        )
        with pytest.raises(ValueError):
            predict_regime(model, X, sequence_length=30)
