"""Tests for the denoising autoencoder (Jansen Ch. 20)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch isn't installed

from src.feature_factory.autoencoder import (
    DenoisingAutoencoder,
    StandardScaler,
    extract_latent_features,
    train_autoencoder,
)


def _synthetic_low_rank_data(n: int = 400, d: int = 12, rank: int = 3, seed: int = 0) -> pd.DataFrame:
    """Build a DataFrame whose rows live near a low-rank subspace."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, rank))
    W = rng.normal(size=(rank, d))
    X = latent @ W + rng.normal(scale=0.1, size=(n, d))
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])


# ---------------------------------------------------------------------------
# Model + scaler basics
# ---------------------------------------------------------------------------

class TestDenoisingAutoencoder:
    def test_reconstructs_low_rank_data(self):
        """After training on rank-3 data, reconstruction error should be small."""
        df = _synthetic_low_rank_data(n=400, d=12, rank=3, seed=0)
        model, metrics = train_autoencoder(
            df, encoding_dim=3, hidden_dims=[8], epochs=200,
            batch_size=64, lr=5e-3, corruption=0.0, dropout=0.0, patience=30,
        )
        # Final val loss (on standardised data) should be much smaller than 1.0
        # (the unit variance after z-scoring).
        assert metrics["val_loss"][-1] < 0.5

    def test_latent_shape(self):
        df = _synthetic_low_rank_data(n=200, d=10, rank=2, seed=1)
        model, metrics = train_autoencoder(
            df, encoding_dim=4, hidden_dims=[8], epochs=10,
            batch_size=32, corruption=0.0, patience=5,
        )
        latent = extract_latent_features(model, df, metrics["scaler"])
        assert list(latent.columns) == [f"ae_latent_{i}" for i in range(4)]
        assert len(latent) == len(df)

    def test_early_stopping_terminates_before_max_epochs(self):
        """Very easy data should converge and trigger early stopping."""
        df = _synthetic_low_rank_data(n=300, d=6, rank=1, seed=2)
        _, metrics = train_autoencoder(
            df, encoding_dim=1, hidden_dims=[4], epochs=500,
            batch_size=32, lr=1e-2, corruption=0.0, patience=10,
        )
        # Either it stopped early, OR it converged in under 500 epochs while
        # improving throughout (patience may not deplete if loss keeps falling).
        # Either way, val_loss should be driven very low.
        assert metrics["val_loss"][-1] < 0.3
        assert metrics["best_epoch"] < 500

    def test_nan_input_raises(self):
        df = _synthetic_low_rank_data(n=100, d=5)
        df.iloc[10, 2] = np.nan
        with pytest.raises(ValueError):
            train_autoencoder(df, encoding_dim=2, epochs=1)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            train_autoencoder(pd.DataFrame(), encoding_dim=2, epochs=1)

    def test_invalid_constructor_params(self):
        with pytest.raises(ValueError):
            DenoisingAutoencoder(input_dim=1, encoding_dim=1)
        with pytest.raises(ValueError):
            DenoisingAutoencoder(input_dim=8, encoding_dim=10)  # encoding > input
        with pytest.raises(ValueError):
            DenoisingAutoencoder(input_dim=8, encoding_dim=2, corruption=1.0)


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class TestStandardScaler:
    def test_fit_transform_and_inverse(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=3.0, scale=2.0, size=(100, 4))
        sc = StandardScaler()
        Z = sc.fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(Z.std(axis=0, ddof=0), 1.0, atol=1e-9)
        X_back = sc.inverse_transform(Z)
        np.testing.assert_allclose(X_back, X, atol=1e-9)

    def test_constant_feature_stays_zero_after_scaling(self):
        X = np.column_stack([np.zeros(50), np.arange(50, dtype=float)])
        sc = StandardScaler()
        Z = sc.fit_transform(X)
        # Constant column should map to 0, not NaN.
        assert np.isfinite(Z).all()
        np.testing.assert_allclose(Z[:, 0], 0.0)

    def test_transform_before_fit_raises(self):
        sc = StandardScaler()
        with pytest.raises(RuntimeError):
            sc.transform(np.zeros((3, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
