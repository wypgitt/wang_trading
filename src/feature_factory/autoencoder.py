"""
Denoising Autoencoder Features (Jansen — ML for Algorithmic Trading, Ch. 20)

A symmetric dense autoencoder trained on the hand-crafted feature matrix
to learn compressed, noise-reduced latent features. Outputs are appended
to the raw feature matrix so the meta-labeler sees both interpretable and
learned representations (design doc §5.8).

PyTorch is imported lazily inside the training / extraction functions so
that importing this module never requires torch to be installed. Downstream
code that doesn't use the autoencoder (e.g. unit tests for other feature
modules) can import freely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# See conftest.py: PyTorch + openblas/Accelerate can ship duplicate libomp
# on macOS and refuse to coexist. The workaround is safe for inference and
# the small training runs we perform in tests / monthly retrains.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ---------------------------------------------------------------------------
# Simple z-score scaler (avoids a hard sklearn dependency)
# ---------------------------------------------------------------------------

@dataclass
class StandardScaler:
    """Minimal z-score scaler with fit / transform / inverse_transform."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute per-column mean and std from ``X``; return self."""
        self.mean = X.mean(axis=0)
        # Replace zero stdev with 1 to avoid NaN outputs for constant features.
        s = X.std(axis=0, ddof=0)
        self.std = np.where(s == 0.0, 1.0, s)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Z-score ``X`` using the stored mean/std. Requires prior ``fit``."""
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called first")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Shortcut for ``fit(X).transform(X)``."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Un-standardise ``X`` back to the original scale."""
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called first")
        return X * self.std + self.mean


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _torch():
    """Lazy-import PyTorch with a friendly error when missing."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Autoencoder requires `torch`. Install it (listed in "
            "requirements.txt under Phase 2) or skip autoencoder features."
        ) from exc
    import torch
    import torch.nn as nn

    return torch, nn


def _build_autoencoder_cls():
    """Build the denoising-autoencoder class lazily so torch is optional."""
    torch, nn = _torch()

    class _DenoisingAutoencoderImpl(nn.Module):
        """Symmetric dense autoencoder with denoising corruption and dropout.

        This class lives inside a factory so the torch imports stay lazy.
        The public entry point is :func:`DenoisingAutoencoder`.
        """

        def __init__(
            self,
            input_dim: int,
            encoding_dim: int = 8,
            hidden_dims: list[int] | None = None,
            dropout: float = 0.2,
            corruption: float = 0.2,
        ) -> None:
            super().__init__()
            if input_dim < 2:
                raise ValueError("input_dim must be >= 2")
            if encoding_dim < 1 or encoding_dim > input_dim:
                raise ValueError(
                    f"encoding_dim must be in [1, {input_dim}]"
                )
            if not (0.0 <= corruption < 1.0):
                raise ValueError("corruption must be in [0, 1)")
            if not (0.0 <= dropout < 1.0):
                raise ValueError("dropout must be in [0, 1)")

            hidden_dims = list(hidden_dims) if hidden_dims else [20]
            self.input_dim = input_dim
            self.encoding_dim = encoding_dim
            self.corruption = corruption

            # Encoder: input → h1 → h2 → ... → encoding_dim. Using ``Any``
            # in the local list hint avoids a mypy "nn.Module not defined"
            # error from the lazy-import pattern; the runtime type is
            # already enforced by torch.
            enc_layers: list[Any] = []
            prev = input_dim
            for h in hidden_dims:
                enc_layers.append(nn.Linear(prev, h))
                enc_layers.append(nn.ReLU())
                enc_layers.append(nn.Dropout(dropout))
                prev = h
            enc_layers.append(nn.Linear(prev, encoding_dim))
            # Tanh bottleneck bounds latents in [-1, 1] and stabilises
            # downstream ML-layer scaling.
            enc_layers.append(nn.Tanh())
            self.encoder = nn.Sequential(*enc_layers)

            # Decoder: encoding_dim → h2 → h1 → ... → input
            dec_layers: list[Any] = []
            prev = encoding_dim
            for h in reversed(hidden_dims):
                dec_layers.append(nn.Linear(prev, h))
                dec_layers.append(nn.ReLU())
                prev = h
            dec_layers.append(nn.Linear(prev, input_dim))
            self.decoder = nn.Sequential(*dec_layers)

        def _corrupt(self, x):
            """Randomly zero out ``corruption`` fraction of inputs at train time."""
            if not self.training or self.corruption <= 0:
                return x
            mask = (torch.rand_like(x) > self.corruption).to(x.dtype)
            return x * mask

        def forward(self, x):
            """Full autoencoder pass: encode (with corruption) then decode."""
            z = self.encoder(self._corrupt(x))
            return self.decoder(z)

        def encode(self, x):
            """Encoder-only pass — returns the bottleneck latents for ``x``."""
            return self.encoder(x)

    return _DenoisingAutoencoderImpl


# Expose the factory at module level under its conventional class-style name.
# It's only instantiated when torch is available; tests that never touch
# the autoencoder don't pay the import cost.
def DenoisingAutoencoder(*args: Any, **kwargs: Any):  # noqa: N802
    """Build a DenoisingAutoencoder instance, lazy-importing torch."""
    cls = _build_autoencoder_cls()
    return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    features: pd.DataFrame,
    encoding_dim: int = 8,
    hidden_dims: list[int] | None = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    corruption: float = 0.2,
    dropout: float = 0.2,
    validation_split: float = 0.2,
    patience: int = 10,
    seed: int = 0,
):
    """
    Train a denoising autoencoder on ``features``.

    Returns (model, metrics_dict) where metrics_dict has:
        scaler           : fitted StandardScaler
        train_loss       : list[float] per epoch
        val_loss         : list[float] per epoch
        best_epoch       : int
        stopped_early    : bool

    Args:
        features:         DataFrame of numeric features. Must not contain NaN.
        encoding_dim:     Size of the bottleneck layer.
        hidden_dims:      Symmetric hidden layer widths (default [20]).
        epochs:           Maximum training epochs.
        batch_size:       Mini-batch size.
        lr:               Adam learning rate.
        corruption:       Fraction of inputs zeroed each forward pass.
        dropout:          Dropout rate in hidden layers.
        validation_split: Fraction of data reserved for validation.
        patience:         Early-stopping patience on val_loss.
        seed:             RNG seed for reproducibility.
    """
    torch, nn = _torch()

    if features.empty:
        raise ValueError("features must be non-empty")
    if features.isna().any().any():
        raise ValueError("features contain NaN — impute or drop before training")
    if not (0.0 < validation_split < 0.5):
        raise ValueError("validation_split must be in (0, 0.5)")

    arr = features.to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("features contain non-finite values")

    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr).astype(np.float32)

    # Split into train / val. A random permutation is enough here — the
    # autoencoder is unsupervised and we're not predicting the future.
    n = len(arr_scaled)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = max(1, int(n * (1.0 - validation_split)))
    train_idx, val_idx = perm[:split], perm[split:]

    torch.manual_seed(seed)
    model = DenoisingAutoencoder(
        input_dim=arr_scaled.shape[1],
        encoding_dim=encoding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        corruption=corruption,
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train = torch.from_numpy(arr_scaled[train_idx])
    X_val = torch.from_numpy(arr_scaled[val_idx]) if len(val_idx) else X_train

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = patience

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        perm_ep = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_train), batch_size):
            batch = X_train[perm_ep[start : start + batch_size]]
            optim.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_loss = float(loss_fn(val_recon, X_val).item())
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.debug(
                    f"autoencoder: early stop at epoch {epoch} "
                    f"(best={best_epoch}, best_val={best_val:.5f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "scaler": scaler,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
        "stopped_early": patience_left <= 0,
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

def extract_latent_features(
    model,
    features: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Encode ``features`` with the trained model's encoder and return the
    latent activations as a DataFrame with columns ``ae_latent_0..{k-1}``.
    """
    torch, _ = _torch()

    if features.isna().any().any():
        raise ValueError("features contain NaN")

    arr = features.to_numpy(dtype=np.float32)
    arr_scaled = scaler.transform(arr).astype(np.float32)

    model.eval()
    with torch.no_grad():
        z = model.encode(torch.from_numpy(arr_scaled))
    latent = z.detach().cpu().numpy()
    cols = [f"ae_latent_{i}" for i in range(latent.shape[1])]
    return pd.DataFrame(latent, index=features.index, columns=cols)
