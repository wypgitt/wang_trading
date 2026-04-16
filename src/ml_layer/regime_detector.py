"""
Tier 2 Regime Detector (design-doc §7.2, Jansen)

An LSTM with an attention pooling head that classifies the prevailing
market regime from a rolling window of feature vectors. The regime
probabilities feed:

    * the Tier-1 meta-labeler as an additional feature block;
    * the bet-sizing cascade as a regime-conditional scaling factor.

Labels for supervised training are generated offline by a Gaussian HMM
fit on the joint ``[returns, volatility]`` series (:func:`label_regimes`).
The HMM's states are post-processed into semantic classes
(``trending_up``, ``trending_down``, ``mean_reverting``, ``high_volatility``)
so the LSTM learns a consistent label schema across retrains.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Silence the OpenMP duplicate-libomp warning when torch is imported alongside
# numpy/scipy that already loaded libomp. This mirrors the guard in
# tests/conftest.py; needed here because the module may be imported outside
# the pytest entrypoint (e.g. from a notebook or the orchestrator).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Imports below must stay after the env-var assignment above — torch reads
# KMP_DUPLICATE_LIB_OK at load time, so moving it earlier would defeat the
# workaround. Pylint flags this as wrong-import-position; suppress locally.
# pylint: disable=wrong-import-position,consider-using-from-import
import torch
import torch.nn as nn
import torch.nn.functional as F
# pylint: enable=wrong-import-position,consider-using-from-import

# Cap torch's intra-op parallelism — prevents the intermittent libomp
# segfault seen inside nn.LSTM on macOS. Setting OMP_NUM_THREADS globally
# would be stronger but breaks LightGBM, so we use the torch-only knob.
try:
    torch.set_num_threads(1)
except RuntimeError:  # pragma: no cover — already set
    pass


# Semantic regime names in canonical order. The HMM's raw state indices are
# remapped to this layout in :func:`label_regimes` so downstream code can
# rely on the ordering (trending_up=0, trending_down=1, ...).
REGIME_NAMES: tuple[str, ...] = (
    "trending_up",
    "trending_down",
    "mean_reverting",
    "high_volatility",
)


# ---------------------------------------------------------------------------
# Attention pooling head
# ---------------------------------------------------------------------------

class _AttentionPool(nn.Module):
    """
    Simple single-head attention that pools a [B, T, H] LSTM output down to
    [B, H] by a softmax-weighted sum across the time dimension.

    Scores are computed with a learned ``W : H → 1`` linear projection
    followed by softmax over the time axis, so the pooled vector emphasises
    the time steps the model finds most informative.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Softmax-weighted sum over the time axis.

        Args:
            hidden_states: Tensor shaped ``[B, T, H]``.

        Returns:
            (pooled ``[B, H]``, attention_weights ``[B, T]``).
        """
        scores = self.W(hidden_states).squeeze(-1)            # [B, T]
        weights = F.softmax(scores, dim=-1)                   # [B, T]
        pooled = torch.bmm(
            weights.unsqueeze(1), hidden_states,
        ).squeeze(1)                                          # [B, H]
        return pooled, weights


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector(nn.Module):
    """
    LSTM + attention pooling + linear head.

    Forward pass returns raw logits of shape ``[B, n_regimes]`` — apply
    ``softmax`` (or use the :func:`predict_regime` helper) for probabilities.

    Hyperparameters stored on the instance for checkpointing / introspection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_regimes: int = 4,
        dropout: float = 0.2,
        sequence_length: int = 60,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if hidden_dim < 1 or n_layers < 1 or n_regimes < 2:
            raise ValueError("invalid hidden_dim / n_layers / n_regimes")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_regimes = n_regimes
        self.dropout = dropout
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = _AttentionPool(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_regimes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, input_dim] float tensor.
            return_attention: if True, also return the [B, T] attention
                              weights (useful for explainability).
        """
        if x.dim() != 3:
            raise ValueError(
                f"expected 3-D tensor [B, T, F]; got shape {tuple(x.shape)}"
            )
        lstm_out, _ = self.lstm(x)                # [B, T, H]
        pooled, attn_weights = self.attention(lstm_out)
        logits = self.head(pooled)                # [B, n_regimes]
        if return_attention:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
# HMM-based regime labeling
# ---------------------------------------------------------------------------

def _postprocess_state_order(
    state_means: np.ndarray,   # [n_states, 2] — mean of [returns, vol]
) -> np.ndarray:
    """
    Map raw HMM states to semantic regimes.

    Returns a permutation vector ``perm`` such that the semantic label for
    raw HMM state ``k`` is ``perm[k]`` with the canonical ordering:

        0 trending_up      — highest mean return
        1 trending_down    — lowest mean return
        2 mean_reverting   — middle return state(s)
        3 high_volatility  — highest volatility among the remaining states

    For ``n_states != 4`` a best-effort mapping is produced: the highest-
    return state maps to 0, the lowest-return state to 1, the highest-
    volatility state to ``min(3, n_states-1)``, and any leftover states are
    mapped to 2 (mean-reverting). This keeps the LSTM target labels inside
    ``{0, 1, 2, 3}`` so the head width stays constant.
    """
    n_states = state_means.shape[0]
    returns = state_means[:, 0]
    vols = state_means[:, 1]

    # Assign high-vol FIRST so an up-trending state with modest vol doesn't
    # win "trending_up" over a genuinely-high-vol state just because the
    # high-vol segment also happened to drift slightly positive.
    high_vol = int(np.argmax(vols))

    remaining = [i for i in range(n_states) if i != high_vol]
    if remaining:
        up = max(remaining, key=lambda i: returns[i])
        down = min(remaining, key=lambda i: returns[i])
    else:
        up = down = high_vol

    perm = np.full(n_states, 2, dtype=int)  # default = mean_reverting (2)
    perm[high_vol] = 3
    perm[up] = 0
    if down != up:
        perm[down] = 1
    return perm


def label_regimes(
    returns: pd.Series,
    volatility: pd.Series,
    n_regimes: int = 4,
    random_state: int = 42,
    n_iter: int = 100,
) -> pd.Series:
    """
    Fit a Gaussian HMM on ``[returns, volatility]`` and emit regime labels.

    Labels follow the canonical mapping in :data:`REGIME_NAMES` so 0 always
    means "trending_up" regardless of the random HMM state ordering.

    Args:
        returns:     Per-bar return series.
        volatility:  Per-bar volatility series (same index as ``returns``).
        n_regimes:   HMM state count. Defaults to 4.
        random_state, n_iter:  HMM fit controls.

    Returns:
        pd.Series of integer labels indexed like ``returns`` (after joint
        dropna of returns & volatility).
    """
    from hmmlearn.hmm import GaussianHMM

    if returns.shape != volatility.shape:
        raise ValueError("returns and volatility must be the same length")
    if n_regimes < 2:
        raise ValueError("n_regimes must be >= 2")

    joint = pd.concat(
        {"ret": returns, "vol": volatility}, axis=1,
    ).dropna()
    if len(joint) < 10 * n_regimes:
        raise ValueError(
            f"need at least {10 * n_regimes} usable rows to fit a "
            f"{n_regimes}-state HMM (got {len(joint)})"
        )

    X = joint.to_numpy(dtype=float)
    # Diagonal covariance keeps the fit stable on finance-style data where
    # returns and vol have very different scales.
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
    )
    try:
        model.fit(X)
    except Exception as exc:  # noqa: BLE001 — hmmlearn raises various fit errors
        raise RuntimeError(f"hmm fit failed: {exc}") from exc

    raw_states = model.predict(X)              # [N] ints in [0, n_regimes)
    means = np.asarray(model.means_)           # [n_regimes, 2]
    perm = _postprocess_state_order(means)     # raw_state -> semantic
    semantic = perm[raw_states]

    labels = pd.Series(
        semantic, index=joint.index, dtype=int, name="regime",
    )
    # Reindex back onto the original returns index so callers can join
    # without losing alignment; missing rows (from dropna) are NaN.
    return labels.reindex(returns.index)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Per-epoch loss / accuracy history + best-epoch bookkeeping."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


def _build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_seqs, y_seqs)`` where ``X_seqs[i]`` = ``features[i:i+seq_len]``."""
    n = len(features)
    if n < seq_len + 1:
        raise ValueError(
            f"need at least {seq_len + 1} rows to build a single "
            f"length-{seq_len} sequence (got {n})"
        )
    n_seqs = n - seq_len
    # [n_seqs, seq_len, input_dim]
    X_seqs = np.stack([features[i : i + seq_len] for i in range(n_seqs)])
    # Label is the regime at the bar AT the end of the window.
    y_seqs = labels[seq_len : seq_len + n_seqs]
    return X_seqs, y_seqs


def train_regime_detector(
    features_df: pd.DataFrame,
    regime_labels: pd.Series,
    sequence_length: int = 60,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    validation_split: float = 0.2,
    hidden_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.2,
    device: str | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[RegimeDetector, dict[str, Any]]:
    """
    Fit an LSTM regime detector on a feature matrix + HMM-generated labels.

    Sequence creation: for each bar ``t >= sequence_length`` the input is
    ``features[t-sequence_length:t]`` and the target is ``regime_labels[t]``.
    Rows where the label is NaN are excluded.

    Split: purged time-series split — the last ``validation_split`` fraction
    of sequences goes to validation; the last ``sequence_length`` sequences
    *before* the val block are dropped so that no validation window shares
    any bar with a training window.

    Returns:
        (model, metrics_dict) where ``metrics_dict`` is a
        :class:`TrainingMetrics`-derived dict with per-epoch loss/accuracy
        and the best epoch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(random_state)

    if features_df.shape[0] != regime_labels.shape[0]:
        raise ValueError(
            "features_df and regime_labels must have the same length"
        )

    # Drop rows with NaN labels; align features to the surviving index.
    mask = regime_labels.notna().to_numpy()
    features = features_df.to_numpy(dtype=np.float32)[mask]
    labels_arr = regime_labels.to_numpy()[mask].astype(np.int64)

    if len(features) <= sequence_length + 1:
        raise ValueError(
            f"sequence_length ({sequence_length}) too large for data "
            f"(usable rows: {len(features)})"
        )

    X_seqs, y_seqs = _build_sequences(features, labels_arr, sequence_length)
    n_seqs = len(X_seqs)

    # Purged time-series split.
    n_val = max(1, int(round(validation_split * n_seqs)))
    val_start = n_seqs - n_val
    train_end = max(0, val_start - sequence_length)  # purge gap
    if train_end < 1:
        raise ValueError(
            "not enough sequences to maintain a purge gap; reduce "
            "sequence_length or validation_split"
        )

    X_tr = torch.from_numpy(X_seqs[:train_end]).float().to(device)
    y_tr = torch.from_numpy(y_seqs[:train_end]).long().to(device)
    X_val = torch.from_numpy(X_seqs[val_start:]).float().to(device)
    y_val = torch.from_numpy(y_seqs[val_start:]).long().to(device)

    n_regimes = int(max(labels_arr.max() + 1, 4))  # at least 4 to hold canonical labels
    input_dim = features.shape[1]
    model = RegimeDetector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_regimes=n_regimes,
        dropout=dropout,
        sequence_length=sequence_length,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    metrics = TrainingMetrics()
    best_state: dict[str, torch.Tensor] | None = None

    n_train = len(X_tr)
    for epoch in range(epochs):
        # --- train ---
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        train_correct = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_correct += (logits.argmax(dim=-1) == yb).sum().item()

        train_loss = train_loss_sum / max(n_train, 1)
        train_acc = train_correct / max(n_train, 1)

        # --- validate ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = float(criterion(val_logits, y_val).item())
            val_acc = float(
                (val_logits.argmax(dim=-1) == y_val).float().mean().item()
            )

        metrics.train_loss.append(float(train_loss))
        metrics.train_accuracy.append(float(train_acc))
        metrics.val_loss.append(val_loss)
        metrics.val_accuracy.append(val_acc)

        if val_loss < metrics.best_val_loss:
            metrics.best_val_loss = val_loss
            metrics.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            logger.info(
                f"regime epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    # Restore best weights.
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_dict = {
        "train_loss": metrics.train_loss,
        "val_loss": metrics.val_loss,
        "train_accuracy": metrics.train_accuracy,
        "val_accuracy": metrics.val_accuracy,
        "best_val_loss": metrics.best_val_loss,
        "best_epoch": metrics.best_epoch,
        "n_train": n_train,
        "n_val": len(X_val),
        "n_regimes": n_regimes,
    }
    return model, metrics_dict


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_regime(
    model: RegimeDetector,
    features_df: pd.DataFrame,
    sequence_length: int = 60,
    batch_size: int = 256,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Run the LSTM on a feature matrix and return per-bar regime probabilities.

    For each bar ``t >= sequence_length`` the window ``features[t-sequence_length:t]``
    is fed through the model. The first ``sequence_length`` rows of the
    output DataFrame are NaN (no history yet).

    Returns:
        DataFrame with columns
            ``regime`` (argmax; int),
            ``prob_trending_up``,
            ``prob_trending_down``,
            ``prob_mean_reverting``,
            ``prob_high_vol``,
        indexed like ``features_df``.
    """
    if sequence_length > len(features_df):
        raise ValueError(
            f"sequence_length ({sequence_length}) exceeds data length "
            f"({len(features_df)})"
        )
    if features_df.shape[1] != model.input_dim:
        raise ValueError(
            f"features_df has {features_df.shape[1]} features; "
            f"model expects {model.input_dim}"
        )
    if device is None:
        device = next(model.parameters()).device.type

    arr = features_df.to_numpy(dtype=np.float32)
    n = len(arr)
    n_seqs = n - sequence_length  # inference seqs end at t >= sequence_length

    prob_cols = [f"prob_{name}" for name in REGIME_NAMES]
    # Rename the last column to prob_high_vol (spec requested short form).
    prob_cols[-1] = "prob_high_vol"

    if n_seqs <= 0:
        # No usable sequences — return an all-NaN frame of the right shape.
        out = pd.DataFrame(
            np.nan, index=features_df.index, columns=["regime", *prob_cols],
        )
        out["regime"] = pd.NA
        return out

    model.eval()
    probs = np.full((n, model.n_regimes), np.nan, dtype=np.float32)
    regime_idx = np.full(n, -1, dtype=np.int64)

    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch_end = min(i + batch_size, n_seqs)
            windows = np.stack([
                arr[t : t + sequence_length] for t in range(i, batch_end)
            ])
            xb = torch.from_numpy(windows).float().to(device)
            logits = model(xb)
            batch_probs = F.softmax(logits, dim=-1).cpu().numpy()
            # Place results at the corresponding output bar t + sequence_length.
            for k, t in enumerate(range(i, batch_end)):
                out_row = t + sequence_length
                if out_row < n:
                    probs[out_row] = batch_probs[k]
                    regime_idx[out_row] = int(np.argmax(batch_probs[k]))

    data: dict[str, np.ndarray] = {"regime": regime_idx}
    for c, col_name in enumerate(prob_cols):
        # Only pad out to the first 4 canonical columns; extras (n_regimes > 4)
        # are not named in the public schema.
        if c < probs.shape[1]:
            data[col_name] = probs[:, c]
        else:
            data[col_name] = np.full(n, np.nan, dtype=np.float32)

    out = pd.DataFrame(data, index=features_df.index)
    # Mark warmup bars as NaN for probabilities and pd.NA for regime.
    out.iloc[:sequence_length, out.columns.get_loc("regime")] = -1
    out.loc[out["regime"] == -1, prob_cols] = np.nan
    # Prefer pd.NA over -1 sentinel for readability; cast to nullable Int.
    regime_series = out["regime"].astype("Int64")
    regime_series[regime_series == -1] = pd.NA
    out["regime"] = regime_series
    return out
