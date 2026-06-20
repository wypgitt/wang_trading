"""Focused tests for production bootstrap adapters."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.bootstrap import (
    DirectTargetOptimizer,
    ModelMetaPipeline,
    load_runtime_config,
)
from src.config import RuntimeConfigError


class _FakeMetaModel:
    feature_names_ = [
        "f1",
        "signal_side",
        "signal_confidence",
        "signal_family_momentum",
    ]

    def predict_proba(self, X):
        assert list(X.columns) == self.feature_names_
        assert X["signal_family_momentum"].iloc[0] == 1.0
        return np.array([0.73])


class _FakeCalibratedMetaModel:
    """Exposes a pre-calibration (raw) probability distinct from the
    isotonic-calibrated one via the ``return_raw`` protocol."""

    feature_names_ = _FakeMetaModel.feature_names_

    def predict_proba(self, X, *, return_raw=False):
        raw = np.array([0.80])
        calibrated = np.array([0.62])
        if return_raw:
            return raw, calibrated
        return calibrated


def test_model_meta_pipeline_aligns_live_features():
    features = pd.DataFrame(
        {"f1": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]),
    )
    signals = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2024-01-01T00:01:00Z"),
            "symbol": "AAPL",
            "family": "momentum",
            "side": 1,
            "confidence": 0.4,
        }
    ])

    out = ModelMetaPipeline(_FakeMetaModel()).predict(features, signals)

    assert len(out) == 1
    # Plain model without return_raw support: both fields equal (fallback).
    assert out["meta_prob"].iloc[0] == 0.73
    assert out["calibrated_prob"].iloc[0] == 0.73


def test_model_meta_pipeline_splits_raw_and_calibrated():
    features = pd.DataFrame(
        {"f1": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]),
    )
    signals = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2024-01-01T00:01:00Z"),
            "symbol": "AAPL",
            "family": "momentum",
            "side": 1,
            "confidence": 0.4,
        }
    ])

    out = ModelMetaPipeline(_FakeCalibratedMetaModel()).predict(features, signals)

    assert len(out) == 1
    # meta_prob is the raw model output; calibrated_prob is post-isotonic.
    assert out["meta_prob"].iloc[0] == 0.80
    assert out["calibrated_prob"].iloc[0] == 0.62


def test_direct_target_optimizer_aggregates_and_caps():
    bets = pd.DataFrame([
        {"symbol": "AAPL", "family": "momentum", "final_size": 0.08},
        {"symbol": "AAPL", "family": "mean_reversion", "final_size": 0.08},
        {"symbol": "MSFT", "family": "momentum", "final_size": -0.04},
    ])

    out = DirectTargetOptimizer(
        max_single_position=0.10,
        max_gross_exposure=0.20,
    ).compute_target_portfolio(bet_sizes=bets)

    assert set(out.columns) == {"symbol", "target_weight", "strategy"}
    assert out.loc[out["symbol"] == "AAPL", "target_weight"].iloc[0] == 0.10
    assert out["target_weight"].abs().sum() <= 0.20


def test_runtime_config_expands_env_values(tmp_path):
    os.environ["BOOTSTRAP_TEST_SECRET"] = "expanded"
    path = tmp_path / "runtime.yaml"
    path.write_text("secret: env:BOOTSTRAP_TEST_SECRET\nnested:\n  x: 1\n")

    out = load_runtime_config(path)

    assert out["secret"] == "expanded"
    assert out["nested"]["x"] == 1


def test_live_runtime_config_validation_reports_bad_paths(tmp_path):
    path = tmp_path / "live.yaml"
    path.write_text(
        "asset_class: equities\n"
        "symbols: []\n"
        "pipeline: []\n"
        "surprise: true\n"
    )

    try:
        load_runtime_config(path, default_name="live_trading")
    except RuntimeConfigError as exc:
        msg = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeConfigError")

    assert "symbols must be a non-empty list" in msg
    assert "pipeline must be a mapping" in msg
    assert "unknown top-level key: surprise" in msg
