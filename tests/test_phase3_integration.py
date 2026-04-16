"""
End-to-end Phase 3 integration test.

Exercises the full learning/serving pipeline:

    synthetic daily bars (5 symbols, 3 years)
        ↓  FeatureAssembler   (Phase 2)
        ↓  CUSUM event filter
        ↓  SignalBattery       (Phase 2; ts_momentum + mean_reversion)
        ↓  MetaLabelingPipeline (P3.04)
        ↓  MetaLabeler (P3.05) with purged-CV early stopping
        ↓  MDI / MDA / SFI     (P3.07) + select_features
        ↓  BetSizingCascade    (P3.12)

Pass criteria:
    * Purged-CV accuracy > 50 % (meta-labeler beats random).
    * Calibrated-prob binned error < 5 %.
    * Every bet size lies in ``[0, max_size]`` and is side-signed; no
      constraint rule produces a size outside the risk budget.
    * Wall-clock < 120 s.

Tagged ``@pytest.mark.integration`` — excluded from the default ``pytest``
run via ``pytest.ini``; opt in with ``pytest -m integration -o addopts=""``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.bet_sizing.cascade import BetSizingCascade, CascadeConfig, FamilyStats
from src.data_engine.bars.cusum_filter import (
    compute_cusum_threshold,
    cusum_filter,
)
from src.feature_factory.assembler import FeatureAssembler
from src.labeling.meta_labeler_pipeline import MetaLabelingPipeline
from src.ml_layer.feature_importance import (
    mda_importance,
    mdi_importance,
    select_features,
    sfi_importance,
)
from src.ml_layer.meta_labeler import MetaLabeler
from src.ml_layer.purged_cv import PurgedKFoldCV, cross_val_score_purged
from src.signal_battery.mean_reversion import MeanReversionSignal
from src.signal_battery.momentum import TimeSeriesMomentumSignal
from src.signal_battery.orchestrator import SignalBattery


# ---------------------------------------------------------------------------
# Synthetic market generator — daily bars, 5 symbols
# ---------------------------------------------------------------------------

def _generate_bars(
    n_days: int = 756,  # ~3 years of daily trading
    seed: int = 11,
) -> dict[str, pd.DataFrame]:
    """
    Build a 5-symbol universe with distinct return-generating processes:

        TREND_UP  — GBM with strong positive drift (momentum-favourable)
        TREND_DN  — GBM with strong negative drift
        MEAN_REV  — AR(1) around 100, phi = 0.92 (mean-reversion-favourable)
        RW_1,RW_2 — zero-drift GBM (no edge)

    Each symbol returns a bars DataFrame with the schema
    :class:`FeatureAssembler` expects (close, volume, dollar_volume,
    buy_volume, sell_volume, tick_count, bar_duration_seconds).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    bars: dict[str, pd.DataFrame] = {}

    def _finalise(symbol: str, close: np.ndarray) -> pd.DataFrame:
        # Log-normal per-day volume; tie buy/sell imbalance loosely to
        # the daily return sign so the microstructure block has signal.
        rets = np.diff(np.log(close), prepend=np.log(close[0]))
        vol = rng.lognormal(mean=12.0, sigma=0.4, size=n_days)
        buy_frac = 0.5 + 0.3 * np.tanh(rets * 20.0)
        df = pd.DataFrame({
            "close": close,
            "volume": vol,
            "dollar_volume": close * vol,
            "buy_volume": vol * buy_frac,
            "sell_volume": vol * (1.0 - buy_frac),
            "tick_count": np.clip((vol / 10.0).astype(int), 10, None),
            "bar_duration_seconds": np.full(n_days, 23400.0),  # one trading day
        }, index=idx)
        return df

    # Trending up.
    drift, sigma = 0.0006, 0.012
    rets_up = rng.normal(drift, sigma, n_days)
    bars["TREND_UP"] = _finalise("TREND_UP", 100.0 * np.exp(np.cumsum(rets_up)))

    # Trending down.
    rets_dn = rng.normal(-drift, sigma, n_days)
    bars["TREND_DN"] = _finalise("TREND_DN", 100.0 * np.exp(np.cumsum(rets_dn)))

    # Mean reverting (AR(1) around 100).
    phi = 0.92
    eps = rng.normal(0.0, 1.2, n_days)
    y = np.zeros(n_days)
    for t in range(1, n_days):
        y[t] = phi * y[t - 1] + eps[t]
    mean_rev_close = 100.0 + y
    # Guard against non-positive prices from the AR shocks.
    mean_rev_close = np.clip(mean_rev_close, 50.0, None)
    bars["MEAN_REV"] = _finalise("MEAN_REV", mean_rev_close)

    # Two random walks.
    for name in ("RW_1", "RW_2"):
        rets = rng.normal(0.0, sigma, n_days)
        bars[name] = _finalise(name, 100.0 * np.exp(np.cumsum(rets)))

    return bars


def _assembler_config() -> dict:
    """
    Fast feature-matrix config: short windows + bypass of the row-hungry
    FFD / post-hoc stationarity passes.

    The default assembler runs two things that eat rows brutally on short
    synthetic slices:

        1. A leading FFD block whose window grows as the threshold tightens
           — on our 756-row series with a 1e-3 default threshold the window
           is hundreds of rows, which cascades through ``dropna`` at the end.
        2. A post-hoc stationarisation pass that refits FFD per column for
           every feature whose ADF p-value ≥ 0.05. On trending and random-
           walk symbols 10–15 features fail ADF and each second-pass FFD
           adds more warmup NaN.

    For the integration test we want a quickly-available, finite feature
    matrix, not perfect stationarity. Disable both:

        * ``ffd.columns = []`` — skip the leading FFD block entirely.
        * ``stationarity.p_value = 1.01`` — make the post-hoc ADF gate
          pass for every feature so no column gets differenced.

    The ML model still learns from the raw features; non-stationarity
    costs a bit of edge but doesn't mask the "purged-CV beats random"
    signal we're asserting.
    """
    return {
        "ffd": {
            "columns": [],  # skip leading FFD block (avoids huge warmup)
            "p_value": 0.05, "max_d": 1.0,
        },
        "structural_breaks": {
            "window": 30, "min_window_sadf": 15, "min_period_chow": 20,
        },
        "entropy": {"window": 40},
        "microstructure": {"window": 30},
        "volatility": {
            "window": 60, "refit_interval": 30,
            "short_window": 5, "long_window": 20, "vvol_window": 20,
        },
        "classical": {
            "rsi_window": 14, "bbw_window": 20, "ret_z_windows": [5, 10, 20],
        },
        "stationarity": {
            "p_value": 1.01,  # bypass the post-hoc stationarity pass
            "ffd_max_d": 1.0,
            "ffd_threshold": 1e-2,
        },
    }


# ---------------------------------------------------------------------------
# Per-symbol feature/signal/label pipeline
# ---------------------------------------------------------------------------

def _run_symbol_pipeline(
    symbol: str,
    bars: pd.DataFrame,
    battery: SignalBattery,
    pipeline: MetaLabelingPipeline,
    assembler_config: dict,
):
    """Returns per-symbol (X, y, weights, features_df, events, signals_df)."""
    assembler = FeatureAssembler(config=assembler_config)
    features = assembler.assemble(bars)
    if features.empty:
        return None

    close = bars["close"].loc[features.index]
    threshold = compute_cusum_threshold(close, multiplier=1.0, lookback=100)
    events = cusum_filter(close, threshold=threshold)
    events = events.intersection(features.index)
    if len(events) < 20:
        return None

    signals = battery.generate_all(
        bars.loc[features.index], event_timestamps=events, symbol=symbol,
    )
    directional = signals[signals["side"] != 0]
    if directional.empty:
        return None

    X, y, w = pipeline.prepare_training_data(close, directional, features)
    if len(X) == 0:
        return None

    return {
        "symbol": symbol,
        "X": X, "y": y, "w": w,
        "features": features,
        "events": events,
        "signals": directional,
        "close": close,
    }


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------

def _max_binned_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    min_bin_count: int = 20,
) -> float:
    """Max |mean_pred − mean_actual| across equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.digitize(y_prob, bins) - 1
    edges = np.clip(edges, 0, n_bins - 1)
    max_err = 0.0
    for b in range(n_bins):
        mask = edges == b
        if mask.sum() < min_bin_count:
            continue
        err = abs(y_prob[mask].mean() - y_true[mask].mean())
        max_err = max(max_err, err)
    return max_err


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_phase3_full_pipeline():
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Generate synthetic market.
    # ------------------------------------------------------------------
    bars_by_symbol = _generate_bars(n_days=756, seed=11)
    assert set(bars_by_symbol.keys()) == {
        "TREND_UP", "TREND_DN", "MEAN_REV", "RW_1", "RW_2",
    }
    logger.info(f"integration p3: generated {len(bars_by_symbol)} symbols")

    # ------------------------------------------------------------------
    # 2. Shared battery + meta-labeling pipeline.
    # ------------------------------------------------------------------
    battery = SignalBattery()
    battery.register(
        TimeSeriesMomentumSignal(
            params={
                "lookbacks": [10, 20, 40],
                "history_window": 80,
                "min_history": 100,
            }
        ),
        kind="bars",
    )
    battery.register(
        MeanReversionSignal(
            params={"min_halflife": 1.0, "max_halflife": 120.0},
        ),
        kind="bars",
    )

    pipeline = MetaLabelingPipeline(
        max_holding_period=20, vol_span=50, time_decay=1.0,
    )

    # ------------------------------------------------------------------
    # 3. Run the per-symbol pipeline and gather training frames.
    # ------------------------------------------------------------------
    per_symbol = []
    for symbol, bars in bars_by_symbol.items():
        out = _run_symbol_pipeline(
            symbol, bars, battery, pipeline, _assembler_config(),
        )
        if out is not None:
            per_symbol.append(out)
    assert len(per_symbol) >= 3, (
        f"expected at least 3 symbols to produce training data; got {len(per_symbol)}"
    )

    # Concatenate across symbols. Features use intersection of columns so
    # the resulting X has a consistent schema.
    common_cols: set[str] | None = None
    for sym in per_symbol:
        cols = set(sym["X"].columns)
        common_cols = cols if common_cols is None else (common_cols & cols)
    assert common_cols, "no common feature columns across symbols"
    column_order = [c for c in per_symbol[0]["X"].columns if c in common_cols]

    X_all = pd.concat(
        [s["X"][column_order].reset_index(drop=True) for s in per_symbol],
        ignore_index=True,
    )
    y_all = pd.concat(
        [s["y"].reset_index(drop=True) for s in per_symbol],
        ignore_index=True,
    ).astype(int)
    w_all = pd.concat(
        [s["w"].reset_index(drop=True) for s in per_symbol],
        ignore_index=True,
    )
    # labels_df for PurgedKFoldCV: event_start = the concatenated event
    # timestamp (rebuilt from each symbol's X index), event_end = event_start
    # + max_holding_period × bar_freq (daily). We don't have the exact exit
    # timestamp here; use event_start + 20 days as a conservative proxy.
    event_starts: list[pd.Timestamp] = []
    event_ends: list[pd.Timestamp] = []
    for s in per_symbol:
        starts = pd.DatetimeIndex(s["X"].index)
        event_starts.extend(starts.tolist())
        event_ends.extend((starts + pd.Timedelta(days=30)).tolist())
    labels_df = pd.DataFrame(
        {"event_start": event_starts, "event_end": event_ends},
    )
    logger.info(
        f"integration p3: training frame rows={len(X_all)} "
        f"cols={X_all.shape[1]} y_mean={y_all.mean():.3f}"
    )
    assert len(X_all) > 200
    assert y_all.nunique() == 2, "need both profitable and unprofitable labels"

    # ------------------------------------------------------------------
    # 4. Train the MetaLabeler with purged-CV early stopping.
    # ------------------------------------------------------------------
    model = MetaLabeler(
        model_type="lightgbm",
        params={
            "n_estimators": 200, "learning_rate": 0.05, "max_depth": 4,
            "min_child_weight": 5,
        },
        calibrate=True,
    ).fit(X_all, y_all, sample_weight=w_all, labels_df=labels_df)
    assert model.oof_predictions_ is not None

    # Purged-CV accuracy check (fresh CV run on the tuned model).
    scores = cross_val_score_purged(
        model.model_, X_all, y_all, labels_df,
        n_splits=3, embargo_pct=0.01, scoring="accuracy",
    )
    cv_accuracy = float(np.nanmean(scores))
    logger.info(f"integration p3: purged-CV accuracy = {cv_accuracy:.3f}")
    assert cv_accuracy > 0.50, (
        f"meta-labeler accuracy {cv_accuracy:.3f} not better than random"
    )

    # Calibration check — use OOF predictions to avoid in-sample optimism.
    oof_mask = np.isfinite(model.oof_predictions_)
    if model.calibrator_ is not None:
        calibrated = model.calibrator_.transform(model.oof_predictions_[oof_mask])
    else:
        calibrated = model.oof_predictions_[oof_mask]
    cal_err = _max_binned_calibration_error(
        y_all.to_numpy()[oof_mask], calibrated, n_bins=10, min_bin_count=15,
    )
    logger.info(f"integration p3: max binned calibration error = {cal_err:.3f}")
    assert cal_err < 0.05, (
        f"calibration error {cal_err:.3f} exceeds 5 % threshold"
    )

    # ------------------------------------------------------------------
    # 5. Feature importance — MDI / MDA / SFI.
    # ------------------------------------------------------------------
    mdi = mdi_importance(model, list(X_all.columns))
    assert np.isclose(mdi.sum(), 1.0)

    # Keep MDA cheap: 2 splits × 2 repeats × n_features.
    mda = mda_importance(
        model, X_all, y_all, labels_df,
        n_splits=2, n_repeats=2, scoring="accuracy",
    )

    # SFI: accuracy per feature with small n_estimators for speed.
    sfi_model_type = "lightgbm"
    sfi = sfi_importance(
        X_all, y_all, labels_df,
        model_type=sfi_model_type, n_splits=2, scoring="accuracy",
    )

    importances = {
        "mdi": mdi.to_frame("importance"),
        "mda": mda,
        "sfi": sfi.to_frame("score"),
    }
    kept = select_features(
        importances, mda_pvalue_threshold=0.20, min_sfi_score=0.51,
    )
    logger.info(
        f"integration p3: selected {len(kept)} / {X_all.shape[1]} features"
    )
    # At least some features must be kept — total pruning would mean nothing
    # is predictive, which contradicts the accuracy check above.
    assert len(kept) >= 1

    # ------------------------------------------------------------------
    # 6. Bet-sizing cascade on the meta-labeler's predictions.
    # ------------------------------------------------------------------
    cascade = BetSizingCascade(
        config=CascadeConfig(
            max_raw_size=1.0, kelly_fraction=0.25,
            max_single_position=0.10, max_family_allocation=0.30,
        ),
        family_stats={
            "ts_momentum":    FamilyStats(avg_win=0.02, avg_loss=0.015),
            "mean_reversion": FamilyStats(avg_win=0.012, avg_loss=0.014),
        },
    )

    # Build a minimal per-row signals frame from what the pipeline recorded
    # (the MetaLabelingPipeline stuffs "signal_side" and "signal_family_*"
    # one-hots into X).
    y_pred_proba = model.predict_proba(X_all)
    sides = X_all["signal_side"].astype(int).to_numpy()
    families = []
    fam_cols = [c for c in X_all.columns if c.startswith("signal_family_")]
    if fam_cols:
        for i in range(len(X_all)):
            active = X_all.iloc[i][fam_cols]
            winners = active[active > 0.5].index.tolist()
            families.append(
                winners[0].replace("signal_family_", "")
                if winners else "unknown"
            )
    else:
        families = ["unknown"] * len(X_all)

    # Apply the cascade to each prediction.
    final_sizes: list[float] = []
    rule_tags: list[list[str]] = []
    for i in range(len(X_all)):
        result = cascade.compute_position_size(
            prob=float(y_pred_proba[i]),
            side=int(sides[i]),
            symbol="AGG",
            signal_family=families[i],
            current_vol=0.02, avg_vol=0.02,
            portfolio_nav=1_000_000.0,
            asset_class="equity",
        )
        final_sizes.append(result["final_size"])
        rule_tags.append(result["constraints_applied"])

    final_sizes_arr = np.asarray(final_sizes, dtype=float)
    # Magnitude bounded by max_single_position; sign matches side.
    assert np.all(np.abs(final_sizes_arr) <= 0.10 + 1e-9), (
        "some bet sizes exceed the 10 % single-position cap"
    )
    signed_match = np.where(
        final_sizes_arr == 0, True,
        np.sign(final_sizes_arr) == np.sign(sides),
    )
    assert signed_match.all(), "some bet sizes have a sign that disagrees with side"

    logger.info(
        "integration p3: cascade ran on "
        f"{len(final_sizes_arr)} predictions, "
        f"{(final_sizes_arr != 0).sum()} non-zero, "
        f"max |size|={float(np.abs(final_sizes_arr).max()):.3f}"
    )

    # ------------------------------------------------------------------
    # 7. Wall-clock budget.
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    logger.info(f"integration p3: elapsed = {elapsed:.2f}s")
    assert elapsed < 120.0, f"phase-3 pipeline too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-o", "addopts="])
