"""
Meta-labeler retraining orchestrator (design-doc §13 Phase 6: weekly retrain).

Production entry point that stitches the Phase-3 pipeline together in one
command:

    python scripts/retrain_model.py --symbol AAPL --tune --n-trials 50
    python scripts/retrain_model.py --symbol AAPL --use-best-params
    python scripts/retrain_model.py --all-symbols --tune --timeout 3600

The script is responsible for:

    1. Loading the latest bars from TimescaleDB.
    2. Running FeatureAssembler + CUSUM event filter + SignalBattery.
    3. Building (X, y, sample_weights) via MetaLabelingPipeline.
    4. Optuna tuning (when ``--tune``) or loading saved best params.
    5. Training the MetaLabeler with purged-CV early stopping + isotonic
       calibration on out-of-fold predictions.
    6. Computing MDI / MDA / SFI feature importances.
    7. Comparing against the incumbent production model.
    8. Logging everything to MLflow and conditionally promoting.

Promotion gate for Phase 3 is a simple mean-CV-accuracy improvement over
the incumbent. Phase 4 replaces this with the deflated-Sharpe three-gate
check (CPCV 60 % positive paths, DSR p<0.05, PBO<40 %); that replacement
lives behind the ``_promotion_gate_phase4_stub`` hook so we just need to
swap the body when Phase 4 lands.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from loguru import logger


# Make ``src`` importable when invoked as a standalone script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.bet_sizing.cascade import BetSizingCascade, CascadeConfig, FamilyStats  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.data_engine.bars.cusum_filter import (  # noqa: E402
    compute_cusum_threshold,
    cusum_filter,
)
from src.data_engine.storage.database import DatabaseManager  # noqa: E402
from src.feature_factory.assembler import FeatureAssembler  # noqa: E402
from src.labeling.meta_labeler_pipeline import MetaLabelingPipeline  # noqa: E402
from src.ml_layer.feature_importance import (  # noqa: E402
    mda_importance,
    mdi_importance,
    select_features,
    sfi_importance,
)
from src.ml_layer.meta_labeler import MetaLabeler  # noqa: E402
from src.ml_layer.model_registry import ModelRegistry  # noqa: E402
from src.ml_layer.purged_cv import cross_val_score_purged  # noqa: E402
from src.ml_layer.tuning import (  # noqa: E402
    retrain_with_best_params,
    tune_meta_labeler,
)
from src.signal_battery.mean_reversion import MeanReversionSignal  # noqa: E402
from src.signal_battery.momentum import TimeSeriesMomentumSignal  # noqa: E402
from src.signal_battery.orchestrator import SignalBattery  # noqa: E402
from src.signal_battery.trend_following import (  # noqa: E402
    DonchianBreakoutSignal,
    MovingAverageCrossoverSignal,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BEST_PARAMS_DIR = _ROOT / "data" / "best_params"
_TRACKING_URI_DEFAULT = f"sqlite:///{_ROOT / 'mlflow.db'}"


def _best_params_path(symbol: str, model_type: str) -> Path:
    return _BEST_PARAMS_DIR / f"{symbol}_{model_type}.json"


def _save_best_params(symbol: str, model_type: str, params: dict[str, Any]) -> Path:
    _BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = _best_params_path(symbol, model_type)
    path.write_text(json.dumps(params, indent=2, sort_keys=True, default=str))
    return path


def _load_best_params(symbol: str, model_type: str) -> dict[str, Any] | None:
    path = _best_params_path(symbol, model_type)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning(f"best-params file {path} unreadable: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_bars(
    db: DatabaseManager,
    symbol: str,
    bar_type: str,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> pd.DataFrame:
    """Return a bars DataFrame indexed by ``timestamp`` or empty on miss."""
    df = db.get_bars(
        symbol=symbol, bar_type=bar_type, start=start, end=end, limit=limit,
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # Columns the FeatureAssembler requires — the DB schema already carries
    # all of them; this guard just fails loudly if someone changes it.
    required = (
        "close", "volume", "dollar_volume", "buy_volume", "sell_volume",
        "tick_count", "bar_duration_seconds",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"bars table missing expected columns {missing} for {symbol}"
        )
    return df


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def _default_signal_battery() -> SignalBattery:
    """Register the set of single-symbol signal families that make sense for a
    per-symbol retrain. Cross-sectional / pair / carry families are
    deliberately out-of-scope here (they require the panel of instruments
    the orchestrator module wires up)."""
    battery = SignalBattery()
    battery.register(TimeSeriesMomentumSignal(), kind="bars")
    battery.register(MeanReversionSignal(), kind="bars")
    battery.register(MovingAverageCrossoverSignal(), kind="bars")
    battery.register(DonchianBreakoutSignal(), kind="bars")
    return battery


def _build_training_frame(
    bars: pd.DataFrame,
    symbol: str,
    cusum_multiplier: float,
    max_holding_period: int,
    vol_span: int,
    time_decay: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame] | None:
    """
    Run feature assembly + CUSUM + signals + meta-labeling; return
    ``(X, y, sample_weights, labels_df)`` or None on degenerate input.
    """
    assembler = FeatureAssembler()
    features = assembler.assemble(bars)
    if features.empty:
        logger.warning(f"{symbol}: feature matrix is empty; skipping")
        return None

    close = bars["close"].loc[features.index]
    threshold = compute_cusum_threshold(close, multiplier=cusum_multiplier)
    events = cusum_filter(close, threshold=threshold)
    events = events.intersection(features.index)
    if len(events) < 50:
        logger.warning(
            f"{symbol}: only {len(events)} CUSUM events; skipping retrain"
        )
        return None

    battery = _default_signal_battery()
    signals = battery.generate_all(
        bars.loc[features.index], event_timestamps=events, symbol=symbol,
    )
    directional = signals[signals["side"] != 0]
    if directional.empty:
        logger.warning(f"{symbol}: signal battery produced no directional signals")
        return None

    pipeline = MetaLabelingPipeline(
        max_holding_period=max_holding_period,
        vol_span=vol_span,
        time_decay=time_decay,
    )
    X, y, w = pipeline.prepare_training_data(close, directional, features)
    if len(X) == 0:
        logger.warning(f"{symbol}: meta-labeling pipeline produced 0 rows")
        return None

    # Build labels_df expected by PurgedKFoldCV. We don't have the triple-
    # barrier exit timestamps here (the pipeline discarded them), so use
    # ``event_start + max_holding_period`` bars as a conservative proxy.
    event_starts = pd.DatetimeIndex(X.index)
    # Spacing of the feature index is the native bar frequency.
    bar_freq = _infer_freq_seconds(features.index)
    delta = pd.Timedelta(seconds=bar_freq * max_holding_period)
    event_ends = event_starts + delta
    labels_df = pd.DataFrame(
        {"event_start": event_starts, "event_end": event_ends},
        index=event_starts,
    )
    return X, y, w, labels_df


def _infer_freq_seconds(index: pd.DatetimeIndex) -> float:
    """Median gap between consecutive bars, in seconds."""
    if len(index) < 2:
        return 86400.0
    diffs = np.diff(index.to_numpy()).astype("timedelta64[s]").astype(float)
    return float(max(np.median(diffs), 1.0))


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def _train_model(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    labels_df: pd.DataFrame,
    *,
    model_type: str,
    params: dict[str, Any] | None,
    calibrate: bool,
) -> MetaLabeler:
    """Train via the tuner helper when params are known; plain fit otherwise."""
    if params:
        return retrain_with_best_params(
            best_params=params,
            X=X, y=y, labels_df=labels_df,
            sample_weight=w,
            model_type=model_type,
            calibrate=calibrate,
        )
    model = MetaLabeler(
        model_type=model_type, params=None, calibrate=calibrate,
    )
    return model.fit(X, y, sample_weight=w, labels_df=labels_df)


def _compute_importances(
    model: MetaLabeler,
    X: pd.DataFrame,
    y: pd.Series,
    labels_df: pd.DataFrame,
    *,
    n_splits: int,
    n_repeats: int,
    model_type: str,
) -> dict[str, pd.DataFrame]:
    mdi = mdi_importance(model, list(X.columns)).to_frame("importance")
    mda = mda_importance(
        model, X, y, labels_df,
        n_splits=n_splits, n_repeats=n_repeats, scoring="accuracy",
    )
    sfi = sfi_importance(
        X, y, labels_df,
        model_type=model_type, n_splits=n_splits, scoring="accuracy",
    ).to_frame("score")
    return {"mdi": mdi, "mda": mda, "sfi": sfi}


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

def _promotion_gate_phase4_stub(
    challenger_metrics: dict[str, float],
    incumbent_metrics: dict[str, float] | None,
) -> tuple[bool, str]:
    """
    Decide whether to promote the challenger.

    **Phase 3 implementation** — simple mean-CV improvement over the
    incumbent. Phase 4 replaces this stub with the three-gate CPCV / DSR /
    PBO check (design-doc §9) — keep the call sites stable by swapping
    the body here.
    """
    challenger = challenger_metrics.get("mean_cv_score", float("nan"))
    if not np.isfinite(challenger):
        return False, "challenger has no finite mean_cv_score"

    if incumbent_metrics is None:
        # First model ever — promote.
        return True, "no incumbent; promoting first model"

    incumbent = incumbent_metrics.get("mean_cv_score", float("nan"))
    if not np.isfinite(incumbent):
        return True, "incumbent has no valid metric; promoting"

    # Require a meaningful improvement (1pp) over the incumbent.
    delta = challenger - incumbent
    if delta > 0.01:
        return True, (
            f"promoting: challenger mean_cv={challenger:.4f} vs "
            f"incumbent {incumbent:.4f} (Δ={delta:+.4f})"
        )
    return False, (
        f"keeping incumbent: challenger mean_cv={challenger:.4f} vs "
        f"incumbent {incumbent:.4f} (Δ={delta:+.4f})"
    )


# ---------------------------------------------------------------------------
# Per-symbol retrain driver
# ---------------------------------------------------------------------------

def retrain_symbol(  # noqa: PLR0913 — matches CLI flag layout
    symbol: str,
    *,
    bar_type: str = "tib",
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 100_000,
    model_type: str = "lightgbm",
    tune: bool = False,
    n_trials: int = 50,
    timeout: int = 600,
    use_best_params: bool = False,
    cusum_multiplier: float = 1.5,
    max_holding_period: int = 20,
    vol_span: int = 100,
    time_decay: float = 1.0,
    n_importance_splits: int = 3,
    n_importance_repeats: int = 3,
    tracking_uri: str = _TRACKING_URI_DEFAULT,
    experiment_name: str = "meta-labeler",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Full retrain for a single symbol. Returns a summary dict:
        {"symbol", "status", "run_id" (or None), "promoted", "reason",
         "elapsed_s", "metrics"}
    """
    t_start = time.perf_counter()
    result: dict[str, Any] = {
        "symbol": symbol, "status": "started", "run_id": None,
        "promoted": False, "reason": "", "elapsed_s": 0.0,
        "metrics": {},
    }

    try:
        # ---- 1. Load bars ---------------------------------------------
        settings = get_settings()
        db = DatabaseManager(settings.database.url)
        bars = _load_bars(db, symbol, bar_type, start, end, limit)
        if bars.empty:
            result.update(status="skipped", reason="no bars in database")
            return result
        logger.info(f"{symbol}: loaded {len(bars)} bars from DB")

        # ---- 2. Build training frame -----------------------------------
        frame = _build_training_frame(
            bars, symbol,
            cusum_multiplier=cusum_multiplier,
            max_holding_period=max_holding_period,
            vol_span=vol_span, time_decay=time_decay,
        )
        if frame is None:
            result.update(status="skipped", reason="insufficient data / events")
            return result
        X, y, w, labels_df = frame
        logger.info(
            f"{symbol}: training frame rows={len(X)} cols={X.shape[1]} "
            f"y_mean={y.mean():.3f}"
        )

        # ---- 3. Resolve hyperparameters --------------------------------
        if tune and use_best_params:
            raise click.UsageError("--tune and --use-best-params are mutually exclusive")

        best_params: dict[str, Any] | None = None
        if tune:
            logger.info(f"{symbol}: tuning with Optuna ({n_trials} trials, {timeout}s timeout)")
            best_params = tune_meta_labeler(
                X=X, y=y, labels_df=labels_df,
                sample_weight=w,
                model_type=model_type,
                n_trials=n_trials, timeout=timeout,
                n_splits=5, scoring="neg_log_loss",
            )
            _save_best_params(symbol, model_type, best_params)
            logger.info(f"{symbol}: saved tuned params to {_best_params_path(symbol, model_type)}")
        elif use_best_params:
            best_params = _load_best_params(symbol, model_type)
            if best_params is None:
                logger.warning(
                    f"{symbol}: no saved best params; falling back to defaults"
                )

        # ---- 4. Train --------------------------------------------------
        model = _train_model(
            X, y, w, labels_df,
            model_type=model_type, params=best_params, calibrate=True,
        )

        # ---- 5. Evaluate ----------------------------------------------
        cv_scores = cross_val_score_purged(
            model.model_, X, y, labels_df,
            n_splits=5, embargo_pct=0.01, scoring="accuracy",
        )
        finite_cv = cv_scores[np.isfinite(cv_scores)]
        metrics = {
            "mean_cv_score": float(finite_cv.mean()) if finite_cv.size else float("nan"),
            "std_cv_score": (
                float(finite_cv.std(ddof=1)) if finite_cv.size > 1 else 0.0
            ),
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "y_prevalence": float(y.mean()),
        }
        logger.info(
            f"{symbol}: mean CV accuracy = {metrics['mean_cv_score']:.4f} "
            f"(±{metrics['std_cv_score']:.4f})"
        )

        # ---- 6. Feature importance ------------------------------------
        importances = _compute_importances(
            model, X, y, labels_df,
            n_splits=n_importance_splits,
            n_repeats=n_importance_repeats,
            model_type=model_type,
        )
        kept = select_features(importances)
        logger.info(f"{symbol}: feature selection kept {len(kept)} / {X.shape[1]}")

        # ---- 7. Promotion gate ----------------------------------------
        registry = ModelRegistry(
            tracking_uri=tracking_uri,
            experiment_name=f"{experiment_name}-{symbol}",
        )
        incumbent_metrics: dict[str, float] | None = None
        top = registry.get_best_model(metric="mean_cv_score", n=1)
        if top:
            incumbent_metrics = top[0]["metrics"]

        promote, reason = _promotion_gate_phase4_stub(metrics, incumbent_metrics)
        result["promoted"] = promote
        result["reason"] = reason

        # ---- 8. Log to MLflow -----------------------------------------
        run_id = registry.log_training_run(
            model=model, X=X, y=y, labels_df=labels_df,
            params=best_params or {},
            cv_scores=cv_scores,
            importances=importances,
            sample_weight=w,
        )
        result["run_id"] = run_id
        logger.info(f"{symbol}: logged to MLflow run {run_id}")

        # ---- 9. Conditional promotion ---------------------------------
        if promote and not dry_run:
            try:
                registry.promote_model(run_id=run_id, stage="production")
                logger.info(f"{symbol}: PROMOTED to production ({reason})")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{symbol}: promote_model failed: {exc}")
                result["promoted"] = False
                result["reason"] = f"promote_model errored: {exc}"
        elif promote and dry_run:
            logger.info(f"{symbol}: would promote (dry-run): {reason}")
        else:
            logger.info(f"{symbol}: not promoting ({reason})")

        result["metrics"] = metrics
        result["status"] = "ok"

    except Exception as exc:  # noqa: BLE001 — surface failure per symbol, not across all
        logger.exception(f"{symbol}: retrain failed")
        result["status"] = "error"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    result["elapsed_s"] = float(time.perf_counter() - t_start)
    return result


# ---------------------------------------------------------------------------
# Batch cascade — for the --all-symbols case
# ---------------------------------------------------------------------------

def _iter_all_symbols() -> list[str]:
    """Configured equity test-universe from settings.yaml."""
    settings = get_settings()
    equities = settings.instruments.equities
    symbols = equities.get("test_symbols") or equities.get("custom_symbols") or []
    return list(symbols)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--symbol", default=None, help="Single-symbol retrain.")
@click.option("--all-symbols", "all_symbols", is_flag=True,
              help="Retrain every symbol in the configured universe.")
@click.option("--bar-type", default="tib",
              type=click.Choice(["tick", "volume", "dollar", "tib", "vib", "time"]))
@click.option("--start-date", default=None, help="ISO-8601 start date.")
@click.option("--end-date", default=None, help="ISO-8601 end date.")
@click.option("--limit", default=100_000, show_default=True,
              help="Max bars per symbol.")
@click.option("--model-type", default="lightgbm",
              type=click.Choice(["lightgbm", "xgboost", "random_forest"]))
@click.option("--tune", is_flag=True, help="Run Optuna hyperparameter search.")
@click.option("--use-best-params", is_flag=True,
              help="Load previously-tuned params from data/best_params/.")
@click.option("--n-trials", default=50, show_default=True,
              help="Optuna trial budget (used with --tune).")
@click.option("--timeout", default=600, show_default=True,
              help="Optuna wall-clock cap in seconds (used with --tune).")
@click.option("--cusum-multiplier", default=1.5, show_default=True, type=float)
@click.option("--max-holding-period", default=20, show_default=True, type=int)
@click.option("--vol-span", default=100, show_default=True, type=int)
@click.option("--time-decay", default=1.0, show_default=True, type=float)
@click.option("--tracking-uri", default=_TRACKING_URI_DEFAULT, show_default=True)
@click.option("--experiment-name", default="meta-labeler", show_default=True)
@click.option("--dry-run", is_flag=True,
              help="Log to MLflow but skip the registry promotion.")
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(
    symbol: str | None,
    all_symbols: bool,
    bar_type: str,
    start_date: str | None,
    end_date: str | None,
    limit: int,
    model_type: str,
    tune: bool,
    use_best_params: bool,
    n_trials: int,
    timeout: int,
    cusum_multiplier: float,
    max_holding_period: int,
    vol_span: int,
    time_decay: float,
    tracking_uri: str,
    experiment_name: str,
    dry_run: bool,
    log_level: str,
) -> None:
    """Retrain the meta-labeler for one or more symbols."""
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    if not symbol and not all_symbols:
        raise click.UsageError("specify --symbol <TICKER> or --all-symbols")
    if symbol and all_symbols:
        raise click.UsageError("--symbol and --all-symbols are mutually exclusive")
    if tune and use_best_params:
        raise click.UsageError("--tune and --use-best-params are mutually exclusive")

    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None

    if symbol:
        symbols = [symbol]
    else:
        symbols = _iter_all_symbols()
        if not symbols:
            raise click.UsageError(
                "no symbols found in settings.yaml; configure instruments.equities"
            )
    logger.info(f"retrain: {len(symbols)} symbol(s) -> {symbols}")

    summaries: list[dict[str, Any]] = []
    for sym in symbols:
        summary = retrain_symbol(
            sym,
            bar_type=bar_type,
            start=start, end=end, limit=limit,
            model_type=model_type,
            tune=tune, n_trials=n_trials, timeout=timeout,
            use_best_params=use_best_params,
            cusum_multiplier=cusum_multiplier,
            max_holding_period=max_holding_period,
            vol_span=vol_span, time_decay=time_decay,
            tracking_uri=tracking_uri, experiment_name=experiment_name,
            dry_run=dry_run,
        )
        summaries.append(summary)
        logger.info(
            f"{sym}: status={summary['status']} "
            f"promoted={summary['promoted']} elapsed={summary['elapsed_s']:.1f}s "
            f"reason={summary['reason']!r}"
        )

    # Print a short aggregate table for operators.
    ok = sum(1 for s in summaries if s["status"] == "ok")
    promoted = sum(1 for s in summaries if s["promoted"])
    logger.info(
        f"retrain complete: ok={ok}/{len(summaries)} promoted={promoted} "
        f"total_elapsed={sum(s['elapsed_s'] for s in summaries):.1f}s"
    )

    # Non-zero exit if nothing succeeded (helps cron wrappers flag failures).
    if ok == 0 and summaries:
        sys.exit(1)


def _parse_date(s: str) -> datetime:
    """Lenient ISO-8601 → aware UTC datetime."""
    ts = pd.to_datetime(s, utc=True)
    return ts.to_pydatetime()


if __name__ == "__main__":
    main()
