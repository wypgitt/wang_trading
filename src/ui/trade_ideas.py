"""Read-only trade idea generation for the local operator UI.

The service intentionally reuses the live bootstrap in paper-rehearsal mode
and stops before order routing. It computes the same upstream artifacts the
live cycle would need - bars, features, signals, meta probabilities, bet sizes,
and target weights - then turns them into operator-readable trade ideas.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TradeIdea:
    symbol: str
    action: str
    target_weight: float
    target_notional: float
    estimated_quantity: float | None
    latest_price: float | None
    latest_bar_at: str | None
    bars_loaded: int
    feature_rows: int
    signal_count: int
    top_signal_family: str | None
    top_signal_side: int | None
    top_signal_confidence: float | None
    avg_signal_confidence: float | None
    meta_probability: float | None
    calibrated_probability: float | None
    bet_size: float | None
    strategy: str | None
    reason: str
    stage_latency_seconds: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "target_weight": self.target_weight,
            "target_notional": self.target_notional,
            "estimated_quantity": self.estimated_quantity,
            "latest_price": self.latest_price,
            "latest_bar_at": self.latest_bar_at,
            "bars_loaded": self.bars_loaded,
            "feature_rows": self.feature_rows,
            "signal_count": self.signal_count,
            "top_signal_family": self.top_signal_family,
            "top_signal_side": self.top_signal_side,
            "top_signal_confidence": self.top_signal_confidence,
            "avg_signal_confidence": self.avg_signal_confidence,
            "meta_probability": self.meta_probability,
            "calibrated_probability": self.calibrated_probability,
            "bet_size": self.bet_size,
            "strategy": self.strategy,
            "reason": self.reason,
            "stage_latency_seconds": dict(self.stage_latency_seconds),
            "errors": list(self.errors),
        }


@dataclass
class TradeIdeaReport:
    generated_at: str
    mode: str
    config_path: str | None
    symbols: list[str]
    bar_type: str
    nav: float
    model_source: str
    live_orders_sent: int
    allow_confidence_fallback: bool
    ideas: list[TradeIdea]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stage_latency_seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        ideas = [idea.to_dict() for idea in self.ideas]
        gross = sum(abs(float(row["target_weight"])) for row in ideas)
        net = sum(float(row["target_weight"]) for row in ideas)
        return {
            "generated_at": self.generated_at,
            "mode": self.mode,
            "config_path": self.config_path,
            "symbols": list(self.symbols),
            "bar_type": self.bar_type,
            "nav": self.nav,
            "model_source": self.model_source,
            "live_orders_sent": self.live_orders_sent,
            "allow_confidence_fallback": self.allow_confidence_fallback,
            "idea_count": len(ideas),
            "totals": {
                "buy": sum(1 for row in ideas if row["action"] == "BUY"),
                "sell": sum(1 for row in ideas if row["action"] == "SELL"),
                "watch": sum(1 for row in ideas if row["action"] == "WATCH"),
                "model_required": sum(
                    1 for row in ideas if row["action"] == "MODEL_REQUIRED"
                ),
                "error": sum(1 for row in ideas if row["action"] == "ERROR"),
                "gross_target_weight": gross,
                "net_target_weight": net,
            },
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "stage_latency_seconds": dict(self.stage_latency_seconds),
            "ideas": ideas,
        }


async def generate_trade_idea_report(
    *,
    config_path: str | Path | None = None,
    symbols: list[str] | None = None,
    bar_limit: int = 500,
    min_abs_weight: float = 0.0025,
    allow_confidence_fallback: bool = False,
) -> TradeIdeaReport:
    """Generate a read-only trade idea report from the live stack.

    The live bootstrap is always invoked with ``paper_rehearsal=True``. This
    preserves production wiring while forcing paper/dry-run broker selection.
    No call in this module invokes ``OrderManager.run_cycle`` or broker submit.
    """

    from src.bootstrap import (
        ConfidenceMetaPipeline,
        build_live_trading_pipeline,
    )

    started = time.perf_counter()
    ctx = build_live_trading_pipeline(config_path, paper_rehearsal=True)
    pipeline = ctx.pipeline
    warnings: list[str] = []
    errors: list[str] = []

    if pipeline.meta_pipeline is None and allow_confidence_fallback:
        pipeline.meta_pipeline = ConfidenceMetaPipeline()
        warnings.append(
            "No production MLflow model was loaded; using confidence fallback for paper-only ideas."
        )

    model_source = _model_source(pipeline.meta_pipeline)
    if pipeline.meta_pipeline is None:
        warnings.append(
            "No production MLflow model is loaded; exact live target generation will stop after signals."
        )

    selected_symbols = [str(s).strip().upper() for s in (symbols or ctx.symbols) if str(s).strip()]
    if not selected_symbols:
        raise RuntimeError("trade idea report requires at least one symbol")

    adapter = getattr(pipeline, "data_adapter", None)
    bar_type = str(getattr(adapter, "bar_type", ""))
    db = getattr(pipeline, "db_manager", None)
    portfolio = getattr(getattr(pipeline, "order_manager", None), "portfolio", None)
    nav = float(getattr(portfolio, "nav", 0.0) or 0.0)
    generated_at = datetime.now(timezone.utc).isoformat()

    ideas: list[TradeIdea] = []
    try:
        if db is None:
            raise RuntimeError("live pipeline has no database manager")
        for symbol in selected_symbols:
            ideas.append(
                await _generate_symbol_idea(
                    pipeline=pipeline,
                    db=db,
                    symbol=symbol,
                    bar_type=bar_type,
                    bar_limit=int(bar_limit),
                    nav=nav,
                    min_abs_weight=float(min_abs_weight),
                )
            )
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()

    ideas.sort(key=_idea_sort_key)
    stage_latency = _sum_stage_latencies(ideas)
    stage_latency["bootstrap_and_total"] = time.perf_counter() - started

    mode = "paper_production_readonly"
    if model_source == "confidence_fallback":
        mode = "paper_confidence_fallback"
    return TradeIdeaReport(
        generated_at=generated_at,
        mode=mode,
        config_path=str(config_path) if config_path is not None else None,
        symbols=selected_symbols,
        bar_type=bar_type,
        nav=nav,
        model_source=model_source,
        live_orders_sent=0,
        allow_confidence_fallback=allow_confidence_fallback,
        ideas=ideas,
        warnings=warnings,
        errors=errors,
        stage_latency_seconds=stage_latency,
    )


def generate_trade_idea_report_sync(**kwargs: Any) -> TradeIdeaReport:
    return asyncio.run(generate_trade_idea_report(**kwargs))


def build_report_from_cycle(
    *,
    symbols: list[str],
    bars: dict[str, Any] | None,
    features: pd.DataFrame | None,
    signals: pd.DataFrame | None,
    meta: pd.DataFrame | None,
    bets: pd.DataFrame | None,
    target: pd.DataFrame | None,
    nav: float,
    bar_type: str,
    meta_pipeline: Any = None,
    config_path: str | None = None,
    min_abs_weight: float = 0.0025,
    stage_latency_seconds: dict[str, float] | None = None,
) -> TradeIdeaReport:
    """Assemble a ``TradeIdeaReport`` from a LIVE cycle's already-computed panel
    artifacts — the post-cycle publish hook (``PaperTradingPipeline.run_cycle``).

    Reuses :func:`_idea_from_artifacts` (the same per-symbol assembly the
    read-only rehearsal uses), so the published snapshot matches what the
    rehearsal would produce WITHOUT re-running the pipeline — trade ideas are
    computed exactly once, in the live cycle. ``bars`` here is the cycle's
    ``{symbol: latest Bar}`` map (one bar/symbol), so ``bars_loaded`` reflects
    the live streaming tick rather than a backfill of history.
    """

    model_source = _model_source(meta_pipeline)
    feat = features if features is not None else pd.DataFrame()
    sig = signals if signals is not None else pd.DataFrame()
    met = meta if meta is not None else pd.DataFrame()
    bet = bets if bets is not None else pd.DataFrame()
    tgt = target if target is not None else pd.DataFrame()
    latencies = dict(stage_latency_seconds or {})
    clean_symbols = [str(s).strip().upper() for s in (symbols or []) if str(s).strip()]

    ideas: list[TradeIdea] = []
    for symbol in clean_symbols:
        try:
            ideas.append(
                _idea_from_artifacts(
                    symbol=symbol,
                    bars=_bars_df_from_cycle(bars, symbol),
                    features=feat,
                    signals=sig,
                    meta=met,
                    bets=bet,
                    target=tgt,
                    nav=float(nav),
                    min_abs_weight=float(min_abs_weight),
                    model_source=model_source,
                    stage_latency_seconds=latencies,
                    errors=[],
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not drop the rest
            ideas.append(
                _empty_idea(
                    symbol=symbol,
                    action="ERROR",
                    nav=float(nav),
                    reason=f"Idea assembly failed: {exc}",
                    stage_latency_seconds=latencies,
                    errors=[str(exc)],
                )
            )

    ideas.sort(key=_idea_sort_key)
    mode = (
        "paper_confidence_fallback"
        if model_source == "confidence_fallback"
        else "paper_production_live"
    )
    return TradeIdeaReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        config_path=str(config_path) if config_path is not None else None,
        symbols=clean_symbols,
        bar_type=str(bar_type),
        nav=float(nav),
        model_source=model_source,
        live_orders_sent=0,
        allow_confidence_fallback=False,
        ideas=ideas,
        warnings=[],
        errors=[],
        stage_latency_seconds=latencies,
    )


def _bars_df_from_cycle(bars: dict[str, Any] | None, symbol: str) -> pd.DataFrame:
    """One-row bars frame (``close`` column + timestamp index) from the cycle's
    latest ``Bar`` for ``symbol`` — matches what ``_latest_price`` /
    ``_latest_bar_at`` read."""

    if not bars:
        return pd.DataFrame()
    bar = bars.get(symbol) or bars.get(symbol.upper()) or bars.get(symbol.lower())
    if bar is None:
        return pd.DataFrame()
    close = _optional_float(getattr(bar, "close", None))
    if close is None:
        return pd.DataFrame()
    ts = getattr(bar, "timestamp", None) or getattr(bar, "open_time", None)
    index = None
    if ts is not None:
        try:
            index = pd.DatetimeIndex([pd.Timestamp(ts)])
        except Exception:  # noqa: BLE001
            index = None
    return pd.DataFrame({"close": [close]}, index=index)


async def _generate_symbol_idea(
    *,
    pipeline: Any,
    db: Any,
    symbol: str,
    bar_type: str,
    bar_limit: int,
    nav: float,
    min_abs_weight: float,
) -> TradeIdea:
    latencies: dict[str, float] = {}
    errors: list[str] = []

    try:
        started = time.perf_counter()
        bars = await asyncio.to_thread(db.get_bars, symbol, bar_type, None, None, bar_limit)
        latencies["data_fetch"] = time.perf_counter() - started
        bars = _normalize_bars(bars)
    except Exception as exc:  # noqa: BLE001
        return _empty_idea(
            symbol=symbol,
            action="ERROR",
            nav=nav,
            reason=f"Data fetch failed: {exc}",
            stage_latency_seconds=latencies,
            errors=[str(exc)],
        )

    if bars.empty:
        return _empty_idea(
            symbol=symbol,
            action="NO_DATA",
            nav=nav,
            reason="No bars are available for the configured bar type.",
            stage_latency_seconds=latencies,
        )

    features = pd.DataFrame()
    signals = pd.DataFrame()
    meta = pd.DataFrame()
    bets = pd.DataFrame()
    target = pd.DataFrame()

    try:
        if getattr(pipeline, "feature_assembler", None) is None:
            raise RuntimeError("feature assembler is not wired")
        started = time.perf_counter()
        features = pipeline.feature_assembler.compute(bars)
        latencies["feature_compute"] = time.perf_counter() - started
        if features is None or features.empty:
            return _idea_from_artifacts(
                symbol=symbol,
                bars=bars,
                features=pd.DataFrame(),
                signals=pd.DataFrame(),
                meta=pd.DataFrame(),
                bets=pd.DataFrame(),
                target=pd.DataFrame(),
                nav=nav,
                min_abs_weight=min_abs_weight,
                model_source=_model_source(getattr(pipeline, "meta_pipeline", None)),
                stage_latency_seconds=latencies,
                errors=errors,
            )

        if getattr(pipeline, "signal_battery", None) is None:
            raise RuntimeError("signal battery is not wired")
        started = time.perf_counter()
        signals = _generate_signals(pipeline.signal_battery, bars, symbol)
        signals = _current_signal_slice(signals)
        latencies["signal_generation"] = time.perf_counter() - started
        if signals is None or signals.empty:
            return _idea_from_artifacts(
                symbol=symbol,
                bars=bars,
                features=features,
                signals=pd.DataFrame(),
                meta=pd.DataFrame(),
                bets=pd.DataFrame(),
                target=pd.DataFrame(),
                nav=nav,
                min_abs_weight=min_abs_weight,
                model_source=_model_source(getattr(pipeline, "meta_pipeline", None)),
                stage_latency_seconds=latencies,
                errors=errors,
            )

        if getattr(pipeline, "meta_pipeline", None) is None:
            return _idea_from_artifacts(
                symbol=symbol,
                bars=bars,
                features=features,
                signals=signals,
                meta=pd.DataFrame(),
                bets=pd.DataFrame(),
                target=pd.DataFrame(),
                nav=nav,
                min_abs_weight=min_abs_weight,
                model_source="none",
                stage_latency_seconds=latencies,
                errors=errors,
            )

        started = time.perf_counter()
        meta = pipeline.meta_pipeline.predict(features, signals)
        latencies["meta_inference"] = time.perf_counter() - started
        if meta is None or meta.empty:
            return _idea_from_artifacts(
                symbol=symbol,
                bars=bars,
                features=features,
                signals=signals,
                meta=pd.DataFrame(),
                bets=pd.DataFrame(),
                target=pd.DataFrame(),
                nav=nav,
                min_abs_weight=min_abs_weight,
                model_source=_model_source(getattr(pipeline, "meta_pipeline", None)),
                stage_latency_seconds=latencies,
                errors=errors,
            )

        if getattr(pipeline, "bet_sizing", None) is None:
            raise RuntimeError("bet sizing is not wired")
        started = time.perf_counter()
        bets = pipeline.bet_sizing.compute(meta, features)
        latencies["sizing"] = time.perf_counter() - started

        if getattr(pipeline, "portfolio_optimizer", None) is None:
            raise RuntimeError("portfolio optimizer is not wired")
        started = time.perf_counter()
        target = pipeline.portfolio_optimizer.compute_target_portfolio(
            strategy_returns={},
            current_signals=signals,
            bet_sizes=bets if bets is not None else pd.DataFrame(),
            regime=getattr(getattr(pipeline, "config", None), "regime", None),
            nav=nav,
        )
        latencies["target_generation"] = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    return _idea_from_artifacts(
        symbol=symbol,
        bars=bars,
        features=features,
        signals=signals,
        meta=meta,
        bets=bets,
        target=target,
        nav=nav,
        min_abs_weight=min_abs_weight,
        model_source=_model_source(getattr(pipeline, "meta_pipeline", None)),
        stage_latency_seconds=latencies,
        errors=errors,
    )


def _idea_from_artifacts(
    *,
    symbol: str,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.DataFrame,
    meta: pd.DataFrame,
    bets: pd.DataFrame,
    target: pd.DataFrame,
    nav: float,
    min_abs_weight: float,
    model_source: str,
    stage_latency_seconds: dict[str, float] | None = None,
    errors: list[str] | None = None,
) -> TradeIdea:
    errors = list(errors or [])
    latest_price = _latest_price(bars)
    latest_bar_at = _latest_bar_at(bars)
    signal_rows = _rows_for_symbol(signals, symbol)
    meta_rows = _rows_for_symbol(meta, symbol)
    bet_rows = _rows_for_symbol(bets, symbol)
    target_rows = _rows_for_symbol(target, symbol)

    signal_count = int(len(signal_rows))
    top_signal = _top_signal(signal_rows)
    meta_row = _top_meta(meta_rows)

    target_weight = _target_weight(target_rows)
    target_notional = target_weight * nav
    estimated_quantity = (
        target_notional / latest_price
        if latest_price is not None and latest_price > 0
        else None
    )
    action = _action_for(
        target_weight=target_weight,
        min_abs_weight=min_abs_weight,
        latest_price=latest_price,
        model_source=model_source,
        signal_count=signal_count,
        errors=errors,
    )

    top_confidence = _optional_float(top_signal.get("confidence") if top_signal else None)
    avg_confidence = _avg_signal_confidence(signal_rows)
    meta_probability = _optional_float(meta_row.get("meta_prob") if meta_row else None)
    calibrated_probability = _optional_float(
        meta_row.get("calibrated_prob") if meta_row else None
    )
    bet_size = _sum_numeric_column(
        bet_rows,
        "final_size" if "final_size" in bet_rows.columns else "size",
    )
    strategy = _last_string(target_rows, "strategy") or _last_string(bet_rows, "family")

    top_family = _optional_str(top_signal.get("family") if top_signal else None)
    return TradeIdea(
        symbol=symbol,
        action=action,
        target_weight=target_weight,
        target_notional=target_notional,
        estimated_quantity=estimated_quantity,
        latest_price=latest_price,
        latest_bar_at=latest_bar_at,
        bars_loaded=int(len(bars)),
        feature_rows=int(len(features)) if features is not None else 0,
        signal_count=signal_count,
        top_signal_family=top_family,
        top_signal_side=_optional_int(top_signal.get("side") if top_signal else None),
        top_signal_confidence=top_confidence,
        avg_signal_confidence=avg_confidence,
        meta_probability=meta_probability,
        calibrated_probability=calibrated_probability,
        bet_size=bet_size,
        strategy=strategy,
        reason=_reason(
            action=action,
            target_weight=target_weight,
            strategy=strategy,
            top_signal=top_signal,
            top_confidence=top_confidence,
            meta_probability=meta_probability,
            bars=bars,
            features=features,
            signals=signals,
            model_source=model_source,
            errors=errors,
        ),
        stage_latency_seconds=dict(stage_latency_seconds or {}),
        errors=errors,
    )


def _generate_signals(signal_battery: Any, bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    for name in ("generate", "generate_all"):
        fn = getattr(signal_battery, name, None)
        if not callable(fn):
            continue
        try:
            return fn(bars, symbol=symbol)
        except TypeError:
            try:
                return fn(bars)
            except TypeError:
                continue
    return pd.DataFrame()


def _current_signal_slice(signals: pd.DataFrame | None) -> pd.DataFrame:
    """Keep only the latest signal timestamp for current-decision displays."""

    if signals is None or signals.empty or "timestamp" not in signals.columns:
        return pd.DataFrame() if signals is None else signals
    work = signals.copy()
    timestamps = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().all():
        return work
    latest = timestamps.max()
    return work.loc[timestamps == latest].copy()


def _normalize_bars(bars: pd.DataFrame | None) -> pd.DataFrame:
    if bars is None:
        return pd.DataFrame()
    frame = bars.copy()
    if frame.empty:
        return frame
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("bars must have a timestamp column or DatetimeIndex")
    return frame.sort_index()


def _empty_idea(
    *,
    symbol: str,
    action: str,
    nav: float,
    reason: str,
    stage_latency_seconds: dict[str, float] | None = None,
    errors: list[str] | None = None,
) -> TradeIdea:
    return TradeIdea(
        symbol=symbol,
        action=action,
        target_weight=0.0,
        target_notional=0.0 * nav,
        estimated_quantity=None,
        latest_price=None,
        latest_bar_at=None,
        bars_loaded=0,
        feature_rows=0,
        signal_count=0,
        top_signal_family=None,
        top_signal_side=None,
        top_signal_confidence=None,
        avg_signal_confidence=None,
        meta_probability=None,
        calibrated_probability=None,
        bet_size=None,
        strategy=None,
        reason=reason,
        stage_latency_seconds=dict(stage_latency_seconds or {}),
        errors=list(errors or []),
    )


def _model_source(meta_pipeline: Any) -> str:
    if meta_pipeline is None:
        return "none"
    name = meta_pipeline.__class__.__name__
    if name == "ModelMetaPipeline":
        return "mlflow_production"
    if name == "ConfidenceMetaPipeline":
        return "confidence_fallback"
    return name


def _action_for(
    *,
    target_weight: float,
    min_abs_weight: float,
    latest_price: float | None,
    model_source: str,
    signal_count: int,
    errors: list[str],
) -> str:
    if errors:
        return "ERROR"
    if latest_price is None:
        return "NO_DATA"
    if abs(target_weight) >= min_abs_weight:
        return "BUY" if target_weight > 0 else "SELL"
    if model_source == "none" and signal_count > 0:
        return "MODEL_REQUIRED"
    return "WATCH"


def _reason(
    *,
    action: str,
    target_weight: float,
    strategy: str | None,
    top_signal: dict[str, Any] | None,
    top_confidence: float | None,
    meta_probability: float | None,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.DataFrame,
    model_source: str,
    errors: list[str],
) -> str:
    if errors:
        return f"Pipeline stage failed: {errors[-1]}"
    if bars is None or bars.empty:
        return "No recent bars were returned."
    if features is None or features.empty:
        return "Feature assembler returned no rows for the latest window."
    if signals is None or signals.empty:
        return "No signal family fired on the latest window."
    if model_source == "none":
        return "Production meta model is unavailable; target generation skipped."
    if action == "WATCH":
        return "Computed target is below the display threshold."
    family = _optional_str(top_signal.get("family") if top_signal else None) or "signal"
    side = _side_label(_optional_int(top_signal.get("side") if top_signal else None))
    confidence = "" if top_confidence is None else f", confidence {top_confidence:.1%}"
    prob = "" if meta_probability is None else f", meta {meta_probability:.1%}"
    strategy_text = strategy or family
    return (
        f"{side} target {target_weight:.2%} from {strategy_text}; "
        f"top signal {family}{confidence}{prob}."
    )


def _rows_for_symbol(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "symbol" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["symbol"].astype(str).str.upper() == symbol.upper()].copy()


def _latest_price(bars: pd.DataFrame) -> float | None:
    if bars is None or bars.empty or "close" not in bars.columns:
        return None
    return _optional_float(bars["close"].iloc[-1])


def _latest_bar_at(bars: pd.DataFrame) -> str | None:
    if bars is None or bars.empty:
        return None
    try:
        return pd.Timestamp(bars.index[-1]).isoformat()
    except Exception:  # noqa: BLE001
        return None


def _target_weight(target_rows: pd.DataFrame) -> float:
    if target_rows is None or target_rows.empty or "target_weight" not in target_rows.columns:
        return 0.0
    return _finite_float(target_rows["target_weight"].sum(), 0.0)


def _top_signal(signals: pd.DataFrame) -> dict[str, Any] | None:
    if signals is None or signals.empty:
        return None
    work = signals.copy()
    if "timestamp" in work.columns:
        work["_timestamp_sort"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    else:
        work["_timestamp_sort"] = pd.Timestamp("1970-01-01", tz="UTC")
    if "confidence" in work.columns:
        work["_confidence_sort"] = pd.to_numeric(work["confidence"], errors="coerce").fillna(0.0)
    else:
        work["_confidence_sort"] = 0.0
    work = work.sort_values(["_timestamp_sort", "_confidence_sort"])
    row = work.iloc[-1].drop(labels=["_timestamp_sort", "_confidence_sort"], errors="ignore")
    return row.to_dict()


def _top_meta(meta: pd.DataFrame) -> dict[str, Any] | None:
    if meta is None or meta.empty:
        return None
    work = meta.copy()
    if "timestamp" in work.columns:
        work["_timestamp_sort"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    else:
        work["_timestamp_sort"] = pd.Timestamp("1970-01-01", tz="UTC")
    score_col = "calibrated_prob" if "calibrated_prob" in work.columns else "meta_prob"
    if score_col in work.columns:
        work["_prob_sort"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    else:
        work["_prob_sort"] = 0.0
    work = work.sort_values(["_timestamp_sort", "_prob_sort"])
    row = work.iloc[-1].drop(labels=["_timestamp_sort", "_prob_sort"], errors="ignore")
    return row.to_dict()


def _avg_signal_confidence(signals: pd.DataFrame) -> float | None:
    if signals is None or signals.empty or "confidence" not in signals.columns:
        return None
    return _optional_float(pd.to_numeric(signals["confidence"], errors="coerce").mean())


def _sum_numeric_column(frame: pd.DataFrame, column: str) -> float | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    return _optional_float(pd.to_numeric(frame[column], errors="coerce").sum())


def _last_string(frame: pd.DataFrame, column: str) -> str | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return str(values.iloc[-1])


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    return out if math.isfinite(out) else None


def _finite_float(value: Any, default: float) -> float:
    out = _optional_float(value)
    return default if out is None else out


def _optional_int(value: Any) -> int | None:
    out = _optional_float(value)
    return None if out is None else int(out)


def _side_label(side: int | None) -> str:
    if side is None:
        return "Neutral"
    if side > 0:
        return "Long"
    if side < 0:
        return "Short"
    return "Neutral"


def _sum_stage_latencies(ideas: list[TradeIdea]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for idea in ideas:
        for stage, seconds in idea.stage_latency_seconds.items():
            totals[stage] = totals.get(stage, 0.0) + float(seconds)
    return totals


def _idea_sort_key(idea: TradeIdea) -> tuple[int, float, str]:
    priority = {
        "BUY": 0,
        "SELL": 0,
        "MODEL_REQUIRED": 1,
        "WATCH": 2,
        "NO_DATA": 3,
        "ERROR": 4,
    }.get(idea.action, 5)
    return priority, -abs(idea.target_weight), idea.symbol
