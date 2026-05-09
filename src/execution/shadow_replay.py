"""Shadow-mode replay for live target generation.

This module reuses the live bootstrap components, forces paper broker wiring,
and walks recent historical bars without sending orders. It is meant for
operator pre-open checks: "would the live stack generate sane targets on the
latest market history?"
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ReplayExpectation:
    max_abs_weight: float = 0.10
    max_gross: float = 1.50
    max_turnover: float | None = None
    max_weight_error: float = 1e-6
    allow_empty_targets: bool = False
    weights: dict[str, dict[str, float]] | None = None


def load_expectation(path: str | Path | None) -> ReplayExpectation:
    if path is None:
        return ReplayExpectation()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReplayExpectation(
        max_abs_weight=float(data.get("max_abs_weight", 0.10)),
        max_gross=float(data.get("max_gross", 1.50)),
        max_turnover=(
            None if data.get("max_turnover") is None
            else float(data.get("max_turnover"))
        ),
        max_weight_error=float(data.get("max_weight_error", 1e-6)),
        allow_empty_targets=bool(data.get("allow_empty_targets", False)),
        weights={
            _timestamp_key(ts): {str(sym): float(w) for sym, w in weights.items()}
            for ts, weights in (data.get("weights") or {}).items()
        },
    )


def target_weights(target: pd.DataFrame | None) -> dict[str, float]:
    if target is None or target.empty:
        return {}
    if "symbol" not in target.columns or "target_weight" not in target.columns:
        return {}
    grouped = target.groupby("symbol")["target_weight"].sum()
    return {str(symbol): float(weight) for symbol, weight in grouped.items()}


def compare_target(
    *,
    timestamp: datetime | pd.Timestamp,
    target: pd.DataFrame | None,
    previous_weights: dict[str, float] | None,
    expectation: ReplayExpectation,
) -> dict[str, Any]:
    weights = target_weights(target)
    gross = sum(abs(w) for w in weights.values())
    max_abs = max((abs(w) for w in weights.values()), default=0.0)
    violations: list[dict[str, Any]] = []

    if not weights and not expectation.allow_empty_targets:
        violations.append({"type": "empty_target", "message": "target is empty"})
    if max_abs > expectation.max_abs_weight + 1e-12:
        violations.append({
            "type": "max_abs_weight",
            "value": max_abs,
            "limit": expectation.max_abs_weight,
        })
    if gross > expectation.max_gross + 1e-12:
        violations.append({
            "type": "max_gross",
            "value": gross,
            "limit": expectation.max_gross,
        })
    if previous_weights is not None and expectation.max_turnover is not None:
        turnover = _turnover(previous_weights, weights)
        if turnover > expectation.max_turnover + 1e-12:
            violations.append({
                "type": "max_turnover",
                "value": turnover,
                "limit": expectation.max_turnover,
            })
    else:
        turnover = None

    ts_key = _timestamp_key(timestamp)
    expected = (expectation.weights or {}).get(ts_key)
    max_error = 0.0
    if expected is not None:
        symbols = set(expected) | set(weights)
        errors = {
            symbol: abs(float(weights.get(symbol, 0.0)) - float(expected.get(symbol, 0.0)))
            for symbol in symbols
        }
        max_error = max(errors.values(), default=0.0)
        if max_error > expectation.max_weight_error:
            violations.append({
                "type": "expected_weight_mismatch",
                "value": max_error,
                "limit": expectation.max_weight_error,
                "errors": errors,
            })

    return {
        "timestamp": ts_key,
        "weights": weights,
        "gross": gross,
        "max_abs_weight": max_abs,
        "turnover": turnover,
        "max_expected_error": max_error,
        "passed": not violations,
        "violations": violations,
    }


async def replay_recent_live_targets(
    *,
    config_path: str | Path | None,
    symbols: list[str] | None = None,
    bar_limit: int = 250,
    replay_points: int = 25,
    expectation: ReplayExpectation | None = None,
) -> dict[str, Any]:
    from src.bootstrap import build_live_trading_pipeline

    expectation = expectation or ReplayExpectation()
    ctx = build_live_trading_pipeline(config_path, paper_rehearsal=True)
    pipeline = ctx.pipeline
    replay_symbols = symbols or ctx.symbols
    if not replay_symbols:
        raise RuntimeError("shadow replay requires at least one symbol")

    bar_type = getattr(getattr(pipeline, "data_adapter", None), "bar_type", "tib")
    db = pipeline.db_manager
    results: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None

    for symbol in replay_symbols:
        bars = await asyncio.to_thread(db.get_bars, symbol, bar_type, None, None, bar_limit)
        if bars.empty:
            results.append({
                "symbol": symbol,
                "passed": False,
                "violations": [{"type": "missing_bars", "message": "no bars found"}],
            })
            continue
        bars = _normalize_bars(bars)
        points = list(bars.index[-max(1, int(replay_points)):])
        for ts in points:
            window = bars.loc[:ts]
            target = _generate_target(pipeline, symbol, window)
            comparison = compare_target(
                timestamp=ts,
                target=target,
                previous_weights=previous,
                expectation=expectation,
            )
            comparison["symbol"] = symbol
            comparison["target_rows"] = int(len(target)) if target is not None else 0
            results.append(comparison)
            previous = comparison["weights"]

    violations = [
        violation
        for result in results
        for violation in result.get("violations", [])
    ]
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow_replay",
        "symbols": replay_symbols,
        "bar_limit": int(bar_limit),
        "replay_points": int(replay_points),
        "passed": not violations,
        "cycles": len(results),
        "violations": violations,
        "results": results,
    }


def format_shadow_replay_report(result: dict[str, Any]) -> str:
    lines = [
        f"# Shadow Replay Report - {result.get('as_of')}",
        "",
        f"Status: {'PASS' if result.get('passed') else 'FAIL'}",
        f"Symbols: {', '.join(result.get('symbols') or [])}",
        f"Replay cycles: {result.get('cycles', 0)}",
        f"Violations: {len(result.get('violations') or [])}",
        "",
        "## Recent Targets",
    ]
    for row in (result.get("results") or [])[-10:]:
        status = "PASS" if row.get("passed") else "FAIL"
        weights = ", ".join(
            f"{symbol}={weight:.2%}"
            for symbol, weight in sorted((row.get("weights") or {}).items())
        ) or "(empty)"
        lines.append(
            f"- {row.get('timestamp')} {row.get('symbol', '')}: "
            f"{status} gross={row.get('gross', 0.0):.2%} weights: {weights}"
        )
    if result.get("violations"):
        lines.extend(["", "## Violations"])
        for violation in result["violations"][:25]:
            lines.append(f"- {violation.get('type')}: {violation}")
    return "\n".join(lines)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("bars must have a timestamp column or DatetimeIndex")
    return frame.sort_index()


def _timestamp_key(timestamp: Any) -> str:
    try:
        return pd.Timestamp(timestamp).isoformat()
    except Exception:  # noqa: BLE001
        return str(timestamp)


def _generate_target(pipeline: Any, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    if pipeline.feature_assembler is None:
        return pd.DataFrame()
    features = pipeline.feature_assembler.compute(bars)
    if features is None or features.empty or pipeline.signal_battery is None:
        return pd.DataFrame()
    signals = _generate_signals(pipeline.signal_battery, bars, symbol)
    if signals is None or signals.empty or pipeline.meta_pipeline is None:
        return pd.DataFrame()
    meta = pipeline.meta_pipeline.predict(features, signals)
    if meta is None or meta.empty or pipeline.bet_sizing is None:
        return pd.DataFrame()
    bets = pipeline.bet_sizing.compute(meta, features)
    if pipeline.portfolio_optimizer is None:
        return pd.DataFrame()
    return pipeline.portfolio_optimizer.compute_target_portfolio(
        strategy_returns={},
        current_signals=signals,
        bet_sizes=bets if bets is not None else pd.DataFrame(),
        regime=getattr(pipeline.config, "regime", None),
        nav=pipeline.order_manager.portfolio.nav,
    )


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(previous) | set(current)
    return sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)


def _generate_signals(signal_battery: Any, bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    for name in ("generate", "generate_all"):
        fn = getattr(signal_battery, name, None)
        if not callable(fn):
            continue
        try:
            return fn(bars, symbol=symbol)
        except TypeError:
            return fn(bars)
    return pd.DataFrame()
