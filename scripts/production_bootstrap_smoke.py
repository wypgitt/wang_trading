#!/usr/bin/env python3
"""Bootstrap-level production smoke test.

This smoke test uses real bootstrap/config/database paths but forces broker
resolution into dry-run paper mode. It intentionally stops at target
portfolio generation; no live order API is called.

Run on the production host after config + DB are in place:

    python scripts/production_bootstrap_smoke.py \
        --config /opt/wang_trading/config/live_trading.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import (  # noqa: E402
    CascadeBetSizingAdapter,
    ConfidenceMetaPipeline,
    DatabaseBarDataAdapter,
    DirectTargetOptimizer,
    ModelMetaPipeline,
    _database_url,
    _load_production_model,
    _settings,
    _symbols,
    build_model_registry,
    load_runtime_config,
)
from src.execution.broker_adapter import PaperBrokerAdapter  # noqa: E402
from src.execution.broker_factory import BrokerFactory  # noqa: E402
from src.execution.models import PortfolioState  # noqa: E402
from src.feature_factory.assembler import FeatureAssembler  # noqa: E402
from src.signal_battery.orchestrator import create_default_battery  # noqa: E402
from src.data_engine.storage.database import DatabaseManager  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("production_bootstrap_smoke")
    p.add_argument("--config", required=True, help="live_trading.yaml path")
    p.add_argument("--settings", default=None, help="optional settings.yaml path")
    p.add_argument("--symbol", default=None, help="single symbol override")
    p.add_argument("--bar-limit", type=int, default=500)
    return p.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = load_runtime_config(args.config, default_name="live_trading")
    runtime = dict(runtime)
    runtime["dry_run"] = True
    settings = _settings(args.settings)
    asset_class = runtime.get("asset_class", "equities")
    symbols = [args.symbol] if args.symbol else _symbols(asset_class, runtime, settings)
    if not symbols:
        raise RuntimeError("no symbols configured")
    symbol = symbols[0]

    db = DatabaseManager(_database_url(runtime, settings))
    with db.engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    broker_factory = BrokerFactory(runtime)
    broker = broker_factory.get_broker(symbol, asset_class=asset_class)
    if not isinstance(broker, PaperBrokerAdapter):
        raise RuntimeError(
            f"dry-run broker factory returned {type(broker).__name__}, "
            "expected PaperBrokerAdapter"
        )

    bar_cfg = getattr(settings.bars, asset_class)
    bar_type = (
        ((runtime.get("bars") or {}).get("type"))
        or runtime.get("bar_type")
        or getattr(bar_cfg, "primary_type", "tib")
    )
    adapter = DatabaseBarDataAdapter(db, bar_type=bar_type, limit=args.bar_limit)
    rows = await adapter.get_bars(symbol)
    if not rows:
        raise RuntimeError(f"no bars found for {symbol}/{bar_type}")
    bars = pd.DataFrame(rows)
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp").sort_index()

    assembler = FeatureAssembler(runtime.get("features"))
    features = assembler.compute(bars) if hasattr(assembler, "compute") else assembler.assemble(bars)
    if features is None or features.empty:
        raise RuntimeError("feature assembler produced no rows")

    battery = create_default_battery(runtime.get("signals"))
    signals = battery.generate(bars, symbol=symbol)
    if signals is None or signals.empty:
        raise RuntimeError("signal battery produced no signals")

    registry = build_model_registry(runtime)
    model = _load_production_model(registry)
    meta_pipe = ModelMetaPipeline(model) if model is not None else ConfidenceMetaPipeline()
    meta = meta_pipe.predict(features, signals)
    if meta is None or meta.empty:
        raise RuntimeError("meta pipeline produced no rows")

    portfolio = PortfolioState(cash=float(runtime.get("initial_cash", 100_000.0)))
    bets = CascadeBetSizingAdapter(
        portfolio,
        asset_class=asset_class,
        asset_class_map={s: asset_class for s in symbols},
    ).compute(meta, features)
    target = DirectTargetOptimizer().compute_target_portfolio(bet_sizes=bets)
    if target.empty:
        raise RuntimeError("target optimizer produced no rows")

    if broker.orders:
        raise RuntimeError("dry-run smoke unexpectedly submitted paper orders")

    return {
        "status": "passed",
        "asset_class": asset_class,
        "symbol": symbol,
        "bar_type": bar_type,
        "bars": int(len(bars)),
        "features_shape": list(features.shape),
        "signals": int(len(signals)),
        "meta_rows": int(len(meta)),
        "target_rows": int(len(target)),
        "model_source": "production" if model is not None else "confidence_fallback",
        "broker": type(broker).__name__,
        "live_orders_sent": 0,
    }


def main() -> int:
    args = _parse_args()
    try:
        result = asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
