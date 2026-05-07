#!/usr/bin/env python3
"""CLI wrapper for `RetrainPipeline.run` (C2).

Uses `src.bootstrap.build_retrain_pipeline` so operators can fire a manual
retrain without writing Python.

Usage:
    python scripts/retrain_now.py --symbol AAPL
    python scripts/retrain_now.py --symbol AAPL --trigger emergency --reason "drift"
    python scripts/retrain_now.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("retrain_now")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="retrain_now")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--all", action="store_true",
                   help="Retrain every symbol in the configured universe")
    p.add_argument("--trigger", choices=["scheduled", "manual", "drift", "emergency"],
                   default="manual")
    p.add_argument("--reason", type=str, default="",
                   help="Required for --trigger emergency")
    p.add_argument("--config", type=str, default=None,
                   help="Optional retrain YAML config path")
    return p.parse_args()


async def _dispatch(args: argparse.Namespace) -> int:
    # src.bootstrap wires the real feature assembler / signal battery /
    # gate / registry.
    try:
        from src.bootstrap import build_retrain_pipeline  # type: ignore
    except ImportError as exc:
        print(
            "ERROR: src.bootstrap.build_retrain_pipeline() unavailable: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2

    pipeline, data_loader, universe = build_retrain_pipeline(args.config)

    if args.all:
        runs = await pipeline.run_all_symbols(universe, data_loader)
        for r in runs:
            print(r.to_dict())
        bad = [r for r in runs if not r.promoted]
        return 0 if not bad else 1

    if not args.symbol:
        print("ERROR: --symbol is required when --all is not set", file=sys.stderr)
        return 2
    close, bars = data_loader(args.symbol)
    if args.trigger == "emergency":
        run = await pipeline.emergency_retrain(args.symbol, args.reason, close, bars)
    else:
        run = await pipeline.run(args.symbol, close, bars, trigger=args.trigger)
    print(run.to_dict())
    return 0 if run.promoted else 1


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return asyncio.run(_dispatch(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
