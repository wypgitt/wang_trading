#!/usr/bin/env python3
"""Run shadow-mode replay over recent historical bars."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.execution.shadow_replay import (  # noqa: E402
    format_shadow_replay_report,
    load_expectation,
    replay_recent_live_targets,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("shadow_replay")
    p.add_argument("--config", required=True, help="live_trading.yaml path")
    p.add_argument("--symbols", default=None, help="comma-separated symbol override")
    p.add_argument("--bar-limit", type=int, default=250)
    p.add_argument("--replay-points", type=int, default=25)
    p.add_argument("--expected", default=None, help="optional expected behavior JSON")
    p.add_argument("--json", action="store_true", help="emit raw JSON instead of Markdown")
    p.add_argument("--output", default=None, help="optional path to write the report")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else None
    )
    result = asyncio.run(replay_recent_live_targets(
        config_path=args.config,
        symbols=symbols,
        bar_limit=args.bar_limit,
        replay_points=args.replay_points,
        expectation=load_expectation(args.expected),
    ))
    text = (
        json.dumps(result, indent=2, default=str)
        if args.json else format_shadow_replay_report(result)
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
