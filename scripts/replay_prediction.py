#!/usr/bin/env python3
"""Replay a persisted meta-label prediction (C4).

Given ``(symbol, timestamp)``, looks up the stored prediction, rebuilds the
feature row from the feature store, re-runs the model at the recorded
version, and reports whether the prediction is reproducible.

Usage:
    python scripts/replay_prediction.py --symbol AAPL \\
        --timestamp 2026-04-17T14:30:00Z
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_factory.assembler import compute_feature_hash  # noqa: E402

log = logging.getLogger("replay_prediction")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="replay_prediction")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timestamp", required=True,
                   help="ISO 8601, e.g. 2026-04-17T14:30:00Z")
    p.add_argument("--window-seconds", type=int, default=1,
                   help="Search window around the target timestamp")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> dict:
    # Project-level bootstrap responsibility — this CLI only orchestrates.
    try:
        from src.bootstrap import build_replay_context  # type: ignore
    except ImportError:
        print(
            "ERROR: src.bootstrap.build_replay_context() not found.",
            file=sys.stderr,
        )
        return {"match": False, "error": "bootstrap_missing"}

    ctx = build_replay_context()  # {db, model_registry, feature_store}

    target = datetime.fromisoformat(args.timestamp.replace("Z", "+00:00"))
    rows = await ctx["db"].get_meta_labels(
        symbol=args.symbol,
        start=target, end=target,
    )
    if len(rows) == 0:
        return {"match": False, "error": "no_persisted_prediction"}
    stored = rows.iloc[0]
    stored_hash = stored["feature_hash"]
    stored_prob = float(stored["calibrated_prob"] or stored["meta_prob"])
    model_version = str(stored["model_version"])

    feature_row = ctx["feature_store"].get_features_at(
        args.symbol, timestamp=target,
    )
    recomputed_hash = compute_feature_hash(feature_row)

    model = ctx["model_registry"].load_model(model_version)
    recomputed_prob = float(model.predict_proba(feature_row.to_frame().T)[0])

    return {
        "symbol": args.symbol,
        "timestamp": args.timestamp,
        "model_version": model_version,
        "original_prob": stored_prob,
        "recomputed_prob": recomputed_prob,
        "match": abs(stored_prob - recomputed_prob) < 1e-6,
        "feature_hash_match": stored_hash == recomputed_hash,
        "stored_hash": stored_hash,
        "recomputed_hash": recomputed_hash,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("match") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
