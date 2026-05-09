#!/usr/bin/env python3
"""Build a local production-like rehearsal environment.

This is intentionally local-only: it creates a SQLite market-data DB, seeds a
fresh feature timestamp, trains/promotes a small MLflow production model, runs
mock HTTP endpoints for Prometheus/Grafana reachability, and then executes the
same shadow replay + preflight code used by live startup.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import build_preflight_checker  # noqa: E402
from src.execution.preflight import PreflightChecker  # noqa: E402
from src.execution.shadow_replay import (  # noqa: E402
    format_shadow_replay_report,
    replay_recent_live_targets,
)
from src.ml_layer.meta_labeler import MetaLabeler  # noqa: E402
from src.ml_layer.model_registry import ModelRegistry  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("local_readiness_rehearsal")
    parser.add_argument(
        "--workdir",
        default="/tmp/wang_trading_readiness_rehearsal",
        help="directory for generated DB, MLflow store, config, and reports",
    )
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--bars", type=int, default=220)
    parser.add_argument("--replay-points", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    args = _parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    _reset_workdir(workdir)

    db_path = workdir / "readiness.db"
    mlflow_db = workdir / "mlflow.db"
    halt_path = workdir / "halt"
    operator_checkin = workdir / "operator_checkin"
    compliance_log = workdir / "compliance.log"
    shadow_json_path = workdir / "shadow_replay.json"
    shadow_report_path = workdir / "shadow_replay.md"
    preflight_json_path = workdir / "preflight.json"

    _seed_sqlite_market_data(db_path, symbol=args.symbol, rows=args.bars)
    operator_checkin.write_text("ok\n", encoding="utf-8")
    run_id = _seed_production_model(mlflow_db, workdir)

    with _local_http_probe_server() as probe_url:
        config_path = _write_live_config(
            workdir=workdir,
            db_path=db_path,
            mlflow_db=mlflow_db,
            symbol=args.symbol,
            probe_url=probe_url,
            halt_path=halt_path,
            operator_checkin=operator_checkin,
            compliance_log=compliance_log,
        )
        shadow = asyncio.run(replay_recent_live_targets(
            config_path=config_path,
            symbols=[args.symbol],
            bar_limit=args.bars,
            replay_points=args.replay_points,
        ))
        shadow_json_path.write_text(
            json.dumps(shadow, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        shadow_report_path.write_text(
            format_shadow_replay_report(shadow) + "\n",
            encoding="utf-8",
        )

        checker = build_preflight_checker(config_path)
        checks = asyncio.run(checker.run_all_checks())
        preflight = PreflightChecker.summary(checks)
        preflight_json_path.write_text(
            json.dumps(preflight, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    result = {
        "status": (
            "passed"
            if shadow.get("passed") and preflight.get("all_passed")
            else "failed"
        ),
        "workdir": str(workdir),
        "config": str(config_path),
        "symbol": args.symbol,
        "production_model_run_id": run_id,
        "shadow_replay": {
            "passed": bool(shadow.get("passed")),
            "cycles": int(shadow.get("cycles", 0)),
            "violations": len(shadow.get("violations") or []),
            "report": str(shadow_report_path),
            "json": str(shadow_json_path),
        },
        "preflight": {
            "all_passed": bool(preflight.get("all_passed")),
            "blockers_failed": int(preflight.get("blockers_failed", 0)),
            "warnings_failed": int(preflight.get("warnings_failed", 0)),
            "json": str(preflight_json_path),
        },
    }
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_format_summary(result))
    return 0 if result["status"] == "passed" else 1


def _reset_workdir(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for name in (
        "readiness.db",
        "mlflow.db",
        "mlflow.db-shm",
        "mlflow.db-wal",
        "live_trading.yaml",
        "shadow_replay.json",
        "shadow_replay.md",
        "preflight.json",
        "halt",
        "operator_checkin",
        "compliance.log",
    ):
        path = workdir / name
        if path.exists():
            path.unlink()
    shutil.rmtree(workdir / "mlruns", ignore_errors=True)


def _seed_sqlite_market_data(
    db_path: Path,
    *,
    symbol: str,
    rows: int,
) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=rows)
    rng = np.random.default_rng(42)
    price = 100.0
    records: list[dict[str, Any]] = []
    for i in range(rows):
        ts = start + timedelta(minutes=i + 1)
        drift = 0.03 + 0.35 * np.sin(i / 17.0)
        shock = float(rng.normal(0.0, 0.08))
        open_px = price
        close_px = max(1.0, open_px + drift + shock)
        high = max(open_px, close_px) + 0.15
        low = min(open_px, close_px) - 0.15
        volume = 1_000.0 + 20.0 * (i % 13)
        records.append({
            "timestamp": ts,
            "open_time": ts - timedelta(minutes=1),
            "symbol": symbol,
            "bar_type": "tib",
            "open": open_px,
            "high": high,
            "low": low,
            "close": close_px,
            "volume": volume,
            "dollar_volume": volume * close_px,
            "tick_count": 100 + (i % 20),
            "buy_volume": volume * 0.52,
            "sell_volume": volume * 0.48,
            "buy_ticks": 55,
            "sell_ticks": 45,
            "imbalance": 0.04,
            "threshold": 50.0,
            "vwap": (open_px + close_px) / 2.0,
            "bar_duration_seconds": 60.0,
        })
        price = close_px

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE bars (
                timestamp TIMESTAMP NOT NULL,
                open_time TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                bar_type TEXT NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                dollar_volume DOUBLE PRECISION NOT NULL,
                tick_count INTEGER NOT NULL,
                buy_volume DOUBLE PRECISION DEFAULT 0,
                sell_volume DOUBLE PRECISION DEFAULT 0,
                buy_ticks INTEGER DEFAULT 0,
                sell_ticks INTEGER DEFAULT 0,
                imbalance DOUBLE PRECISION DEFAULT 0,
                threshold DOUBLE PRECISION DEFAULT 0,
                vwap DOUBLE PRECISION DEFAULT 0,
                bar_duration_seconds DOUBLE PRECISION DEFAULT 0
            )
        """))
        conn.execute(text("""
            CREATE TABLE features (
                timestamp TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                value DOUBLE PRECISION NOT NULL
            )
        """))
        conn.execute(
            text("""
                INSERT INTO bars (
                    timestamp, open_time, symbol, bar_type, open, high, low,
                    close, volume, dollar_volume, tick_count, buy_volume,
                    sell_volume, buy_ticks, sell_ticks, imbalance, threshold,
                    vwap, bar_duration_seconds
                )
                VALUES (
                    :timestamp, :open_time, :symbol, :bar_type, :open, :high,
                    :low, :close, :volume, :dollar_volume, :tick_count,
                    :buy_volume, :sell_volume, :buy_ticks, :sell_ticks,
                    :imbalance, :threshold, :vwap, :bar_duration_seconds
                )
            """),
            records,
        )
        conn.execute(
            text("""
                INSERT INTO features (timestamp, symbol, feature_name, value)
                VALUES (:timestamp, :symbol, :feature_name, :value)
            """),
            {
                "timestamp": now,
                "symbol": symbol,
                "feature_name": "readiness_freshness",
                "value": 1.0,
            },
        )
    engine.dispose()


def _seed_production_model(mlflow_db: Path, workdir: Path) -> str:
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        registry = ModelRegistry(
            tracking_uri=f"sqlite:///{mlflow_db}",
            experiment_name="local-readiness-meta-labeler",
        )
        rng = np.random.default_rng(7)
        idx = pd.date_range(
            datetime.now(timezone.utc) - timedelta(hours=120),
            periods=120,
            freq="1h",
        )
        X = pd.DataFrame(
            {
                "signal_side": rng.choice([-1.0, 1.0], size=len(idx)),
                "signal_confidence": rng.uniform(0.1, 0.95, size=len(idx)),
            },
            index=idx,
        )
        logits = 1.2 * X["signal_side"] * X["signal_confidence"]
        probs = 1.0 / (1.0 + np.exp(-logits.to_numpy()))
        y = pd.Series((rng.uniform(size=len(idx)) < probs).astype(int), index=idx)
        model = MetaLabeler(
            model_type="random_forest",
            params={"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 2},
            calibrate=False,
        ).fit(X, y)
        run_id = registry.log_training_run(
            model,
            X,
            y,
            params={
                "n_training_events": len(X),
                "gates": {"cpcv": True, "dsr": True, "pbo": True},
            },
            metrics={"gate_cpcv": 1.0, "gate_dsr": 1.0, "gate_pbo": 1.0},
            cv_scores=np.array([0.61, 0.64, 0.63]),
        )
        registry.promote_model(run_id, stage="production")
        return run_id
    finally:
        os.chdir(cwd)


def _write_live_config(
    *,
    workdir: Path,
    db_path: Path,
    mlflow_db: Path,
    symbol: str,
    probe_url: str,
    halt_path: Path,
    operator_checkin: Path,
    compliance_log: Path,
) -> Path:
    config = {
        "asset_class": "equities",
        "symbols": [symbol],
        "dry_run": True,
        "initial_cash": 100_000,
        "halt_file": str(halt_path),
        "operator_checkin_path": str(operator_checkin),
        "compliance_log_path": str(compliance_log),
        "storage": {"database_url": f"sqlite:///{db_path}"},
        "broker": {
            "adapter": "paper",
            "initial_cash": 100_000,
            "fill_delay_ms": 0,
        },
        "paper_prices": {symbol: 100.0},
        "bars": {"type": "tib"},
        "mlflow": {
            "tracking_uri": f"sqlite:///{mlflow_db}",
            "experiment_name": "local-readiness-meta-labeler",
        },
        "circuit_breakers": {
            "max_order_pct": 0.50,
            "daily_loss_limit_pct": 0.02,
            "max_positions": 20,
            "max_single_position_pct": 0.10,
            "max_gross_exposure": 1.50,
        },
        "paper_stats": {
            "weeks_history": 9,
            "sharpe": 1.25,
            "max_drawdown": 0.05,
            "win_rate": 0.56,
            "n_trades": 60,
        },
        "operator": {
            "risk_acknowledged": True,
            "emergency_contact": "local-readiness@example.invalid",
            "within_working_hours": True,
        },
        "infra": {
            "prometheus_url": probe_url,
            "grafana_url": probe_url,
            "alert_ping_enabled": True,
            "db_disk_path": str(workdir),
            "probe_timeout_s": 2.0,
        },
        "preflight": {
            "broker": {"min_buying_power_per_asset": 1.0},
            "model": {
                "max_age_days": 30,
                "min_training_events": 50,
                "require_regime_detector": False,
            },
            "paper": {
                "min_weeks_history": 8,
                "min_sharpe": 1.0,
                "max_drawdown": 0.15,
                "min_win_rate": 0.5,
                "min_completed_trades": 50,
            },
            "infra": {
                "max_db_disk_pct": 95,
                "feature_freshness_max_h": 24,
            },
        },
        "pipeline": {
            "drift_check_every": 100,
            "snapshot_every": 50,
            "sleep_seconds": 0.0,
        },
        "features": {
            "structural_breaks": {
                "window": 20,
                "min_window_sadf": 10,
                "min_period_chow": 12,
            },
            "entropy": {"window": 20},
            "microstructure": {"window": 20},
            "volatility": {
                "window": 30,
                "refit_interval": 20,
                "short_window": 3,
                "long_window": 10,
                "vvol_window": 10,
            },
            "classical": {
                "rsi_window": 5,
                "bbw_window": 10,
                "ret_z_windows": [3, 5],
            },
        },
        "signals": {
            "ts_momentum": {
                "lookbacks": [2, 4, 8],
                "history_window": 10,
                "min_history": 20,
            },
            "ma_crossover": {"fast_period": 3, "slow_period": 8},
            "donchian": {"entry_period": 8, "exit_period": 4},
        },
    }
    path = workdir / "live_trading.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@contextlib.contextmanager
def _local_http_probe_server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def _format_summary(result: dict[str, Any]) -> str:
    return "\n".join([
        f"Local readiness rehearsal: {result['status'].upper()}",
        f"Workdir: {result['workdir']}",
        f"Config: {result['config']}",
        f"Production model run: {result['production_model_run_id']}",
        (
            "Shadow replay: "
            f"passed={result['shadow_replay']['passed']} "
            f"cycles={result['shadow_replay']['cycles']} "
            f"violations={result['shadow_replay']['violations']} "
            f"report={result['shadow_replay']['report']}"
        ),
        (
            "Preflight: "
            f"all_passed={result['preflight']['all_passed']} "
            f"blockers_failed={result['preflight']['blockers_failed']} "
            f"warnings_failed={result['preflight']['warnings_failed']} "
            f"json={result['preflight']['json']}"
        ),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
