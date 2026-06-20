"""Tmpfs trade-idea publisher.

Writes the read-only trade-idea report to a tmpfs JSON file after each
generation cycle so the BFF (``src.web``) can serve it without paying
the live-bootstrap latency on every HTTP request. The file is written
to a ``.tmp`` sibling and ``os.replace``-d into place so readers never
observe a partial document.

The on-disk format is:

    {
        "schema_version": <int>,
        "as_of": "<ISO-8601 UTC timestamp>",
        "report": <TradeIdeaReport.to_dict() output>
    }

The BFF's ``TmpfsTradeIdeasCache`` parses this shape; the publisher and
the cache are the only writers/readers of the file format, so any field
additions should be coordinated across both.

CLI usage:

    python -m src.execution.trade_idea_publisher --once
    python -m src.execution.trade_idea_publisher --period 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# On-disk snapshot schema version (engine->BFF contract). Bump on shape change.
SNAPSHOT_SCHEMA_VERSION = 1


# ── Path resolution ───────────────────────────────────────────────────

_ENV_VAR = "WANG_TRADE_IDEAS_PATH"
_FILE_NAME = "trade_ideas.json"


def default_output_path() -> Path:
    """Pick the platform-appropriate tmpfs path for the trade-idea file.

    Resolution order:
    1. ``$WANG_TRADE_IDEAS_PATH`` if set (full path including filename).
    2. ``${XDG_RUNTIME_DIR}/wang/trade_ideas.json`` if ``XDG_RUNTIME_DIR``
       is set (typical on Linux desktop sessions; also our macOS dev
       convention per docs/backend_design.md §24a).
    3. ``/run/wang/trade_ideas.json`` on Linux (production tmpfs).
    4. ``/tmp/wang/trade_ideas.json`` as the last-resort fallback.

    The parent directory is created (``mkdir -p``) so callers can write
    directly without retry boilerplate.
    """

    env_override = os.environ.get(_ENV_VAR)
    if env_override:
        path = Path(env_override)
    else:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg:
            path = Path(xdg) / "wang" / _FILE_NAME
        elif sys.platform.startswith("linux"):
            path = Path("/run/wang") / _FILE_NAME
        else:
            path = Path("/tmp/wang") / _FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── Publisher ─────────────────────────────────────────────────────────

class TradeIdeaPublisher:
    """Generate trade-idea reports on a schedule and write to tmpfs.

    The publisher delegates report generation to
    :func:`src.ui.trade_ideas.generate_trade_idea_report_sync` — it
    never duplicates pipeline logic. After each cycle the report is
    serialised to ``{"as_of": ..., "report": <report.to_dict()>}``,
    written to ``<output_path>.tmp``, and then ``os.replace``-d onto
    ``output_path`` for atomic visibility.

    ``run()`` is the async loop suitable for embedding in the trading
    process; ``publish_once()`` exposes a single cycle for tests and
    one-shot CLI use; ``run_sync()`` wraps ``run()`` for synchronous
    callers.
    """

    def __init__(
        self,
        *,
        config_path: str | None = None,
        symbols: list[str] | None = None,
        bar_limit: int = 500,
        min_abs_weight: float = 0.0025,
        allow_confidence_fallback: bool = False,
        output_path: str | Path | None = None,
        period_seconds: float = 60.0,
    ) -> None:
        self.config_path = config_path
        self.symbols = list(symbols) if symbols else None
        self.bar_limit = int(bar_limit)
        self.min_abs_weight = float(min_abs_weight)
        self.allow_confidence_fallback = bool(allow_confidence_fallback)
        self.output_path = Path(output_path) if output_path else default_output_path()
        # Ensure the directory exists even when the caller supplied a path.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.period_seconds = float(period_seconds)

    def publish_once(self) -> Any:
        """Generate a report and write it atomically to ``output_path``.

        Returns the ``TradeIdeaReport`` instance so callers (and tests)
        can inspect what was just written.
        """

        # Lazy import keeps the publisher importable even when the full
        # pipeline isn't available (e.g. unit tests that monkeypatch the
        # generator out).
        from src.ui import trade_ideas as trade_ideas_module

        report = trade_ideas_module.generate_trade_idea_report_sync(
            config_path=self.config_path,
            symbols=self.symbols,
            bar_limit=self.bar_limit,
            min_abs_weight=self.min_abs_weight,
            allow_confidence_fallback=self.allow_confidence_fallback,
        )

        payload = {
            # Bump when the on-disk snapshot shape changes. The BFF reader
            # (TmpfsTradeIdeasCache) checks this across the engine->BFF
            # boundary — the highest-leverage failure point (1 of 2 sources).
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "report": report.to_dict() if hasattr(report, "to_dict") else report,
        }
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        # Write -> fsync -> rename for atomicity. The temp file is in
        # the same directory so ``os.replace`` is an atomic rename.
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str, sort_keys=True)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # fsync can fail on certain tmpfs setups; the rename is
                # still atomic, so log and move on.
                log.debug("fsync failed for %s; continuing", tmp_path, exc_info=True)
        os.replace(tmp_path, self.output_path)
        log.info(
            "trade idea snapshot published path=%s idea_count=%d",
            self.output_path,
            len(getattr(report, "ideas", []) or []),
        )
        return report

    async def run(self) -> None:
        """Run an indefinite publish loop, one cycle per ``period_seconds``.

        Individual cycle exceptions are logged and swallowed so the loop
        stays up across transient pipeline failures (data source hiccup,
        empty model registry, etc). ``asyncio.CancelledError`` is the
        only signal that stops the loop.
        """

        log.info(
            "trade idea publisher starting period=%.1fs path=%s",
            self.period_seconds,
            self.output_path,
        )
        try:
            while True:
                try:
                    await asyncio.to_thread(self.publish_once)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    log.exception("trade idea publish cycle failed; will retry")
                try:
                    await asyncio.sleep(max(0.0, self.period_seconds))
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            log.info("trade idea publisher cancelled")
            raise

    def run_sync(self) -> None:
        """Convenience entry point for non-asyncio callers."""

        asyncio.run(self.run())


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish read-only trade-idea reports to a tmpfs JSON file "
            "consumed by the BFF and other read-only viewers."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Live trading YAML to load in paper-rehearsal mode (defaults to bootstrap's auto-discovery).",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol override; defaults to the live config.",
    )
    parser.add_argument("--bar-limit", type=int, default=500)
    parser.add_argument("--min-abs-weight", type=float, default=0.0025)
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to $WANG_TRADE_IDEAS_PATH or the platform tmpfs path.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=60.0,
        help="Seconds between publish cycles when running in loop mode.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Generate a single report and exit (useful for cron/oneshot service units).",
    )
    parser.add_argument(
        "--allow-confidence-fallback",
        action="store_true",
        help="Use confidence-based meta probabilities when no production model is loaded.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser.parse_args(argv)


def _parse_symbols(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return symbols or None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    publisher = TradeIdeaPublisher(
        config_path=args.config,
        symbols=_parse_symbols(args.symbols),
        bar_limit=args.bar_limit,
        min_abs_weight=args.min_abs_weight,
        allow_confidence_fallback=args.allow_confidence_fallback,
        output_path=args.output,
        period_seconds=args.period,
    )

    if args.once:
        publisher.publish_once()
        return 0

    # Install a clean SIGINT/SIGTERM handler so the loop's
    # CancelledError path drains the publisher.
    loop = asyncio.new_event_loop()
    task = loop.create_task(publisher.run())

    def _stop(signum: int, _frame: Any) -> None:
        log.info("received signal %s; stopping publisher", signum)
        loop.call_soon_threadsafe(task.cancel)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):
            # Not on the main thread (or platform without that signal);
            # rely on KeyboardInterrupt fallthrough.
            pass

    try:
        loop.run_until_complete(task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        loop.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
