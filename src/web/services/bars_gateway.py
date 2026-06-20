"""Read-only access to the ``bars`` hypertable for ``/markets`` + ``/symbols``.

The bars hypertable is one of the two real data sources. This gateway is the
single place the BFF reads it: a parameterized, mandatory-LIMIT, recency-first
query that degrades to ``None`` (never raises) when the database is
unreachable — so the routes can render a row with ``data_available=False``
rather than 500.

``DatabaseManager.get_bars`` orders ``timestamp ASC LIMIT n`` (oldest-N), which
is wrong for a "latest bars" view, so the gateway issues its own
``ORDER BY timestamp DESC LIMIT n`` against the manager's pooled engine and
reverses to oldest-first for charting.

Tests inject a ``fetch`` callable and never touch a real database.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# (symbol, bar_type, limit) -> rows oldest-first, [] (no rows), or None (down).
FetchFn = Callable[[str, str, int], Optional[list[dict[str, Any]]]]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def closes_of(rows: list[dict[str, Any]]) -> list[float]:
    """The non-null close series for a bar window (the spark/line source).

    Shared by ``/markets`` and ``/symbols`` so the two surfaces derive ``spark``
    identically — close is NOT NULL in the bars schema, so this is normally
    every row, but filtering keeps the ``list[float]`` contract honest.
    """

    return [c for c in (_to_float(r.get("close")) for r in rows) if c is not None]


def window_change_pct(closes: list[float]) -> Optional[float]:
    """``(last - first) / first`` over the window, or ``None``.

    Returns ``None`` for a degenerate window (< 2 closes, or a zero first
    close) rather than a fabricated ``0.0`` — a single-bar window has no
    honest "change over the window".
    """

    if len(closes) < 2:
        return None
    first, last = closes[0], closes[-1]
    if first == 0:
        return None
    return (last - first) / first


def _resolve_db_url() -> Optional[str]:
    """Best-effort DB URL: explicit env first, then settings, else None."""

    url = os.environ.get("WANG_DATABASE_URL")
    if url:
        return url
    try:
        from config.settings import load_settings

        return load_settings().database.url
    except Exception:  # noqa: BLE001 — config optional; degrade to no-DB
        return None


class BarsGateway:
    """Recency-first reader over the bars hypertable, degradation-safe."""

    def __init__(
        self,
        *,
        fetch: FetchFn | None = None,
        db_url: str | None = None,
        default_limit: int = 120,
    ) -> None:
        self._fetch = fetch
        self._db_url = db_url
        self._default_limit = int(default_limit)
        self._db: Any = None  # lazily constructed DatabaseManager

    def recent_bars(
        self, symbol: str, bar_type: str, limit: int | None = None
    ) -> Optional[list[dict[str, Any]]]:
        """Most-recent ``limit`` bars (oldest-first).

        Returns ``[]`` when the table simply has no rows for the symbol, and
        ``None`` when the database is unreachable (caller degrades).
        """

        n = int(limit or self._default_limit)
        if self._fetch is not None:
            try:
                return self._fetch(symbol.upper(), bar_type, n)
            except Exception as exc:  # noqa: BLE001
                log.warning("bars gateway fetch override failed symbol=%s: %s", symbol, exc)
                return None
        return self._db_recent_bars(symbol.upper(), bar_type, n)

    # ── DB path ───────────────────────────────────────────────────────

    def _manager(self) -> Any:
        if self._db is not None:
            return self._db
        url = self._db_url or _resolve_db_url()
        if not url:
            return None
        try:
            from src.data_engine.storage.database import DatabaseManager

            self._db = DatabaseManager(url)
        except Exception as exc:  # noqa: BLE001
            log.warning("bars gateway: cannot construct DatabaseManager: %s", exc)
            return None
        return self._db

    def _db_recent_bars(
        self, symbol: str, bar_type: str, n: int
    ) -> Optional[list[dict[str, Any]]]:
        mgr = self._manager()
        if mgr is None:
            return None
        try:
            import pandas as pd
            from sqlalchemy import text

            query = text(
                "SELECT * FROM bars WHERE symbol = :s AND bar_type = :bt "
                "ORDER BY timestamp DESC LIMIT :n"
            )
            df = pd.read_sql(query, mgr.engine, params={"s": symbol, "bt": bar_type, "n": n})
        except Exception as exc:  # noqa: BLE001
            log.warning("bars gateway query failed symbol=%s: %s", symbol, exc)
            return None
        if df is None or len(df) == 0:
            return []
        df = df.iloc[::-1]  # DESC -> oldest-first for charting
        return df.to_dict(orient="records")


__all__ = ["BarsGateway", "FetchFn"]
