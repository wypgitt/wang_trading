"""Database migration runner for production schema management.

This wraps the existing DDL strings in explicit, versioned migrations. It is
deliberately small: no external migration framework, but enough tracking to
know which schema phases were applied and to avoid replaying everything as an
opaque one-shot setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.data_engine.storage.database import SCHEMA_SQL
from src.execution.audit_log import AUDIT_LOG_DDL
from src.execution.storage import EXECUTION_SCHEMA_SQL


MIGRATION_TABLE = "schema_migrations"


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    sql: str


MIGRATIONS: tuple[Migration, ...] = (
    Migration("0001", "core market-data schema", SCHEMA_SQL),
    Migration("0002", "execution storage schema", EXECUTION_SCHEMA_SQL),
    Migration("0003", "audit log schema", AUDIT_LOG_DDL),
)


class MigrationRunner:
    def __init__(self, db_url: str, migrations: Iterable[Migration] = MIGRATIONS):
        self.db_url = db_url
        self.migrations = tuple(migrations)
        self.engine: Engine = create_engine(db_url)

    def close(self) -> None:
        self.engine.dispose()

    def run(self) -> list[str]:
        """Apply pending migrations and return applied versions."""
        self._ensure_table()
        applied = self.applied_versions()
        changed: list[str] = []
        for migration in self.migrations:
            if migration.version in applied:
                continue
            logger.info(
                "Applying DB migration {}: {}",
                migration.version,
                migration.description,
            )
            self._apply_sql(migration.sql)
            self._record(migration)
            changed.append(migration.version)
        return changed

    def applied_versions(self) -> set[str]:
        with self.engine.connect() as conn:
            try:
                rows = conn.execute(
                    text(f"SELECT version FROM {MIGRATION_TABLE}")
                ).fetchall()
            except Exception:
                return set()
        return {str(row[0]) for row in rows}

    def _ensure_table(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
                    version     TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()

    def _record(self, migration: Migration) -> None:
        with self.engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {MIGRATION_TABLE} (version, description)
                    VALUES (:version, :description)
                """),
                {
                    "version": migration.version,
                    "description": migration.description,
                },
            )
            conn.commit()

    def _apply_sql(self, sql: str) -> None:
        for stmt in _split_sql_statements(sql):
            stmt = stmt.strip()
            if not stmt:
                continue
            stmt = _statement_for_dialect(stmt, self.engine.dialect.name)
            if stmt is None:
                continue
            with self.engine.connect() as conn:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception as exc:  # noqa: BLE001
                    conn.rollback()
                    if _is_tolerable_schema_warning(stmt, exc):
                        logger.warning("Schema statement warning: {}", exc)
                        continue
                    raise


def run_migrations(db_url: str) -> list[str]:
    runner = MigrationRunner(db_url)
    try:
        return runner.run()
    finally:
        runner.close()


def _is_tolerable_schema_warning(stmt: str, exc: Exception) -> bool:
    msg = str(exc).lower()
    low_stmt = stmt.lower()
    return (
        "already a hypertable" in msg
        or "create_hypertable" in msg
        or "no such function" in msg
        or (
            "does not exist" in msg
            and "create_hypertable" in low_stmt
        )
        or low_stmt.startswith("select create_hypertable")
    )


def _statement_for_dialect(stmt: str, dialect_name: str) -> str | None:
    """Return a backend-compatible DDL statement, or ``None`` to skip it."""
    if dialect_name != "sqlite":
        return stmt
    low_stmt = _without_leading_comments(stmt).lstrip().lower()
    if low_stmt.startswith("create extension"):
        return None
    if low_stmt.startswith("select create_hypertable"):
        return None
    return stmt.replace("DEFAULT NOW()", "DEFAULT CURRENT_TIMESTAMP")


def _without_leading_comments(stmt: str) -> str:
    lines = stmt.splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("--")):
        lines.pop(0)
    return "\n".join(lines)


def _split_sql_statements(sql: str) -> list[str]:
    """Split SQL on statement delimiters without breaking comments/strings."""
    statements: list[str] = []
    buf: list[str] = []
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if not in_single_quote and not in_double_quote and ch == "-" and nxt == "-":
            in_line_comment = True
            buf.extend([ch, nxt])
            i += 2
            continue

        if ch == "'" and not in_double_quote:
            if in_single_quote and nxt == "'":
                buf.extend([ch, nxt])
                i += 2
                continue
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        if ch == ";" and not in_single_quote and not in_double_quote:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements
