"""Regression tests for versioned database migrations."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from src.data_engine.storage.migrations import MIGRATION_TABLE, run_migrations


def test_migrations_run_idempotently_on_sqlite(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'migrations.db'}"

    applied = run_migrations(db_url)
    applied_again = run_migrations(db_url)

    assert applied == ["0001", "0002", "0003"]
    assert applied_again == []

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            versions = [
                row[0]
                for row in conn.execute(
                    text(f"SELECT version FROM {MIGRATION_TABLE} ORDER BY version")
                )
            ]
            tables = {
                row[0]
                for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
            }
    finally:
        engine.dispose()

    assert versions == ["0001", "0002", "0003"]
    assert {"bars", "orders", "audit_log"}.issubset(tables)
