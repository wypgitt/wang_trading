"""
Database Setup Script

Creates the TimescaleDB database and schema.
Run this once before starting ingestion.

Usage:
    python scripts/setup_db.py
    python scripts/setup_db.py --reset   # WARNING: drops all data
"""

import sys
import click
from loguru import logger

sys.path.insert(0, ".")

from src.config import get_settings
from src.data_engine.storage.database import DatabaseManager


@click.command()
@click.option("--reset", is_flag=True, help="Drop and recreate all tables (DESTRUCTIVE)")
def setup(reset: bool):
    """Set up the TimescaleDB schema."""
    settings = get_settings()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info(f"Database: {settings.database.host}:{settings.database.port}/{settings.database.name}")

    if reset:
        click.confirm(
            "This will DROP ALL TABLES and delete all data. Continue?",
            abort=True,
        )
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.database.url)
        with engine.connect() as conn:
            for table in ["signals", "features", "cusum_events", "bars", "raw_ticks"]:
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                    logger.info(f"Dropped table: {table}")
                except Exception as e:
                    logger.warning(f"Could not drop {table}: {e}")
            conn.commit()
        engine.dispose()

    db = DatabaseManager(settings.database.url)

    try:
        db.setup_schema()
        logger.info("Database schema created successfully!")

        # Verify
        from sqlalchemy import text
        with db.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ))
            tables = [row[0] for row in result]
            logger.info(f"Tables: {tables}")

            # Check TimescaleDB
            result = conn.execute(text(
                "SELECT hypertable_name FROM timescaledb_information.hypertables"
            ))
            hypertables = [row[0] for row in result]
            logger.info(f"Hypertables: {hypertables}")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        logger.info(
            "Make sure PostgreSQL with TimescaleDB extension is running.\n"
            "Quick setup:\n"
            "  docker run -d --name timescaledb \\\n"
            "    -p 5432:5432 \\\n"
            "    -e POSTGRES_DB=quantsystem \\\n"
            "    -e POSTGRES_USER=quant \\\n"
            "    -e POSTGRES_PASSWORD=password \\\n"
            "    timescale/timescaledb:latest-pg16"
        )
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    setup()
