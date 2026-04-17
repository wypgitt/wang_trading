# pylint: disable=no-value-for-parameter  # Click injects CLI args
"""
Bar Validation Runner

Loads bars from the database or feature store, runs validation,
and compares bar types to verify AFML's predictions about
information-driven bars.

Usage:
    python -m src.data_engine.validation.runner --symbol AAPL
    python -m src.data_engine.validation.runner --symbol AAPL --bar-type tib
    python -m src.data_engine.validation.runner --symbol AAPL --compare-all
"""

from __future__ import annotations

import sys
import click
from loguru import logger

from src.config import get_settings
from src.data_engine.storage.database import DatabaseManager
from src.data_engine.validation.bar_validator import (
    validate_bars, compare_bar_types, BarValidationReport,
)


@click.command()
@click.option("--symbol", required=True, help="Symbol to validate")
@click.option("--bar-type", default=None, help="Specific bar type (tick, volume, dollar, tib, vib)")
@click.option("--compare-all", is_flag=True, help="Compare all bar types for this symbol")
@click.option("--limit", default=10_000, help="Max bars to load")
def main(symbol: str, bar_type: str | None, compare_all: bool, limit: int):
    """Validate bar quality and compare bar types."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    settings = get_settings()
    db = DatabaseManager(settings.database.url)

    if compare_all or bar_type is None:
        bar_types = ["tick", "volume", "dollar", "tib", "vib"]
    else:
        bar_types = [bar_type]

    reports: list[BarValidationReport] = []

    for bt in bar_types:
        try:
            df = db.get_bars(symbol, bt, limit=limit)
            if df.empty:
                logger.info(f"No bars found for {symbol}/{bt}")
                continue

            report = validate_bars(df, symbol, bt)
            reports.append(report)
            print(report.summary())
            print()

        except Exception as e:
            logger.error(f"Validation failed for {symbol}/{bt}: {e}")

    if len(reports) > 1:
        print("═══ Bar Type Comparison ═══")
        compare_bar_types(reports)

    db.close()


if __name__ == "__main__":
    main()
