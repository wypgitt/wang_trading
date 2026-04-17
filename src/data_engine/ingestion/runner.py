# pylint: disable=no-value-for-parameter  # Click injects CLI args
"""
Ingestion Runner

Main entry point for the data ingestion pipeline. Connects to data
sources, streams ticks, constructs bars in real-time, and stores
everything to TimescaleDB and the feature store.

Usage:
    # Real-time streaming
    python -m src.data_engine.ingestion.runner --asset-class equities

    # Historical backfill
    python -m src.data_engine.ingestion.runner --backfill --symbol AAPL --start 2024-01-01
"""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import click
from loguru import logger

from src.config import get_settings
from src.data_engine.models import Tick, Bar, AssetClass
from src.data_engine.bars.constructors import create_bar_constructor, BaseBarConstructor
from src.data_engine.storage.database import DatabaseManager
from src.data_engine.storage.feature_store import FeatureStore
from src.data_engine.ingestion.base_adapter import BaseAdapter


class IngestionPipeline:
    """
    Orchestrates data ingestion: adapter → tick classification → bar
    construction → storage.

    Manages multiple bar constructors per symbol (primary + secondary)
    and handles buffered database writes for efficiency.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        db: DatabaseManager,
        feature_store: FeatureStore,
        asset_class: str = "equities",
        write_buffer_size: int = 100,
    ):
        self.adapter = adapter
        self.db = db
        self.feature_store = feature_store
        self.asset_class = asset_class
        self.write_buffer_size = write_buffer_size

        self._bar_constructors: dict[str, list[BaseBarConstructor]] = {}
        self._tick_buffer: list[Tick] = []
        self._bar_buffer: list[Bar] = []
        self._running = False

        # Stats
        self._tick_count = 0
        self._bar_count = 0
        self._start_time: Optional[datetime] = None

    def setup_constructors(self, symbols: list[str]) -> None:
        """
        Create bar constructors for each symbol based on config.

        Each symbol gets both a primary and secondary bar constructor
        as defined in the config for this asset class.
        """
        settings = get_settings()
        bar_config = getattr(settings.bars, self.asset_class)

        for symbol in symbols:
            constructors = []

            # Primary bar constructor
            primary = create_bar_constructor(
                symbol=symbol,
                bar_type=bar_config.primary_type,
                ewma_span=bar_config.imbalance_ewma_span,
                bar_size=bar_config.tick_bar_size,
                initial_threshold=float(bar_config.tick_bar_size),
            )
            constructors.append(primary)
            logger.info(f"  {symbol}: primary={bar_config.primary_type}")

            # Secondary bar constructor
            if bar_config.secondary_type != bar_config.primary_type:
                size_map = {
                    "tick": bar_config.tick_bar_size,
                    "volume": bar_config.volume_bar_size,
                    "dollar": bar_config.dollar_bar_size,
                }
                sec_size = size_map.get(bar_config.secondary_type, bar_config.dollar_bar_size)
                secondary = create_bar_constructor(
                    symbol=symbol,
                    bar_type=bar_config.secondary_type,
                    bar_size=sec_size,
                    ewma_span=bar_config.imbalance_ewma_span,
                    initial_threshold=float(sec_size),
                )
                constructors.append(secondary)
                logger.info(f"  {symbol}: secondary={bar_config.secondary_type}")

            self._bar_constructors[symbol] = constructors

    async def run_stream(self, symbols: list[str]) -> None:
        """
        Main streaming loop: connect, subscribe, process ticks.
        """
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        self.setup_constructors(symbols)

        logger.info(f"Starting ingestion for {len(symbols)} symbols...")

        async with self.adapter:
            await self.adapter.subscribe(symbols)

            async for tick in self.adapter.stream_ticks():
                if not self._running:
                    break

                self._process_tick(tick)

                # Periodic status
                if self._tick_count % 10_000 == 0 and self._tick_count > 0:
                    elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
                    rate = self._tick_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Ingestion stats: {self._tick_count:,} ticks, "
                        f"{self._bar_count:,} bars, {rate:.0f} ticks/s"
                    )

        # Flush remaining buffers
        self._flush_ticks()
        self._flush_bars()
        logger.info(
            f"Ingestion stopped. Total: {self._tick_count:,} ticks, "
            f"{self._bar_count:,} bars"
        )

    async def run_backfill(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """
        Historical backfill: fetch historical ticks and construct bars.
        """
        self._start_time = datetime.now(timezone.utc)
        self.setup_constructors([symbol])

        logger.info(f"Backfilling {symbol} from {start.date()} to {end.date()}...")

        async with self.adapter:
            ticks = await self.adapter.get_historical_ticks(symbol, start, end)
            logger.info(f"Fetched {len(ticks):,} historical ticks")

            for tick in ticks:
                self._process_tick(tick)

        self._flush_ticks()
        self._flush_bars()

        logger.info(
            f"Backfill complete: {self._tick_count:,} ticks → "
            f"{self._bar_count:,} bars"
        )

    def _process_tick(self, tick: Tick) -> None:
        """Process a single tick through all constructors."""
        self._tick_count += 1
        self._tick_buffer.append(tick)

        # Flush tick buffer periodically
        if len(self._tick_buffer) >= self.write_buffer_size:
            self._flush_ticks()

        # Feed to all bar constructors for this symbol
        constructors = self._bar_constructors.get(tick.symbol, [])
        for constructor in constructors:
            bar = constructor.process_tick(tick)
            if bar is not None:
                self._bar_count += 1
                self._bar_buffer.append(bar)
                logger.debug(
                    f"Bar [{bar.bar_type.value}] {bar.symbol}: "
                    f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
                    f"C={bar.close:.2f} V={bar.volume:.0f} "
                    f"ticks={bar.tick_count} dur={bar.bar_duration_seconds:.0f}s"
                )

                # Flush bar buffer periodically
                if len(self._bar_buffer) >= self.write_buffer_size:
                    self._flush_bars()

    def _flush_ticks(self) -> None:
        """Write buffered ticks to database."""
        if self._tick_buffer:
            try:
                count = self.db.insert_ticks(self._tick_buffer)
                logger.debug(f"Flushed {count} ticks to DB")
            except Exception as e:
                logger.error(f"Failed to flush ticks: {e}")
            self._tick_buffer.clear()

    def _flush_bars(self) -> None:
        """Write buffered bars to database and feature store."""
        if self._bar_buffer:
            try:
                count = self.db.insert_bars(self._bar_buffer)
                logger.debug(f"Flushed {count} bars to DB")
            except Exception as e:
                logger.error(f"Failed to flush bars: {e}")
            self._bar_buffer.clear()

    def stop(self) -> None:
        """Signal the pipeline to stop."""
        self._running = False
        logger.info("Stop signal received")


def _create_adapter(asset_class: str) -> BaseAdapter:
    """Create the appropriate adapter for the asset class."""
    settings = get_settings()

    if asset_class == "equities":
        from src.data_engine.ingestion.adapters.alpaca import AlpacaAdapter
        return AlpacaAdapter(
            api_key=settings.data_sources.alpaca.api_key,
            secret_key=settings.data_sources.alpaca.secret_key,
            feed=settings.data_sources.alpaca.feed,
        )
    elif asset_class == "crypto":
        from src.data_engine.ingestion.adapters.ccxt_adapter import CCXTAdapter
        return CCXTAdapter(
            exchange_id="binance",
            api_key=settings.data_sources.binance.api_key,
            secret_key=settings.data_sources.binance.secret_key,
            testnet=settings.data_sources.binance.testnet,
        )
    elif asset_class == "futures":
        raise NotImplementedError(
            "IBKR adapter not yet implemented — coming in Phase 5"
        )
    else:
        raise ValueError(f"Unsupported asset class: {asset_class}")


def _get_symbols(asset_class: str) -> list[str]:
    """Get the symbol list for the asset class."""
    settings = get_settings()
    instruments = getattr(settings.instruments, asset_class, {})

    if asset_class == "equities":
        return instruments.get("test_symbols", ["AAPL", "SPY"])
    elif asset_class == "crypto":
        return instruments.get("symbols", ["BTC/USDT"])
    elif asset_class == "futures":
        return instruments.get("symbols", ["ES"])
    return []


@click.command()
@click.option("--asset-class", type=click.Choice(["equities", "crypto", "futures"]), default="equities")
@click.option("--backfill", is_flag=True, help="Run historical backfill instead of live stream")
@click.option("--symbol", default=None, help="Single symbol for backfill")
@click.option("--start", default=None, help="Backfill start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="Backfill end date (YYYY-MM-DD)")
@click.option("--days", default=30, help="Backfill N days from today (if no start/end)")
def main(
    asset_class: str,
    backfill: bool,
    symbol: Optional[str],
    start: Optional[str],
    end: Optional[str],
    days: int,
):
    """Run the data ingestion pipeline."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=get_settings().system.log_level)
    logger.add("logs/ingestion_{time}.log", rotation="100 MB", retention="7 days")

    # Initialize components
    settings = get_settings()
    db = DatabaseManager(settings.database.url)
    feature_store = FeatureStore(settings.feature_store.local_path)
    adapter = _create_adapter(asset_class)

    pipeline = IngestionPipeline(
        adapter=adapter,
        db=db,
        feature_store=feature_store,
        asset_class=asset_class,
    )

    async def _run_with_signals(coro) -> None:
        """Wrap a coroutine with asyncio-native signal handlers."""
        loop = asyncio.get_running_loop()

        def _stop(sig: signal.Signals) -> None:
            logger.info(f"Received signal {sig.name}, shutting down...")
            pipeline.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _stop, sig)

        await coro

    if backfill:
        # Historical backfill mode
        if not symbol:
            symbol = _get_symbols(asset_class)[0]

        end_dt = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
        start_dt = datetime.fromisoformat(start) if start else end_dt - timedelta(days=days)

        try:
            asyncio.run(_run_with_signals(pipeline.run_backfill(symbol, start_dt, end_dt)))
        except KeyboardInterrupt:
            pass
    else:
        # Live streaming mode
        symbols = _get_symbols(asset_class)
        try:
            asyncio.run(_run_with_signals(pipeline.run_stream(symbols)))
        except KeyboardInterrupt:
            pass

    db.close()


if __name__ == "__main__":
    main()
