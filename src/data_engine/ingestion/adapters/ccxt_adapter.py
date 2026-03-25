"""
CCXT Crypto Exchange Adapter

Unified adapter for crypto exchanges using the CCXT library.
Supports Binance, Coinbase, Kraken, Bybit, and 100+ other exchanges.

Used for:
- Real-time trade streaming
- Historical OHLCV backfill
- Multi-exchange data for cross-exchange arbitrage signals
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator, Optional

from loguru import logger

from src.data_engine.models import Tick, Side
from src.data_engine.ingestion.base_adapter import BaseAdapter


class CCXTAdapter(BaseAdapter):
    """
    CCXT-based crypto exchange adapter.

    Wraps ccxt's async exchange clients to provide a unified
    tick stream from any supported exchange.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        secret_key: str = "",
        testnet: bool = True,
    ):
        super().__init__(name=f"ccxt-{exchange_id}")
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._secret_key = secret_key
        self._testnet = testnet
        self._exchange = None
        self._symbols: list[str] = []

    async def connect(self) -> None:
        """Initialize the CCXT exchange client."""
        try:
            import ccxt.async_support as ccxt
        except ImportError:
            raise ImportError("Install ccxt: pip install ccxt")

        exchange_class = getattr(ccxt, self._exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {self._exchange_id}")

        config = {
            "apiKey": self._api_key,
            "secret": self._secret_key,
            "enableRateLimit": True,
        }

        if self._testnet:
            config["sandbox"] = True

        self._exchange = exchange_class(config)
        await self._exchange.load_markets()
        self._connected = True
        logger.info(f"Connected to {self._exchange_id} ({'testnet' if self._testnet else 'live'})")

    async def disconnect(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
        self._connected = False
        logger.info(f"Disconnected from {self._exchange_id}")

    async def subscribe(self, symbols: list[str]) -> None:
        """Set symbols for streaming."""
        self._symbols = symbols
        logger.info(f"Crypto symbols set: {symbols}")

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """
        Stream trades by polling watch_trades (if supported) or
        falling back to periodic fetch_trades.
        """
        if not self._exchange or not self._symbols:
            raise ConnectionError("Not connected or no symbols set")

        has_watch = self._exchange.has.get("watchTrades", False)

        if has_watch:
            async for tick in self._stream_websocket():
                yield tick
        else:
            async for tick in self._stream_polling():
                yield tick

    async def _stream_websocket(self) -> AsyncIterator[Tick]:
        """Stream via WebSocket (preferred)."""
        while self._connected:
            try:
                for symbol in self._symbols:
                    trades = await self._exchange.watch_trades(symbol)
                    for trade in trades:
                        yield self._parse_trade(trade, symbol)
            except Exception as e:
                logger.error(f"WebSocket stream error: {e}")
                await asyncio.sleep(5)

    async def _stream_polling(self) -> AsyncIterator[Tick]:
        """Fallback: poll fetch_trades periodically."""
        last_ids: dict[str, str] = {}

        while self._connected:
            for symbol in self._symbols:
                try:
                    trades = await self._exchange.fetch_trades(symbol, limit=100)
                    for trade in trades:
                        trade_id = str(trade.get("id", ""))
                        if trade_id != last_ids.get(symbol):
                            yield self._parse_trade(trade, symbol)
                    if trades:
                        last_ids[symbol] = str(trades[-1].get("id", ""))
                except Exception as e:
                    logger.warning(f"Polling error for {symbol}: {e}")

            await asyncio.sleep(1)  # 1s polling interval

    async def get_historical_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Tick]:
        """
        Fetch historical trades. Falls back to OHLCV if trades
        are not available historically.
        """
        if not self._exchange:
            raise ConnectionError("Not connected")

        ticks: list[Tick] = []
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        try:
            while since < end_ms:
                trades = await self._exchange.fetch_trades(
                    symbol, since=since, limit=1000
                )
                if not trades:
                    break

                for trade in trades:
                    ts = trade.get("timestamp", 0)
                    if ts > end_ms:
                        return ticks
                    ticks.append(self._parse_trade(trade, symbol))

                since = trades[-1]["timestamp"] + 1
                await asyncio.sleep(self._exchange.rateLimit / 1000)

        except Exception as e:
            logger.error(f"Historical fetch error for {symbol}: {e}")

        logger.info(f"Fetched {len(ticks)} historical trades for {symbol}")
        return ticks

    def _parse_trade(self, trade: dict, symbol: str) -> Tick:
        """Parse a CCXT trade dict into a Tick."""
        ts = trade.get("timestamp", 0)
        side_str = trade.get("side", "")

        if side_str == "buy":
            side = Side.BUY
        elif side_str == "sell":
            side = Side.SELL
        else:
            side = Side.UNKNOWN

        return Tick(
            timestamp=datetime.utcfromtimestamp(ts / 1000) if ts else datetime.utcnow(),
            symbol=symbol,
            price=float(trade.get("price", 0)),
            volume=float(trade.get("amount", 0)),
            side=side,
            exchange=self._exchange_id,
            trade_id=str(trade.get("id", "")),
        )
