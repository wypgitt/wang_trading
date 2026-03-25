"""
Alpaca Market Data Adapter

Connects to Alpaca's WebSocket stream for real-time trade data
and REST API for historical backfill. Supports both IEX (free)
and SIP (paid) data feeds.

Alpaca is the primary data source for US equities in Phase 1.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import aiohttp
from loguru import logger

from src.data_engine.models import Tick, Side
from src.data_engine.ingestion.base_adapter import BaseAdapter


class AlpacaAdapter(BaseAdapter):
    """
    Alpaca market data adapter for US equities.

    Supports:
    - Real-time trade streaming via WebSocket
    - Historical trade data via REST API
    - Automatic reconnection on disconnect
    """

    # Alpaca WebSocket URLs
    WS_URL_IEX = "wss://stream.data.alpaca.markets/v2/iex"
    WS_URL_SIP = "wss://stream.data.alpaca.markets/v2/sip"

    # REST API
    REST_URL = "https://data.alpaca.markets/v2"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: str = "iex",
    ):
        super().__init__(name="alpaca")
        self._api_key = api_key
        self._secret_key = secret_key
        self._feed = feed
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._symbols: list[str] = []
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=10_000)

    async def connect(self) -> None:
        """Connect to Alpaca WebSocket."""
        self._session = aiohttp.ClientSession()

        ws_url = self.WS_URL_SIP if self._feed == "sip" else self.WS_URL_IEX
        logger.info(f"Connecting to Alpaca WebSocket ({self._feed})...")

        self._ws = await self._session.ws_connect(ws_url)

        # Authenticate
        auth_msg = {
            "action": "auth",
            "key": self._api_key,
            "secret": self._secret_key,
        }
        await self._ws.send_json(auth_msg)

        # Wait for auth response
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                for item in data:
                    if item.get("T") == "success" and item.get("msg") == "authenticated":
                        logger.info("Alpaca WebSocket authenticated")
                        self._connected = True
                        return
                    elif item.get("T") == "error":
                        raise ConnectionError(f"Alpaca auth failed: {item}")
                break

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._connected = False
        logger.info("Alpaca disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time trades for given symbols."""
        if not self._ws:
            raise ConnectionError("Not connected")

        self._symbols = symbols
        subscribe_msg = {
            "action": "subscribe",
            "trades": symbols,
        }
        await self._ws.send_json(subscribe_msg)
        logger.info(f"Subscribed to trades: {symbols}")

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """
        Yield ticks from the WebSocket stream.

        Handles reconnection automatically on disconnect.
        """
        if not self._ws:
            raise ConnectionError("Not connected")

        while True:
            try:
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        for item in data:
                            if item.get("T") == "t":  # trade message
                                tick = self._parse_trade(item)
                                if tick:
                                    yield tick

                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.warning(f"WebSocket closed/error: {msg.type}")
                        break

            except Exception as e:
                logger.error(f"Stream error: {e}")

            # Reconnect
            if self._symbols:
                logger.info("Attempting reconnection in 5s...")
                await asyncio.sleep(5)
                try:
                    await self.connect()
                    await self.subscribe(self._symbols)
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
                    await asyncio.sleep(30)

    async def get_historical_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Tick]:
        """
        Fetch historical trade data via REST API.

        Alpaca's v2 API returns trades paginated by page_token.
        This method handles pagination to fetch all trades in range.
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        ticks: list[Tick] = []
        page_token = None

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

        while True:
            params = {
                "start": start.isoformat() + "Z",
                "end": end.isoformat() + "Z",
                "feed": self._feed,
                "limit": 10_000,
            }
            if page_token:
                params["page_token"] = page_token

            url = f"{self.REST_URL}/stocks/{symbol}/trades"

            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Alpaca REST error {resp.status}: {error_text}")
                    break

                data = await resp.json()
                trades = data.get("trades", [])

                for trade in trades:
                    tick = Tick(
                        timestamp=datetime.fromisoformat(
                            trade["t"].replace("Z", "+00:00")
                        ),
                        symbol=symbol,
                        price=float(trade["p"]),
                        volume=float(trade["s"]),
                        side=Side.UNKNOWN,  # will be classified by tick rule
                        exchange=trade.get("x", ""),
                        trade_id=str(trade.get("i", "")),
                    )
                    ticks.append(tick)

                page_token = data.get("next_page_token")
                if not page_token or not trades:
                    break

            logger.debug(f"Fetched {len(ticks)} historical trades for {symbol}")

        logger.info(
            f"Historical backfill: {len(ticks)} trades for {symbol} "
            f"({start.date()} to {end.date()})"
        )
        return ticks

    def _parse_trade(self, data: dict) -> Optional[Tick]:
        """Parse an Alpaca WebSocket trade message into a Tick."""
        try:
            return Tick(
                timestamp=datetime.fromisoformat(
                    data["t"].replace("Z", "+00:00")
                ),
                symbol=data["S"],
                price=float(data["p"]),
                volume=float(data["s"]),
                side=Side.UNKNOWN,
                exchange=data.get("x", ""),
                trade_id=str(data.get("i", "")),
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse trade: {e} — data: {data}")
            return None
