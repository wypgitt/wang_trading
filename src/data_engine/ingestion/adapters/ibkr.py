"""Interactive Brokers market-data adapter for futures ingestion.

This is separate from the execution-layer ``IBKRBrokerAdapter``: it only
subscribes to quotes/trades and emits normalized ``Tick`` objects for bar
construction. The live order gate in the execution adapter is not involved.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from src.data_engine.ingestion.base_adapter import BaseAdapter
from src.data_engine.models import Side, Tick
from src.execution.ibkr_adapter import FuturesContractRegistry, contract_expiry

log = logging.getLogger(__name__)


class IBKRFuturesAdapter(BaseAdapter):
    """IBKR/TWS market-data adapter for futures symbols."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 11,
        registry: FuturesContractRegistry | None = None,
        poll_interval_s: float = 0.25,
    ) -> None:
        super().__init__("ibkr_futures")
        self.host = host
        self.port = port
        self.client_id = client_id
        self.registry = registry or FuturesContractRegistry()
        self.poll_interval_s = float(poll_interval_s)
        self._ib: Any = None
        self._contracts: dict[str, Any] = {}
        self._tickers: dict[str, Any] = {}
        self._subscribed: list[str] = []
        self._last_price: dict[str, float] = {}

    @property
    def ib(self) -> Any:
        if self._ib is None:
            try:
                from ib_insync import IB  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ib_insync not installed; pip install ib_insync") from exc
            self._ib = IB()
        return self._ib

    async def connect(self) -> None:
        if self._connected:
            return
        await asyncio.to_thread(
            self.ib.connect,
            self.host,
            self.port,
            clientId=self.client_id,
        )
        self._connected = True

    async def disconnect(self) -> None:
        if self._ib is not None and self._connected:
            await asyncio.to_thread(self._ib.disconnect)
        self._connected = False

    async def subscribe(self, symbols: list[str]) -> None:
        self._subscribed = list(symbols)
        for symbol in symbols:
            contract = self._contract(symbol)
            self._contracts[symbol] = contract
            ticker = await asyncio.to_thread(self.ib.reqMktData, contract, "", False, False)
            self._tickers[symbol] = ticker
        log.info("IBKR futures market-data subscribed: %s", symbols)

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        while self._connected:
            for symbol in list(self._subscribed):
                tick = self._tick_from_ticker(symbol, self._tickers.get(symbol))
                if tick is not None:
                    yield tick
            await asyncio.sleep(self.poll_interval_s)

    async def get_historical_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Tick]:
        contract = self._contract(symbol)
        duration_seconds = max(1, int((end - start).total_seconds()))
        duration = f"{duration_seconds} S"
        try:
            rows = await asyncio.to_thread(
                self.ib.reqHistoricalTicks,
                contract,
                start,
                end,
                min(1000, max(1, duration_seconds)),
                "TRADES",
                0,
                True,
            )
        except Exception:
            rows = []
        ticks = [self._historical_tick(symbol, row) for row in rows or []]
        ticks = [tick for tick in ticks if tick is not None]
        if ticks:
            return ticks

        # Fallback for IBKR installations that only allow historical bars.
        bars = await asyncio.to_thread(
            self.ib.reqHistoricalData,
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )
        return [
            Tick(
                timestamp=_coerce_dt(getattr(bar, "date", end)),
                symbol=symbol,
                price=float(getattr(bar, "close", 0.0) or 0.0),
                volume=float(getattr(bar, "volume", 0.0) or 0.0),
                side=Side.UNKNOWN,
                exchange="IBKR",
                trade_id=f"histbar-{uuid.uuid4()}",
            )
            for bar in bars or []
            if float(getattr(bar, "close", 0.0) or 0.0) > 0
        ]

    def _contract(self, symbol: str) -> Any:
        from ib_insync import Future  # type: ignore

        spec = self.registry.get_spec(symbol)
        base = FuturesContractRegistry._base_symbol(symbol)
        expiry = contract_expiry(
            symbol if symbol != base else self.registry.get_front_month(symbol)
        )
        return Future(
            symbol=base,
            lastTradeDateOrContractMonth=f"{expiry.year:04d}{expiry.month:02d}",
            exchange=spec["exchange"],
            currency=spec["currency"],
        )

    def _tick_from_ticker(self, symbol: str, ticker: Any) -> Tick | None:
        if ticker is None:
            return None
        price = (
            float(getattr(ticker, "last", 0.0) or 0.0)
            or _mid(getattr(ticker, "bid", 0.0), getattr(ticker, "ask", 0.0))
        )
        if price <= 0:
            return None
        prev = self._last_price.get(symbol)
        self._last_price[symbol] = price
        side = Side.UNKNOWN
        if prev is not None:
            side = Side.BUY if price > prev else Side.SELL if price < prev else Side.UNKNOWN
        size = float(
            getattr(ticker, "lastSize", 0.0)
            or getattr(ticker, "volume", 0.0)
            or 1.0
        )
        return Tick(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            price=price,
            volume=max(size, 1.0),
            side=side,
            exchange="IBKR",
            trade_id=f"live-{uuid.uuid4()}",
        )

    @staticmethod
    def _historical_tick(symbol: str, row: Any) -> Tick | None:
        price = float(getattr(row, "price", 0.0) or 0.0)
        if price <= 0:
            return None
        size = float(getattr(row, "size", 0.0) or 0.0)
        return Tick(
            timestamp=_coerce_dt(getattr(row, "time", None)),
            symbol=symbol,
            price=price,
            volume=max(size, 1.0),
            side=Side.UNKNOWN,
            exchange="IBKR",
            trade_id=str(getattr(row, "tickAttribLast", "") or uuid.uuid4()),
        )


def _mid(bid: Any, ask: Any) -> float:
    bid_f = float(bid or 0.0)
    ask_f = float(ask or 0.0)
    return (bid_f + ask_f) / 2.0 if bid_f > 0 and ask_f > 0 else 0.0


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
