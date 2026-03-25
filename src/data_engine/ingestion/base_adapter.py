"""
Base adapter interface for market data ingestion.

All exchange/broker adapters implement this interface, providing
a unified tick stream regardless of the data source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Optional, Callable

from src.data_engine.models import Tick


class BaseAdapter(ABC):
    """
    Abstract base for market data adapters.

    Adapters handle:
    - Connection management (connect/disconnect)
    - Data normalization (exchange format → Tick)
    - Reconnection on failure
    - Historical data backfill
    """

    def __init__(self, name: str):
        self.name = name
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly disconnect from the data source."""
        ...

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time trade data for given symbols."""
        ...

    @abstractmethod
    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """Yield ticks as they arrive from the data source."""
        ...

    @abstractmethod
    async def get_historical_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Tick]:
        """Fetch historical tick data for backfill."""
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
