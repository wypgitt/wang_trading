"""Broker factory + smart order router (P6.04).

``BrokerFactory`` picks the right adapter for a symbol/asset-class. Adapters
are cached so the rest of the system always sees the same Alpaca/CCXT/IBKR
instance for the same asset class.

``SmartOrderRouter`` distributes crypto orders across multiple CCXT exchanges
to take advantage of price differentials or bigger aggregated depth.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from src.execution.broker_adapter import (
    AlpacaBrokerAdapter,
    BaseBrokerAdapter,
    CCXTBrokerAdapter,
)
from src.execution.circuit_breakers import _is_crypto
from src.execution.ibkr_adapter import FuturesContractRegistry, IBKRBrokerAdapter
from src.execution.models import Order

log = logging.getLogger(__name__)

# Futures root symbols we know about out of the box — anything else that
# doesn't look like crypto falls through to equities.
_DEFAULT_FUTURES_ROOTS: frozenset[str] = frozenset({
    "ES", "NQ", "CL", "GC", "ZN", "ZC", "RTY", "YM", "SI", "HG",
})


def _strip_contract_suffix(symbol: str) -> str:
    """``ESZ25`` → ``ES``. Leaves plain equity tickers alone."""
    s = symbol.upper()
    tail = s.rstrip("0123456789")
    if tail and tail[-1] in "FGHJKMNQUVXZ" and len(tail) < len(s):
        return tail[:-1] or s
    return s


# ── Broker factory ────────────────────────────────────────────────────────

class BrokerFactory:
    """Maps (symbol, asset_class) → broker adapter. Caches singletons."""

    ASSET_EQUITIES = "equities"
    ASSET_CRYPTO = "crypto"
    ASSET_FUTURES = "futures"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        futures_roots: Iterable[str] | None = None,
    ) -> None:
        self.config = config or {}
        self._cache: dict[str, BaseBrokerAdapter] = {}
        self._futures_roots: set[str] = set(futures_roots) if futures_roots else set(
            _DEFAULT_FUTURES_ROOTS
        )
        # Extend futures roots from loaded registry if possible.
        try:
            for sym in FuturesContractRegistry().symbols():
                self._futures_roots.add(sym)
        except Exception:
            pass

    # ── Classification ────────────────────────────────────────────────

    def classify(self, symbol: str) -> str:
        if _is_crypto(symbol):
            return self.ASSET_CRYPTO
        if _strip_contract_suffix(symbol) in self._futures_roots:
            return self.ASSET_FUTURES
        return self.ASSET_EQUITIES

    # ── Resolution ────────────────────────────────────────────────────

    def get_broker(
        self, symbol: str, asset_class: str | None = None,
    ) -> BaseBrokerAdapter:
        klass = asset_class or self.classify(symbol)
        if klass == self.ASSET_CRYPTO:
            return self._get_or_create("crypto", self._build_crypto)
        if klass == self.ASSET_FUTURES:
            return self._get_or_create("futures", self._build_ibkr)
        return self._get_or_create("equities", self._build_alpaca)

    def _get_or_create(self, key: str, builder) -> BaseBrokerAdapter:
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def _build_alpaca(self) -> AlpacaBrokerAdapter:
        cfg = self.config.get("alpaca", {}) or {}
        return AlpacaBrokerAdapter(
            api_key=cfg.get("api_key", ""),
            secret_key=cfg.get("secret_key", cfg.get("api_secret", "")),
            paper=bool(cfg.get("paper", True)),
            base_url=cfg.get("base_url"),
        )

    def _build_crypto(self) -> CCXTBrokerAdapter:
        # Prefer the first configured crypto broker. Defaults to binance.
        for name in ("binance", "coinbase", "kraken", "bybit"):
            cfg = self.config.get(name)
            if cfg:
                return CCXTBrokerAdapter(
                    exchange_name=name,
                    api_key=cfg.get("api_key", ""),
                    secret_key=cfg.get("secret_key", cfg.get("api_secret", "")),
                    sandbox=bool(cfg.get("sandbox", True)),
                    passphrase=cfg.get("passphrase"),
                    options=cfg.get("options"),
                )
        return CCXTBrokerAdapter(exchange_name="binance", api_key="", secret_key="")

    def _build_ibkr(self) -> IBKRBrokerAdapter:
        cfg = self.config.get("ibkr", {}) or {}
        return IBKRBrokerAdapter(
            host=cfg.get("host", "127.0.0.1"),
            port=cfg.get("port"),
            client_id=int(cfg.get("client_id", 1)),
            account_id=cfg.get("account_id", ""),
            live=bool(cfg.get("live", False)),
            allow_extended=bool(cfg.get("allow_extended", False)),
        )

    # ── Aggregate operations ──────────────────────────────────────────

    def get_all_brokers(self) -> dict[str, BaseBrokerAdapter]:
        return dict(self._cache)

    async def heartbeat_all(self) -> dict[str, bool]:
        if not self._cache:
            return {}
        names = list(self._cache.keys())
        results = await asyncio.gather(
            *(self._cache[n].heartbeat() for n in names),
            return_exceptions=True,
        )
        return {
            n: (bool(r) if not isinstance(r, Exception) else False)
            for n, r in zip(names, results)
        }

    async def shutdown_all(self) -> None:
        for name, broker in list(self._cache.items()):
            close = getattr(broker, "close", None)
            disconnect = getattr(getattr(broker, "ib", None), "disconnect", None)
            try:
                if callable(close):
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
                elif callable(disconnect):
                    await asyncio.to_thread(disconnect)
            except Exception as exc:  # pragma: no cover
                log.warning("shutdown failed for %s: %s", name, exc)
        self._cache.clear()


# ── Smart order router ────────────────────────────────────────────────────

class SmartOrderRouter:
    """Routes crypto orders across multiple CCXT venues for best execution."""

    def __init__(self, exchanges: list[CCXTBrokerAdapter]) -> None:
        if not exchanges:
            raise ValueError("SmartOrderRouter requires at least one exchange")
        self.exchanges = list(exchanges)

    def _exchange_name(self, broker: CCXTBrokerAdapter) -> str:
        return broker.exchange_name

    async def _quotes(self, symbol: str) -> list[tuple[str, dict[str, float]]]:
        names = [self._exchange_name(b) for b in self.exchanges]
        results = await asyncio.gather(
            *(b.get_quote(symbol) for b in self.exchanges),
            return_exceptions=True,
        )
        out: list[tuple[str, dict[str, float]]] = []
        for n, r in zip(names, results):
            if isinstance(r, Exception):
                log.warning("quote failed on %s: %s", n, r)
                continue
            out.append((n, r))
        return out

    async def get_best_quote(self, symbol: str, side: int) -> tuple[str, float]:
        """Return ``(exchange_name, price)`` for best-fill direction.

        ``side > 0`` (buy) → lowest ask. ``side < 0`` (sell) → highest bid.
        """
        quotes = await self._quotes(symbol)
        if not quotes:
            raise RuntimeError(f"No quotes available for {symbol!r}")
        if side > 0:
            best = min(quotes, key=lambda kv: kv[1].get("ask") or float("inf"))
            return best[0], float(best[1]["ask"])
        best = max(quotes, key=lambda kv: kv[1].get("bid") or float("-inf"))
        return best[0], float(best[1]["bid"])

    async def get_aggregated_depth(self, symbol: str, levels: int = 10) -> pd.DataFrame:
        """Combined order book across exchanges as a DataFrame with columns
        ``[exchange, side, price, size]`` sorted best-price-first per side."""
        async def _ob(broker):
            return await asyncio.to_thread(lambda: None) or await broker.client.fetch_order_book(
                broker.normalize_symbol(symbol), levels,
            )

        results = await asyncio.gather(
            *(_ob(b) for b in self.exchanges),
            return_exceptions=True,
        )
        rows: list[dict[str, Any]] = []
        for broker, r in zip(self.exchanges, results):
            if isinstance(r, Exception):
                continue
            name = self._exchange_name(broker)
            for p, sz in (r.get("bids") or [])[:levels]:
                rows.append({"exchange": name, "side": "bid", "price": float(p), "size": float(sz)})
            for p, sz in (r.get("asks") or [])[:levels]:
                rows.append({"exchange": name, "side": "ask", "price": float(p), "size": float(sz)})
        if not rows:
            return pd.DataFrame(columns=["exchange", "side", "price", "size"])
        df = pd.DataFrame(rows)
        # Sort: asks ascending, bids descending; stable within side.
        df = pd.concat([
            df[df.side == "bid"].sort_values("price", ascending=False),
            df[df.side == "ask"].sort_values("price", ascending=True),
        ], ignore_index=True)
        return df

    async def route_order(self, order: Order) -> list[Order]:
        """Split ``order`` across exchanges to minimize slippage.

        Strategy:
          • For each exchange, compute available depth at or inside our limit.
          • Walk venues from best-priced first, consuming depth until the
            order is fully allocated.
          • Submit a child order per venue used.
        """
        quotes = await self._quotes(order.symbol)
        if not quotes:
            raise RuntimeError(f"No quotes available for {order.symbol!r}")

        # Rank exchanges by execution price (buy→ask asc, sell→bid desc)
        if order.side > 0:
            quotes.sort(key=lambda kv: kv[1].get("ask") or float("inf"))
            depth_field = "asks"
        else:
            quotes.sort(key=lambda kv: -(kv[1].get("bid") or 0.0))
            depth_field = "bids"

        remaining = abs(order.quantity)
        children: list[Order] = []
        name_to_broker = {self._exchange_name(b): b for b in self.exchanges}

        for name, _q in quotes:
            if remaining <= 0:
                break
            broker = name_to_broker[name]
            try:
                ob = await broker.client.fetch_order_book(
                    broker.normalize_symbol(order.symbol), 10,
                )
            except Exception:
                continue
            depth = sum(float(size) for _, size in (ob.get(depth_field) or []))
            if depth <= 0:
                continue
            slice_qty = min(depth, remaining)
            signed = slice_qty if order.side > 0 else -slice_qty
            child = Order(
                order_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=signed,
                limit_price=order.limit_price,
                execution_algo=order.execution_algo,
                parent_order_id=order.order_id,
                signal_family=order.signal_family,
            )
            child = await broker.submit_order(child)
            children.append(child)
            remaining -= slice_qty
        return children
