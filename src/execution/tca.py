"""Post-trade Transaction Cost Analysis (Phase 5 / P5.06).

Narang §10.4: every executed order is analyzed for arrival-price slippage,
market impact, timing cost, and benchmark comparison vs TWAP/VWAP. Persistent
negative TCA against the benchmark indicates algo degradation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Iterable

import numpy as np
import pandas as pd

from src.execution.models import Order


@dataclass
class TCAResult:
    order_id: str
    symbol: str
    side: int
    arrival_price: float
    execution_price: float
    slippage_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    total_cost_bps: float
    commission: float
    algo_used: str
    execution_duration_seconds: float
    fill_rate: float
    benchmark_vs_twap_bps: float
    benchmark_vs_vwap_bps: float


def _vwap(prices: pd.Series, volumes: pd.Series) -> float:
    w = volumes.values.astype(float)
    total = w.sum()
    if total <= 0:
        return float(prices.mean()) if len(prices) else 0.0
    return float(np.dot(prices.values.astype(float), w) / total)


class TCAAnalyzer:
    """Compute TCA metrics for orders and roll them up for monitoring."""

    def analyze_order(
        self,
        order: Order,
        arrival_mid: float,
        market_prices_during: pd.Series,
        *,
        market_volumes_during: pd.Series | None = None,
    ) -> TCAResult:
        if arrival_mid <= 0:
            raise ValueError("arrival_mid must be positive")

        side = order.side
        if order.fills:
            filled_qty = sum(f.quantity for f in order.fills)
            exec_price = (
                sum(f.price * f.quantity for f in order.fills) / filled_qty
                if filled_qty
                else 0.0
            )
            commission = sum(f.commission for f in order.fills)
            first_ts = order.fills[0].timestamp
            last_ts = order.fills[-1].timestamp
            duration = max((last_ts - first_ts).total_seconds(), 0.0)
        else:
            exec_price = 0.0
            commission = 0.0
            duration = 0.0

        slippage_bps = side * (exec_price - arrival_mid) / arrival_mid * 10_000 \
            if exec_price > 0 else 0.0

        # Market impact: price drift from arrival to end of execution window
        if len(market_prices_during) >= 2:
            end_price = float(market_prices_during.iloc[-1])
            market_impact_bps = side * (end_price - arrival_mid) / arrival_mid * 10_000
        elif len(market_prices_during) == 1:
            end_price = float(market_prices_during.iloc[0])
            market_impact_bps = side * (end_price - arrival_mid) / arrival_mid * 10_000
        else:
            market_impact_bps = 0.0

        timing_cost_bps = slippage_bps - market_impact_bps
        total_cost_bps = slippage_bps  # slippage captures the realized cost

        # Benchmarks during the window
        if len(market_prices_during):
            twap = float(market_prices_during.mean())
            if market_volumes_during is not None and len(market_volumes_during):
                vwap = _vwap(market_prices_during, market_volumes_during)
            else:
                vwap = twap
            benchmark_vs_twap_bps = (
                side * (exec_price - twap) / twap * 10_000 if exec_price > 0 and twap > 0 else 0.0
            )
            benchmark_vs_vwap_bps = (
                side * (exec_price - vwap) / vwap * 10_000 if exec_price > 0 and vwap > 0 else 0.0
            )
        else:
            benchmark_vs_twap_bps = 0.0
            benchmark_vs_vwap_bps = 0.0

        fill_rate = (
            abs(order.filled_quantity) / abs(order.quantity)
            if order.quantity
            else 0.0
        )

        return TCAResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=side,
            arrival_price=arrival_mid,
            execution_price=exec_price,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            total_cost_bps=total_cost_bps,
            commission=commission,
            algo_used=order.execution_algo.value,
            execution_duration_seconds=duration,
            fill_rate=fill_rate,
            benchmark_vs_twap_bps=benchmark_vs_twap_bps,
            benchmark_vs_vwap_bps=benchmark_vs_vwap_bps,
        )

    def analyze_batch(
        self,
        orders: list[Order],
        arrival_prices: dict[str, float],
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run TCA on a batch of orders.

        `market_data` is a DataFrame with a 'price' column (and optionally
        'volume'), indexed per-symbol or with a 'symbol' column. For test
        convenience, we accept either a single price series (applied to all
        symbols) or a DataFrame keyed by symbol.
        """
        rows: list[dict] = []
        for order in orders:
            arrival = arrival_prices.get(order.symbol)
            if arrival is None:
                continue
            prices = self._prices_for(order.symbol, market_data)
            volumes = self._volumes_for(order.symbol, market_data)
            result = self.analyze_order(
                order, arrival, prices, market_volumes_during=volumes,
            )
            rows.append(asdict(result))
        return pd.DataFrame(rows)

    @staticmethod
    def _prices_for(symbol: str, market_data: pd.DataFrame) -> pd.Series:
        if isinstance(market_data, pd.Series):
            return market_data
        if "symbol" in market_data.columns:
            slice_ = market_data[market_data["symbol"] == symbol]
            return slice_["price"] if "price" in slice_.columns else pd.Series(dtype=float)
        if symbol in market_data.columns:
            return market_data[symbol]
        if "price" in market_data.columns:
            return market_data["price"]
        return pd.Series(dtype=float)

    @staticmethod
    def _volumes_for(symbol: str, market_data: pd.DataFrame) -> pd.Series | None:
        if isinstance(market_data, pd.Series):
            return None
        if "symbol" in market_data.columns:
            slice_ = market_data[market_data["symbol"] == symbol]
            return slice_["volume"] if "volume" in slice_.columns else None
        if "volume" in market_data.columns:
            return market_data["volume"]
        return None

    def get_tca_summary(self, results: list[TCAResult]) -> dict:
        if not results:
            return {
                "n_orders": 0, "mean_slippage_bps": 0.0, "mean_impact_bps": 0.0,
                "mean_timing_cost_bps": 0.0, "total_commission": 0.0,
                "cost_by_algo": {}, "flags": [],
            }

        slippages = [r.slippage_bps for r in results]
        impacts = [r.market_impact_bps for r in results]
        timing = [r.timing_cost_bps for r in results]
        commission = sum(r.commission for r in results)

        cost_by_algo: dict[str, dict[str, float]] = {}
        for r in results:
            bucket = cost_by_algo.setdefault(
                r.algo_used, {"n": 0, "slippage_sum": 0.0, "commission_sum": 0.0},
            )
            bucket["n"] += 1
            bucket["slippage_sum"] += r.slippage_bps
            bucket["commission_sum"] += r.commission
        for bucket in cost_by_algo.values():
            bucket["mean_slippage_bps"] = bucket["slippage_sum"] / bucket["n"]

        flags: list[str] = []
        twap_benches = [r.benchmark_vs_twap_bps for r in results]
        mean_vs_twap = fmean(twap_benches)
        if mean_vs_twap > 5.0:  # persistent underperformance vs TWAP
            flags.append(
                f"Persistent negative TCA vs TWAP: mean +{mean_vs_twap:.1f}bps"
            )

        return {
            "n_orders": len(results),
            "mean_slippage_bps": fmean(slippages),
            "mean_impact_bps": fmean(impacts),
            "mean_timing_cost_bps": fmean(timing),
            "total_commission": commission,
            "mean_vs_twap_bps": mean_vs_twap,
            "mean_vs_vwap_bps": fmean([r.benchmark_vs_vwap_bps for r in results]),
            "cost_by_algo": cost_by_algo,
            "flags": flags,
        }

    def detect_execution_degradation(
        self,
        recent_results: list[TCAResult],
        historical_results: list[TCAResult],
        threshold_std: float = 2.0,
    ) -> list[str]:
        if not recent_results or len(historical_results) < 2:
            return []

        hist_slippage = [r.slippage_bps for r in historical_results]
        hist_mean = fmean(hist_slippage)
        hist_std = pstdev(hist_slippage) if len(hist_slippage) > 1 else 0.0
        recent_mean = fmean(r.slippage_bps for r in recent_results)

        warnings: list[str] = []
        if hist_std > 0 and recent_mean > hist_mean + threshold_std * hist_std:
            warnings.append(
                f"Slippage degradation: recent mean {recent_mean:.1f}bps "
                f"> historical {hist_mean:.1f}bps + {threshold_std}σ ({hist_std:.1f}bps)"
            )
        elif hist_std == 0 and recent_mean > hist_mean * 2 and hist_mean > 0:
            warnings.append(
                f"Slippage doubled: {hist_mean:.1f}bps → {recent_mean:.1f}bps"
            )

        # Fill rate degradation
        hist_fill = [r.fill_rate for r in historical_results]
        recent_fill = [r.fill_rate for r in recent_results]
        if hist_fill and recent_fill:
            hist_fill_mean = fmean(hist_fill)
            recent_fill_mean = fmean(recent_fill)
            if recent_fill_mean < hist_fill_mean - 0.10:
                warnings.append(
                    f"Fill rate degradation: {hist_fill_mean:.1%} → {recent_fill_mean:.1%}"
                )
        return warnings
