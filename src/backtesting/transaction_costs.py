"""
Transaction cost model (Johnson — Algorithmic Trading & DMA).

Models the all-in cost of a trade as the sum of four components:

    total = commission + spread_cost + slippage + market_impact

Market impact uses Johnson's square-root model:

    impact_per_share ∝ σ · sqrt(order_size / ADV)

which captures the empirical observation that impact grows sub-linearly
with participation rate. Multiplying by ``price * order_size`` converts
the per-share impact into a dollar cost on the full order.

Design-doc §9.4 (realistic transaction costs) and §10.1 (market impact).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    """Breakdown of a single trade's estimated execution cost (all in $)."""

    commission: float
    spread_cost: float
    slippage: float
    market_impact: float
    total_cost: float
    cost_bps: float


_REQUIRED_KEYS = {
    "commission_per_share",
    "min_commission",
    "spread_bps",
    "slippage_bps",
    "impact_coefficient",
}


class TransactionCostModel:
    """Estimates $ cost of executing a single trade given per-asset-class params.

    Parameters are supplied as dicts keyed by asset class. Each dict must contain:

        commission_per_share  — $/share (or $/contract for futures)
        min_commission        — floor applied to commission
        spread_bps            — half-spread cost in bps of notional
        slippage_bps          — execution slippage in bps of notional
        impact_coefficient    — multiplier on Johnson's sqrt impact term
    """

    def __init__(
        self,
        equities_config: dict | None = None,
        crypto_config: dict | None = None,
        futures_config: dict | None = None,
    ) -> None:
        self._configs: dict[str, dict] = {}
        if equities_config is not None:
            self._configs["equities"] = self._validate(equities_config)
        if crypto_config is not None:
            self._configs["crypto"] = self._validate(crypto_config)
        if futures_config is not None:
            self._configs["futures"] = self._validate(futures_config)

    @staticmethod
    def _validate(cfg: dict) -> dict:
        missing = _REQUIRED_KEYS - cfg.keys()
        if missing:
            raise ValueError(f"cost config missing required keys: {sorted(missing)}")
        return cfg

    def estimate(
        self,
        order_size: float,
        price: float,
        adv: float,
        volatility: float,
        asset_class: str = "equities",
    ) -> CostEstimate:
        if asset_class not in self._configs:
            raise ValueError(
                f"no cost config for asset_class={asset_class!r}; "
                f"configured: {sorted(self._configs)}"
            )
        if price <= 0:
            raise ValueError("price must be positive")
        if adv <= 0:
            raise ValueError("adv must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        cfg = self._configs[asset_class]
        qty = abs(order_size)
        notional = price * qty

        commission = max(cfg["commission_per_share"] * qty, cfg["min_commission"])
        spread_cost = (cfg["spread_bps"] / 10_000.0) * notional
        slippage = (cfg["slippage_bps"] / 10_000.0) * notional

        participation = qty / adv
        market_impact = (
            cfg["impact_coefficient"]
            * volatility
            * math.sqrt(participation)
            * notional
        )

        total_cost = commission + spread_cost + slippage + market_impact
        cost_bps = (total_cost / notional) * 10_000.0 if notional > 0 else 0.0

        return CostEstimate(
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
            market_impact=market_impact,
            total_cost=total_cost,
            cost_bps=cost_bps,
        )


def estimate_round_trip(
    entry_size: float,
    entry_price: float,
    exit_price: float,
    adv: float,
    volatility: float,
    asset_class: str = "equities",
    model: TransactionCostModel | None = None,
) -> CostEstimate:
    """Total cost of entering and exiting a position (two legs).

    The ``model`` argument lets callers inject a custom-configured model;
    when omitted, the module-level :data:`DEFAULT_MODEL` (wired to the
    pre-canned EQUITIES/CRYPTO/FUTURES configs) is used.
    """

    cost_model = model if model is not None else DEFAULT_MODEL
    entry = cost_model.estimate(entry_size, entry_price, adv, volatility, asset_class)
    exit_ = cost_model.estimate(entry_size, exit_price, adv, volatility, asset_class)

    total = entry.total_cost + exit_.total_cost
    avg_notional = 0.5 * (
        abs(entry_size) * entry_price + abs(entry_size) * exit_price
    )
    cost_bps = (total / avg_notional) * 10_000.0 if avg_notional > 0 else 0.0

    return CostEstimate(
        commission=entry.commission + exit_.commission,
        spread_cost=entry.spread_cost + exit_.spread_cost,
        slippage=entry.slippage + exit_.slippage,
        market_impact=entry.market_impact + exit_.market_impact,
        total_cost=total,
        cost_bps=cost_bps,
    )


EQUITIES_COSTS: dict = {
    "commission_per_share": 0.005,
    "min_commission": 1.0,
    "spread_bps": 2.0,
    "slippage_bps": 1.0,
    "impact_coefficient": 0.1,
}

CRYPTO_COSTS: dict = {
    "commission_per_share": 0.001,
    "min_commission": 0.0,
    "spread_bps": 3.0,
    "slippage_bps": 2.0,
    "impact_coefficient": 0.15,
}

FUTURES_COSTS: dict = {
    "commission_per_share": 1.25,
    "min_commission": 1.25,
    "spread_bps": 1.0,
    "slippage_bps": 0.5,
    "impact_coefficient": 0.08,
}


DEFAULT_MODEL = TransactionCostModel(
    equities_config=EQUITIES_COSTS,
    crypto_config=CRYPTO_COSTS,
    futures_config=FUTURES_COSTS,
)
