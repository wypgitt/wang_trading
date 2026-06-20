"""Static reference data the bars hypertable does not carry.

The persisted ``bars`` row holds only ``symbol`` + ``bar_type`` (no display
name, no asset class), and the signal classes carry only an ``id`` (no
category/author/thesis). This module supplies that *reference* metadata —
display names, asset classes, primary bar type, and the per-family
descriptive copy — so ``/markets``, ``/symbols`` and ``/signals`` can render
human-legible cards.

This is reference data, not engine output: it describes instruments and
strategies, it never reports a *measured* value. Performance numbers
(price, sharpe, win-rate, …) always come from the engine, never from here.
"""

from __future__ import annotations

from typing import TypedDict


class Instrument(TypedDict):
    name: str
    asset_class: str  # equity | index | crypto | future
    bar_type: str     # primary BarType.value for this asset class


# Primary bar type per asset class: crypto trades on dollar bars, everything
# else on tick-imbalance bars (mirrors the client + config/settings bars block).
_TIB = "tib"
_DOLLAR = "dollar"


INSTRUMENTS: dict[str, Instrument] = {
    # Equities
    "NVDA": {"name": "NVIDIA Corp.", "asset_class": "equity", "bar_type": _TIB},
    "AAPL": {"name": "Apple Inc.", "asset_class": "equity", "bar_type": _TIB},
    "MSFT": {"name": "Microsoft Corp.", "asset_class": "equity", "bar_type": _TIB},
    "GOOGL": {"name": "Alphabet Inc.", "asset_class": "equity", "bar_type": _TIB},
    "AMZN": {"name": "Amazon.com Inc.", "asset_class": "equity", "bar_type": _TIB},
    "TSLA": {"name": "Tesla Inc.", "asset_class": "equity", "bar_type": _TIB},
    "META": {"name": "Meta Platforms", "asset_class": "equity", "bar_type": _TIB},
    "JPM": {"name": "JPMorgan Chase", "asset_class": "equity", "bar_type": _TIB},
    # Indexes
    "SPX": {"name": "S&P 500 Index", "asset_class": "index", "bar_type": _TIB},
    "NDX": {"name": "Nasdaq 100", "asset_class": "index", "bar_type": _TIB},
    "RUT": {"name": "Russell 2000", "asset_class": "index", "bar_type": _TIB},
    "VIX": {"name": "CBOE Volatility", "asset_class": "index", "bar_type": _TIB},
    # Crypto
    "BTC": {"name": "Bitcoin", "asset_class": "crypto", "bar_type": _DOLLAR},
    "ETH": {"name": "Ethereum", "asset_class": "crypto", "bar_type": _DOLLAR},
    "SOL": {"name": "Solana", "asset_class": "crypto", "bar_type": _DOLLAR},
    "AVAX": {"name": "Avalanche", "asset_class": "crypto", "bar_type": _DOLLAR},
    # Futures
    "ES": {"name": "E-mini S&P 500", "asset_class": "future", "bar_type": _TIB},
    "CL": {"name": "Crude Oil WTI", "asset_class": "future", "bar_type": _TIB},
    "GC": {"name": "Gold", "asset_class": "future", "bar_type": _TIB},
}


def instrument(symbol: str) -> Instrument | None:
    """Reference row for ``symbol`` (case-insensitive), or ``None`` if unknown."""

    return INSTRUMENTS.get(symbol.upper())


def all_symbols() -> list[str]:
    """The instrument universe, in declaration order."""

    return list(INSTRUMENTS.keys())


# ----------------------------------------------------------------------
# Signal-family descriptive metadata
# ----------------------------------------------------------------------
#
# Keyed by the family id the engine's ``create_default_battery`` registers.
# ``status`` is NOT stored here — it is derived at request time from the
# battery's dispatch ``kind`` (bars-runnable => live, otherwise dormant),
# so the surface reflects what the engine can actually run, not a guess.


class FamilyMeta(TypedDict):
    name: str
    category: str
    source: str
    thesis: str
    asset_classes: list[str]
    params: list[dict[str, str]]


FAMILY_META: dict[str, FamilyMeta] = {
    "ts_momentum": {
        "name": "Time-Series Momentum",
        "category": "Momentum",
        "source": "Clenow · Chan",
        "asset_classes": ["equity", "crypto", "future"],
        "params": [
            {"key": "lookbacks", "value": "21 / 63 / 126 / 252"},
            {"key": "history_window", "value": "252"},
            {"key": "vol_normalize", "value": "true"},
        ],
        "thesis": (
            "Per-asset momentum across multiple lookbacks, volatility-normalized "
            "to z-scores and weighted into a single conviction. Goes long winners "
            "/ short losers. Best in persistent trends."
        ),
    },
    "cs_momentum": {
        "name": "Cross-Sectional Momentum",
        "category": "Momentum",
        "source": "Jansen",
        "asset_classes": ["equity"],
        "params": [
            {"key": "lookback", "value": "252 bars"},
            {"key": "skip", "value": "21 bars"},
            {"key": "deciles", "value": "top 0.9 / bottom 0.1"},
        ],
        "thesis": (
            "12-month momentum with a 1-month skip to dodge short-term reversal. "
            "Ranks the cross-section, longs the top decile and shorts the bottom. "
            "Panel-relative alpha."
        ),
    },
    "mean_reversion": {
        "name": "Mean Reversion (O-U)",
        "category": "Mean Reversion",
        "source": "Chan",
        "asset_classes": ["equity", "index"],
        "params": [
            {"key": "entry_z", "value": "2.0"},
            {"key": "exit_z", "value": "0.5"},
            {"key": "half_life", "value": "1–100 bars"},
            {"key": "adf_pvalue", "value": "≤ 0.05"},
        ],
        "thesis": (
            "Fits an Ornstein-Uhlenbeck process; trades stationary series back "
            "toward the mean when the z-score breaches ±2σ. Half-life and ADF gate "
            "which names are tradeable."
        ),
    },
    "ma_crossover": {
        "name": "MA Crossover",
        "category": "Trend",
        "source": "Clenow",
        "asset_classes": ["equity", "future"],
        "params": [
            {"key": "fast", "value": "20 EMA"},
            {"key": "slow", "value": "50 EMA"},
            {"key": "triple_ma", "value": "on"},
        ],
        "thesis": (
            "Classic EMA crossover with optional triple-MA 2-of-3 voting. Slow, "
            "robust trend capture that complements the faster momentum sleeves."
        ),
    },
    "donchian_breakout": {
        "name": "Donchian Breakout",
        "category": "Trend",
        "source": "Clenow",
        "asset_classes": ["crypto", "future"],
        "params": [
            {"key": "entry_channel", "value": "55 bars"},
            {"key": "exit_channel", "value": "20 bars"},
            {"key": "sizing", "value": "ATR units"},
        ],
        "thesis": (
            "Turtle-style channel breakout: enter on a new 55-bar extreme, exit on "
            "the opposing 20-bar channel. ATR-normalized position sizing. Captures "
            "fat-tailed trends."
        ),
    },
    "vrp": {
        "name": "Volatility Risk Premium",
        "category": "Volatility",
        "source": "Sinclair",
        "asset_classes": ["index"],
        "params": [
            {"key": "vrp_lookback", "value": "30 bars"},
            {"key": "high_pct", "value": "75"},
            {"key": "low_pct", "value": "25"},
        ],
        "thesis": (
            "Trades the gap between implied and realized vol. Sells vol when VRP is "
            "rich, buys when cheap, and emits a regime modifier that scales the "
            "other families."
        ),
    },
    "futures_carry": {
        "name": "Futures Carry",
        "category": "Carry",
        "source": "Clenow",
        "asset_classes": ["future"],
        "params": [
            {"key": "annualize", "value": "true"},
            {"key": "conf_window", "value": "252"},
        ],
        "thesis": (
            "Roll-yield harvest from the front/back futures spread. Long "
            "backwardation (positive roll), short contango (roll drag). Slow, "
            "diversifying carry."
        ),
    },
    "funding_rate_arb": {
        "name": "Funding-Rate Arb",
        "category": "Carry",
        "source": "Crypto",
        "asset_classes": ["crypto"],
        "params": [
            {"key": "entry", "value": "10% annualized"},
            {"key": "exit", "value": "2% annualized"},
            {"key": "cadence", "value": "3×/day"},
        ],
        "thesis": (
            "Delta-neutral crypto carry: long spot, short perpetual, collect "
            "funding. Fires when annualized funding clears the entry threshold. "
            "High Sharpe, capacity-limited."
        ),
    },
    "stat_arb": {
        "name": "Statistical Arbitrage",
        "category": "Arbitrage",
        "source": "Chan",
        "asset_classes": ["equity"],
        "params": [
            {"key": "entry_z", "value": "2.0"},
            {"key": "exit_z", "value": "0.5"},
            {"key": "hedge", "value": "Kalman dynamic"},
        ],
        "thesis": (
            "Cointegration pairs trading with a Kalman-filtered dynamic hedge "
            "ratio. Trades the mean-reverting spread; Engle-Granger / Johansen gate "
            "the pair selection."
        ),
    },
    "cross_exchange_arb": {
        "name": "Cross-Exchange Arb",
        "category": "Arbitrage",
        "source": "Crypto",
        "asset_classes": ["crypto"],
        "params": [
            {"key": "min_spread", "value": "10 bps"},
            {"key": "fee_estimate", "value": "20 bps"},
        ],
        "thesis": (
            "Bar-level spot arbitrage across venues. Buys the cheap book, sells the "
            "rich one when the spread clears fees. Delta-neutral; depends on live "
            "venue keys."
        ),
    },
}


def family_meta(family_id: str) -> FamilyMeta | None:
    return FAMILY_META.get(family_id)


__all__ = [
    "FAMILY_META",
    "INSTRUMENTS",
    "FamilyMeta",
    "Instrument",
    "all_symbols",
    "family_meta",
    "instrument",
]
