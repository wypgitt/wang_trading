"""
Signal Battery Orchestrator

Runs every registered signal generator on the same bars + context, filters
the output to CUSUM-triggered event timestamps, and returns a flat
DataFrame for downstream consumption by the Labeling Engine / meta-labeler.

Conflict resolution is DELIBERATELY absent: when multiple families fire at
the same (timestamp, symbol), all signals pass through. The meta-labeler
(Phase 3) decides which to act on. The orchestrator's only job is
fan-out generation, input routing, and bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from loguru import logger

from src.signal_battery.base_signal import BaseSignalGenerator, Signal


# ---------------------------------------------------------------------------
# Registration model
# ---------------------------------------------------------------------------

# Each generator has a ``kind`` that tells the orchestrator how to call it:
#
#   "bars"            : generate(bars, symbol=...)           — default
#   "panel"           : generate(panel=...)                  — CS momentum
#   "pair"            : generate(y_series=..., x_series=...) — stat-arb
#   "exchange_prices" : generate(exchange_prices_df, symbol=..) — xchg arb
#   "bars_extra"      : generate(extra_df, symbol=...)
#                       where extra_df comes from ``context[<context_key>]``
#
# "bars_extra" covers everything that expects a DataFrame shaped like bars
# but with different columns (funding_rate, front/back prices, iv/rv, ...),
# avoiding a combinatorial explosion of kinds.

_KINDS = {"bars", "panel", "pair", "exchange_prices", "bars_extra"}


@dataclass
class _Registration:
    name: str
    generator: BaseSignalGenerator
    kind: str
    context_key: str | None = None  # used when kind == "bars_extra"
    # Optional adapter: (bars, context, symbol) -> list[Signal] override.
    adapter: Callable | None = None


# ---------------------------------------------------------------------------
# SignalBattery
# ---------------------------------------------------------------------------

class SignalBattery:
    """
    Registry + runner for signal generators.

    Usage:
        battery = SignalBattery()
        battery.register(TimeSeriesMomentumSignal(), kind="bars")
        battery.register(
            FuturesCarrySignal(), kind="bars_extra", context_key="futures_curve"
        )
        df = battery.generate_all(
            bars=equity_bars,
            event_timestamps=cusum_events,
            symbol="AAPL",
            multi_asset_prices=panel,           # consumed by "panel" gens
            exchange_prices=binance_coinbase_df, # consumed by "exchange_prices"
            stat_arb_pair=(y, x),
            futures_curve=front_back_df,
        )
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config) if config else {}
        self._registry: list[_Registration] = []

    # ------------------------------------------------------------------ API --

    def register(
        self,
        generator: BaseSignalGenerator,
        kind: str = "bars",
        context_key: str | None = None,
        adapter: Callable | None = None,
    ) -> None:
        """
        Add a generator to the battery.

        Args:
            generator:   Instance of a subclass of ``BaseSignalGenerator``.
            kind:        One of {"bars", "panel", "pair", "exchange_prices",
                         "bars_extra"}. See module docstring.
            context_key: Required when kind == "bars_extra"; names the key in
                         the ``**context`` kwargs to pull the DataFrame from.
            adapter:     Optional custom callable
                         ``(bars, context, symbol) -> list[Signal]``. When
                         provided, ``kind`` is ignored.
        """
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        if kind == "bars_extra" and not context_key and adapter is None:
            raise ValueError("kind='bars_extra' requires context_key or adapter")
        self._registry.append(
            _Registration(
                name=generator.name,
                generator=generator,
                kind=kind,
                context_key=context_key,
                adapter=adapter,
            )
        )

    def get_active_families(self) -> list[str]:
        """List the names of all registered signal generators."""
        return [r.name for r in self._registry]

    def generate_all(
        self,
        bars: pd.DataFrame,
        event_timestamps: pd.DatetimeIndex | list | None = None,
        symbol: str = "UNKNOWN",
        **context: Any,
    ) -> pd.DataFrame:
        """
        Run every registered generator and return a flat DataFrame of signals.

        Args:
            bars:             Primary single-asset bars DataFrame.
            event_timestamps: If provided, output signals are filtered to
                              these timestamps only (CUSUM events).
                              Pass None/empty to keep everything.
            symbol:           Stamped onto generators that don't emit their
                              own symbol metadata.
            **context:        Extra inputs routed to non-"bars" generators
                              (multi_asset_prices, exchange_prices,
                              stat_arb_pair, plus any context_key keys).

        Returns:
            DataFrame with columns: timestamp, symbol, family, side,
            confidence, plus one column per metadata key encountered
            (prefixed ``meta_<key>``). Empty DataFrame (with correct schema)
            when nothing fires.
        """
        all_signals: list[Signal] = []

        for reg in self._registry:
            try:
                sigs = self._dispatch(reg, bars=bars, symbol=symbol, context=context)
            except Exception as exc:  # noqa: BLE001
                # A generator crashing must not take the battery down; log
                # and skip. The meta-labeler will just see fewer signals.
                logger.warning(
                    f"SignalBattery: generator {reg.name!r} raised {exc!r}; skipping"
                )
                continue
            all_signals.extend(sigs)

        # Event-filter semantics:
        #   None       → keep all signals (no filtering)
        #   []         → keep none (explicit empty event list)
        #   [ts, ...]  → keep only signals whose timestamp is in the set
        if event_timestamps is not None:
            # Convert each Timestamp to a plain datetime via .to_pydatetime()
            # on the element, not the index (DatetimeIndex.to_pydatetime()
            # is missing from pandas-stubs even though it exists at runtime).
            event_set: set = {
                ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                for ts in pd.DatetimeIndex(event_timestamps)
            }
            all_signals = [s for s in all_signals if s.timestamp in event_set]

        return self._to_dataframe(all_signals)

    def get_signal_stats(self, df: pd.DataFrame | None = None) -> dict:
        """
        Per-family summary statistics.

        Args:
            df: Optional DataFrame as returned by ``generate_all``. If None,
                returns an empty-stats dict (callers usually pass the result
                they already have).

        Returns:
            dict mapping family → {count, avg_confidence, long_ratio,
            short_ratio, neutral_ratio}. Empty dict if ``df`` is empty/None.
        """
        if df is None or df.empty:
            return {}
        stats: dict[str, dict] = {}
        for family, group in df.groupby("family"):
            n = len(group)
            sides = group["side"].to_numpy()
            stats[family] = {
                "count": int(n),
                "avg_confidence": float(group["confidence"].mean()),
                "long_ratio": float((sides == 1).sum() / n),
                "short_ratio": float((sides == -1).sum() / n),
                "neutral_ratio": float((sides == 0).sum() / n),
            }
        return stats

    # ------------------------------------------------------------------ helpers --

    def _dispatch(
        self,
        reg: _Registration,
        *,
        bars: pd.DataFrame,
        symbol: str,
        context: dict[str, Any],
    ) -> list[Signal]:
        """Call a single generator according to its ``kind``."""
        if reg.adapter is not None:
            return list(reg.adapter(bars, context, symbol) or [])

        gen = reg.generator
        if reg.kind == "bars":
            return list(gen.generate(bars, symbol=symbol))

        if reg.kind == "panel":
            panel = context.get("multi_asset_prices")
            if not panel:
                return []
            # Allow an optional per-call rebalance timestamp.
            ts = context.get("panel_timestamp")
            return list(gen.generate(panel=panel, timestamp=ts))

        if reg.kind == "pair":
            pair = context.get("stat_arb_pair")
            if not pair:
                return []
            y_series, x_series = pair
            y_symbol = context.get("pair_y_symbol", "Y")
            x_symbol = context.get("pair_x_symbol", "X")
            return list(
                gen.generate(
                    y_series=y_series,
                    x_series=x_series,
                    y_symbol=y_symbol,
                    x_symbol=x_symbol,
                )
            )

        if reg.kind == "exchange_prices":
            px = context.get("exchange_prices")
            if px is None or px.empty:
                return []
            return list(gen.generate(px, symbol=symbol))

        if reg.kind == "bars_extra":
            if reg.context_key is None:
                raise RuntimeError(
                    f"bars_extra registration {reg.name!r} missing context_key"
                )
            extra = context.get(reg.context_key)
            if extra is None or (hasattr(extra, "empty") and extra.empty):
                return []
            return list(gen.generate(extra, symbol=symbol))

        raise RuntimeError(f"unknown kind {reg.kind!r}")

    @staticmethod
    def _to_dataframe(signals: list[Signal]) -> pd.DataFrame:
        """Flatten a list of Signals to a tidy DataFrame."""
        base_cols = ["timestamp", "symbol", "family", "side", "confidence"]
        if not signals:
            return pd.DataFrame(columns=base_cols)

        rows: list[dict] = []
        for s in signals:
            row = {
                "timestamp": s.timestamp,
                "symbol": s.symbol,
                "family": s.family,
                "side": s.side,
                "confidence": s.confidence,
            }
            # Flatten metadata with a prefix so keys don't collide with base.
            for k, v in s.metadata.items():
                row[f"meta_{k}"] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        # Always keep the base columns first, metadata after.
        meta_cols = [c for c in df.columns if c not in base_cols]
        return df[base_cols + sorted(meta_cols)]


# ---------------------------------------------------------------------------
# Default battery factory
# ---------------------------------------------------------------------------

def create_default_battery(config: dict[str, Any] | None = None) -> SignalBattery:
    """
    Return a ``SignalBattery`` with the 7 canonical families registered.

    Families (matching design doc §4):
        1. time-series momentum (ts_momentum, kind=bars)
        2. cross-sectional momentum (cs_momentum, kind=panel)
        3. mean reversion (mean_reversion, kind=bars)
        4. stat-arb pairs (stat_arb, kind=pair)
        5. trend following: MA crossover + Donchian breakout (kind=bars)
        6. carry: futures roll + funding rate (kind=bars_extra)
        7. cross-exchange arb (kind=exchange_prices)
        +  volatility risk premium (kind=bars_extra)

    Each generator is instantiated with ``config[<family_key>]`` params if
    provided, else its module defaults.
    """
    # Local imports avoid any circular-import risk between orchestrator and
    # the individual signal modules.
    from src.signal_battery.carry import FundingRateArbSignal, FuturesCarrySignal
    from src.signal_battery.cross_exchange_arb import CrossExchangeArbSignal
    from src.signal_battery.mean_reversion import MeanReversionSignal
    from src.signal_battery.momentum import (
        CrossSectionalMomentumSignal,
        TimeSeriesMomentumSignal,
    )
    from src.signal_battery.stat_arb import StatArbSignal
    from src.signal_battery.trend_following import (
        DonchianBreakoutSignal,
        MovingAverageCrossoverSignal,
    )
    from src.signal_battery.volatility_signal import VolatilityRiskPremiumSignal

    cfg = config or {}
    battery = SignalBattery(config=cfg)

    battery.register(
        TimeSeriesMomentumSignal(params=cfg.get("ts_momentum")),
        kind="bars",
    )
    battery.register(
        CrossSectionalMomentumSignal(params=cfg.get("cs_momentum")),
        kind="panel",
    )
    battery.register(
        MeanReversionSignal(params=cfg.get("mean_reversion")),
        kind="bars",
    )
    battery.register(
        StatArbSignal(params=cfg.get("stat_arb")),
        kind="pair",
    )
    battery.register(
        MovingAverageCrossoverSignal(params=cfg.get("ma_crossover")),
        kind="bars",
    )
    battery.register(
        DonchianBreakoutSignal(params=cfg.get("donchian")),
        kind="bars",
    )
    battery.register(
        FuturesCarrySignal(params=cfg.get("futures_carry")),
        kind="bars_extra",
        context_key="futures_curve",
    )
    battery.register(
        FundingRateArbSignal(params=cfg.get("funding_arb")),
        kind="bars_extra",
        context_key="funding_rates",
    )
    battery.register(
        CrossExchangeArbSignal(params=cfg.get("cross_exchange_arb")),
        kind="exchange_prices",
    )
    battery.register(
        VolatilityRiskPremiumSignal(params=cfg.get("vrp")),
        kind="bars_extra",
        context_key="vol_features",
    )
    return battery
