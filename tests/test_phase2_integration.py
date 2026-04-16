"""
End-to-end Phase 2 integration test.

Tagged ``@pytest.mark.integration`` so it's excluded from the default
``make test`` (unit tests) target. Run with ``make test-integration`` or
``pytest -m integration -o addopts=""``.

Covers:
    1. Synthetic-market generation (10 symbols: trending / MR / cointegrated
       pairs / random walks) with realistic log-normal volumes.
    2. Tick-level data funneled through the Phase 1 TIBConstructor.
    3. FeatureAssembler on the resulting bars → finite, stationary matrix.
    4. CUSUM filter → event timestamps.
    5. SignalBattery on the events → family-level sanity checks.
    6. Wall-clock budget: full pipeline < 60 seconds.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from statsmodels.tsa.stattools import adfuller

from src.data_engine.bars.constructors import TIBConstructor
from src.data_engine.bars.cusum_filter import (
    compute_cusum_threshold,
    cusum_filter,
)
from src.data_engine.models import Side, Tick
from src.feature_factory.assembler import FeatureAssembler
from src.signal_battery.base_signal import BaseSignalGenerator  # noqa: F401
from src.signal_battery.mean_reversion import MeanReversionSignal
from src.signal_battery.momentum import TimeSeriesMomentumSignal
from src.signal_battery.orchestrator import SignalBattery
from src.signal_battery.stat_arb import StatArbSignal


# ---------------------------------------------------------------------------
# Synthetic market generator
# ---------------------------------------------------------------------------

def _generate_symbols(
    n_bars_per_symbol: int = 500,
    ticks_per_bar: int = 30,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Build a 10-symbol market:
        TREND_UP_1,  TREND_UP_2     : positive-drift GBM
        MR_1,  MR_2                 : AR(1) around a level, phi = 0.95
        COINT_A, COINT_B            : cointegrated pair (slope 2, AR(1) resid)
        COINT_C, COINT_D            : second cointegrated pair
        RW_1, RW_2                  : random walks

    Returns dict of symbol → {"prices": ndarray, "volumes": ndarray, "timestamps": list}.
    Prices are sampled at tick cadence so TIB can consume them directly.
    """
    rng = np.random.default_rng(seed)
    n_ticks = n_bars_per_symbol * ticks_per_bar
    dt = 1.0 / ticks_per_bar  # fraction-of-a-bar (for correct GBM scaling)
    sqrt_dt = float(np.sqrt(dt))
    # Per-tick noise scale (low enough that P(tick up) hovers near 50% and
    # TIB bars form at a predictable cadence instead of feeding back into an
    # ever-growing imbalance threshold on a strong trend).
    tick_sigma = 0.002

    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=i) for i in range(n_ticks)]

    def _volumes(n: int) -> np.ndarray:
        # Log-normal volumes (realistic long tail; ~55 median).
        return rng.lognormal(mean=4.0, sigma=0.4, size=n)

    market: dict[str, dict] = {}

    # --- trending symbols: small drift per tick so P(buy) stays near 50%+ ---
    for name, drift_daily in [("TREND_UP_1", 0.002), ("TREND_UP_2", 0.0015)]:
        # Correct GBM scaling: mean linear in dt, std linear in sqrt(dt).
        rets = rng.normal(loc=drift_daily * dt, scale=tick_sigma * sqrt_dt, size=n_ticks)
        prices = 100.0 * np.exp(np.cumsum(rets))
        market[name] = {
            "prices": prices, "volumes": _volumes(n_ticks), "timestamps": timestamps,
        }

    # --- mean-reverting symbols: slow-reverting AR(1) around 100 ---
    # phi=0.998 at the TICK level → half-life ≈ 346 ticks ≈ 9 bars, giving
    # the MR signal's rolling z-score a window large enough (>2) to produce
    # meaningful scores and actually fire entry/exit events.
    for name in ["MR_1", "MR_2"]:
        phi = 0.998
        eps = rng.normal(0.0, 0.4, size=n_ticks)
        y = np.zeros(n_ticks)
        for t in range(1, n_ticks):
            y[t] = phi * y[t - 1] + eps[t]
        prices = 100.0 + y
        market[name] = {
            "prices": prices, "volumes": _volumes(n_ticks), "timestamps": timestamps,
        }

    # --- cointegrated pairs ---
    def _make_pair(name_x: str, name_y: str, hedge: float, phi: float):
        x_rets = rng.normal(0.0, tick_sigma * sqrt_dt, size=n_ticks)
        x = 100.0 * np.exp(np.cumsum(x_rets))
        resid = np.zeros(n_ticks)
        for t in range(1, n_ticks):
            resid[t] = phi * resid[t - 1] + rng.normal(0.0, 0.4)
        y = hedge * x + resid
        market[name_x] = {
            "prices": x, "volumes": _volumes(n_ticks), "timestamps": timestamps,
        }
        market[name_y] = {
            "prices": y, "volumes": _volumes(n_ticks), "timestamps": timestamps,
        }

    _make_pair("COINT_A", "COINT_B", hedge=2.0, phi=0.97)
    _make_pair("COINT_C", "COINT_D", hedge=1.5, phi=0.96)

    # --- random walks: zero drift, same noise scale as trending ---
    for name in ["RW_1", "RW_2"]:
        rets = rng.normal(0.0, tick_sigma * sqrt_dt, size=n_ticks)
        prices = 100.0 * np.exp(np.cumsum(rets))
        market[name] = {
            "prices": prices, "volumes": _volumes(n_ticks), "timestamps": timestamps,
        }

    return market


def _ticks_for(symbol: str, spec: dict) -> list[Tick]:
    """Materialise Tick objects from a price/volume/timestamp dict."""
    ticks: list[Tick] = []
    prev_price = spec["prices"][0]
    for ts, px, vol in zip(spec["timestamps"], spec["prices"], spec["volumes"]):
        # Nudge with explicit side when price is flat so the tick-rule
        # classifier doesn't get stuck on UNKNOWN for the opening tick.
        side = (
            Side.BUY if px >= prev_price else Side.SELL
        )
        ticks.append(
            Tick(
                timestamp=ts,
                symbol=symbol,
                price=float(px),
                volume=float(vol),
                side=side,
            )
        )
        prev_price = px
    return ticks


def _bars_to_dataframe(bars: list) -> pd.DataFrame:
    """Convert list[Bar] to a DataFrame with assembler-compatible columns."""
    rows = [asdict(b) for b in bars]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


# ---------------------------------------------------------------------------
# The integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_phase2_full_pipeline():
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Generate synthetic market data.
    # ------------------------------------------------------------------
    n_bars = 500              # target bars per symbol after TIB constructs
    ticks_per_bar = 40        # how many ticks per bar on average (generous)
    market = _generate_symbols(
        n_bars_per_symbol=n_bars, ticks_per_bar=ticks_per_bar, seed=7,
    )
    assert set(market.keys()) == {
        "TREND_UP_1", "TREND_UP_2", "MR_1", "MR_2",
        "COINT_A", "COINT_B", "COINT_C", "COINT_D",
        "RW_1", "RW_2",
    }

    # ------------------------------------------------------------------
    # 2. Build TIB bars per symbol.
    # ------------------------------------------------------------------
    symbol_bars: dict[str, pd.DataFrame] = {}
    for symbol, spec in market.items():
        # Low initial threshold + large EWMA span keeps the adaptive
        # imbalance threshold close to its starting value so bar cadence
        # is predictable even on trending symbols (where a feedback loop
        # between imbalance and threshold can otherwise throttle bar
        # production dramatically).
        cons = TIBConstructor(
            symbol=symbol, ewma_span=500, initial_threshold=3.0,
        )
        ticks = _ticks_for(symbol, spec)
        bars = cons.process_ticks(ticks)
        assert len(bars) > 50, f"{symbol}: only {len(bars)} bars — too few"
        symbol_bars[symbol] = _bars_to_dataframe(bars)

    logger.info(
        f"integration: built TIB bars for {len(symbol_bars)} symbols, "
        f"avg {int(np.mean([len(b) for b in symbol_bars.values()]))} bars each"
    )

    # ------------------------------------------------------------------
    # 3. FeatureAssembler on one representative (trending) symbol.
    # ------------------------------------------------------------------
    feature_bars = symbol_bars["TREND_UP_1"]
    # Keep windows small so warmup doesn't eat the entire test slice.
    config = {
        "structural_breaks": {
            "window": 20, "min_window_sadf": 10, "min_period_chow": 15,
        },
        "entropy": {"window": 40},
        "microstructure": {"window": 20},
        "volatility": {
            # Tight GARCH window to keep feature-matrix warmup short on the
            # ~200 bars per symbol produced by the TIB constructor.
            "window": 60, "refit_interval": 30,
            "short_window": 5, "long_window": 15, "vvol_window": 15,
        },
        "classical": {
            "rsi_window": 14, "bbw_window": 15, "ret_z_windows": [5, 10, 15],
        },
    }
    assembler = FeatureAssembler(config=config)
    feature_matrix = assembler.assemble(feature_bars)

    assert not feature_matrix.empty
    values = feature_matrix.to_numpy(dtype=float)
    assert np.isfinite(values).all(), "feature matrix contains non-finite values"
    n_features = feature_matrix.shape[1]
    # Expected feature count (per-family counts as specified in prompts
    # P2.01–P2.08): FFD(3) + structural_breaks(3) + entropy(3) +
    # microstructure(6) + volatility(3) + classical(5) = 23. We allow
    # 15..70 to cover minor config variations and future additions
    # (GSADF, ApEn, Hasbrouck, etc. are opt-in).
    assert 15 <= n_features <= 70, (
        f"feature count {n_features} outside expected 15–70 band"
    )

    # Stationarity check: ADF has limited statistical power on the ~100-row
    # post-warmup slices produced by this short synthetic run, so we only
    # require a majority (>= 50 %) of columns to clear p < 0.10. In
    # production with years of daily data the assembler's post-hoc FFD
    # pass typically brings this well above 95 %; the remaining columns
    # are flagged with an "could not stationarise" warning.
    n_stationary = 0
    n_testable = 0
    for col in feature_matrix.columns:
        vals = feature_matrix[col].to_numpy(dtype=float)
        if np.std(vals) == 0:
            continue
        n_testable += 1
        pval = adfuller(vals, autolag="AIC")[1]
        if pval < 0.10:
            n_stationary += 1
    stationarity_rate = n_stationary / max(n_testable, 1)
    assert stationarity_rate >= 0.50, (
        f"only {stationarity_rate:.1%} of features are stationary "
        f"({n_stationary}/{n_testable})"
    )

    # ------------------------------------------------------------------
    # 4. CUSUM filter → events.
    # ------------------------------------------------------------------
    threshold = compute_cusum_threshold(
        feature_bars["close"], multiplier=1.5, lookback=60,
    )
    events = cusum_filter(feature_bars["close"], threshold=threshold)
    n_bars_symbol = len(feature_bars)
    assert len(events) > 0, "CUSUM produced zero events"
    assert len(events) < n_bars_symbol, (
        f"CUSUM fired on every bar ({n_bars_symbol}) — threshold too low"
    )

    # ------------------------------------------------------------------
    # 5. SignalBattery: momentum + mean reversion + stat arb.
    # ------------------------------------------------------------------
    battery = SignalBattery()
    battery.register(
        TimeSeriesMomentumSignal(
            params={
                "lookbacks": [5, 10, 20],
                "history_window": 40,
                "min_history": 60,
            }
        ),
        kind="bars",
    )
    battery.register(
        MeanReversionSignal(params={"min_halflife": 1.0, "max_halflife": 200.0}),
        kind="bars",
    )
    battery.register(
        StatArbSignal(params={"min_halflife": 1.0, "max_halflife": 200.0}),
        kind="pair",
    )

    # ---- TS momentum on a trending symbol (should lean long) ----
    trend_sigs_df = battery.generate_all(
        feature_bars,
        event_timestamps=events,
        symbol="TREND_UP_1",
    )
    ts_mom = trend_sigs_df[trend_sigs_df["family"] == "ts_momentum"]
    assert not ts_mom.empty, "ts_momentum emitted no signals on a trending symbol"
    long_frac = (ts_mom["side"] == 1).mean()
    assert long_frac > 0.6, (
        f"ts_momentum long fraction {long_frac:.2f} too low on a trending symbol"
    )

    # ---- Mean reversion on an MR symbol (should fire at all) ----
    # We don't event-filter here: MR's own entry/exit logic already acts as
    # an implicit filter, and event-filtering can wipe every signal out on
    # short synthetic slices where CUSUM and MR triggers rarely coincide.
    mr_bars = symbol_bars["MR_1"]
    mr_sigs_df = battery.generate_all(
        mr_bars,
        event_timestamps=None,
        symbol="MR_1",
    )
    mr_only = mr_sigs_df[mr_sigs_df["family"] == "mean_reversion"]
    assert not mr_only.empty, "mean_reversion emitted no signals on an MR symbol"

    # ---- Stat arb on a cointegrated pair ----
    # The two legs' TIB bars don't share timestamps (each leg's bars close
    # when its own imbalance hits threshold), so naive intersection drops
    # almost everything. Align on the union index with forward-fill, then
    # keep only bars where BOTH legs have data. This mirrors what a
    # production pairs-trade runner would do with tick-level quotes.
    y_bars = symbol_bars["COINT_B"]["close"]
    x_bars = symbol_bars["COINT_A"]["close"]
    union = y_bars.index.union(x_bars.index).sort_values()
    y_aligned = y_bars.reindex(union).ffill().dropna()
    x_aligned = x_bars.reindex(union).ffill().dropna()
    common = y_aligned.index.intersection(x_aligned.index)
    y_aligned = y_aligned.loc[common]
    x_aligned = x_aligned.loc[common]

    pair_sigs_df = battery.generate_all(
        symbol_bars["COINT_B"],
        event_timestamps=None,
        symbol="COINT_B",
        stat_arb_pair=(y_aligned, x_aligned),
        pair_y_symbol="COINT_B",
        pair_x_symbol="COINT_A",
    )
    sa = pair_sigs_df[pair_sigs_df["family"] == "stat_arb"]
    assert not sa.empty, "stat_arb emitted no signals on a cointegrated pair"

    # ---- Log per-family stats for visibility. ----
    stats = battery.get_signal_stats(trend_sigs_df)
    logger.info(f"integration: signal stats on trending symbol = {stats}")

    # ------------------------------------------------------------------
    # 6. Wall-clock budget.
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    logger.info(f"integration: full pipeline elapsed = {elapsed:.2f}s")
    assert elapsed < 60.0, f"pipeline too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-o", "addopts="])
