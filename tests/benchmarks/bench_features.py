"""
Phase 2 Performance Benchmarks

Standalone script (not a pytest file) that times the expensive components
of the Feature Factory and Signal Battery against documented targets.

Run with:
    make bench
    # or directly:
    python tests/benchmarks/bench_features.py

Prints a table with component / elapsed / target / status. Components
exceeding their target are flagged in red along with a suggestion for
where to look first when optimising (numba, vectorisation, sub-sampling).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the project root importable when running directly as a script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Match the conftest workaround so torch and openblas don't fight over libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from src.feature_factory.assembler import FeatureAssembler
from src.feature_factory.entropy import compute_entropy_features
from src.feature_factory.fractional_diff import frac_diff_features, frac_diff_ffd, find_min_d
from src.feature_factory.microstructure import compute_microstructure_features
from src.feature_factory.structural_breaks import sadf_test
from src.feature_factory.volatility import fit_garch
from src.signal_battery.carry import FundingRateArbSignal, FuturesCarrySignal
from src.signal_battery.mean_reversion import MeanReversionSignal
from src.signal_battery.momentum import (
    CrossSectionalMomentumSignal,
    TimeSeriesMomentumSignal,
)
from src.signal_battery.orchestrator import SignalBattery
from src.signal_battery.stat_arb import StatArbSignal
from src.signal_battery.trend_following import (
    DonchianBreakoutSignal,
    MovingAverageCrossoverSignal,
)
from src.signal_battery.volatility_signal import VolatilityRiskPremiumSignal


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()
_RED = "\033[31m" if _USE_COLOR else ""
_GREEN = "\033[32m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


@dataclass
class BenchResult:
    name: str
    elapsed: float
    target: float
    suggestion: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.elapsed < self.target else "SLOW"

    def fmt(self) -> str:
        status_color = _GREEN if self.status == "PASS" else _RED
        status = f"{status_color}{self.status}{_RESET}"
        return (
            f"  {self.name:<45s}  "
            f"{self.elapsed:>7.2f}s  "
            f"<  {self.target:>6.1f}s   {status}"
        )


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _price_series(n: int, seed: int = 0, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    return pd.Series(close, index=idx, name="close")


def _bars_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    close = pd.Series(100.0 + rng.normal(0.0, 0.5, size=n).cumsum(), index=idx)
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n)))
    volume = pd.Series(rng.integers(1000, 10000, size=n).astype(float), index=idx)
    buy = volume * rng.uniform(0.3, 0.7, size=n)
    sell = volume - buy
    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "dollar_volume": close * volume,
            "buy_volume": buy,
            "sell_volume": sell,
            "tick_count": pd.Series(rng.integers(50, 500, size=n), index=idx),
            "bar_duration_seconds": pd.Series(np.full(n, 60.0), index=idx),
        },
        index=idx,
    )


def _time_it(func, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_ffd() -> BenchResult:
    s = _price_series(10_000, seed=1)
    # FFD computation proper — a fixed d, one pass. ``find_min_d`` is a
    # separate optimisation loop (~20 ADFs at different d values) and is
    # not the "FFD" itself; it's benchmarked implicitly inside the
    # assembler bench below.
    elapsed = _time_it(frac_diff_ffd, s, d=0.4, threshold=1e-3)
    return BenchResult(
        name="FFD (frac_diff_ffd, n=10k, d=0.4)",
        elapsed=elapsed,
        target=1.0,
        suggestion=(
            "vectorise the weight convolution (scipy.signal.lfilter or "
            "numba @njit) — the inner python loop is the hot path"
        ),
    )


def bench_sadf() -> BenchResult:
    s = _price_series(1_000, seed=2)
    elapsed = _time_it(sadf_test, s, min_window=20, max_lag=1, fast=True)
    return BenchResult(
        name="SADF (n=1k, fast=True)",
        elapsed=elapsed,
        target=10.0,
        suggestion=(
            "default to fast=True; for slower granular scans, cache the "
            "ADF regression residuals across overlapping windows"
        ),
    )


def bench_entropy() -> BenchResult:
    s = _price_series(5_000, seed=3)
    elapsed = _time_it(compute_entropy_features, s, window=100)
    return BenchResult(
        name="entropy features (shannon + lz + sampen, n=5k)",
        elapsed=elapsed,
        target=5.0,
        suggestion=(
            "SampEn's O(N^2) template match is the bottleneck; use a KD-tree "
            "(scipy.spatial.cKDTree) or numba @njit over the sliding windows"
        ),
    )


def bench_microstructure() -> BenchResult:
    bars = _bars_frame(10_000, seed=4)
    elapsed = _time_it(compute_microstructure_features, bars, window=50)
    return BenchResult(
        name="microstructure features (n=10k bars)",
        elapsed=elapsed,
        target=2.0,
        suggestion=(
            "everything is already rolling; the Hasbrouck VAR is opt-in. "
            "If still slow, precompute dp and sv once and share across "
            "Kyle / Roll / OFI instead of recomputing."
        ),
    )


def bench_garch() -> BenchResult:
    rng = np.random.default_rng(5)
    rets = pd.Series(rng.normal(0.0, 0.01, size=1_000))
    elapsed = _time_it(fit_garch, rets)
    return BenchResult(
        name="GARCH(1,1) fit (n=1k returns)",
        elapsed=elapsed,
        target=2.0,
        suggestion=(
            "arch's default BFGS with analytic gradients is already fast; "
            "if hit in a hot loop, switch to the rolling-refit pattern "
            "(volatility.garch_volatility uses refit_interval for this)"
        ),
    )


def bench_assembler() -> BenchResult:
    bars = _bars_frame(5_000, seed=6)
    # Tight windows keep warmup bounded while still exercising every family.
    cfg = {
        "structural_breaks": {"window": 50, "min_window_sadf": 20, "min_period_chow": 30},
        "entropy": {"window": 100},
        "microstructure": {"window": 50},
        "volatility": {
            "window": 252, "refit_interval": 50,
            "short_window": 5, "long_window": 30, "vvol_window": 30,
        },
        "classical": {"rsi_window": 14, "bbw_window": 20, "ret_z_windows": [5, 10, 20]},
    }
    assembler = FeatureAssembler(config=cfg)
    elapsed = _time_it(assembler.assemble, bars)
    return BenchResult(
        name="FeatureAssembler.assemble (n=5k bars)",
        elapsed=elapsed,
        target=30.0,
        suggestion=(
            "SADF and entropy dominate; they have their own fast flags. "
            "Also consider computing feature blocks in parallel with "
            "concurrent.futures.ProcessPoolExecutor."
        ),
    )


def bench_signal_battery() -> BenchResult:
    n_bars = 5_000
    bars = _bars_frame(n_bars, seed=7)

    # Extra inputs per family that needs something beyond plain bars.
    rng = np.random.default_rng(8)
    # CS momentum panel expects {symbol: DataFrame-with-close}, not a Series.
    panel = {
        f"S{i:02d}": pd.DataFrame({"close": _price_series(n_bars, seed=100 + i).values})
        for i in range(25)
    }

    futures_curve = pd.DataFrame(
        {
            "front_price": bars["close"].values,
            "back_price": bars["close"].values * 1.01,
            "days_to_expiry": np.full(n_bars, 30.0),
        },
        index=bars.index,
    )
    funding_rates = pd.DataFrame(
        {"funding_rate": rng.normal(0.0001, 0.0001, size=n_bars)},
        index=bars.index,
    )
    vol_features = pd.DataFrame(
        {
            "iv": rng.uniform(0.15, 0.30, size=n_bars),
            "rv": rng.uniform(0.10, 0.25, size=n_bars),
        },
        index=bars.index,
    )
    stat_arb_pair = (
        bars["close"],
        pd.Series(bars["close"].values * 2.0 + rng.normal(0, 0.3, size=n_bars), index=bars.index),
    )
    # 200 event timestamps sampled across the series.
    event_ix = np.linspace(200, n_bars - 1, 200).astype(int)
    events = bars.index[event_ix]

    battery = SignalBattery()
    battery.register(
        TimeSeriesMomentumSignal(params={"lookbacks": [21, 63], "history_window": 252}),
        kind="bars",
    )
    battery.register(CrossSectionalMomentumSignal(), kind="panel")
    battery.register(MeanReversionSignal(), kind="bars")
    battery.register(StatArbSignal(), kind="pair")
    battery.register(MovingAverageCrossoverSignal(), kind="bars")
    battery.register(DonchianBreakoutSignal(), kind="bars")
    battery.register(FuturesCarrySignal(), kind="bars_extra", context_key="futures_curve")
    battery.register(
        FundingRateArbSignal(), kind="bars_extra", context_key="funding_rates"
    )
    battery.register(
        VolatilityRiskPremiumSignal(), kind="bars_extra", context_key="vol_features"
    )

    def _run():
        battery.generate_all(
            bars,
            event_timestamps=events,
            symbol="BENCH",
            multi_asset_prices=panel,
            stat_arb_pair=stat_arb_pair,
            pair_y_symbol="Y",
            pair_x_symbol="X",
            futures_curve=futures_curve,
            funding_rates=funding_rates,
            vol_features=vol_features,
        )

    elapsed = _time_it(_run)
    return BenchResult(
        name="SignalBattery.generate_all (n=5k bars, 200 events)",
        elapsed=elapsed,
        target=10.0,
        suggestion=(
            "stat-arb's Kalman loop and MR's bar-by-bar z-score loop are "
            "python; port to numpy cumulative math or numba for a 5–10x "
            "speedup"
        ),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    header = (
        f"{_BOLD}Phase 2 Performance Benchmarks — "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
        f"{_RESET}"
    )
    print(header)
    print("-" * len(header))
    print(f"  {'component':<45s}  elapsed    target    status")
    print("-" * 80)

    results: list[BenchResult] = [
        bench_ffd(),
        bench_sadf(),
        bench_entropy(),
        bench_microstructure(),
        bench_garch(),
        bench_assembler(),
        bench_signal_battery(),
    ]

    for r in results:
        print(r.fmt())

    print("-" * 80)
    slow = [r for r in results if r.status != "PASS"]
    if slow:
        print()
        print(f"{_YELLOW}{len(slow)} component(s) exceeded target:{_RESET}")
        for r in slow:
            print(f"  - {r.name}: {r.elapsed:.2f}s > {r.target:.1f}s")
            print(f"      {_YELLOW}→ {r.suggestion}{_RESET}")
        return 1
    print(f"{_GREEN}All {len(results)} benchmarks under target.{_RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
