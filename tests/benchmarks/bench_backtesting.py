"""
Phase 4 Performance Benchmarks

Times the hot paths of the backtesting + portfolio-construction stack
against documented targets.

Run with:
    make bench-backtest
    # or directly:
    python tests/benchmarks/bench_backtesting.py

Prints a table with component / elapsed / target / status. Components
exceeding their target are flagged in red; the status tally is the
process exit code so CI can gate on regressions.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make the project root importable when running directly as a script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from src.backtesting.cpcv import CPCVEngine
from src.backtesting.deflated_sharpe import compute_dsr_from_backtest
from src.backtesting.gate_orchestrator import StrategyGate
from src.backtesting.pbo import compute_pbo
from src.backtesting.transaction_costs import EQUITIES_COSTS, TransactionCostModel
from src.backtesting.walk_forward import BacktestResult, WalkForwardBacktester
from src.portfolio.hrp import compute_hrp_weights
from src.portfolio.risk_parity import compute_risk_parity_weights


# ── pretty-printing helpers ────────────────────────────────────────────


_USE_COLOR = sys.stdout.isatty()
_RED = "\033[31m" if _USE_COLOR else ""
_GREEN = "\033[32m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


@dataclass
class BenchResult:
    name: str
    elapsed: float
    target: float
    note: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.elapsed < self.target else "SLOW"

    def fmt(self) -> str:
        status_color = _GREEN if self.status == "PASS" else _RED
        status = f"{status_color}{self.status}{_RESET}"
        row = (
            f"  {self.name:<46s}  "
            f"{self.elapsed:>7.2f}s  <  "
            f"{self.target:>6.1f}s   {status}"
        )
        if self.note:
            row += f"   ({self.note})"
        return row


# ── synthetic-data helpers ─────────────────────────────────────────────


def _close_panel(n_bars: int, n_syms: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    data = np.cumsum(rng.normal(0, 0.01, size=(n_bars, n_syms)), axis=0)
    return pd.DataFrame(
        100 * np.exp(data),
        index=idx,
        columns=[f"S{i}" for i in range(n_syms)],
    )


def _signals(close: pd.DataFrame) -> pd.DataFrame:
    mom = close.pct_change(20).fillna(0)
    sig = pd.DataFrame(
        np.sign(mom.values), index=close.index, columns=close.columns
    )
    sig.iloc[:25] = 0
    return sig


def _returns_panel(T: int, N: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # add a few latent factors for realistic covariance structure
    loadings = rng.normal(0, 0.3, size=(N, max(N // 5, 1)))
    factors = rng.normal(0, 0.01, size=(T, loadings.shape[1]))
    idio = rng.normal(0, 0.003, size=(T, N))
    data = factors @ loadings.T + idio
    return pd.DataFrame(
        data,
        index=pd.date_range("2024-01-01", periods=T, freq="B"),
        columns=[f"A{i}" for i in range(N)],
    )


def _fake_backtest_result(n_bars: int = 500, seed: int = 0) -> BacktestResult:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    r = rng.normal(1.0 / 252, 1 / np.sqrt(252), n_bars)
    equity = pd.Series(100_000 * np.exp(np.cumsum(r)), index=idx)
    return BacktestResult(
        trades=[],
        equity_curve=equity,
        returns=equity.pct_change().fillna(0),
        drawdown_curve=pd.Series(0.0, index=idx),
        metrics={
            "sharpe": 1.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "max_drawdown": 0.0,
            "total_trades": 0,
        },
    )


# ── individual benchmarks ──────────────────────────────────────────────


def bench_walk_forward() -> BenchResult:
    close = _close_panel(n_bars=5_000, n_syms=10)
    signals = _signals(close)
    probs = pd.DataFrame(0.7, index=close.index, columns=close.columns)
    bets = pd.DataFrame(0.05, index=close.index, columns=close.columns)

    bt = WalkForwardBacktester(
        cost_model=TransactionCostModel(equities_config=EQUITIES_COSTS),
        max_holding_period=10,
    )
    t0 = time.perf_counter()
    r = bt.run(close, signals, probs, bets)
    elapsed = time.perf_counter() - t0
    return BenchResult(
        "walk-forward (5 000 bars × 10 symbols)",
        elapsed,
        30.0,
        note=f"{len(r.trades)} trades",
    )


def bench_cpcv() -> BenchResult:
    """CPCV without per-path retraining (no ML model in this benchmark).

    The prompt's 300 s target assumes each path retrains a meta-labeler.
    Our benchmark re-runs the (cheap) walk-forward backtester on each
    path's test slice, so the number comes in well under target and gives
    an upper bound on the partition/iteration overhead.
    """
    close = _close_panel(n_bars=5_000, n_syms=6)
    signals = _signals(close)
    probs = pd.DataFrame(0.7, index=close.index, columns=close.columns)
    bets = pd.DataFrame(0.05, index=close.index, columns=close.columns)
    bt = WalkForwardBacktester(
        cost_model=TransactionCostModel(equities_config=EQUITIES_COSTS),
        max_holding_period=10,
    )

    idx = close.index
    n = len(idx)
    labels_df = pd.DataFrame(
        {"event_start": idx, "event_end": idx.shift(10, freq="B")},
        index=idx,
    )
    engine = CPCVEngine(n_groups=10, n_test_groups=2, embargo_pct=0.01)
    # Use a 1-column "features" panel — generate_paths only needs lengths.
    feats = pd.DataFrame(np.zeros((n, 1)), index=idx)

    t0 = time.perf_counter()
    paths = engine.generate_paths(feats, None, labels_df)
    assert len(paths) == 45
    for path in paths:
        _, test_idx = path[0]
        if len(test_idx) < 20:
            continue
        sl = idx[test_idx]
        bt.run(
            close.loc[sl],
            signals.loc[sl],
            probs.loc[sl],
            bets.loc[sl],
        )
    elapsed = time.perf_counter() - t0
    return BenchResult(
        "CPCV (45 paths, no model retrain)",
        elapsed,
        300.0,
        note="per-path retrain adds model-fit cost on top",
    )


def bench_hrp() -> BenchResult:
    rets = _returns_panel(T=252, N=50)
    t0 = time.perf_counter()
    w = compute_hrp_weights(rets)
    elapsed = time.perf_counter() - t0
    assert w.sum() > 0
    return BenchResult(
        "HRP (50 assets × 252 returns)", elapsed, 1.0
    )


def bench_risk_parity() -> BenchResult:
    rets = _returns_panel(T=252, N=50, seed=1)
    cov = rets.cov()
    t0 = time.perf_counter()
    w = compute_risk_parity_weights(cov)
    elapsed = time.perf_counter() - t0
    assert w.sum() > 0
    return BenchResult(
        "Risk Parity (50 assets)", elapsed, 2.0
    )


def bench_pbo() -> BenchResult:
    rng = np.random.default_rng(0)
    T = 1_000
    N = 20
    data = rng.normal(0, 0.01, size=(T, N))
    idx = pd.date_range("2024-01-01", periods=T, freq="B")
    matrix = pd.DataFrame(data, index=idx, columns=[f"v{i}" for i in range(N)])
    t0 = time.perf_counter()
    pbo, _ = compute_pbo(matrix, n_partitions=10)
    elapsed = time.perf_counter() - t0
    return BenchResult(
        "PBO (20 variants × 1 000 bars)",
        elapsed,
        60.0,
        note=f"pbo={pbo:.3f}",
    )


def bench_quick_gate() -> BenchResult:
    gate = StrategyGate()
    br = _fake_backtest_result(n_bars=5_000)
    t0 = time.perf_counter()
    out = gate.quick_validate(br, n_trials=10)
    elapsed = time.perf_counter() - t0
    assert "recommendation" in out
    return BenchResult(
        "StrategyGate.quick_validate (5 000 bars)",
        elapsed,
        60.0,
        note="DSR only — no CPCV/PBO",
    )


# ── runner ─────────────────────────────────────────────────────────────


def main() -> int:
    print(f"\n{_BOLD}Phase 4 Performance Benchmarks{_RESET}\n")
    header = (
        f"  {'component':<46s}  "
        f"{'elapsed':>8s}  "
        f"{'target':>11s}   {'status':<6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: list[BenchResult] = []
    for fn in (
        bench_walk_forward,
        bench_cpcv,
        bench_hrp,
        bench_risk_parity,
        bench_pbo,
        bench_quick_gate,
    ):
        r = fn()
        print(r.fmt())
        results.append(r)

    slow = [r for r in results if r.status == "SLOW"]
    print()
    if slow:
        print(
            f"  {_RED}{len(slow)} / {len(results)} component(s) over target{_RESET}"
        )
    else:
        print(
            f"  {_GREEN}All {len(results)} components within target{_RESET}"
        )
    return 0 if not slow else 1


if __name__ == "__main__":
    sys.exit(main())
