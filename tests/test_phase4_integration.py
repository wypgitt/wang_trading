"""
End-to-end Phase 4 integration test.

Generates a 4-year, 10-symbol synthetic market (2 trending up, 2 mean-reverting,
2 cointegrated pairs, 4 random walks) and drives the Phase 4 machinery:

    synthetic bars
        ↓  lightweight per-symbol signals (momentum / z-score)
        ↓  WalkForwardBacktester with realistic transaction costs
        ↓  CPCV (n_groups=6, n_test=2 → 15 paths) for gate 1
        ↓  Deflated Sharpe Ratio (gate 2)
        ↓  Probability of Backtest Overfitting (gate 3)
        ↓  HRP + Factor risk model
        ↓  StrategyGate (quick + full recommendation wiring)

Pass criteria
-------------
* Signal-bearing symbols (trending + mean-reverting + cointegrated) produce
  meaningfully positive gross P&L, while the pure random-walk names land
  near zero (net of costs is usually slightly negative).
* HRP weights stay diversified — no single symbol > 40%.
* The PCA factor risk model picks up at least 2 factors with non-trivial
  explained variance.
* Wall-clock < 300 s on CI hardware.

Tagged ``@pytest.mark.integration`` — excluded from the default pytest run;
opt in with ``pytest -m integration -o addopts=""`` (or ``make test-integration``).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.backtesting.cpcv import CPCVEngine, validate_strategy
from src.backtesting.deflated_sharpe import compute_dsr_from_cpcv
from src.backtesting.gate_orchestrator import StrategyGate
from src.backtesting.pbo import compute_pbo
from src.backtesting.transaction_costs import EQUITIES_COSTS, TransactionCostModel
from src.backtesting.walk_forward import WalkForwardBacktester
from src.portfolio.factor_risk import FactorRiskModel
from src.portfolio.hrp import compute_hrp_weights


pytestmark = pytest.mark.integration


# ── synthetic market ───────────────────────────────────────────────────


def _generate_universe(
    n_days: int = 1008,  # ~4 years daily
    seed: int = 17,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a 10-symbol close-price panel with labelled regimes.

    Returns (close, tag_map). ``tag_map`` is symbol → {"trend","meanrev","coint","rw"}.
    """

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B")

    data: dict[str, np.ndarray] = {}
    tags: dict[str, str] = {}

    # 2 trending-up
    for i in range(2):
        drift = 0.0008 + 0.0003 * i
        noise = rng.normal(0, 0.01, n_days)
        data[f"TREND_{i}"] = 100 * np.exp(np.cumsum(drift + noise))
        tags[f"TREND_{i}"] = "trend"

    # 2 mean-reverting AR(1) with tight phi
    for i in range(2):
        phi = 0.93
        level = np.zeros(n_days)
        noise = rng.normal(0, 0.8, n_days)
        level[0] = 100.0
        for t in range(1, n_days):
            level[t] = 100 * (1 - phi) + phi * level[t - 1] + noise[t]
        data[f"MEANREV_{i}"] = np.clip(level, 10.0, None)
        tags[f"MEANREV_{i}"] = "meanrev"

    # 2 cointegrated pairs — a common driver + small idiosyncratic noise
    common = rng.normal(0, 0.01, n_days).cumsum()
    for i in range(2):
        spread = rng.normal(0, 0.3, n_days).cumsum() * 0.01
        data[f"COINT_{i}"] = 100 * np.exp(common + spread)
        tags[f"COINT_{i}"] = "coint"

    # 4 random walks — no edge
    for i in range(4):
        data[f"RW_{i}"] = 100 * np.exp(
            np.cumsum(rng.normal(0, 0.01, n_days))
        )
        tags[f"RW_{i}"] = "rw"

    close = pd.DataFrame(data, index=idx)
    return close, tags


def _build_signals(close: pd.DataFrame, tags: dict[str, str]) -> pd.DataFrame:
    """Trivial per-tag signal generator — stand-in for Phases 1–3.

    * trend → 20-bar momentum sign
    * meanrev → negative z-score of deviation from rolling mean
    * coint → sign of the pair's spread vs its rolling mean
    * rw → random ±1 noise (no edge)
    """

    signals = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=float)
    rng = np.random.default_rng(0)

    for sym, tag in tags.items():
        px = close[sym]
        if tag == "trend":
            mom = px.pct_change(20)
            signals[sym] = np.sign(mom.fillna(0))
        elif tag == "meanrev":
            ma = px.rolling(20).mean()
            sd = px.rolling(20).std(ddof=0)
            z = (px - ma) / sd.replace(0, np.nan)
            signals[sym] = -np.sign(z.fillna(0))
        elif tag == "coint":
            # diff vs partner coint symbol
            partner = "COINT_0" if sym == "COINT_1" else "COINT_1"
            spread = (close[sym] / close[partner]).apply(np.log)
            ma = spread.rolling(20).mean()
            signals[sym] = -np.sign((spread - ma).fillna(0))
        else:  # rw
            signals[sym] = rng.choice([-1, 0, 1], size=len(close), p=[0.15, 0.70, 0.15])

    # don't trade the first 30 bars while rolling stats warm up
    signals.iloc[:30] = 0
    return signals


# ── the integration test ───────────────────────────────────────────────


def test_phase4_end_to_end():
    start = time.perf_counter()
    close, tags = _generate_universe()
    signals = _build_signals(close, tags)
    n_bars, n_symbols = close.shape
    assert n_symbols == 10

    cost_model = TransactionCostModel(equities_config=EQUITIES_COSTS)

    # ── walk-forward backtest on the full panel ──
    backtester = WalkForwardBacktester(
        cost_model=cost_model,
        initial_capital=100_000,
        execution_delay_bars=1,
        max_positions=10,
        max_holding_period=10,
        upper_multiplier=2.0,
        lower_multiplier=2.0,
    )
    probs = pd.DataFrame(0.7, index=close.index, columns=close.columns)
    bets = pd.DataFrame(0.05, index=close.index, columns=close.columns)
    result = backtester.run(close, signals, probs, bets)

    # ── per-regime P&L check: signal-bearing tags make money gross, RW doesn't ──
    pnl_by_tag: dict[str, float] = {"trend": 0.0, "meanrev": 0.0, "coint": 0.0, "rw": 0.0}
    gross_by_tag: dict[str, float] = {"trend": 0.0, "meanrev": 0.0, "coint": 0.0, "rw": 0.0}
    for t in result.trades:
        tag = tags[t.symbol]
        pnl_by_tag[tag] += t.net_pnl
        gross_by_tag[tag] += t.gross_pnl

    assert len(result.trades) > 50, (
        f"expected a meaningful trade count; got {len(result.trades)}"
    )
    # Signal-bearing regimes produce positive gross P&L in aggregate
    assert gross_by_tag["trend"] + gross_by_tag["meanrev"] + gross_by_tag["coint"] > 0, (
        f"signal tags lost money gross: {gross_by_tag}"
    )
    # Random-walk gross P&L is small relative to the signal tags'
    signal_gross = gross_by_tag["trend"] + gross_by_tag["meanrev"] + gross_by_tag["coint"]
    assert abs(gross_by_tag["rw"]) < max(abs(signal_gross), 1.0)

    # ── CPCV: 6 groups, 2 test → C(6,2) = 15 paths ──
    idx = close.index
    labels_df = pd.DataFrame(
        {
            "event_start": idx,
            "event_end": idx.shift(10, freq="B"),
        },
        index=idx,
    )
    engine = CPCVEngine(n_groups=6, n_test_groups=2, embargo_pct=0.01)

    # For CPCV we just need the test-slice backtest results — we don't
    # retrain a model here; we re-run the synthetic signals on each path's
    # test slice.
    cpcv_results = []
    paths = engine.generate_paths(
        pd.DataFrame(np.zeros((n_bars, 1)), index=idx), None, labels_df
    )
    assert len(paths) == 15
    for path in paths:
        _, test_idx = path[0]
        if len(test_idx) < 20:
            continue
        test_index = idx[test_idx]
        slice_close = close.loc[test_index]
        slice_sig = signals.loc[test_index]
        slice_probs = probs.loc[test_index]
        slice_bets = bets.loc[test_index]
        r = backtester.run(slice_close, slice_sig, slice_probs, slice_bets)
        cpcv_results.append(r)

    assert len(cpcv_results) == 15
    stats = engine.get_path_statistics(cpcv_results)
    assert "mean" in stats.index
    g1_passed, g1_stats = validate_strategy(cpcv_results, min_positive_paths_pct=0.6)
    # We don't require g1 to pass — synthetic data is too noisy; just assert
    # the machinery produced a sane statistic.
    assert 0.0 <= g1_stats["positive_pct"] <= 1.0
    assert g1_stats["path_count"] == 15

    # ── Deflated Sharpe across CPCV paths ──
    dsr = compute_dsr_from_cpcv(cpcv_results, n_total_trials=10)
    for key in ("observed_sharpe", "expected_max_sharpe", "dsr_statistic", "p_value", "passed"):
        assert key in dsr
    assert 0.0 <= dsr["p_value"] <= 1.0

    # ── PBO over a mini-grid of per-symbol returns ──
    # Build a (T × N) matrix where each column is one symbol's strategy
    # returns after applying that symbol's signal (gross of costs). This is
    # the simplest realistic input for PBO.
    rets = close.pct_change().fillna(0.0)
    variant_returns = (rets * signals.shift(1).fillna(0.0)).fillna(0.0)
    pbo_value, pbo_details = compute_pbo(
        variant_returns.iloc[40:], n_partitions=8
    )
    assert 0.0 <= pbo_value <= 1.0
    assert len(pbo_details) == 70  # C(8, 4)

    # ── HRP weights over the full returns panel ──
    hrp_w = compute_hrp_weights(rets.iloc[20:])
    assert hrp_w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (hrp_w >= 0).all()
    assert hrp_w.max() < 0.40, f"HRP is over-concentrated: max={hrp_w.max():.3f}"

    # ── Factor risk model: at least 2 factors with nontrivial variance ──
    factor_model = FactorRiskModel(n_factors=5).fit(rets.iloc[20:])
    evr = factor_model.explained_variance_ratio
    assert (evr > 0.01).sum() >= 2

    # ── StrategyGate full wiring + quick path ──
    gate = StrategyGate(backtester=backtester)
    quick = gate.quick_validate(result, n_trials=5)
    assert {"passed", "recommendation", "gate_2_dsr"}.issubset(quick.keys())
    assert isinstance(quick["recommendation"], str)

    # Wall-clock budget
    duration = time.perf_counter() - start
    assert duration < 300.0, f"phase-4 integration took {duration:.1f}s"
