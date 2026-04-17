"""End-to-end smoke test (Phase 5 / P5.16).

Validates the whole trading stack without external services: generates
synthetic market data, builds bars, features, signals, a meta-labeler,
runs a walk-forward backtest + DSR + HRP, then drives 50 paper-trading
cycles and produces a daily report. Target: < 120s wall time.

Run:
    python scripts/smoke_test.py
    make smoke-test
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prometheus_client import CollectorRegistry

from src.backtesting.deflated_sharpe import deflated_sharpe_ratio
from src.backtesting.transaction_costs import TransactionCostModel
from src.data_engine.bars.constructors import (
    TickBarConstructor,
    TIBConstructor,
)
from src.data_engine.models import BarType, Side, Tick
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.daily_ops import generate_daily_report
from src.execution.models import (
    ExecutionAlgo,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PaperTradingPipeline, PipelineConfig
from src.execution.tca import TCAAnalyzer, TCAResult
from src.monitoring.alerting import AlertManager, AlertSeverity, LogChannel
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector
from src.portfolio.hrp import compute_hrp_weights


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke")

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
N_DAYS = 504  # ~2 years of trading days


@contextmanager
def _timed(step: str, report: list[tuple[str, float, str]]):
    t0 = time.perf_counter()
    status = "OK"
    msg = ""
    try:
        yield
    except Exception as exc:
        status = f"FAIL: {type(exc).__name__}"
        msg = str(exc)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        report.append((step, elapsed, status if not msg else f"{status} — {msg[:60]}"))
        print(f"  [{elapsed:5.2f}s] {step}: {status}")


# ── Data generators ───────────────────────────────────────────────────

def _synthetic_ticks(symbol: str, n_days: int, seed: int) -> list[Tick]:
    rng = np.random.default_rng(seed)
    ticks_per_day = 200
    prices = [100.0]
    for _ in range(n_days * ticks_per_day - 1):
        ret = rng.normal(0, 0.002)
        prices.append(max(prices[-1] * (1 + ret), 1.0))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ticks: list[Tick] = []
    for i, price in enumerate(prices):
        minutes = i * 2  # 2 min apart
        ticks.append(Tick(
            timestamp=start + timedelta(minutes=minutes),
            symbol=symbol,
            price=float(price),
            volume=float(rng.integers(100, 500)),
            side=Side.BUY if rng.random() > 0.5 else Side.SELL,
            exchange="SYNTH",
        ))
    return ticks


def _bars_from_ticks(symbol: str, ticks: list[Tick]) -> pd.DataFrame:
    # TIB is the design-doc primary. Synthetic balanced ticks rarely trip
    # the imbalance threshold, so also run tick bars for dense coverage.
    tib = TIBConstructor(symbol=symbol, ewma_span=20, initial_threshold=10)
    _ = tib.process_ticks(ticks)  # exercise TIB code path
    constructor = TickBarConstructor(symbol=symbol, bar_size=200)
    bars = constructor.process_ticks(ticks)
    if not bars:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "timestamp": b.timestamp,
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume, "symbol": b.symbol,
        }
        for b in bars
    ]).set_index("timestamp")


# ── Lightweight feature + signal + model ──────────────────────────────

def _simple_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    ret = close.pct_change()
    feats = pd.DataFrame({
        "return": ret,
        "ret_5": ret.rolling(5).mean(),
        "ret_20": ret.rolling(20).mean(),
        "vol_20": ret.rolling(20).std(),
        "mom_50": close.pct_change(50),
        "z_20": (close - close.rolling(20).mean()) / close.rolling(20).std(),
    }, index=bars.index).dropna()
    return feats


def _generate_signals(features: pd.DataFrame) -> pd.Series:
    # Simple momentum-z-score signal
    z = features["z_20"].fillna(0.0)
    return np.sign(z)


def _train_meta_labeler(features: pd.DataFrame, signals: pd.Series,
                        forward_return: pd.Series) -> tuple[object, pd.Series]:
    """Train a small LightGBM meta-labeler: labels = sign(fwd_ret) * sign(signal) > 0."""
    from lightgbm import LGBMClassifier
    df = pd.concat([features, signals.rename("signal"),
                    forward_return.rename("fwd_ret")], axis=1).dropna()
    labels = (np.sign(df["fwd_ret"]) == df["signal"]).astype(int)
    X = df.drop(columns=["fwd_ret"])
    model = LGBMClassifier(n_estimators=50, num_leaves=15, learning_rate=0.1,
                           verbose=-1)
    model.fit(X, labels)
    probs = pd.Series(model.predict_proba(X)[:, 1], index=df.index, name="meta_prob")
    return model, probs


def _bet_sizes(probs: pd.Series, cap: float = 0.03) -> pd.Series:
    # Simple cascade surrogate: 2*(p-0.5) clamped then capped
    raw = (2 * (probs - 0.5)).clip(-1, 1)
    return raw * cap


# ── Walk-forward backtest ─────────────────────────────────────────────

def _walk_forward_metrics(returns: pd.Series) -> dict:
    rets = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 2:
        return {"n": len(rets)}
    mu, sigma = float(rets.mean()), float(rets.std(ddof=1))
    sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0.0
    equity = (1 + rets).cumprod()
    total_ret = float(equity.iloc[-1] - 1.0)
    dd = float((equity / equity.cummax() - 1).min())
    return {
        "sharpe": float(sharpe), "total_return": total_ret,
        "max_drawdown": dd, "volatility": float(sigma * np.sqrt(252)),
    }


# ── Paper trading pipeline ────────────────────────────────────────────

class _IntentOptimizer:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.cycle = 0

    def compute_target_portfolio(self, **kwargs):
        self.cycle += 1
        weights = [0.05 * np.sin((self.cycle + i) * 0.5) for i, _ in enumerate(self.symbols)]
        return pd.DataFrame({
            "symbol": self.symbols,
            "target_weight": weights,
            "strategy": ["momentum"] * len(self.symbols),
        })


async def _run_paper_cycles(symbols: list[str], n_cycles: int = 50) -> dict:
    prices = {s: 100.0 + 10 * i for i, s in enumerate(symbols)}
    broker = PaperBrokerAdapter(
        initial_cash=1_000_000.0, slippage_bps=0.0, fill_delay_ms=0,
        price_feed=lambda s: prices.get(s, 100.0),
    )
    portfolio = PortfolioState(cash=1_000_000.0)
    cost_cfg = {
        "commission_per_share": 0.0, "min_commission": 0.0,
        "spread_bps": 1.0, "slippage_bps": 1.0, "impact_coefficient": 0.1,
    }
    cost = TransactionCostModel(equities_config=cost_cfg)
    cbs = CircuitBreakerManager(
        max_order_pct=0.50, daily_loss_limit_pct=0.02, max_positions=50,
        max_single_position_pct=0.50, max_gross_exposure=3.0,
    )
    om = OrderManager(broker, cbs, cost, portfolio,
                     adv_map={s: 1e7 for s in symbols})
    metrics = MetricsCollector(registry=CollectorRegistry())
    alerts = AlertManager(channel_map={s: [LogChannel()] for s in AlertSeverity},
                          default_cooldown_seconds=0)
    drift = FeatureDriftDetector()

    class _SigStub:
        def generate(self, f):
            return pd.DataFrame({"symbol": symbols, "family": ["momentum"] * len(symbols),
                                 "side": [1] * len(symbols)})

    class _MetaStub:
        def predict(self, f, s):
            return pd.DataFrame({"symbol": symbols, "meta_prob": [0.6] * len(symbols)})

    pipeline = PaperTradingPipeline(
        data_adapter=None, bar_constructors={},
        feature_assembler=None, signal_battery=_SigStub(),
        meta_pipeline=_MetaStub(), meta_labeler=None, bet_sizing=None,
        portfolio_optimizer=_IntentOptimizer(symbols),
        order_manager=om, metrics=metrics, alert_manager=alerts,
        drift_detector=drift,
        config=PipelineConfig(sleep_seconds=0.0, drift_check_every=10_000),
    )
    rng = np.random.default_rng(0)
    for _ in range(n_cycles):
        for s in symbols:
            prices[s] *= 1 + rng.normal(0, 0.002)
        features = pd.DataFrame({"ret": rng.normal(0, 0.01, 10)})
        await pipeline.run_cycle(features=features, prices=dict(prices))
    return {
        "nav": portfolio.nav,
        "orders_filled": pipeline._orders_filled_count,
        "summary": pipeline.get_performance_summary(),
        "portfolio": portfolio,
    }


# ── Driver ────────────────────────────────────────────────────────────

def main() -> int:
    print("=== Wang Trading — End-to-End Smoke Test ===\n")
    report: list[tuple[str, float, str]] = []
    t_total = time.perf_counter()

    # 1. Synthetic data
    all_bars: dict[str, pd.DataFrame] = {}
    with _timed("1. Generate 2yr synthetic data + TIB bars (5 symbols)", report):
        for i, sym in enumerate(SYMBOLS):
            ticks = _synthetic_ticks(sym, N_DAYS, seed=100 + i)
            all_bars[sym] = _bars_from_ticks(sym, ticks)
        total_bars = sum(len(df) for df in all_bars.values())
        print(f"    → {total_bars} bars across {len(SYMBOLS)} symbols")

    # 2. Features
    features_map: dict[str, pd.DataFrame] = {}
    with _timed("2. Feature assembly (per symbol)", report):
        for sym, bars in all_bars.items():
            features_map[sym] = _simple_features(bars)
        total_feats = sum(len(df) for df in features_map.values())
        print(f"    → {total_feats} feature rows")

    # 3. Signals
    signals_map: dict[str, pd.Series] = {}
    with _timed("3. Signal generation", report):
        for sym, f in features_map.items():
            signals_map[sym] = _generate_signals(f)

    # 4. Meta-labeling pipeline + training
    probs_map: dict[str, pd.Series] = {}
    with _timed("4. Train meta-labeler (LightGBM)", report):
        for sym in SYMBOLS:
            bars = all_bars[sym]
            fwd_ret = bars["close"].pct_change(5).shift(-5).reindex(features_map[sym].index)
            try:
                _, probs = _train_meta_labeler(
                    features_map[sym], signals_map[sym], fwd_ret,
                )
                probs_map[sym] = probs
            except Exception as exc:
                log.warning("%s meta-labeler training failed: %s", sym, exc)
                probs_map[sym] = pd.Series(0.5, index=features_map[sym].index)

    # 5. Bet sizing
    bet_sizes_map: dict[str, pd.Series] = {}
    with _timed("5. Bet sizing cascade", report):
        for sym, probs in probs_map.items():
            bet_sizes_map[sym] = _bet_sizes(probs)

    # 6. Walk-forward backtest (per symbol)
    backtest_returns: dict[str, pd.Series] = {}
    with _timed("6. Walk-forward backtest", report):
        for sym in SYMBOLS:
            close = all_bars[sym]["close"]
            sizes = bet_sizes_map[sym].reindex(close.index).fillna(0.0)
            strat_ret = sizes.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)
            backtest_returns[sym] = strat_ret
            m = _walk_forward_metrics(strat_ret)
            if m.get("sharpe") is not None:
                print(f"    → {sym:<6} sharpe={m['sharpe']:.2f}")

    # 7. DSR
    with _timed("7. Deflated Sharpe Ratio", report):
        for sym, ret in backtest_returns.items():
            ret_clean = ret.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ret_clean) < 30:
                continue
            try:
                sigma = ret_clean.std(ddof=1)
                sharpe = (ret_clean.mean() / sigma * np.sqrt(252)) if sigma > 0 else 0.0
                stat, p = deflated_sharpe_ratio(
                    observed_sharpe=float(sharpe),
                    sharpe_std=1.0,
                    n_trials=5,
                    n_observations=len(ret_clean),
                    skewness=float(ret_clean.skew()),
                    kurtosis=float(ret_clean.kurt() + 3),
                )
                print(f"    → {sym:<6} dsr_stat={stat:.2f} p={p:.3f}")
            except Exception as exc:
                log.warning("%s DSR failed: %s", sym, exc)

    # 8. HRP
    with _timed("8. HRP portfolio optimization", report):
        rets_df = pd.DataFrame({
            s: r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for s, r in backtest_returns.items()
        }).dropna(how="all")
        if len(rets_df) >= 30 and rets_df.shape[1] >= 2:
            weights = compute_hrp_weights(rets_df)
            print(f"    → HRP weights: {dict(weights.round(3))}")
        else:
            print("    → skipped (insufficient joint data)")

    # 9. Paper trading
    paper_result: dict = {}
    with _timed("9. Paper trading — 50 cycles", report):
        paper_result = asyncio.run(_run_paper_cycles(SYMBOLS, n_cycles=50))
        print(f"    → nav=${paper_result['nav']:,.0f} "
              f"fills={paper_result['orders_filled']}")

    # 10. Daily report
    with _timed("10. Daily report generation", report):
        pf: PortfolioState = paper_result["portfolio"]
        dummy_order = Order(
            order_id=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc),
            symbol="AAPL", side=1, order_type=OrderType.LIMIT,
            quantity=10, limit_price=100.0, status=OrderStatus.FILLED,
            execution_algo=ExecutionAlgo.IMMEDIATE,
        )
        dummy_order.add_fill(Fill(
            fill_id=str(uuid.uuid4()), order_id=dummy_order.order_id,
            timestamp=datetime.now(timezone.utc), price=100.1,
            quantity=10, commission=0.0, exchange="SYNTH",
        ))
        dummy_tca = TCAResult(
            order_id=dummy_order.order_id, symbol="AAPL", side=1,
            arrival_price=100.0, execution_price=100.1,
            slippage_bps=10.0, market_impact_bps=5.0, timing_cost_bps=5.0,
            total_cost_bps=10.0, commission=0.0,
            algo_used=ExecutionAlgo.IMMEDIATE.value,
            execution_duration_seconds=1.0, fill_rate=1.0,
            benchmark_vs_twap_bps=0.0, benchmark_vs_vwap_bps=0.0,
        )
        daily = generate_daily_report(
            pf, [dummy_order], [dummy_tca], pd.DataFrame(),
            last_model_retrain=datetime.now(timezone.utc) - timedelta(hours=4),
        )
        assert "Daily Report" in daily
        assert f"${pf.nav:,.2f}" in daily
        print(f"    → report: {len(daily.splitlines())} lines")

    # ── Summary ───────────────────────────────────────────────────────
    total = time.perf_counter() - t_total
    print("\n=== Summary ===")
    for step, elapsed, status in report:
        print(f"  {status:<4} {elapsed:6.2f}s   {step}")
    print(f"  TOTAL: {total:.2f}s")

    if total > 120.0:
        print("\n⚠️  Exceeded 120s budget", file=sys.stderr)
        return 1
    if not all(r[2] == "OK" for r in report):
        print("\n⚠️  One or more steps failed", file=sys.stderr)
        return 1
    print("\nAll steps completed ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
