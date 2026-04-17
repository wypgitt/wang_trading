"""
Standardized backtest report (design-doc §9.5).

Produces a single text artifact plus a set of tabular DataFrames summarising
one backtest run: headline metrics, gate-by-gate promotion decision (CPCV /
DSR / PBO), per-trade log, monthly returns, regime-conditional performance,
strategy-family breakdown, drawdown table, and (when available) top-feature
importance.

Intended to be the operator-facing "did this thing earn its seat?" sheet
before a model is promoted from research → paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.cpcv import CPCVEngine, validate_strategy
from src.backtesting.deflated_sharpe import compute_dsr_from_backtest
from src.backtesting.pbo import validate_pbo
from src.backtesting.walk_forward import BacktestResult


@dataclass
class _GateVerdict:
    name: str
    passed: bool | None
    detail: str


class BacktestReport:
    def __init__(
        self,
        result: BacktestResult,
        cpcv_results: list[BacktestResult] | None = None,
        dsr_result: dict | None = None,
        pbo_result: dict | None = None,
        feature_importance: pd.DataFrame | None = None,
        vol_series: pd.Series | None = None,
    ) -> None:
        self.result = result
        self.cpcv_results = cpcv_results or []
        self.dsr_result = dsr_result
        self.pbo_result = pbo_result
        self.feature_importance = feature_importance
        self.vol_series = vol_series

    # ── gates ──────────────────────────────────────────────────────────

    def _gate_cpcv(self) -> _GateVerdict:
        if not self.cpcv_results:
            return _GateVerdict("CPCV", None, "not run")
        passed, stats = validate_strategy(self.cpcv_results)
        detail = (
            f"{stats['positive_count']}/{stats['path_count']} paths positive "
            f"({stats['positive_pct']*100:.1f}%, threshold "
            f"{stats['threshold']*100:.0f}%)"
        )
        return _GateVerdict("CPCV", passed, detail)

    def _gate_dsr(self) -> _GateVerdict:
        if not self.dsr_result:
            return _GateVerdict("DSR", None, "not run")
        d = self.dsr_result
        detail = (
            f"statistic={d['dsr_statistic']:.3f}, p={d['p_value']:.4f} "
            f"(SR={d['observed_sharpe']:.2f} vs E[max]={d['expected_max_sharpe']:.2f}, "
            f"n_trials={d['n_trials']})"
        )
        return _GateVerdict("DSR", bool(d.get("passed", False)), detail)

    def _gate_pbo(self) -> _GateVerdict:
        if not self.pbo_result:
            return _GateVerdict("PBO", None, "not run")
        d = self.pbo_result
        pbo_val = d["pbo"]
        passed, msg = validate_pbo(pbo_val, max_pbo=d.get("max_pbo", 0.40))
        return _GateVerdict("PBO", passed, f"{pbo_val:.3f} — {msg}")

    def overall_verdict(self) -> tuple[str, list[str]]:
        gates = [self._gate_cpcv(), self._gate_dsr(), self._gate_pbo()]
        failures = [g.name for g in gates if g.passed is False]
        ran = [g for g in gates if g.passed is not None]
        if failures:
            return "FAIL", failures
        if not ran:
            return "INCOMPLETE", []
        return "PASS", []

    # ── tabular extractions ────────────────────────────────────────────

    def _monthly_returns(self) -> pd.DataFrame:
        eq = self.result.equity_curve
        if not isinstance(eq.index, pd.DatetimeIndex) or len(eq) < 2:
            return pd.DataFrame()
        monthly = eq.resample("ME").last().pct_change().dropna()
        if monthly.empty:
            return pd.DataFrame()
        tbl = monthly.to_frame("return")
        tbl["year"] = tbl.index.year
        tbl["month"] = tbl.index.month
        return tbl.pivot_table(
            index="year", columns="month", values="return"
        )

    def _trade_log(self) -> pd.DataFrame:
        if not self.result.trades:
            return pd.DataFrame()
        rows = []
        for t in self.result.trades:
            rows.append(
                {
                    "entry_timestamp": t.entry_timestamp,
                    "exit_timestamp": t.exit_timestamp,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "signal_family": t.signal_family,
                    "gross_pnl": t.gross_pnl,
                    "total_cost": t.costs.total_cost,
                    "net_pnl": t.net_pnl,
                    "holding_period_bars": t.holding_period_bars,
                    "meta_label_prob": t.meta_label_prob,
                    "return_pct": t.return_pct,
                }
            )
        return pd.DataFrame(rows)

    def _strategy_breakdown(self) -> pd.DataFrame:
        log = self._trade_log()
        if log.empty:
            return pd.DataFrame()
        grouped = log.groupby("signal_family").agg(
            n_trades=("net_pnl", "size"),
            win_rate=("net_pnl", lambda x: (x > 0).mean()),
            avg_net_pnl=("net_pnl", "mean"),
            total_net_pnl=("net_pnl", "sum"),
            avg_holding=("holding_period_bars", "mean"),
        )
        return grouped

    def _drawdown_table(self, top_n: int = 5) -> pd.DataFrame:
        eq = self.result.equity_curve
        if len(eq) < 3:
            return pd.DataFrame()
        running = eq.cummax()
        dd = eq / running - 1.0
        # Identify drawdown runs
        runs: list[dict] = []
        start: Any = None
        trough_ts = None
        trough_val = 0.0
        for ts, val in dd.items():
            if val < 0:
                if start is None:
                    start = ts
                    trough_ts = ts
                    trough_val = val
                elif val < trough_val:
                    trough_val = val
                    trough_ts = ts
            else:
                if start is not None:
                    runs.append(
                        {
                            "start": start,
                            "trough": trough_ts,
                            "end": ts,
                            "depth": trough_val,
                            "duration_bars": int(
                                eq.index.get_loc(ts) - eq.index.get_loc(start)
                            ),
                        }
                    )
                    start = None
                    trough_val = 0.0
        if start is not None:
            runs.append(
                {
                    "start": start,
                    "trough": trough_ts,
                    "end": eq.index[-1],
                    "depth": trough_val,
                    "duration_bars": int(
                        eq.index.get_loc(eq.index[-1]) - eq.index.get_loc(start)
                    ),
                }
            )
        if not runs:
            return pd.DataFrame()
        df = pd.DataFrame(runs).sort_values("depth").head(top_n).reset_index(drop=True)
        return df

    def _regime_breakdown(self) -> pd.DataFrame:
        if self.vol_series is None or len(self.result.returns) < 10:
            return pd.DataFrame()
        vol = self.vol_series.reindex(self.result.returns.index).ffill().bfill()
        median = float(vol.median())
        hi = self.result.returns[vol > median]
        lo = self.result.returns[vol <= median]

        def _stats(r: pd.Series) -> dict:
            if len(r) < 2 or r.std(ddof=0) == 0:
                return {"mean": float(r.mean()) if len(r) else 0.0, "vol": 0.0, "sharpe": 0.0}
            return {
                "mean": float(r.mean()),
                "vol": float(r.std(ddof=0)),
                "sharpe": float(r.mean() / r.std(ddof=0) * np.sqrt(252)),
            }

        return pd.DataFrame(
            {"high_vol": _stats(hi), "low_vol": _stats(lo)}
        ).T

    def _cpcv_path_stats(self) -> pd.DataFrame:
        if not self.cpcv_results:
            return pd.DataFrame()
        return CPCVEngine.get_path_statistics(self.cpcv_results)

    # ── public surface ─────────────────────────────────────────────────

    def generate_dataframes(self) -> dict[str, pd.DataFrame]:
        return {
            "monthly_returns": self._monthly_returns(),
            "drawdown_table": self._drawdown_table(),
            "trade_log": self._trade_log(),
            "strategy_breakdown": self._strategy_breakdown(),
            "cpcv_path_stats": self._cpcv_path_stats(),
            "regime_breakdown": self._regime_breakdown(),
        }

    def generate_text_report(self) -> str:
        m = self.result.metrics
        verdict, failures = self.overall_verdict()
        gates = [self._gate_cpcv(), self._gate_dsr(), self._gate_pbo()]

        def _fmt(key: str, spec: str = ".4f") -> str:
            v = m.get(key, 0.0)
            if v == float("inf"):
                return "inf"
            return format(v, spec)

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 72)
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 72)
        lines.append(f"  Total return               : {_fmt('total_return', '.2%')}")
        lines.append(f"  Annualised return          : {_fmt('annualized_return', '.2%')}")
        lines.append(f"  Annualised vol             : {_fmt('annualized_vol', '.2%')}")
        lines.append(f"  Sharpe ratio               : {_fmt('sharpe', '.3f')}")
        lines.append(f"  Sortino ratio              : {_fmt('sortino', '.3f')}")
        lines.append(f"  Calmar ratio               : {_fmt('calmar', '.3f')}")
        lines.append(f"  Max drawdown               : {_fmt('max_drawdown', '.2%')}")
        lines.append(f"  Max drawdown duration (bar): {int(m.get('max_drawdown_duration_bars', 0))}")
        lines.append("")
        lines.append("PROMOTION GATES")
        lines.append("-" * 72)
        for g in gates:
            status = (
                "PASS" if g.passed is True
                else "FAIL" if g.passed is False
                else "n/a"
            )
            lines.append(f"  [{status:4}] {g.name:4} — {g.detail}")
        lines.append("")
        lines.append(f"  OVERALL VERDICT: {verdict}")
        if failures:
            lines.append(f"    failed gates: {', '.join(failures)}")
        lines.append("")
        lines.append("TRADE STATISTICS")
        lines.append("-" * 72)
        lines.append(f"  Total trades               : {int(m.get('total_trades', 0))}")
        lines.append(f"  Win rate                   : {_fmt('win_rate', '.2%')}")
        lines.append(f"  Profit factor              : {_fmt('profit_factor', '.3f')}")
        lines.append(f"  Average trade net P&L      : {_fmt('avg_trade', '.2f')}")
        lines.append(f"  Avg holding period (bars)  : {_fmt('avg_holding_period_bars', '.1f')}")
        lines.append(f"  Trades per month           : {_fmt('trades_per_month', '.2f')}")
        lines.append("")
        lines.append("COST ANALYSIS")
        lines.append("-" * 72)
        total_costs = sum(t.costs.total_cost for t in self.result.trades)
        gross = sum(t.gross_pnl for t in self.result.trades)
        pct_of_gross = (
            total_costs / abs(gross) * 100 if gross != 0 else 0.0
        )
        lines.append(f"  Total costs (cash)         : {total_costs:,.2f}")
        lines.append(f"  Cost drag                  : {_fmt('cost_drag_bps', '.1f')} bps")
        lines.append(f"  Turnover                   : {_fmt('turnover', '.2f')}")
        lines.append(f"  Costs / |gross profit|     : {pct_of_gross:.2f}%")
        lines.append("")

        dfs = self.generate_dataframes()

        if not dfs["monthly_returns"].empty:
            lines.append("MONTHLY RETURNS")
            lines.append("-" * 72)
            lines.append(
                dfs["monthly_returns"].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else ""
                ).to_string()
            )
            lines.append("")

        if not dfs["regime_breakdown"].empty:
            lines.append("REGIME-CONDITIONAL PERFORMANCE (split at median vol)")
            lines.append("-" * 72)
            lines.append(dfs["regime_breakdown"].round(4).to_string())
            lines.append("")

        if not dfs["strategy_breakdown"].empty:
            lines.append("STRATEGY-FAMILY BREAKDOWN")
            lines.append("-" * 72)
            lines.append(dfs["strategy_breakdown"].round(4).to_string())
            lines.append("")

        if not dfs["drawdown_table"].empty:
            lines.append("TOP DRAWDOWNS")
            lines.append("-" * 72)
            lines.append(dfs["drawdown_table"].to_string(index=False))
            lines.append("")

        if self.feature_importance is not None and not self.feature_importance.empty:
            lines.append("FEATURE IMPORTANCE (top 10 by MDA)")
            lines.append("-" * 72)
            fi = self.feature_importance
            col = (
                "mda_mean" if "mda_mean" in fi.columns
                else fi.select_dtypes(include="number").columns[0]
            )
            top = fi.sort_values(col, ascending=False).head(10)
            lines.append(top.to_string())
            lines.append("")

        if not dfs["cpcv_path_stats"].empty:
            lines.append("CPCV PATH STATISTICS (summary)")
            lines.append("-" * 72)
            summary_rows = ["mean", "std", "min", "max"]
            present = [r for r in summary_rows if r in dfs["cpcv_path_stats"].index]
            if present:
                lines.append(
                    dfs["cpcv_path_stats"].loc[present].round(4).to_string()
                )
                lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    def save_report(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.txt").write_text(self.generate_text_report())
        for name, df in self.generate_dataframes().items():
            if df is None or df.empty:
                continue
            df.to_csv(out / f"{name}.csv")
        # Also persist the equity/drawdown series for plotting downstream
        self.result.equity_curve.to_csv(out / "equity_curve.csv", header=True)
        self.result.drawdown_curve.to_csv(out / "drawdown_curve.csv", header=True)


# ── end-to-end convenience ─────────────────────────────────────────────


def generate_report_from_pipeline(
    backtester,
    cpcv_engine: CPCVEngine,
    close: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.DataFrame,
    meta_pipeline,
    model: Any,
    cascade,
    n_trials: int = 1,
    labels_df: pd.DataFrame | None = None,
    pbo_matrix: pd.DataFrame | None = None,
) -> BacktestReport:
    """Full-pipeline convenience: walk-forward backtest + CPCV + DSR + PBO."""

    from src.backtesting.pbo import compute_pbo

    bets = cascade(signals) if callable(cascade) else None
    primary = backtester.run(
        close=close,
        signals_df=signals,
        bet_sizes=bets,
    )

    cpcv_results: list[BacktestResult] = []
    if labels_df is not None:
        paths = cpcv_engine.generate_paths(features, None, labels_df)
        cpcv_results = cpcv_engine.run_backtest_paths(
            backtester=backtester,
            paths=paths,
            close=close,
            features_df=features,
            signals_df=signals,
            meta_labeling_pipeline=meta_pipeline,
            model_class=type(model),
        )

    dsr = compute_dsr_from_backtest(primary, n_trials=n_trials)

    pbo_result = None
    if pbo_matrix is not None and not pbo_matrix.empty:
        pbo_val, _ = compute_pbo(pbo_matrix)
        pbo_result = {"pbo": pbo_val, "max_pbo": 0.40}

    return BacktestReport(
        result=primary,
        cpcv_results=cpcv_results,
        dsr_result=dsr,
        pbo_result=pbo_result,
    )
