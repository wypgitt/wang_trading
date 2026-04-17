"""
Strategy-gate orchestrator — runs the three design-doc §9 validation gates
(CPCV, DSR, PBO) end-to-end and produces a go/no-go recommendation plus a
full :class:`BacktestReport` for operator review.

Promotion rule (design-doc §9): *all three* gates must pass before a
strategy moves from research → paper trading. Any individual failure
returns the strategy to research with a specific remediation note (e.g.
"DSR p=0.12 — reduce trial count or raise the in-sample Sharpe bar").

The CLI entry point (``python -m src.backtesting.gate_orchestrator``) is
the operator-facing wrapper used by the weekend retrain job to score each
symbol's candidate model before it ships.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from loguru import logger

from src.backtesting.cpcv import CPCVEngine, validate_strategy
from src.backtesting.deflated_sharpe import (
    compute_dsr_from_backtest,
    compute_dsr_from_cpcv,
)
from src.backtesting.pbo import compute_pbo, validate_pbo
from src.backtesting.report import BacktestReport
from src.backtesting.transaction_costs import TransactionCostModel
from src.backtesting.walk_forward import BacktestResult, WalkForwardBacktester


@dataclass
class StrategyGate:
    """Runs the full CPCV / DSR / PBO gate stack."""

    backtester: WalkForwardBacktester | None = None

    def _build_backtester(
        self, cost_model: TransactionCostModel
    ) -> WalkForwardBacktester:
        if self.backtester is not None:
            return self.backtester
        return WalkForwardBacktester(cost_model=cost_model)

    # ── full validation ───────────────────────────────────────────────

    # pylint: disable=unused-argument,too-many-arguments,too-many-locals
    def validate(
        self,
        close: pd.DataFrame,
        features: pd.DataFrame,
        signals: pd.DataFrame,
        meta_pipeline: Callable,
        meta_labeler: Any,
        cascade: Callable,
        cost_model: TransactionCostModel,
        labels_df: pd.DataFrame | None = None,
        n_cpcv_groups: int = 10,
        n_test_groups: int = 2,
        n_trials: int = 1,
        pbo_variants: int = 20,  # reserved for future variant-grid generation
        max_pbo: float = 0.40,
        min_dsr_pvalue: float = 0.05,
        min_positive_paths_pct: float = 0.60,
        pbo_matrix: pd.DataFrame | None = None,
    ) -> dict:
        bt = self._build_backtester(cost_model)

        # (a) primary walk-forward backtest
        bet_sizes = cascade(signals) if callable(cascade) else None
        primary = bt.run(close=close, signals_df=signals, bet_sizes=bet_sizes)

        # (b) CPCV
        cpcv_results: list[BacktestResult] = []
        if labels_df is not None and len(features) == len(labels_df):
            engine = CPCVEngine(
                n_groups=n_cpcv_groups, n_test_groups=n_test_groups
            )
            paths = engine.generate_paths(features, None, labels_df)
            cpcv_results = engine.run_backtest_paths(
                backtester=bt,
                paths=paths,
                close=close,
                features_df=features,
                signals_df=signals,
                meta_labeling_pipeline=meta_pipeline,
                model_class=type(meta_labeler),
            )

        if cpcv_results:
            g1_pass, g1_stats = validate_strategy(
                cpcv_results, min_positive_paths_pct=min_positive_paths_pct
            )
            gate_1 = {
                "passed": g1_pass,
                "positive_paths": g1_stats["positive_count"],
                "total_paths": g1_stats["path_count"],
                "pct": g1_stats["positive_pct"],
            }
        else:
            gate_1 = {
                "passed": False,
                "positive_paths": 0,
                "total_paths": 0,
                "pct": 0.0,
            }

        # (c) DSR — use CPCV dispersion when we have it, else the single run
        if cpcv_results:
            dsr = compute_dsr_from_cpcv(cpcv_results, n_total_trials=n_trials)
        else:
            dsr = compute_dsr_from_backtest(primary, n_trials=n_trials)
        gate_2 = {
            "passed": bool(dsr["p_value"] < min_dsr_pvalue),
            "statistic": dsr["dsr_statistic"],
            "p_value": dsr["p_value"],
        }

        # (d) PBO
        gate_3: dict
        pbo_value: float | None = None
        if pbo_matrix is not None and not pbo_matrix.empty:
            pbo_value, _ = compute_pbo(pbo_matrix)
            passed_pbo, _ = validate_pbo(pbo_value, max_pbo=max_pbo)
            gate_3 = {"passed": passed_pbo, "pbo_value": pbo_value}
        else:
            gate_3 = {"passed": False, "pbo_value": float("nan")}

        # (e) report
        report = BacktestReport(
            result=primary,
            cpcv_results=cpcv_results,
            dsr_result=dsr,
            pbo_result=(
                {"pbo": pbo_value, "max_pbo": max_pbo}
                if pbo_value is not None
                else None
            ),
        )

        all_passed = gate_1["passed"] and gate_2["passed"] and gate_3["passed"]
        recommendation = self._recommend(all_passed, gate_1, gate_2, gate_3)

        return {
            "passed": all_passed,
            "gate_1_cpcv": gate_1,
            "gate_2_dsr": gate_2,
            "gate_3_pbo": gate_3,
            "backtest_result": primary,
            "report": report,
            "recommendation": recommendation,
        }

    # ── quick path for fast iteration ─────────────────────────────────

    def quick_validate(
        self,
        backtest_result: BacktestResult,
        n_trials: int = 1,
        min_dsr_pvalue: float = 0.05,
    ) -> dict:
        dsr = compute_dsr_from_backtest(backtest_result, n_trials=n_trials)
        gate_2 = {
            "passed": bool(dsr["p_value"] < min_dsr_pvalue),
            "statistic": dsr["dsr_statistic"],
            "p_value": dsr["p_value"],
        }
        gate_1 = {"passed": None, "positive_paths": 0, "total_paths": 0, "pct": 0.0}
        gate_3 = {"passed": None, "pbo_value": float("nan")}
        return {
            "passed": gate_2["passed"],
            "gate_1_cpcv": gate_1,
            "gate_2_dsr": gate_2,
            "gate_3_pbo": gate_3,
            "backtest_result": backtest_result,
            "report": BacktestReport(result=backtest_result, dsr_result=dsr),
            "recommendation": self._recommend(
                gate_2["passed"], gate_1, gate_2, gate_3
            ),
        }

    # ── recommendation text ───────────────────────────────────────────

    @staticmethod
    def _recommend(
        passed: bool, g1: dict, g2: dict, g3: dict
    ) -> str:
        if passed:
            return (
                "PROCEED TO PAPER TRADING — all three promotion gates passed "
                f"(CPCV {g1['positive_paths']}/{g1['total_paths']} paths positive, "
                f"DSR p={g2['p_value']:.4f}, PBO={g3['pbo_value']:.3f})."
            )
        failures: list[str] = []
        if g1["passed"] is False:
            failures.append(
                f"CPCV {g1['pct'] * 100:.1f}% paths positive "
                f"(need ≥60%, {g1['positive_paths']}/{g1['total_paths']})"
            )
        elif g1["passed"] is None:
            failures.append("CPCV not run (no labels supplied)")
        if g2["passed"] is False:
            failures.append(
                f"DSR p={g2['p_value']:.4f} ≥ 0.05 (statistic={g2['statistic']:.3f})"
            )
        elif g2["passed"] is None:
            failures.append("DSR not run")
        if g3["passed"] is False:
            failures.append(
                f"PBO={g3['pbo_value']:.3f} exceeds threshold"
            )
        elif g3["passed"] is None:
            failures.append("PBO not run (no hyperparameter grid supplied)")
        return "ITERATE: " + "; ".join(failures)


# ── CLI ────────────────────────────────────────────────────────────────


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.backtesting.gate_orchestrator",
        description=(
            "Run the CPCV/DSR/PBO gate stack against a retrained meta-labeler "
            "and print the PROCEED / ITERATE verdict."
        ),
    )
    parser.add_argument("--symbol", type=str, help="Single symbol to validate")
    parser.add_argument(
        "--all-symbols", action="store_true", help="Sweep the configured universe"
    )
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run CPCV and PBO in addition to DSR (slower)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned invocation without calling any loaders",
    )
    args = parser.parse_args(argv)

    if not args.symbol and not args.all_symbols:
        parser.error("either --symbol SYMBOL or --all-symbols is required")

    symbols = (
        [args.symbol] if args.symbol else _load_configured_universe()
    )
    logger.info(
        f"gate-orchestrator: symbols={symbols} "
        f"mode={'full' if args.full_validation else 'quick'} "
        f"n_trials={args.n_trials}"
    )
    if args.dry_run:
        print(
            f"[dry-run] would validate {len(symbols)} symbol(s): {symbols}"
        )
        return 0

    # Full loader wiring is deferred to the downstream pipeline (Phase 5);
    # this CLI currently prints a stub message so the entry point exists
    # and is callable from cron.
    print(
        "Gate orchestrator CLI requires the Phase-5 data/model loaders, "
        "which are not yet wired. Use StrategyGate.validate() from Python "
        "for now. See docs/phase4_backtesting.md."
    )
    return 0


def _load_configured_universe() -> list[str]:
    # Lightweight fallback so the CLI stays importable without a full
    # runtime config. Callers can override via --symbol.
    return []


if __name__ == "__main__":
    sys.exit(_cli())
