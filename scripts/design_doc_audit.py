#!/usr/bin/env python3
"""Design-doc conformance audit (C5).

Walks the repo for evidence that every item called out in the design doc is
implemented. Evidence is intentionally loose — a DB table name in a schema
file, a class name in a module, a marker string in a template — so the audit
stays fast and additive as the spec evolves.

Exit code:
    0  overall coverage >= 95%
    1  overall coverage <  95%
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent


# ── Evidence helpers ─────────────────────────────────────────────────────

def _any_file_contains(patterns: list[str], globs: list[str]) -> bool:
    for pat in globs:
        for path in ROOT.glob(pat):
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if all(re.search(p, text, re.IGNORECASE) for p in patterns):
                return True
    return False


def _identifier(name: str, globs: list[str]) -> Callable[[], bool]:
    return lambda: _any_file_contains([rf"\b{name}\b"], globs)


def _table(name: str) -> Callable[[], bool]:
    return _identifier(name, ["src/**/*.py", "src/**/*.sql"])


def _class(name: str) -> Callable[[], bool]:
    return lambda: _any_file_contains([rf"class\s+{name}\b"], ["src/**/*.py"])


def _feature_module(module: str, tokens: list[str]) -> Callable[[], bool]:
    def check() -> bool:
        path = ROOT / "src" / "feature_factory" / f"{module}.py"
        if not path.exists():
            return False
        text = path.read_text(errors="ignore")
        return any(tok.lower() in text.lower() for tok in tokens)
    return check


# ── Spec definition ──────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    description: str
    check: Callable[[], bool]
    present: bool = False


@dataclass
class Section:
    name: str
    checks: list[Check] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def present(self) -> int:
        return sum(1 for c in self.checks if c.present)

    @property
    def pct(self) -> float:
        return (self.present / self.total * 100.0) if self.total else 100.0


def _build_spec() -> list[Section]:
    return [
        Section("DB tables", [
            Check(name, f"hypertable `{name}`",
                  _table(name))
            for name in (
                "bars", "raw_ticks", "cusum_events", "features", "signals",
                "labels", "meta_labels", "positions_history",
                "orders", "fills", "tca_results", "portfolio_snapshots",
                "audit_log",
            )
        ]),
        Section("Feature families", [
            Check("FFD", "fractional differentiation",
                  _feature_module("fractional_diff",
                                  ["frac_diff_ffd", "fractional"])),
            Check("structural_breaks", "CUSUM / SADF / explosive regimes",
                  _feature_module("structural_breaks",
                                  ["compute_structural_break_features"])),
            Check("entropy", "entropy features",
                  _feature_module("entropy", ["compute_entropy_features"])),
            Check("microstructure", "Kyle/Amihud/VPIN",
                  _feature_module("microstructure",
                                  ["compute_microstructure_features"])),
            Check("GARCH", "volatility / GARCH features",
                  _feature_module("volatility", ["garch", "compute_volatility_features"])),
            Check("NLP", "FinBERT sentiment",
                  _identifier("FinBERT",
                              ["src/feature_factory/**/*.py"])),
            Check("on-chain", "on-chain crypto features",
                  lambda: _any_file_contains(
                      [r"on.?chain"],
                      ["src/feature_factory/**/*.py", "src/signal_battery/**/*.py"],
                  )),
            Check("autoencoder", "autoencoder latents",
                  lambda: (ROOT / "src" / "feature_factory" / "autoencoder.py").exists()),
            Check("classical", "RSI / Bollinger / classical indicators",
                  _feature_module("assembler", ["_rsi", "_bollinger_width"])),
        ]),
        Section("Signal families", [
            Check(fam, f"signal family `{fam}`",
                  lambda f=fam: _any_file_contains(
                      [rf"{f}"], ["src/signal_battery/**/*.py"]
                  ))
            for fam in (
                "momentum", "mean_reversion", "stat_arb", "trend",
                "carry", "cross_exchange", "vol_premium",
            )
        ]),
        Section("Validation gates", [
            Check("CPCV", "combinatorial purged cross-validation",
                  _class("CPCV") if False else lambda: _any_file_contains(
                      [r"CPCV|combinatorial_purged"],
                      ["src/backtesting/**/*.py", "src/ml_layer/**/*.py"],
                  )),
            Check("DSR", "deflated Sharpe ratio",
                  lambda: _any_file_contains(
                      [r"deflated.*sharpe|DSR\b"],
                      ["src/backtesting/**/*.py"],
                  )),
            Check("PBO", "probability of backtest overfitting",
                  lambda: _any_file_contains(
                      [r"\bPBO\b|probability_of_backtest_overfitting"],
                      ["src/backtesting/**/*.py"],
                  )),
        ]),
        Section("Circuit breakers", [
            Check(name, f"circuit breaker `{name}`",
                  lambda token=token: _any_file_contains(
                      [token], ["src/execution/circuit_breakers.py"],
                  ))
            for name, token in (
                ("fat_finger", r"fat.?finger|MAX_ORDER_SIZE"),
                ("daily_loss", r"daily.?loss|MAX_DAILY_LOSS"),
                ("drawdown", r"drawdown"),
                ("max_positions", r"max_positions|MAX_POSITIONS"),
                ("gross_exposure", r"gross.?exposure"),
                ("crypto_limit", r"crypto|spot"),
                ("model_staleness", r"model.?stale"),
                ("connectivity", r"connectivity|heartbeat"),
                ("dead_man_switch", r"dead.?man"),
                ("correlation", r"correlation"),
            )
        ]),
        Section("Alert templates", [
            Check(name, f"alert template `{name}`",
                  lambda pat=pat: _any_file_contains(
                      [pat], ["src/monitoring/alerting.py"],
                  ))
            for name, pat in (
                ("drawdown", r"alert_drawdown|drawdown"),
                ("daily_loss", r"alert_daily_loss|daily_loss"),
                ("circuit_breaker", r"alert_circuit_breaker"),
                ("model_stale", r"alert_model_stale|model_stale"),
                ("data_gap", r"alert_data_gap|data_gap"),
                ("execution_failure", r"alert_execution_failure"),
                ("position_reconciliation", r"alert_position_reconciliation|reconciliation"),
                ("feature_drift", r"alert_feature_drift|feature_drift"),
            )
        ]),
        Section("Portfolio optimizers", [
            Check("HRP", "hierarchical risk parity",
                  lambda: _any_file_contains(
                      [r"\bHRP\b|hierarchical_risk_parity"],
                      ["src/portfolio/**/*.py"],
                  )),
            Check("risk_parity", "coordinate-descent risk parity",
                  lambda: _any_file_contains(
                      [r"risk_parity|RiskParity"],
                      ["src/portfolio/**/*.py"],
                  )),
            Check("multi_strategy", "multi-strategy allocator",
                  lambda: _any_file_contains(
                      [r"multi.?strategy|MultiStrategy"],
                      ["src/portfolio/**/*.py"],
                  )),
        ]),
        Section("Execution algorithms", [
            Check(name, f"execution algo `{name}`",
                  lambda cls=cls: _any_file_contains(
                      [rf"class\s+{cls}"], ["src/execution/algorithms.py"],
                  ))
            for name, cls in (
                ("immediate", "ImmediateAlgo"),
                ("twap", "TWAPAlgo"),
                ("vwap", "VWAPAlgo"),
                ("iceberg", "IcebergAlgo"),
            )
        ]),
        Section("Broker adapters", [
            Check("paper", "PaperBrokerAdapter", _class("PaperBrokerAdapter")),
            Check("alpaca", "AlpacaBrokerAdapter", _class("AlpacaBrokerAdapter")),
            Check("ccxt", "CCXTBrokerAdapter", _class("CCXTBrokerAdapter")),
            Check("ibkr", "IBKRBrokerAdapter", _class("IBKRBrokerAdapter")),
        ]),
    ]


# ── Runner ───────────────────────────────────────────────────────────────

def run_audit() -> tuple[list[Section], float]:
    sections = _build_spec()
    total_present = 0
    total_checks = 0
    for sec in sections:
        for chk in sec.checks:
            try:
                chk.present = bool(chk.check())
            except Exception:
                chk.present = False
        total_present += sec.present
        total_checks += sec.total
    overall_pct = (total_present / total_checks * 100.0) if total_checks else 100.0
    return sections, overall_pct


def _print_report(sections: list[Section], overall: float, *, verbose: bool = True) -> None:
    print("=" * 68)
    print("wang_trading design-doc conformance audit")
    print("=" * 68)
    for sec in sections:
        print(f"\n[{sec.pct:5.1f}%]  {sec.name}  ({sec.present}/{sec.total})")
        if verbose:
            for c in sec.checks:
                mark = "OK " if c.present else "-- "
                print(f"   {mark} {c.name:28s}  {c.description}")
    print("\n" + "=" * 68)
    status = "PASS" if overall >= 95.0 else "FAIL"
    print(f"OVERALL: {overall:.1f}%  ({status}, threshold 95%)")
    print("=" * 68)


def _as_dict(sections: list[Section], overall: float) -> dict:
    return {
        "overall_pct": overall,
        "sections": [
            {
                "name": s.name,
                "present": s.present,
                "total": s.total,
                "pct": s.pct,
                "checks": [
                    {"name": c.name, "present": c.present,
                     "description": c.description}
                    for c in s.checks
                ],
            }
            for s in sections
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="design_doc_audit")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of the text report")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-check detail")
    args = parser.parse_args(argv)

    sections, overall = run_audit()
    if args.json:
        print(json.dumps(_as_dict(sections, overall), indent=2))
    else:
        _print_report(sections, overall, verbose=not args.quiet)
    return 0 if overall >= 95.0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
