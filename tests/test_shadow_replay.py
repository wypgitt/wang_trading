"""Tests for shadow replay target comparisons."""

from __future__ import annotations

import pandas as pd

from src.execution.shadow_replay import (
    ReplayExpectation,
    compare_target,
    format_shadow_replay_report,
    target_weights,
)


def test_target_weights_groups_symbols():
    target = pd.DataFrame([
        {"symbol": "AAPL", "target_weight": 0.03},
        {"symbol": "AAPL", "target_weight": 0.02},
        {"symbol": "MSFT", "target_weight": -0.01},
    ])

    assert target_weights(target) == {"AAPL": 0.05, "MSFT": -0.01}


def test_compare_target_flags_risk_envelope_violation():
    target = pd.DataFrame([
        {"symbol": "AAPL", "target_weight": 0.12},
    ])

    result = compare_target(
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        target=target,
        previous_weights=None,
        expectation=ReplayExpectation(max_abs_weight=0.10),
    )

    assert not result["passed"]
    assert result["violations"][0]["type"] == "max_abs_weight"


def test_shadow_replay_report_is_operator_readable():
    report = format_shadow_replay_report({
        "as_of": "2026-01-01T00:00:00Z",
        "passed": True,
        "symbols": ["AAPL"],
        "cycles": 1,
        "violations": [],
        "results": [{
            "timestamp": "2026-01-01T00:00:00Z",
            "symbol": "AAPL",
            "passed": True,
            "gross": 0.05,
            "weights": {"AAPL": 0.05},
        }],
    })

    assert "Shadow Replay Report" in report
    assert "Status: PASS" in report
    assert "AAPL=5.00%" in report
