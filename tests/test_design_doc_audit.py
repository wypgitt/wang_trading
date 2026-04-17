"""Tests for the design-doc conformance auditor (C5)."""

from __future__ import annotations

import pytest

from scripts.design_doc_audit import main, run_audit

pytestmark = pytest.mark.integration


class TestAuditRuns:
    def test_runs_to_completion(self):
        sections, overall = run_audit()
        assert len(sections) >= 9  # we defined nine sections
        assert 0.0 <= overall <= 100.0

    def test_core_subsystems_detected(self):
        sections, _ = run_audit()
        by_name = {s.name: s for s in sections}
        # These four must be fully implemented
        for section_name in (
            "DB tables", "Validation gates",
            "Portfolio optimizers", "Execution algorithms",
            "Broker adapters", "Alert templates",
        ):
            assert section_name in by_name
            assert by_name[section_name].pct == 100.0, (
                f"{section_name}: "
                f"{by_name[section_name].present}/{by_name[section_name].total} "
                f"({by_name[section_name].pct:.1f}%)"
            )

    def test_overall_above_threshold(self):
        _, overall = run_audit()
        assert overall >= 95.0, f"conformance dropped to {overall:.1f}%"

    def test_cli_exit_code_zero_when_above_threshold(self, capsys):
        code = main([])
        captured = capsys.readouterr()
        assert "OVERALL" in captured.out
        assert code == 0


class TestCheckIdentification:
    def test_known_present_items(self):
        sections, _ = run_audit()
        present_map = {
            c.name: c.present
            for s in sections for c in s.checks
        }
        # Recently-added C1 tables and audit log must be detected
        for name in ("labels", "meta_labels", "positions_history", "audit_log",
                     "bars", "orders", "fills"):
            assert present_map.get(name) is True, f"{name!r} should be detected"
        # Core gates
        for gate in ("CPCV", "DSR", "PBO"):
            assert present_map.get(gate) is True
        # All four broker adapters
        for broker in ("paper", "alpaca", "ccxt", "ibkr"):
            assert present_map.get(broker) is True
