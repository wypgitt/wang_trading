"""Tests for Grafana dashboard/alert-rule generators (Phase 5 / P5.11)."""

from __future__ import annotations

import json

import pytest

from src.monitoring.dashboards import (
    generate_alerting_rules,
    generate_main_dashboard,
)


EXPECTED_ROW_TITLES = {
    "Portfolio Overview",
    "Positions",
    "Signals & Model",
    "Execution",
    "Data Health",
    "Risk",
}


class TestMainDashboard:
    def test_returns_dict(self):
        d = generate_main_dashboard()
        assert isinstance(d, dict)
        assert d["uid"] == "wang-trading-main"
        assert d["title"].startswith("Wang Trading")

    def test_roundtrips_as_json(self):
        d = generate_main_dashboard()
        blob = json.dumps(d)
        parsed = json.loads(blob)
        assert parsed["uid"] == d["uid"]

    def test_contains_expected_rows(self):
        d = generate_main_dashboard()
        row_titles = {p["title"] for p in d["panels"] if p["type"] == "row"}
        assert EXPECTED_ROW_TITLES.issubset(row_titles)

    def test_panels_have_unique_ids_and_grid(self):
        d = generate_main_dashboard()
        ids = [p["id"] for p in d["panels"]]
        assert len(ids) == len(set(ids))
        for p in d["panels"]:
            assert "gridPos" in p
            grid = p["gridPos"]
            assert {"x", "y", "w", "h"}.issubset(grid.keys())

    def test_has_minimum_panel_count(self):
        d = generate_main_dashboard()
        # 6 rows + > 10 data panels
        assert len(d["panels"]) >= 16

    def test_targets_reference_metrics(self):
        d = generate_main_dashboard()
        data_panels = [p for p in d["panels"] if p["type"] != "row"]
        for p in data_panels:
            assert p["targets"], f"panel {p['title']} has no targets"


class TestAlertRules:
    def test_returns_rules(self):
        rules = generate_alerting_rules()
        assert rules["apiVersion"] == 1
        assert rules["groups"]
        rule_list = rules["groups"][0]["rules"]
        uids = {r["uid"] for r in rule_list}
        # Expected rule uids
        assert {"dd-warning", "dd-critical", "daily-loss",
                "model-stale", "data-gap", "feature-drift"}.issubset(uids)

    def test_severity_labels(self):
        rules = generate_alerting_rules()
        for rule in rules["groups"][0]["rules"]:
            assert rule["labels"]["severity"] in {"warning", "critical"}

    def test_drawdown_thresholds_increasing(self):
        rules = generate_alerting_rules()
        by_uid = {r["uid"]: r for r in rules["groups"][0]["rules"]}
        assert by_uid["dd-warning"]["threshold"] < by_uid["dd-critical"]["threshold"]
