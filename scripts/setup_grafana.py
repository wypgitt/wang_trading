"""Provision Grafana datasources, dashboards, and alert rules.

Usage:
    python scripts/setup_grafana.py \
        --grafana-url http://localhost:3000 \
        --api-key <grafana-api-key>

Uses the Grafana HTTP API. Datasources default to:
    Prometheus → http://prometheus:9090
    TimescaleDB → postgres://quant:password@timescaledb:5432/quantsystem

Override with --prometheus-url / --timescale-url if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

# Allow script to be run from repo root or from scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.dashboards import (
    generate_alerting_rules,
    generate_main_dashboard,
)


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def create_datasource(url: str, api_key: str, payload: dict) -> None:
    r = requests.post(
        f"{url}/api/datasources", headers=_headers(api_key),
        data=json.dumps(payload), timeout=10,
    )
    if r.status_code in (200, 201):
        print(f"  ✓ datasource {payload['name']} created")
    elif r.status_code == 409:
        print(f"  • datasource {payload['name']} already exists")
    else:
        print(f"  ✗ datasource {payload['name']} failed: {r.status_code} {r.text}")


def import_dashboard(url: str, api_key: str, dashboard: dict) -> None:
    payload = {"dashboard": dashboard, "overwrite": True, "folderId": 0}
    r = requests.post(
        f"{url}/api/dashboards/db", headers=_headers(api_key),
        data=json.dumps(payload), timeout=10,
    )
    if r.status_code in (200, 201):
        print(f"  ✓ dashboard imported (uid={dashboard.get('uid')})")
    else:
        print(f"  ✗ dashboard import failed: {r.status_code} {r.text}")


def push_alert_rules(url: str, api_key: str, rules: dict) -> None:
    # Grafana v10+ provisioning API: PUT /api/v1/provisioning/alert-rules
    for group in rules.get("groups", []):
        for rule in group.get("rules", []):
            r = requests.post(
                f"{url}/api/v1/provisioning/alert-rules",
                headers=_headers(api_key), data=json.dumps(rule), timeout=10,
            )
            if r.status_code in (200, 201, 202):
                print(f"  ✓ alert rule {rule['uid']} pushed")
            else:
                print(
                    f"  ✗ alert rule {rule['uid']} failed: "
                    f"{r.status_code} {r.text[:200]}"
                )


def configure_telegram(
    url: str, api_key: str, bot_token: str, chat_id: str
) -> None:
    payload = {
        "name": "telegram",
        "type": "telegram",
        "isDefault": True,
        "settings": {"bottoken": bot_token, "chatid": chat_id},
    }
    r = requests.post(
        f"{url}/api/alert-notifications", headers=_headers(api_key),
        data=json.dumps(payload), timeout=10,
    )
    if r.status_code in (200, 201):
        print("  ✓ Telegram notification channel configured")
    elif r.status_code == 409:
        print("  • Telegram channel already exists")
    else:
        print(f"  ✗ Telegram config failed: {r.status_code} {r.text[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grafana-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--prometheus-url", default="http://prometheus:9090")
    ap.add_argument("--timescale-url",
                    default="postgres://quant:password@timescaledb:5432/quantsystem")
    ap.add_argument("--telegram-bot-token", default="")
    ap.add_argument("--telegram-chat-id", default="")
    args = ap.parse_args()

    url = args.grafana_url.rstrip("/")

    print("Creating datasources…")
    create_datasource(url, args.api_key, {
        "name": "Prometheus", "type": "prometheus",
        "access": "proxy", "url": args.prometheus_url, "isDefault": True,
    })
    create_datasource(url, args.api_key, {
        "name": "TimescaleDB", "type": "postgres",
        "access": "proxy", "url": args.timescale_url,
        "database": "quantsystem", "user": "quant",
        "jsonData": {"sslmode": "disable", "postgresVersion": 1600},
        "secureJsonData": {"password": "password"},
    })

    print("Importing dashboard…")
    import_dashboard(url, args.api_key, generate_main_dashboard())

    print("Pushing alert rules…")
    push_alert_rules(url, args.api_key, generate_alerting_rules())

    if args.telegram_bot_token and args.telegram_chat_id:
        print("Configuring Telegram notification channel…")
        configure_telegram(url, args.api_key, args.telegram_bot_token,
                           args.telegram_chat_id)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
