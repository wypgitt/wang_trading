# Go-live checklist

The 17 items that must be signed off before a single live dollar is deployed.
Every box must be ticked. Every row links to the subsystem that verifies it.

## Pre-flight (system)

- [ ] **1. Preflight all-blockers-pass** — `make preflight` exits 0. Re-run
      within 30 minutes of starting live trading.
- [ ] **2. All brokers respond to heartbeat** — Alpaca, CCXT venue, IBKR
      all return healthy within 5 seconds (preflight `broker.heartbeat`).
- [ ] **3. Buying power per asset class** — each configured broker has
      ≥ $1,000 available (preflight `broker.buying_power`).
- [ ] **4. Production meta-labeler present** — MLflow has a `production`
      alias pointing to a run that passed CPCV, DSR, and PBO, trained
      within the last 30 days on ≥ 500 labeled events
      (preflight `model.*`).
- [ ] **5. Regime detector loaded and predicting** — LSTM responds to a
      smoke-test observation (preflight `model.regime_detector`).

## Paper-trading proof

- [ ] **6. 8+ weeks of paper history** — Sharpe > 1.0, max drawdown < 15%,
      win rate > 50%, ≥ 50 completed paper trades
      (preflight `paper.*`).
- [ ] **7. No unresolved SEV1/SEV2 incidents in the last 14 days** — ops
      log clean; all follow-ups closed.

## Infrastructure

- [ ] **8. TimescaleDB reachable, disk < 80%** — preflight
      `infra.timescaledb`.
- [ ] **9. Prometheus + Grafana reachable** — preflight `infra.prometheus`
      and `infra.grafana`; "Trading" dashboard loads.
- [ ] **10. Alert channel verified end-to-end** — send a synthetic test
      alert through Telegram, confirm receipt on the ops phone.
- [ ] **11. Feature store freshness** — `infra.feature_freshness` ≤ 24h.
- [ ] **12. MLflow tracking server reachable** — preflight
      `infra.mlflow`.

## Risk + operational guardrails

- [ ] **13. Risk limits sensible** — `max_single_position ≤ 10%`,
      `daily_loss_limit ≤ 2%`, all circuit breakers configured (preflight
      `risk.*`).
- [ ] **14. Dead-man's switch fresh** — `.dead_mans_switch` file < 24 h
      old (preflight `risk.dead_mans_switch`).
- [ ] **15. Secrets loaded via SecretsManager** — no API key committed to
      the repo; `WANG_SECRETS_BACKEND` configured (env/file/aws/gcp).
      `grep -rn "api_key.*=.*['\"][^'\"]" src/` returns nothing.

## Human sign-off

- [ ] **16. Operator check-in** — `.operator_checkin` touched within the
      last hour; named operator available for the first 60 minutes.
- [ ] **17. Capital deployment phase explicitly chosen** — phase 1 by
      default; any deviation requires written justification in the ops
      log.

## Execution

When all 17 boxes are ticked, a named operator runs:

```bash
sudo -u wang touch /opt/wang_trading/.operator_checkin
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.execution.preflight --full-check   # must exit 0
sudo systemctl start wang-live-trading
sudo journalctl -u wang-live-trading -f
```

Stay at the console for the first 60 minutes. If anything goes wrong,
HALT:

```bash
sudo systemctl stop wang-live-trading
```

A clean HALT is a successful trading day. An "we were lucky" day is not.
