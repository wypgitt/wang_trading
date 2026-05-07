# Daily operations

Follow this runbook every trading day. Equities follow the US cash session;
crypto trades 24/7 so the weekend section applies there.

## Morning checklist (T-60 minutes to open)

1. **Operator check-in.** Touch the sentinel to prove a human is present:
   ```bash
   sudo -u wang touch /opt/wang_trading/.operator_checkin
   ```
   Without this the pipeline refuses to restart.

2. **Preflight.** Run the full check; expect exit 0.
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python \
       -m src.execution.preflight --full-check \
       --config /opt/wang_trading/config/live_trading.yaml
   ```
   Any blocker → STOP. Do not proceed with live trading until resolved.

3. **Broker heartbeats.** Confirm all three adapters respond:
   - Alpaca: preflight `broker.heartbeat` + `broker.buying_power`
   - CCXT venues: same
   - IBKR: TWS / Gateway must be up *before* the service starts.

4. **Data freshness.**
   - Grafana dashboard "Ingestion freshness" — every asset class ≤ 5 min.
   - Feature store heartbeat: `tail -F /var/log/wang_trading/monitoring.out.log`.

5. **Model age + gates.** Preflight covers this; spot-check MLflow for
   the latest production run ID.

6. **Shadow replay.** Replay recent bars through the exact live target stack:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python \
       /opt/wang_trading/scripts/shadow_replay.py \
       --config /opt/wang_trading/config/live_trading.yaml \
       --output /opt/wang_trading/logs/shadow_replay_report.md
   ```
   A FAIL means do not start live trading until the target violation is understood.

7. **Paper/live divergence.** Generate the morning operator report:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python \
       -m src.execution.daily_ops --paper-live-divergence \
       --paper-returns-csv /opt/wang_trading/logs/paper_returns.csv \
       --live-returns-csv /opt/wang_trading/logs/live_returns.csv
   ```

8. **Review overnight alerts.** Telegram channel → acknowledge or escalate.

## Intraday

- Dashboards to keep open in another tab:
  - **Trading** — NAV, drawdown, open orders, reconciliation diffs.
  - **Ingestion** — per-venue lag.
  - **RL shadow** — divergence between HRP and RL target.
- Spot-check the compliance log once per session:
  ```bash
  tail -F /opt/wang_trading/logs/live_trading_compliance.log
  ```
- Do **not** bypass any alert. A muted alert becomes tomorrow's incident.

## End-of-day (EOD) — equities close

1. Confirm the pipeline has flattened to the target book. If a run-on order
   is stuck, cancel it manually:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python \
       -m src.execution.live_trading --config ... --cancel-open-orders
   ```
2. Snapshot verification:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python -c \
       "from src.execution.disaster_recovery import SnapshotManager; \
        print(SnapshotManager().verify_snapshot_chain())"
   ```
3. Drop a one-line note in the ops log (date, NAV, anything notable).

## Weekend (crypto 24/7)

Crypto never closes. On weekends we do the *same* morning checklist but the
staffing expectation is **monitor-only unless alerted**.

- No new code deploys Friday 16:00 → Monday 09:00 ET.
- Primary on-call monitors Telegram; secondary is 30-min recall.
- If an incident requires HALT, do it — losing a few hours of crypto alpha
  beats losing tomorrow's confidence.

## Off-hours shutdown

Only do this if you cannot monitor:
```bash
sudo systemctl stop wang-live-trading
```
This writes `.live_halt`. The next operator must remove it + re-run
preflight before restarting.

## Kill-switch commands

- Halt new live starts:
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python \
      -m src.execution.live_trading \
      --config /opt/wang_trading/config/live_trading.yaml \
      --halt --halt-reason operator_halt
  ```
- Flatten and halt:
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python \
      -m src.execution.live_trading \
      --config /opt/wang_trading/config/live_trading.yaml \
      --flatten
  ```
- Verify flat:
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python \
      -m src.execution.live_trading \
      --config /opt/wang_trading/config/live_trading.yaml \
      --verify-flat
  ```
