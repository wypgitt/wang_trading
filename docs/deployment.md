# Deployment Runbook

This runbook covers production deployment of the wang_trading stack on a
single Linux host. The host runs four supervised services: data ingestion,
monitoring, retrain scheduler, and the live trading pipeline. Only the live
trading pipeline requires manual operator approval to start.

## Contents

- [Prerequisites](#prerequisites)
- [Initial deployment](#initial-deployment)
- [Starting live trading](#starting-live-trading)
- [Day-to-day operations](#day-to-day-operations)
- [Rollback](#rollback)
- [Log tailing](#log-tailing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Linux host (Ubuntu 22.04 LTS or equivalent), 8+ CPU, 32+ GB RAM, 500+ GB SSD
- Python 3.11 available as `python3`
- TimescaleDB 2.13+ reachable from the host (local or networked)
- Redis 7+ (used by the feature store cache)
- Prometheus 2.45+ and Grafana 10+ for monitoring
- Outbound network access to broker APIs (Alpaca, CCXT venues, IBKR Gateway)
- Incoming firewall open only on the monitoring scrape port (default 9090)

Operator checklist before first deploy:

- [ ] Broker API keys created, trading scope enabled, IP-allowlisted
- [ ] Telegram bot + chat ID for alerts
- [ ] Encrypted backup location configured
- [ ] On-call rotation agreed with at least one secondary

## Initial deployment

Run the deploy script as root. It creates the `wang` service user, installs
the Python virtualenv, copies source into `/opt/wang_trading`, installs the
systemd unit and (if present) supervisord configs, and sets up logrotate.

```bash
sudo ./scripts/deploy.sh --repo /path/to/wang_trading
```

The script **does not start live trading**. Starting it is a manual
operator step — see below.

After the script finishes:

1. Edit `/opt/wang_trading/config/live_trading.yaml` — fill in broker
   credentials, symbols, deployment plan.
2. Edit `/opt/wang_trading/config/live_trading.env` — export the
   asset-class-specific enable flags:
   ```ini
   WANG_ALLOW_LIVE_TRADING=yes        # Alpaca equities
   WANG_ALLOW_LIVE_CRYPTO=yes         # CCXT venues
   WANG_ALLOW_LIVE_FUTURES=yes        # IBKR futures
   ```
   Leave these blank until you actually intend to go live.
3. Touch `.operator_checkin` to confirm recent operator presence:
   ```bash
   sudo -u wang touch /opt/wang_trading/.operator_checkin
   ```

## Starting live trading

Live trading refuses to start if `/opt/wang_trading/.live_halt` exists or if
preflight fails. These are by design.

1. **Remove any stale HALT file** (only after you understand why it was
   written — check the compliance log):
   ```bash
   sudo rm -f /opt/wang_trading/.live_halt
   ```

2. **Run preflight**:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python \
       -m src.execution.preflight --full-check
   ```
   Exit codes: `0` clean, `2` warnings only (still OK), `1` blocker failed.

3. **Start the service**:
   ```bash
   sudo systemctl start wang-live-trading
   sudo systemctl status wang-live-trading
   ```

4. **Watch the first few minutes**:
   ```bash
   sudo journalctl -u wang-live-trading -f
   ```
   Expect the startup alert (`Live trading starting`) in Telegram inside 30s.

## Day-to-day operations

- **Graceful stop**:
  ```bash
  sudo systemctl stop wang-live-trading
  ```
  The pipeline writes a HALT file and sends a shutdown alert before exiting.

- **Emergency flatten** (cancel orders + close positions immediately):
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python \
      -m src.execution.live_trading --emergency-flatten
  ```

- **Check RL promotion eligibility**:
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python \
      -m src.execution.live_trading --check-rl-promotion
  ```

- **Approve RL promotion** (only after eligibility check passes):
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python \
      -m src.execution.live_trading --approve-rl-promotion
  ```

- **Revert to HRP**:
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python \
      -m src.execution.live_trading --revert-to-hrp "reason here"
  ```

## Rollback

Rolling back a bad deploy does not automatically flatten positions. Consider
whether you want a flat book first.

1. Stop live trading:
   ```bash
   sudo systemctl stop wang-live-trading
   ```

2. If needed, flatten:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python \
       -m src.execution.live_trading --emergency-flatten
   ```

3. Redeploy the previous revision:
   ```bash
   sudo ./scripts/deploy.sh --repo /path/to/prior_revision
   ```

4. Re-run preflight and start as normal.

## Log tailing

- **Live trading**:
  `sudo journalctl -u wang-live-trading -f` (systemd) or
  `sudo supervisorctl tail -f wang_live_trading` (supervisor).
- **Ingestion**:
  `tail -F /var/log/wang_trading/ingest_equities.out.log`
- **Monitoring**:
  `tail -F /var/log/wang_trading/monitoring.out.log`
- **Compliance**:
  `tail -F /opt/wang_trading/logs/live_trading_compliance.log`

## Troubleshooting

| Symptom                                     | Likely cause                                                    | Fix                                                                                       |
| ------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `service refused to start`                  | HALT file present                                               | Investigate the compliance log for the shutdown reason; remove the HALT file when safe.   |
| `preflight failed: N blockers`              | One or more blocker checks failed                               | Run `preflight --full-check` and address each blocker before restarting.                  |
| `Operator check-in older than 1h`           | `.operator_checkin` not touched recently                        | `sudo -u wang touch /opt/wang_trading/.operator_checkin` (only if you really are present) |
| `Live trading blocked: set WANG_ALLOW_LIVE_TRADING=yes`  | Env-var gate not set                                            | Set the appropriate env var in `live_trading.env` and reload the service.                 |
| RL auto-reverted                            | >5% drawdown in the first 3 days after RL promotion             | Investigate before re-approving; check the shadow comparison report for regression.       |
| Rapid alert loop                            | Missing / crashed Telegram channel                              | Check `monitoring.out.log`; verify bot credentials in `settings.yaml`.                    |

If the system is misbehaving in any way that isn't covered here, **halt
first, investigate second**:

```bash
sudo systemctl stop wang-live-trading
sudo -u wang touch /opt/wang_trading/.live_halt
```
