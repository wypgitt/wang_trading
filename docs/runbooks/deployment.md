# Deployment

Short runbook for deploys. The long-form walkthrough (first install,
systemd config, supervisor configs) lives in [`../deployment.md`](../deployment.md).

## Pre-deployment checklist

- [ ] PR reviewed and merged to `master`.
- [ ] `make test` green on CI.
- [ ] `make preflight` passes against staging data.
- [ ] Changelog entry drafted.
- [ ] Operator available for the deploy + first hour of monitoring.
- [ ] Ops log entry opened ("deploying <sha> at <UTC>").

Additional gates for trading-path changes:

- [ ] Paper-trading Sharpe vs previous revision regression-tested over
      the past 8 weeks.
- [ ] Any model change has CPCV/DSR/PBO gate results attached.
- [ ] Risk committee sign-off for changes to sizing, circuit breakers,
      or deployment phases.

## Staging → production

Staging is a separate host with the same code + a sandbox broker keyset.
Never skip staging for trading-path changes.

1. **Deploy to staging.**
   ```bash
   git checkout <sha>
   sudo ./scripts/deploy.sh --repo .
   ```
2. **Burn-in.** Leave staging running for at least one full trading
   session. Watch for:
   - Non-zero rejection rate on the OMS.
   - Drift alerts.
   - Compliance-log gaps.
3. **Promote to production.** Same command on the production host.
   Systemd is `autostart=false` — deploy does NOT auto-start.
4. **Operator approves:** remove any `.live_halt`, run preflight,
   `systemctl start wang-live-trading`.
5. **First 60 minutes:** sit at the dashboard. `journalctl -u
   wang-live-trading -f`. If anything looks off, HALT.

## Rollback

Rollback does not automatically flatten — decide first whether you want a
flat book.

1. `sudo systemctl stop wang-live-trading`.
2. (Optional) `--emergency-flatten`.
3. Redeploy the prior commit:
   ```bash
   git checkout <previous-sha>
   sudo ./scripts/deploy.sh --repo .
   ```
4. Re-run preflight and start.

## Verification

After every production start:

- [ ] `systemctl status wang-live-trading` = active (running).
- [ ] Startup alert fired on Telegram ("Live trading starting").
- [ ] Grafana "Trading" dashboard shows non-zero cycles within 5 min.
- [ ] Audit-log chain still intact:
  ```bash
  sudo -u wang /opt/wang_trading/venv/bin/python -c \
      "from src.execution.audit_log import ComplianceAuditLogger; \
       print(ComplianceAuditLogger(signing_key='...').verify_chain())"
  ```
