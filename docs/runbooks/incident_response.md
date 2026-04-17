# Incident response

**HALT first, investigate second** for anything under SEV3.

## Severity

| SEV | Definition | Response time | Example |
|-----|------------|---------------|---------|
| SEV1 | Money at risk *right now* | Page immediately | Position mismatch, runaway order loop, >5% intraday DD |
| SEV2 | Trading degraded or blind | 15 min | Broker connectivity lost, data gap >10 min |
| SEV3 | Observability/model health | 4 h | Drift alert, stale model nearing 30-day age |
| SEV4 | Cosmetic / informational | Next business day | Dashboard typo, log spam |

## Common incidents

### 1. Broker connectivity lost (SEV2)

**Signal.** `broker.heartbeat` fails for >2 consecutive cycles; alert title
"broker heartbeat timeout".

**Steps.**
1. Confirm external status pages (Alpaca, Binance, IBKR Gateway).
2. If the outage is upstream, the circuit breaker should already have
   paused new order submission. Verify:
   ```bash
   grep HEARTBEAT /opt/wang_trading/logs/live_trading_compliance.log | tail
   ```
3. If our side is broken (TWS crashed, DNS, expired token), HALT and fix.
4. After recovery, run preflight before restart.

### 2. Unexpected drawdown (SEV1)

**Signal.** `drawdown > 5%` alert on live NAV.

**Steps.**
1. Stop the pipeline:
   ```bash
   sudo systemctl stop wang-live-trading
   ```
2. If the drawdown is still accelerating (stale data, rogue position) —
   flatten:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python \
       -m src.execution.live_trading --emergency-flatten
   ```
3. Freeze deployment; do NOT restart until root cause is known.
4. Pull the audit log window covering the loss:
   ```bash
   sudo -u wang /opt/wang_trading/venv/bin/python -c \
       "from src.execution.audit_log import ComplianceAuditLogger; \
        print(ComplianceAuditLogger(signing_key='...').export_report(format='csv'))"
   ```

### 3. Model staleness (SEV3)

**Signal.** `model.age` blocker in preflight or drift_detector alert.

**Steps.**
1. Schedule the emergency retrain — see
   [model_operations.md](model_operations.md#emergency-retrain).
2. Until the new model passes all three gates, reduce deployment phase:
   operator runs `--revert-to-hrp` if RL was promoted.

### 4. Data gap (SEV2)

**Signal.** Ingestion lag >10 min; Grafana panel turns red.

**Steps.**
1. Check the ingestion process:
   ```bash
   sudo supervisorctl status wang_data_equities wang_data_crypto
   ```
2. Restart the failed group:
   ```bash
   sudo supervisorctl restart wang_data_crypto
   ```
3. If the gap affects features used in the current cycle, trust nothing:
   HALT live trading and backfill before resuming:
   ```bash
   make backfill SYMBOL=BTC/USDT DAYS=1
   ```

### 5. Position mismatch (SEV1)

**Signal.** Reconciliation diff count > 0 in the compliance log.

**Steps.**
1. Freeze new orders: `sudo systemctl stop wang-live-trading`.
2. Run disaster recovery:
   ```bash
   make recover
   ```
   The recovery manager reconciles the snapshot with the broker and
   cancels orphan orders.
3. If the diff is unexplained, flatten and file a SEV1 post-mortem.

### 6. Pipeline crash (SEV1/2 depending on state)

**Signal.** `systemctl status wang-live-trading` shows `failed`, and the
`.live_crash` sentinel exists.

**Steps.**
1. Do not restart the service yet.
2. Inspect the last compliance entries:
   ```bash
   tail -200 /var/log/wang_trading/live_trading.err.log
   ```
3. Run disaster recovery (`make recover`). This clears the crash sentinel
   after reconciling.
4. Preflight → restart.

### 7. Paper/live divergence (SEV2)

**Signal.** Daily divergence check flags `|live_sharpe - paper_sharpe| > 1.0`.

**Steps.**
1. Capital deployment controller auto-halts — verify:
   ```bash
   grep "capital deployment halted" /var/log/wang_trading/live_trading.err.log
   ```
2. Do not resume until a human-reviewed RCA identifies the delta. Common
   causes: slippage model too optimistic, venue maintenance window,
   fill-timing bug.

## Escalation

- **Primary on-call** responds within 15 minutes (all SEV1/2).
- **Secondary** called at 30 min no-ack.
- **Engineering lead** called for any SEV1 or any SEV2 > 2 h.
- **Compliance** notified for any incident that touches live money ≥ SEV2.

## Post-incident template

Copy this into the ops log after every SEV1/2:

```
## Incident: <title>
- Severity   :
- Start      : <UTC>
- Detected   : <UTC>  (detection source: alert / operator / other)
- Mitigated  : <UTC>
- Resolved   : <UTC>
- Impact     : <P&L impact, downtime, blast radius>
- Root cause :
- Timeline   :
    - t+0   : ...
    - t+3m  : ...
- Action items (owner / due):
    - [ ] ...
```

File the completed post-mortem in `docs/postmortems/YYYY-MM-DD-<slug>.md`
and link it from this runbook if it uncovers a new class of failure.
