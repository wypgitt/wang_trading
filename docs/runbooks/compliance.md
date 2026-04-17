# Compliance

Audit log handling, tax reporting, regulatory considerations, retention.

## Audit log

Every decision (signal → meta-label → sizing → order → fill → breaker →
operator action) writes an entry to the compliance audit log, HMAC-signed
with the key from `WANG_AUDIT_HMAC_KEY`. Entries chain via
`prev_signature`; tampering with any row breaks the chain downstream.

### Export (yearly, on demand)

```bash
sudo -u wang /opt/wang_trading/venv/bin/python -c \
    "from datetime import datetime, timezone; \
     from src.execution.audit_log import ComplianceAuditLogger; \
     import os; \
     logger = ComplianceAuditLogger(signing_key=os.environ['WANG_AUDIT_HMAC_KEY']); \
     print(logger.export_report( \
         start=datetime(2026,1,1,tzinfo=timezone.utc), \
         end=datetime(2027,1,1,tzinfo=timezone.utc), \
         format='csv'))" > audit_2026.csv
```

### Verify chain integrity

```bash
sudo -u wang /opt/wang_trading/venv/bin/python -c \
    "from src.execution.audit_log import ComplianceAuditLogger; \
     import os, json; \
     logger = ComplianceAuditLogger(signing_key=os.environ['WANG_AUDIT_HMAC_KEY']); \
     print(json.dumps(logger.verify_chain(), indent=2))"
```

If `ok: false`, freeze exports and investigate. A broken chain is itself
evidence of tampering — preserve the raw database state before anything
else.

## Tax reporting

1. Export every fill for the tax year (query the `fills` table *and* the
   audit log; cross-check row counts):
   ```sql
   SELECT order_id, fill_id, timestamp, symbol, quantity, price, commission
   FROM fills
   WHERE timestamp >= '2026-01-01' AND timestamp < '2027-01-01'
   ORDER BY timestamp;
   ```
2. Download the authoritative broker statements (Alpaca 1099-B, crypto
   venue 1099-K/MISC where applicable, IBKR Form 1099-B consolidated).
3. Reconcile internal rows vs broker rows — any diff > $0.01 is a bug
   (or a P&L lost in a stale corporate action).
4. File the annual package under `compliance/tax/YYYY/` with:
   - Internal fills CSV
   - Broker statements (PDF/CSV)
   - Reconciliation diff report
   - Audit-log chain verification receipt

## Regulatory considerations

This stack is a **personal-account** system. It is not registered as an
investment adviser or broker-dealer. Before ever trading third-party
money, the following are non-negotiable:

- Registration status (investment adviser / CPO / CTA) with the SEC or
  CFTC, depending on strategy and client base.
- Written compliance program (books, records, disclosure) reviewed by
  counsel.
- KYC/AML onboarding via a regulated partner.
- Performance reporting that conforms to GIPS or equivalent.

Treat the current audit log as a *foundation* for those requirements,
not a substitute.

## Record retention

| Record | Retention |
|--------|-----------|
| Fills / orders (database rows) | 7 years |
| Audit log | 7 years |
| Compliance runbook snapshots (`docs/runbooks/` git history) | Indefinite |
| Operator check-in sentinel | 90 days |
| State snapshots (`logs/snapshots/`) | 30 days (rotating) |
| Application logs | 180 days (logrotate) |

Encrypted nightly backups go to S3 with object-lock enabled — deletions
require multi-party approval.
