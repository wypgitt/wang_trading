# Runbooks

Operational runbooks for the wang_trading live stack. Each runbook covers a
single theme and is written assuming the reader is the on-call operator at
02:00, half awake, with one laptop.

| Runbook | When to open it |
|---------|----------------|
| [Daily operations](daily_operations.md) | Every trading day — morning, intraday, end-of-day, weekend. |
| [Incident response](incident_response.md) | Something is wrong right now. |
| [Deployment](deployment.md) | Shipping new code. Pre-flight → staging → prod → rollback. |
| [Model operations](model_operations.md) | Retraining, promotion, RL shadow, A/B tests. |
| [Capital management](capital_management.md) | Phase transitions, adding/withdrawing capital, flatten. |
| [Compliance](compliance.md) | Tax reporting, audit export/verify, retention. |

See also the deeper deployment walkthrough at [`../deployment.md`](../deployment.md).

## Conventions

- **Severity levels** — SEV1 (money at risk), SEV2 (trading degraded),
  SEV3 (observability gap), SEV4 (cosmetic). Definitions in
  [incident_response.md](incident_response.md#severity).
- **HALT first, investigate second** for anything unclear under live capital.
- **Dead-man sentinels** — `.live_halt` (clean), `.live_crash` (unclean)
  live in `/opt/wang_trading/`.
- **Alerts** fan out through Telegram; every alert includes a correlation
  ID that also appears in the compliance audit log.
