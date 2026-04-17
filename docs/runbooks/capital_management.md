# Capital management

Moving money in and out of the trading account, phase transitions, and
emergency flatten.

## Deployment phases

The default plan (see `src/execution/capital_deployment.py`):

| Phase | Name    | Target capital | Size mult | Min duration |
|-------|---------|----------------|-----------|--------------|
| 1     | pilot   | $5,000         | 0.25×     | 14 days      |
| 2     | beta    | $15,000        | 0.50×     | 28 days      |
| 3     | scale   | $50,000        | 0.75×     | 42 days      |
| 4     | full    | (unbounded)    | 1.00×     | —            |

Promotion criteria per phase: Sharpe > 1.0 and max drawdown < 10% over the
duration of the phase. The controller enforces both before advancing.

## Phase transitions

**Promotion is operator-gated.** Weekly, the pipeline checks eligibility
and alerts when ready. Operator approves:

```bash
# Inspect eligibility first
sudo -u wang /opt/wang_trading/venv/bin/python -c \
    "from src.execution.capital_deployment import CapitalDeploymentController; \
     import json; print(json.dumps(... .get_deployment_status(), indent=2))"
```

To advance:
```bash
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.execution.capital_deployment --promote
```

Compliance logs the promotion; Telegram receives an INFO alert.

## Adding capital

1. **Stop live trading cleanly:**
   ```bash
   sudo systemctl stop wang-live-trading
   ```
2. **Wire the funds at the broker.** Confirm settlement (Alpaca: end of
   next business day; crypto: on-chain confirmations).
3. **Update the phase target** if the new capital pushes you into the
   next tier. Edit `config/live_trading.yaml` → `deployment.phases`.
4. **Preflight** and restart.

Never "let the system notice" new cash — always restart after funding.

## Withdrawing capital

Same procedure, inverted:

1. Stop live trading.
2. (Optional but recommended) flatten the book to USD/USDT. Withdrawing
   from a leveraged position silently raises effective leverage.
3. Wire the funds out.
4. Adjust `deployment.phases` *downward* if you've dropped below a
   target threshold.
5. Preflight + restart.

## Emergency flatten

When the goal is simply "get to cash now":

```bash
sudo systemctl stop wang-live-trading
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.execution.live_trading --emergency-flatten
```

This cancels every open order and submits market exits for every open
position. Expect slippage; that's the cost of certainty.

## Tax reporting

Every fill hits both the `orders` table and the compliance audit log.
See [compliance.md](compliance.md#tax-reporting) for the yearly export
procedure. Keep **raw broker statements** — our internal ledger is
authoritative only until a tax authority says otherwise.
