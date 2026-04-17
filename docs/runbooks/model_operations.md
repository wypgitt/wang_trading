# Model operations

All retraining, promotion, RL shadow, and A/B test procedures.

## Scheduled retrain

The retrain scheduler (`wang_retrain_scheduler` supervisor program) runs
weekly on Sunday 02:00 UTC. It:

1. Assembles the previous 2-year training window.
2. Trains a fresh meta-labeler candidate.
3. Runs CPCV, DSR, and PBO gates.
4. If all gates pass, stages the run in MLflow under the `staging` alias.
5. Emits a summary to the compliance log and Telegram.

A human promotes `staging` → `production` (see below). Automation never
self-promotes.

## Emergency retrain

Triggered when `model.age` blocker fires or drift causes a SEV3.

```bash
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.ml_layer.retrain_scheduler --now
```

Follow-up:
1. Inspect the MLflow run — all three gates must pass.
2. If any gate fails, do NOT promote; open a SEV3 and investigate the
   failure mode (leakage, label drift, regime change).

## Model promotion

After staging, promote a model to production:

```bash
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.ml_layer.model_registry --promote <run_id>
```

Promotion writes an operator audit entry, updates MLflow's `production`
alias, and emits a Telegram alert. The live pipeline picks up the new
model at the next cycle.

### Rollback

```bash
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.ml_layer.model_registry --promote <previous_run_id>
```

## RL shadow monitoring

The shadow agent runs alongside HRP in every live cycle. Check status:

```bash
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.execution.live_trading --check-rl-promotion
```

The weekly eligibility check also fires a Telegram alert the day the RL
agent first becomes eligible. **Do not promote without operator approval.**

## RL promotion

Criteria (enforced in code):

- ≥6 months of shadow history.
- RL Sharpe > HRP Sharpe by 0.3.
- RL max-drawdown within 2% of HRP.
- Paired t-test p < 0.05.
- RL passes CPCV, DSR, PBO.

When eligible:

```bash
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.execution.live_trading --approve-rl-promotion
```

The pipeline logs a compliance entry, swaps the optimizer, sends an
EMERGENCY alert, and starts the 3-day auto-revert watchdog (5% drawdown
triggers auto-revert).

### Revert RL → HRP

```bash
sudo -u wang /opt/wang_trading/venv/bin/python \
    -m src.execution.live_trading --revert-to-hrp "reason here"
```

Always supply a human-readable reason — it shows up in the audit log.

## Feature engineering A/B test

Protocol for adding a new feature to the factory:

1. **Offline.** Add to the factory behind a config flag; ensure all
   backtests pass with the flag both on and off.
2. **Shadow.** Run the candidate factory in paper mode for 8 weeks;
   confirm Sharpe within ±0.1 of baseline with no drift.
3. **Gated retrain.** Retrain the meta-labeler with the feature enabled;
   all three gates must pass.
4. **Staged rollout.** Enable in staging for one full session before
   touching production.

A/B test review meeting is mandatory before step 4.
