# Architecture overview

End-to-end dataflow from a market tick to an executed trade, across all
six phases.

```
                                   ┌──────────────────────────────────────────────┐
                                   │                    Phase 6                    │
                                   │   Live Capital + RL + Production Hardening    │
                                   └──────────────────────────────────────────────┘
                                                         ▲
                                                         │
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                  │
 │   Market tick       Phase 1                 Phase 2                Phase 3       │
 │   ─────────────     ────────────            ──────────────         ────────────  │
 │   alpaca/poly/──►   ingest → TSDB  ──►      feature factory  ──►   signal       │
 │   binance/ibkr      bar constructors        (120+ features)        battery      │
 │                     ETF trick                                       (9 families) │
 │                                                                                  │
 │                                                                      │           │
 │                                                                      ▼           │
 │   Phase 3 (cont'd)            Phase 4                                            │
 │   ──────────────────          ──────────────────                                 │
 │   meta-labeler      ──►       walk-forward / CPCV / DSR / PBO                    │
 │   regime detector             portfolio construction (HRP)                       │
 │                                                                      │           │
 │                                                                      ▼           │
 │   Phase 5                                                                        │
 │   ──────────────                                                                 │
 │   bet-sizing cascade  ──►  order manager  ──►  broker adapter  ──►  exchange     │
 │   circuit breakers       execution algos       (Alpaca/CCXT/IBKR)                │
 │                                                                                  │
 └──────────────────────────────────────────────────────────────────────────────────┘
                                                         ▲
                                                         │
                                 observability + state  ──┤
                                 (Phase 5 + 6)            │
                                                          │
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                  │
 │   MetricsCollector ──► Prometheus ──► Grafana dashboards                         │
 │   AlertManager    ──► Telegram                                                   │
 │   ComplianceAuditLogger ──► HMAC-signed chain ──► audit_log hypertable           │
 │   SnapshotManager ──► logs/snapshots/*.pkl (30-day retention)                    │
 │   RecoveryManager ──► reconciles on crash; orphan-order cancellation             │
 │                                                                                  │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

## Phase-by-phase responsibilities

### Phase 1 — Data engine
Real-time + historical ingestion from Alpaca/Polygon/Binance/IBKR into a
TimescaleDB hypertable, tick → bar construction (time / tick / volume /
dollar), ETF-trick synthetic continuous contracts, validation, feature
store.

### Phase 2 — Feature factory + signal battery
120+ features (microstructure, fractional differentiation, volatility,
regimes, sentiment via FinBERT, etc.). Nine orthogonal signal families:
trend, mean-reversion, carry, cross-exchange arb, pairs, sentiment,
options-flow, macro, microstructure.

### Phase 3 — Core ML
Triple-barrier labeling, meta-labeler (LightGBM) with calibrated
probabilities, HMM regime detector + LSTM predictor, feature-importance
gate (SHAP), model registry on MLflow.

### Phase 4 — Validation + allocation
Combinatorial purged cross-validation, deflated Sharpe ratio, probability
of backtest overfitting, walk-forward backtester with realistic costs,
hierarchical risk-parity allocation, factor-risk model.

### Phase 5 — Execution + monitoring
Bet-sizing cascade (AFML sizing → Kelly → vol adjust → ATR cap → risk
budget), order manager, execution algorithms (Immediate / TWAP / VWAP /
Iceberg), broker adapters (paper + skeletons for live), circuit breakers,
Prometheus metrics, Telegram alerts, drift detector.

### Phase 6 — Live capital + RL + hardening
- **P6.01–P6.04** — live broker adapters (Alpaca, CCXT, IBKR) + factory +
  smart-order router.
- **P6.05** — preflight checks (18 blockers + 3 warnings).
- **P6.06** — four-phase graduated capital deployment controller.
- **P6.07** — live trading pipeline (subclass of paper) with HALT file,
  operator check-in, deployment multiplier.
- **P6.08–P6.11** — Gymnasium trading environment, PPO agent, shadow
  comparison engine, shadow-mode integration + auto-revert.
- **P6.12** — supervisor / systemd configs + deploy script.
- **P6.13** — disaster recovery (checksummed snapshots, crash detection,
  broker reconciliation).
- **P6.14** — HMAC-chained compliance audit log.
- **P6.15** — pluggable secrets management (env / Fernet file / AWS / GCP).
- **P6.16** — operational runbooks.
- **P6.17** — this document, go-live checklist, production smoke test.

## Control plane

```
                   ┌───────────────────────┐
                   │     Operator (CLI)    │
                   └──┬────────────────────┘
                      │  make preflight
                      │  make live-start | live-stop | live-flatten
                      │  make recover
                      │  --approve-rl-promotion / --revert-to-hrp
                      ▼
   ┌───────────────────────────────────────┐
   │        LiveTradingPipeline            │
   │  ┌──────────────┐  ┌────────────────┐ │
   │  │ Preflight    │  │  Deployment    │ │
   │  │ Checker      │  │  Controller    │ │
   │  └──────────────┘  └────────────────┘ │
   │  ┌──────────────┐  ┌────────────────┐ │
   │  │ Shadow Aware │  │  Promotion     │ │
   │  │  Optimizer   │──│  Controller    │ │
   │  └──────────────┘  └────────────────┘ │
   │  ┌──────────────┐  ┌────────────────┐ │
   │  │ Snapshot +   │  │ Audit Log      │ │
   │  │ Recovery     │  │ (HMAC chain)   │ │
   │  └──────────────┘  └────────────────┘ │
   └───────────────────────────────────────┘
                      │
                      ▼
         HRP optimizer  ⇆  RL shadow agent
                      │
                      ▼
            Bet-sizing cascade
                      │
                      ▼
            OrderManager + CircuitBreakers
                      │
                      ▼
       BrokerFactory → Alpaca / CCXT / IBKR
                      │
                      ▼
                  Exchange
```

## State + persistence

| Artifact | Where | Retention |
|----------|-------|-----------|
| Bars + features | TimescaleDB hypertable | 7 y |
| Orders + fills | TimescaleDB | 7 y |
| Audit log | TimescaleDB (HMAC chain) | 7 y |
| Model runs | MLflow (S3 backend) | 7 y |
| Snapshots | `logs/snapshots/*.pkl` | 30 d |
| App logs | `/var/log/wang_trading/` | 180 d |
| Compliance log | `logs/live_trading_compliance.log` | 7 y |

## Failure domains

| Failure | Detection | Automatic action | Operator action |
|---------|-----------|------------------|-----------------|
| Broker disconnect | heartbeat | pause new orders (CB) | investigate, preflight before restart |
| Drawdown > 5% | circuit breaker | HALT_AND_FLATTEN | post-mortem |
| Paper/live Sharpe divergence > 1.0 | daily check | halt via `CapitalDeploymentController` | investigate; do not resume blindly |
| RL > 5% drawdown in 3 days of promotion | per-cycle check | auto-revert to HRP | investigate before re-approving |
| Process crash | signal handler | write snapshot + `.live_crash` | `make recover` |
| Audit chain broken | `verify_chain()` | none | freeze + investigate tamper |
```
