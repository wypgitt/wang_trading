# Unified Quantitative Trading System

A multi-strategy, multi-asset quantitative trading engine integrating research from
López de Prado (AFML), Chan, Jansen, Clenow, Narang, Johnson, Isichenko, and Sinclair.

## Architecture

```
src/
├── data_engine/       # Phase 1: Bars, ingestion, storage
│   ├── bars/          # Bar constructors (tick, volume, dollar, TIB, VIB)
│   ├── ingestion/     # WebSocket/REST data collection
│   │   └── adapters/  # Exchange-specific adapters (Alpaca, Binance, IBKR)
│   ├── storage/       # TimescaleDB interface, feature store
│   └── validation/    # Bar quality validation
├── feature_factory/   # Phase 2: FFD, entropy, microstructure, GARCH, NLP
├── signal_battery/    # Phase 2: Momentum, mean-rev, trend, stat-arb, carry, vol
├── labeling/          # Phase 3: Triple-barrier, meta-labeling, sample weights
├── ml_layer/          # Phase 3: XGBoost meta-labeler, LSTM regime, RL agent
├── bet_sizing/        # Phase 3: AFML sizing, Kelly, GARCH adjustment
├── portfolio/         # Phase 4: HRP, factor risk, risk parity
├── backtesting/       # Phase 4: CPCV, deflated Sharpe, PBO
├── execution/         # Phase 5: VWAP/TWAP, market impact, TCA
└── monitoring/        # Phase 5: Grafana dashboards, alerting
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit config
cp config/settings.example.yaml config/settings.yaml
# Add your API keys (Alpaca, Polygon, etc.)

# 3. Set up database
python scripts/setup_db.py

# 4. Run ingestion (equities)
python -m src.data_engine.ingestion.runner --asset-class equities

# 5. Validate bars
python -m src.data_engine.validation.runner --symbol AAPL --bar-type tib

# 6. Paper trading (Phase 5)
cp config/paper_trading.example.yaml config/paper_trading.yaml
# edit the config, then
python -m src.execution.paper_trading --config config/paper_trading.yaml

# 7. Start the monitoring stack (Grafana + Prometheus)
docker compose up -d prometheus grafana
python scripts/setup_grafana.py \
    --grafana-url http://localhost:3000 --api-key <grafana-api-key>

# 8. Smoke-test the full pipeline in < 10s (no external services)
make smoke-test

# 9. Production smoke test — exercises every Phase 6 system (preflight,
#    deployment controller, audit log, snapshots, recovery, alerts).
python scripts/production_smoke_test.py
```

## Live trading

**⚠️ DANGER.** Live trading puts real capital at risk. Do not start the
live pipeline without reading [`docs/go_live_checklist.md`](docs/go_live_checklist.md)
and completing every item.

```bash
# First-time install on a Linux host (creates user, venv, systemd unit)
sudo ./scripts/deploy.sh

# 1. Touch the operator check-in sentinel (proves a human is present)
sudo -u wang touch /opt/wang_trading/.operator_checkin

# 2. Run preflight — 18 blocker checks must pass
make preflight

# 3. If (and only if) preflight is green:
make live-start       # or: sudo systemctl start wang-live-trading

# Graceful stop (writes HALT sentinel — operator must clear before next start):
make live-stop

# Emergency: cancel all orders + close all positions
make live-flatten

# Disaster recovery after a crash or unclean exit:
make recover
```

## Testing

```bash
make test               # unit tests (excludes integration)
make test-integration   # end-to-end Phase 2/3/4 pipeline tests
make bench              # Phase 2 performance benchmarks
make bench-backtest     # Phase 4 backtesting / portfolio benchmarks
```

## Configuration

All settings are in `config/settings.yaml`. API keys should be set via
environment variables or the config file (never committed to git).

## Phase Status

- [x] **Phase 1: Data Foundation** (complete)
  - TIB/VIB/dollar/volume/tick bars; ETF trick; CUSUM filter; TimescaleDB storage
- [x] **Phase 2: Feature Factory + Signal Battery** (complete)
  - Features: FFD, structural breaks, entropy, microstructure, GARCH, NLP sentiment, on-chain, autoencoder, classical
  - Signals: 10 generators across the 7 canonical families plus a unified orchestrator
  - 326 unit tests + 1 end-to-end integration test, all green
  - Benchmarks: all 7 components under target (`make bench`)
  - See: [docs/phase2_features.md](docs/phase2_features.md) · [docs/phase2_signals.md](docs/phase2_signals.md)
- [x] **Phase 3: Labeling + Core ML + Bet Sizing** (complete)
  - Labeling Engine: triple-barrier, meta-labeling pipeline, AFML sample weights (uniqueness + sequential bootstrap + return attribution + time decay)
  - ML Layer: LightGBM/XGBoost/RandomForest meta-labeler with purged CV + isotonic calibration, Optuna TPE tuning with MedianPruner, MDI/MDA/SFI/SHAP feature importance, MLflow model registry, LSTM regime detector (Tier 2)
  - Bet Sizing: 5-layer cascade (AFML sizing → Kelly cap → vol adjustment → ATR cap → risk budget) with full audit trail
  - Retraining: `scripts/retrain_model.py` with `--tune` / `--use-best-params` / `--all-symbols` / `--dry-run`; `make retrain*` targets
  - 571 unit tests + 2 end-to-end integration tests (Phase 2 and Phase 3), all green
  - See: [docs/phase3_ml_pipeline.md](docs/phase3_ml_pipeline.md)
- [x] **Phase 4: Backtesting + Portfolio** (complete)
  - Backtesting: walk-forward engine with realistic transaction costs (Johnson square-root impact + IBKR-style commissions, spread, slippage); CPCV (45 paths) / Deflated Sharpe / PBO validation gates; `BacktestReport` with monthly returns, drawdown table, regime-conditional stats; `StrategyGate` orchestrator with `quick_validate` fast path
  - Portfolio: HRP (default), PCA factor risk model with `neutralize_factors`, risk parity (Griveau-Billion coordinate descent), multi-strategy allocator with risk-budget enforcement (design-doc §8.5)
  - 756 unit tests + 3 end-to-end integration tests (Phase 2, 3, 4), all green
  - Benchmarks: all 6 Phase 4 components under target (`make bench-backtest`)
  - See: [docs/phase4_backtesting.md](docs/phase4_backtesting.md) · [docs/phase4_portfolio.md](docs/phase4_portfolio.md)
- [x] **Phase 5: Execution + Paper Trading + Monitoring** (complete)
  - Execution Engine: Order/Fill/Position/PortfolioState models, 8 circuit breakers (fat-finger, daily-loss, drawdown throttle, model staleness, connectivity, data quality, correlation spike, dead-man), execution algorithms (ImmediateAlgo / TWAP / VWAP / Iceberg) with `select_execution_algo()` router, broker adapters (PaperBroker full implementation; Alpaca + CCXT skeletons with position reconciliation)
  - Order Manager: end-to-end orchestration — target portfolio → pre-trade checks → cost estimate → algo selection → execution → PortfolioState update; triple-barrier exit checks (stop-loss / take-profit / time expiry); `run_cycle` per-tick entry point
  - TCA: post-trade analyzer with slippage / impact / timing-cost / TWAP-VWAP benchmarks; execution-degradation detection; full storage schema (`orders`, `fills`, `tca_results`, `portfolio_snapshots`)
  - Monitoring: Prometheus `MetricsCollector` (15 metrics across portfolio, orders, signals, execution, data health, features, model, breakers); tiered `AlertManager` (Log + Telegram) with 8 templated alerts and duplicate suppression; `FeatureDriftDetector` (KL / KS / mean-shift / variance-ratio)
  - Grafana: `generate_main_dashboard()` (6 rows, 17+ panels) + `generate_alerting_rules()` + `scripts/setup_grafana.py`; Prometheus + Grafana wired into `docker-compose.yaml`
  - Paper Trading: `PaperTradingPipeline` top-level runner with `PipelineConfig`, full `run_cycle` flow (features → signals → meta → sizing → optimizer → execution → metrics → drift); `DailyReconciliation` + `generate_daily_report`; `RetrainScheduler` with purged-CV promotion gate
  - 913 unit tests + 4 end-to-end integration tests (including Phase 5 100-cycle integration); `scripts/smoke_test.py` runs the full stack end-to-end in under 10 seconds (`make smoke-test`)
  - See: [docs/phase5_execution.md](docs/phase5_execution.md)
- [x] **Phase 6: Live Capital + RL Agent + Production Hardening** (complete)
  - Live broker adapters: Alpaca (equities), CCXT (crypto — Binance/Coinbase/Kraken/Bybit), IBKR (futures) — each gated by an env-var switch (`WANG_ALLOW_LIVE_TRADING`, `WANG_ALLOW_LIVE_CRYPTO`, `WANG_ALLOW_LIVE_FUTURES`)
  - `BrokerFactory` + `SmartOrderRouter` for cross-venue best-execution and depth aggregation
  - `PreflightChecker` with 18 blocker checks across broker connectivity, model readiness, paper-trading proof, infrastructure, risk limits, and operator acknowledgment
  - Graduated capital deployment (`CapitalDeploymentController`): 4-phase ramp (pilot $5K / beta $15K / scale $50K / full) with paper-vs-live divergence detection and auto-halt
  - `LiveTradingPipeline` (subclass of paper): HALT/CRASH sentinels, operator check-in, deployment multiplier, daily divergence check, weekly promotion check, emergency flatten, HMAC-chained compliance audit log
  - RL portfolio optimizer: Gymnasium `TradingEnv` + PPO (`stable_baselines3`) agent + `ShadowComparisonEngine` with paired t-test promotion gate + `RLPromotionController` with 3-day / 5%-drawdown auto-revert watchdog
  - Disaster recovery: checksummed state snapshots, crash signal handler, broker reconciliation with orphan-order cancellation
  - Secrets management: pluggable backends (env, Fernet-encrypted file, AWS Secrets Manager, GCP Secret Manager) with 5-min TTL cache
  - Deployment: `scripts/deploy.sh`, supervisor configs for all four services, systemd unit, logrotate, runbooks in `docs/runbooks/`
  - 1079 unit tests + 28 integration tests, all green; `scripts/production_smoke_test.py` runs every Phase 6 subsystem in < 10 s
  - See: [docs/go_live_checklist.md](docs/go_live_checklist.md) · [docs/runbooks/](docs/runbooks/README.md) · [docs/architecture_overview.md](docs/architecture_overview.md) · [docs/deployment.md](docs/deployment.md)

## Production operations

Once live, the system is run by a named on-call operator following the
runbooks. Every decision — signal, meta-label, bet size, order, fill,
circuit-breaker trigger, phase promotion, operator action — is written to
the HMAC-chained compliance audit log.

| Runbook | Opens… |
|---------|--------|
| [Daily operations](docs/runbooks/daily_operations.md) | Every trading day |
| [Incident response](docs/runbooks/incident_response.md) | SEV1–SEV4 triage |
| [Deployment](docs/runbooks/deployment.md) | Ships + rollbacks |
| [Model operations](docs/runbooks/model_operations.md) | Retrain, promotion, RL |
| [Capital management](docs/runbooks/capital_management.md) | Adding / withdrawing funds |
| [Compliance](docs/runbooks/compliance.md) | Tax, audit export, retention |
| [Go-live checklist](docs/go_live_checklist.md) | Before first live dollar |
