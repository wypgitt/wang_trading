# Wang Trading — Comprehensive Design Document

**Version:** 2.0 (design spec) / Phases 1–6 shipped
**Scope:** Single authoritative reference for the architecture, algorithms,
technology, and operational design of the Wang Trading system.
**Audience:** Solo operator of the system + future contributors.
**Companion docs:**
- [`architecture_overview.md`](architecture_overview.md) — 1-page overview
- [`data_model.md`](data_model.md) — storage schemas, retention
- [`technology_stack.md`](technology_stack.md) — libraries and rationale
- Phase-specific: `phase2_features.md`, `phase2_signals.md`,
  `phase3_ml_pipeline.md`, `phase4_backtesting.md`, `phase4_portfolio.md`,
  `phase5_execution.md`
- Ops: `deployment.md`, `go_live_checklist.md`, `runbooks/`

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Design philosophy](#2-design-philosophy)
3. [Canonical sources and their roles](#3-canonical-sources-and-their-roles)
4. [Architecture: Narang's 5 layers, 10 subsystems](#4-architecture-narangs-5-layers-10-subsystems)
5. [End-to-end dataflow](#5-end-to-end-dataflow)
6. [Subsystem 1 — Data Engine](#6-subsystem-1--data-engine)
7. [Subsystem 2 — Feature Factory](#7-subsystem-2--feature-factory)
8. [Subsystem 3 — Signal Battery](#8-subsystem-3--signal-battery)
9. [Subsystem 4 — Labeling Engine](#9-subsystem-4--labeling-engine)
10. [Subsystem 5 — ML Model Layer](#10-subsystem-5--ml-model-layer)
11. [Subsystem 6 — Bet Sizing Cascade](#11-subsystem-6--bet-sizing-cascade)
12. [Subsystem 7 — Portfolio Construction](#12-subsystem-7--portfolio-construction)
13. [Subsystem 8 — Backtesting and Validation](#13-subsystem-8--backtesting-and-validation)
14. [Subsystem 9 — Execution Engine](#14-subsystem-9--execution-engine)
15. [Subsystem 10 — Monitoring](#15-subsystem-10--monitoring)
16. [Validation gate: the three tests](#16-validation-gate-the-three-tests)
17. [Data model](#17-data-model)
18. [Technology stack and rationale](#18-technology-stack-and-rationale)
19. [Operational architecture](#19-operational-architecture)
20. [Deployment topology](#20-deployment-topology)
21. [Security and risk posture](#21-security-and-risk-posture)
22. [Phase roadmap](#22-phase-roadmap)
23. [Testing strategy](#23-testing-strategy)
24. [Future work](#24-future-work)

---

## 1. Executive summary

Wang Trading is a **multi-strategy, multi-asset, multi-timeframe quantitative
trading engine** for a single operator, trading US equities, crypto (spot +
perpetuals), and futures. It is designed as a **research-grade production
system**: every module is built from primary references in the quant
literature, and every strategy must clear three statistical overfit tests
before a live dollar touches it.

The thesis is simple to state and hard to execute:

> Alpha does not come from one large edge. It comes from combining **many
> uncorrelated modest edges** through meta-labeling and hierarchical risk
> parity, wrapped in **AFML-grade validation** and **institutional
> execution**.

The system is organised into **10 subsystems**, structured into **Narang's 5
layers** (Alpha / Risk / TCost / Portfolio Construction / Execution), and
rolled out across **6 phases**. All six phases are shipped and merged to
master as of 2026-04.

### At a glance

| Dimension | Value |
|-----------|-------|
| Asset classes | US equities, crypto (spot + perp), futures |
| Brokers | Alpaca (equities), CCXT (crypto, multi-venue), IBKR (futures + backup) |
| Primary timeframe | Information-driven bars (TIB), not clock bars |
| Feature count | ~200 per bar (9 families) |
| Signal families | 10 generators across 7 canonical families |
| ML tiers | Tier 1 meta-labeler (LightGBM), Tier 2 regime (LSTM), Tier 3 allocator (PPO) |
| Portfolio default | Hierarchical Risk Parity; PPO shadow-mode optional |
| Validation | CPCV (45 paths) + Deflated Sharpe (p<0.05) + PBO (<40%) |
| Live protection | 18 preflight blockers + 8 circuit breakers + HMAC audit chain |
| Deployment | 4-phase capital ramp: $5K pilot → $15K beta → $50K scale → full |
| Targets | Annual Sharpe >1.5 net (deflated) · MaxDD <20% · win rate >55% · SPY corr <0.3 |

### What makes it interesting

1. **Research-grade validation** — Standard k-fold CV is never used; every
   cross-validation is purged with embargo. Every candidate strategy must
   pass CPCV / Deflated Sharpe / PBO jointly before promotion.
2. **Meta-labeling, not signal stacking** — Raw signals are treated as
   proposals. A calibrated LightGBM learns which proposals to *take*, using
   triple-barrier labels and AFML sample weights.
3. **A 5-layer bet sizing cascade** — Probability → Kelly cap → vol adjust
   → ATR cap → risk budget. Every decision has a full audit trail.
4. **Graduated live deployment** — Capital ramps in four phases with
   automatic halt on paper-vs-live divergence.
5. **Tamper-evident operations** — Every signal, fill, and operator action
   is written to an HMAC-chained audit log.

---

## 2. Design philosophy

The system is opinionated. The opinions are consequential, so each is stated
explicitly with its reason and its blast radius.

### P1. Combine many modest edges; don't chase a large one

A single-strategy system with a Sharpe of 2.0 is either a fluke, overfit,
or about to decay. A portfolio of ten weakly-correlated Sharpe-0.5
strategies is robust — the math of diversification does the work. The
system's job is to **keep adding uncorrelated proposals and let the
meta-labeler and HRP arbitrate**.

**Blast radius:** we tolerate modest per-strategy quality. We do *not*
tolerate correlated strategies sneaking in — the correlation-spike
circuit breaker halts if portfolio correlation exceeds 0.80.

### P2. Validate as if every strategy is a lie

López de Prado's core observation is that the probability of a backtest
being overfit, given 20+ trials, approaches certainty. Our defense:

- **Purged CV with embargo** everywhere. Never plain `KFold`.
- **CPCV** instead of single-path walk-forward, giving 45 paths not 1.
- **Deflated Sharpe** to correct for multiple-testing selection bias.
- **PBO** to estimate the probability the backtest is actually worthless.

**Blast radius:** many strategies that look promising in plain backtests
will never make it to live. That is the intended behavior.

### P3. Narang's 5-layer separation is non-negotiable

Each layer has a clean API; no layer leaks responsibilities into another:

- **Alpha** produces `side ∈ {-1,0,+1}` and `confidence ∈ [0,1]`.
  It never sees NAV.
- **Risk** produces the meta-label — `P(profitable | signal, features)`.
  It never sees position size.
- **TCost** produces cost estimates and post-trade TCA.
  It never makes allocation decisions.
- **Portfolio Construction** produces target weights given a covariance.
  It never knows how signals were generated.
- **Execution** produces fills.
  It never decides whether to trade.

**Blast radius:** one subsystem's bug cannot corrupt another. This is worth
the interface ceremony.

### P4. Be conservative at the boundary; be liberal inside

At **system boundaries** (broker, exchange, operator) we are paranoid:
HMAC-signed audit chain, 18 preflight blockers, 8 circuit breakers, 3
explicit env-var gates for live trading. Inside the system, components
trust each other with duck-typed protocols — easier to test, easier to
extend.

### P5. The operator is a human. Assume they miss things

Every manual action is confirmed; every risky action requires an explicit
flag. The dead-man switch halts trading if the operator fails to check in
for 24 hours. Preflight blocks a restart even if the operator is sure
everything is fine. These gates cost seconds; an unauthorised live order
can cost a month's P&L.

---

## 3. Canonical sources and their roles

The system is explicitly composed from eight canonical texts. Every major
design decision traces to a specific chapter or idea in one of them.

| # | Source | Role in system | Key modules |
|---|--------|----------------|-------------|
| 1 | **López de Prado — Advances in Financial ML (AFML)** | Infra + safeguards — the spine of the system | bars (TIB/VIB/dollar), triple-barrier, meta-labeling, purged CV, CPCV, deflated Sharpe, HRP, sequential bootstrap |
| 2 | **Chan — Quantitative + Algorithmic Trading** | Alpha — classical systematic strategies | mean reversion (O-U, z-score), momentum, Kelly sizing, Engle-Granger cointegration, Kalman hedge ratios |
| 3 | **Jansen — ML for Algorithmic Trading** | Alpha enrichment + risk meta-layer | LightGBM/XGBoost meta-labeler, LSTM/Transformer regime, PPO RL portfolio, FinBERT, autoencoders |
| 4 | **Clenow — Following the Trend** | Alpha — trend-following toolkit | MA crossover, Donchian breakout, ATR position sizing, carry models |
| 5 | **Narang — Inside the Black Box** | Architecture blueprint | the 5-layer separation, TCA philosophy |
| 6 | **Johnson — Algo Trading & DMA** | Execution | VWAP/TWAP algos, square-root market impact model, smart-order routing |
| 7 | **Isichenko — Quant Portfolio Management** | Portfolio construction | PCA factor risk model, risk parity, factor neutralisation |
| 8 | **Sinclair — Volatility Trading** | Volatility alpha + risk | GARCH(1,1), volatility risk premium, regime-conditional sizing |

### Mapping: source → module

```
AFML  ──► data_engine/bars/              (TIB, VIB, dollar bars)
      ──► data_engine/bars/etf_trick.py  (continuous contracts)
      ──► labeling/triple_barrier.py     (triple-barrier labels)
      ──► labeling/sample_weights.py     (uniqueness + sequential bootstrap)
      ──► labeling/meta_labeler_pipeline (meta-labeling pipeline)
      ──► ml_layer/purged_cv.py          (purged k-fold + embargo)
      ──► backtesting/cpcv.py            (combinatorial purged CV)
      ──► backtesting/deflated_sharpe.py (DSR)
      ──► backtesting/pbo.py             (probability of backtest overfit)
      ──► portfolio/hrp.py               (hierarchical risk parity)

Chan  ──► signal_battery/mean_reversion.py  (O-U half-life + z-score)
      ──► signal_battery/stat_arb.py        (Engle-Granger + Kalman)
      ──► bet_sizing/kelly.py               (fractional Kelly)

Jansen──► ml_layer/meta_labeler.py       (LightGBM + isotonic calibration)
      ──► ml_layer/regime_detector.py    (LSTM + attention pooling)
      ──► ml_layer/rl_agent.py, rl_env.py (Gymnasium + PPO)
      ──► feature_factory/sentiment.py   (FinBERT)
      ──► feature_factory/autoencoder.py (noise-reducing latents)

Clenow──► signal_battery/trend_following.py (EMA/Donchian)
      ──► signal_battery/carry.py          (futures roll + funding)
      ──► bet_sizing/cascade.py (Layer 4)  (ATR risk-per-trade cap)

Narang──► src/*                          (the 5-layer separation itself)
      ──► execution/tca.py               (post-trade TCA)

Johnson ► execution/algorithms.py        (TWAP, VWAP, Iceberg)
      ──► execution/broker_factory.py    (Smart Order Router)
      ──► backtesting/transaction_costs  (square-root impact)

Isichenko► portfolio/factor_risk.py      (PCA factor model)
      ──► portfolio/risk_parity.py       (Griveau-Billion coord descent)

Sinclair► feature_factory/volatility.py  (GARCH + VRP)
      ──► signal_battery/volatility_signal.py (VRP-regime sizing)
      ──► bet_sizing/cascade.py (Layer 3)     (vol adjustment + VRP haircut)
```

---

## 4. Architecture: Narang's 5 layers, 10 subsystems

The system is a composition of **10 subsystems** that map cleanly onto
**Narang's 5 layers**. Each layer has a single responsibility and a
stable interface to its neighbours.

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                         NARANG LAYER 1 — ALPHA                        │
 │  ──────────────────────────────────────────────────────────────────── │
 │   Subsystem 1: Data Engine         src/data_engine/                   │
 │   Subsystem 2: Feature Factory     src/feature_factory/               │
 │   Subsystem 3: Signal Battery      src/signal_battery/                │
 │                                                                       │
 │   output:  signals_df {timestamp, symbol, family, side, confidence}   │
 └──────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                          NARANG LAYER 2 — RISK                        │
 │  ──────────────────────────────────────────────────────────────────── │
 │   Subsystem 4: Labeling Engine     src/labeling/                      │
 │   Subsystem 5: ML Model Layer      src/ml_layer/                      │
 │                                                                       │
 │   output:  meta_probability p ∈ [0,1] per signal                      │
 │           regime probabilities over {trend_up, trend_down, MR, HV}    │
 └──────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                      NARANG LAYER 3 — TRANSACTION COST                │
 │  ──────────────────────────────────────────────────────────────────── │
 │   (shared module — backtesting/transaction_costs.py + execution/tca)  │
 │   square-root impact + commission + spread + slippage                 │
 │                                                                       │
 │   output:  cost estimate (pre-trade)   +  TCA (post-trade)            │
 └──────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                 NARANG LAYER 4 — PORTFOLIO CONSTRUCTION               │
 │  ──────────────────────────────────────────────────────────────────── │
 │   Subsystem 6: Bet Sizing          src/bet_sizing/                    │
 │   Subsystem 7: Portfolio           src/portfolio/                     │
 │   Subsystem 8: Backtesting (gate)  src/backtesting/                   │
 │                                                                       │
 │   output:  target_weights  (per symbol, signed, clipped to limits)    │
 └──────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                        NARANG LAYER 5 — EXECUTION                     │
 │  ──────────────────────────────────────────────────────────────────── │
 │   Subsystem 9:  Execution Engine   src/execution/                     │
 │   Subsystem 10: Monitoring         src/monitoring/                    │
 │                                                                       │
 │   output:  fills  + metrics + alerts + audit log entries              │
 └──────────────────────────────────────────────────────────────────────┘
```

### Subsystem → layer crosswalk

| # | Subsystem | Narang layer | Phase shipped | Primary path |
|---|-----------|--------------|---------------|--------------|
| 1 | Data Engine | Alpha | 1 | `src/data_engine/` |
| 2 | Feature Factory | Alpha | 2 | `src/feature_factory/` |
| 3 | Signal Battery | Alpha | 2 | `src/signal_battery/` |
| 4 | Labeling Engine | Risk | 3 | `src/labeling/` |
| 5 | ML Model Layer | Risk | 3 | `src/ml_layer/` |
| 6 | Bet Sizing | Portfolio Construction | 3 | `src/bet_sizing/` |
| 7 | Portfolio Construction | Portfolio Construction | 4 | `src/portfolio/` |
| 8 | Backtesting | Portfolio Construction (validation gate) | 4 | `src/backtesting/` |
| 9 | Execution Engine | Execution | 5, 6 | `src/execution/` |
| 10 | Monitoring | Execution (crosscutting) | 5 | `src/monitoring/` |

---

## 5. End-to-end dataflow

### 5.1 Happy-path trade lifecycle

```
  (1) tick                (2) bar             (3) features         (4) signals
  ──────────►   Data      ──────────►  Feature  ──────────►   Signal
  alpaca/ccxt/  Engine    TimescaleDB  Factory  assembler df  Battery
  ibkr stream  (Phase 1)              (Phase 2)              (Phase 2)
                                                                    │
                                                                    ▼
        (7) target            (6) meta-label            (5) signals_df
         weights              + regime probs
  ◄──────────── Portfolio ◄────────────  ML Layer  ◄────────────  Signals
                Construction           (Phase 3)                  arrive
                (Phase 4)
  HRP or PPO
       │
       │ (8) per-symbol            (9) order             (10) fill
       ▼   target weight
  Bet Sizing  ──────────►  Order       ──────────►  Broker      ──────────►
  Cascade                 Manager                    Factory                Exchange
  (Phase 3)              + circuit      chooses        (Alpaca/
                         breakers       algo           CCXT/IBKR)
                         (Phase 5)      (Imm/TWAP/
                                         VWAP/Iceberg)
                                                                    │
                                                                    ▼
                                                            (11) fill →
                                                             Portfolio state,
                                                             MetricsCollector,
                                                             AuditLog,
                                                             TCA (T+0)
```

### 5.2 The per-cycle pipeline (one "tick")

The `PaperTradingPipeline.run_cycle()` and `LiveTradingPipeline.run_cycle()`
methods are the single entry point per trading tick. The flow:

```
  run_cycle(bars_df)
     │
     ├──► FeatureAssembler.assemble(bars_df)         ▸ ~200 features
     │
     ├──► SignalBattery.generate_all(bars_df)        ▸ signals_df
     │
     ├──► MetaLabeler.predict_proba(features,        ▸ p per signal
     │                              signals)
     │
     ├──► RegimeDetector.predict(features)           ▸ regime probs
     │
     ├──► BetSizingCascade.compute_position_sizes
     │        _batch(signals, metaprobs, features)    ▸ size per signal
     │
     ├──► Portfolio.compute_target_portfolio(        ▸ target_weights
     │         sizes, regime)
     │
     ├──► OrderManager.execute_target_portfolio(
     │         target_weights, prices, nav)
     │        ├─ circuit_breakers.check_pre_trade()
     │        ├─ transaction_costs.estimate()
     │        ├─ select_execution_algo()
     │        ├─ algo.execute() via BrokerFactory
     │        └─ update PortfolioState from fills
     │
     ├──► MetricsCollector.update(...)               ▸ Prometheus
     │
     ├──► FeatureDriftDetector.check_drift(...)      ▸ retrain trigger
     │
     └──► AuditLog.log(cycle_summary)                ▸ HMAC chain
```

### 5.3 The retrain loop (weekly + drift-triggered)

```
  RetrainScheduler.should_retrain()
     │  (conditions: 7-day cadence OR drift > 20% OR manual)
     ▼
  RetrainPipeline.run()
     │
     ├──► load bars + signals + features              ▸ training set
     │
     ├──► MetaLabelingPipeline.prepare_training_data  ▸ (X, y, weights)
     │        triple_barrier + sample_weights
     │
     ├──► (optional) tune_meta_labeler (Optuna TPE)   ▸ best params
     │
     ├──► MetaLabeler.train(X, y, weights)
     │        LightGBM + purged CV + isotonic calib
     │
     ├──► StrategyGate.validate(candidate)            ◄── PROMOTION GATE
     │        ├─ CPCVEngine: ≥60% of 45 paths > 0?
     │        ├─ DSR: p-value < 0.05?
     │        └─ PBO: < 0.40?
     │
     ├──► ModelRegistry.log_run(candidate)            ▸ MLflow
     │
     └──► if gate passed AND improvement > 5%:        ▸ promote
             ModelRegistry.promote(candidate)
           else:
             keep incumbent; audit-log rejection
```

### 5.4 Control plane (live)

```
                   ┌───────────────────────┐
                   │     Operator (CLI)    │
                   └──┬────────────────────┘
                      │  touch .operator_checkin
                      │  make preflight
                      │  make live-start | live-stop | live-flatten
                      │  make recover
                      │  --approve-rl-promotion / --revert-to-hrp
                      ▼
   ┌──────────────────────────────────────────┐
   │           LiveTradingPipeline            │
   │  ┌──────────────┐  ┌──────────────────┐  │
   │  │ Preflight    │  │  Capital         │  │
   │  │ Checker      │  │  Deployment Ctrl │  │
   │  │ (18 blocker) │  │  (pilot/beta/…)  │  │
   │  └──────────────┘  └──────────────────┘  │
   │  ┌──────────────┐  ┌──────────────────┐  │
   │  │ Shadow-aware │  │  RL Promotion    │  │
   │  │ Optimizer    │──│  Controller      │  │
   │  │ (HRP ⇆ RL)   │  │  (3d/5%DD gate)  │  │
   │  └──────────────┘  └──────────────────┘  │
   │  ┌──────────────┐  ┌──────────────────┐  │
   │  │ Snapshot +   │  │  Compliance      │  │
   │  │ Recovery     │  │  Audit Log       │  │
   │  │ (checksum)   │  │  (HMAC chain)    │  │
   │  └──────────────┘  └──────────────────┘  │
   └──────────────────────────────────────────┘
                      │
                      ▼
            Bet sizing cascade
                      │
                      ▼
            OrderManager + CircuitBreakers (8)
                      │
                      ▼
        BrokerFactory → Alpaca / CCXT / IBKR
                      │
                      ▼
                  Exchange
```

---

## 6. Subsystem 1 — Data Engine

**Phase:** 1 · **Narang layer:** Alpha · **Path:** `src/data_engine/`

### 6.1 Responsibilities

Ingest real-time + historical market data from multiple venues, classify
ticks, construct **information-driven bars**, persist to TimescaleDB,
and validate bar quality. The data engine is the only part of the system
that knows about raw ticks. Everything upstream of the Feature Factory
consumes bars, not ticks.

### 6.2 Why information-driven bars, not clock bars?

AFML Ch. 2 observes that clock bars (e.g., 1-minute OHLC) under-sample
active periods and over-sample quiet periods, distorting the volatility
estimate and biasing everything downstream. **Information-driven bars**
close when a fixed amount of *information* (ticks, volume, dollars, or
imbalance) has accumulated. This produces closer-to-i.i.d. returns,
which is what every statistical test downstream assumes.

### 6.3 Bar types (`bars/constructors.py`)

| Bar type | Closes when… | Use |
|----------|-------------|-----|
| TickBar | fixed #ticks accumulated | generic, fast |
| VolumeBar | fixed volume | crypto, equities |
| DollarBar | fixed notional traded | cross-asset comparability |
| TIB (Tick Imbalance) | cumulative \|b_t\| > EWMA expected | **default** — surfaces directional flow |
| VIB (Volume Imbalance) | cumulative \|b_t·v_t\| > EWMA | volume-weighted variant |

TIB and VIB use EWMA of imbalance (span 100 bars) to set a dynamic
threshold; minimum threshold of 5 ticks prevents degenerate single-tick
bars. `BaseBarConstructor` is the abstract parent; subclasses override
`_is_bar_complete()`.

### 6.4 ETF trick (`bars/etf_trick.py`)

Synthetic continuous contracts for futures by rolling on open-interest
crossover (when back_oi > front_oi). Unlike back-adjustment, preserves
returns and avoids introducing artefacts at the roll. Initial value 100.

### 6.5 CUSUM filter (`bars/cusum_filter.py`)

Symmetric CUSUM over log-returns. Fires when cumulative positive or
negative drift exceeds threshold (default 1.5× rolling std). Used by the
Signal Battery as an event filter — rather than generating signals
every bar, we generate them only on CUSUM events (reduces noise +
multiple-comparisons burden).

### 6.6 Ingestion (`ingestion/`)

```
        BaseIngestionAdapter  (abstract)
         │
         ├── AlpacaAdapter       (equities: WebSocket + REST; IEX or SIP)
         ├── CCXTAdapter         (crypto: 100+ exchanges via ccxt)
         └── IBKRAdapter         (futures + equities backup, via ib_insync)

        IngestionPipeline  (runner.py)
          │
          ├─ adapter.stream_ticks()      ──► tick queue
          ├─ classify direction (Lee-Ready)
          ├─ fan out to [TIB, VIB, Dollar, Volume, Tick] constructors
          └─ buffered insert (100 rows) into TimescaleDB
```

Resilience: async reconnect with backoff on WS disconnect, structured
logs via `loguru`, heartbeat to `MetricsCollector`.

### 6.7 Storage (`storage/`)

- **`database.py`** — SQLAlchemy-based TimescaleDB wrapper. Hypertables:
  `raw_ticks`, `bars`, `cusum_events`, `features`, `signals`, `labels`,
  `meta_labels`, `positions_history`. 7-year retention.
- **`feature_store.py`** — Parquet-backed versioned feature matrices
  (local filesystem or GCS). Partitioned by symbol × date. The feature
  store is the single source of truth for training data; TimescaleDB
  is the source of truth for live execution.

### 6.8 Validation (`validation/bar_validator.py`)

Statistical QA on the bar series:

| Check | Threshold | Failure mode |
|-------|-----------|--------------|
| Minimum bars | ≥100 | too little history |
| Return mean std | >0 | constant series |
| Skewness | \|skew\| < 5 | extreme tail |
| Kurtosis | <50 | fat tails out of control |
| Avg ticks/bar | ≥5 | threshold too tight |
| Jarque-Bera | reported | normality sanity |
| Ljung-Box | reported | residual autocorr |
| Variance ratio | reported | random-walk sanity |

Outputs `HEALTHY` / `WARNING` / `FAIL` status.

---

## 7. Subsystem 2 — Feature Factory

**Phase:** 2 · **Narang layer:** Alpha · **Path:** `src/feature_factory/`

### 7.1 Responsibilities

Given a bar series, produce a dense feature matrix that captures
microstructure, memory, regimes, volatility, sentiment, on-chain flow,
and compressed latents. The result is one row per bar, ~200 columns.

### 7.2 Feature families

| # | Family | Module | Key features | Typical window |
|---|--------|--------|--------------|----------------|
| 1 | Fractional Diff | `fractional_diff.py` | FFD(close), FFD(volume), FFD(dollar_vol) — min d preserving stationarity + memory | weight threshold 1e-5 |
| 2 | Structural Breaks | `structural_breaks.py` | CUSUM rolling magnitude, SADF, GSADF, Chow test | 50-bar default |
| 3 | Entropy | `entropy.py` | Shannon (10 bins), Lempel-Ziv (LZ76), ApEn, SampEn | 100 bars |
| 4 | Microstructure | `microstructure.py` | Kyle λ, Amihud, Roll spread, VPIN, order-flow imbalance, trade intensity | 50 bars |
| 5 | Volatility | `volatility.py` | GARCH(1,1) refit every 25 bars, RV_short/RV_long, vol-of-vol, IV−RV | 252 bars |
| 6 | Classical | `assembler.py` | RSI-14, Bollinger-width-20, return z-scores [5,10,20] | various |
| 7 | Sentiment | `sentiment.py` | FinBERT per-bar score, multi-window momentum, article count | lazy-loaded |
| 8 | On-Chain | `onchain.py` | exchange inflow/outflow, whale activity, network health, stablecoin supply (Glassnode) | varies |
| 9 | Autoencoder | `autoencoder.py` | noise-reduced latents of all upstream features | trained offline |

### 7.3 Why fractional differentiation (FFD)?

AFML Ch. 5: integer differencing destroys memory that may be predictive;
level data is non-stationary. FFD finds the **minimum fractional d** that
makes the series stationary while preserving the maximum memory. Weight
series is truncated at `|w_k| < 1e-5` to keep computation tractable.

### 7.4 Why GARCH(1,1)?

Sinclair: volatility clusters. A single realised-vol number is naïve.
GARCH(1,1) captures persistence with only 3 parameters; refitting every
25 bars keeps it responsive without overfitting.

### 7.5 Why an autoencoder on top?

Jansen Ch. 20: a dense tail of near-duplicate features hurts tree models
with limited depth. The autoencoder produces compressed latents that the
meta-labeler can consume in addition to raw features. Appended, not
replacing.

### 7.6 Assembler pipeline

```
 raw bars
    │
    ├── fractional_diff ──┐
    ├── structural breaks─┤
    ├── entropy ──────────┤
    ├── microstructure ───┤          ┌── pass 1: ADF test each column;
    ├── volatility ───────├─► align ─┤            FFD if non-stationary
    ├── classical ────────┤          └── pass 2: re-verify; drop if fails
    ├── sentiment ────────┤
    ├── onchain ──────────┤
    └── autoencoder ──────┘
             │
             ▼
     FeatureMatrix [N_bars × ~200]
             │
             └── get_feature_hash(row) → SHA-256 → reproducible replay
```

The two-pass stationarization is important: some features (e.g., FFD on a
different series) may still fail ADF on the assembled set. Pass 2 drops
any residual non-stationary columns.

---

## 8. Subsystem 3 — Signal Battery

**Phase:** 2 · **Narang layer:** Alpha · **Path:** `src/signal_battery/`

### 8.1 Responsibilities

Produce **uncorrelated modest-edge proposals** across the 7 canonical
families. The battery never decides whether to trade — it only proposes.
The meta-labeler decides which proposals to take.

### 8.2 Base abstraction (`base_signal.py`)

```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    family: str                 # "ts_momentum", "mean_reversion", …
    side: int                   # -1, 0, +1
    confidence: float           # [0.0, 1.0]
    metadata: dict              # per-family extras

class BaseSignalGenerator(ABC):
    REQUIRED_COLUMNS: set[str]
    @abstractmethod
    def generate(self, bars, **kwargs) -> list[Signal]: ...
```

### 8.3 The 10 generators across 7 families

| Family | Generator | Module | Core idea | Edge source |
|--------|-----------|--------|-----------|-------------|
| Momentum | TS Momentum | `momentum.py` | multi-lookback [21,63,126,252]; per-lookback z-score; sum | Chan |
| Momentum | Cross-sectional Momentum | `momentum.py` | top/bottom decile long/short with 1-month skip | Jegadeesh-Titman |
| Mean Reversion | O-U z-score | `mean_reversion.py` | O-U half-life (OLS lag), ADF stationarity, entry \|z\|>1.5 exit \|z\|<0.5 | Chan |
| Stat Arb | Pairs | `stat_arb.py` | Engle-Granger (p<0.05), Kalman-filtered hedge ratio, spread z-score | Chan |
| Trend | MA / Donchian | `trend_following.py` | dual/triple EMA crossover + Donchian breakout, ATR-normalized | Clenow |
| Carry | Futures roll + Funding | `carry.py` | annualized front-back roll yield; crypto funding arb (3/day) | Clenow |
| Cross-Exchange Arb | Spread scanner | `cross_exchange_arb.py` | best-ask vs best-bid across CCXT venues; threshold 10bps+20bps buffer | crypto microstructure |
| Volatility | VRP regime | `volatility_signal.py` | VRP = IV − RV; top quartile → short vol, bottom → long vol | Sinclair |

Note: the design doc lists 7 signal families; there are 10 generators
because some families have multiple variants (e.g., TS + cross-sectional
momentum).

### 8.4 Orchestrator (`orchestrator.py`)

```
   SignalBattery
     │
     │  register(generator, kind=...)
     │    kinds: "bars" | "panel" | "pair" | "exchange_prices" | "bars_extra"
     │
     └── generate_all(bars_df, context)
           │
           │  fan-out dispatch on kind
           │  INTENTIONALLY NO conflict resolution — the meta-labeler arbitrates
           │
           └── returns flat DataFrame:
                 timestamp, symbol, family, side, confidence,
                 meta_<k1>, meta_<k2>, ...  (from Signal.metadata)
```

Optional CUSUM event filtering — if passed, signals are only emitted at
CUSUM events (reduces multiple-comparisons burden, AFML Ch. 3).

### 8.5 Why not resolve conflicts in the battery?

Design choice. A momentum signal and a mean-reversion signal on AAPL can
both be right — at different regimes. Resolving in the battery requires
the battery to know about regime, which breaks P3 (layer separation).
Instead, we emit both; the meta-labeler learns which regime favours
which, and HRP handles residual correlation.

---

## 9. Subsystem 4 — Labeling Engine

**Phase:** 3 · **Narang layer:** Risk · **Path:** `src/labeling/`

### 9.1 Responsibilities

Convert signals + price paths into `(X, y, sample_weights)` suitable for
supervised ML. Use **triple-barrier labels** and **AFML sample weights**
to neutralise bias from overlapping events.

### 9.2 Triple-barrier labeling (`triple_barrier.py`)

Every signal is labelled by which of three barriers is touched first:

```
                    ▲
     upper barrier ─┼─ entry · (1 + upper_mult · vol)      → label = +1 (TP)
                    │
             entry  ●────────────────────────────────────►  time (bars)
                    │                                        │
                    │                                        │ vertical barrier
                    │                                       (max_holding_period)
                    │                                        │
     lower barrier ─┼─ entry · (1 − lower_mult · vol)      → label = -1 (SL)
                    │
                    │    if vertical hit first              → label = sign(Δ)
                    ▼                                         (or 0 if flat)
```

Volatility scaling uses an EWM std (span 100 bars, configurable).
Max holding period defaults to 50 bars; it clamps to final bar near end.

**Asymmetric multipliers by family** — different families have different
natural risk-reward profiles; the labels should reflect that:

| Family | upper_mult | lower_mult | Reasoning |
|--------|-----------|-----------|-----------|
| Carry / funding | 3.0 | 1.0 | wide TP, tight SL — slow-moving positive carry |
| Mean reversion | 1.0 | 1.5 | tight TP, wider SL — band reversion |
| Momentum / trend / breakout | 2.5 | 1.0 | wide TP, tight SL — asymmetric payoff |
| Arb | 1.5 | 1.5 | symmetric |
| Default | 2.0 | 2.0 | symmetric |

### 9.3 Meta-labeling pipeline (`meta_labeler_pipeline.py`)

```
 signals_df + features_df + close
     │
     ├── drop neutral (side=0) signals
     │
     ├── compute daily volatility (EWM std, span from config)
     │
     ├── per-family: apply family-specific mult; make_labels()
     │
     ├── concat per-family frames   (multi-signal events stay separate)
     │
     ├── compute sample weights      ──► see 9.4
     │
     ├── backward-fill features to event timestamps
     │
     ├── append signal metadata: one-hot family, signal_side, signal_conf
     │
     └── binary meta-label:
            y = 1   if triple-barrier label > 0 (profitable)
            y = 0   otherwise
```

Binary target is the key insight of meta-labeling: we are not predicting
direction (the signal already picks direction) — we are predicting
**should this signal be taken?**

### 9.4 Sample weighting (`sample_weights.py`)

AFML Ch. 4. Overlapping events violate the i.i.d. assumption of every
tree-based model. We correct with four weighting schemes combined:

1. **Average Uniqueness** — `mean(1 / concurrent_count)` over each
   event's life. Events fully overlapping others get weight 0.5;
   isolated events get 1.0. Implemented in O(N+T) via difference-array.
2. **Sequential Bootstrap** (AFML 4.5.2) — respects overlap: after each
   draw, probabilities for correlated remaining samples are downweighted.
   Fallback to uniqueness-weighted draw if event ranges omitted.
3. **Return Attribution** — `w_i = |Σ_t r_t / c_t|` over each event's
   life. Rescales so Σw_i = N. Bigger-move events weigh more.
4. **Time Decay** — rank-based linear decay in `[time_decay, 1.0]`,
   so recent events weigh more. Stable against duplicate timestamps.

**Combined weight:** `raw = uniqueness × ret_attr × decay`, renormalised
to sum to N.

---

## 10. Subsystem 5 — ML Model Layer

**Phase:** 3 · **Narang layer:** Risk · **Path:** `src/ml_layer/`

### 10.1 Tier structure

Three tiers; each can be run independently; live default is Tier 1 + Tier 2.

```
 Tier 1: Meta-labeler                 LightGBM + isotonic calib
         p ∈ [0, 1] per signal        (default live — always on)
           │
           ▼
 Tier 2: Regime detector              LSTM + attention → 4 classes
         regime probabilities         (trend_up / trend_down / MR / HV)
           │
           ▼
 Tier 3: RL portfolio agent           Gymnasium TradingEnv + PPO
         shadow or live allocator     (shadow mode by default)
                                      HRP is the live baseline
```

### 10.2 Tier 1 — Meta-labeler (`meta_labeler.py`)

- **Model choice:** LightGBM (default) / XGBoost / RandomForest, picked
  at config time.
- **Default LightGBM params:** `lr=0.05, n_est=500, max_depth=5,
  min_child_weight=10, subsample=0.8, colsample=0.8, reg_alpha=0.1,
  reg_lambda=1.0`.
- **Calibration:** `ProbabilityCalibrator` fits isotonic regression on
  out-of-fold predictions; outputs are clipped to [0,1] and monotone
  non-decreasing. This matters — raw tree outputs are poorly calibrated,
  and the bet-sizing cascade assumes `p` is a probability.
- **Purged CV:** integrated via `PurgedKFoldCV` (see 10.6).
- **Output:** `p(profitable | signal, features)` per signal.

### 10.3 Tier 2 — Regime detector (`regime_detector.py`)

- **Architecture:** LSTM (hidden_dim configurable) + single-head
  attention pooling → softmax over 4 classes.
- **Labels:** Gaussian HMM on `[returns, volatility]`, 4 latent states
  mapped by inspection to `{trend_up, trend_down, mean_reverting, high_vol}`.
- **Integration:** regime probabilities feed back to:
  - the meta-labeler as a feature block (so it learns regime-conditional
    effectiveness per family),
  - the bet-sizing cascade as a regime-conditional scaling hook,
  - the portfolio as a tilt signal (±20%).

### 10.4 Tier 3 — RL agent (`rl_agent.py`, `rl_env.py`, `rl_shadow.py`)

**Environment** (`rl_env.py`, Gymnasium `gym.Env`):
- Observation: multi-symbol closes `[T, n_symbols]`, features
  `[T, n_features]`, meta-probs, regime probs.
- Action: per-symbol tier 0..K (0 = flat, K long-only sizes).
- Reward: rolling-20 Sharpe − drawdown penalty (λ=10, thr=0.05)
  − turnover cost (μ=0.1).
- Config: initial_capital $100K, max_position_pct 0.20, tcost 5bps.

**Agent** (`rl_agent.py`):
- PPO via `stable_baselines3` with `MultiInputPolicy` default.
- Defaults: `lr=3e-4, n_steps=2048, batch=64, n_epochs=10, gamma=0.99,
  gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
  max_grad_norm=0.5`.

**Shadow comparison** (`rl_shadow.py`):
- `ShadowComparisonEngine` logs paired HRP vs RL decisions every cycle.
- Promotion gate: paired t-test on realised P&L — p < 0.05 AND
  t > 0 (RL better on average).
- **Auto-revert:** `RLPromotionController` monitors RL after promotion;
  3 consecutive down-days or 5% drawdown → automatic revert to HRP.
  Design-doc §6 requires **6 months of paper** before RL can go live.

### 10.5 Model registry (`model_registry.py`)

Thin MLflow wrapper:
- `tracking_uri = sqlite:///mlflow.db` by default (swap to S3 + Postgres
  for prod).
- Experiment: "meta-labeler".
- Serialisation via joblib (handles custom `MetaLabeler` class).
- API: `log_training_run`, `load_model`, `rank_runs`, `promote_model`.

### 10.6 Purged cross-validation (`purged_cv.py`)

Implements AFML Ch. 7:

```
 k-fold split:
   fold:  [   train   |PURGE| test |EMBARGO|   train   ]
                        └─── purge: drop train rows whose labels
                        │        overlap test label periods
                        └─── forward embargo: drop train rows in
                             [test_end, test_end + embargo_pct]
```

Defaults: `n_splits=5, embargo_pct=0.01`. Exposed via `split(X, y, labels_df)`.

### 10.7 Hyperparameter tuning (`tuning.py`)

- Optuna **TPE** sampler (default) over LR, n_estimators, max_depth,
  min_child_weight, subsample, colsample, reg_alpha, reg_lambda.
- **MedianPruner** — prune unpromising trials mid-CV to save compute.
- **Scoring:** ROC-AUC default; supports accuracy, F1, precision,
  recall, log-loss. Higher-is-better normalised.
- Evaluated via `PurgedKFoldCV` above — no tuning ever uses plain k-fold.

### 10.8 Feature importance (`feature_importance.py`)

Four methods, combined for feature selection:

| Method | Source | Tradeoff |
|--------|--------|----------|
| MDI | tree split gain | fast, biased toward high-cardinality features |
| MDA | OOF permutation | most trustworthy, expensive; uses purged CV |
| SFI | 1-feature model CV | detects interactions that MDA misses |
| SHAP | per-row attribution | global + local explanations |

**Selection rule:** keep feature if MDA-significant OR SFI-significant
(union — conservative about dropping).

### 10.9 Retrain pipeline (`retrain_pipeline.py`)

End-to-end: bars → features → signals → labels → train → gate (CPCV +
DSR + PBO) → registry promotion. Rejection if improvement < 5% (default
`min_improvement_pct`) or any gate fails. Async entry point `run()`.

Trigger sources:
- **Scheduled** — 7-day cadence, min 100 new bars.
- **Drift-triggered** — FeatureDriftDetector reports >20% features drifted.
- **Manual** — operator via `scripts/retrain_model.py`.
- **Emergency** — after a circuit-breaker trip.

---

## 11. Subsystem 6 — Bet Sizing Cascade

**Phase:** 3 · **Narang layer:** Portfolio Construction · **Path:** `src/bet_sizing/`

### 11.1 Responsibilities

Given `(signal, meta_probability, features)` produce a signed position
size `f ∈ [-1, +1]` that is clipped by a stack of risk constraints. Every
layer's output is stored for audit.

### 11.2 The five-layer cascade

```
   Layer 1  AFML Sizing              p → raw fraction
            (prob → concave size)
                      │
                      ▼
   Layer 2  Kelly Cap                cap at ¼-Kelly (default)
            (risk-of-ruin bound)
                      │
                      ▼
   Layer 3  Vol Adjust + VRP         scale by realized vol;
            haircut                  25% haircut in top-quartile VRP
                      │
                      ▼
   Layer 4  ATR Cap                  Clenow risk-per-trade cap
            (trend/futures only)     (skipped otherwise)
                      │
                      ▼
   Layer 5  Risk Budget              single / family / gross / crypto /
            (hard limits)            sector caps
                      │
                      ▼
            final_size = side × magnitude
            (audit log stores every layer's output)
```

### 11.3 Layer 1 — AFML sizing (`afml_sizing.py`)

From AFML Ch. 10: convert probability to size via

```
  z(p)    = (p − 0.5) / √[p·(1−p)]
  size(p) = (2·Φ(z) − 1) · max_raw_size
```

where `Φ` is the standard normal CDF. Concave in `p`, monotonic in
`p`, and goes to 0 at `p = 0.5`. `max_raw_size` defaults to 1.0 (100%
NAV raw — clipped downstream). Output is unsigned magnitude.

### 11.4 Layer 2 — Kelly cap (`kelly.py`)

Fractional Kelly:

```
  f* = (p · W − (1 − p) · L) / (W · L)
  f_frac = fraction × f*         (fraction default 0.25 = ¼ Kelly)
```

`W` and `L` are average win/loss for the signal family from historical
backtest. If unavailable, layer logs debug and passes through. Clamps to
`[0, 1]`, then `min(afml_size, kelly_cap)`.

Why fractional Kelly? Full Kelly is optimal growth but assumes `W`, `L`,
`p` are known exactly. They aren't. Quarter Kelly trades a small expected
growth loss for a large reduction in drawdown variance.

### 11.5 Layer 3 — Vol adjust + VRP haircut (`cascade.py`)

```
  vol_adjusted = kelly_capped × (avg_vol / current_vol)    (if ratio ≠ 1)
  if vrp_quartile == 3:                                    # top quartile VRP
      vol_adjusted *= (1 − vrp_haircut)                    # default 25% haircut
  vol_adjusted = clip(vol_adjusted, 0, max_raw_size)
```

Rationale (Sinclair): volatility matters twice — as a *current*
scaling factor (bigger positions in quieter markets) and as a *regime*
factor (top-quartile VRP means options are expensive ⇒ short-vol
strategies face inventory risk).

### 11.6 Layer 4 — ATR cap (`cascade.py`, Clenow-style)

Conditional — applied only for trend families or futures:

```
  per_unit_risk = atr × atr_multiplier × point_value         # default mult 2.0
  max_units     = (nav × risk_per_trade) / per_unit_risk     # default 1% NAV
  max_fraction  = max_units × price × point_value / nav
  result        = min(vol_adjusted, max_fraction)
```

Ensures a single trade can lose at most `risk_per_trade × NAV` if the ATR
stop is hit.

### 11.7 Layer 5 — Risk budget (`cascade.py`)

Hard limits enforced against the whole portfolio:

| Limit | Default | Notes |
|-------|---------|-------|
| Max single position | 10% NAV | always enforced |
| Max family allocation | 30% NAV | excluding the symbol itself |
| Max gross exposure | 150% | 1.0 default no-leverage; 1.5× with leverage |
| Max crypto allocation | 30% NAV | asset_class="crypto" only |
| Max sector exposure | 20% NAV | equities with GICS tags |

### 11.8 Orchestrator output

`BetSizingCascade.compute_position_size()` returns a dict with:
- `final_size` (signed)
- outputs of all 5 layers individually
- `constraints_applied` list (audit trail)

`compute_position_sizes_batch()` is the vectorised path used in live.

---

## 12. Subsystem 7 — Portfolio Construction

**Phase:** 4 · **Narang layer:** Portfolio Construction · **Path:** `src/portfolio/`

### 12.1 Responsibilities

Given per-signal desired sizes and a covariance estimate, produce
**target portfolio weights** that:
- diversify risk (not dollars),
- respect factor exposure constraints,
- enforce risk-budget hard limits from §11.

### 12.2 Hierarchical Risk Parity (`hrp.py`) — default

AFML Ch. 20. Three steps:

```
   returns [T × N]
      │
      │ 1. Distance:  d_{ij} = √(½ (1 − corr_{ij}))
      │ 2. Linkage:   scipy.cluster.hierarchy.linkage(d, method="single")
      │ 3. Quasi-diagonalize: reorder covariance by leaf order
      │ 4. Recursive bisection:
      │      α = 1 − σ²(left) / (σ²(left) + σ²(right))
      │      scale left by α, right by (1-α), recurse
      ▼
   HRP weights (N,)
```

Why HRP over mean-variance optimisation? MVO explodes under noisy
covariance. HRP never inverts the covariance, is stable under mis-specified
inputs, and produces interpretable cluster-based risk decomposition.

**Streaming variant:** `HRPPortfolioOptimizer` rebalances every 5 bars
(configurable).

### 12.3 Factor risk model (`factor_risk.py`)

PCA factor model:
- **5 latent factors** (hardcoded for stability; Isichenko recommends
  5–8 for US equities).
- **252-bar lookback.**
- Fits via eigendecomposition of covariance.
- Tracks `explained_variance_ratio_` for monitoring.
- **Risk decomposition:** `get_risk_decomposition(weights)` returns
  `total_risk, systematic, idio, factor_contributions, pct_systematic`.
- **Neutralisation:** `neutralize_factors(weights, [factor_ids])`
  projects weights onto the null-space of selected factors via Woodbury
  solve (regularised `gram += 1e-12·I` for collinear factors).

### 12.4 Risk parity (`risk_parity.py`)

Griveau-Billion coordinate descent:
- Minimises `Σ_i (RC_i − budget_i)²` s.t. `Σw = 1, w ≥ 0`.
- Seed: inverse-volatility `w ∝ 1/σ`.
- Up to 1000 iterations, tolerance 1e-8.
- Provably converges for PSD covariance.

### 12.5 Multi-strategy allocator (`multi_strategy.py`)

Design-doc §8.5. Four-layer allocator that combines everything:

```
  L1 — Strategy-level weighting across families
        (HRP default; options: risk_parity, equal, momentum)
            │
            ▼
  L2 — Instrument-level weighting within each strategy
        (same choice set)
            │
            ▼
  L3 — Signed positions:  target × signal_side × bet_size
            │
            ▼
  L4 — Clipping / budgets:
        · single-position   ±10% NAV
        · single-strategy   30% NAV
        · gross exposure    ≤150%
        · crypto allocation ≤30%
        · min_trade         ≥1% (ignore tiny rebalances)
        · regime tilt       ±20% per family group
```

Residual from clipping goes to cash (per design-doc §8.5 — we do *not*
renormalise, so the risk budget is honoured strictly rather than
smeared across remaining positions).

**Regime tilt map:**
- `trending / bull` → boost `momentum`, `trend` by 20%.
- `mean_reverting / range` → boost `mean_reversion`, `stat_arb`, `arb`
  by 20%.

---

## 13. Subsystem 8 — Backtesting and Validation

**Phase:** 4 · **Narang layer:** Portfolio Construction (validation gate)
· **Path:** `src/backtesting/`

### 13.1 Responsibilities

Simulate the full signal → label → ML → sizing → execution pipeline against
historical data, and subject every candidate strategy to three gates before
promotion.

### 13.2 Walk-forward backtester (`walk_forward.py`)

Bar-by-bar simulator:
- triple-barrier exits with realistic price paths,
- configurable execution delay (default 1 bar — signal at t-1 → entry at t),
- realistic transaction costs on both entry and exit (see 13.3),
- mark-to-market NAV tracking with equity curve,
- max holding 20 bars; positions force-closed at end.

Outer loop: `run_expanding_window()` wraps with retraining (default
252-bar interval, 504-bar init) to avoid lookahead in model parameters.

### 13.3 Transaction costs (`transaction_costs.py`)

Johnson square-root market impact model:

```
  cost = commission + spread + slippage + impact
  impact = σ · √(participation) · notional
```

Asset-class defaults:

| Asset class | Commission | Spread | Slip | Impact coeff |
|-------------|-----------|--------|------|--------------|
| Equities | 0.5¢/share (min $1) | 2bps | 1bps | 0.10 |
| Crypto | 0.1¢/share (no min) | 3bps | 2bps | 0.15 |
| Futures | $1.25/contract | 1bps | 0.5bps | 0.08 |

Models include IBKR commission tiers. Used by both backtester and by
`OrderManager.estimate_cost()` pre-trade.

### 13.4 CPCV — Combinatorial Purged CV (`cpcv.py`)

AFML Ch. 12 — the keystone of our validation gate.

```
                          # enumerate all (N choose k) splits
  timeline ─┬─ group 1 ─┐                    N = 10 groups (default)
            ├─ group 2  │                    k = 2  test groups
            ├─ …        │── C(10, 2) = 45
            └─ group 10 ┘   paths
                           
  per path: train on the 8 non-test groups (purged + embargoed
            around each test group), backtest on test groups,
            stitch contributions into a full-history P&L path
```

`CPCVEngine.generate_paths()` yields 45 train/test splits; `run_backtest_paths`
fits the meta-labeler per path and produces per-path returns. 45 paths
means a real distribution of Sharpes, not a point estimate.

### 13.5 Deflated Sharpe Ratio (`deflated_sharpe.py`)

Bailey & López de Prado (2014). Corrects the single backtest Sharpe for:
- multiple-testing selection bias (we tried many strategies),
- skew and kurtosis of returns (non-normality),
- finite-sample bias.

Returns `(dsr_stat, p_value)`. `compute_dsr_from_cpcv()` uses the
mean Sharpe across CPCV paths as input.

**Gate:** one-tailed `p < 0.05`.

### 13.6 Probability of Backtest Overfit (`pbo.py`)

CSCV — Combinatorially Symmetric Cross-Validation (Bailey et al. 2013):
- S = 10 timeline partitions.
- C(10, 5) = 252 combinations.
- For each combo, compute IS Sharpe ranks; find OOS rank of IS champion.
- PBO = fraction of combos where IS champion underperforms median OOS.

**Gate:** `PBO < 0.40` (design-doc §9.2). PBO > 0.50 means worse than
coin-flip overfit detection.

### 13.7 Gate orchestrator (`gate_orchestrator.py`)

```
   StrategyGate
     │
     ├── .quick_validate(candidate)
     │     DSR only, ~1 second. First-pass sanity check during development.
     │
     └── .validate(candidate)
           ├─ CPCVEngine.run → 45 path returns
           │   Gate 1: ≥60% of paths have positive net returns
           │
           ├─ compute_dsr_from_cpcv → (dsr, p_value)
           │   Gate 2: p_value < 0.05
           │
           └─ compute_pbo → pbo_value
               Gate 3: pbo < 0.40
```

Used by `RetrainPipeline` to gate every promotion. Candidate must pass
**all three gates**; failing one rejects even if the other two pass.

### 13.8 BacktestReport (`report.py`)

Full reporting artifact:
- monthly returns table,
- trade log,
- regime-conditional stats (split at volatility median),
- drawdown table (top 5 episodes),
- strategy-family breakdown,
- CPCV path stats aggregation (median/mean/std of Sharpe, hit rate).

---

## 14. Subsystem 9 — Execution Engine

**Phase:** 5 (paper) + 6 (live) · **Narang layer:** Execution · **Path:** `src/execution/`

### 14.1 Responsibilities

Turn target weights into fills while respecting risk. Handle broker
connectivity, execution algo selection, circuit breakers, TCA, disaster
recovery, and compliance audit.

### 14.2 Data model (`models.py`)

```
  Order           Fill            Position         PortfolioState
  ─────           ────            ────────         ──────────────
  id              order_id        symbol           nav
  symbol          symbol          qty              cash
  side            side            avg_price        positions [Position]
  qty             qty             realized_pnl     daily_pnl
  type            price           market_value     drawdown
  status          timestamp                        gross_exposure
  algo            commission                       net_exposure
  filled_qty      slippage_bps                     position_count
  avg_fill_price  fee
```

Status enum: `NEW → ACCEPTED → (PARTIAL_FILL) → FILLED | REJECTED |
CANCELLED`. Execution-algo enum: `IMMEDIATE | TWAP | VWAP | ICEBERG`.

### 14.3 Broker abstraction

```
             ┌────────────────────────────────┐
             │      BaseBrokerAdapter         │
             │ submit_order / cancel_order /  │
             │ get_positions / get_account /  │
             │ heartbeat / poll_fills         │
             └────────────────────────────────┘
                      ▲         ▲         ▲
                      │         │         │
           ┌──────────┘         │         └───────────┐
           │                    │                      │
     ┌─────────────┐    ┌─────────────┐       ┌─────────────┐
     │ PaperBroker │    │ AlpacaAdapt │       │ CCXTAdapter │
     │ (full sim)  │    │ (equities)  │       │ (crypto,    │
     │             │    │             │       │  100+ venues)│
     └─────────────┘    └─────────────┘       └─────────────┘
                              │
                        ┌─────────────┐
                        │ IBKRAdapter │
                        │ (futures +  │
                        │  equities   │
                        │  backup)    │
                        └─────────────┘
```

**BrokerFactory** (`broker_factory.py`): symbol → asset class → cached
broker singleton. **SmartOrderRouter** (same file): for crypto, queries
top-of-book depth across CCXT venues and splits an order across venues
to minimise weighted slippage.

### 14.4 Live-trading env-var gates

Live trading is **off by default** — three env-vars unlock it:

```
  WANG_ALLOW_LIVE_TRADING=yes   # required for ANY live order
  WANG_ALLOW_LIVE_CRYPTO=yes    # crypto (CCXT) live routes
  WANG_ALLOW_LIVE_FUTURES=yes   # IBKR futures live routes
```

Also relevant:

```
  WANG_SECRETS_BACKEND=env|file|aws|gcp
  WANG_MASTER_KEY=<fernet-key>        # for encrypted-file backend
  WANG_SECRETS_PATH=<path>            # for encrypted-file backend
```

### 14.5 Circuit breakers (`circuit_breakers.py`)

Eight named breakers in two modes:

| # | Breaker | Trigger | Action |
|---|---------|---------|--------|
| 1 | Fat-finger | single order > 5% NAV | block |
| 2 | Daily loss | P&L < −2% NAV | halt new entries; exits only |
| 3 | Drawdown throttle | DD 10% / 15% / 20% | reduce 50% / reduce 75% / HALT+FLATTEN |
| 4 | Model staleness | >30d warn, >60d halt | warn / halt |
| 5 | Connectivity | no heartbeat 60s | pause new orders |
| 6 | Data quality | bar-rate outlier >3σ | pause new orders |
| 7 | Correlation spike | portfolio corr > 0.80 | halt new entries |
| 8 | Dead-man switch | operator check-in >24h | HALT+FLATTEN |

Actions are a typed enum `CircuitBreakerAction`:
`HALT | FLATTEN | REDUCE_SIZE_50 | REDUCE_SIZE_75 | PAUSE | NONE`.

Sync vs async separation: `check_pre_trade()` runs before each order
(fast, blocking). `check_portfolio_health()` runs on a loop (slower,
covers stale/dead-man/correlation). `OperatorCheckin` watches the
sentinel file `.operator_checkin`.

### 14.6 Execution algorithms (`algorithms.py`)

| Algo | How it works | Default config |
|------|-------------|----------------|
| Immediate | limit-at-mid, fallback to market | 30s timeout |
| TWAP | N equal-time slices; carry unfilled qty to next slice | 10 slices over 10 min |
| VWAP | U-shaped intraday volume profile (heavier open/close) | 30 min duration |
| Iceberg | show 10% visible qty; refresh on fill | up to 100 tranches |

**Router** (`select_execution_algo`):

```
  participation = order_qty / ADV

  if participation < 0.001:                 Immediate
  elif participation < 0.01:                TWAP
  else:                                     VWAP

  # crypto override: if order > 0.5% top-of-book depth
  if asset_class == "crypto" and qty / depth > 0.005:
      Iceberg
```

### 14.7 Order manager (`order_manager.py`)

The orchestration hub:

```
  run_cycle(prices_dict)
    │
    ├── circuit_breakers.check_portfolio_health()  → possibly HALT
    │
    ├── check_position_exits(prices)
    │     # triple-barrier exits on existing positions:
    │     #   stop-loss / take-profit / vertical (time) barrier
    │
    ├── execute_target_portfolio(target_weights)
    │     │
    │     ├── diff vs current: trade_list
    │     ├── circuit_breakers.check_pre_trade(trade) for each
    │     ├── transaction_costs.estimate(trade)
    │     ├── if expected_cost > alpha_cost_ratio × expected_alpha: skip
    │     ├── select_execution_algo(trade, urgency)
    │     ├── broker = BrokerFactory.get_broker(trade.symbol)
    │     ├── algo.execute(trade, broker)
    │     └── update PortfolioState from fills
    │
    └── reconcile_positions()
          broker positions ⇆ internal PortfolioState
          orphan-order cancellation on mismatch
```

### 14.8 TCA (`tca.py`)

Post-trade Transaction Cost Analysis:
- Arrival-price slippage, market impact, timing cost.
- Benchmarks: TWAP, VWAP.
- **Degradation detection:** fires if slippage > 2σ of historical mean,
  or fill rate drops >10pp, or TWAP underperformance >5bps sustained.

### 14.9 Preflight (`preflight.py`)

**18 blocker checks** across 6 categories (plus 3 warnings). Must all
pass before `make live-start`.

| Category | Example checks |
|----------|---------------|
| Broker connectivity | heartbeat ok; buying power sufficient; account permissions |
| Model readiness | model age <30d; meta-labeler loaded; regime model loaded |
| Paper-trading proof | 8+ weeks history; Sharpe >1.0; MaxDD <15%; win rate >50% |
| Infrastructure | DB reachable; feature store accessible; logs writable |
| Risk limits | risk-budget config loaded; crypto cap sane; gross cap ≤150% |
| Operator | check-in fresh; risk acknowledgment recent |

Blockers are hard stops. Warnings are logged but don't prevent start.

### 14.10 Capital deployment (`capital_deployment.py`)

Four-phase ramp with automatic divergence halt:

```
   ┌──────────┬─────────┬──────────┬──────────┐
   │  Pilot   │  Beta   │  Scale   │   Full   │
   ├──────────┼─────────┼──────────┼──────────┤
   │  $5K cap │  $15K   │  $50K    │  ∞       │
   │  0.25×   │  0.50×  │  0.75×   │  1.00×   │
   │  size    │  size   │  size    │  size    │
   │          │         │          │          │
   │  ≥14d    │  ≥28d   │  ≥42d    │  —       │
   │  promo   │  promo  │  promo   │          │
   │  gate    │  gate   │  gate    │          │
   └──────────┴─────────┴──────────┴──────────┘
```

Promotion criteria (per phase):
- minimum duration reached,
- live Sharpe ≥ 1.0,
- max drawdown < 15%,
- win rate > 50%.

**Divergence check** (daily): 7-day rolling Sharpe divergence between paper
and live. If |ΔSharpe| > 1.0, auto-halt; operator must investigate
before resume.

### 14.11 Disaster recovery (`disaster_recovery.py`)

- **Snapshots**: portfolio state + open orders + phase + model version +
  metrics, pickled with SHA-256 checksum every 50 bars. 30-day
  retention.
- **Sentinel files**:
  - `.live_crash` — written by signal handler on unclean exit;
    presence blocks restart until operator clears.
  - `.live_halt` — operator-written graceful-stop sentinel.
- **RecoveryManager**: reconciles internal state vs broker state on
  restart; cancels orphan orders; rehydrates PortfolioState from
  checksummed snapshot (or fallback to broker reconciliation).

### 14.12 Compliance audit log (`audit_log.py`)

HMAC-SHA256 **chained** audit trail:
- Each entry signed with HMAC(prev_signature || serialize(entry));
- Tamper-evident — `verify_chain()` catches any mutation.
- EventType enum covers: SIGNAL, ORDER_SUBMIT, ORDER_FILL, ORDER_REJECT,
  POSITION_UPDATE, CIRCUIT_BREAKER_TRIP, PHASE_PROMOTION, RL_PROMOTION,
  OPERATOR_ACTION, STARTUP, SHUTDOWN, RECOVERY.
- Persistent backends: TimescaleDB hypertable (primary) + JSON file
  fallback.
- 7-year retention — matches regulatory norms.

### 14.13 Secrets management (`src/config/secrets.py`, `scripts/setup_secrets.py`)

Pluggable backend via `WANG_SECRETS_BACKEND`:

| Backend | Use case | Notes |
|---------|----------|-------|
| `env` | dev / CI | simplest |
| `file` | single-host prod | Fernet-encrypted, master key in env |
| `aws` | multi-host AWS | Secrets Manager, 5-min TTL cache |
| `gcp` | GCP | Secret Manager, 5-min TTL cache |

`scripts/setup_secrets.py` is the bootstrap CLI: generate Fernet key,
rotate master key, interactive seed for Alpaca / Binance / Coinbase /
IBKR / Telegram keys.

### 14.14 Daily ops (`daily_ops.py`)

`DailyReconciliation` — end-of-day routine:
1. reconcile positions (internal ⇆ broker),
2. compute daily P&L,
3. roll up TCA,
4. run drift detector,
5. persist daily snapshot,
6. generate human-readable daily report (`generate_daily_report`),
7. alert on any discrepancy.

---

## 15. Subsystem 10 — Monitoring

**Phase:** 5 · **Narang layer:** Execution (crosscutting) · **Path:** `src/monitoring/`

### 15.1 Responsibilities

Observe everything, alert on anomalies, detect data/feature drift,
present a single pane of glass (Grafana).

### 15.2 Prometheus metrics (`metrics.py`)

**15 metrics grouped by domain**:

| Domain | Metrics |
|--------|---------|
| Portfolio (5) | NAV, drawdown, daily P&L, gross exposure, net exposure |
| Positions (1) | position count |
| Orders (3) | orders submitted / filled / rejected (counters) |
| Signals (1) | signal count per family (gauge) |
| ML (1) | meta-label probability histogram |
| Execution (1) | slippage bps histogram |
| Data (2) | bar formation rate, data gap per symbol |
| Features (1) | KL drift per feature |
| Model (1) | model age (hours) |
| Breakers (1) | circuit-breaker triggers per type |

Exposed on HTTP `:9090/metrics`. Histogram buckets tuned: slippage
0.1–500bps (11 buckets); meta-label prob 0–1 in 0.1 steps.

### 15.3 Alerting (`alerting.py`)

Tiered `AlertManager`:
- `LogChannel` (always on)
- `TelegramChannel` (rate-limited 5s per send)

Severity: `INFO | WARNING | CRITICAL | EMERGENCY`.

**8 templated alerts:**

| Alert | Severity logic |
|-------|----------------|
| Drawdown | ≥10% critical; ≥5% warning |
| Daily loss | P&L ≤ −2% critical |
| Circuit breaker | always critical |
| Model stale | >24h warn; >60min… (see below) |
| Data gap | >300s warning |
| Execution failure | critical |
| Position reconciliation | warning |
| Feature drift | KL >1.0 critical; >0.5 warning |

Duplicate suppression via `(source, title)` key with 300s cooldown;
persisted to JSON for cross-process durability.

### 15.4 Feature drift detector (`drift_detector.py`)

Four statistics per feature vs training baseline:

| Statistic | Threshold |
|-----------|-----------|
| KL divergence | >0.5 (50-bin histogram) |
| KS test (p-value) | <0.01 |
| Mean shift (z-score) | >3.0 |
| Variance ratio | >2.0 or <0.5 |

A feature is "drifted" if **any** statistic exceeds its threshold.
`recommend_action()`:

```
  frac_drifted < 20%     → monitor
  20% ≤ x < 50%          → warning + retrain soon
  ≥ 50%                  → critical + immediate retrain
```

### 15.5 Grafana dashboards (`dashboards.py`)

`generate_main_dashboard()` produces a 6-row dashboard JSON with
17+ panels:

| Row | Panels |
|-----|--------|
| 1 Portfolio Overview | NAV time-series, drawdown, daily P&L, exposure gauge |
| 2 Positions | count, top holdings, per-symbol P&L |
| 3 Signals & Model | signals/family stacked, meta-label distribution, model age |
| 4 Execution | slippage histogram, TCA rollup, fill rate |
| 5 Data Health | bar-rate, data gap, feed status |
| 6 Risk | breaker triggers, KL drift heatmap, correlation matrix |

`generate_alerting_rules()` produces 6 Grafana rule definitions
(drawdown warn/crit, daily loss, model stale, data gap, feature drift).

Refresh: 30s. Wired into `docker-compose.yaml` with Prometheus +
Grafana containers.

---

## 16. Validation gate: the three tests

The validation gate is the load-bearing safety of this entire system.
Every candidate model (meta-labeler or otherwise) must clear all three
before promotion to live.

### 16.1 Why three and not one?

Each test corrects for a different failure mode. No single test is
sufficient:

| Test | What it catches | What it misses |
|------|----------------|----------------|
| CPCV (≥60%) | single-path luck; a strategy with fragile Sharpe on one split | selection bias across many strategies tried |
| Deflated Sharpe (p<0.05) | selection bias; non-normal returns | can be passed by a genuinely overfit model if CPCV happened to be lucky |
| PBO (<40%) | overfit probability directly via combinatorial ranking | says nothing about absolute performance |

All three in series = high confidence that what we're promoting actually
has edge.

### 16.2 Gate diagram

```
  candidate ─► CPCV (45 paths) ─► are ≥60% positive? ─No──► REJECT
                     │
                     │ Yes
                     ▼
                 DSR computed on paths
                     │
                     │ p < 0.05 ? ─No──► REJECT
                     │
                     │ Yes
                     ▼
                 PBO computed (CSCV)
                     │
                     │ < 0.40 ? ─No──► REJECT
                     │
                     │ Yes
                     ▼
                 improvement vs incumbent > 5% ? ─No──► REJECT
                     │
                     │ Yes
                     ▼
                 PROMOTE (log to MLflow + audit log)
```

### 16.3 Fast path

`StrategyGate.quick_validate()` runs DSR only (~1s). Used during
development to fail fast; never the authoritative promotion gate.

---

## 17. Data model

Full schema reference lives in [`data_model.md`](data_model.md). Summary:

### 17.1 TimescaleDB hypertables (primary)

| Table | Rows per day (typical) | Retention | Owner |
|-------|-----------------------|-----------|-------|
| `raw_ticks` | ~millions | 90 days | Data Engine |
| `bars` | ~10–50K | 7 years | Data Engine |
| `cusum_events` | ~100s | 7 years | Data Engine |
| `features` | = bars | 7 years | Feature Factory |
| `signals` | ~100–1000 | 7 years | Signal Battery |
| `labels` | = signals | 7 years | Labeling |
| `meta_labels` | = signals | 7 years | ML Layer |
| `orders` | ~10–100 | 7 years | Execution |
| `fills` | ~20–500 | 7 years | Execution |
| `tca_results` | = orders | 7 years | Execution |
| `portfolio_snapshots` | ~1/bar | 7 years | Execution |
| `audit_log` (HMAC-chained) | ~100–1000 | 7 years | Execution |

### 17.2 Other persistence

| Artifact | Store | Retention |
|----------|-------|-----------|
| Feature matrices (training) | Parquet (local or GCS), partitioned by symbol × date | 7 years (DVC optional) |
| Model runs | MLflow (SQLite + filesystem local; Postgres + S3 prod) | 7 years |
| Snapshots (state) | `logs/snapshots/*.pkl` (SHA-256 checksum) | 30 days |
| App logs | `/var/log/wang_trading/` | 180 days |
| Compliance log copy | `logs/live_trading_compliance.log` | 7 years (file) |

### 17.3 Data flow summary

```
  ticks ─► bars ─► features ─► signals ─► (labels, meta-labels)
                                                      │
                                                      ▼
                                      target weights → orders → fills
                                                      │
                                                      ▼
                                       portfolio_snapshots + tca_results
                                                      │
                                                      ▼
                                      everything materially decision-
                                      related → audit_log (HMAC chain)
```

---

## 18. Technology stack and rationale

Full dependency list lives in [`technology_stack.md`](technology_stack.md).
Summary of why each piece.

### 18.1 Core language + runtime

| Tech | Rationale |
|------|-----------|
| Python 3.11+ | ecosystem dominance for quant; match-case, faster interpreter |
| `numba` (hot paths) | JIT for bar constructors and CUSUM — 10–100× vs pure Python |
| `asyncio` | concurrency model for broker adapters; non-blocking SOR |
| `dataclasses` | typed models (Order, Fill, Position) without ORM overhead |

### 18.2 Data + storage

| Tech | Rationale |
|------|-----------|
| TimescaleDB | hypertables are purpose-built for time-series; SQL native |
| SQLAlchemy | portable ORM; avoids TimescaleDB lock-in |
| Parquet + `pyarrow` | columnar training-set format; plays well with Spark/DVC |
| DVC (optional) | versioned training data — reproducibility |
| GCS (optional) | long-term warehouse + cross-VM access |

### 18.3 Market data / brokers

| Tech | Rationale |
|------|-----------|
| `alpaca-py` / `alpaca-trade-api` | US equities primary (low fees, good API) |
| `ccxt` | 100+ crypto exchanges behind one API |
| `ib_insync` | IBKR for futures + equities backup |
| `aiohttp` | WebSocket + REST for custom feeds (Polygon, Glassnode) |

### 18.4 ML + DL

| Tech | Rationale |
|------|-----------|
| `lightgbm` (default) / `xgboost` / `sklearn` | meta-labeler choice — LightGBM wins on speed + accuracy for our feature count |
| `torch` | LSTM regime model, autoencoder, RL nets |
| `stable-baselines3` | PPO implementation; battle-tested |
| `gymnasium` | RL env standard |
| `transformers` (HuggingFace) | FinBERT for sentiment |
| `hmmlearn` | HMM for regime labelling |
| `optuna` | TPE hyperparameter search + MedianPruner |
| `mlflow` | model registry + run tracking |
| `shap` | feature attribution |

### 18.5 Stats + math

| Tech | Rationale |
|------|-----------|
| `statsmodels` | ADF, Johansen cointegration, Ljung-Box |
| `arch` | GARCH(1,1) fit |
| `scipy.stats` | KS test, normal CDF for DSR |
| `scipy.cluster.hierarchy` | HRP linkage |

### 18.6 Observability + ops

| Tech | Rationale |
|------|-----------|
| `prometheus_client` | metrics surface |
| Prometheus server | scrape + store |
| Grafana | dashboards + alerting rules |
| Telegram Bot API | mobile-first alert channel |
| `loguru` | structured logging |
| `cryptography.fernet` | encrypted-file secrets backend |
| `boto3` / `google-cloud-secret-manager` | cloud secrets |

### 18.7 Deploy + control

| Tech | Rationale |
|------|-----------|
| Docker + docker-compose | Prometheus + Grafana + dev convenience |
| systemd | single-host live service |
| supervisor | multi-process live (data + exec + monitoring + retrain) |
| `scripts/deploy.sh` | first-time install on Linux host |
| Makefile | operator entry points (`make preflight`, `make live-start`, `make recover`) |

### 18.8 Why these choices in one sentence each

- **TimescaleDB over InfluxDB** — SQL semantics for joins between bars, features, and orders; native `time_bucket`.
- **LightGBM over XGBoost default** — categorical support and equal accuracy on our data sizes.
- **HRP over Markowitz** — MVO explodes on noisy covariance; HRP is stable.
- **PPO over DDPG/SAC** — stable, fewer knobs, well-tested in `stable_baselines3`.
- **Alpaca + IBKR over one broker** — redundancy; IBKR is the backup.
- **Prometheus + Grafana over SaaS APM** — no per-metric cost; local first.
- **Fernet-encrypted file before cloud KMS** — works offline for dev; swap backend for prod without touching code.

---

## 19. Operational architecture

### 19.1 Lifecycle states

```
  ┌──────────────┐   preflight passes     ┌──────────────┐
  │  STOPPED     │ ─────────────────────► │  RUNNING     │
  │ (halt file)  │ ◄───────────────────── │  (phase N)   │
  └──────────────┘  make live-stop        └──────────────┘
        ▲                                         │
        │                                         │ crash / SIGTERM
        │                                         ▼
        │                                  ┌──────────────┐
        │   make recover                   │   CRASHED    │
        └──────────────────────────────────│ (.live_crash)│
                                           └──────────────┘
```

### 19.2 Operator runbooks

All runbooks live in `docs/runbooks/`:

| Runbook | Opens… |
|---------|--------|
| [Daily operations](runbooks/daily_operations.md) | Every trading day |
| [Incident response](runbooks/incident_response.md) | SEV1–SEV4 triage |
| [Deployment](runbooks/deployment.md) | Ships + rollbacks |
| [Model operations](runbooks/model_operations.md) | Retrain, promotion, RL |
| [Capital management](runbooks/capital_management.md) | Adding / withdrawing funds |
| [Compliance](runbooks/compliance.md) | Tax, audit export, retention |
| [Go-live checklist](go_live_checklist.md) | Before first live dollar |

### 19.3 Failure-domain matrix

| Failure | Detection | Automatic action | Operator action |
|---------|-----------|------------------|-----------------|
| Broker disconnect | heartbeat | CB pauses new orders | investigate; preflight before restart |
| Drawdown > 5% | CB warn | alert | watch; post-mortem if continues |
| Drawdown > 15% | CB critical | REDUCE_SIZE_75 | investigate before un-throttling |
| Drawdown > 20% | CB halt | HALT_AND_FLATTEN | post-mortem + gate review |
| Paper/live Sharpe divergence > 1.0 | daily check | auto-halt via CapitalDeploymentController | investigate; resume requires operator approval |
| RL > 5% DD within 3 days of promotion | per-cycle check | auto-revert to HRP | investigate before re-approving |
| Process crash | signal handler | snapshot + `.live_crash` | `make recover` |
| Audit chain broken | `verify_chain()` | none (alert only) | freeze; investigate tamper |
| Data gap > 300s | monitoring | alert; may trigger data-quality CB | contact provider; check feed |
| Feature drift > 50% | drift detector | trigger immediate retrain | review retrain; manual override if needed |

---

## 20. Deployment topology

### 20.1 Single-host (default live)

```
 ┌─────────────────────────── Linux host (cloud VM / on-prem) ─────────────────┐
 │                                                                              │
 │   systemd unit:  wang-live-trading.service                                   │
 │       │                                                                       │
 │       ├── supervisor:  data_ingestion.conf     ─► ingestion workers          │
 │       ├── supervisor:  live_trading.conf       ─► LiveTradingPipeline        │
 │       ├── supervisor:  monitoring.conf         ─► MetricsCollector + drift   │
 │       └── supervisor:  retrain_scheduler.conf  ─► weekly retrain             │
 │                                                                               │
 │   Docker:                                                                     │
 │       ├── prometheus  (scrapes :9090)                                         │
 │       └── grafana     (localhost:3000)                                        │
 │                                                                               │
 │   Databases:                                                                  │
 │       ├── TimescaleDB  (local or remote managed Postgres)                     │
 │       ├── MLflow        (SQLite + local FS; swap to Postgres + S3 for prod)   │
 │       └── FeatureStore  (Parquet local or GCS)                                │
 │                                                                               │
 │   State:                                                                      │
 │       ├── /opt/wang_trading/   — code + venv                                  │
 │       ├── /var/log/wang_trading/ — app logs (logrotate)                       │
 │       └── /opt/wang_trading/logs/snapshots/ — pickled state (30d)             │
 │                                                                               │
 └──────────────────────────────────────────────────────────────────────────────┘
                   │                                    │
                   ▼                                    ▼
             Broker APIs                          Telegram (alerts)
         Alpaca / CCXT / IBKR
```

### 20.2 Cloud (aspirational)

Per the design-doc tech stack:
- Cloud Run for ingestion / execution / monitoring (24/7 services).
- GPU VM on-demand for research (retrain, hyperparameter search,
  RL training). Provisioned, used, torn down.

This is not required — the single-host topology is fully supported.

### 20.3 First-time install

```bash
sudo ./scripts/deploy.sh
# creates user, venv, systemd unit, supervisor configs, logrotate
```

See [`deployment.md`](deployment.md) for full walkthrough.

---

## 21. Security and risk posture

### 21.1 Threat model (short version)

We defend against: (a) accidental operator mistakes (the most frequent
threat), (b) silent data corruption, (c) runaway automation, (d) credential
exposure. We do **not** defend against a determined attacker with
root on the host; assume host compromise means game over.

### 21.2 Defence layers

| Layer | Mechanism |
|-------|-----------|
| Process | three env-var gates for live (`WANG_ALLOW_LIVE_*`) |
| Pre-trade | 18 preflight blockers (hard stop at start) |
| Real-time | 8 circuit breakers (hard stop during run) |
| Operator | dead-man switch, check-in sentinel, HALT sentinel |
| Credentials | pluggable secrets (env / Fernet / AWS / GCP); 5-min cache |
| Integrity | HMAC-chained audit log; `verify_chain()` check |
| State | SHA-256 checksummed snapshots; `RecoveryManager` |
| Divergence | daily paper/live Sharpe check; auto-halt |
| RL safety | shadow for 6 months; 3-day / 5%-DD auto-revert |

### 21.3 Secrets

- Never committed.
- Dev: `env`.
- Single-host prod: `file` (Fernet + master key in env).
- Multi-host prod: `aws` or `gcp` with 5-min TTL cache.
- Rotation: `scripts/setup_secrets.py --rotate-master-key`.

### 21.4 Compliance posture

- 7-year audit log retention.
- HMAC chain ensures tamper-evidence.
- `compliance.md` runbook covers tax export, audit-log export, retention.

---

## 22. Phase roadmap

```
  Phase 1 (Wk 1–4)    Data Foundation                ✅ shipped
     ├─ TimescaleDB + ingestion
     ├─ All bar types + ETF trick
     └─ Feature store

  Phase 2 (Wk 5–10)   Feature Factory + Signal       ✅ shipped
     ├─ FFD, structural breaks, entropy
     ├─ Microstructure, GARCH, NLP
     └─ 7 signal families + orchestrator

  Phase 3 (Wk 11–14)  Labeling + Core ML + Sizing    ✅ shipped
     ├─ Triple-barrier + sample weights
     ├─ LightGBM meta-labeler + calibration
     ├─ Purged CV + Optuna tuning
     └─ 5-layer bet sizing cascade

  Phase 4 (Wk 15–18)  Backtesting + Portfolio         ✅ shipped
     ├─ Walk-forward + realistic costs
     ├─ CPCV (45-path) / DSR / PBO gate
     └─ HRP default, PCA factor, RP, multi-strat

  Phase 5 (Wk 19–24)  Execution + Paper + Monitoring  ✅ shipped
     ├─ OrderManager + 8 circuit breakers
     ├─ Imm / TWAP / VWAP / Iceberg + router
     ├─ PaperTradingPipeline + retrain scheduler
     └─ Prometheus + Grafana + drift

  Phase 6 (Wk 25–32+) Live Capital + RL + Hardening   ✅ shipped
     ├─ Live adapters (Alpaca / CCXT / IBKR)
     ├─ BrokerFactory + SmartOrderRouter
     ├─ 18-check preflight
     ├─ 4-phase graduated deployment
     ├─ LiveTradingPipeline w/ HALT + HMAC audit
     ├─ RL (Gymnasium + PPO) + ShadowEngine + auto-revert
     ├─ Disaster recovery (checksums, signals, reconcile)
     └─ Secrets (env/file/AWS/GCP) + deploy.sh + runbooks
```

All six phases are merged to master. Paper-trading history is being
accumulated; live deployment begins after the go-live checklist is
complete.

---

## 23. Testing strategy

### 23.1 Test layers

| Layer | Count | Examples |
|-------|-------|----------|
| Unit tests | 1079+ | per-module correctness (one `test_<module>.py` per module) |
| Integration tests | 28+ | end-to-end phase tests (`test_phase{2,3,4,5}_integration.py`, `test_rl_integration.py`) |
| Benchmarks | ~15 | `make bench`, `make bench-backtest` — perf budgets per subsystem |
| Smoke tests | 2 | `scripts/smoke_test.py` (< 10s), `scripts/production_smoke_test.py` (< 10s) |
| Design-doc audit | 1 | `scripts/design_doc_audit.py` + `test_design_doc_audit.py` — enforces design-spec conformance |

### 23.2 What each phase's integration test covers

```
  test_phase2_integration.py  bars → features → signals → signals_df
  test_phase3_integration.py  signals → labels → meta-labeler → bet size
  test_phase4_integration.py  candidate model → CPCV/DSR/PBO gate → promote
  test_phase5_integration.py  100-cycle PaperTradingPipeline end-to-end
  test_rl_integration.py      shadow loop + promotion controller
```

### 23.3 Makefile entry points

```
  make test                 # all unit tests (no integration)
  make test-integration     # phase integration tests
  make bench                # phase 2 perf benchmarks
  make bench-backtest       # phase 4 perf benchmarks
  make smoke-test           # full pipeline in < 10s
  make preflight            # 18-check live readiness
  make live-start/stop/flatten
  make recover
  make retrain / retrain-dry-run
```

### 23.4 CI posture

- Unit tests on every PR.
- Integration tests nightly (they take longer).
- Smoke test on every merge to master.
- Design-doc audit blocks merges that drift from spec.

---

## 24. Future work

### 24.1 Known gaps (open work)

- **Options** — currently equities / crypto / futures only. Options
  would need: surface features, Greeks, a new labeling regime,
  and exchange adapters (IBKR has options, but we have not wired it).
- **Full cloud topology** — Cloud Run + GPU VM on-demand is designed
  but not implemented. Current deployment is single-host.
- **Alternative data beyond Glassnode + FinBERT** — satellite imagery,
  credit-card data, web scraping are not integrated.
- **Options volatility surface** in `feature_factory/volatility.py` —
  currently just GARCH + realized.
- **Multi-operator workflows** — system assumes a single operator.
  Multi-operator would require role-based gates on the compliance log.

### 24.2 Research tracks

- **Transformer regime detector** as alternative to LSTM — Jansen
  notes transformers outperform on longer sequences.
- **Reinforcement-learning sizing** (a Tier-4 layer feeding the
  bet-sizing cascade).
- **Cross-asset stat-arb** (commodities ⇆ equities) via the existing
  pairs engine.
- **Alt microstructure features** — hidden liquidity estimators,
  queue-position models (for HFT-ish execution).

### 24.3 Operational improvements

- **Automated hot-standby** — second host tracking snapshots, ready to
  promote on primary-host failure.
- **Web-based operator console** — today operations are CLI-only.
- **Richer TCA** — implementation shortfall decomposition, benchmark
  families beyond TWAP/VWAP.

---

## Appendix A — Configuration surface

Primary config files (all start from `.example.yaml` templates):

| File | Purpose |
|------|---------|
| `config/settings.yaml` | DB, brokers, feature-store paths, global flags |
| `config/paper_trading.yaml` | PaperTradingPipeline config (`PipelineConfig`) |
| `config/live_trading.yaml` | LiveTradingPipeline config + deployment phase |
| `config/futures_contracts.yaml` | IBKR contract specs (point value, tick size, expiry) |
| `config/prometheus.yml` | Prometheus scrape targets |
| `config/supervisor/*.conf` | supervisord program definitions |
| `config/systemd/wang-live-trading.service` | systemd unit |

Env vars (operational):

| Var | Default | Purpose |
|-----|---------|---------|
| `WANG_ALLOW_LIVE_TRADING` | unset | hard gate — required for any live order |
| `WANG_ALLOW_LIVE_CRYPTO` | unset | unlocks CCXT live routes |
| `WANG_ALLOW_LIVE_FUTURES` | unset | unlocks IBKR live futures routes |
| `WANG_SECRETS_BACKEND` | `env` | env / file / aws / gcp |
| `WANG_MASTER_KEY` | unset | Fernet key (file backend) |
| `WANG_SECRETS_PATH` | unset | secrets file path (file backend) |

---

## Appendix B — Reading order

For someone new to the codebase:

1. This document (top-to-bottom).
2. `architecture_overview.md` — 1-page visual refresher.
3. `phase2_features.md`, `phase2_signals.md` — alpha side.
4. `phase3_ml_pipeline.md` — ML + sizing.
5. `phase4_backtesting.md`, `phase4_portfolio.md` — validation + allocation.
6. `phase5_execution.md` — paper trading + monitoring.
7. `go_live_checklist.md` + `runbooks/` — before touching live.
8. `data_model.md` + `technology_stack.md` — reference material.
