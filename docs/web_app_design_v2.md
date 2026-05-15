# Wang Trading Web App Design — v2

Status: design proposal
Version: 2.0 (supersedes [docs/web_app_design.md](web_app_design.md))
Audience: product, quant engineering, execution engineering, frontend engineering, operations
Scope: real web application that exposes every model and pipeline output, optimised for an institutional operator cockpit
Out of scope: implementation PR, deployment migration, broker credential handling

## 0. What Changed Since v1

v1 captured the right backbone (decision-chain explainability, operator-grade density, phased rollout). v2 makes three changes:

1. **Folds in every model and pipeline output.** v1 covered most of the ~200 fields the model and pipeline emit; v2 surfaces the rest — per-row SHAP, regime probabilities, RL shadow comparison, per-family signal metadata, calibration history, expected pre-trade cost, source-level staleness, sizing constraint provenance, per-bar microstructure, structural-break statistics, on-chain flows, sentiment article counts, labeling barriers, and Prometheus stage metrics.
2. **Adds four new pages.** Scenarios & Stress Test, Preflight & Go-Live, Track Record, Replay / Time Travel. Each maps to an existing pipeline capability that v1 under-exposed.
3. **Adds five new UX patterns.** Diff-since-last-refresh rail, command palette (`Cmd+K`), edge-vs-size scatter on the Command Center, pre-trade portfolio simulator on Portfolio & Risk, natural-language explanation per idea on Trade Ideas.

Page count: 11 → 15. All v1 pages are retained. v1 section numbering is retained where possible to keep cross-references valid.

## 1. Executive Summary

The current local Trade Ideas UI proves the live stack can produce a read-only symbol-level decision table. The real web app should be an operator-grade trading cockpit that exposes the full decision chain end-to-end:

```text
Ticks -> Bars -> Features -> Signals -> Regime -> Meta Probability -> Calibrated Probability -> Bet Size (5-layer cascade) -> Target Weight -> Action -> Pre-Trade Cost -> Risk/Execution State -> Outcome -> Track Record
```

At all times the app must answer five questions, not four:

1. What should I do now?
2. Why is the system saying this? (Bars → Features → Signals → Meta → Sizing → Target)
3. Can I trust it? (Calibration, drift, regime fit, RL shadow, gate history)
4. What happened after acting? (TCA, reconciliation, P&L)
5. **How well has the system done at this kind of call before?** (Track record — new in v2)

The application should be dense, fast, auditable, and calm. It should feel closer to an internal hedge-fund risk terminal than a SaaS marketing dashboard, but with explainability and counterfactual tooling that real terminals lack.

## 2. Product Goals

### 2.1 Primary Goals

- Show current trade ideas with **full** context — every input, every intermediate value, every constraint that bound the final size, every expected cost.
- Explain every action through a traceable chain.
- Surface trust state: calibration, drift, regime fit, model readiness, broker health, breakers, RL shadow divergence.
- Provide portfolio, execution, backtest, model, signal, monitoring, audit, scenarios, preflight, track-record, and replay views from a single web application.
- Keep live actions gated until explicitly authorised, with typed confirmation.
- Make all critical state visible without hover.
- **(v2)** Surface what changed since the last refresh, not just the current state.
- **(v2)** Allow what-if simulation: pre-trade portfolio impact, parametric stress, historical replay.

### 2.2 Non-Goals

(unchanged from v1)

## 3. Users And Operating Context

### 3.1 Personas

| Persona | Needs | Primary Screens |
|---|---|---|
| Daily operator | Decide what needs attention now, confirm health, review ideas | Command Center, Trade Ideas, Monitoring |
| Quant researcher | Understand signal/model behavior, calibration, drift, RL shadow | Signals, Model & Features, Backtests, Track Record |
| Risk manager | Monitor exposure, concentration, drawdown, breakers, scenarios | Portfolio & Risk, Monitoring, Scenarios |
| Execution engineer | Routing, fills, slippage, reconciliation | Execution & TCA, Audit |
| Compliance reviewer | Reconstruct decisions, verify audit chain | Audit & Compliance, Replay |
| **(v2) Go-live operator** | Run preflight, verify readiness for live | Preflight & Go-Live, Monitoring |

### 3.2 Usage Modes
(unchanged from v1)

## 4. Design Principles

### 4.1 Operational Density
(unchanged)

### 4.2 Explainability First

Every trade idea must expose, at minimum:

- latest price, bar time, bar type, bars loaded, feature rows
- per-family signal metadata (z-scores, half-lives, hedge ratios, ADF p-values, carry, VRP percentile, etc.)
- meta probability and calibrated probability
- **per-row SHAP top contributors** (v2 addition)
- regime probabilities and which regime the model is conditioned on
- full bet-sizing cascade with each layer's value (`afml_size → kelly_capped → vol_adjusted → atr_capped → final_size`)
- `constraints_applied` list — which caps actually bound the size
- target weight, target notional, estimated quantity
- expected execution cost in bps (from rolling TCA)
- action and natural-language reason
- warnings/errors
- stage latencies for each of `data_fetch / feature_compute / signal_generation / meta_inference / sizing / target_generation / order_routing`

### 4.3 Trust State Is Always Visible

The global status bar must always show:

- data freshness (oldest source among bars/features/signals/onchain/sentiment)
- model age and source
- regime label and confidence (v2 addition)
- calibration recency (v2 addition)
- drift severity (v2 addition)
- broker heartbeat
- circuit breaker state
- paper/live divergence
- active alert counts
- last generated timestamp

### 4.4 Safety By Default
(unchanged)

### 4.5 Progressive Disclosure
(unchanged)

### 4.6 Show The Diff, Not Just The State (v2)

Operators do not re-read the entire table every refresh. They look for what changed. Every page that lists current state must offer a diff toggle that shows what is new, what flipped sides, what changed weight by more than a threshold, what error states appeared, and what error states cleared, since the previous refresh.

### 4.7 Show Calibration History, Not Just Current Prediction (v2)

A probability is only as useful as its calibration. Every page that displays `meta_prob` or `calibrated_prob` must offer a one-click drill-down to a reliability plot binned over the trailing N days.

## 5. Recommended Technical Architecture

### 5.1 Frontend
(unchanged from v1)

### 5.2 Backend
(unchanged)

### 5.3 High-Level System

```text
Browser
  -> React application
  -> API client
  -> FastAPI BFF
  -> src.ui.trade_ideas
  -> src.bootstrap
  -> src.data_engine.storage.database
  -> src.data_engine.storage.feature_store
  -> src.signal_battery.orchestrator
  -> src.ml_layer.meta_labeler
  -> src.ml_layer.regime_detector
  -> src.ml_layer.feature_importance
  -> src.ml_layer.rl_shadow
  -> src.bet_sizing.cascade
  -> src.portfolio.multi_strategy
  -> src.portfolio.factor_risk
  -> src.execution.storage
  -> src.execution.tca
  -> src.execution.preflight
  -> src.execution.audit_log
  -> src.monitoring.metrics
  -> src.monitoring.drift_detector
  -> src.monitoring.alerting
  -> src.backtesting.report
  -> src.backtesting.gate_orchestrator
  -> TimescaleDB / Parquet feature store / MLflow / Prometheus / logs
```

### 5.4 Realtime Strategy
(unchanged)

## 6. Data Exposure Policy

### 6.1 Expose To Client

In addition to v1's list:

- regime label and four regime probabilities
- per-row SHAP top contributors (top 5–10)
- per-family signal metadata payloads
- bet-sizing cascade with each layer value and `constraints_applied`
- RL shadow records and aggregate comparison stats
- calibration reliability buckets
- per-source freshness vector (bars/features/signals/onchain/sentiment/factor model)
- structural-break statistics (CUSUM, SADF, GSADF)
- microstructure features (Kyle's lambda, Amihud, Roll, VPIN, Hasbrouck, OFI)
- on-chain flow metrics (per supported asset)
- sentiment scores and article counts per symbol
- triple-barrier labeling outputs for historical events
- expected execution cost (bps) from rolling TCA per symbol-side-algo
- Prometheus stage-latency histogram quantiles
- promotion gate history (CPCV / DSR / PBO over time)
- preflight check pass/fail vector

### 6.2 Do Not Expose To Client
(unchanged from v1)

### 6.3 Sanitization Rules
(unchanged from v1, plus:)

- Per-row SHAP must be capped to top 10 contributors and sent with feature names from `meta_labeler.feature_names_`, never raw column slices.
- Per-family signal metadata must be returned as a dict and never as raw pickled bytes.
- Probabilities must always be returned with `model_version` so the client can compare against the registry.

## 7. Global Application Shell

### 7.1 Layout

```text
+----------------------------------------------------------------------------------+
| Global Status Bar: NAV | DPNL | DD | Gross | Net | Model | Data | Regime |        |
|                   Drift | Cal | Broker | Breakers | Alerts | Last refresh        |
+----------------------------------------------------------------------------------+
| Diff Rail: new BUY: NVDA | flipped: META | weight +80bps: AAPL | err+: TSLA       |
+--------------+-------------------------------------------------------------------+
| Sidebar      | Page content                                                       |
|              |                                                                    |
| Command      | Page toolbar (filters, refresh, export, save view, copy link)      |
| Trade Ideas  |                                                                    |
| Symbol       | Primary panels/tables/charts                                       |
| Portfolio    |                                                                    |
| Execution    |                                                                    |
| Signals      |                                                                    |
| Model        |                                                                    |
| Backtests    |                                                                    |
| Scenarios    |   <-- v2                                                           |
| Track Rec.   |   <-- v2                                                           |
| Replay       |   <-- v2                                                           |
| Preflight    |   <-- v2                                                           |
| Monitoring   |                                                                    |
| Audit        |                                                                    |
| Settings     |                                                                    |
+--------------+-------------------------------------------------------------------+
```

### 7.2 Left Sidebar

Width: 220 px desktop, collapsible to icons at 64 px. Mobile: drawer navigation.

Navigation items (v2 order):

- Command Center
- Trade Ideas
- Symbol Detail
- Portfolio & Risk
- Execution & TCA
- Signals
- Model & Features
- Backtests & Research
- **Scenarios & Stress** *(new)*
- **Track Record** *(new)*
- **Replay** *(new)*
- **Preflight & Go-Live** *(new)*
- Monitoring & Alerts
- Audit & Compliance
- Settings

Each item shows a status marker (red / amber / green / gray) plus an optional counter badge for unread items (alerts, new ideas, drift events).

### 7.3 Top Global Status Bar

Height: 48 to 56 px. Sticky. Always visible.

Fields, in priority order:

- NAV
- Daily P&L (signed, color-coded)
- Drawdown
- Gross exposure
- Net exposure
- Model source/age
- Data freshness (worst-of source vector)
- **Regime label + dominant probability** *(new)*
- **Drift severity badge** *(new — number of features past KL threshold)*
- **Calibration recency** *(new — days since last calibration refit)*
- Broker heartbeat
- Active breakers
- Active alert counts (severity-grouped)
- Last refresh (clock since last successful pull)

Each field is a `HealthPill`. Click navigates to the relevant page filtered to that domain.

### 7.4 Diff Rail (v2)

A thin horizontal strip directly under the status bar. Width: full. Height: 28 px. Background: `surface-muted`.

Contents:

- New BUY/SELL symbols since last refresh
- Symbols whose side flipped
- Symbols whose `|target_weight|` changed by ≥ 25 bps
- New ERROR or MODEL_REQUIRED states
- Cleared error states
- New active alerts (top severity only)

Each chip is clickable, scrolls/filters the active page to that symbol or alert. The diff baseline is the previous successful pull stored in the client; on page load the rail is empty until the second refresh fires.

### 7.5 Command Palette (v2)

Triggered globally by `Cmd+K` / `Ctrl+K`. Inputs are typed prefixes; results are ranked by recency.

Examples:

- `idea AAPL` → opens Trade Ideas drawer for AAPL
- `symbol AAPL` → navigates to Symbol Detail
- `signal ts_momentum` → Signals page filtered to family
- `feature vrp_percentile_rank` → Model & Features filtered to that feature's drift row
- `alert open` → Monitoring filtered to active unacknowledged
- `replay 2026-05-15T13:00Z` → Replay page seeded to that timestamp
- `halt` → guarded halt dialog (gated by role)
- `flatten` → guarded flatten dialog
- `verify-flat` → preflight verify
- `promote` → model promote dialog (gated)
- `cmd ?` → help with all available commands

The palette is keyboard-first, mouse-optional. Arrows + Enter select. Esc closes.

## 8. Visual Design System

### 8.1 Tone
(unchanged)

### 8.2 Color Tokens

Add to v1 set:

| Token | Hex | Usage |
|---|---:|---|
| `regime-trend-up` | `#0f7a54` | trending_up regime |
| `regime-trend-down` | `#b42318` | trending_down regime |
| `regime-mean-rev` | `#7c3aed` | mean_reverting regime |
| `regime-high-vol` | `#b76e00` | high_volatility regime |
| `diff-new` | `#0f7a54` | new entries in diff rail |
| `diff-changed` | `#2563a9` | mutated entries |
| `diff-cleared` | `#667085` | cleared / removed entries |
| `shadow` | `#7c3aed` | RL shadow accent |

Regime badges, drift severity badges, and SHAP contribution bars use these tokens. Do not rely on color alone.

### 8.3 Typography
(unchanged)

### 8.4 Spacing And Density
(unchanged)

### 8.5 Status Badges

Add to v1 set:

- REGIME_TREND_UP / TREND_DOWN / MEAN_REV / HIGH_VOL: colored per regime token, all caps, with confidence inline.
- DRIFT_OK / DRIFT_WARN / DRIFT_CRIT: drift severity.
- CAL_OK / CAL_STALE: calibration recency.
- RL_OK / RL_DIVERGE: RL shadow alignment.
- PRE_PASS / PRE_FAIL: preflight result.
- SHADOW: badge for RL shadow rows in audit / replay.

## 9. Buttons And Controls
(unchanged from v1, with one addition)

### 9.3 Additional Buttons (v2)

- **Simulate Targets**: Portfolio & Risk — runs pre-trade portfolio simulator with current targets.
- **Run Stress Scenario**: Scenarios page — submits a parametric shock and renders impact.
- **Open Replay**: any timestamp cell — opens Replay page seeded to that time.
- **Pin to Diff Rail**: any row — explicitly track a symbol's mutations even if below threshold.
- **Generate NL Explanation**: Trade Ideas drawer — produces a one-paragraph plain-English summary from the structured chain.

## 10. Common Components

In addition to v1's set:

### 10.6 RegimeBadge (v2)

Props:

- `regime`: `trending_up | trending_down | mean_reverting | high_volatility | unknown`
- `confidence`: float [0, 1]
- `regimeProbabilities`: dict of all four probabilities for tooltip
- `onClick`: navigates to Model & Features regime panel

### 10.7 SizingWaterfall (v2)

Props:

- `layers`: ordered list of `{ name, value, capped: bool, capReason?: string }`
- `constraintsApplied`: list of binding constraints
- `final`: signed final size
- Renders a stepped horizontal bar where each cap is marked with a kink and a color-coded reason chip.

### 10.8 ShapContributors (v2)

Props:

- `contributors`: list of `{ feature, value, contribution_signed, abs_contribution, percentile }`
- `topN`: default 10
- Renders a horizontal bar chart sorted by `abs_contribution`, with signed direction shown via diverging color and feature name on the y-axis.

### 10.9 CalibrationReliabilityPlot (v2)

Props:

- `buckets`: list of `{ predicted_bucket_lo, predicted_bucket_hi, predicted_mean, observed_rate, count }`
- `diagonalRef`: bool — show y=x reference line
- Renders predicted vs observed with bucket counts labelled.

### 10.10 EdgeSizeScatter (v2)

Props:

- `points`: list of `{ symbol, action, calibrated_prob, target_weight, family }`
- `xRef`: 0.5 (decision threshold)
- `yRef`: 0 (neutral weight)
- Quadrant labels: "high edge / small bet", "high edge / big bet", "low edge / big bet" (warn), "low edge / small bet"

### 10.11 FreshnessHeatmap (v2)

Props:

- `rows`: symbols
- `columns`: data sources (`bars`, `features`, `signals`, `onchain`, `sentiment`, `factor_model`)
- `cellValue`: staleness seconds
- Cells colored on a sequential scale, threshold-banded (green / amber / red).

### 10.12 DiffStrip (v2)
Props for the rail described in 7.4.

### 10.13 CommandPalette (v2)
Props for the palette described in 7.5.

### 10.14 NaturalLanguageExplain (v2)

Props:

- `chain`: DecisionChainStep[]
- `topShap`: ShapContributor[]
- `topSignal`: SignalRow
- `regime`: regime label + confidence
- `bet`: SizingWaterfall
- `cost`: expected bps
- Renders a one-paragraph explanation generated server-side. Falls back to the structured chain if generation fails.

## 11. Information Architecture

Primary pages (v2):

1. Command Center
2. Trade Ideas
3. Symbol Detail
4. Portfolio & Risk
5. Execution & TCA
6. Signals
7. Model & Features
8. Backtests & Research
9. **Scenarios & Stress Test** *(new)*
10. **Track Record** *(new)*
11. **Replay / Time Travel** *(new)*
12. **Preflight & Go-Live** *(new)*
13. Monitoring & Alerts
14. Audit & Compliance
15. Settings

## 12. Page Specification: Command Center

### 12.1 Purpose

First screen. Tells the operator what needs attention now, what changed, and how confident the system is.

### 12.2 Layout

Top row (status tiles):

- NAV
- Daily P&L
- Drawdown
- Gross / Net exposure
- Positions count
- Model health
- Data health (worst-of, with source breakdown tooltip)
- Regime + dominant probability *(new)*
- Drift severity *(new)*
- Calibration recency *(new)*

Diff rail:

(see 7.4)

Second row:

- Action counts (BUY / SELL / WATCH / MODEL_REQUIRED / NO_DATA / ERROR)
- Top actionable ideas (compact table)
- Active warnings / errors
- Stage latency horizontal bars

Third row *(new)*:

- **Edge-vs-size scatter** — `calibrated_prob` (x) vs `target_weight` (y) for all current ideas. Markers colored by signal family, shape by action. Quadrants annotated. Click point → opens drawer for that symbol.
- **Regime tile expanded** — four regime probabilities with sparkline of the last N bars and current dominant label.
- **Calibration recency tile expanded** — last refit timestamp, days since, OOS reliability snapshot.

Fourth row:

- Portfolio exposure chart
- Broker / circuit breaker status
- Paper / live divergence
- Recent audit / alert events

### 12.3 Components

#### Action Counts
(unchanged)

#### Top Actionable Ideas

Compact table columns (v2 adds two):

- symbol
- action
- target weight
- notional
- meta probability
- calibrated probability
- top family
- **top SHAP feature** *(new)*
- **est cost bps** *(new — rolling TCA on same algo/symbol-side)*
- reason

Row sort priority (unchanged from v1).

#### Stage Latency
(unchanged)

#### Edge vs Size Scatter (v2)
See 10.10. Highlights outliers visually that a table buries.

### 12.4 Buttons

- Refresh Dashboard (primary)
- Open Trade Ideas
- Export Summary
- Toggle Auto Refresh
- **Open Replay From Now − N min** *(new)*
- **Open Scenario Library** *(new)*

## 13. Page Specification: Trade Ideas

### 13.1 Purpose

Core daily decision table. v2 makes it the deepest page in the app by adding pre-trade cost, regime fit, per-row SHAP teaser, and natural-language explanation.

### 13.2 Table Columns

Required columns (v2 superset):

- Symbol
- Action
- Target Weight
- Target Notional
- Estimated Quantity
- Latest Price
- Latest Bar Time
- Bars Loaded
- Feature Rows
- Signal Count
- Top Signal Family
- Top Signal Side
- Top Signal Confidence
- Average Signal Confidence
- Meta Probability
- Calibrated Probability
- **Regime Fit Score** *(new — alignment between top family and current regime; e.g. momentum family in trending_up = high)*
- Bet Size
- Strategy
- **Est Cost bps** *(new)*
- **Top SHAP Feature** *(new — single-feature teaser)*
- **Track Record Win Rate** *(new — trailing 90d for this family + symbol)*
- Reason
- Data Stage / Feature Stage / Signal Stage / Model Stage / Sizing Stage / Target Stage (latencies)
- Errors

Default visible:

- Symbol
- Action
- Target Weight
- Notional
- Est Qty
- Price
- Meta / Calibrated
- Top Confidence
- Signals
- Top Family
- Regime Fit
- Est Cost bps
- Reason

### 13.3 Filters

In addition to v1:

- Regime fit minimum
- Calibrated probability range
- Has SHAP signal beyond threshold
- Estimated cost ≤ X bps
- Track record win rate ≥ X %
- Recently changed (diff filter — only rows that appear in current diff rail)

### 13.4 Row Detail Drawer

Tabs (v2 superset):

1. Explainability (full chain)
2. Signals (per-family metadata)
3. Model (SHAP + meta/calibrated + history)
4. Sizing (waterfall + constraints)
5. Data (bars + features + microstructure + structural breaks)
6. **Regime** *(new)*
7. **Cost** *(new — pre-trade)*
8. **Track Record** *(new — historical calls for this symbol-family)*
9. **NL Explanation** *(new — paragraph)*
10. Related Alerts
11. Related Audit Entries
12. JSON (raw)

#### Explainability Tab
(unchanged from v1; renders the canonical chain with status / count / latency per step)

#### Signals Tab

Now displays the **full per-family metadata payload**:

- For each fired family, render a typed sub-panel:
  - `ts_momentum`: lookbacks, weights, per-lookback z-scores, aggregate z
  - `cs_momentum`: decile rank, lookback return, z-score, skip periods
  - `mean_reversion`: half-life, ADF p-value, z-score, entry/exit thresholds
  - `stat_arb`: cointegration p-value, pair symbol, hedge ratio, spread z, spread half-life
  - `futures_carry`: front/back prices, days to expiry, carry yield
  - `vrp`: IV, RV, VRP, percentile rank, regime modifier
  - `ma_crossover`, `donchian_breakout`, `funding_rate_arb`, `cross_exchange_arb`: per-family field set
- Each panel includes timestamp, side, confidence.

#### Model Tab

Fields:

- model source (mlflow_production / confidence_fallback / none)
- model version, run id, MLflow tracking source
- trained at, model age, n_training_events
- meta probability, calibrated probability, calibration status, calibration recency
- **per-row SHAP top 10 contributors** (10.8)
- feature hash for reproducibility
- global feature importance (MDI / MDA / SFI / SHAP-summary) as collapsible reference

#### Sizing Tab

Renders `SizingWaterfall` (10.7):

```text
afml_size -> kelly_capped -> vol_adjusted (with VRP haircut step if active) -> atr_capped -> final_size
```

Plus the explicit `constraints_applied` list with which were binding. Sidebar lists every configured cap and its current headroom: `max_single_position`, `max_family_allocation`, `max_gross_exposure`, `max_crypto_allocation`, `max_sector_exposure`, `kelly_cap`, `vol_scaling`, `vrp_haircut`, `atr_cap`, `deployment_multiplier`.

#### Data Tab

In addition to v1:

- Latest bar microstructure: VWAP, tick_count, buy_volume / sell_volume, buy_ticks / sell_ticks, imbalance, threshold, bar_duration_seconds
- Microstructure feature snapshot: Kyle's lambda, Amihud, Roll spread, VPIN, Hasbrouck, OFI, trade_intensity
- Volatility feature snapshot: GARCH conditional vol, vol_term_structure, vol_of_vol, realized-vs-implied spread
- Structural-break statistics: CUSUM, SADF, GSADF, Chow
- On-chain features (if applicable): exchange inflow/outflow, whale activity, network health, stablecoin supply
- Sentiment snapshot (if applicable): FinBERT sentiment score, momentum, article count
- Triple-barrier preview for current event: upper / lower / vertical barriers, EWM vol used

#### Regime Tab (v2)

- Current regime label + four probabilities (`prob_trending_up`, `prob_trending_down`, `prob_mean_reverting`, `prob_high_vol`)
- Recent regime sequence sparkline (last N bars)
- Regime modifier the meta-labeler applied to this family
- Regime detector training metrics: last refit timestamp, val accuracy, val loss, best epoch
- "Regime fit" badge: how well this family typically performs in the dominant regime, computed from track record

#### Cost Tab (v2)

- Expected slippage bps from rolling TCA over last N orders, segmented by algo
- Expected market impact bps
- Expected commission
- Total expected cost as a fraction of target weight
- Comparison to TWAP and VWAP benchmarks from history
- Side-by-side: rebalance via VWAP vs TWAP vs market depending on user role

#### Track Record Tab (v2)

For this symbol-family:

- Trailing 90d / 180d / 365d / all-time count of calls
- Win rate, average return, median holding period
- Equity curve of "act on every call" P&L
- Worst drawdown
- Comparison: this family on this symbol vs this family on the rest of the universe

#### NL Explanation Tab (v2)

A single paragraph generated server-side from the structured chain. Example:

> The system is calling **BUY AAPL** at **target weight 1.2%** (≈ \$120k notional). The strongest signal is `ts_momentum` (long, confidence 0.74) — aggregate z-score 1.9 across 1/3/6-month lookbacks. The meta-labeler probability is **0.66**, calibrated to **0.61**. Regime is **trending_up** at 0.72 confidence, which favours momentum. The top SHAP contributor is `vol_term_structure` (positive). Sizing started at 1.8% (AFML), was clipped to 1.4% by Kelly, scaled to 1.3% by vol-adjustment, and the ATR cap took it to 1.2%. Expected cost is **6 bps**. Historical track record on this symbol-family: 58% win-rate over 24 trades in the last 180 days. No active alerts.

The structured chain remains primary; the paragraph is a convenience layer.

#### Related Alerts / Audit / JSON
(unchanged)

### 13.5 Buttons

In addition to v1:

- **Simulate Apply All** — opens Portfolio simulator with this row's target included
- **Open Replay** — opens Replay seeded to `latest_bar_at`
- **Show Track Record** — opens Track Record page filtered to this symbol-family
- **Explain in English** — toggles NL paragraph at top of drawer
- **Pin to Diff Rail**

## 14. Page Specification: Symbol Detail

### 14.1 Purpose

One-symbol forensic view. Permalinkable, multi-monitor-friendly. The drawer's deep cousin.

### 14.2 Header

Includes v1 fields plus:

- Regime fit badge for this symbol's strategy
- Estimated cost to rebalance now
- Track record summary (win rate / N trades / 90d)

### 14.3 Panels

#### Price And Bars

Candlestick / OHLC with volume below. v2 additions:

- Bar-type switcher (TICK / VOLUME / DOLLAR / TIB / VIB / TIME)
- Multi-bar overlay (primary + secondary bar type)
- **News overlay** — vertical event markers from sentiment article count, hover shows top headlines and counts
- **Trade markers** — historical entries / exits with side and P&L
- **Target / action markers** — historical action transitions
- **Stale-data badge** — when the latest bar is older than threshold

#### Target Weight History
(unchanged)

#### Signal Timeline
(unchanged)

#### Meta Probability Timeline

Adds:

- Calibrated probability overlay
- Recent regime overlay band (color by regime)
- Decision threshold line

#### Calibration History (v2)

Reliability plot for this symbol's predictions over trailing 90d/180d:

- predicted probability bucket vs observed win rate
- bucket count labels
- diagonal reference

#### Feature Snapshot

Adds:

- Drift status per feature (KL, KS p-value, mean shift, var ratio, drifted boolean) — pulled from `FeatureDriftDetector`
- Recommendation column

#### Microstructure Snapshot (v2)

Compact table: Kyle's lambda, Amihud, Roll spread, VPIN, Hasbrouck, OFI, trade_intensity — with rolling percentile rank.

#### Position State
(unchanged)

#### Orders And Fills
(unchanged)

#### Per-Symbol Track Record (v2)

Equity curve of "act on every call" P&L for this symbol over all time + family breakdown.

### 14.4 Buttons

In addition to v1:

- **Open Replay** seeded to current state
- **Open Symbol Track Record** as a full page
- **Run Symbol-Only Scenario** (apply parametric shock to just this symbol's drivers)

## 15. Page Specification: Portfolio & Risk

### 15.1 Purpose

Current portfolio, target portfolio, risk constraints, and — new in v2 — pre-trade simulation of applying the targets.

### 15.2 Panels

#### Current Holdings Table
(unchanged)

#### Target Portfolio Table

Adds:

- `unclipped_target_weight` (pre-risk-budget)
- `clipped_target_weight` (post)
- Which constraint clipped it (e.g. `max_family_allocation`)
- Expected cost bps for the delta
- Estimated time to fill given recent fill rates

#### Pre-Trade Portfolio Simulator (v2)

A panel that, given the current target table:

- Shows resulting gross / net exposure deltas
- Shows resulting factor exposures (from `FactorRiskModel`): factor loadings, factor returns, factor covariance, idiosyncratic variance, explained variance ratio
- Shows family and asset-class concentration
- Shows breaker headroom after the rebalance (max position, daily loss, drawdown)
- Estimates total cost in bps and absolute USD
- Estimates execution duration
- Toggle: "apply only ideas with calibrated_prob ≥ X" / "apply only BUYs" / "apply all"

Pure read-only simulation. No mutation.

#### Exposure Summary

Adds:

- Factor exposure bars (one per principal component, ordered by `explained_variance_ratio`)
- Systematic vs idiosyncratic variance breakdown

#### Risk Metrics

In addition to v1:

- Per-factor exposure
- Per-factor contribution to total variance
- Systematic variance fraction
- Idiosyncratic variance fraction
- Effective N (diversification ratio)

#### Correlation Matrix

Adds:

- Toggle between symbol-level, family-level, and factor-level matrices
- Rolling window selector
- Highlight negative diversifiers

### 15.3 Buttons

In addition to v1:

- **Simulate Targets** — opens pre-trade simulator
- **Export Simulation Result**
- **Compare to Yesterday** — diff vs previous day's target portfolio

## 16. Page Specification: Execution & TCA

### 16.1 Purpose
(unchanged)

### 16.2 Panels

#### Orders By Status
(unchanged)

#### Orders Table

Adds columns:

- Expected cost bps (at submission)
- Realized cost bps (post-fill)
- Cost surprise (realized − expected)
- SHAP top contributor at time of decision (for forensic linkage)

#### Fills Table
(unchanged)

#### TCA Table

(unchanged, with the full v1 column set: arrival, execution, slippage, market impact, timing cost, total cost, commission, algo, duration, fill rate, vs TWAP, vs VWAP)

#### TCA Charts

Adds:

- Expected vs realized cost scatter
- Cost by signal family box plot
- Algo recommendation panel: given recent slippage by algo for each symbol-side, recommend an algo

#### Reconciliation
(unchanged)

### 16.3 Buttons
(unchanged plus:)

- **Recommend Algo** — runs the algo selector for the current target list
- **Export Cost Forecast**

## 17. Page Specification: Signals

### 17.1 Purpose
(unchanged)

### 17.2 Panels

#### Active Families
(unchanged list)

#### Family Summary Table

Adds:

- Average regime fit for current regime
- Trailing 90d win rate when this family fired
- Average meta probability for this family's current fires

#### Latest Signals Table
(unchanged)

#### Signal Metadata Viewer

Full per-family metadata payloads (matches Trade Ideas drawer Signals tab):

- ts_momentum: lookbacks, weights, z-scores, aggregate
- cs_momentum: decile_rank, lookback_return, z_score, skip_periods
- mean_reversion: half_life, adf_pvalue, z_score, entry_threshold, exit_threshold
- stat_arb: cointegration_pvalue, pair_symbol, hedge_ratio, spread_z_score, spread_halflife
- futures_carry: front_price, back_price, days_to_expiry, carry
- funding_rate_arb / cross_exchange_arb: full payload
- vrp: iv, rv, vrp, vrp_percentile_rank, regime_modifier
- ma_crossover, donchian_breakout: EMA values, channel levels

#### Signal Correlation Matrix
(unchanged)

#### Signal-to-Meta Attribution (v2)

For each family, the average SHAP contribution of that family's signal features to the meta probability across recent fires. Identifies which families the model actually leans on vs decorates with.

#### Family Performance Attribution
(unchanged, plus regime-conditioned slice)

#### Regime-Conditioned Family Performance (v2)

For each (family, regime) pair: trade count, win rate, average P&L, average holding period. Heatmap visualization.

### 17.3 Buttons
(unchanged)

## 18. Page Specification: Model & Features

### 18.1 Purpose
(unchanged)

### 18.2 Panels

#### Production Model Status

In addition to v1:

- MLflow run id
- Alias (`production` / `staging`)
- Hyperparameters (learning_rate, n_estimators, depth, subsample, regularization)
- CV metrics (mean cv score, std cv score, per-fold cv_fold_i, train_accuracy)
- Gate history snapshot (CPCV / DSR / PBO booleans + last run timestamp)
- Calibration enabled boolean and calibrator type

#### Probability Distribution
(unchanged)

#### Calibration Reliability Plot (v2)

For trailing 90d / 180d:

- predicted probability bucket vs observed win rate
- bucket counts labelled
- diagonal reference line
- Brier score and ECE displayed as headline stats

#### Feature Importance

Tabs for:

- MDI (mean decrease impurity, summed to 1)
- MDA (mean / std / p-value per feature)
- SFI (single feature score per feature)
- SHAP global (mean |SHAP| per feature)

#### Per-Idea SHAP Browser (v2)

A dedicated panel that lets the user select any active idea (or any historical event from the audit log) and shows the full per-row SHAP decomposition.

#### Feature Drift Table
(unchanged: KL, KS statistic, KS p-value, mean shift, variance ratio, drifted, recommendation)

#### Drift Heatmap
(unchanged)

#### Retrain Status

In addition to v1:

- Last retrain timestamp + model run id
- Next scheduled retrain
- Emergency retrain state
- Drift-triggered retrain cooldown
- Latest validation recommendation
- **Retrain timeline (v2)** — vertical chart of retrains over time with success / failure markers and overlaid drift events

#### Regime Detector Panel (v2)

- Current regime + four probabilities
- Recent regime sequence
- Regime detector training metrics: train_loss, val_loss, train_accuracy, val_accuracy, best_val_loss, best_epoch, n_train, n_val
- HMM labelling stats: adf_pvalue, half_life

#### RL Shadow Comparison (v2)

- Latest paired t-stat, p-value, `rl_is_better`, significance level
- HRP vs RL metrics: total return, Sharpe, max drawdown, n_trades
- Recent decision diff: side-by-side `hrp_target` vs `rl_target` for last N decisions
- Promotion eligibility: months_span, gate status, blocking reasons

#### Backtest Gates Summary

Adds historical gate runs over time, not just the latest.

### 18.3 Buttons

In addition to v1:

- Open Per-Idea SHAP Browser
- Open RL Shadow Detail
- Run Calibration Refit (Phase 4, gated)
- Request Retrain / Emergency Retrain / Promote — gated

## 19. Page Specification: Backtests & Research

### 19.1 Purpose
(unchanged)

### 19.2 Panels

#### Headline Metrics
(unchanged)

#### Equity And Drawdown
(unchanged)

#### Trade Log

Full columns from `BacktestReport.trade_log`:

- entry_timestamp, exit_timestamp
- symbol, side
- entry_price, exit_price, size
- signal_family
- gross_pnl, costs (commission / impact / slippage), net_pnl
- holding_period_bars
- meta_label_prob
- return_pct

#### Monthly Returns
(unchanged calendar heatmap)

#### Regime Breakdown
(unchanged)

#### Strategy Breakdown
(unchanged: family, trades, win_rate, avg_net_pnl, total_net_pnl, avg_holding)

#### Drawdown Table (v2)

Top-N drawdown events from `BacktestReport.drawdown_table` with start, trough, recovery dates, magnitude.

#### Promotion Gates

CPCV: positive_count, path_count, positive_pct, threshold
DSR: dsr_statistic, p_value, observed_sharpe, expected_max_sharpe, n_trials
PBO: pbo, max_pbo
Plus pass/fail and recommendation per gate.

#### Head-to-Head Compare (v2)

Select two backtest run ids and overlay equity curves, drawdowns, monthly returns, gate verdicts side by side. Difference table for each headline metric.

### 19.3 Buttons
(unchanged plus Head-to-Head)

## 20. Page Specification: Scenarios & Stress Test *(new in v2)*

### 20.1 Purpose

Apply parametric or historical shocks to the current portfolio and target list, and observe the impact on factor exposures, P&L, drawdown, breaker headroom, and which trade ideas would be cancelled, kept, or amplified.

### 20.2 Inputs

Scenario library:

- Parametric: directional shock (`SPY ±3%`, `BTC ±10%`, custom symbol shocks)
- Volatility shock (`VIX ×1.5`, `realized vol ×2`)
- Correlation breakdown (cross-asset correlations → 1 or → 0)
- Factor flip (invert top factor return)
- Liquidity dry-up (slippage × N)
- Historical replay: pick a prior date and shock with that day's returns / vols / correlations
- Custom: user-defined dict of `{symbol: pct_move}` and `{factor: shift}`

### 20.3 Output Panels

- Portfolio P&L impact (signed, with confidence bands)
- Drawdown impact
- Factor exposure deltas
- Breaker headroom: which breaker would trip, how close
- Trade idea impact table: each current idea with old target, new target after shock, and whether it would still be a BUY/SELL
- Suggested hedges (long/short pairs that would neutralise the impact)

### 20.4 Buttons

- Run Scenario
- Save Scenario to Library
- Export Result
- Compare Scenarios (overlay two)

### 20.5 Backend

Calls `src.portfolio.factor_risk` for factor-level impact and `src.bet_sizing.cascade` re-run for per-idea impact. No state mutation.

## 21. Page Specification: Track Record *(new in v2)*

### 21.1 Purpose

Convert the model's history of calls into a track record so the operator can ask: "Has the system been right about this kind of call?"

### 21.2 Inputs

Filters:

- Symbol
- Signal family
- Regime (filter to calls made when regime was X)
- Action (BUY / SELL / WATCH)
- Probability bucket
- Time window

### 21.3 Output Panels

#### Headline Track Record Stats

- Total calls
- Win rate
- Average return
- Median holding period
- Sharpe of "act on every call" P&L
- Hit rate by probability bucket (calibration check)

#### Equity Curve of Calls

The implicit P&L of acting on every filtered call, scaled by signal-recommended size and assuming realised execution costs.

#### Hit Rate Heatmap

Rows: signal family. Columns: regime. Cells: win rate. Reveals which family-regime combinations the model is genuinely good at.

#### Call-Level Table

Sortable list of historical calls with timestamp, symbol, action, target weight, calibrated prob, regime, realised P&L, holding period, signal family, and whether the call agreed with the eventual outcome.

#### Track Record Diff vs Backtest

For each family, compare live track record win rate to the latest backtest's per-family win rate. Significant divergence flags model decay.

### 21.4 Backend

Joins audit-log entries (`signal_generated`, `meta_label_predicted`, `bet_sized`, `position_opened`, `position_closed`) with execution storage. New service layer module `src/web/services/track_record_service.py`.

### 21.5 Buttons

- Apply Filters
- Export Track Record CSV
- Open Decay Alert Setup

## 22. Page Specification: Replay / Time Travel *(new in v2)*

### 22.1 Purpose

Reconstruct the system's state and decisions at any past timestamp using only the audit chain and persisted data. Critical for forensic debugging, compliance reviews, and model post-mortems.

### 22.2 Layout

Top: time picker (calendar + time + symbol filter) and a horizontal "scrub bar" of audit events.

Body: a clone of the Trade Ideas page rendered as it was at `T = picked_time`. The drawer, charts, regime banner, and SHAP all reflect the world at `T`, not now.

### 22.3 Sources

- `src.execution.audit_log` event chain (`signal_generated`, `meta_label_predicted`, `bet_sized`, `order_submitted`, `fill_received`, `position_opened`, `position_closed`, `breaker_triggered`, `phase_promoted`, `rl_shadow_decision`, `operator_action`)
- `src.data_engine.storage.database` for bars / features / signals up to `T`
- Model registry for the model active at `T` (`model_version` field on audit events)
- Drift / regime / calibration values from `T`

### 22.4 Panels

#### Audit Event Timeline

Horizontal timeline with event markers colored by event type. Click to jump.

#### State At T

The Trade Ideas table as it appeared at `T`, with the active model version, regime, drift status, breaker status.

#### Diff vs T+δ

Side-by-side: state at `T` vs state at `T + 5 min`, with the audit events that fired between. Operators see exactly which event caused which mutation.

#### Chain Verification Inline

Shows that the audit signatures verify from genesis to `T`.

### 22.5 Constraints

- Replay is strictly read-only.
- All recomputation is server-side from persisted data; the client only receives the final snapshot.
- If a feature or bar is missing from `T`, the affected fields are shown as "no_data" with a diagnostic.

### 22.6 Buttons

- Pick Time
- Step Forward / Backward by audit event
- Step Forward / Backward by N minutes
- Export Snapshot
- Diff vs Now

## 23. Page Specification: Preflight & Go-Live *(new in v2)*

### 23.1 Purpose

Surface the live preflight checklist (currently in `src.execution.preflight` and `docs/go_live_checklist.md`) as a real page so a go-live operator can see, in one screen, what is blocking live trading and what is ready.

### 23.2 Panels

#### Blocker Checks Vector

A vertically stacked list of every preflight blocker from `src.execution.preflight`, each as a row:

- check name
- status (PASS / FAIL / SKIPPED / UNKNOWN)
- last evaluated at
- reason if FAIL
- runbook link (deep-link into `docs/runbooks/`)

#### Operator Sentinel

- `.operator_checkin` sentinel state (touched / stale)
- HALT sentinel state
- Last operator action

#### Capital Deployment Ramp

- Current deployment phase (1–4) and multiplier
- Days in phase, next phase eligibility, blockers

#### Infra Probes

- DB health, feature store health, MLflow health, broker health, Prometheus health
- Disk usage, CPU, memory

#### Recent Disaster Recovery State

- Last clean shutdown, last crash, last `recover` invocation
- Recovery health summary

#### Reconciliation Snapshot

- Internal vs broker quantity per symbol, delta, severity

### 23.3 Buttons

- Run Preflight Now (Phase 4 gated)
- Touch Check-in (Phase 4 gated)
- Clear HALT (Phase 4 gated, typed confirmation)
- Open Runbook
- Export Preflight Report

### 23.4 Backend

`src/web/services/preflight_service.py` wraps `src.execution.preflight` and surfaces structured results.

## 24. Page Specification: Monitoring & Alerts

### 24.1 Purpose
(unchanged)

### 24.2 Panels

#### Metrics Overview

Full list from `src.monitoring.metrics`:

- portfolio_nav
- portfolio_drawdown
- portfolio_daily_pnl
- portfolio_gross_exposure
- portfolio_net_exposure
- positions_count
- target_weight (labelled by symbol / strategy)
- orders_submitted_total
- orders_filled_total
- orders_rejected_total
- signal_count (labelled by family)
- meta_label_prob histogram
- execution_slippage_bps histogram
- stage_latency_seconds (labelled by stage)
- stage_cost_usd_total
- stage_items_total
- bar_formation_rate (labelled by symbol)
- data_gap_seconds (labelled by symbol)
- feature_drift_kl (labelled by feature)
- feature_freshness_hours
- model_last_retrain_age_hours
- broker_heartbeat
- circuit_breaker_state

Each can be expanded to its own time-series chart.

#### Freshness Heatmap (v2)

`FreshnessHeatmap` component (10.11). Rows: symbols. Columns: sources (bars / features / signals / onchain / sentiment / factor model). Cells: staleness seconds with color banding.

#### Alert Feed

Adds:

- Severity filter
- Source filter (signal / model / drift / breaker / broker / data / preflight)
- Symbol filter
- Acknowledged / unacknowledged toggle
- **Diff filter: only alerts new since last refresh**

#### Circuit Breaker State
(unchanged)

#### Data Health

Adds:

- Per-source freshness time series
- Bar formation rate by symbol
- Latest bar age by symbol
- Sentiment article count rate
- On-chain pull cadence

#### Escalation Channels (v2)

A read-only panel showing configured Telegram bots, escalation policies, and recent delivery success/failure. Does not expose tokens.

### 24.3 Buttons
(unchanged plus:)

- Acknowledge Alert (Phase 4)
- Silence Alert (Phase 4)
- Test Escalation Channel (Phase 4, admin)

## 25. Page Specification: Audit & Compliance

### 25.1 Purpose
(unchanged)

### 25.2 Panels

#### Audit Summary
(unchanged)

#### Audit Table

Full column set:

- entry_id
- timestamp
- event_type (full enum from `src.execution.audit_log`: SIGNAL_GENERATED, META_LABEL_PREDICTED, BET_SIZED, ORDER_SUBMITTED, FILL_RECEIVED, POSITION_OPENED, POSITION_CLOSED, BREAKER_TRIGGERED, PHASE_PROMOTED, RL_SHADOW_DECISION, OPERATOR_ACTION)
- symbol
- model_version
- context summary (truncated)
- output summary (truncated)
- signature status (chain-ok / chain-broken)

#### Audit Entry Drawer

Tabs:

- Context (full `decision_context` dict)
- Output (full `decision_output` dict)
- Signature (`prev_signature`, `signature`)
- Related Objects (linked order, fill, position, breaker, model run, shadow record)
- **Open in Replay (v2)** — jumps to Replay page seeded to this event's timestamp and symbol

#### Chain Verification
(unchanged)

#### Event Type Timeline (v2)

A horizontal stacked-bar chart of event counts per type per time bucket. Reveals system rhythm and anomalies (e.g. spike in BREAKER_TRIGGERED).

### 25.3 Buttons
(unchanged plus Open in Replay)

## 26. Settings

(unchanged from v1, plus:)

- Default scenario presets
- Custom column sets per page
- Diff rail thresholds (target_weight delta, alert severity floor)
- Command palette shortcuts editor

## 27. API Design

### 27.1 Standard Response Envelope

(unchanged from v1, plus `model_version` and `regime` fields on responses that depend on them)

```ts
type ApiEnvelope<T> = {
  as_of: string;
  source: string;
  staleness_seconds?: number;
  source_freshness?: Record<string, number>; // v2: per-source breakdown
  model_version?: string;                    // v2
  regime?: RegimeSnapshot;                   // v2
  warnings: string[];
  errors: string[];
  data: T;
};

type RegimeSnapshot = {
  label: "trending_up" | "trending_down" | "mean_reverting" | "high_volatility" | "unknown";
  probabilities: {
    trending_up: number;
    trending_down: number;
    mean_reverting: number;
    high_volatility: number;
  };
  as_of: string;
};
```

### 27.2 Core Read Endpoints

v1 endpoints retained. New endpoints:

```text
GET  /api/v1/overview                        (v1)
GET  /api/v1/trade-ideas                     (v1)
GET  /api/v1/trade-ideas/{symbol}            (v1, extended payload)
GET  /api/v1/symbols/{symbol}/bars           (v1, with bar_type query)
GET  /api/v1/symbols/{symbol}/features/latest (v1)
GET  /api/v1/symbols/{symbol}/microstructure (v2, new)
GET  /api/v1/symbols/{symbol}/structural-breaks (v2, new)
GET  /api/v1/symbols/{symbol}/onchain         (v2, new)
GET  /api/v1/symbols/{symbol}/sentiment       (v2, new)
GET  /api/v1/symbols/{symbol}/signals         (v1)
GET  /api/v1/symbols/{symbol}/probabilities   (v1, with calibration history)
GET  /api/v1/symbols/{symbol}/shap            (v2, new — per-row SHAP for latest event)
GET  /api/v1/symbols/{symbol}/track-record    (v2, new)
GET  /api/v1/symbols/{symbol}/news-overlay    (v2, new — sentiment article markers for chart)
GET  /api/v1/portfolio/summary                (v1)
GET  /api/v1/portfolio/positions              (v1)
GET  /api/v1/portfolio/targets                (v1)
GET  /api/v1/portfolio/risk                   (v1)
GET  /api/v1/portfolio/factor-decomposition   (v2, new)
POST /api/v1/portfolio/simulate               (v2, new — pre-trade simulator)
GET  /api/v1/execution/orders                 (v1)
GET  /api/v1/execution/fills                  (v1)
GET  /api/v1/execution/tca                    (v1)
GET  /api/v1/execution/reconciliation         (v1)
GET  /api/v1/execution/cost-forecast          (v2, new)
GET  /api/v1/signals/summary                  (v1)
GET  /api/v1/signals/latest                   (v1, full per-family metadata)
GET  /api/v1/signals/correlation              (v1)
GET  /api/v1/signals/family-regime-attribution (v2, new)
GET  /api/v1/signals/family-shap-attribution  (v2, new)
GET  /api/v1/model/status                     (v1, extended)
GET  /api/v1/model/calibration                (v2, new)
GET  /api/v1/model/features/drift             (v1)
GET  /api/v1/model/features/importance        (v1)
GET  /api/v1/model/regime                     (v2, new)
GET  /api/v1/model/rl-shadow                  (v2, new)
GET  /api/v1/model/retrain-history            (v2, new)
GET  /api/v1/backtests/runs                   (v1)
GET  /api/v1/backtests/{run_id}               (v1)
GET  /api/v1/backtests/compare?a=...&b=...    (v2, new — head-to-head)
POST /api/v1/scenarios/run                    (v2, new)
GET  /api/v1/scenarios/library                (v2, new)
GET  /api/v1/track-record                     (v2, new — global view)
GET  /api/v1/replay?ts=...&symbol=...         (v2, new)
GET  /api/v1/preflight                        (v2, new)
GET  /api/v1/monitoring/metrics/snapshot      (v1)
GET  /api/v1/monitoring/freshness-heatmap     (v2, new)
GET  /api/v1/monitoring/escalation-channels   (v2, new)
GET  /api/v1/alerts                           (v1)
GET  /api/v1/audit/entries                    (v1)
GET  /api/v1/audit/verify                     (v1)
GET  /api/v1/audit/event-timeline             (v2, new)
GET  /api/v1/stream/ops                       (v1 SSE — extended to include regime/drift/calibration ticks)
GET  /api/v1/stream/diff                      (v2, new SSE — delivers diff rail events)
```

### 27.3 Phase 4 Mutation Endpoints

v1 endpoints retained. New:

```text
POST /api/v1/preflight/check-in              (v2)
POST /api/v1/preflight/clear-halt            (v2, typed confirmation)
POST /api/v1/model/calibration/refit         (v2, gated)
POST /api/v1/escalation/test                 (v2, admin)
POST /api/v1/saved-views                     (v2)
DELETE /api/v1/saved-views/{id}              (v2)
```

### 27.4 Trade Idea DTO (v2)

```ts
type TradeIdea = {
  symbol: string;
  action: "BUY" | "SELL" | "WATCH" | "MODEL_REQUIRED" | "NO_DATA" | "ERROR";
  target_weight: number;
  target_notional: number;
  estimated_quantity: number | null;
  latest_price: number | null;
  latest_bar_at: string | null;
  bar_type: string | null;
  bars_loaded: number;
  feature_rows: number;
  signal_count: number;
  top_signal_family: string | null;
  top_signal_side: -1 | 0 | 1 | null;
  top_signal_confidence: number | null;
  avg_signal_confidence: number | null;
  meta_probability: number | null;
  calibrated_probability: number | null;
  regime: RegimeSnapshot | null;             // v2
  regime_fit_score: number | null;           // v2
  bet_size: number | null;
  sizing_constraints_applied: string[];      // v2
  strategy: string | null;
  reason: string;
  expected_cost_bps: number | null;          // v2
  top_shap_feature: ShapContributor | null;  // v2
  track_record_win_rate: number | null;      // v2
  track_record_n: number | null;             // v2
  stage_latency_seconds: Record<string, number>;
  errors: string[];
};

type ShapContributor = {
  feature: string;
  value: number;
  contribution: number;
  abs_contribution: number;
  percentile: number;
};
```

### 27.5 Trade Idea Detail DTO (v2)

```ts
type TradeIdeaDetail = {
  idea: TradeIdea;
  chain: DecisionChainStep[];
  signals: SignalRow[];                     // v1
  signal_metadata: Record<string, FamilyMetadata>;  // v2 — typed per family
  model: ModelInferenceDetail;
  shap: ShapContributor[];                  // v2 — per-row top N
  sizing: SizingWaterfall;                  // v2 — typed layers + constraints
  features: FeatureSnapshot;
  microstructure: MicrostructureSnapshot;   // v2
  structural_breaks: StructuralBreakSnapshot; // v2
  onchain: OnchainSnapshot | null;          // v2
  sentiment: SentimentSnapshot | null;      // v2
  bars: BarSummary;
  cost_forecast: CostForecast;              // v2
  track_record: TrackRecordSummary;         // v2
  nl_explanation: string | null;            // v2
  related_alerts: AlertSummary[];
  related_audit_entries: AuditSummary[];
};
```

### 27.6 Sizing Waterfall DTO (v2)

```ts
type SizingWaterfall = {
  layers: Array<{
    name: "afml" | "kelly" | "vol" | "atr" | "final";
    value: number;
    capped: boolean;
    cap_reason: string | null;
  }>;
  constraints_applied: string[];
  side: -1 | 0 | 1;
  final: number;
};
```

### 27.7 Family Metadata DTO (v2)

Typed union by family — each family returns its native payload as defined in `src.signal_battery.*`. Examples:

```ts
type TsMomentumMetadata = {
  family: "ts_momentum";
  lookbacks: number[];
  weights: number[];
  z_scores: Record<string, number>;
  aggregate: number;
};

type StatArbMetadata = {
  family: "stat_arb";
  pair_symbol: string;
  cointegration_pvalue: number;
  hedge_ratio: number;
  spread_z_score: number;
  spread_halflife: number;
};

// (similar for cs_momentum, mean_reversion, futures_carry, vrp, funding_rate_arb, cross_exchange_arb, ma_crossover, donchian_breakout)
```

### 27.8 Decision Chain Step (v2)

```ts
type DecisionChainStep = {
  name: "bars" | "features" | "signals" | "regime" | "model" | "calibration" | "sizing" | "target" | "cost" | "risk" | "execution";
  status: "ok" | "warning" | "error" | "skipped" | "unknown";
  value: string | number | null;
  count?: number;
  timestamp?: string;
  latency_seconds?: number;
  message?: string;
};
```

### 27.9 Replay Request / Response (v2)

```ts
type ReplayRequest = {
  ts: string;       // ISO timestamp
  symbol?: string;
};

type ReplaySnapshot = {
  ts: string;
  model_version: string;
  regime: RegimeSnapshot;
  ideas: TradeIdea[];
  audit_chain_verified_to: string;
  warnings: string[];
};
```

### 27.10 Scenario Request / Response (v2)

```ts
type ScenarioRequest = {
  shocks: {
    symbol_pct?: Record<string, number>;
    vol_multiplier?: number;
    correlation_target?: number | null;
    factor_shift?: Record<string, number>;
    liquidity_multiplier?: number;
  };
  apply_to: "current_portfolio" | "current_targets" | "both";
};

type ScenarioResult = {
  pnl_impact_usd: number;
  drawdown_impact: number;
  factor_exposure_deltas: Record<string, number>;
  breaker_headroom: Record<string, number>;
  affected_ideas: Array<{
    symbol: string;
    old_target: number;
    new_target: number;
    flipped: boolean;
  }>;
  suggested_hedges: Array<{ long: string; short: string; ratio: number }>;
  warnings: string[];
};
```

## 28. Backend Module Interfaces

### 28.1 Existing Source Modules

(v1 list retained — see [docs/web_app_design.md](web_app_design.md) §24.1)

Additional in v2:

| Web App Domain | Existing Module |
|---|---|
| Regime detection | `src.ml_layer.regime_detector` |
| Per-row SHAP | `src.ml_layer.feature_importance` |
| Calibration | `src.ml_layer.meta_labeler` (isotonic calibrator) |
| Microstructure features | `src.feature_factory.microstructure` |
| Structural breaks | `src.feature_factory.structural_breaks` |
| On-chain features | `src.feature_factory.onchain` |
| Sentiment features | `src.feature_factory.sentiment` |
| Labeling barriers | `src.labeling` (triple-barrier outputs) |
| Factor risk decomposition | `src.portfolio.factor_risk` |
| Preflight | `src.execution.preflight` |
| Disaster recovery | `src.execution.disaster_recovery` |
| Capital deployment | `src.execution.capital_deployment` |
| Infra probes | `src.execution.infra_probe` |
| Retrain pipeline | `src.ml_layer.retrain_pipeline` |
| Retrain scheduler | `src.execution.retrain_scheduler` |

### 28.2 Backend Service Layer

In addition to v1's services:

```text
src/web/services/regime_service.py
src/web/services/shap_service.py
src/web/services/microstructure_service.py
src/web/services/onchain_service.py
src/web/services/sentiment_service.py
src/web/services/cost_forecast_service.py
src/web/services/track_record_service.py
src/web/services/scenario_service.py
src/web/services/replay_service.py
src/web/services/preflight_service.py
src/web/services/calibration_service.py
src/web/services/rl_shadow_service.py
src/web/services/diff_service.py
src/web/services/freshness_service.py
src/web/services/escalation_service.py
src/web/services/nl_explain_service.py
```

Each service:

- accepts request filters
- calls existing repository modules
- normalizes response shape
- applies pagination/limits
- strips sensitive fields
- adds `as_of`, `source`, `staleness_seconds`, `source_freshness`, `model_version`, `regime` metadata as appropriate
- returns typed DTOs

## 29. Visualization Standards

(v1 §25 retained — no 3D, no gradients, no dark backgrounds, server-side downsampling, explicit stale/empty/error states)

### 29.1 Additional Chart Types (v2)

| Data | Chart |
|---|---|
| Edge vs size (Command Center) | Scatter with quadrant labels |
| Sizing cascade per idea | Stepped horizontal waterfall |
| Per-row SHAP | Horizontal diverging bars |
| Calibration reliability | Predicted-vs-observed plot with diagonal |
| Source freshness per symbol | Heatmap, sequential scale, threshold-banded |
| Regime sequence | Colored band time-series |
| Per-family regime attribution | Heatmap, family × regime |
| Track record equity curve | Line chart with confidence band |
| Scenario impact | Diverging bar / waterfall hybrid |
| Audit event timeline | Stacked bar over time buckets |

### 29.2 Diff-Rail Visual Spec (v2)

Inline chips, 4 px horizontal padding, 24 px height, colored by `diff-*` tokens. Truncate at viewport width with a "+N more" chip.

## 30. Performance Design

(v1 §26 retained)

### 30.1 Additional Frontend Considerations (v2)

- Diff rail computes diffs on the client from cached previous payload; no extra request.
- SHAP arrays are capped server-side to top N per row.
- Per-family metadata is sent only inside the drawer fetch, never in the list fetch.
- Replay snapshot is computed server-side; client receives a single immutable snapshot per timestamp.
- Scenario simulation is server-side; the result is cached by `(scenario_hash, portfolio_hash, targets_hash)` for 60 s.

### 30.2 Refresh Tiers (v2)

| Data Type | Refresh |
|---|---:|
| Alert feed | SSE or 10 s |
| Broker heartbeat | 10 s |
| Circuit breaker state | 10 s |
| Regime snapshot | 30 s |
| Drift severity | 60 s |
| Calibration recency | 5 min |
| Portfolio summary | 15 to 30 s |
| Trade ideas | 1 to 5 min |
| Track record | 5 min (cached) |
| Replay snapshot | manual / one-shot |
| Scenario result | manual / one-shot |
| Model status | 5 min |
| Backtest reports | manual / 10 min |
| Audit table | 30 s or manual |
| Freshness heatmap | 30 s |

## 31. Reliability And Empty States

(v1 §27 retained, plus:)

- For Replay: if audit chain is broken between genesis and requested `ts`, render a banner with the first broken entry id and disable forward navigation past that point.
- For Scenarios: if the factor model is stale beyond threshold, banner the result with a `STALE_FACTOR_MODEL` warning and degrade gracefully (still show direct symbol shock impacts).
- For Per-row SHAP: if SHAP cannot be computed for the active model (e.g. RF without TreeSHAP), show "SHAP unavailable for this model class" with the model class name.
- For NL Explanation: if generation fails or is disabled, render the structured chain only.

## 32. Security And Permissions

(v1 §28 retained, plus:)

| Role | Replay | Scenario | Track Record | Preflight Run | Calibration Refit |
|---|---|---|---|---|---|
| viewer | yes | yes (read-only library) | yes | no | no |
| operator | yes | yes | yes | no | no |
| live_operator | yes | yes | yes | yes | no |
| quant_admin | yes | yes | yes | yes | yes |
| admin | yes | yes | yes | yes | yes |

### 32.1 Dangerous Action Confirmation
(unchanged from v1)

## 33. Accessibility

(v1 §29 retained, plus:)

- Diff rail chips have ARIA-live region announcements on update.
- Command palette is fully keyboard navigable, screen-reader friendly, with role=combobox.
- Edge-vs-size scatter has a tabular fallback rendered when "high contrast" or "no animation" preferences are set.
- Heatmaps include a tabular dump option.

## 34. Testing Strategy

(v1 §30 retained, plus:)

### 34.1 v2 Additional Tests

- Component tests for `RegimeBadge`, `SizingWaterfall`, `ShapContributors`, `CalibrationReliabilityPlot`, `EdgeSizeScatter`, `FreshnessHeatmap`, `DiffStrip`, `CommandPalette`, `NaturalLanguageExplain`.
- Contract tests for new DTOs: `TradeIdea` (v2), `TradeIdeaDetail` (v2), `SizingWaterfall`, `FamilyMetadata` union, `ReplaySnapshot`, `ScenarioResult`.
- Replay determinism test: pulling the same `ts` twice returns byte-identical snapshots.
- Scenario idempotency test: same inputs return cached result.
- Diff rail correctness test: simulated payload mutation produces correct chip set.
- Permission tests for new endpoints.
- Calibration redaction test: no per-row SHAP leaks beyond top N.

### 34.2 End-to-End Scenarios (v2)

In addition to v1's scenarios:

- Open Command Center, observe edge-vs-size scatter and diff rail populate on second refresh.
- Filter Trade Ideas to BUY with calibrated_prob ≥ 0.6 and regime fit ≥ 0.7.
- Open AAPL drawer, navigate to Cost tab, confirm expected_cost_bps populates.
- Run a parametric scenario, save it to library.
- Open Replay seeded to a past timestamp, confirm state reconstructs and chain verifies.
- Open Preflight, observe blocker vector and runbook deep-links.
- Navigate to Track Record, filter by family + regime, confirm equity curve renders.
- Use Cmd+K to navigate to `feature vpin` and confirm it lands on the drift row.

## 35. Implementation Plan

### Phase 1: Real Read-Only Operator App

Build:

- React app shell
- FastAPI BFF
- Command Center (without v2 advanced tiles)
- Trade Ideas table with v2 columns (regime fit, expected cost, top SHAP, track record win rate)
- Trade Idea drawer with v2 tabs (Explainability, Signals with full family metadata, Model with per-row SHAP, Sizing with waterfall + constraints, Data with microstructure / structural breaks, Regime, Cost, Track Record, NL Explanation, JSON)
- Global status bar with regime, drift, calibration tiles
- Diff rail
- Cmd+K command palette
- Basic API contracts around existing trade ideas + new endpoints listed in §27.2 that are read-only and not gated

Do not build:

- Live mutation actions
- Scenarios mutation (read-only scenario engine is OK)
- Admin settings
- Preflight write actions

### Phase 2: Diagnostics And Drilldowns

Build:

- Symbol Detail (full v2 panel set including news overlay, calibration history, microstructure)
- Signals page (full per-family metadata viewer, regime-conditioned attribution)
- Model & Features (calibration plot, per-row SHAP browser, regime panel, RL shadow comparison, retrain timeline)
- Portfolio & Risk with factor decomposition + pre-trade simulator (read-only)
- Feature drift visualisations
- Freshness heatmap

### Phase 3: Execution, Monitoring, Audit, Replay, Track Record

Build:

- Execution & TCA with v2 cost columns and algo recommender
- Monitoring & Alerts with full Prometheus surface, freshness heatmap, escalation panel
- Audit & Compliance with event timeline
- Replay / Time Travel page (read-only)
- Track Record page (read-only)
- Backtests & Research head-to-head compare

### Phase 4: Controlled Operator Actions And Live Workflows

Build:

- Acknowledge / silence alerts
- Mark review complete
- Request paper replay
- Request retrain
- Halt, verify flat, flatten
- Gated promotion actions
- Preflight & Go-Live full page with mutation actions (check-in, clear HALT, etc.)
- Scenarios page with persistent library
- Calibration refit request
- Saved views CRUD

## 36. Acceptance Criteria

The app is Phase 1 production-ready when:

- Command Center loads from real APIs, with regime, drift, calibration, and freshness tiles populating from live data.
- Diff rail correctly reflects deltas across consecutive refreshes.
- Cmd+K palette routes to every primary destination.
- Trade Ideas table exposes all v2 columns by default-visible set, with the full v2 column set available via column-visibility toggle.
- Every trade idea opens a drawer with the v2 tabs and renders SHAP, per-family metadata, regime, sizing waterfall, and NL explanation.
- Model / data / broker / breaker / regime / drift / calibration status is visible globally.
- All critical warnings / errors are visible without hover.
- No secrets are exposed in network payloads.
- Tables are sortable / filterable / exportable.
- Empty / stale / error / permission-denied states are implemented.
- Read-only mode cannot mutate trading state.
- API responses are typed and documented, including v2 envelope fields (`source_freshness`, `model_version`, `regime`).

## 37. Open Design Questions

v1 questions retained, plus:

- Should NL Explanation be a deterministic templated paragraph (recommended for Phase 1) or use an LLM via a controlled prompt (Phase 4)?
- Should the pre-trade simulator support hypothetical capital adjustments, or only the current NAV?
- Should Replay re-run the model from persisted features, or only reconstruct what the audit log already records? (Recommended: reconstruct from audit log first; allow opt-in model re-run for forensic deep-dives in Phase 4.)
- Should Track Record include hypothetical "act on every WATCH" P&L for upside calibration, or only realised P&L?
- How long should the diff rail baseline persist? (Recommended: previous successful pull only, within a session.)
- For Scenarios, what's the canonical factor library — top-K PCA factors from `factor_risk`, or a named factor set?
- Should the freshness heatmap include MLflow as a column? (Recommended: yes, as model-source freshness.)
- Per-row SHAP requires the model to support TreeSHAP / KernelSHAP. Which models in the registry currently support it, and what's the fallback ordering?

## 38. Design Summary

The web app is a complete operator cockpit. The first screen answers what matters now, what changed since last refresh, and how confident the system is. Every recommendation is explainable down to the row-level SHAP contribution and the binding sizing constraint. Every trust signal — model age, calibration recency, regime, drift, broker, breakers, source-by-source freshness — is in the top bar. Every action and outcome is traceable through audit, with Replay reconstructing past state on demand. The operator can stress-test the portfolio without touching production, simulate a rebalance before submitting, and judge the model by its track record, not just its current call.

Dense, table-first, chart-supported, calm, auditable, counterfactual-aware. Beyond what an internal terminal usually does, because the model under it is more transparent than most.
