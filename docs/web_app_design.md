# Wang Trading Web App Design

Status: design proposal
Audience: product, quant engineering, execution engineering, frontend engineering, operations
Scope: real web application for read-only and controlled operator workflows
Out of scope for this document: implementation PR, deployment migration, broker credential handling

## 1. Executive Summary

The current Trade Ideas UI proves that the live stack can produce a read-only symbol-level decision table. The real web app should be an operator-grade trading cockpit that exposes the entire decision chain:

```text
Bars -> Features -> Signals -> Meta Probability -> Bet Size -> Target Weight -> Action -> Risk/Execution State -> Outcome
```

The application must answer four questions at all times:

1. What should I do now?
2. Why is the system saying this?
3. Can I trust it?
4. What happened after acting?

The application should be dense, fast, auditable, and calm. It should feel closer to an internal hedge-fund risk terminal than a SaaS marketing dashboard.

## 2. Product Goals

### 2.1 Primary Goals

- Show current trade ideas with enough context to make an informed operator decision.
- Explain every action through a traceable chain of bars, features, signals, model output, sizing, and risk state.
- Surface trust and safety state: model readiness, data freshness, drift, preflight, circuit breakers, broker health, reconciliation.
- Provide portfolio, execution, backtest, model, signal, monitoring, and audit views from a single web application.
- Keep live trading actions out of the default experience until gated workflows are explicitly designed and implemented.
- Make all critical state visible without relying on hover-only interactions.

### 2.2 Non-Goals

- This is not a retail trading app.
- This is not a broker replacement.
- This is not a public client portal.
- This is not a marketing landing page.
- This should not execute live trades in Phase 1.
- This should not expose secrets, credentials, unrestricted SQL, or raw infrastructure internals.

## 3. Users And Operating Context

### 3.1 Personas

| Persona | Needs | Primary Screens |
|---|---|---|
| Daily operator | Decide what needs attention now, confirm system health, review ideas | Command Center, Trade Ideas, Monitoring |
| Quant researcher | Understand signal/model behavior and promotion readiness | Signals, Model & Features, Backtests |
| Risk manager | Monitor exposure, concentration, drawdown, breakers | Portfolio & Risk, Monitoring, Audit |
| Execution engineer | Inspect order routing, fills, slippage, reconciliation | Execution & TCA, Audit |
| Compliance reviewer | Reconstruct decisions and verify audit chain | Audit & Compliance |

### 3.2 Usage Modes

| Mode | Description | Allowed Actions |
|---|---|---|
| Read-only | Default mode for Phase 1 to Phase 3 | Filter, inspect, export |
| Paper operator | Paper trading or rehearsal review | Acknowledge alerts, request replay, export reports |
| Live operator | Real capital workflow after explicit gating | Halt, flatten, acknowledge, review complete, export |
| Admin | Configuration and user management | Role and token administration |

## 4. Design Principles

### 4.1 Operational Density

Use compact panels, tables, and inline status. Avoid oversized hero sections, decorative cards, or narrative copy. Every pixel should help the operator scan, compare, or investigate.

### 4.2 Explainability First

Every trade idea must expose:

- latest price and bar time
- bars loaded
- feature rows produced
- signals fired
- top signal family and confidence
- full signal stack
- model source and version
- meta probability and calibrated probability
- bet sizing waterfall
- constraints applied
- target weight and notional
- action and reason
- warnings/errors
- stage latency

### 4.3 Trust State Is Always Visible

The app must always show:

- data freshness
- model age/source
- broker heartbeat
- circuit breaker state
- paper/live divergence
- active alerts
- last generated timestamp

### 4.4 Safety By Default

Read-only views are default. Any externally visible action needs:

- explicit role permission
- clear preview
- confirmation dialog
- audit entry
- result state

High-risk actions like halt, flatten, live order override, or model promotion require a stronger confirmation pattern.

### 4.5 Progressive Disclosure

The top-level app shows status and decisions. Detail panels explain why. Deep pages expose raw diagnostics. The operator should not need to leave the app to reconstruct a decision.

## 5. Recommended Technical Architecture

### 5.1 Frontend

Recommended:

- React + TypeScript SPA.
- Vite for local development and production bundle.
- TanStack Query for server-state fetching, caching, invalidation, polling, and retries.
- TanStack Table or equivalent headless grid for dense, virtualized, sortable tables.
- Apache ECharts for most operational charts and heatmaps.
- TradingView Lightweight Charts or equivalent financial chart component for price/candlestick charts if license and requirements fit.
- CSS modules, Tailwind, or design-token-driven plain CSS. Keep visual system local and explicit.

Rationale:

- The app is highly interactive and data-heavy.
- Most state is server state, not local client state.
- Tables need full control over density, pinned columns, custom cells, and virtualization.
- Charts need mixed time series, histograms, heatmaps, and waterfall displays.

### 5.2 Backend

Recommended:

- FastAPI backend-for-frontend service.
- Pydantic response models for every API contract.
- Existing Python modules remain the source of trading logic.
- Read-only endpoints in Phase 1 to Phase 3.
- Controlled mutation endpoints in Phase 4.

Do not move core quant logic into the frontend.

### 5.3 High-Level System

```text
Browser
  -> React application
  -> API client
  -> FastAPI BFF
  -> src.ui.trade_ideas
  -> src.data_engine.storage.database
  -> src.execution.storage
  -> src.monitoring.metrics
  -> src.monitoring.alerting
  -> src.execution.audit_log
  -> src.backtesting.report
  -> TimescaleDB / MLflow / Prometheus / logs
```

### 5.4 Realtime Strategy

Use a mixed approach:

- Poll trade ideas every 1 to 5 minutes by default.
- Poll operational health every 10 to 30 seconds.
- Use Server-Sent Events for alert feed, breaker state, broker heartbeat, and cycle updates.
- Avoid WebSockets until bidirectional workflows are needed.

## 6. Data Exposure Policy

### 6.1 Expose To Client

- trade ideas
- target weights
- target notional
- model metadata
- meta probabilities
- signal summaries
- signal metadata relevant to explainability
- feature names and latest values
- drift statistics
- portfolio state
- target portfolio
- order/fill summaries
- TCA metrics
- backtest metrics
- promotion gates
- monitoring metrics
- alert feed
- audit trail summaries
- audit chain verification status

### 6.2 Do Not Expose To Client

- broker API keys
- signing keys
- raw secrets
- database credentials
- MLflow credentials
- Telegram bot tokens
- unrestricted SQL
- raw environment variables
- private infrastructure topology beyond health/status
- full raw tick stream unless explicitly needed and permissioned

### 6.3 Sanitization Rules

Every API response must:

- include `as_of`
- include `source`
- include `staleness_seconds` when the data is time-sensitive
- include warnings separately from errors
- never include credential-like keys
- cap array sizes unless explicitly paginated
- use UTC timestamps in ISO 8601

## 7. Global Application Shell

### 7.1 Layout

```text
+--------------------------------------------------------------------------+
| Global Status Bar: NAV | P&L | DD | Model | Data | Broker | Breakers     |
+--------------+-----------------------------------------------------------+
| Sidebar      | Page content                                              |
|              |                                                           |
| Command      | Page toolbar                                              |
| Trade Ideas  | Primary panels/tables/charts                              |
| Symbol       | Side drawer for row drilldown                             |
| Portfolio    |                                                           |
| Execution    |                                                           |
| Signals      |                                                           |
| Model        |                                                           |
| Backtests    |                                                           |
| Monitoring   |                                                           |
| Audit        |                                                           |
+--------------+-----------------------------------------------------------+
```

### 7.2 Left Sidebar

Width: 220 px desktop, collapsible to icons at 64 px.
Mobile: drawer navigation.

Navigation items:

- Command Center
- Trade Ideas
- Symbol Detail
- Portfolio & Risk
- Execution & TCA
- Signals
- Model & Features
- Backtests & Research
- Monitoring & Alerts
- Audit & Compliance
- Settings

Each item may show a small status marker:

- red: critical active issue
- amber: warning
- green: healthy
- gray: no data

### 7.3 Top Global Status Bar

Height: 48 to 56 px. Sticky. Always visible.

Fields:

- NAV
- Daily P&L
- Drawdown
- Gross exposure
- Net exposure
- Model source/age
- Data freshness
- Broker heartbeat
- Active breakers
- Last refresh

Interaction:

- Clicking `Model` opens Model & Features filtered to production model.
- Clicking `Data` opens Monitoring filtered to data health.
- Clicking `Broker` opens Execution/Monitoring filtered to broker.
- Clicking `Breakers` opens Monitoring filtered to breakers.

## 8. Visual Design System

### 8.1 Tone

Quiet, precise, institutional. Avoid decorative imagery, gradient-heavy backgrounds, animation-heavy transitions, or large empty cards.

### 8.2 Color Tokens

Use semantic colors, not strategy-specific random colors.

| Token | Hex | Usage |
|---|---:|---|
| `bg` | `#f4f6f8` | application background |
| `surface` | `#ffffff` | panels, tables |
| `surface-muted` | `#f8fafc` | table headers, section bands |
| `ink` | `#111827` | primary text |
| `muted` | `#667085` | secondary text |
| `line` | `#d7dde6` | borders |
| `buy` | `#0f7a54` | positive directional action |
| `sell` | `#b42318` | negative directional action |
| `watch` | `#8a6100` | watch state |
| `model` | `#285dad` | model-required/model metadata |
| `ok` | `#16794c` | healthy |
| `warning` | `#b76e00` | warnings |
| `critical` | `#b42318` | critical issues |
| `info` | `#2563a9` | informational |

Do not rely on red/green alone. Pair color with labels, icons, or shape.

### 8.3 Typography

- Base font: system UI or Inter.
- Base size: 13 to 14 px.
- Table text: 12 to 13 px.
- Section headings: 15 to 18 px.
- Page titles: 20 to 24 px.
- Numeric cells: tabular numbers.
- Letter spacing: 0.

### 8.4 Spacing And Density

- Page padding: 16 to 20 px desktop.
- Panel gap: 12 px.
- Panel radius: 6 px.
- Table row height: compact 36 px, comfortable 44 px.
- Icon button size: 32 px.
- Form controls: 32 to 36 px.

### 8.5 Status Badges

Badges are compact, all-caps optional, with strong contrast:

- BUY: green text on pale green background.
- SELL: red text on pale red background.
- WATCH: amber text on pale amber background.
- MODEL_REQUIRED: blue text on pale blue background.
- NO_DATA: gray text on light gray background.
- ERROR: red text on pale red background with error icon.

## 9. Buttons And Controls

### 9.1 Button Types

| Type | Appearance | Usage |
|---|---|---|
| Primary | filled green or blue, 32 to 36 px high | refresh, apply filters, run read-only query |
| Secondary | white, border, dark text | clear filters, export, copy link |
| Icon | square 32 px, icon only with tooltip | refresh, expand, download, settings |
| Warning | amber filled or outlined | acknowledge warning, request replay |
| Danger | red filled, confirmation required | halt, flatten, live destructive actions |
| Disabled | low opacity, visible tooltip reason | unavailable due to permission/state |

### 9.2 Required Buttons

Global:

- Refresh: fetch latest page data.
- Auto-refresh toggle: on/off with interval selector.
- Export: page-specific CSV/JSON/report export.
- Copy link: copies current filters and selected object.

Trade Ideas:

- Refresh Ideas: calls trade ideas API.
- Apply Filters: applies symbols/action/model/status filters.
- Reset: resets filters to defaults.
- Export Ideas: downloads current filtered table.
- Open Detail: row action, opens side drawer.

Monitoring:

- Acknowledge Alert: Phase 4 mutation, requires audit entry.
- Silence Alert: Phase 4, permissioned, time-bound.
- Open Source: navigates to relevant page.

Live Controls, later phase:

- Write HALT: danger button, requires typed confirmation.
- Flatten: danger button, requires typed confirmation and preflight summary.
- Verify Flat: read-only check button.

## 10. Common Components

### 10.1 Health Pill

Props:

- label
- state: `ok | warning | critical | unknown`
- value
- staleness_seconds
- onClick

Use for model, data, broker, breaker, drift.

### 10.2 Metric Tile

Props:

- label
- value
- unit
- delta
- severity
- sparkline optional
- timestamp optional

Avoid oversized cards. Metric tiles should be compact and scannable.

### 10.3 Data Table

Capabilities:

- server-side pagination
- server-side sorting
- server-side filtering
- column pinning
- column visibility
- density toggle
- row expansion
- row side drawer
- CSV export
- sticky header
- sticky first column for wide tables
- numeric right alignment
- status left alignment

### 10.4 Side Drawer

Width:

- 520 px default
- 720 px for explainability chain
- full-screen modal on mobile

Use for:

- trade idea details
- order details
- alert details
- audit entry details
- model run details

### 10.5 Timeline

Use for:

- signal timeline
- order lifecycle
- audit chain
- incident history

Each event has:

- timestamp
- title
- severity
- source
- details

## 11. Information Architecture

Primary pages:

1. Command Center
2. Trade Ideas
3. Symbol Detail
4. Portfolio & Risk
5. Execution & TCA
6. Signals
7. Model & Features
8. Backtests & Research
9. Monitoring & Alerts
10. Audit & Compliance
11. Settings

## 12. Page Specification: Command Center

### 12.1 Purpose

The Command Center is the first screen. It tells the operator what needs attention now.

### 12.2 Layout

Top row:

- NAV
- Daily P&L
- Drawdown
- Gross exposure
- Net exposure
- Positions
- Model health
- Data health

Middle row:

- Action counts
- Top actionable ideas
- Active warnings/errors
- Stage latency

Bottom row:

- Portfolio exposure chart
- Broker/circuit breaker status
- Paper/live divergence
- Recent audit/alert events

### 12.3 Components

#### Action Counts

Show counts for:

- BUY
- SELL
- WATCH
- MODEL_REQUIRED
- NO_DATA
- ERROR

Clicking a count navigates to Trade Ideas with that action filter.

#### Top Actionable Ideas

Compact table columns:

- symbol
- action
- target weight
- notional
- meta probability
- confidence
- top family
- reason

Rows sorted:

1. ERROR
2. MODEL_REQUIRED
3. BUY/SELL by absolute target weight
4. WATCH
5. NO_DATA

#### Stage Latency

Display as horizontal bars:

- data_fetch
- feature_compute
- signal_generation
- meta_inference
- sizing
- target_generation
- order_routing

Color:

- green under threshold
- amber near threshold
- red over threshold

### 12.4 Buttons

- Refresh Dashboard: primary.
- Open Trade Ideas: secondary.
- Export Summary: secondary.
- Toggle Auto Refresh: segmented control.

## 13. Page Specification: Trade Ideas

### 13.1 Purpose

This is the core daily decision table.

### 13.2 Table Columns

Required columns:

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
- Bet Size
- Strategy
- Reason
- Data Stage
- Feature Stage
- Signal Stage
- Model Stage
- Sizing Stage
- Target Stage
- Errors

Default visible columns:

- Symbol
- Action
- Target Weight
- Notional
- Est Qty
- Price
- Meta Probability
- Top Confidence
- Signals
- Top Family
- Latest Bar Time
- Reason

Hidden by default but available:

- feature rows
- calibrated probability
- average confidence
- bet size
- strategy
- all stage latencies
- errors

### 13.3 Filters

- Symbols input
- Action multi-select
- Model source
- Minimum absolute target weight
- Minimum meta probability
- Minimum confidence
- Has errors
- Has warnings
- Latest bar age
- Signal family
- Strategy

### 13.4 Row Detail Drawer

Tabs:

1. Explainability
2. Signals
3. Model
4. Sizing
5. Data
6. Related Alerts
7. JSON

#### Explainability Tab

Show chain:

```text
Bars -> Features -> Signals -> Meta Probability -> Bet Size -> Target Weight -> Action -> Risk/Execution State
```

Each step has:

- status
- timestamp
- count
- primary value
- latency
- warning/error

#### Signals Tab

Table:

- timestamp
- family
- side
- confidence
- metadata summary

Use small directional badges:

- Long
- Short
- Neutral

#### Model Tab

Fields:

- model source
- model version
- meta probability
- calibrated probability
- calibration status
- feature hash
- top feature contributions when available

#### Sizing Tab

Waterfall chart:

```text
Raw AFML size -> Kelly cap -> Vol adjusted -> ATR capped -> Risk budget -> Deployment multiplier -> Final size
```

Show constraints:

- max_single_position
- max_family_allocation
- max_gross_exposure
- max_crypto_allocation
- max_sector_exposure
- kelly_cap
- vol_scaling
- vrp_haircut
- atr_cap
- deployment_multiplier

#### Data Tab

Show:

- bars loaded
- latest bar timestamp
- latest price
- feature rows
- data freshness
- stage latencies

### 13.5 Buttons

- Refresh Ideas
- Apply Filters
- Reset Filters
- Save View
- Export CSV
- Open Drawer
- Copy Row Link

Phase 4:

- Mark Reviewed
- Request Paper Replay

## 14. Page Specification: Symbol Detail

### 14.1 Purpose

One-symbol forensic view.

### 14.2 Header

Show:

- symbol
- current action
- latest price
- target weight
- target notional
- meta probability
- top signal family
- latest bar time

### 14.3 Panels

#### Price And Bars

Chart:

- candlestick or OHLC bars
- volume bars below
- latest target/action marker
- optional entry/exit markers

Time ranges:

- 1D
- 5D
- 1M
- 3M
- 1Y
- Custom

#### Target Weight History

Line chart:

- target weight
- current weight
- zero line
- max single-position cap bands

#### Signal Timeline

Stacked event timeline by family:

- y-axis: family
- x-axis: time
- point color: side
- point opacity/size: confidence

#### Meta Probability Timeline

Line chart:

- meta probability
- calibrated probability
- threshold line at 0.5
- optional action markers

#### Feature Snapshot

Table:

- feature name
- current value
- training median
- z-score/shift
- drift status

#### Position State

Fields:

- side
- quantity
- average entry price
- current price
- market value
- unrealized P&L
- realized P&L
- return %
- stop loss
- take profit
- vertical barrier
- holding period

#### Orders And Fills

Table:

- order id
- side
- type
- quantity
- filled quantity
- status
- algo
- limit price
- average fill
- commission

### 14.4 Buttons

- Refresh Symbol
- Export Symbol Report
- Open Latest Trade Idea
- Open Orders
- Open Audit Trail

## 15. Page Specification: Portfolio & Risk

### 15.1 Purpose

Show current portfolio, target portfolio, and risk constraints.

### 15.2 Panels

#### Current Holdings Table

Columns:

- symbol
- side
- quantity
- current price
- market value
- current weight
- unrealized P&L
- realized P&L
- return %
- signal family

#### Target Portfolio Table

Columns:

- symbol
- strategy
- current weight
- target weight
- delta weight
- target shares
- trade required
- reason

#### Exposure Summary

Charts:

- gross/net exposure over time
- long/short exposure bars
- exposure by strategy/family
- exposure by asset class
- cap usage gauges

#### Risk Metrics

Show:

- portfolio volatility
- weighted asset volatility
- diversification ratio
- effective N
- estimated max drawdown
- gross exposure
- net exposure
- factor exposures
- risk decomposition

#### Correlation Matrix

Heatmap:

- families or symbols
- diverging color scale
- labels shown for small matrices
- tooltips for exact values

### 15.3 Buttons

- Refresh Portfolio
- Export Holdings
- Export Targets
- Show Only Trade Required
- Open Risk Constraints

Phase 4:

- Mark Risk Review Complete

## 16. Page Specification: Execution & TCA

### 16.1 Purpose

Inspect what happened after target generation.

### 16.2 Panels

#### Orders By Status

Status cards:

- pending
- submitted
- partial fill
- filled
- cancelled
- rejected
- expired

#### Orders Table

Columns:

- order id
- timestamp
- symbol
- side
- order type
- quantity
- filled quantity
- fill %
- status
- execution algo
- signal family
- meta probability
- limit price
- estimated cost
- rejection reason

#### Fills Table

Columns:

- fill id
- order id
- timestamp
- symbol
- price
- quantity
- commission
- exchange
- maker/taker

#### TCA Table

Columns:

- order id
- symbol
- side
- arrival price
- execution price
- slippage bps
- market impact bps
- timing cost bps
- total cost bps
- commission
- algo
- duration
- fill rate
- TWAP benchmark bps
- VWAP benchmark bps

#### TCA Charts

- Slippage histogram.
- Slippage by algo box plot.
- Slippage over time line chart.
- TWAP/VWAP benchmark comparison.
- Fill rate distribution.

#### Reconciliation

Table:

- symbol
- internal signed quantity
- broker signed quantity
- delta
- severity
- last checked

### 16.3 Buttons

- Refresh Execution
- Export Orders
- Export TCA
- Open Reconciliation Details

Phase 4:

- Verify Flat
- Cancel Stale Orders
- Flatten, danger gated

## 17. Page Specification: Signals

### 17.1 Purpose

Show signal health, behavior, and family-level diagnostics.

### 17.2 Panels

#### Active Families

List:

- ts_momentum
- cs_momentum
- mean_reversion
- stat_arb
- ma_crossover
- donchian_breakout
- futures_carry
- funding_rate_arb
- cross_exchange_arb
- vrp

#### Family Summary Table

Columns:

- family
- count
- average confidence
- long ratio
- short ratio
- neutral ratio
- latest timestamp
- latest symbols

#### Latest Signals Table

Columns:

- timestamp
- symbol
- family
- side
- confidence
- metadata

#### Signal Metadata Viewer

Metadata examples:

- lookbacks
- z-scores
- aggregate momentum
- half-life
- ADF p-value
- hedge ratio
- spread z-score
- EMA values
- Donchian channel levels
- carry
- funding rate
- spread bps
- VRP percentile
- regime modifier

#### Signal Correlation Matrix

Heatmap:

- signal family vs signal family
- rolling correlation window selector
- highlight negative diversifiers

#### Family Performance Attribution

Table/chart:

- family
- trades
- win rate
- total net P&L
- average P&L
- average holding period
- max drawdown

### 17.3 Buttons

- Refresh Signals
- Filter Family
- Export Signals
- Open Family Documentation

## 18. Page Specification: Model & Features

### 18.1 Purpose

Explain whether the model is ready and whether the input distribution is trustworthy.

### 18.2 Panels

#### Production Model Status

Fields:

- production model exists
- model version/run id
- model source
- MLflow tracking source label
- trained at
- model age
- training event count
- required gates status
- calibration status
- regime detector status

#### Probability Distribution

Histogram:

- meta probabilities
- calibrated probabilities
- optional selected symbol/family filters

#### Feature Importance

Charts:

- horizontal bar for top features
- grouped tabs for MDI, MDA, SFI, SHAP when available

#### Feature Drift Table

Columns:

- feature
- KL divergence
- KS statistic
- KS p-value
- mean shift
- variance ratio
- drifted
- recommendation

#### Drift Heatmap

Rows: features.
Columns: time buckets.
Color: drift intensity.

#### Retrain Status

Fields:

- last retrain
- next scheduled retrain
- emergency retrain state
- drift-triggered retrain cooldown
- latest validation recommendation

#### Backtest Gates Summary

Show:

- CPCV
- DSR
- PBO
- overall recommendation

### 18.3 Buttons

- Refresh Model Status
- Export Drift Report
- Open Backtest Gates

Phase 4:

- Request Retrain
- Request Emergency Retrain, permissioned
- Promote Model, permissioned and gate-backed

## 19. Page Specification: Backtests & Research

### 19.1 Purpose

Evaluate whether a strategy/model earned deployment.

### 19.2 Panels

#### Headline Metrics

Show:

- total return
- annualized return
- annualized vol
- Sharpe
- Sortino
- Calmar
- max drawdown
- win rate
- profit factor
- turnover
- cost drag bps

#### Equity And Drawdown

Charts:

- equity curve
- drawdown curve below equity
- mark worst drawdowns

#### Trade Log

Columns:

- entry timestamp
- exit timestamp
- symbol
- side
- entry price
- exit price
- size
- signal family
- gross P&L
- costs
- net P&L
- holding period
- meta probability
- return %

#### Monthly Returns

Heatmap:

- rows: year
- columns: month
- cell: return %

#### Regime Breakdown

Table:

- regime
- mean return
- volatility
- Sharpe
- trades

#### Strategy Breakdown

Table:

- signal family
- trades
- win rate
- average net P&L
- total net P&L
- average holding

#### Promotion Gates

Cards:

- CPCV positive paths
- DSR statistic and p-value
- PBO value
- pass/fail
- recommendation

### 19.3 Buttons

- Select Backtest Run
- Compare Runs
- Export Report
- Download Trade Log
- Open Promotion Details

## 20. Page Specification: Monitoring & Alerts

### 20.1 Purpose

Operational observability and incident triage.

### 20.2 Panels

#### Metrics Overview

Prometheus-style charts:

- portfolio NAV
- drawdown
- daily P&L
- gross exposure
- net exposure
- positions count
- target weights
- orders submitted/filled/rejected
- signal count
- meta probability histogram
- execution slippage
- stage latency
- stage cost
- bar formation rate
- data gap seconds
- feature drift KL
- feature freshness
- model age
- broker heartbeat
- circuit breaker triggers/state

#### Alert Feed

Columns:

- severity
- timestamp
- title
- source
- message
- status
- metadata summary

Filters:

- severity
- source
- acknowledged/unacknowledged
- time range
- symbol

#### Circuit Breaker State

Table:

- breaker type
- active
- severity
- action
- reason
- timestamp

#### Data Health

Charts:

- data gap time series
- bar formation rate by symbol
- latest bar age by symbol

### 20.3 Buttons

- Refresh Monitoring
- Acknowledge Alert, Phase 4
- Silence Alert, Phase 4
- Open Source Object
- Export Alerts

## 21. Page Specification: Audit & Compliance

### 21.1 Purpose

Reconstruct decisions and verify tamper-evident audit chain.

### 21.2 Panels

#### Audit Summary

Show:

- total entries
- chain verification status
- broken entries count
- last entry timestamp
- event counts by type

#### Audit Table

Columns:

- timestamp
- event type
- symbol
- model version
- context summary
- output summary
- signature status

Event types:

- signal_generated
- meta_label_predicted
- bet_sized
- order_submitted
- fill_received
- position_opened
- position_closed
- breaker_triggered
- phase_promoted
- rl_shadow_decision
- operator_action

#### Audit Entry Drawer

Tabs:

- Context
- Output
- Signature
- Related Objects

#### Chain Verification

Show:

- total checked
- ok/broken
- first broken entry
- previous signature
- current signature

### 21.3 Buttons

- Verify Chain
- Export CSV
- Export JSON
- Filter Event Type
- Filter Symbol

## 22. Settings

Settings should be limited at first.

Sections:

- display density
- timezone display preference
- default refresh intervals
- saved views
- visible columns
- API connection status
- user/role information

Do not expose credentials in Settings.

## 23. API Design

### 23.1 Standard Response Envelope

```ts
type ApiEnvelope<T> = {
  as_of: string;
  source: string;
  staleness_seconds?: number;
  warnings: string[];
  errors: string[];
  data: T;
};
```

### 23.2 Core Read Endpoints

```text
GET /api/v1/overview
GET /api/v1/trade-ideas
GET /api/v1/trade-ideas/{symbol}
GET /api/v1/symbols/{symbol}/bars
GET /api/v1/symbols/{symbol}/features/latest
GET /api/v1/symbols/{symbol}/signals
GET /api/v1/symbols/{symbol}/probabilities
GET /api/v1/portfolio/summary
GET /api/v1/portfolio/positions
GET /api/v1/portfolio/targets
GET /api/v1/portfolio/risk
GET /api/v1/execution/orders
GET /api/v1/execution/fills
GET /api/v1/execution/tca
GET /api/v1/execution/reconciliation
GET /api/v1/signals/summary
GET /api/v1/signals/latest
GET /api/v1/model/status
GET /api/v1/model/features/drift
GET /api/v1/model/features/importance
GET /api/v1/backtests/runs
GET /api/v1/backtests/{run_id}
GET /api/v1/monitoring/metrics/snapshot
GET /api/v1/alerts
GET /api/v1/audit/entries
GET /api/v1/audit/verify
GET /api/v1/stream/ops
```

### 23.3 Phase 4 Mutation Endpoints

```text
POST /api/v1/alerts/{alert_id}/ack
POST /api/v1/operator/review-complete
POST /api/v1/paper/replay
POST /api/v1/reports/export
POST /api/v1/live/halt
POST /api/v1/live/verify-flat
POST /api/v1/live/flatten
POST /api/v1/model/retrain-request
POST /api/v1/model/promote-request
```

All mutation endpoints require:

- authenticated user
- role authorization
- idempotency key
- audit entry
- confirmation payload for dangerous actions

### 23.4 Trade Idea DTO

```ts
type TradeIdea = {
  symbol: string;
  action: "BUY" | "SELL" | "WATCH" | "MODEL_REQUIRED" | "NO_DATA" | "ERROR";
  target_weight: number;
  target_notional: number;
  estimated_quantity: number | null;
  latest_price: number | null;
  latest_bar_at: string | null;
  bars_loaded: number;
  feature_rows: number;
  signal_count: number;
  top_signal_family: string | null;
  top_signal_side: -1 | 0 | 1 | null;
  top_signal_confidence: number | null;
  avg_signal_confidence: number | null;
  meta_probability: number | null;
  calibrated_probability: number | null;
  bet_size: number | null;
  strategy: string | null;
  reason: string;
  stage_latency_seconds: Record<string, number>;
  errors: string[];
};
```

### 23.5 Trade Idea Detail DTO

```ts
type TradeIdeaDetail = {
  idea: TradeIdea;
  chain: DecisionChainStep[];
  signals: SignalRow[];
  model: ModelInferenceDetail;
  sizing: SizingWaterfall;
  features: FeatureSnapshot;
  bars: BarSummary;
  related_alerts: AlertSummary[];
  related_audit_entries: AuditSummary[];
};
```

### 23.6 Decision Chain Step

```ts
type DecisionChainStep = {
  name: "bars" | "features" | "signals" | "model" | "sizing" | "target" | "risk" | "execution";
  status: "ok" | "warning" | "error" | "skipped" | "unknown";
  value: string | number | null;
  count?: number;
  timestamp?: string;
  latency_seconds?: number;
  message?: string;
};
```

## 24. Backend Module Interfaces

### 24.1 Existing Source Modules

| Web App Domain | Existing Module |
|---|---|
| Trade ideas | `src.ui.trade_ideas` |
| Current UI server | `src.ui.trade_ideas_app` |
| Live/paper bootstrap | `src.bootstrap` |
| Bars/features/signals DB | `src.data_engine.storage.database` |
| Feature store | `src.data_engine.storage.feature_store` |
| Signal battery | `src.signal_battery.orchestrator` |
| Model inference | `src.ml_layer.meta_labeler`, `src.bootstrap.ModelMetaPipeline` |
| Bet sizing | `src.bet_sizing.cascade` |
| Portfolio targets | `src.portfolio.multi_strategy`, `src.bootstrap.DirectTargetOptimizer` |
| Execution state | `src.execution.models`, `src.execution.storage` |
| TCA | `src.execution.tca` |
| Daily ops | `src.execution.daily_ops` |
| Preflight | `src.execution.preflight` |
| Live controls | `src.execution.live_trading` |
| Monitoring metrics | `src.monitoring.metrics` |
| Alerting | `src.monitoring.alerting` |
| Drift | `src.monitoring.drift_detector` |
| Backtest reports | `src.backtesting.report` |
| Promotion gates | `src.backtesting.gate_orchestrator` |
| Audit | `src.execution.audit_log` |
| RL shadow | `src.ml_layer.rl_shadow` |

### 24.2 Backend Service Layer

Create a thin service layer before API routes:

```text
src/web/services/trade_ideas_service.py
src/web/services/portfolio_service.py
src/web/services/execution_service.py
src/web/services/signals_service.py
src/web/services/model_service.py
src/web/services/backtest_service.py
src/web/services/monitoring_service.py
src/web/services/audit_service.py
```

Each service:

- accepts request filters
- calls existing repository modules
- normalizes response shape
- applies pagination/limits
- strips sensitive fields
- adds freshness metadata
- returns typed DTOs

## 25. Visualization Standards

### 25.1 General Chart Rules

- No 3D charts.
- No decorative gradients.
- No dark, low-contrast chart backgrounds.
- Use consistent time axis formatting.
- Show units on axes.
- Use tooltips for exact values.
- Downsample long time series server-side.
- Show stale/empty/error states explicitly.

### 25.2 Chart Types

| Data | Chart |
|---|---|
| NAV over time | Line chart |
| Drawdown | Negative area chart |
| Daily P&L | Bar chart |
| Exposure by family | Horizontal stacked bar |
| Target vs current weight | Diverging bar |
| Meta probability distribution | Histogram |
| Slippage distribution | Histogram |
| Feature drift over time | Heatmap |
| Correlation matrix | Heatmap |
| Sizing cascade | Waterfall |
| Signal events | Dot timeline |
| Order lifecycle | Timeline |
| Monthly returns | Calendar-like heatmap |

### 25.3 Financial Charts

Price chart:

- candlesticks when OHLC is available
- line chart fallback when only close is available
- volume subpanel
- trade/order markers
- latest bar marker
- stale-data badge

### 25.4 Heatmaps

Use diverging scale for correlations:

- negative: blue
- zero: white/light gray
- positive: red

Use sequential scale for drift:

- low: light gray/green
- medium: amber
- high: red

### 25.5 Waterfall Charts

Use for bet sizing:

- start with AFML size
- each cap/adjustment is a step down
- final size highlighted
- constraints listed next to chart

## 26. Performance Design

### 26.1 Frontend Performance Budgets

Target budgets:

- initial app shell under 250 KB compressed excluding chart chunks
- command center first meaningful render under 1.5 seconds on local network
- table interaction under 100 ms for visible rows
- page data refresh under 2 seconds for normal payloads
- no single table renders more than 200 visible DOM rows
- no unbounded client-side arrays

### 26.2 Frontend Techniques

- Code split by route.
- Lazy load heavy chart libraries.
- Use server-side sorting/filtering/pagination for large tables.
- Virtualize long tables.
- Memoize column definitions.
- Keep chart data immutable and downsampled.
- Use TanStack Query stale times per data class.
- Use SSE for event streams instead of aggressive polling.
- Use skeleton states for page-level loading.
- Use inline cell loading only for incremental detail fields.

### 26.3 Backend Performance

- Precompute overview aggregates.
- Add materialized views or cached summaries for heavy dashboard panels.
- Cache trade idea reports for short TTL when parameters match.
- Use DB indexes on timestamp, symbol, status, family, model version.
- Use pagination for audit/orders/fills/signals.
- Use time-window limits for charts.
- Return summary payloads by default; deep payloads only in detail endpoints.
- Avoid recomputing features/signals repeatedly when the same generated report can be shared across panels.

### 26.4 Data Refresh Tiers

| Data Type | Refresh |
|---|---:|
| Alert feed | SSE or 10 s |
| Broker heartbeat | 10 s |
| Circuit breaker state | 10 s |
| Portfolio summary | 15 to 30 s |
| Trade ideas | 1 to 5 min |
| Model status | 5 min |
| Backtest reports | manual or 10 min |
| Audit table | 30 s or manual |

## 27. Reliability And Empty States

Every page needs explicit states:

- Loading
- Loaded
- Empty
- Partial data
- Stale data
- Error
- Permission denied

Examples:

- `NO_DATA`: show latest known bar query details and DB freshness.
- `MODEL_REQUIRED`: show model source, registry status, and preflight link.
- `ERROR`: show failed stage and diagnostic message.
- Stale broker: show last heartbeat timestamp and age.

## 28. Security And Permissions

### 28.1 Roles

| Role | Read | Export | Acknowledge | Replay | Live Halt | Flatten | Promote |
|---|---|---|---|---|---|---|---|
| viewer | yes | no | no | no | no | no | no |
| operator | yes | yes | yes | yes | no | no | no |
| live_operator | yes | yes | yes | yes | yes | yes | no |
| quant_admin | yes | yes | yes | yes | no | no | yes |
| admin | yes | yes | yes | yes | yes | yes | yes |

### 28.2 Dangerous Action Confirmation

For halt/flatten/promote:

- modal shows action impact
- modal shows current state
- user must type exact confirmation phrase
- button remains disabled until phrase matches
- API receives idempotency key
- audit entry is written
- result is displayed

## 29. Accessibility

- Keyboard navigation for sidebar, tables, drawers, dialogs.
- Visible focus states.
- No color-only status.
- ARIA labels for icon buttons.
- Tooltips are supplemental, not required for critical meaning.
- Tables have column headers and row labels.
- Charts have textual summaries.
- All timestamps include timezone in tooltip or detail text.

## 30. Testing Strategy

### 30.1 Frontend Tests

- Component tests for badges, metric tiles, tables, drawers.
- Contract tests for API DTOs.
- Route tests for loading/empty/error states.
- Accessibility tests for dialogs and tables.
- Visual regression for primary pages.
- Performance tests for large tables.

### 30.2 Backend Tests

- DTO serialization tests.
- Permission tests.
- Sensitive field redaction tests.
- Pagination/filter/sort tests.
- Freshness metadata tests.
- Endpoint smoke tests using fake repositories.

### 30.3 End-to-End Tests

Scenarios:

- open Command Center
- filter Trade Ideas to BUY
- open symbol drawer
- inspect sizing waterfall
- navigate to Model & Features from model badge
- view alert and source object
- export a report
- verify audit chain

## 31. Implementation Plan

### Phase 1: Real Read-Only Operator App

Build:

- React app shell
- FastAPI BFF
- Command Center
- Trade Ideas table
- Trade Idea drawer
- Global status bar
- basic API contracts around existing trade ideas

Do not build:

- live mutation actions
- complex backtest UI
- admin settings

### Phase 2: Diagnostics And Drilldowns

Build:

- Symbol Detail
- Signals page
- Model & Features page
- Portfolio & Risk page
- feature drift visualizations
- signal family diagnostics

### Phase 3: Execution, Monitoring, Audit

Build:

- Execution & TCA
- Monitoring & Alerts
- Audit & Compliance
- Daily ops report view
- exports

### Phase 4: Controlled Operator Actions

Build:

- acknowledge alerts
- mark review complete
- request paper replay
- request retrain
- halt
- verify flat
- flatten
- gated promotion actions

## 32. Acceptance Criteria

The app is ready for Phase 1 production use when:

- Command Center loads from real APIs.
- Trade Ideas table exposes all existing report fields.
- Every trade idea opens a drawer with explainability chain.
- Model/data/broker/breaker status is visible globally.
- All critical warnings/errors are visible without hover.
- No secrets are exposed in network payloads.
- Tables are sortable/filterable and exportable.
- Empty/stale/error states are implemented.
- Read-only mode cannot mutate trading state.
- API responses are typed and documented.

## 33. Open Design Questions

- Should the first real app live in this repository or a separate frontend repository?
- Should the BFF be served by the same process as the trading service or a separate service?
- Which chart library should be standardized for all non-price charts?
- Do we need authenticated multi-user access in Phase 1, or is local/VPN access sufficient initially?
- Which backtest artifacts are persisted today and which need new storage?
- How much raw feature data should be exposed versus summarized?
- What is the minimum viable audit export format for compliance review?

## 34. Design Summary

The web app should be a complete operator cockpit. The first screen answers what matters now. Every recommendation must be explainable. Every trust signal must be visible. Every action and outcome must be traceable. The frontend should be dense, table-first, chart-supported, and built for repeated daily use by serious operators.
