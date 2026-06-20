# Aperture — App Design (Web + iOS)

> **Status update (2026-06-13):** This document's vision is **folded into the canonical
> [aperture_v1_design.md](aperture_v1_design.md)**, now the single design of record (one product,
> web + iOS, data-honest v1). Kept here for the original consumer-product framing.

**A consumer-grade interface for the `wang_trading` quant engine.**

Version 1.0 · 2026-06-01 · Owner: YW

---

## 0. What this document is

This is the end-to-end UI/UX design for a real application — a **responsive web app** and a
**native iOS app** — that surfaces everything the engine produces: live trade ideas, the full
decision chain behind each one, all ten trading strategies in detail, portfolio & risk, backtests
& validation, the ML model internals, and rich visualization of every instrument (equities,
indexes, crypto, futures).

It is paired with a **working interactive prototype** at [`apps/prototype/`](../apps/prototype)
that implements the design with realistic mock data modeled on the real engine schemas. The
prototype lets you click through both platforms today; this doc is the specification behind it and
the plan for turning it into the production product.

### Relationship to existing design docs

| Doc | Direction | Status |
|-----|-----------|--------|
| [`docs/web_app_design_v2.md`](web_app_design_v2.md) | Dense **operator cockpit** (Bloomberg-terminal density, 15 pages, max information) | Spec + Sprint-1 scaffold in `apps/web` |
| [`docs/api_contracts_v2.md`](api_contracts_v2.md) | BFF API contracts (`ApiEnvelope<T>`, ~40 endpoints) | Partially wired |
| **This doc** | **Consumer-grade** reframe (clean, beautiful, great visualization) **+ a native iOS app** | Design + interactive prototype |

The two front-end directions are **complementary, not competing**: the v2 cockpit optimizes for a
power-user at a desk; Aperture optimizes for clarity, beauty, and mobility while preserving the same
explainability. Both read the **same BFF and the same engine outputs** — the API contracts are
shared. You can ship either, or both (cockpit on desktop, Aperture on web + phone).

---

## 1. Product vision & principles

**Vision.** *Your quant engine, in focus.* A single, beautiful surface where you can see what the
machine is doing, why, and what it's about to do — at your desk or on your phone — without losing
any of the rigor the engine was built with.

**Audience.** A single operator/owner (you). Single-tenant, single-portfolio, read-first with
gated, confirmed write actions. No multi-user, no onboarding funnels.

### Principles

1. **Explain the whole chain.** Every actionable number traces back through the pipeline:
   `ticks → bars → features → signals → regime → meta-probability → calibrated probability →
   bet size → target weight → action → cost → outcome → track record`. The UI's job is to make that
   chain legible, never to hide it.
2. **Consumer-grade, not dumbed-down.** Robinhood/Linear-level polish and whitespace — *and* the
   full depth, revealed through progressive disclosure (tap a row → drawer/detail with everything).
3. **Trust state is always visible.** Data freshness, model age, regime, drift, calibration recency,
   breakers, and mode (Paper/Live) are persistent. You should never wonder if you're looking at
   stale numbers.
4. **Show the diff.** Operators look for *what changed*, not absolute state. New ideas, flipped
   sides, weight moves, new errors are surfaced as changes.
5. **Safety by default.** Live, money-moving actions are gated, biometric-confirmed, and require a
   typed confirmation phrase. The default app is read-only.
6. **Platform-native, design-shared.** One design system → a responsive web app and a *native-feeling*
   iOS app (large titles, tab bar, push, widgets, haptics). Not a webview in a frame.
7. **Data-honest visualization.** Up/down color is consistent and never the only signal; charts
   never imply precision the data doesn't have.

---

## 2. What we visualize (grounded in the engine)

Everything below maps to real outputs in the repo. Field names are taken from the actual code so the
UI ↔ engine contract is unambiguous.

### 2.1 The decision chain (the spine of the product)

```
Data Engine → Feature Factory → Signal Battery → Labeling → ML Layer → Bet Sizing → Portfolio → Execution
  bars         ~200 features     10 families      triple-     meta-p +    5-layer      HRP /        orders +
 (TIB/VIB/$)   (9 families)      (Signal)         barrier     regime      cascade      multi-strat  TCA
```

### 2.2 Data domains, source schemas, and where they appear

| Domain | Source in repo | Key fields used by the UI | Web screens | iOS screens |
|--------|----------------|---------------------------|-------------|-------------|
| **Trade ideas** | `src/ui/trade_ideas.py` · `TradeIdea` | `action`, `target_weight`, `target_notional`, `meta_probability`, `calibrated_probability`, `regime_fit_score`, `bet_size`, `sizing_constraints`, `top_signal_family/side/confidence`, `reason`, `expected_cost_bps`, `top_shap`, `track_record_*`, `stage_latency` | Overview, Trade Ideas (+ drawer), Symbol | Home, Ideas, Idea, Symbol |
| **Signals** | `src/signal_battery/*` · `Signal` | `family`, `side`, `confidence`, family `metadata` (e.g. `z_scores`, `half_life`, `adf_pvalue`, `hedge_ratio`, `annualized_funding`) | Strategies, Strategy detail, Idea drawer | Strategies, Idea |
| **Strategies (10 families)** | `src/signal_battery/{momentum,mean_reversion,trend_following,volatility_signal,carry,stat_arb,cross_exchange_arb}.py` | thesis, params, regime-fit, Sharpe/win/contribution | Strategies (+ detail) | Strategies (+ detail) |
| **Backtest / validation** | `src/backtesting/*` · `walk_forward.compute_metrics`, `StrategyGate` | `sharpe`, `sortino`, `calmar`, `max_drawdown`, `win_rate`, `profit_factor`, `turnover`, `cost_drag_bps`, CPCV paths, Deflated-Sharpe p, PBO | Research & Backtests | (roadmap) |
| **Model & features** | `src/ml_layer/*` · `MetaLabeler`, `ProbabilityCalibrator`, `RegimeDetector`, `RLShadow` | AUC/Brier/ECE, calibration buckets, feature importance, drift KL, meta-prob histogram, RL agreement | Model & Features | (roadmap) |
| **Portfolio & risk** | `src/portfolio/*` · `MultiStrategyAllocator`, `FactorRiskModel` | positions, gross/net/long/short, factor `exposure`/`contribution`/`explained_var`, systematic vs idiosyncratic | Portfolio & Risk | (roadmap card on Home) |
| **Regime** | `src/ml_layer/regime_detector.py` | `label` + `{trending_up, trending_down, mean_reverting, high_volatility}` | global status + Overview | Home + status |
| **Live metrics** | `src/monitoring/metrics.py` (Prometheus) | NAV, daily P&L, drawdown, exposure, breaker state, slippage, freshness | status bar + Overview | status + Home |
| **Markets / instruments** | universe config + `data_engine/models.py` `Bar` | OHLCV, `vwap`, `dollar_volume`, `bar_type` | Markets, Symbol | Markets, Symbol |

### 2.3 Universe

Equities (AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, META, JPM…), Indexes (SPX, NDX, RUT, VIX),
Crypto spot + perps (BTC, ETH, SOL, AVAX…), Futures (ES, CL, GC…). Each instrument carries bar-type
metadata (`tib`/`vib`/`dollar`/`time`) so the UI can label what kind of bar it's drawing.

---

## 3. Information architecture

### 3.1 Web sitemap

```
Aperture (web)
├── Cockpit
│   ├── Overview            NAV, equity curve, key stats, regime, top ideas, movers
│   ├── Markets             all instruments, filter by class, sparklines → Symbol
│   ├── Trade Ideas         decision table + decision-chain drawer
│   └── Strategies          10 family cards → Strategy detail
├── Portfolio
│   ├── Portfolio & Risk    positions, exposure, allocation, factor risk, drawdown
│   ├── Research & Backtests equity vs benchmark, 3 gates, monthly heatmap, trade log
│   └── Model & Features    calibration, importance, drift, meta-prob, RL shadow
├── (detail routes)
│   ├── Symbol/{symbol}      price chart + "what the engine sees" + live features
│   └── Strategy/{id}        thesis, params, regime-fit, equity, active ideas
└── (roadmap) Execution & TCA · Monitoring · Scenarios · Track Record · Replay · Preflight · Settings
```

Layout: persistent left **sidebar** (grouped nav) + sticky **top bar** (page title, ⌘K search,
regime chip, freshness, NAV/day-P&L, alerts, account). Content max-width 1480px, generous padding.

### 3.2 iOS sitemap

```
Aperture (iOS)
├── Tab: Home          portfolio snapshot, equity, regime, top ideas, movers
├── Tab: Markets       search + class filter, instrument list → Symbol (push)
├── Tab: Ideas         summary chips, filter, idea cards → Idea (push)
├── Tab: Strategies    category filter, strategy cards → Strategy (push)
├── Push: Symbol       price chart, engine read, snapshot
├── Push: Idea         full decision chain, conviction, signals, cascade, SHAP
├── Push: Strategy     thesis, stats, equity, regime-fit, params
└── System extensions  Home-screen widgets · Lock-screen widget · Dynamic Island Live Activity ·
                       Push notifications · Face ID gate (roadmap)
```

Native patterns: large titles, bottom **tab bar**, navigation **push/pop** with swipe-back, sticky
nav header on detail, pull-to-refresh, haptics on actions.

### 3.3 Cross-platform parity

| Capability | Web | iOS (v1) | iOS (later) |
|-----------|:---:|:---:|:---:|
| Overview / portfolio snapshot | ✓ | ✓ | |
| Markets + Symbol detail | ✓ | ✓ | |
| Trade Ideas + full decision chain | ✓ (drawer) | ✓ (push) | |
| Strategies + detail | ✓ | ✓ | |
| Portfolio & Risk (positions, factor) | ✓ | snapshot | full |
| Research / Model deep dives | ✓ | — | condensed |
| Candlestick + volume | ✓ | area (candles later) | candles |
| Widgets / Live Activity / push | — | — | ✓ |
| Live (gated) actions | confirm phrase | Face ID + phrase | |

The phone is intentionally **decision-focused** (what's happening, what to do); the desktop is the
**research surface** (backtests, model internals, deep tables).

---

## 4. Design system

Brand: **Aperture** — a lens onto the engine. Default theme is a premium **dark** palette (markets
read best in dark; charts pop). Tokens are structured so a light theme can be added later.

### 4.1 Color tokens

Source of truth: [`apps/prototype/src/styles/theme.css`](../apps/prototype/src/styles/theme.css)
(CSS vars) mirrored as hex in [`src/lib/colors.ts`](../apps/prototype/src/lib/colors.ts) for charts.

| Token | Hex | Use |
|-------|-----|-----|
| `--bg-0` / `--bg-1` | `#0a0c10` / `#0e1116` | App backdrop / sections |
| `--surface-1/2/3` | `#14181f` / `#1a1f27` / `#222933` | Cards / hover / popovers |
| `--surface-inset` | `#0c0f14` | Chart wells, tracks |
| `--text-1/2/3` | `#eef1f6` / `#a3adbb` / `#6c7787` | Primary / secondary / tertiary |
| `--pos` / `--neg` | `#1ecb8b` / `#f6465d` | Up / down, buy / sell, P&L |
| `--accent` / `--accent-2` | `#7c5cff` / `#4d9fff` | Brand (violet→blue), interactive, ML |
| `--warn` / `--info` | `#f0a93b` / `#4d9fff` | Caution / informational |
| Regime: up / down / MR / HV | `#1ecb8b` / `#f6465d` / `#b07cff` / `#f0a93b` | LSTM regime classes |
| Category: Momentum / MeanRev / Trend / Vol / Carry / Arb | `#4d9fff` / `#b07cff` / `#1ecb8b` / `#f0a93b` / `#22d3ee` / `#f6679a` | Strategy taxonomy |

**Rules:** green/red reserved for *direction/P&L only*; brand violet/blue for interactive + model;
never encode meaning by color alone (pair with label/arrow/icon).

### 4.2 Typography

- **Inter** (UI) + **JetBrains Mono** (data: tickers, params, raw values).
- All numerics use **tabular figures** (`font-variant-numeric: tnum`) so columns align.
- Scale: page title 21–30px, section 15–18px, body 13–14px, eyebrow/label 10–11px uppercase.
- iOS swaps to the **SF system stack** inside the device frame for native feel; sizes map to Dynamic
  Type categories in the production SwiftUI build.

### 4.3 Spacing, radius, elevation, motion

- 4-pt spacing scale; card radius 13–20px, pills full-round, controls 8–11px.
- Soft layered shadows; a subtle brand glow on primary CTAs.
- Motion: 0.12–0.32s `cubic-bezier(.22,1,.36,1)`; `fade-up` on route change, `slide-in-right` for the
  drawer, gentle pulse for the live dot. Respect `prefers-reduced-motion`.

### 4.4 Component catalog

All implemented in the prototype (`apps/prototype/src/components`):

| Component | What it shows | Where |
|-----------|---------------|-------|
| **ActionPill** | BUY/SELL/WATCH/MODEL? as colored pill | tables, cards, drawers |
| **RegimeBar / RegimeChip** | 4-class regime probabilities (stacked bar + chip) | status, Overview, Home |
| **ProbRing** | meta / calibrated / regime-fit probability dial | drawer, Symbol, Idea |
| **Decision-chain stepper** | signals → meta → calibrated → fit → bet → target | Idea drawer / Idea screen |
| **Sizing-cascade bars** | 5-layer cascade with the *binding* constraint highlighted | drawer, Idea |
| **SHAP MiniBars** | signed top feature contributions per idea | drawer, Symbol, Idea |
| **Donut** | allocation by asset class, systematic vs idiosyncratic | Portfolio |
| **Sparkline** (SVG) | inline 30/46-bar trend, auto up/down color | tables, cards, movers |
| **AreaChart** (Recharts) | Robinhood-style gradient area + tooltip | NAV, strategy & symbol curves, drawdown |
| **CandleChart** (custom SVG) | OHLC candles + volume + last-price tag | Symbol detail |
| **Stat strip** | dense row of labeled metrics with dividers | Overview, Research, Strategy |
| **Decision drawer** | the full chain for one idea, slide-in | Trade Ideas |
| **Tables** (`.tbl`) | sortable-ready dense data (ideas, positions, trade log) | Ideas, Portfolio, Research |

### 4.5 Charting guidelines

- **Consumer hero charts** use smooth gradient **area** curves with a thin stroke and a hover tag.
- **Candlesticks** are available on Symbol detail (pro view) with volume and a live last-price label.
- **Reliability/calibration** plots show observed-vs-predicted against a y=x reference.
- **Distributions** (meta-prob, CPCV paths) use bars; positive/negative colored consistently.
- Axis chrome is minimal (right-aligned y, hidden x) for the consumer surfaces; research surfaces add
  gridlines and a benchmark overlay.

---

## 5. Web screens (spec)

Each screen lists **purpose · key components · data contract · interactions · states**. Endpoints
reference [`api_contracts_v2.md`](api_contracts_v2.md) (all wrapped in `ApiEnvelope<T>`).

### 5.1 Overview (`/overview`)
- **Purpose.** One-glance health: where the book stands, the market regime, and the highest-conviction
  actions right now.
- **Components.** NAV hero + timeframe area chart; 6-up stat strip (Sharpe, Max DD, Vol, Win, Gross/Net,
  positions); Top-ideas list; Regime panel; P&L-contribution bars; Movers grid.
- **Data.** `GET /overview` (NAV, day P&L, exposure, regime, action counts) + `GET /trade-ideas`
  (top N) + market sparks.
- **Interactions.** Timeframe toggle (1W…All); idea row → Symbol; "View all" → Trade Ideas; mover →
  Symbol.
- **States.** Skeleton hero + shimmer rows; stale banner if `staleness_seconds` over threshold.

### 5.2 Markets (`/markets`)
- **Purpose.** Browse the whole universe with live trend and a flag for active ideas.
- **Components.** Class filter chips; row list (glyph, symbol/name, 30-bar sparkline, price, 1D/1W/1M,
  market-cap/volume, active-idea dot).
- **Data.** `GET /markets` (or derived from bars + ideas). Each row links to `/symbols/{symbol}`.
- **Interactions.** Filter (All/Equities/Indexes/Crypto/Futures); row → Symbol. (Roadmap: sort,
  watchlists, search.)

### 5.3 Trade Ideas (`/ideas`) — flagship
- **Purpose.** The live decision table and the **full decision chain** behind any idea.
- **Components.** Summary tiles (Buy/Sell/Watch/Gross/Net); filter tabs; dense table (Symbol, Action,
  Strategy, Top conf, Meta p, Cal p, Regime fit, Target wt, Notional, Cost); **decision drawer**.
- **Drawer sections.** NL reason · decision-chain stepper · meta+calibrated rings + track record ·
  signals (family/side/conf + metadata) · 5-layer **sizing cascade** (binding highlighted) · **SHAP**
  contributions · pre-trade cost + constraints · pipeline latency · *Stage for execution* (gated).
- **Data.** `GET /trade-ideas`; per-symbol enrich `GET /symbols/{symbol}/shap`,
  `/cost-forecast`, `/track-record`.
- **Interactions.** Row → drawer; "Open" → Symbol; tab filter; (roadmap ⌘K `idea AAPL`).

### 5.4 Symbol Detail (`/symbols/{symbol}`)
- **Purpose.** Bridge "a stock/coin" ↔ "what the model thinks."
- **Components.** Price hero; Candles/Area + timeframe toggle (custom candlestick w/ volume); change
  strip (1W/1M/YTD, cap, bar type); **"What the engine sees"** (action, reason, meta/cal/regime rings,
  target, cost); **Why** (SHAP); **Live features** grid (GARCH vol, realized vol, RSI-14, order-flow
  imbalance, Kyle λ, VPIN, Amihud, Roll spread).
- **Data.** `GET /symbols/{symbol}/bars`, `/microstructure`, `/shap`, + the symbol's idea.

### 5.5 Strategies (`/strategies`) + Strategy Detail (`/strategy/{id}`)
- **Purpose.** All ten families at a glance and each one in depth.
- **Cards.** name, category dot, status (Live/Shadow/Paused), equity sparkline, Sharpe/Win/P&L-share,
  allocation bar.
- **Detail.** thesis + source (book/author) + asset classes; stat strip (Sharpe, win, P&L share, YTD,
  allocation, avg hold); equity area; **regime-fit** bars; **parameters** table; **active ideas** from
  this family.
- **Data.** `GET /signals/family-*`, strategy registry, filtered `/trade-ideas`.

### 5.6 Portfolio & Risk (`/portfolio`)
- **Components.** Exposure (gross/net/long/short + split bar + Sharpe/Sortino/MaxDD); allocation donut
  by asset class; positions table (side, strategy, qty, entry, mark, weight, notional, unreal P&L,
  day); **factor-risk model** (systematic vs idiosyncratic donut + PCA factor exposures); drawdown
  area.
- **Data.** `GET /portfolio` (positions, exposure), `/portfolio/factor-decomposition`.

### 5.7 Research & Backtests (`/research`)
- **Components.** metrics strip; walk-forward **equity vs SPX benchmark**; the **three promotion
  gates** (CPCV %, Deflated-Sharpe p, PBO) with pass badges + viz (CPCV path distribution, threshold
  bars); **monthly-returns heatmap**; performance-by-regime; trade log.
- **Data.** `GET /backtests/*`, `StrategyGate` output.

### 5.8 Model & Features (`/model`)
- **Components.** model header (version, type, trained/age, AUC/Brier/ECE); **calibration reliability**
  plot; **meta-probability histogram**; **feature importance** (colored by family) + legend; **feature
  drift** (KL with ok/warn/alert); **RL shadow** card (agreement, shadow vs live Sharpe, auto-revert);
  **retrain timeline** (promoted/rejected, gated by the 3 checks).
- **Data.** `GET /model/calibration`, `/model/regime`, `/model/rl-shadow`, `/model/retrain-history`.

### 5.9 Roadmap screens
Execution & TCA · Monitoring & Alerts (freshness heatmap, alert feed) · Scenarios & Stress · Track
Record · Replay/Time-Travel · Preflight & Go-Live · Settings. These exist in the v2 cockpit spec and
slot into the same IA when wired.

### 5.10 Screen states (all screens)
Loading = skeletons/shimmer; Empty = explanatory copy (e.g. "No active idea above the entry gate");
Error = inline with retry + request-id; **Stale** = amber banner driven by `staleness_seconds` /
`source_freshness`; Live actions = confirm modal with typed phrase.

---

## 6. iOS app (spec)

A **native SwiftUI** app sharing the Aperture design tokens. The prototype demonstrates the layouts;
production is SwiftUI (not a webview).

### 6.1 Why native iOS
Glanceable home, push notifications for alerts/breakers, Home-screen & Lock-screen widgets, a Dynamic
Island Live Activity for the trading session, Face ID to gate sensitive views/actions, smooth
60–120fps charts, offline cache. None of these are possible (well) in a wrapped web view.

### 6.2 Screens (in the prototype)
- **Home.** Large title; NAV hero + area chart + timeframe; Sharpe/MaxDD/Win cards; **Market regime**
  card; **Top ideas** list; **Movers** horizontal scroller.
- **Markets.** Search + class filter; instrument rows (glyph, sparkline, price, change) → Symbol.
- **Ideas.** Buy/Sell/Watch chips; filter; idea cards (action, symbol, cal p, target, cost, regime-fit
  bar) → Idea.
- **Strategies.** Category filter; strategy cards (sparkline + Sharpe/Win/P&L/Alloc) → Strategy.
- **Symbol (push).** Price hero + area chart + timeframe; change cards; **"What the engine sees"**
  (rings + reason + target/cost); snapshot grid.
- **Idea (push).** Reason; **decision-chain** grid; conviction rings + track record; signals; sizing
  cascade; SHAP; *Open symbol* / *Stage for execution*.
- **Strategy (push).** Thesis; stat grid; equity; regime-fit; parameters.

### 6.3 Native patterns & system integration (production)
- **Navigation.** `TabView` (Home/Markets/Ideas/Strategies) + `NavigationStack` push/pop, swipe-back,
  sticky nav header, large titles collapsing on scroll.
- **Refresh.** Pull-to-refresh; background refresh pulls the publisher's `trade_ideas.json` cadence.
- **Widgets.** Home-screen widgets: NAV + day P&L; Regime; Top idea. Lock-screen complication: NAV
  delta. (WidgetKit, App Group shared cache.)
- **Live Activity / Dynamic Island.** Active trading session: NAV, day P&L, breaker state — live on
  the Lock Screen and in the Island.
- **Notifications.** Push for new high-conviction ideas, regime flips, drift alerts, breaker trips
  (mirrors the engine's Telegram alerting). Deep-link into the relevant screen.
- **Security.** Face ID to open the app / reveal P&L / confirm any live action; typed confirmation
  phrase for halt-clear and live capital (mirrors the engine's 18-check preflight philosophy).
- **Dynamic Type & VoiceOver.** Respect text-size settings; numeric values get spoken labels;
  color-blind-safe by pairing color with arrows/labels.
- **iPad / Apple Watch.** iPad split-view (master list + detail) and a Watch glance (NAV, regime, top
  idea) are natural later extensions.

---

## 7. Data & API contracts

- **Envelope.** Every response is `ApiEnvelope<T>` with `as_of`, `source`, `staleness_seconds`,
  `source_freshness`, `model_version`, `regime`, `warnings`, `errors`, `data` — the UI's persistent
  trust state binds directly to these fields.
- **Read endpoints.** `/overview`, `/trade-ideas`(+`/{symbol}`), `/symbols/{symbol}/{bars,
  microstructure,shap,track-record}`, `/portfolio`(+`/factor-decomposition`), `/signals/family-*`,
  `/backtests/*`, `/model/{calibration,regime,rl-shadow,retrain-history}`. (See `api_contracts_v2.md`.)
- **Real-time.** SSE `/stream/ops` and `/stream/diff` for live status + the "what changed" rail; the
  `TradeIdeaPublisher` already writes `trade_ideas.json` to tmpfs on a cadence, which the BFF serves
  with a staleness guard — the iOS app polls/streams the same.
- **Write endpoints (gated).** `POST /preflight/clear-halt` (header `X-Confirmation-Phrase: CLEAR
  HALT`), `/model/calibration/refit`, etc. — always role-checked + typed confirmation; on iOS also
  Face ID.
- **Caching.** Web uses React Query (`staleTime` ~30s). iOS caches last-good payloads in an App Group
  for instant cold-launch + widgets, revalidating in the background.

---

## 8. Key interaction patterns

- **Decision-chain drill-down.** The core gesture: tap any idea/row → drawer (web) / push (iOS)
  exposing signals → meta → calibrated → regime-fit → cascade → SHAP → cost. This is *the* product.
- **The diff rail.** Surface mutations since last refresh (new ideas, side flips, weight moves
  ≥25bps, new errors). Web: a strip/rail; iOS: a "What changed" card + push.
- **Command palette (web).** ⌘K to jump: `idea NVDA`, `strategy ts_momentum`, `symbol BTC`,
  `replay 2026-05-15T13:00Z`.
- **Safety confirmations.** Any live action → modal with consequence summary + typed phrase (+ Face ID
  on iOS). Read-only is the default everywhere.

---

## 9. Accessibility & performance

- **Contrast.** Body/secondary text and semantic colors meet WCAG AA on the dark surfaces; verify any
  new pairing.
- **Not-color-alone.** Direction always paired with arrow/label; regimes with labels; gates with
  Pass/Fail text + icon.
- **Dynamic Type / VoiceOver (iOS).** Scale type; expose numeric semantics; logical focus order.
- **Performance budgets.** First meaningful paint < 1.5s on web; charts virtualized/animated off for
  large series; iOS keeps scrolling at 60–120fps by capping on-screen candle counts and reusing cells.

---

## 10. Build roadmap

| Phase | Scope | Status |
|------|-------|:---:|
| **0 — Prototype** | Web (9 screens) + iOS (7 screens) with mock data on real schemas; design system; charts | ✅ in `apps/prototype` |
| **1 — Web, live data** | Wire Overview, Trade Ideas (+drawer), Symbol, Strategies to the BFF; status bar from `ApiEnvelope` | next |
| **2 — Web, research** | Portfolio & Risk, Research & Backtests, Model & Features from real endpoints | |
| **3 — iOS native** | SwiftUI app from shared tokens; Home/Markets/Ideas/Strategies/details; pull-to-refresh; cache | |
| **4 — iOS system** | Widgets, Lock-screen, Dynamic Island Live Activity, push (mirror Telegram alerts), Face ID | |
| **5 — Actions & advanced** | Gated live actions, Scenarios & Stress, Track Record, Replay, Preflight | |

**Tech recommendations.**
- *Web:* keep the prototype stack (React 18 + TypeScript + Vite + Recharts) — it already matches
  `apps/web`. Promote shared primitives into a small package; swap mock data for the React-Query hooks
  already present in `apps/web` (`useTradeIdeas`, etc.).
- *iOS:* native **SwiftUI** + **Swift Charts**, consuming the same BFF. Export the design tokens as a
  shared JSON (`theme.css` → `tokens.json`) so web and iOS stay in lockstep. (React Native is an option
  if you want one codebase, but native SwiftUI gives the best charts/widgets/Live-Activity story.)
- *Tokens as source of truth:* generate platform files (CSS vars, Swift `Color` extension) from one
  token JSON to guarantee parity.

---

## 11. The prototype

**Location:** [`apps/prototype/`](../apps/prototype) · **Run:**

```bash
pnpm -C apps/prototype install
pnpm -C apps/prototype dev      # http://localhost:5174  → toggle "Web / iOS" in the top bar
```

**What's real vs mocked.** Layouts, design system, components, charts, and interactions are
production-shaped. Data is **mock** but modeled field-for-field on the engine schemas
([`src/data/mock.ts`](../apps/prototype/src/data/mock.ts)) — `TradeIdea`, `Signal` families, backtest
metrics + 3 gates, calibration buckets, factor model, regime. Series are seeded (stable across
reloads) via [`src/lib/rng.ts`](../apps/prototype/src/lib/rng.ts).

**File map.**

```
apps/prototype/src/
├── App.tsx                 platform switch (Web ⇄ iOS)
├── data/mock.ts            dataset modeled on real engine schemas
├── lib/                    format, colors, seeded RNG, useWidth
├── components/
│   ├── charts/             AreaChart, CandleChart, Sparkline, MiniBars
│   ├── ui/                 primitives (pills, rings, donut, segmented…), Panel
│   ├── Icon.tsx, IdeaDrawer.tsx, PrototypeBar.tsx
├── web/                    Sidebar, TopBar, WebApp + pages/ (9 screens)
└── ios/                    IPhoneFrame, IOSApp (tab bar) + screens/ (7 screens)
```

**Screens implemented.** Web: Overview · Markets · Trade Ideas (+drawer) · Symbol (candles) ·
Strategies (+detail) · Portfolio & Risk · Research & Backtests · Model & Features. iOS: Home ·
Markets · Ideas (+detail) · Strategies (+detail) · Symbol.

---

*Naming, palette, and scope are all easy to change — “Aperture” is a placeholder brand. This doc plus
the prototype are meant to be iterated on together.*
