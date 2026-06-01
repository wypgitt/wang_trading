# Aperture — UI prototype (web + iOS)

An interactive, clickable prototype of a **consumer-grade** interface for the `wang_trading` quant
engine. One codebase renders both a **responsive web app** and a **native-style iOS app** (toggle
"Web / iOS" in the top bar).

Design spec: [`docs/aperture_app_design.md`](../../docs/aperture_app_design.md).

## Run

```bash
pnpm -C apps/prototype install
pnpm -C apps/prototype dev      # → http://localhost:5174
```

Build / type-check:

```bash
pnpm -C apps/prototype type-check
pnpm -C apps/prototype build
```

## What it is (and isn't)

- **Real:** layout, design system, components, charts (area / candlestick / sparkline / reliability),
  navigation, and the decision-chain drill-down.
- **Mock:** all data — but modeled field-for-field on the engine's real schemas (`TradeIdea`,
  `Signal` families, backtest metrics + the 3 promotion gates, calibration, factor model, regime).
  See [`src/data/mock.ts`](src/data/mock.ts). Series are seeded so charts are stable across reloads.

This is a **design artifact**, not wired to the BFF. Production wiring is Phase 1 in the design doc.

## Screens

**Web:** Overview · Markets · Trade Ideas (+ decision drawer) · Symbol (candlesticks) · Strategies
(+ detail) · Portfolio & Risk · Research & Backtests · Model & Features.

**iOS:** Home · Markets · Ideas (+ detail) · Strategies (+ detail) · Symbol.

## Stack

React 18 · TypeScript · Vite · Recharts (analytics charts) + custom SVG (candles/sparklines).
Matches `apps/web` so primitives can be promoted into the real app.

## Layout

```
src/
├── App.tsx              platform switch
├── data/mock.ts         dataset modeled on engine schemas
├── lib/                 format · colors · seeded rng · useWidth
├── components/charts/   AreaChart · CandleChart · Sparkline · MiniBars
├── components/ui/        primitives · Panel
├── web/                 Sidebar · TopBar · pages/
└── ios/                 IPhoneFrame · IOSApp · screens/
```
