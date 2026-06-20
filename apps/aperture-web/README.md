# Aperture — Web (Next.js)

**Your quant engine, in focus.** The production web app for the `wang_trading` engine —
decision support + monitoring, premium dark, Robinhood/Linear polish. Built to the canonical
[`docs/aperture_v1_design.md`](../../docs/aperture_v1_design.md) and **data-honest** per
[`docs/data_readiness.md`](../../docs/data_readiness.md): every screen renders only what the engine
produces today; everything else is a dignified *coming* state with its exact unlock condition — never
a faked number.

## Run

```bash
pnpm -C apps/aperture-web install
pnpm -C apps/aperture-web dev      # http://localhost:4180
```

`predev`/`prebuild` regenerate the design tokens from `tokens.json` (see below). Production:

```bash
pnpm -C apps/aperture-web build && pnpm -C apps/aperture-web start
```

## What's live vs coming (v1)

| Group | Screens | State |
|---|---|---|
| **Cockpit** | Overview · Markets · Trade Ideas (+ decision drawer) · Strategies | **LIVE** |
| **Engine** | Model & Features | **LIVE**, model-gated |
| (detail) | Symbol detail · Strategy detail | **LIVE** |
| **Portfolio / Research / Operate** | Portfolio · Execution · Backtests · Scenarios · Track Record · Monitoring · Preflight · Replay | **COMING** — visible, locked, with verbatim unlock conditions |

The honest gaps (regime fit, sizing cascade, SHAP, cost, track record, NAV/equity, per-strategy
performance, calibration/drift/RL) render via the **seven-state `DataState` system**
(`src/components/ui/honesty.tsx`) — Live / Loading / Empty / Stale / Error / **Not-yet-available** /
**Model-gated**. The locked nav rows and unlock copy are data-driven from a single
`screenReadiness` map (`src/lib/readiness.ts`), the machine-readable twin of `data_readiness.md`.

## Architecture

```
src/
├── app/                 App Router — one folder per route + globals.css + tokens.generated.css
│   ├── layout.tsx       root: AppShell (sidebar + trust bar + density)
│   ├── overview · markets · ideas · strategies · model        (LIVE)
│   ├── symbols/[symbol] · strategy/[id]                        (LIVE detail)
│   └── portfolio · execution · backtests · … (8 COMING via ComingScreen)
├── components/
│   ├── shell/           Sidebar, TrustBar (3 honest pills), AppShell
│   ├── ui/              primitives, Panel, honesty (DataState/ComingState/…)
│   ├── charts/          Sparkline, AreaChart (Recharts), CandleChart, MiniBars
│   ├── ideas/           IdeaDrawer (the decision chain)
│   └── ComingScreen.tsx locked-destination template
├── data/                api.ts (access layer) · envelope.ts (ApiEnvelope) · mock.ts
└── lib/                 colors (generated) · format · rng · readiness · density
```

## Tokens — single source of truth

`tokens.json` is the **only** place a color/space/radius is authored. `scripts/gen-tokens.mjs`
emits, in lockstep (committed; CI can `git diff --exit-code` after regenerating):

- `src/app/tokens.generated.css` — web CSS vars
- `src/lib/tokens.generated.ts` — chart hex (CSS vars don't resolve in SVG attributes)
- `../aperture-ios/Aperture/DesignSystem/ApertureTokens.generated.swift` — SwiftUI `Color`/`Tok`

## Wiring the live BFF

The screens bind to `ApiEnvelope<T>` (`src/data/envelope.ts`), mirroring `src/web/envelope.py`.
The data layer is mock-first — swap the bodies in `src/data/api.ts` for `fetch(\`${APERTURE_BFF_URL}/…\`)`
calls and the screens don't change. v1 honesty (what's null/coming) is encoded once, in `api.ts`.
The one required BFF addition is `request_id` on the envelope (for the Error state's copyable id).
