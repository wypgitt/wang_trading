# Aperture — v1 Design (Canonical)

**The single converged design for the Aperture trading app — web + native iOS, built on the `wang_trading` engine.**

Version 1.0 · 2026-06-13 · Owner: YW · Status: **canonical**

---

## 0. What this document is

This is the **one** design of record for Aperture's first shippable version. It supersedes the
fork between the consumer app and the operator cockpit, and it is **honest about data**: every
screen is designed only for what the engine produces today (per
[data_readiness.md](data_readiness.md)), with everything else held in a dignified "coming" state
that shows its exact unlock condition — never a faked number.

### Relationship to the earlier docs

| Doc | Role now |
|-----|----------|
| **This doc** | **Canonical v1 design** — one product, one design language, data-honest screens |
| [aperture_app_design.md](aperture_app_design.md) | Source consumer-product vision (folded in here) |
| [web_app_design_v2.md](web_app_design_v2.md) | **Depth catalog** — the field-level detail that lands progressively as data arrives |
| [web_app_design.md](web_app_design.md) | Superseded (early operator-cockpit draft) |
| [data_readiness.md](data_readiness.md) | Ground truth for what's live vs coming |

### Product decisions (locked)

- **One product = Aperture.** Premium dark, Robinhood/Linear polish. The dense cockpit is not a
  separate app — its depth folds in as drawers and an optional **pro density** toggle. One design
  language, one design system, one BFF.
- **Web + native iOS**, kept in lockstep by a single `tokens.json` source of truth (→ web CSS vars
  + SwiftUI `Color`). Both consume the same FastAPI BFF.
- **Decision support + monitoring** (read-first). No live in-app execution in v1.
- **Simple surface, deep on demand.** The Overview answers "what now" in a glance; one tap reveals
  the full decision chain. "Huge but simple" is achieved by progressive disclosure, never two apps.

---

## Part I — Foundations

## Design System & Token Source-of-Truth

Aperture has **one** design language. The premium dark palette in `apps/prototype/src/styles/theme.css` is canonical; the v2 cockpit's light, "no-gradients / no-dark-backgrounds" system (`docs/web_app_design_v2.md` §8.2, §25 — regime tokens `#0f7a54 / #b42318 / #7c3aed / #b76e00`) is **retired**. The cockpit's *density and depth* survive — folded into Aperture as drawers and a "pro density" toggle — but its visual system does not. Web (CSS vars), iOS (SwiftUI `Color`), and charts (hex, since CSS vars don't resolve inside SVG presentation attributes) all derive from a single `tokens.json`.

### 1. Token set (canonical)

All values below are the locked source-of-truth. Hex mirrors live today in `apps/prototype/src/lib/colors.ts` (`C`, `CAT`, `REGIME_HEX`, `ASSET_TINT`); CSS vars in `theme.css`. `tokens.json` (§3) replaces both as the generator input.

**Surfaces** (near-black, faint cool tint — never pure `#000`):
| Token | Hex | Use |
|---|---|---|
| `bg.0` | `#0a0c10` | App backdrop |
| `bg.1` | `#0e1116` | Page sections, drawer body |
| `surface.1` | `#14181f` | Cards (`.card`) |
| `surface.2` | `#1a1f27` | Hover, elevated, chips |
| `surface.3` | `#222933` | Popovers, inputs, active segment |
| `surface.inset` | `#0c0f14` | Chart wells, bar tracks, donut rail |

**Hairlines:** `border` `rgba(255,255,255,0.07)` · `border.strong` `rgba(255,255,255,0.13)` · `grid` `rgba(255,255,255,0.055)` (chart gridlines only).

**Text:** `text.1` `#eef1f6` (primary) · `text.2` `#a3adbb` (secondary, `.muted`) · `text.3` `#6c7787` (tertiary/labels, `.dim`) · `text.inverse` `#0a0c10` (on accent fills).

**Semantic — direction/P&L only** (rule: green/red are *never* used for non-directional meaning, and never the sole signal — always paired with arrow/label/icon):
`pos` `#1ecb8b` · `pos.dim` `#15a673` · `pos.soft` `rgba(30,203,139,0.13)` · `neg` `#f6465d` · `neg.dim` `#d23a4e` · `neg.soft` `rgba(246,70,93,0.13)`.

**Brand (violet→blue) — interactive + ML/model only:** `accent` `#7c5cff` · `accent.2` `#4d9fff` · `accent.soft` `rgba(124,92,255,0.15)` · gradients `accent.grad` `linear-gradient(135deg,#7c5cff,#4d9fff)`, `pos.grad` `linear-gradient(135deg,#1ecb8b,#4d9fff)` (hero area fills, primary CTA only).

**Status:** `warn` `#f0a93b` / `warn.soft` · `info` `#4d9fff` / `info.soft`.

**Regime (4 LSTM classes):** `regime.up` `#1ecb8b` · `regime.down` `#f6465d` · `regime.mr` `#b07cff` · `regime.hv` `#f0a93b`. Labels: `Trending ↑ / Trending ↓ / Mean-revert / High vol`.

**Action pills:** `buy` `#1ecb8b` · `sell` `#f6465d` · `watch` `#4d9fff` · `neutral` `#8a93a3` (each with a `.soft` 0.14-alpha fill). `MODEL_REQUIRED` and `NO_DATA` both map to `neutral`.

**Strategy category** (taxonomy color, distinct from direction): `Momentum` `#4d9fff` · `Mean Reversion` `#b07cff` · `Trend` `#1ecb8b` · `Volatility` `#f0a93b` · `Carry` `#22d3ee` · `Arbitrage` `#f6679a`.

**Asset tint** (glyph backgrounds): `equity` `#4d9fff` · `index` `#b07cff` · `crypto` `#f0a93b` · `future` `#22d3ee`.

**Typography:** `font.sans` Inter (UI) · `font.mono` JetBrains Mono (tickers, params, raw values, all numerics). Body 14px / line-height 1.45 / `letter-spacing -0.011em`; ligature features `cv11, ss01`. Headings weight 650, `letter-spacing -0.02em`. **All numbers carry `font-variant-numeric: tabular-nums` (`.num` / `.mono`)** so columns align. Scale: page title 21–30, section 15.5–18, body 13–14, eyebrow/label 11 uppercase (`letter-spacing 0.08em`). iOS swaps to the SF system stack and maps sizes to Dynamic Type categories; mono numerics use SF Mono.

**Spacing (4-pt):** `sp.1`=4 … `sp.4`=16 `sp.5`=20 `sp.6`=24 `sp.8`=32 `sp.10`=40 `sp.12`=48. Layout: `sidebar.w` 244, content max-width 1480.

**Radius:** `r.xs`=6 `r.sm`=9 `r.md`=13 `r.lg`=18 `r.xl`=24 `r.pill`=999.

**Elevation:** `shadow.1` `0 1px 2px rgba(0,0,0,.4)` (cards) · `shadow.2` `0 6px 24px rgba(0,0,0,.35)` (lift/hover) · `shadow.pop` `0 18px 50px rgba(0,0,0,.55)` (drawer/popover) · `shadow.glow` `0 8px 32px rgba(124,92,255,.25)` (primary CTA only).

**Motion:** duration `0.12–0.32s`, easing `cubic-bezier(.22,1,.36,1)`. Named: `fade-up` (route change, 0.32s), `slide-in-right` (drawer, 0.26s), `fade-in` (backdrop, 0.18s), `pulse-soft` (live dot, 1.8s loop). All motion respects `prefers-reduced-motion`.

### 2. tokens.json → web + iOS (single source of truth)

`tokens.json` is the only place a value is authored. Two generators emit platform files in lockstep; CI fails if generated output drifts from committed.

**Schema** (grouped, primitives + alias layer so semantics can re-point without touching call sites):
```jsonc
{
  "$schema": "./tokens.schema.json",
  "color": {
    "bg":      { "0": "#0a0c10", "1": "#0e1116" },
    "surface": { "1": "#14181f", "2": "#1a1f27", "3": "#222933", "inset": "#0c0f14" },
    "border":  { "_": "rgba(255,255,255,0.07)", "strong": "rgba(255,255,255,0.13)" },
    "text":    { "1": "#eef1f6", "2": "#a3adbb", "3": "#6c7787", "inverse": "#0a0c10" },
    "pos": { "_": "#1ecb8b", "dim": "#15a673", "soft": "rgba(30,203,139,0.13)" },
    "neg": { "_": "#f6465d", "dim": "#d23a4e", "soft": "rgba(246,70,93,0.13)" },
    "accent": { "_": "#7c5cff", "2": "#4d9fff", "soft": "rgba(124,92,255,0.15)" },
    "warn": { "_": "#f0a93b" }, "info": { "_": "#4d9fff" },
    "regime": { "up": "#1ecb8b", "down": "#f6465d", "mr": "#b07cff", "hv": "#f0a93b" },
    "action": { "buy": "#1ecb8b", "sell": "#f6465d", "watch": "#4d9fff", "neutral": "#8a93a3" },
    "category": { "momentum": "#4d9fff", "meanRev": "#b07cff", "trend": "#1ecb8b",
                  "vol": "#f0a93b", "carry": "#22d3ee", "arb": "#f6679a" },
    "asset": { "equity": "#4d9fff", "index": "#b07cff", "crypto": "#f0a93b", "future": "#22d3ee" }
  },
  "gradient": { "accent": "135deg,#7c5cff,#4d9fff", "pos": "135deg,#1ecb8b,#4d9fff" },
  "radius":  { "xs": 6, "sm": 9, "md": 13, "lg": 18, "xl": 24, "pill": 999 },
  "space":   { "1": 4, "2": 8, "3": 12, "4": 16, "5": 20, "6": 24, "8": 32, "10": 40, "12": 48 },
  "shadow":  { "1": "0 1px 2px rgba(0,0,0,.4)", "2": "0 6px 24px rgba(0,0,0,.35)",
               "pop": "0 18px 50px rgba(0,0,0,.55)", "glow": "0 8px 32px rgba(124,92,255,.25)" },
  "font":    { "sans": "Inter, -apple-system, system-ui, sans-serif",
               "mono": "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace" },
  "motion":  { "ease": "cubic-bezier(.22,1,.36,1)", "fast": 120, "base": 180, "slow": 320 }
}
```

**Generators** (one tiny build step, run in CI + pre-commit):
- **Web** → `theme.css`: flatten dot-paths to kebab CSS vars (`color.surface.1`→`--surface-1`, `color.border._`→`--border`, `color.pos.soft`→`--pos-soft`, `gradient.accent`→`--accent-grad: linear-gradient(<v>)`). This file already exists by hand today; the generator makes it derived, not authored.
- **Web charts** → `colors.ts`: emit the `C`, `CAT`, `REGIME_HEX`, `ASSET_TINT` objects (hex required because CSS vars don't resolve in SVG attributes — already the convention in the prototype).
- **iOS** → `ApertureTokens.swift`: a `Color` extension + `enum Tokens` for spacing/radius. Hex strings become `Color(hex:)`; `rgba()` alphas become `.opacity()`; gradients become `LinearGradient`. Example:
```swift
extension Color {
  static let bg0      = Color(hex: 0x0A0C10)
  static let surface1 = Color(hex: 0x14181F)
  static let pos      = Color(hex: 0x1ECB8B)
  static let accent   = Color(hex: 0x7C5CFF)
  static let regimeMr = Color(hex: 0xB07CFF)
}
enum Radius { static let md: CGFloat = 13; static let lg = 18; static let pill = 999 }
enum Space  { static let s4: CGFloat = 16; static let s6 = 24 }
```

**Rule:** no raw hex, no magic px in any component, on either platform. Lint forbids hex literals outside `colors.ts` / `ApertureTokens.swift`. Both platforms consume the same FastAPI BFF (`src/web`); tokens are the only other shared contract.

### 3. Component catalog (locked)

Names match `apps/prototype/src/components`. Most already exist; the two new states (`ComingSoon`, `DataUnavailable`) are the honesty primitives that make a data-thin v1 shippable without faking numbers.

| Component | Props (canonical) | Renders | Screens |
|---|---|---|---|
| **ActionPill** | `action: BUY\|SELL\|WATCH\|MODEL_REQUIRED\|NO_DATA` | colored pill via `.pill-{buy,sell,watch,neutral}`; MODEL_REQUIRED→"Model?", NO_DATA→"No data" | Overview, Ideas, drawer, Symbol |
| **RegimeChip** | `regime: {label, probabilities}` | dot + top-class label + % (regime color) | top bar, Overview, Home — **gated: regime is null today → render `DataUnavailable` chip variant** |
| **RegimeBar** | `probs, height?` | 4-class stacked probability bar | Overview/Home (same gate) |
| **ProbRing** | `value: number\|null, size?, label?` | SVG dial; color steps `≥.65 pos / ≥.55 warn / else text3`; `—` when null | drawer (Meta, Calibrated), Symbol, iOS Idea — **model-gated** |
| **DecisionStepper** | `steps: {label,value,tone}[]` | horizontal `›`-joined chain (Signals→Meta→Cal→Fit→Bet→Target) | Ideas drawer, iOS Idea. Today: Signals + latency real, Meta/Cal model-gated, **Regime-fit/Bet steps render `DataUnavailable`** |
| **CascadeBars** | `cascade: {stage,value,binding}[]` | 5-layer bars, binding constraint in `warn` + "binding" tag | drawer, iOS Idea — **wrapped in `ComingSoon` (Wave 4: needs `constraints_applied` surfaced)** |
| **MiniBars** *(SHAP)* | `items, signed?, labelWidth?, max?, fmt?` | signed diverging horizontal bars (pos/neg) | drawer "Why", Symbol, Model importance — **`ComingSoon` (Wave 4: persist `shap_importance`)** |
| **Donut** | `segments: {label,value,color}[], size?, thickness?, center?` | SVG ring, inset rail | Portfolio (`ComingSoon`), Model splits |
| **Sparkline** | `data, width?, height?, color?, fill?` | inline SVG trend, auto up/down color | Markets, Overview movers, Strategy cards |
| **AreaChart** | Recharts: `data, color?, height?, gradient?` | smooth gradient area + hover tag, minimal axis | NAV hero *(NAV null → `DataUnavailable`)*, Symbol/Strategy curves |
| **CandleChart** | `candles, height?` | custom SVG OHLC + volume + last-price tag | Symbol detail — **real (bars table)** |
| **Stat / StatStrip** | `label, value, sub?, subColor?, valueSize?` | labeled metric, divider row | Overview, Research, Strategy |
| **Delta** | `value, kind?, dp?, arrow?` | signed value, ▲/▼, pos/neg/dim | tables, stats, movers |
| **Segmented** | `options, value, onChange, size?` | timeframe / filter toggle | charts, Markets, Ideas tabs |
| **AssetGlyph** | `sym:{symbol,type}, size?` | tinted rounded ticker badge | Markets, Symbol, idea rows |
| **StatusDot** | `ok, live?` | freshness/live dot (pulse when live) | top bar, status |
| **Panel / Card** | `title?, action?, children` | `.card` container, section header | everywhere |
| **Table** (`.tbl`) | right-aligned, mono numerics, clickable rows | ideas / positions / trade log | Ideas, Portfolio, Research |
| **IdeaDrawer** | `idea, onClose, onOpenSymbol` | slide-in full decision chain | Trade Ideas (web); iOS = push screen |
| **ComingSoon** *(new)* | `feature, unlock` | dignified locked panel: dimmed iconographic preview + title + the **exact unlock condition** from `data_readiness.md` (e.g. "Sizing cascade — unlocks when the engine surfaces `constraints_applied` at the idea boundary · Wave 4"). Never renders fake numbers. | Portfolio & Risk, Research & Backtests, Execution & TCA, Monitoring, Scenarios, Track Record, Replay; and per-section inside partial screens |
| **DataUnavailable** *(new)* | `field, reason?` | inline em-dash slot with a subtle `?`-affordance; tooltip explains *why* this number is absent today (null at BFF / model-gated). Used **in place of a value**, not as a whole-panel takeover. | regime chips, regime-fit step, cost, track-record, NAV/exposure stats |

**ComingSoon vs DataUnavailable rule:** a whole capability that doesn't exist → `ComingSoon` (panel-level, with unlock condition). A single field that's null inside an otherwise-real panel → `DataUnavailable` (inline `—` + reason). This is the mechanism that lets Overview, Symbol, drawer, Strategies, and Model ship as **partial-but-honest** in Waves 1–3.

### 4. "Pro density" toggle

A single global preference (persisted; web top-bar switch, iOS Settings) — **not** a second app, not a second theme. It only re-points spacing/size aliases and disclosure defaults; colors, components, and tokens.json are unchanged:
- **Comfort (default):** card padding `sp.4–sp.6`, table rows 44px, hero charts on, drawer sections expanded, eyebrow labels visible.
- **Pro:** padding collapses to `sp.2–sp.3`, table rows ~30px, sparklines replace hero charts where space-bound, more columns surfaced, drawer sections default-collapsed, mono numerics tighten. This is where the v2 cockpit's Bloomberg-density lives — same components, denser tokens.
Implemented as a `data-density="comfort|pro"` attribute (web) / `@Environment(\.density)` (iOS) that switches a small alias subset (`space.*`, row-height, `chartMode`). No component re-implements itself; it reads the alias.

### 5. Charting guidelines

Consumer/hero surfaces use gradient **area** curves (thin stroke, hover tag, minimal right-aligned y-axis, hidden x). Research surfaces add `grid` lines + benchmark overlay. **Candlesticks** (the one real chart today) carry volume + live last-price tag. Distributions (meta-prob histogram, CPCV) use bars with consistent pos/neg color. Calibration/reliability plots overlay observed-vs-predicted against a `y=x` reference. Direction is never color-alone — always arrow + label. Charts never imply precision the data lacks (the whole point of `DataUnavailable`/`ComingSoon`).

## 3. Information Architecture & Navigation

> One product, two shells, one BFF. The web shell is a persistent **sidebar + sticky trust bar**; iOS is a **bottom tab bar + navigation stack**. Both render the same screens from the same `ApiEnvelope`. Depth is never a second app — it lives in **drawers, detail pushes, and a "Pro density" toggle**. Every destination is either **LIVE** (real engine data today) or **COMING** (visible, dignified, with its exact unlock condition shown on open). We never render a placeholder as a live number.

### 3.1 The converged sitemap (web)

The shell keeps the prototype's grouped sidebar (`apps/prototype/src/web/Sidebar.tsx`) but corrects two things: the COMING destinations become **visible-but-locked nav rows** (today they're simply absent), and `research` is regrouped under a new **Research** section so the LIVE Cockpit/Engine block reads cleanly.

```
Aperture (web)  — sidebar groups, top→bottom
├── COCKPIT            [all LIVE v1]
│   ├── Overview       /overview        "what now" in one glance
│   ├── Markets        /markets         the universe, real bars
│   ├── Trade Ideas    /ideas           decision table + decision drawer
│   └── Strategies     /strategies      live ideas per family
├── ENGINE             [LIVE v1, model-gated]
│   ├── Symbol         /symbols/{sym}   (reached from Markets/Ideas; not a top-level row)
│   └── Model & Features /model         meta-prob histogram + retrain timeline
├── PORTFOLIO          [COMING — locked rows]
│   ├── Portfolio & Risk   🔒
│   └── Execution & TCA    🔒
├── RESEARCH           [COMING — locked rows]
│   ├── Backtests          🔒
│   ├── Scenarios & Stress 🔒
│   └── Track Record       🔒
└── OPERATE            [COMING — locked rows]
    ├── Monitoring & Alerts 🔒
    ├── Replay / Time Travel 🔂  (deferred, see note)
    └── Preflight & Go-Live 🔧  (cheap-win candidate — wire post-v1)
```

Layout: persistent left **sidebar** (`--sidebar-w`, `--bg-1` background, `1px solid var(--border)` right edge) + sticky **trust bar** (`height:68`, `rgba(10,12,16,0.72)` + `backdrop-filter: blur(12px)`, matching the prototype `TopBar`). Content max-width 1480px. **Symbol is a detail route, not a sidebar row** — it's reached by clicking a Markets row or an idea's "Open symbol", exactly as the prototype's `PARENT` map already encodes (`symbol → markets`).

#### Why this grouping (locked)
- **COCKPIT** = the "simple surface": the four screens that answer *what now*. This is the only group a casual glance needs.
- **ENGINE / PORTFOLIO / RESEARCH / OPERATE** = "deep on demand": progressively heavier. The v2 cockpit's 15 pages map 1:1 into PORTFOLIO/RESEARCH/OPERATE — they are **the same IA, folded in**, not a second app.
- The **Audit & Compliance** and **Settings** v2 rows are intentionally dropped from v1 nav (Settings is single-tenant noise; Audit has no rows written — `ComplianceAuditLogger` is never instantiated). They re-appear when their stores land.

### 3.2 The converged sitemap (iOS)

iOS gives the four COCKPIT screens **first-class tabs** (matching the prototype `IOSTab = 'home' | 'markets' | 'ideas' | 'strategies'`), and everything heavier lives behind a **More** tab as a grouped list — preserving parity without cramming a 13-item tab bar onto a phone. "Home" is the iOS label for Overview.

```
Aperture (iOS)  — bottom tab bar + NavigationStack
├── Tab: Home        (= Overview)   [LIVE]
├── Tab: Markets                    [LIVE]   → push: Symbol
├── Tab: Ideas                      [LIVE]   → push: Idea (decision chain)
├── Tab: Strategies                 [LIVE]   → push: Strategy
└── Tab: More                       grouped list:
    ├── Model & Features  [LIVE, model-gated]  → push
    ├── Portfolio & Risk  🔒  → "coming" detail
    ├── Backtests / Scenarios / Track Record 🔒
    ├── Monitoring · Execution · Preflight 🔒/🔧
    └── Settings (mode, theme, density)
```

Native patterns: large titles collapsing on scroll, `NavigationStack` push/pop with swipe-back, pull-to-refresh, haptics on row tap. The **decision drawer on web = a detail push on iOS** (`IdeaScreen`) — same content, platform-native container.

### 3.3 Web ⇄ iOS parity map

| Screen | Web | iOS | Container difference |
|---|:--:|:--:|---|
| Overview / Home | ✓ sidebar | ✓ tab "Home" | — |
| Markets | ✓ sidebar | ✓ tab | — |
| Symbol detail | ✓ route (from Markets/Ideas) | ✓ push | candles both; iOS may start area-only |
| Trade Ideas list | ✓ sidebar | ✓ tab "Ideas" | — |
| Decision drawer / chain | ✓ **slide-in drawer** | ✓ **push (Idea screen)** | same payload, different container |
| Strategies + detail | ✓ sidebar + route | ✓ tab + push | — |
| Model & Features | ✓ sidebar (Engine) | ✓ More → push | iOS condensed (histogram + timeline) |
| Portfolio · Execution · Backtests · Scenarios · Track Record · Monitoring · Replay · Preflight | 🔒 locked rows | 🔒 More list | identical "coming" treatment |

**Parity rule:** any screen LIVE on web is LIVE on iOS in v1; any COMING screen is COMING on both. Platform differs only in *container* (drawer vs push, sidebar vs tab), never in *which data is real*. The phone is decision-focused (Cockpit tabs are first-class); the desktop surfaces the heavier Engine/Research depth more prominently — but neither hides a destination the other has.

### 3.4 How a COMING destination behaves (the locked-row contract)

A COMING row is **always visible, never a dead link, never fake data.** This is the mechanism that lets us promise "huge but simple" honestly.

**Nav row appearance**
- Rendered at reduced emphasis: label in `--text-3`, a trailing lock glyph, and **no status dot** (LIVE rows carry the green `live-dot`). Reuse the existing sidebar `<button>` with `disabled`-styling, not a removed item.
- Three lock variants, by readiness from `docs/data_readiness.md`:
  - 🔒 **Gated** — needs net-new engine persistence (Portfolio, Execution, Backtests, Scenarios, Track Record, Monitoring).
  - 🔧 **Wireable** — runnable engine code exists, only a BFF stub to rewire (**Preflight & Go-Live** — point `PreflightService` at the real `PreflightChecker`/`InfrastructureProbe`). Marked as a fast-follow, not v1.
  - 🔂 **Deferred** — needs an audit chain that is never written (**Replay**); lowest priority.

**On click → a dignified "Coming" screen** (not a toast, not a 404). Standard template, reused for every locked destination:
- Title + one-line purpose (lifted from the v2 page spec, e.g. Portfolio: *"Positions, exposure, factor risk, drawdown"*).
- A muted **wireframe ghost** of the eventual layout (low-opacity skeleton of the real panels) so the depth is legible without inventing numbers.
- An **"Unlock condition"** card stating the exact engine gate, verbatim from data_readiness:
  - Portfolio & Risk → *"Unlocks when the engine persists positions + a NAV series. Today no orders are routed (`live_orders_sent=0`) and `src/portfolio` has zero production callers."*
  - Execution & TCA → *"Unlocks when the order-routing path writes `ExecutionStorage` (orders/fills/TCA). The deployed path stops before routing."*
  - Backtests → *"Unlocks when backtest runs are persisted **and** the retrain gate is fixed (`retrain_pipeline.py:265` falls through to `gate_unavailable`)."*
  - Scenarios → *"Unlocks when `ScenarioService` calls the real `factor_risk` engine instead of returning mock numbers."*
  - Track Record → *"Unlocks when a call-history store exists (the trade-ideas snapshot is overwritten each publish)."*
  - Monitoring → *"Unlocks when pipeline metrics are scraped over HTTP (the registry is currently unscraped; `/metrics` serves only `bff_*`)."*
  - Replay → *"Unlocks when the audit chain is written (`ComplianceAuditLogger` is never instantiated)."*
- For 🔧 Preflight only: a **"Wire this next"** affordance, since it needs no persistence.

**Rule:** the lock state is **data-driven**, derived from a single `screenReadiness` map (the machine-readable twin of `docs/data_readiness.md`), so when an engine wave lands, a screen flips LIVE by changing one entry — not by editing nav code.

### 3.5 The trust-state top bar (v1 — real signals only)

The prototype `TopBar` currently binds to mock NAV/P&L/regime. For v1 it must bind **only to fields the `ApiEnvelope` actually carries** (`src/web/envelope.py`: `as_of`, `staleness_seconds`, `source_freshness`, `model_version`, `regime`) and to the trade-ideas snapshot. Everything else is removed from the bar until its producer exists.

**v1 trust bar contents (left → right):**

| Element | Source (real today) | States / tokens |
|---|---|---|
| Page title + sub | route | — |
| ⌘K search (symbols/strategies) | client (Markets + snapshot) | placeholder only; deep palette is COMING |
| **Mode pill** = `Paper` | static config (`SYSTEM.mode`; engine is read-only, `live_orders_sent=0`) | always reads **PAPER**; `--info` chip. Live mode is COMING. |
| **Model pill** | envelope `model_version` / idea `action` | `MODEL LOADED` (`--pos`) when a version is present; **`MODEL REQUIRED`** (`--warn`) when ideas return `action=MODEL_REQUIRED`. Drives whether meta/cal probabilities render. |
| **Freshness pill** | envelope `as_of` + `staleness_seconds` (snapshot `latest_bar_at`) | `Ns ago` with green `live-dot` when fresh; **amber "STALE"** when `staleness_seconds` > threshold. The single most important honesty signal. |

**Explicitly DEFERRED from the v1 trust bar** (present in the v2 §7.3 spec, removed here because no producer is wired):
- **NAV / Daily P&L / Drawdown / Gross / Net / Positions** — `/overview` returns these as `null` (no persisted portfolio). Removing them from the bar is the headline change vs. the mock `TopBar`, which currently shows fake NAV + "today" P&L.
- **Regime label + probability** — `RegimeDetector` has zero runtime callers; `regime` rides in the envelope but is null. Drop the `RegimeChip` from the v1 bar.
- **Drift severity** — emits a hardcoded `1.0`, never sets a baseline → meaningless. Defer.
- **Calibration recency** — no calibration history persisted. Defer.
- **Broker heartbeat / Active breakers / Alert counts** — no broker wired, no scraped metrics, no alert store. Defer.

So the v1 bar is exactly **three live pills — Mode · Model · Freshness — plus search and the page title.** Three honest signals beat eleven hopeful ones. Each pill, when clicked, deep-links to its source (Model → Model & Features; Freshness → Overview).

### 3.6 The "Pro density" toggle (reconciling huge-but-simple)

A single **Settings → Density: Comfortable | Pro** switch (CSS-var driven, persisted per device). Comfortable is the default consumer surface (generous `--sp` spacing, hero charts). **Pro** tightens spacing, shows more table columns by default, and reveals secondary panels inline (e.g. Symbol's bar-microstructure columns, the Ideas table's latency-per-stage). It is **one design language at two densities** — the v2 "operator cockpit" *is* Aperture in Pro density, not a separate build. Pro never unlocks COMING data; it only changes how much *real* data is shown at once.

### 3.7 v1 nav, final ordering (drop-in for `Sidebar.tsx` GROUPS)

```
COCKPIT   → Overview · Markets · Trade Ideas · Strategies        [LIVE]
ENGINE    → Model & Features                                     [LIVE, model-gated]
PORTFOLIO → Portfolio & Risk 🔒 · Execution & TCA 🔒             [COMING]
RESEARCH  → Backtests 🔒 · Scenarios & Stress 🔒 · Track Record 🔒 [COMING]
OPERATE   → Monitoring & Alerts 🔒 · Preflight & Go-Live 🔧 · Replay 🔂 [COMING]
```
Symbol & Strategy detail remain **non-nav detail routes** (`PARENT` map). This is the complete, honest v1 IA: 5 LIVE destinations + 1 detail layer, 8 dignified COMING rows, three real trust pills, full web↔iOS parity.

## The disclosure & data-state system — how "huge but simple" is real

This section defines the interaction contract that lets Aperture be a one-glance consumer app *and* the full operator cockpit, without forking into two products. The mechanism is two orthogonal axes — **a 4-rung disclosure ladder** (how deep you've drilled) and a **density toggle** (how much the current rung shows) — plus one rule that makes the whole thing honest: **every data element resolves to exactly one of seven canonical states, and "Not yet available" is a first-class state, never a fake zero.**

> Ground truth: this pillar is written against what the engine produces *today* (`docs/data_readiness.md`). The BFF already ships the metadata that drives these states — `apps/prototype`'s consumer envelope maps directly to `src/web/envelope.py`'s `ApiEnvelope`: `as_of`, `staleness_seconds`, `source_freshness`, `model_version`, `regime`, `warnings[]`, `errors[]`, `data`, serialized with `exclude_none=True` (so ABSENT fields arrive null or absent — the client must treat both identically).

---

### 1. The progressive-disclosure ladder

Four rungs. Each rung adds detail; **nothing on a deeper rung is required to act on a shallower one.** The same `TradeIdea` object renders at all four — only the projection changes.

| Rung | Surface | Answers | What lives here (hard rule) | Components |
|---|---|---|---|---|
| **L0 — Glance** | Overview hero + tiles, iOS Home, widgets | "What now?" | Exactly **one verdict per concept**: action counts (BUY/SELL/WATCH), top-N actionable ideas, the single binding number. **No raw model internals.** If it needs a sentence to explain, it doesn't belong at L0. | `Stat`, `ActionPill`, `RegimeChip`, `Sparkline` |
| **L1 — Row** | Trade Ideas table, Markets list, Strategies grid | "Which one, and roughly why?" | One scannable line: symbol, `ActionPill`, top-signal family, `meta_probability` / `calibrated_probability` (model-gated), `target_weight`, `bet_size`. Columns that are ABSENT today (regime fit, cost) are **hidden by default at standard density**, not shown blank. | `.tbl` row, `ActionPill`, `Delta`, `ProbRing` (compact) |
| **L2 — Drawer / Detail** | `IdeaDrawer` (web slide-in), iOS push, Symbol Detail | "Show me the whole decision chain." | The full chain that exists today: NL `reason`, decision-chain stepper (`signal_count` → meta → calibrated → bet → target), per-stage `stage_latency_seconds`, signals + their metadata. The four ABSENT sections (regime-fit, sizing-cascade waterfall, SHAP, expected cost, track record) render here as **"Not yet available"** placeholders that hold their layout slot. | `IdeaDrawer` Sections, `ProbRing`, decision-chain stepper, sizing-cascade `Bar`, `MiniBars` (SHAP) |
| **L3 — Pro density** | Same drawer/detail with the toggle on | "Give me everything, tight." | Raw values alongside formatted ones, `bars_loaded` / `feature_rows` counts, full per-stage latency table, `errors[]` verbatim, `model_version`, `as_of` / `staleness_seconds` exact, request id. Adds rows/columns; **never adds new screens.** | density-aware variants of the L1/L2 components |

**The core gesture is invariant across platform:** tap any L1 row → L2 (drawer on web, push on iOS). This is *the* product. Everything below L2 is reachable from L2; nothing requires hunting through navigation.

**Promotion/demotion rule (what decides a rung).** A field's home rung = the shallowest rung at which a non-expert can *act on it without further explanation*. `action` is L0. `calibrated_probability` is L1 (a number you can rank by). `top_shap_feature` is L2 (needs the chain as context). Raw `feature_rows` count is L3 (diagnostic only). When in doubt, push it **down** a rung — simplicity is the default, depth is opt-in.

---

### 2. The "pro density" toggle

One global toggle, persisted per-user (localStorage on web; `@AppStorage` on iOS), surfaced in the top bar and Settings. It is **not** a separate mode or skin — it is a density multiplier on the rung you're already on.

**Behavior contract:**
- **Standard (default).** Generous spacing (`--sp-4`/`--sp-5` row padding), hero numbers at `valueSize ≥ 22`, ABSENT columns hidden, charts use the consumer gradient `AreaChart`, one decimal on probabilities, relative timestamps ("2m ago").
- **Pro.** Tightens to `--sp-2`/`--sp-3` padding, reveals diagnostic columns (`bars_loaded`, `feature_rows`, exact `stage_latency_seconds`, `model_version`), exposes raw-vs-formatted values, switches Symbol Detail's default chart to `CandleChart` + microstructure columns, shows absolute timestamps + `staleness_seconds`, monospace (`--font-mono`) everywhere numeric.
- **What it must NOT do:** change navigation, hide any action, alter colors/semantics, or fabricate data. Pro density reveals *more real fields and tighter layout* — it never turns a "Not yet available" into a number.
- **Respects the data-state system fully:** a field ABSENT in standard is still ABSENT in pro — pro just also shows you the raw envelope metadata around it.

Rationale: this is how the "operator cockpit" folds in. The cockpit *is* Aperture-at-pro-density-with-drawers-open. We never ship two apps; we ship one app with a density dial.

---

### 3. The canonical honest data-state system

**The rule: every data-bearing element renders exactly one of seven states. A component must declare its state before it renders a value. There is no eighth "just show 0" path.** This is the single most important rule in the design system — it is what makes "deep on demand" *honest* given that most depth isn't wired yet.

State is derived from the envelope, in this precedence order (first match wins):

```
errors[] present for this field        → Error
data === undefined (field not in DTO)  → Not yet available   ← the ABSENT contract
data === null (producer wired, no value)→ Empty OR Not-yet-available (see §3 table)
in-flight, no cached value             → Loading
in-flight, has cached value            → Live (show cached) + subtle refreshing affticator
staleness_seconds > threshold          → Stale
otherwise                              → Live
```

| State | Visual | Wording pattern | Token / class | When |
|---|---|---|---|---|
| **Live** | Full value, `--text-1`, optional pulsing `.live-dot` (`--pos`) on the freshness indicator | the value itself | `--text-1`, `.live-dot` | fresh, real data |
| **Loading** | Skeleton block matching the value's footprint (never a spinner inside content) — shimmer sweep | *(no text)* | new: `.skeleton` on `--surface-2`, `@keyframes shimmer` | first fetch, no cache |
| **Empty** | Calm, centered, low-emphasis line in the slot's footprint | "No active idea above the entry gate." / "No movers today." — describe the *real, valid* zero-state, never an error | `--text-3`, `--surface-inset` well | producer ran, legitimately returned nothing |
| **Stale** | Last-good value dimmed to `--text-2` + an amber chip; section gets a thin `--warn` top-border | "As of {relative} · refreshing" or banner "Showing data from {time} — snapshot is {age} old" | `--warn`, `--warn-soft` | `staleness_seconds` over threshold (default **90s**; the publisher writes `trade_ideas.json` on a cadence) |
| **Error** | Inline within the slot (never a full-page takeover for one field), retry affordance + copyable request id | "Couldn't load {thing}. Retry" + mono `req: {id}` | `--neg`, `--neg-soft`, `.btn` | `errors[]` has an entry for this field |
| **Not yet available** | **The dignified state.** Slot keeps its full layout; a small `lock`/`clock` icon; label of what it *will* be + the exact unlock condition. Muted, **never** colored as good/bad. | "Regime fit — coming when the regime detector is wired into the live cycle." / "Sizing cascade — coming when the 5-layer sizing constraints are surfaced at the idea boundary." | new: `.coming` on `--surface-inset`, `--text-3`, dashed `--border-strong` | field is ABSENT today (`regime`, `regime_fit_score`, `sizing_constraints_applied`, `expected_cost_bps`, `top_shap_feature`, `track_record_*`) |
| **Model-gated** *(specialization of Not-yet-available)* | Same treatment, but the unlock is dynamic, not roadmap | "Meta probability — load an MLflow production model. Currently `MODEL_REQUIRED`." | `--text-3` + `ActionPill` `pill-neutral` "Model?" | `meta_probability` / `calibrated_probability` null because no production model is loaded |

**Hard prohibitions (enforce in review):**
1. **Never render `0`, `—`, or `N/A` for an ABSENT field.** `—` is reserved for a *legitimately empty* live value (e.g. no track record yet *but the producer exists*). ABSENT → always the "Not yet available" treatment with its unlock copy. Today `ProbRing` and `IdeaDrawer` render `'—'` for null meta/cal/regime/SHAP — **this is the gap to fix**: route null-because-absent through `.coming`, reserve `—` for null-because-empty.
2. **Never let one field's Error blank a whole screen.** State is per-element. The Overview must render its real action-count tiles even if the equity curve errors.
3. **Stale is amber and reversible, not an error.** Stale data is still shown (last-good), just flagged — operators would rather see 90s-old truth than a spinner.
4. **The "Not yet available" copy must name the *real* unlock condition** from `data_readiness.md`, not a vague "coming soon." This turns the honest gaps into a visible product roadmap and builds trust instead of eroding it.

**Per-screen application (today's reality):**
- **Overview:** Live — action counts, top-N actionable, summed `stage_latency_seconds`. Not-yet-available — `nav`, `daily_pnl`, `drawdown`, `gross/net_exposure`, `positions_count`, equity curve, regime (all already null in `src/web/routes/overview.py` with the warning `"portfolio metrics … unavailable: no persisted portfolio"` — surface that warning as the unlock copy).
- **Decision drawer (L2):** Live — `reason`, chain stepper, signals, latency. Model-gated — meta/calibrated. Not-yet-available — regime-fit, sizing-cascade waterfall, SHAP, expected cost, track record (5 sections that hold their slots).
- **Symbol Detail:** Live — candles, volume, bar microstructure cols, "what the engine sees." Not-yet-available — live computed feature values (features never persisted), per-symbol history/track record.
- **Strategies / Model & Features:** Live — live ideas per family, static thesis/params, meta-prob histogram (model-gated), thin retrain timeline. Not-yet-available — per-strategy Sharpe/win/PnL, calibration/ECE/Brier, drift, RL shadow.

---

### 4. The diff / "what changed" model

Operators scan for mutations, not absolute state. Verdict: a **client-side snapshot diff is v1-feasible and should ship**; the richer server-streamed diff is **deferred**.

**v1 (ship it) — client-side snapshot diff.** The publisher overwrites one tmpfs `trade_ideas.json` each cycle; the client already polls it. On each successful fetch, keep the **previous** payload in memory (and the last-good in localStorage / iOS App Group for cold-launch). Diff the two keyed by `symbol`:
- **New idea** (symbol absent before) → `new` badge on the row.
- **Side flip** (`action` BUY↔SELL, or `top_signal_side` sign change) → `flipped` badge, animated.
- **Weight move** ≥ 25 bps in `target_weight` → `↑`/`↓` delta chip on the row.
- **New error** (`errors[]` grew for a symbol) → `--neg` flag.

Surface as: web — a dismissible "What changed" strip above the Ideas table + per-row badges; iOS — a "What changed" card on Home. Scope strictly to fields **PRODUCED today** (`action`, `target_weight`, `top_signal_side`, `errors`) so the diff is always real. Diffing is itself state-aware: a field going Live→Not-yet-available is **not** a "change" worth flagging (it's an engine gap, not a market event) — only diff within-Live transitions.

**Deferred (needs engine work).** True event-level diff (SSE `/stream/diff`, "weight moved at 13:42:07"), and any cross-time history, require an **append-only call-history store** — the snapshot is overwritten each publish, so there is no server-side previous state. Mark the time-travel/Replay surface "coming when call history is persisted." Until then, the client-side two-snapshot diff is the honest, shippable version of principle #4 ("show the diff").

---

### 5. Tokens & components this pillar adds

To build the above, three net-new primitives join the existing catalog (`apps/prototype/src/components`):

- **`<DataState>`** — the wrapper every data element passes through. Props: `{ state, value, unlock?, requestId?, asOf?, children }`. Centralizes the seven-state switch so the prohibition in §3 is structurally enforced, not hoped for. Drawer Sections, `Stat`, `ProbRing`, table cells all delegate to it.
- **`<ComingState label unlock />`** — the dignified Not-yet-available renderer. New CSS class `.coming` (`--surface-inset` fill, dashed `--border-strong`, `--text-3`, `clock` icon).
- **`<Skeleton w h r? />`** — shimmer placeholder. New `@keyframes shimmer` + `.skeleton` (on `--surface-2`). Pairs with `prefers-reduced-motion` (static `--surface-2` fill, no sweep).

New tokens (extend `apps/prototype/src/styles/theme.css`, mirror into `tokens.json` → web CSS vars + iOS `Color`): `--stale-threshold-seconds: 90` (config, not CSS), reuse existing `--warn`/`--warn-soft` for Stale, `--neg`/`--neg-soft` for Error, `--text-3` + `--surface-inset` for Empty/Coming. **No new colors needed** — honesty rides entirely on the existing semantic palette, which keeps the surface calm.

**One required BFF addition:** the envelope carries `errors[]` but no per-response **request id**. Add `request_id` to `ApiEnvelope` (`src/web/envelope.py`) so the Error state can render the copyable id promised in §3.

---

## Part II — v1 Screens

Each screen below is build-ready: purpose, real data sources, the deferred elements with their unlock conditions, components, states, and interactions. Specs are written against the foundations above.

UNCHANGED — see notes; the original markdown is fully honest and returned verbatim below.

## Overview / Home (`/overview`) — v1 spec

**Readiness:** Wave 1, partial-real. **Source of truth:** `docs/data_readiness.md`. **Backing endpoint:** `GET /api/v1/overview` (`src/web/routes/overview.py`).

### Purpose
The L0 "what now?" glance. One verdict per concept, no raw model internals. The engine produces a decision-flow snapshot, not a portfolio — so the v1 hero is reframed from a money dashboard to a decision dashboard. The headline is how many actionable ideas this cycle; the body is the top-N ideas (one tap into the full decision chain); trust lives in the bar. Everything portfolio (NAV, P&L, equity, risk, regime, movers) is dignified-coming, never faked.

> The v1 hero when NAV is null. /overview returns nav, daily_pnl, drawdown, gross/net_exposure, positions_count as null with the warning "portfolio metrics (nav/pnl/drawdown/exposure) unavailable: no persisted portfolio." We do not render $0.00 or a flat fake curve. The primary hero is the action-count verdict ("4 actionable ideas this cycle — 3 BUY · 1 SELL"); the NAV+equity panel sits beneath it as a ComingState ghost with its exact unlock condition. The layout still reads as a complete command center.

### Layout (top to bottom)
1. Decision headline (hero, LIVE). "N actionable ideas this cycle", with BUY / SELL / WATCH as three large tabular-nums figures under ActionPill-colored labels; a muted line "M need a model · K no data" when those counts are non-zero. Source: data.action_counts.
2. What-changed strip (LIVE, client-side diff). Dismissible. new / flipped / up-down Nbps badges from diffing the current snapshot against the previous (in-memory + localStorage), scoped to action, target_weight (>=25bps), top_signal_side, errors[]. Absent on cold launch.
3. Top trade ideas (LIVE, the core gesture). 5 rows: ActionPill · symbol · strategy/top_signal_family · target_weight (signed %) · calibrated_probability via compact ProbRing. Source: data.top_actionable (server pre-sorted by abs(target_weight)). Tap to decision drawer (web) / Idea push (iOS). calibrated_probability is model-gated — Model-gated treatment, not a fake number, when no production model is loaded.
4. Engine pulse (LIVE). Summed stage_latency_seconds as a compact stage breakdown + total-cycle figure. The one honest system-health signal today. Expands to the full per-stage table in Pro density.
5. Portfolio value & equity curve (COMING). ComingState ghost in the hero-secondary slot. Unlock: "persists positions + a NAV series; today live_orders_sent=0 and src/portfolio has zero production callers" — Wave 5.
6. Risk & performance band (COMING). Replaces the prototype's 6-up Sharpe/DD/Vol/Win/Exposure/Positions StatStrip with a single ComingState band (avoids six identical em-dashes). Unlock: same portfolio gate; backtest metrics never persisted (retrain gate broken, retrain_pipeline.py:265) — Wave 5.
7. Market regime (COMING). ComingState panel. Unlock: "wired into the live cycle and persisted; RegimeDetector has zero runtime callers" — Wave 6. The RegimeChip is dropped from the v1 trust bar (no null chip).
8. Market movers — OMITTED in v1, with a "View Markets" affordance to the fully-LIVE Markets screen. Movers need multi-symbol history; only ~1 symbol fires live, so a movers grid would imply breadth the data lacks. Returns sourced from the bars table once the universe has multiple live symbols.
9. P&L contribution — OMITTED in v1 to keep L0 calm; live ideas-per-family live on Strategies.

### Trust bar (shared, anchored here)
Three honest pills only: Mode = PAPER (static config), Model (LOADED/MODEL REQUIRED, drives whether ProbRings render), Freshness (Ns ago + live-dot; amber STALE when staleness_seconds > 90). Removed vs. the prototype TopBar: fake NAV, "today" P&L, and the regime chip. Three honest signals beat eleven hopeful ones.

### Components
Card/Panel · Stat/StatStrip (action counts, latency total) · ActionPill · ProbRing (compact, model-gated) · Sparkline (latency sparkbar) · Delta (target-weight, weight-move chips) · ComingState · DataUnavailable · Skeleton · StatusDot/live-dot · DataState wrapper (enforces the 7-state switch).

### Data states (every slot passes through DataState)
- Live — full value, --text-1, pulsing freshness dot.
- Loading — Skeleton shimmer matching footprint; ComingState panels render immediately; respects prefers-reduced-motion.
- Empty — action_counts all-zero & top_actionable empty to "No ideas above the entry gate this cycle." (a valid quiet engine, never an error).
- Stale — staleness_seconds > 90: last-good dimmed --text-2 + amber chip + --warn top-border; Freshness pill to amber STALE.
- Error — errors[] / aggregation fall-through ("overview trade-ideas aggregation unavailable"): inline retry + copyable req:{request_id} in the ideas slot only; never blanks other panels.
- Not yet available (first-class) — NAV+equity, risk band, regime to ComingState: full footprint, low-opacity wireframe shape (no numbers, no drawn lines), clock/lock glyph, verbatim unlock copy from data_readiness.md / the envelope warning. Muted, never good/bad colored.
- Model-gated (dynamic sub-variant) — calibrated_probability null with no model to "Calibrated probability — load an MLflow production model. Currently MODEL_REQUIRED."

> Prohibition (review-enforced): never render 0, —, or N/A for an ABSENT field. "—" is reserved for a legitimately-empty live value whose producer exists; ABSENT to always ComingState/DataUnavailable with real unlock copy.

### Interactions
Tap idea row to drawer (web) / Idea push (iOS) — the invariant L0 to L2 gesture · "View all ideas" to /ideas · dismiss what-changed strip (session) · Freshness/Model pills deep-link to source · hover/tap ?/ComingState to unlock-condition tooltip (gap-as-roadmap) · Pro-density to latency strip expands to per-stage table, rows tighten, absolute timestamps, raw bars_loaded/feature_rows columns (never unlocks coming data) · iOS pull-to-refresh + haptics; web auto-poll on publisher cadence, animating flipped/new badges. All empty/stale/error affordances are per-element and non-blocking.

### One BFF addition required
Add request_id to ApiEnvelope (src/web/envelope.py) so the Error state can render the copyable id promised by the data-state system.

### Web to iOS parity
Identical data; container differs only: web sidebar row vs. iOS "Home" tab; idea row to web drawer vs. iOS push. Any slot LIVE on web is LIVE on iOS; every COMING slot is COMING on both.

---

## Markets

**Route:** `/markets` (web sidebar · Cockpit group) · iOS tab "Markets"
**Readiness:** Wave 1 — REAL. Backed entirely by the TimescaleDB `bars` hypertable (price, sparkline, 1D/1W/1M changes, volume) joined to the tmpfs `trade_ideas.json` snapshot (active-idea flag). No new engine persistence required — only a thin read-only `GET /api/v1/markets` BFF route plus a small static instrument-reference map (name + asset class), which the bars row alone does not carry.

> Markets is the most *finished* screen in v1: almost everything on it is real. Its design job is therefore confidence, not apology — a clean, dense, scannable universe table that gets you from "all instruments" to "this one" (and onward into its decision chain) in one glance and one tap.

### Layout (web)

A single `.card` table inside the 1480px content column, under a filter/summary bar:

```
[ All ][ Equities ][ Indexes ][ Crypto ][ Futures ]      19 instruments · 4 with active ideas
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Instrument            30d trend     Price      1D      1W      1M     Volume      › │  ← header (sortable)
├──────────────────────────────────────────────────────────────────────────────────┤
│ ◧ NVDA •  NVIDIA Corp.   ╱╲╱  ▁▃▅    118.42   ▲+1.9%  +4.2%  +12.1%   142.3M     › │
│ ◧ AAPL    Apple Inc.     ╲╱╲      196.04   ▼-0.4%  +1.1%   +3.8%    88.1M     › │
│ ◧ BTC  •  Bitcoin        ╱╱╱  ▅▇█  61,204    ▲+2.1%  +6.7%  +11.4%   2.41B      › │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Column grid (from the prototype `MarketRow`, market-cap column dropped): `Instrument 1.7fr · 30d trend 130 · Price 110 · 1D 84 · 1W 84 · 1M 84 · Volume 130 · chevron 22`. All numerics are `.num` (`font-variant-numeric: tabular-nums`, JetBrains Mono) so columns align. The `•` after a ticker is the `--accent` active-idea dot.

### Layout (iOS)

Large title "Markets" collapsing on scroll, a search field, a horizontally-scrolling chip strip, then a single `IOSCard` of rows. Each row is condensed: `AssetGlyph(36) · ticker + idea-dot + name · 62×28 sparkline · right-aligned price + 1D Delta`. Tap pushes Symbol Detail (`NavigationStack`, swipe-back, haptic). Pull-to-refresh re-polls and runs the client diff. Parity rule holds: every field real on web is real on iOS.

### Real data → source map

| Element | Source (real today) | Component |
|---|---|---|
| Last price | `bars` latest close per symbol | `.num` + `fmtPrice` |
| 30d trend | `bars` close series (~46 pts) | `Sparkline` (auto pos/neg) |
| 1D / 1W / 1M | `bars` `last/close_{1,5,21}back − 1` | `Delta` (▲/▼ + sign + color) |
| Volume | `bars` `volume` / `dollar_volume` | `.num` + `fmtCompact` |
| Active-idea dot | `trade_ideas.json` snapshot (action∈BUY/SELL/WATCH) | `.dot` `--accent` |
| Ticker / glyph | `bars.symbol`; asset class from static ref map | `AssetGlyph` |
| Class filter + counts | client over ref map + snapshot join | `.chip` |
| Freshness / stale | envelope `as_of` + `staleness_seconds` | `.live-dot` / `--warn` |

### Honesty rules applied here

- **Market cap is gone, not faked.** The engine has no cap producer (it is a hardcoded mock, `0` for indexes/futures). v1 ships one truthful **Volume** column from the bars table and removes the cap column entirely — we never render `0` or `—` where a missing producer would go.
- **The active-idea dot is honestly scarce.** Only ~4 bars-families fire live on a single symbol today, so in production few rows carry the dot. The `empty` state ("…· no active ideas") treats a zero count as a valid engine outcome, not a failure — and the price table renders fully regardless.
- **Regime stays off the row.** `RegimeDetector` has zero runtime callers and `regime` is null in the envelope, so **no `RegimeChip` renders in any Markets row.** A regime column appears only in Pro density, and even then as a single panel-level `ComingState` ("Regime tags — coming when the regime detector runs in the live cycle"), never as per-row fake chips.
- **The em-dash is reserved.** Inside a Markets cell, `—` appears only for a *legitimately empty live* value (e.g. a brand-new symbol with `<2` bars, so no change/sparkline can be computed). Absent producers get column-removal or a `ComingState`, never an em-dash.
- **One field's error never blanks the table.** If the snapshot join fails but bars load, prices and sparklines render and only the idea dots are suppressed with a quiet "idea status unavailable" note. The `Error` state is scoped to the card body with a retry + copyable `req:{id}`.

### States

- **Loading:** real header row + 8–12 `Skeleton` shimmer rows (respects `prefers-reduced-motion`); chips live-but-disabled. No content spinner.
- **Empty:** (1) filter matches nothing → "No equities in the universe yet." + "Clear filter"; (2) zero active ideas → header reads "N instruments · no active ideas" (rows still render).
- **Stale (`staleness_seconds > 90`):** amber Freshness pill, `--warn` top-border, dismissible "prices as of {relative} — feed is {age} old" strip; last-good prices dimmed to `--text-2`. Reversible, not an error.
- **Error:** inline `--neg` block in the card body, "Couldn't load the universe. Retry" + `req:{id}`.
- **Not-yet-available:** market-cap column removed; Pro-density Regime column = one `ComingState`; missing ref-map entry falls back to ticker-as-name + default tint.

### Interactions

Class filter · sortable Price/1D/1W/1M/Volume headers (default: active-idea rows pinned, then `|1D|` desc) · client search by ticker/name · row → Symbol Detail · idea-dot → decision drawer/Idea screen · iOS pull-to-refresh + web auto-poll running the client snapshot diff (flip/Δprice badges) · Pro-density tightens rows and reveals real extra columns (never fabricated).

### Build notes

- **New BFF route required:** `GET /api/v1/markets` reading last-bar-per-symbol + close series from the `bars` hypertable, wrapped in the standard `ApiEnvelope` (carries `as_of`/`staleness_seconds`). Not yet registered in `src/web/app.py` (only `overview`, `trade-ideas`, `scenarios`, `replay`, `preflight` are). Active-idea flag joins via the existing `TradeIdeasService` snapshot read.
- **Static instrument-reference map** (symbol → display name + asset class) ships with the BFF; the bars row carries only `symbol` + `bar_type`. Unknown symbol → ticker-as-name + default accent tint.
- **Reuse, don't reinvent:** `MarketsPage.tsx` / iOS `Markets.tsx` are the canonical starting points; swap mock `SYMBOLS` for the real route, drop the market-cap column for Volume, add sort + the `Skeleton`/`DataState` wiring. `Sparkline`, `Delta`, `AssetGlyph`, `.chip`, `.row-hover`, `.dot` all already exist.

---

## Trade Ideas + Decision Drawer

> **Readiness:** List = **LIVE** (Wave 1, the tmpfs snapshot's ~21 PRODUCED fields). Drawer = **PARTIAL** (Wave 2 — real reason + signals + latency + model-gated conviction; four chain sections "coming"). Web container = **slide-in drawer**; iOS = **push (`IdeaScreen`)**. Same payload, different shell.
>
> This is the signature screen. It is the proof that Aperture can be "complete in shape" while being truthful field-by-field — every absent value holds its slot with its exact unlock condition, and no placeholder is ever rendered as a live number.

### Why this screen exists
The Trade Ideas list answers *which ideas, roughly why* in one scan (L1). One tap opens the Decision Drawer — *the whole decision chain* (L2) — which is the heart of the product. The list is fully real today. The drawer is honest: its real spine (NL `reason`, signals, per-stage latency, model-gated conviction) reads alongside dignified "coming" steps so the decision chain is legible end-to-end without faking the four sections the engine hasn't wired yet.

### A. The list (L1 — LIVE)

**Header tiles** — four action-count tiles (`Buy` · `Sell` · `Watch` · `Model?`) derived client-side from each idea's `action`, plus a fifth **freshness tile** bound to `as_of`/`staleness_seconds`. *Drop* the prototype's `Gross/Net target … on $NAV` tiles — `PORTFOLIO.nav` is mock and the BFF returns NAV null.

**Filter tabs** — `All · Buy · Sell · Watch · Model?`. The `Model?` tab promotes `MODEL_REQUIRED` ideas to a first-class cohort, not an error bucket.

**Table** (`.tbl`, mono right-aligned, clickable rows). Columns and their real sources:

| Column | Source | Today |
|---|---|---|
| Symbol | `symbol` | ✅ Live |
| Action | `action` | ✅ Live (`ActionPill`; `MODEL_REQUIRED`→"Model?") |
| Strategy | `strategy` / `top_signal_family` | ✅ Live |
| Top conf | `top_signal_confidence` | ✅ Live |
| Meta p | `meta_probability` | ⚠️ **Model-gated** — null→"Model?" chip, never `—` |
| Cal. p | `calibrated_probability` | ⚠️ **Model-gated** (raw==calibrated today) |
| Regime fit | `regime_fit_score` | ❌ **Hidden** at standard density (absent) |
| Target wt | `target_weight` | ✅ Live (signed + arrow) |
| Notional | `target_notional` | ✅ Live |
| Cost | `expected_cost_bps` | ❌ **Hidden** at standard density (absent) |

> **Column honesty rule:** absent columns (`Regime fit`, `Cost`) are *hidden* at standard density — not shown blank — and reappear only in **Pro density** as coming headers with an unlock tooltip. This is the §3 "ABSENT columns hidden, not blank" contract.

**"What changed" strip** — a client-side two-snapshot diff (the publisher overwrites one tmpfs file; the client keeps the prior payload). Badges, scoped strictly to PRODUCED fields: `new` (symbol appeared), `flipped` (`action` BUY↔SELL / `top_signal_side` sign flip, animated), `↑/↓` (`target_weight` moved ≥25bps), neg flag (`errors[]` grew). A field going Live→absent is **not** a change worth flagging.

### B. The Decision Drawer (L2 — PARTIAL, the heart)

**Build constraint:** `GET /trade-ideas/{symbol}` is a **501 stub** (`trade_ideas.py:46-54`). The drawer hydrates **entirely from the list-row object already in memory** — no detail round-trip. It opens instantly.

**Header** — `symbol` · `ActionPill` · `strategy · sideLabel(top_signal_side) · bar_type bars`. Footer **"Open symbol"** routes to the real Symbol candle route. The prototype's "Stage for execution" CTA is **removed** — v1 is read-first.

**1. Reason** *(✅ Live)* — the `reason` NL string in a `surface-1` card. The richest real field; always present.

**2. Decision chain** *(mixed)* — a six-step `DecisionStepper`, the product's spine:

`Signals → Meta → Calibrated → ⏳Regime fit → Bet size → Target`

- **Signals** (`signal_count`) ✅ · **Target** (`target_weight`) ✅ · **Bet size** (`bet_size`) ✅ when present.
- **Meta / Calibrated** ⚠️ model-gated — real dial when a model is loaded, otherwise the model-gated treatment.
- **Regime fit** ❌ renders a `ComingState` cell in-place (clock glyph, tap → *"Coming when RegimeDetector is wired into the live cycle · Wave 6"*).

The chain reads as a complete 6-step decision spine; the gap is visibly **pending**, never a `0.00`.

**3. Model conviction** *(model-gated + coming)* — two `ProbRing`s (Meta, Calibrated) and a track-record slot.
- **Model loaded:** real rings. Footnote: *"raw == calibrated — the pipeline exposes one calibrated scalar today."*
- **`MODEL_REQUIRED`:** rings render the **lock-glyph "Load model"** state (not `—`, not a 0-filled ring — the fix to the prototype's `ProbRing` rendering `'—'`). The trust-bar **Model pill** reads `MODEL REQUIRED` (warn).
- **Track record** slot → `ComingState`: *"Coming when an append-only call-history store exists — the snapshot is overwritten each publish · Wave 6."*

**4. Signals** *(✅ Live)* — per-signal rows: `family` · side pill · `confidence` · metadata chips. Only the ~4 live bars-families fire (ts_momentum, mean_reversion, ma_crossover, donchian) on a single symbol — render exactly what's present; **don't pad to 10 families**.

**5. Bet-sizing cascade** *(❌ ComingSoon — Wave 4)* — a dimmed five-row `CascadeBars` **ghost** (no numbers) + unlock card: *"Coming when the engine surfaces `constraints_applied` at the idea boundary · Wave 4."* The waterfall's shape is legible; no fake binding constraint.

**6. Why — SHAP** *(❌ ComingSoon — Wave 4)* — a dimmed `MiniBars` ghost + *"Coming when `shap_importance` is persisted per idea · Wave 4."*

**7. Execution & timing** *(mixed)* —
- **Pre-trade cost** → `DataUnavailable` inline (em-dash + `?`, tooltip *"No cost-forecast service wired yet"*).
- **Pipeline latency** ✅ Live — per-stage bars from `stage_latency_seconds` + total. (`bars_loaded`/`feature_rows` counts are **Pro-density only**; `feature_rows` is a count, never per-feature values.)

### C. The seven data-states on this screen

Every value passes through `<DataState>`; first match wins. **`—` is reserved for a legitimately-empty live value** (a WATCH idea's `target_weight=0`). An **absent** field is *never* `0`/`—`/`N/A` — it is `ComingState`/`DataUnavailable` with its real unlock copy.

- **Loading** — Skeleton shimmer on tiles/rows/rings (no spinner; `prefers-reduced-motion` → static fill).
- **Empty** — "No ideas above the entry gate this cycle." / per-filter empties.
- **Error** — per-element inline card + retry + copyable `req: {request_id}` (**requires the one envelope addition: `request_id`**); one row's error never blanks the screen.
- **Stale** (`staleness_seconds > 90`) — last-good dimmed, amber top-border, Freshness pill → `STALE · as of {relative}`, dismissible banner. Reversible, not an error.
- **Not yet available** — the dignified state, two flavors: **model-gated** (conviction, dynamic unlock = "load a model") and **roadmap-gated** (regime-fit, cascade, SHAP, cost, track-record — each with its verbatim `data_readiness.md` unlock + wave).

### D. Interactions
Row → drawer (web slide-in `0.26s`) / push (iOS, swipe-back). Coming steps tap → unlock card. Trust-pill deep-links (Model → Model & Features; Freshness → Overview). Pro-density reveals diagnostics + tightens rows but **never turns a coming slot into a number**. "What changed" strip is dismissible; per-row badges persist the cycle. All motion respects `prefers-reduced-motion`.

### E. Required BFF change
Add `request_id` to `ApiEnvelope` (`src/web/envelope.py`) so the **Error** state can render the promised copyable id. No other backend change is needed for this screen — and critically, **do not** wire `GET /trade-ideas/{symbol}`; the drawer is satisfied entirely by the list payload in v1.

---

## 5.4 Symbol Detail — `/symbols/{symbol}`

**Wave 1 (candles + bar microstructure, LIVE) · Wave 2 (engine read, model-gated)**
Detail route — reached from a Markets row or an idea's "Open symbol", never a sidebar item (`PARENT: symbol → markets`). iOS: push `SymbolScreen` from the Markets tab or the Idea screen.

### Purpose
Bridge "a ticker" to "what the engine thinks of it." This is the **only screen carrying a first-class real chart today** (candles from the `bars` hypertable), so it does two honest jobs at once: show the **price truth** (OHLC + volume + the microstructure the engine actually persists *on each bar*), and show the **engine's current read** (action / reason / target + model-gated probability) with a one-tap bridge to the full decision chain.

> **The load-bearing distinction this screen must get right.** *Bar-level microstructure* — `vwap`, `dollar_volume`, `tick_count`, `buy/sell_volume`, `volume_imbalance`, `tick_imbalance_ratio`, `imbalance`, `threshold`, `bar_duration_seconds` — is **real**; it lives on every `Bar` row (`src/data_engine/models.py:84–123`). The *feature-factory "feature grid"* — GARCH vol, realized vol, RSI-14, order-flow imbalance, Kyle λ, VPIN, Amihud, Roll spread — is **computed-then-discarded** (`FeatureStore.save_features` has zero callers) and must **never** render as a live number. Today's prototype fabricates all eight from a hash of the symbol string; v1 deletes that and renders the real microstructure plus a dignified "coming" panel for the factory features.

### Anatomy (top → bottom)

**1 · Price hero.** `AssetGlyph` + last price (34px mono, tabular-nums) + 1D `Delta` (arrow + pos/neg, never color-alone) + name · asset class. Source: snapshot `latest_price`; falls back to the bars table's last close when the symbol has no active idea — price never degrades to a placeholder. Right-aligned: `Segmented` Candles/Area · `Segmented` 1M/3M/6M.

**2 · Chart — the one real chart.** `CandleChart` (custom SVG OHLC + volume + dashed last-price tag) is the **default** on this screen. Source: `bars` hypertable. The x-axis is **bar index, not wall-clock** — honest for event-driven (imbalance) bars, which are unevenly spaced in time. Candles↔Area toggles the projection over the same rows; Area is the consumer-soft view, candles the engine-true view.

**3 · Change strip.** `1W / 1M / YTD` `Delta`s (computed client-side from the loaded bar series) + **Bar type** — the real `Bar.bar_type` enum (`tib / vib / dollar / volume / tick / tick_run / time`), with an info affordance explaining the type and why the axis is bar-indexed. This replaces the prototype's hardcoded `crypto ? 'dollar' : 'tib'`.

**4 · Bar microstructure** *(real — the honesty centerpiece).* A grid of genuine per-bar stats: **VWAP · dollar volume · tick count · bar duration** (latest bar), and an imbalance row — **buy/sell volume · volume imbalance · tick-imbalance ratio · cumulative imbalance · threshold**. Mono, right-aligned, tabular-nums. Collapsed behind a disclosure at Comfort density; **expanded inline at Pro density**. For bar types where an imbalance field is legitimately `0` (e.g. `volume_imbalance` on a time bar), that `0` is shown as a **real value** with quiet "flat / n/a for {bar_type} bars" copy — an Empty state, *not* a coming-state.

**5 · What the engine sees.** `ActionPill` + NL `reason` + `strategy` + top-signal side (all PRODUCED). Rings: **Meta / Calibrated** `ProbRing` — *model-gated* (real GBM output only when an MLflow production model is loaded; else the model-gated state, `action=MODEL_REQUIRED`, pill reads "Model?"). Per `data_readiness.md` the live pipeline collapses meta and calibrated to one scalar — show a **single ring** at Comfort ("Model probability"); expose both labels only in Pro; never imply a raw-vs-calibrated split the data lacks. Target weight (signed, arrow) is live; **Pre-trade cost** and **Regime-fit** render as honest gaps (below). A "Full decision chain" chip bridges to the `IdeaDrawer` (web) / Idea push (iOS) for the same `TradeIdea`. When there's no idea this cycle → the legitimate Empty line: *"No active idea for {symbol} this cycle. The signal battery produced no actionable edge above the entry gate."*

### Honest gaps (held in shape, never faked)
| Element | Treatment | Unlock condition (verbatim) |
|---|---|---|
| **Live feature grid** (GARCH/RSI/VPIN/Kyle-λ/Amihud/Roll) | `ComingSoon` panel — dimmed 8-tile ghost, labels visible, values in `.coming` slots; copy distinguishes it from the *real* microstructure above | Unlocks when the engine persists computed features (a `save_features` caller exists) · **Wave 4** |
| **Why — SHAP** | `ComingSoon` ghost of `MiniBars` | Unlocks when `shap_importance` (TreeSHAP) is persisted at the idea boundary (real code, never invoked) · **Wave 4** |
| **Regime-fit ring** | inline `DataUnavailable` dashed dial beside the live rings | Unlocks when `RegimeDetector` is wired into the live cycle (zero runtime callers; `regime` null at BFF) · **Wave 6** |
| **Pre-trade cost** | inline `DataUnavailable` (`—?` slot, not a bare `—`) | Unlocks when a cost service writes `expected_cost_bps` (none exists; null at BFF) · **Wave 5** |
| **Track record** | `ComingSoon` strip | Unlocks when an append-only call-history store exists (snapshot is overwritten each publish) · **Wave 6** |

Each holds its layout slot (`surface-inset` + dashed `border-strong` + `text-3` + clock icon), is never colored good/bad, and names the **real** gate — turning the data gaps into a visible roadmap. Critically, the prototype's bare `'—'` for absent cost/regime is replaced by the `.coming` affordance; `—` is reserved for legitimately-empty *live* values only.

### States
- **Loading** — `Skeleton` for the chart footprint, change strip, and microstructure tiles (shimmer; static under `prefers-reduced-motion`); cached last-good shown instantly on return visits with a subtle refreshing affticator. Never a spinner inside content.
- **Empty** — (a) no idea → calm centered line, chart/price/microstructure stay live; (b) a microstructure field that's genuinely `0` for the bar type → real `0` with "flat / n/a" copy.
- **Stale** (`staleness_seconds > 90`) — last-good candles + price dimmed to `text-2`, amber "As of {rel} · refreshing" chip + thin `warn` top-border. Reversible, still-shown, not an error.
- **Error** — per-element only. Bars error → chart slot shows inline "Couldn't load bars. Retry" + copyable `req:{id}`, while the engine-read card stays up; snapshot error → only "What the engine sees" shows the error. One field never blanks the screen.
- **Not-yet-available / Model-gated** — see the gaps table; model-gated rings flip Live the moment a production model loads.

### Interactions
Candles↔Area · timeframe slice · chart hover (per-bar OHLC + real VWAP/dollar-vol/imbalance at Pro) · bar-type info tooltip · "Full decision chain" → drawer/push · microstructure disclosure (collapsed Comfort / expanded Pro) · Pro-density (candles default, raw microstructure surfaced, absolute timestamps + exact `staleness_seconds`, never un-gates absent data) · per-element retry. iOS: collapsing large title, swipe-back, pull-to-refresh, haptic on the chain push. **Parity:** identical data-reality on web and iOS; only the container differs (drawer vs push; iOS may start area-only on narrow windows).

### Data contract
- **LIVE today:** `bars` hypertable (OHLCV + `vwap`, `dollar_volume`, `tick_count`, `buy/sell_volume`, `buy/sell_ticks`, `imbalance`, `threshold`, `bar_duration_seconds`, `bar_type`); trade-ideas snapshot (`action`, `reason`, `strategy`, `target_weight`, `estimated_quantity`, `target_notional`, `top_signal_*`, `stage_latency_seconds`); model-gated `meta_probability` / `calibrated_probability`.
- **BFF work (Wave 1):** add the symbol bars/microstructure read route — there is no markets/symbol route in `src/web` yet; the data exists, the endpoint is the wiring. `GET /trade-ideas/{symbol}` is a 501 stub today, so the engine-read card sources the symbol's row from the list snapshot until that detail endpoint lands.
- **ABSENT (route through coming-states, never fake):** feature-factory features, per-row SHAP, regime-fit, expected cost, per-symbol track record.

---

## Strategies — grid + detail (Wave 1, partial)

> **Readiness:** the family roster, each family's thesis/params/source/asset-classes, an honest active-vs-dormant status, and the live ideas each active family produced this cycle are **LIVE**. Every per-strategy performance number (Sharpe, win, P&L-share, allocation, equity curve, regime-fit) is **COMING** and rendered as a dignified locked state with its exact engine unlock condition. **No fabricated metric ships.**

### Purpose
Show all ten signal families at a glance, make it obvious **which are actually firing right now**, and give each family's full thesis on demand — without inventing a single performance statistic. This screen's defining job is honesty: the prototype today shows fabricated Sharpe/win/PnL/equity/regime-fit for all 10 families and marks every one "live." v1 corrects both lies — 6 of 10 families read as **dignified INACTIVE** (they can't fire on today's single-symbol bars path), and all backtest-derived metrics are replaced by `ComingSoon` panels carrying their real unlock conditions.

### The active / dormant truth (data-driven, not hand-typed)
Status is derived from each generator's **dispatch `kind`** in `src/signal_battery/orchestrator.py`. The deployed cycle supplies only single-symbol `bars`; families needing extra context never fire:

| Family | `kind` | v1 status | Why dormant |
|---|---|---|---|
| Time-Series Momentum | `bars` | **Active** | — |
| Mean Reversion (O-U) | `bars` | **Active** | — |
| MA Crossover | `bars` | **Active** | — |
| Donchian Breakout | `bars` | **Active** | — |
| Cross-Sectional Momentum | `panel` | Inactive | needs `multi_asset_prices` panel feed |
| Statistical Arbitrage | `pair` | Inactive | needs a cointegrated `stat_arb_pair` |
| Futures Carry | `bars_extra` | Inactive | needs `futures_curve` context |
| Volatility Risk Premium | `bars_extra` | Inactive | needs `vol_features` context |
| Funding-Rate Arb | `bars_extra` | Inactive | needs `funding_rates` feed |
| Cross-Exchange Arb | `exchange_prices` | Inactive | needs multi-venue prices |

`4 active · 6 inactive` is computed from a single `familyReadiness` map (the machine-readable twin of `data_readiness.md` §bug-5). When a context feed is wired, a family flips Active by changing one entry — not by editing the screen.

### Grid card anatomy
A card is **fully rendered for every family, active or dormant** — the only difference is the status chip. Inactive must read *intentional*, never broken.

- **Header:** name · category color-dot + label · `StatusDot` (green pulsing = Active) or neutral **"Inactive — no live feed"** chip (reason on hover).
- **Thesis (truncated):** real static description of what the family does.
- **Asset-class pills:** tinted `equity / index / crypto / future` coverage.
- **Live footprint:** count of ideas this family produced this cycle (real, from the snapshot). Active families link into their idea rows; dormant families show "No live ideas — feed not wired."
- **Performance slot:** a single `ComingSoon` strip — *"Performance — coming when backtests are persisted · Wave 5"* — replacing the prototype's fabricated Sharpe/Win/P&L trio, equity sparkline, and allocation bar. **No `0.00`, no fake bar fill.**

Header line: **`4 active · 6 inactive`** (replaces the prototype's fabricated "X live · Y shadow").

### Detail route (`/strategy/{id}` · iOS push)
Reachable for **every** family, including dormant ones — a dormant family is explorable, not a dead link.

- **Thesis hero (LIVE):** full thesis · source attribution (book/author) · asset-class pills · category chip · status chip (with dormancy reason for inactive families).
- **Parameters (LIVE):** mono config table — entry/exit z, lookbacks, channels, thresholds — mirroring the generator's real defaults.
- **Active ideas from this family (LIVE):** `.tbl` rows — `ActionPill` · symbol · calibrated prob (`ProbRing`, **model-gated** → "Model?" when no production model) · target weight. Rows → Symbol detail. Two distinct empty states: *"No live ideas above the entry gate this cycle"* (active, quiet) vs *"No live ideas — context feed not wired"* (dormant).
- **Track record & performance (COMING):** one `ComingSoon` panel absorbing Sharpe / win / P&L-share / YTD / allocation / avg-hold / trades — *"coming when backtests are persisted and the retrain gate is fixed (`retrain_pipeline.py:265`) · Wave 5."*
- **Equity curve (COMING):** dimmed `AreaChart` **wireframe ghost** (axes only, no plotted line) + the same Wave-5 copy. Conveys eventual shape; invents zero data points.
- **Regime fit (COMING):** the four regime labels (`Trending ↑ / Trending ↓ / Mean-revert / High vol`) as **unfilled** `RegimeBar` rails + *"coming when the regime detector is wired into the live cycle · Wave 6"* (`RegimeDetector` has zero runtime callers today). Labels stay so the layout reads intentional; bars are visibly empty, never a made-up %.

### States
- **Loading:** static catalog renders instantly (config, no fetch); the live-ideas overlay shows `Skeleton` shimmer matching idea-row footprints.
- **Empty:** two deliberately-worded zero-states (active-but-quiet vs dormant-no-feed) so the user can tell which.
- **Error:** per-element — a `/trade-ideas` failure still renders all 10 cards with thesis/params/status; only the live overlay shows inline *"Couldn't load live ideas. Retry"* + copyable `req:{request_id}`.
- **Stale:** `staleness_seconds > 90s` dims live-idea values to `--text-2`, adds a `--warn` top-border and *"As of {t} · refreshing"*. Last-good counts stay. Static catalog never stales.
- **Not yet available:** the defining state — every performance metric, the equity curve, and regime-fit routed through `ComingSoon` with the **exact** `data_readiness.md` unlock condition. Hard rule: never `0` / `—` / `N/A` for an absent metric; `—` is reserved for legitimately-empty LIVE values only.

### Interactions
Card → detail (works for dormant families) · category filter (client-side) · active-ideas row → Symbol · hover a dormant chip → precise unlock reason · hover a `ComingSoon` clock → verbatim unlock condition + Wave # · **Pro density** surfaces params + the required-context-key inline (never unlocks a coming metric) · pull-to-refresh re-fetches the live overlay · client-side diff badges new/flipped ideas (scoped to PRODUCED fields; an Active→Dormant flip is *not* flagged as a market change).

### What changed vs the prototype (the hardening)
1. **Deleted** fabricated `s.sharpe / s.winRate / s.contributionPct / s.pnlYtd / s.allocation` and the RNG `equityCurve` sparkline/area — they are not engine output.
2. **Replaced** the blanket `live / shadow` pills with a **data-driven Active/Inactive** status from generator `kind`.
3. **Replaced** the fabricated `s.regimeFit` bars with empty rails + a regime `ComingSoon`.
4. **Added** an honest two-way Empty state and a dormancy reason so an inactive family reads intentional.
5. **Kept** everything genuinely real — roster, thesis, params, source, asset classes, and the live ideas each active family is producing right now.

---

## Model & Features (`/model`) — thin v1

**Group:** ENGINE · **Readiness:** Wave 2 (partial, model-gated) · **Web:** sidebar route · **iOS:** More → push · **Container parity:** identical data on both platforms; iOS condenses to header + histogram + timeline, coming panels stacked.

### Purpose
The trust surface for the meta-labeler. It answers one question at a glance — *is the brain loaded, and what is it predicting right now?* — using only the model facts the engine actually persists. Most of this screen's eventual depth (calibration, importance, drift, RL shadow) isn't wired yet, so v1 is deliberately **thin and honest**: a small real core, and every gap shown as a dignified "coming" panel with its exact unlock condition. This is not the research console it will become; it is the smallest honest version of one.

> **Design stance:** This screen is *mostly coming*, and the design says so out loud. The win is not hiding that — it's making "loaded + predicting + the gaps named" feel finished. A user should leave trusting the app *more* for being candid about what it can't yet show.

### Layout (top → bottom)
1. **Model header** *(real, model-gated)* — `meta-labeler v{n}` (mono) + `production` alias pill; `{type} · trained {date} · {run_id} · {age} ago`; right-aligned `Stat` strip: **CV score** (mean·5-fold), **Train acc** (in-sample), **Events** (`n_training_events`). All from `ModelRegistry.get_production_model()` / MLflow run metadata. **AUC / Brier / ECE are removed** — they were never logged and the mock fabricated them.
2. **Meta-probability distribution** *(conditional-real)* — histogram of `calibrated_prob` over [0,1] buckets from the `meta_labels` hypertable. Empty (not coming) when the table has no rows for the loaded `model_version`.
3. **Promotion gate** *(real fields, honestly false)* — CPCV / DSR / PBO from the production run's `gates`. Rendered as neutral **"not run"** with a warn note: gates are uniformly `false` because the retrain gate is hard-broken (`retrain_pipeline.py:265`). Never green-pass, never red-fail.
4. **Retrain timeline** *(thin real)* — vertical stepper of MLflow runs: promoted/rejected dot, `cv_score`, trigger, rejection reason. Every node reads "gates not run" (same upstream cause).
5. **Coming panels** *(not-yet-available)* — `Calibration reliability` (Wave 6), `Feature importance` (Wave 4), `Feature drift` (Wave 6), `RL shadow` (Wave 6); `Regime detector` + `Per-idea SHAP` collapsed/lowest-priority. Each holds its slot with a dimmed ghost + verbatim unlock condition.

### Real vs coming (ground truth)
| Element | Source | State |
|---|---|---|
| Version · alias · type · trained · run id · age | `get_production_model()` + envelope `model_version` | **Live** (model-gated) |
| CV score · train acc · training events | MLflow `metrics`/`params` | **Live** |
| Meta-prob histogram | `meta_labels` hypertable (`get_meta_labels`) | **Live** if written, else **Empty** |
| CPCV / DSR / PBO flags | `get_production_model()['gates']` | **Live value, "not run"** (false-because-broken-gate) |
| Retrain timeline | MLflow `search_runs` | **Live (thin)** |
| Calibration reliability / ECE / Brier | — no calibration history persisted | **Coming · Wave 6** |
| Feature importance (MDI/MDA/SFI/SHAP) | `get_feature_importance` exists, never persisted | **Coming · Wave 4** |
| Feature drift (KL/KS) | hardcoded `1.0`, no baseline | **Coming · Wave 6** |
| RL shadow comparison | no runnable producer | **Coming · Wave 6** |
| Regime detector panel | `RegimeDetector` zero callers | **Coming · Wave 6** |

### States
- **Loading** — header + histogram = `Skeleton`; coming panels render immediately (static).
- **Empty (no model)** — whole screen → one calm panel: `Model?` pill, "No production model loaded", explains `MODEL_REQUIRED` and that meta/cal probabilities are unavailable; timeline below may still list past runs.
- **Empty (no predictions)** — histogram alone shows "No predictions recorded yet for this model." (producer exists, legitimately empty — *not* a coming panel).
- **Stale** — `staleness_seconds > 90`: last-good header dimmed, warn top-border, "as of {t} · refreshing". Timeline exempt.
- **Error** — per-element inline Retry + copyable `req:{request_id}`; never blanks the screen.
- **Not yet available** — the four+ ComingState panels; gate flags as neutral "not run" with the `retrain_pipeline.py:265` note.

### Interactions
- Trust-bar **Model pill** deep-links here; **run-id** click copies the full MLflow `run_id`.
- **Gate dot** tooltip explains *why* false ("did not run" ≠ "failed validation").
- **Timeline node** expands inline (trigger, rows, rejection reason; +per-fold CV & hyperparams under Pro).
- **Pro density** reveals the hyperparameter / per-fold-CV row + absolute timestamps + a trigger `Segmented` filter — and *never* turns a coming panel into a number.

### Honesty rules enforced on this screen
1. **Delete every fabricated number** from the current mock (AUC/Brier/ECE, calibration observed rates, importance values, drift severities, RL Sharpe). They have no producer.
2. **A real-but-broken flag is "not run", not "fail"** — the false CPCV/DSR/PBO booleans are shown with their upstream-bug provenance, the one subtle truth this screen must get right.
3. **Coming ≠ Empty** — absent capabilities get `ComingState` + unlock condition; a wired-but-empty `meta_labels` table gets `Empty`. The histogram is the boundary case that proves the rule.
4. **One BFF addition:** add `request_id` to `ApiEnvelope` so the per-element Error state can render its copyable id. The `/model` route reads `ModelRegistry.get_production_model()` + `get_meta_labels(model_version)` + `search_runs`, mirroring the `/overview` degrade-don't-500 pattern (`src/web/routes/overview.py`).

---


---

## Part III — Deferred screens (designed as "coming")

These appear in the nav (never as dead links) and open to a dignified ComingSoon panel showing the
real value-to-be plus its unlock condition. They are **not** built on faked data. Unlock conditions
trace to [data_readiness.md](data_readiness.md).

| Screen | Unlocks when | Wave |
|--------|--------------|------|
| **Portfolio & Risk** | the engine persists positions / NAV / `portfolio_snapshots` (today `_save_snapshot` is a no-op; deployed path never routes orders) | 5 |
| **Research & Backtests** | backtest runs are persisted with run ids **and** the broken retrain gate is fixed (gate flags are uniformly `0/False` today) | 5 |
| **Execution & TCA** | an order-routing path writes `ExecutionStorage` orders/fills (today `live_orders_sent=0`, zero rows written) | 5 |
| **Monitoring & Alerts** | the pipeline metrics registry is exposed over HTTP (today nothing scrapes it) and alerts are persisted | 6 |
| **Scenarios & Stress** | a real scenario engine wires `factor_risk` + the sizing cascade (today the service returns mock numbers) | 6 |
| **Track Record** | an append-only published-idea + realized-outcome store exists (the snapshot is overwritten each publish) | 6 |
| **Replay / Time-Travel** | the HMAC audit chain is actually written (`ComplianceAuditLogger` is never instantiated) | 6 |

Each deferred screen's ComingSoon panel uses the ghost-component treatment defined in the
disclosure foundation (a dimmed shell of the real component + the unlock line), so the user sees
*what is coming*, framed as a roadmap, not a broken page.

---

## Part IV — Build sequencing

Screens light up as the engine's data lands. The frontend can be **built in full on mock data now**
(as the prototype already is); these waves describe when each screen flips from mock → live.

| Wave | Screens that go live | Gated on |
|------|----------------------|----------|
| **1 — now** | Markets · Trade Ideas (list) · Symbol Detail (candles + bar microstructure) · Overview (action counts, top ideas, stage latency) | the `bars` table + the trade-ideas snapshot — both exist today |
| **2 — model loaded** | Decision Drawer conviction (meta/calibrated) · Model & Features (thin) · Strategies (live ideas per family) | a registered MLflow production model (else `MODEL_REQUIRED`) |
| **4 — feature/SHAP bridge** | Symbol live-feature overlay · drawer sizing-cascade waterfall + SHAP | a `save_features` caller + surfacing `constraints_applied`/`shap_importance` at the idea boundary |
| **5 — execution/portfolio/backtest persistence** | Portfolio & Risk · Execution & TCA · Research & Backtests · Overview NAV/equity tiles | an order-routing path that writes `ExecutionStorage`; backtest-run persistence; the retrain gate fixed |
| **6 — net-new stores** | Monitoring · Scenarios · Track Record · Replay · regime/cost everywhere | metrics-over-HTTP, scenario engine, call-history store, the audit chain |

(Wave 3 from the engine audit — an on-demand Preflight screen — was reclassified after triage:
preflight is the live-capital go/no-go gate and its infra probes can fire alerts, so it is treated
as money-adjacent and deferred, not a cheap win.)

---

## Part V — How "simple but huge" holds

The whole product is reconcilable to four disclosure levels (detailed in the disclosure
foundation): **L1 glance** (Overview / list rows answer "what now"), **L2 row** (a row's inline
chips), **L3 drawer/detail** (the full decision chain), **L4 pro density** (the cockpit-grade field
firehose, opt-in). A first-time user lives at L1–L2 and never sees the firehose; a power user flips
pro density on. Nothing is removed to achieve simplicity — it is deferred down the ladder. That is
the answer to "can it be both huge and simple": it is huge in depth, simple in default surface.

*Aperture is a placeholder brand; palette and naming are easy to change. This doc + the prototype
are meant to be iterated together.*
