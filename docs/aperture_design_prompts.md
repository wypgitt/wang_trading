# Aperture — Design Prompt Pack

A set of copy-paste prompts to hand to **Claude Design** (or any Claude that produces UI
artifacts/mockups) so it designs every screen — look, information, layout, styles, colors — for the
Aperture web + iOS apps.

> Companion to [`aperture_app_design.md`](aperture_app_design.md) (the spec) and the working
> prototype in [`apps/prototype/`](../apps/prototype).

---

## How to use this pack

1. **Start a Claude Design conversation.** Optionally attach `docs/aperture_app_design.md` and a few
   prototype screenshots for extra grounding — but the prompts below are written to be **self-contained**
   so they work even with nothing attached.
2. **Paste the `0 · FOUNDATION` prompt first.** It defines the brand, design system, and sample data.
   Keep it in context for the whole session (re-paste at the top if the conversation gets long).
3. **Then paste one screen prompt at a time** (sections 1–N). Each produces one self-contained HTML
   mockup. Review, then ask for tweaks ("make the hero chart taller", "try a light theme", "tighten
   the table").
4. **For consistency**, design the screens in order and tell Claude *"match the visual language of the
   previous screen exactly."*
5. **To get variations**, append: *"Give me 2 alternative layouts for this screen."*

Each prompt is in its own code block for easy copying.

---

## 0 · FOUNDATION  (paste first, reuse all session)

```text
You are a senior product designer and front-end engineer. You produce high-fidelity, presentation-ready
UI mockups as a SINGLE self-contained HTML file (inline CSS; load Inter and JetBrains Mono from Google
Fonts). Dark theme. Pixel-polished and consumer-grade. No lorem ipsum — use the SAMPLE DATA below.

PRODUCT — "Aperture": a clean, consumer-grade cockpit for a single-operator quantitative trading engine
that trades US equities, indexes, crypto (spot + perpetuals), and futures. It surfaces live trade ideas,
the full model decision chain behind each one, 10 trading strategies, portfolio & risk, backtests, and
the ML model internals. The user is the owner/operator (one person). Mode shown: Paper trading.

AESTHETIC: premium and calm like Robinhood + Linear — generous whitespace, soft rounded cards, subtle
shadows, gorgeous data visualization. NOT a dense Bloomberg terminal. Information-rich but uncluttered;
depth is revealed progressively (a row expands into full detail).

DESIGN SYSTEM (use these exact tokens):
• Theme dark. App bg #0a0c10; section bg #0e1116; cards #14181f with 1px border rgba(255,255,255,.07),
  radius 16–20px, soft shadow 0 6px 24px rgba(0,0,0,.35); elevated #1a1f27; inset wells #0c0f14.
• Text #eef1f6 (primary) / #a3adbb (secondary) / #6c7787 (tertiary).
• Direction & P&L ONLY: positive/up/buy #1ecb8b, negative/down/sell #f6465d. Use 12–14% tints for pill
  backgrounds.
• Brand accent (interactive, model, links, logo): violet #7c5cff → blue #4d9fff (135° gradient for
  primary buttons & the logo mark). Warn #f0a93b. Info #4d9fff.
• Regime colors: trending-up #1ecb8b, trending-down #f6465d, mean-reverting #b07cff, high-vol #f0a93b.
• Strategy category colors: Momentum #4d9fff, Mean Reversion #b07cff, Trend #1ecb8b, Volatility #f0a93b,
  Carry #22d3ee, Arbitrage #f6679a.
• Type: Inter for UI, JetBrains Mono for tickers/params/raw numbers. ALL numbers tabular
  (font-variant-numeric: tabular-nums). Page title 22–30px/700; section 15–18px/650; body 13–14px;
  eyebrow labels 10–11px UPPERCASE, letter-spaced, color #6c7787.
• Pills: full-round. Buy=green tint, Sell=red tint, Watch=blue tint, neutral/Model?=grey tint.
• Never encode meaning by color alone — pair with an arrow, label, or icon.

KEY CONCEPT — THE DECISION CHAIN: every trade idea traces through
  signals → meta-probability → calibrated probability → regime fit → 5-layer bet-size cascade
  → target weight → action → pre-trade cost.
Make this chain legible wherever ideas appear.

COMPONENT VOCABULARY (reuse consistently): action pill; regime stacked-bar + chip; probability ring
(a 0–1 dial); decision-chain stepper (6 small value boxes with arrows); 5-layer sizing-cascade bars
(highlight the BINDING constraint in amber); SHAP signed-contribution bars (green +, red −); allocation
donut; sparkline; smooth gradient area chart; candlestick + volume chart; dense stat strip with dividers.

SAMPLE DATA (use these exact values across every screen so they form one coherent app):
• Portfolio: NAV $1.21M; today +$9.8K (+0.81%); since inception +21.3%. Sharpe 1.84, Sortino 2.61,
  Calmar 2.12, Max DD −8.3%, Ann vol 11.4%, Win 56.3%, Profit factor 1.71, Beta 0.21.
  Exposure Gross 55%, Net +31% (Long 43% / Short 12%). 10 open positions.
• Regime (LSTM detector): Trending-up 71%, Trending-down 7%, Mean-reverting 15%, High-vol 7%.
• Model: meta_v1.7.2 (LightGBM meta-labeler + isotonic calibration), trained 2026-05-12, 62h ago.
  AUC 0.673, Brier 0.214, ECE 0.031.
• Validation gates: CPCV 71% (32/45 combinatorial paths positive; ≥60% required); Deflated Sharpe
  p=0.011 (<0.05 required); PBO 0.28 (<0.40 required). All PASS.
• Trade ideas — [symbol · action · strategy · meta p / calibrated p · regime fit · target wt · cost]:
  NVDA · Buy · ts_momentum · 0.74/0.71 · 0.86 · +9.2% · 4.2bps
  BTC · Buy · ts_momentum · 0.70/0.68 · 0.81 · +8.3% · 6.1bps
  META · Buy · cs_momentum · 0.69/0.66 · 0.78 · +7.1% · 3.4bps
  ETH · Buy · donchian_breakout · 0.64/0.62 · 0.74 · +5.7% · 5.8bps
  TSLA · Sell · mean_reversion · 0.67/0.64 · 0.71 · −5.2% · 4.9bps
  GOOGL · Buy · cs_momentum · 0.61/0.59 · 0.69 · +4.8% · 3.1bps
  SOL · Buy · funding_rate_arb · 0.66/0.63 · 0.70 · +3.9% · 7.4bps
  CL · Sell · futures_carry · 0.58/0.56 · 0.61 · −3.6% · 2.8bps
  AVAX · Watch · donchian_breakout · 0.52/0.50 · 0.55 · — · 8.1bps
• 10 strategies — [name · category · status · Sharpe · win · P&L share · allocation]:
  Time-Series Momentum · Momentum · Live · 1.12 · 54% · 26% · 22%
  Cross-Sectional Momentum · Momentum · Live · 0.94 · 52% · 17% · 15%
  Mean Reversion (O-U) · Mean Reversion · Live · 1.03 · 61% · 14% · 12%
  MA Crossover · Trend · Live · 0.71 · 46% · 7% · 8%
  Donchian Breakout · Trend · Live · 0.88 · 43% · 10% · 10%
  Volatility Risk Premium · Volatility · Live · 1.21 · 64% · 9% · 7%
  Futures Carry · Carry · Live · 0.79 · 57% · 5% · 6%
  Funding-Rate Arb · Carry · Live · 1.42 · 72% · 8% · 6%
  Statistical Arbitrage · Arbitrage · Live · 0.97 · 59% · 6% · 5%
  Cross-Exchange Arb · Arbitrage · Shadow · 1.55 · 81% · 3% · 2%
• Prices: NVDA $118, AAPL $196, MSFT $421, META $504, BTC $61,000, ETH $3,380, SOL $142, SPX 5,430,
  NDX 19,200, VIX 14.2.

OUTPUT RULES: design ONE screen per request as a single self-contained HTML file at the canvas size I
specify; dark background; realistic numbers from the sample data; polished and presentation-ready; keep
the visual language identical across screens. Confirm you've understood, then wait for the screen brief.
```

---

# WEB SCREENS  (desktop canvas 1440 wide)

## 1 · Web — Overview

```text
Using the Aperture design system and sample data, design the WEB — OVERVIEW screen at 1440×1080.

Chrome:
• Left sidebar (244px): logo "✦ Aperture" (gradient mark) + "wang · quant engine"; nav group "COCKPIT":
  Overview (active), Markets, Trade Ideas, Strategies; group "PORTFOLIO": Portfolio & Risk, Research &
  Backtests, Model & Features; footer card: green pulsing dot + "Paper trading · live", "0 breakers ·
  1 alert · data 2.4s".
• Sticky top bar: page title "Overview" + subtitle "Your engine at a glance"; a ⌘K search field;
  a regime chip "● Trending ↑ · 71%"; a live chip "● 8s ago"; NAV "$1.21M / +0.81% today" (green);
  bell icon; round avatar "YW".

Main content (cards, 18px gaps):
1) NAV HERO — eyebrow "NET ASSET VALUE · PAPER"; "$1.21M" at ~40px; "▲ +$9.8K (+0.81%) today" green;
   subline "+1.5% over 3M · since inception +21.3%"; timeframe pills 1W/1M/3M/6M(active)/All on the
   right; a large smooth GREEN gradient area chart (~220px tall).
2) STAT STRIP — one card, 6 equal columns with dividers: Sharpe 1.84 (sub "Sortino 2.61") · Max DD
   −8.3% ("Calmar 2.12") · Ann vol 11.4% ("Beta 0.21") · Win 56.3% ("PF 1.71") · Gross 55% ("Net 31%")
   · Open positions 10 ("9 new ideas").
3) TWO COLUMNS (1.55fr / 1fr): LEFT card "Top trade ideas" + "View all 12" link — 5 rows: [action pill]
   [ticker bold] [strategy grey mono] [small sparkline] [target weight, green/red] [calibrated p, green
   if ≥0.60]. RIGHT column = two cards: "Market regime" (a 4-segment stacked bar, then 4 rows: label +
   thin colored bar + %) and "P&L contribution" (horizontal bars per strategy, colored by category, with
   % at right).
4) "Market movers" card — 5 mini cards in a row: ticker, day % (green/red), sparkline, price.

Deliver one self-contained HTML file. Match tokens exactly; numbers tabular.
```

## 2 · Web — Markets

```text
Using the Aperture design system and sample data, design the WEB — MARKETS screen at 1440×1080 (same
sidebar + top bar chrome as Overview; active nav = Markets; top-bar subtitle "Equities · Indexes ·
Crypto · Futures").

Body:
• A row of filter chips: All (active), Equities, Indexes, Crypto, Futures; right-aligned caption
  "19 instruments · 12 with active ideas".
• One large card containing a header row of eyebrow labels (Instrument / 30D trend / Price / 1D / 1W /
  1M / Mkt cap·Vol) then ~14 instrument rows. Each row: a rounded "asset glyph" tile (tinted by class,
  showing the ticker letters) + ticker (bold) + a small accent dot if it has an active idea + company
  name (grey, smaller); a 120px sparkline (green if up over the window, red if down); price (tabular);
  1D / 1W / 1M percent columns (green/red, right-aligned); market-cap-or-volume (e.g. $2.90T, $7.72B);
  a chevron.
Rows for: NVDA, AAPL, MSFT, GOOGL, AMZN, TSLA, META, JPM (equities); SPX, NDX, RUT, VIX (indexes);
BTC, ETH, SOL, AVAX (crypto); ES, CL, GC (futures). Vary the up/down colors realistically.

Whole rows are clickable (hover highlight). One self-contained HTML file.
```

## 3 · Web — Trade Ideas (table)

```text
Using the Aperture design system and sample data, design the WEB — TRADE IDEAS screen at 1440×1080 (same
chrome; active nav = Trade Ideas; subtitle "The full decision chain, live").

Body:
• 5 summary tiles in a row: Buy 7 (green) "long candidates"; Sell 3 (red) "short candidates"; Watch 2
  (blue) "below entry gate"; Gross target 55% "of NAV"; Net target +31.5% "on $1.21M".
• Filter tabs (chips): All (active), Buy, Sell, Watch, Model?.
• A clean dense data table with columns: Symbol · Action (pill) · Strategy (grey mono) · Top conf ·
  Meta p · Cal. p (bold, green if ≥0.60) · Regime fit · Target wt (green/red) · Notional · Cost · ›.
  Use the sample ideas (NVDA, BTC, META, ETH, TSLA, GOOGL, SOL, CL, JPM, AVAX, AAPL, plus one
  "RUT · Model?" row with em-dashes). Row hover highlight; the › hints a drill-down.

Keep it scannable and calm, not crowded. One self-contained HTML file.
```

## 4 · Web — Idea decision-chain drawer  ★ the centerpiece

```text
Using the Aperture design system and sample data, design the WEB — TRADE-IDEA DECISION DRAWER: a 500px
right-side slide-over panel over a dimmed Trade Ideas table (show the table faintly behind a 55% black
scrim). Canvas 1440×1080. Design it for the NVDA · Buy idea.

Drawer contents, top to bottom (each section separated by a hairline, with an UPPERCASE eyebrow label):
• Header: "NVDA" large + Buy pill; subline "ts_momentum · LONG · tib bars"; an "Open ↗" button and an
  "✕" close.
• Reason card: "Strong 3M time-series momentum (z=1.9), confirmed by MA crossover. Meta-labeler 0.74 /
  calibrated 0.71 in a trending-up regime (fit 0.86)."
• DECISION CHAIN: 6 small value boxes with "›" separators — Signals 2 / Meta p 0.74 (blue) / Calibrated
  0.71 (violet) / Regime fit 0.86 (green) / Bet size 9.2% (amber) / Target +9.20% (green).
• MODEL CONVICTION: two probability rings (Meta 0.74, Calibrated 0.71) + "Historical track record 50%
  win over 40 similar calls".
• SIGNALS (2): card per signal — "ts_momentum" LONG conf 0.79 with metadata chips (lookbacks 21/63/126/
  252, z_63 1.9, aggregate 0.72); "ma_crossover" LONG conf 0.41 (fast_ema 95.83, slow_ema 93.90).
• BET-SIZING CASCADE: 5 horizontal bars decreasing — AFML size 15.6%, Kelly cap 12.4%, Vol target 10.3%
  (BINDING, amber, with a "binding" tag), ATR cap 9.6%, Risk budget 9.2%. Each bar labeled + value.
• WHY (SHAP): 6 signed horizontal bars — ts_mom_63 +0.44 (green), regime_trending_up +0.29, order_flow_
  imbalance +0.19, garch_vol −0.17 (red), rsi_14 −0.14, vpin −0.08.
• EXECUTION & TIMING: pre-trade cost "+4.2 bps", constraints chips ("vol_target"); pipeline latency
  bars summing ~220ms (data_fetch, feature_compute, signal_generation, meta_inference, sizing,
  target_generation).
• Footer: primary gradient button "✓ Stage for execution" + secondary "Symbol detail".

This screen is the product's signature — make it beautiful and information-dense yet legible. One
self-contained HTML file.
```

## 5 · Web — Symbol detail

```text
Using the Aperture design system and sample data, design the WEB — SYMBOL DETAIL screen for NVDA at
1440×1080 (same chrome; top bar shows a back chevron + "NVDA" + "NVIDIA Corp.").

Body:
1) PRICE HERO card: asset glyph + "$118.40" large + "▲ +1.27%" green + "NVIDIA Corp. · Equity"; on the
   right two segmented toggles: Candles|Area and 1M|3M|6M. Below, a CANDLESTICK chart (~340px) with
   up/down candles, a volume sub-panel, faint horizontal gridlines + right-edge price labels, and a
   dashed last-price line with a green price tag "118.40". Under it a 5-column change strip: 1 week
   −4.4%, 1 month −0.2%, YTD +14%, Mkt cap $2.90T, Bar type tib.
2) Two columns (1.5fr / 1fr): LEFT "What the engine sees" — Buy pill + "via ts_momentum · LONG"; the
   reason paragraph; three probability rings (Meta 0.74, Calibrated 0.71, Regime fit 0.86); Target
   weight +9.20% and Pre-trade cost +4.2 bps. RIGHT "Why" — 6 SHAP signed bars.
3) "Live features" card: 8 small stat tiles (GARCH vol, Realized vol (5), RSI-14, Order-flow imbalance,
   Kyle λ, VPIN, Amihud illiq., Roll spread) each with a value + a tiny grey hint.

One self-contained HTML file.
```

## 6 · Web — Strategies grid

```text
Using the Aperture design system and sample data, design the WEB — STRATEGIES screen at 1440×1080 (same
chrome; active nav = Strategies; subtitle "10 signal families across 6 categories").

Body: a row of category filter chips (All active, then Momentum, Mean Reversion, Trend, Volatility,
Carry, Arbitrage — each with its colored dot), right caption "9 live · 1 shadow". Then a 3-column grid
of strategy cards (use all 10 from sample data). Each card: name (bold) + a Live/Shadow pill; a category
row (colored dot + category); an equity sparkline (green/red); a 3-up mini-stat row (Sharpe / Win rate /
P&L share); an allocation bar at the bottom (category-colored) with "X% of gross". Cards lift on hover.

One self-contained HTML file.
```

## 7 · Web — Strategy detail

```text
Using the Aperture design system and sample data, design the WEB — STRATEGY DETAIL screen for "Time-
Series Momentum" at 1440×1080 (same chrome; back chevron + "Time-Series Momentum" + "Momentum · Clenow ·
Chan").

Body:
1) THESIS hero card: a category chip + "source · Clenow · Chan" + mono id "ts_momentum" + asset-class
   pills (Equity, Crypto, Future); then the thesis paragraph: "Per-asset momentum across multiple
   lookbacks, volatility-normalized to z-scores and weighted into a single conviction. Goes long winners
   / short losers. Best in persistent trends."
2) 6-column stat strip: Sharpe 1.12 · Win rate 54% (318 trades) · P&L share 26% · YTD +8.2% ·
   Allocation 22% · Avg hold 34 bars (6 active).
3) Two columns: LEFT "Strategy equity curve" (green area, rebased to 100, baseline line at 100); RIGHT
   "Regime fit" — 4 labeled bars: Trending ↑ 86%, Trending ↓ 62%, Mean-revert 21%, High vol 40%.
4) Two columns: LEFT "Parameters" table (lookbacks 21/63/126/252; history_window 252; vol_normalize
   true); RIGHT "Active ideas from this strategy" list (NVDA Buy +9.2% cal 0.71; BTC Buy +8.3% cal 0.68).

One self-contained HTML file.
```

## 8 · Web — Portfolio & Risk

```text
Using the Aperture design system and sample data, design the WEB — PORTFOLIO & RISK screen at 1440×1080
(same chrome; active nav = Portfolio & Risk; subtitle "Positions · exposure · factor risk").

Body:
1) Two columns: LEFT "Exposure" card — Gross 55% / Net +31% (green) / Long 43% (green) / Short 12% (red);
   a long/short split bar (green 79% | red 21%) with "79% long / 21% short"; a divider; Sharpe 1.84 /
   Sortino 2.61 / Max DD −8.3%. RIGHT "Allocation by asset class" — a donut (center "55% gross") with a
   legend: Equity 33% (blue), Crypto 18% (amber), Future 4% (cyan).
2) "Positions (10)" table: Symbol · Side (pill) · Strategy (mono) · Qty · Entry · Mark · Weight · Notional
   · Unreal. P&L (green/red, with % in parens) · Day. Use: NVDA Long, META Long, BTC Long, ETH Long,
   GOOGL Long, AMZN Long, SOL Long, TSLA Short, CL Short, JPM Short — plausible numbers.
3) Two columns: LEFT "Factor risk model" — a systematic-vs-idiosyncratic donut (68% / 32%, center "11.4%
   ann. risk") + factor exposure bars: Market (PC1) 0.31, Momentum (PC2) 0.58, Size (PC3) −0.18, Crypto-
   beta (PC4) 0.42, Vol (PC5) −0.22. RIGHT "Drawdown" — a red underwater area chart from a running peak.

One self-contained HTML file.
```

## 9 · Web — Research & Backtests

```text
Using the Aperture design system and sample data, design the WEB — RESEARCH & BACKTESTS screen at
1440×1200 (same chrome; active nav = Research & Backtests; subtitle "Walk-forward · validation gates").

Body:
1) 8-column metrics strip: Total return +194% · Ann return +21.3% · Sharpe 1.84 · Sortino 2.61 · Max DD
   −10.1% · Win 56.3% · Profit factor 1.71 · Turnover 8.4×.
2) "Walk-forward equity curve" card: a violet gradient area (Strategy) plus a dashed grey line
   (SPX benchmark), faint gridlines, right-edge y-axis; legend below.
3) "Promotion gates · all three must pass" — 3 cards: CPCV 71% (Pass) "32/45 combinatorial paths positive
   (≥60% required)" with a 45-bar mini distribution (mostly green, a few red); Deflated Sharpe 0.011
   (Pass) "p=0.011 after deflating for trials (<0.05 required)" with a progress bar vs a threshold tick;
   PBO 0.28 (Pass) "<0.40 required" with a progress bar.
4) Two columns: LEFT "Monthly returns" heatmap (rows 2024/2025/2026, 12 month columns, cells tinted
   green/red by magnitude, blank for future months); RIGHT "Performance by regime" — 4 bars: Trending-up
   Sharpe 2.31, Mean-reverting 1.74, Trending-down 1.12, High-vol 0.82.
5) "Recent trade log" table: Symbol · Strategy · Side · Entry · Exit · Hold · Meta p · Return · Net P&L.

One self-contained HTML file.
```

## 10 · Web — Model & Features

```text
Using the Aperture design system and sample data, design the WEB — MODEL & FEATURES screen at 1440×1200
(same chrome; active nav = Model & Features; subtitle "Meta-labeler · calibration · drift").

Body:
1) Header card: "meta_v1.7.2" (mono) + a green "Promoted" pill; "LightGBM meta-labeler + isotonic
   calibration · trained 2026-05-12 · 62h ago"; right side three big metrics AUC 0.673, Brier 0.214,
   ECE 0.031.
2) Two columns: LEFT "Calibration reliability" — a line chart, x=predicted 0→1, y=observed 0→1, a dashed
   y=x reference line + a violet observed curve with dots, gridlines. RIGHT "Meta-probability
   distribution" — a violet bar histogram over buckets 0.0–0.9 (counts rising to ~196 at 0.4 then
   tapering).
3) Two columns: LEFT "Feature importance" — horizontal bars colored by feature family: ts_mom_63 14.2%
   (Momentum/blue), garch_vol 11.8% (Volatility/amber), regime_prob_up 9.7% (Regime/green),
   order_flow_imbalance 8.3% (Microstructure/pink), ffd_close_d04 7.1% (violet), rsi_14 6.4% (cyan),
   vpin 5.8%, cs_mom_decile 5.2%, kyle_lambda 4.7%, sentiment_24h 3.9% — plus a family legend. RIGHT
   "Feature drift" — KL bars with status dots: ts_mom_63 0.04 (ok/green), garch_vol 0.11 (ok),
   order_flow_imbalance 0.27 (warn/amber), vpin 0.42 (alert/red), sentiment_24h 0.09, rsi_14 0.06; note
   "vpin breached the alert threshold — retrain trigger armed".
4) Two columns: LEFT "RL agent · shadow mode" — Agreement 78%, Shadow Sharpe 1.91, Live Sharpe 1.84, and
   a status pill "shadow · auto-revert armed". RIGHT "Retrain timeline" — a vertical timeline: Promoted
   meta_v1.7.2 (Sharpe 1.84, 2026-05-12), Promoted v1.7.1 (1.79), Rejected — PBO 0.46 (red, 1.71),
   Promoted v1.7.0 (1.77).

One self-contained HTML file.
```

---

# iOS SCREENS  (canvas: an iPhone, 393×852, with Dynamic Island + bottom tab bar)

> Tell Claude: render each iOS screen inside a realistic iPhone frame (rounded bezel, Dynamic Island
> pill, status bar "9:41" + signal/wifi/battery, a bottom home indicator) using the SF system font.

## 11 · iOS — Home

```text
Using the Aperture design system and sample data, design the iOS — HOME screen inside an iPhone frame
(393×852, Dynamic Island, status bar, bottom tab bar). Use the SF system font; dark theme.

Content (scrolling, 16px side padding):
• Large title "Overview" + subtitle "Paper · live" + a green pulsing dot.
• NAV hero card: eyebrow "NET ASSET VALUE"; "$1.21M" ~38px; "▲ +$9.8K (+0.81%) today" green; a green
  gradient area chart (~150px); timeframe pills 1W/1M/3M(active)/All.
• A 3-up card row: Sharpe 1.84 · Max DD −8.3% · Win rate 56%.
• "Market regime" card: title + "Trending ↑ · 71%" (green) on the right; a 4-segment stacked bar; a row
  of 4 tiny labels with % (Trending↑ 71, Trending↓ 7, Mean-revert 15, High vol 7).
• Section header "Top ideas" + "All" link; a card with 4 rows: action pill, ticker, strategy (grey
  mono), target weight (green/red), "cal 0.71", chevron.
• Section header "Movers" + "Markets" link; a horizontal scroller of mini cards (ticker, sparkline,
  price, day %).
Bottom TAB BAR (4 items, Home active in blue): Home, Markets, Ideas, Strategies — icon + label.

One self-contained HTML file showing the phone centered on a dark backdrop.
```

## 12 · iOS — Trade idea detail (mobile decision chain)

```text
Using the Aperture design system and sample data, design the iOS — TRADE IDEA DETAIL for NVDA inside an
iPhone frame (393×852). SF font; dark theme.

• Nav header (sticky): a back chevron (blue), centered title "NVDA" + subtitle "ts_momentum", and a
  "Buy" pill on the right.
• Reason card: "Strong 3M time-series momentum (z=1.9), confirmed by MA crossover. Meta-labeler 0.74 /
  calibrated 0.71 in a trending-up regime (fit 0.86)."
• Section "Decision chain": a 3×2 grid of value cards — Signals 2 / Meta p 0.74 (blue) / Calibrated 0.71
  (violet) / Regime fit 0.86 (green) / Bet size 9.2% (amber) / Target +9.20% (green).
• Section "Model conviction": a card with two probability rings (Meta 0.74, Calibrated 0.71) and "50%
  win · 40n".
• Section "Signals (2)": two cards (ts_momentum LONG 0.79 + metadata chips; ma_crossover LONG 0.41).
• Section "Bet-sizing cascade": 5 horizontal bars (Vol target is the binding one, amber).
• Section "Why · SHAP": 6 signed bars.
• Two footer buttons: gradient "Open NVDA" + outline "Stage for execution".
Bottom tab bar (Ideas active).

One self-contained HTML file.
```

## 13 · iOS — Markets, Ideas, Strategies, Symbol (one prompt, 4 frames)

```text
Using the Aperture design system and sample data, design FOUR iOS screens, each inside its own iPhone
frame (393×852), arranged in a 2×2 grid on a dark backdrop. SF font; dark theme; a bottom tab bar on
each (highlight the relevant tab).

A) MARKETS — large title "Markets", a search field, a horizontal chip filter (All/Equities/Indexes/
   Crypto/Futures), then a list card of instrument rows (asset glyph, ticker + name, sparkline, price,
   day %). 6–7 rows.
B) IDEAS — large title "Trade Ideas" + "12 this cycle · +31.5% net"; a 3-up chip row (Buy 7 / Sell 3 /
   Watch 2); a filter chip row; then idea CARDS (not a table): each card has a top row [action pill,
   ticker, strategy, chevron] and a stat row [Cal. p · Target · Cost · a small regime-fit bar].
C) STRATEGIES — large title "Strategies", category chip filter, then strategy cards: name + category
   dot + Live pill + sparkline + a 4-up stat row (Sharpe/Win/P&L share/Alloc).
D) SYMBOL (NVDA) — back-chevron nav header; price hero ("$118.40 ▲ +1.27%"); a green area chart with
   1M/3M/6M pills; a 3-up change row (1W/1M/YTD); a "What the engine sees" card (Buy pill, reason,
   three rings, target + cost).

One self-contained HTML file with all four phones.
```

---

# CROSS-CUTTING PROMPTS

## 14 · Design system / style guide sheet

```text
Using the Aperture design system above, produce a STYLE GUIDE artboard (1440 wide, dark) that documents
the system: a color palette (swatches with hex + names for surfaces, text, semantic up/down, brand
gradient, regime, category); the type scale (Inter + JetBrains Mono samples, tabular-number demo);
spacing & radius tokens; elevation/shadow samples; and a COMPONENT SHEET showing every component in all
states: action pills (Buy/Sell/Watch/Model?), probability ring, regime stacked-bar + chip, decision-
chain stepper, sizing-cascade bars (with a binding one), SHAP bars, allocation donut, sparkline, a
gradient area chart, a candlestick snippet, buttons (primary gradient / secondary / icon), chips, and a
table row. One self-contained HTML file — this becomes the single source of truth for all screens.
```

## 15 · States (loading / empty / error / stale)

```text
Using the Aperture design system, design the SCREEN STATES for the Trade Ideas table at 1440 wide, dark,
as four stacked panels: (1) LOADING — skeleton/shimmer rows; (2) EMPTY — calm illustration + "No ideas
above the entry gate this cycle" + a subtle explanation; (3) ERROR — inline error card with a retry
button and a request-id; (4) STALE — the normal table with an amber banner "Data is 94s old — last
good pull 18:45Z" driven by source freshness. One self-contained HTML file.
```

## 16 · Marketing / hero shot (optional)

```text
Using the Aperture design system and sample data, design a HERO MARKETING SHOT (1600×1000, dark): the
Aperture web Overview floating at a slight 3D tilt next to the iOS Home screen in an iPhone frame, with
a short headline "Your quant engine, in focus." and 3 feature bullets (Live trade ideas with the full
decision chain · 10 strategies, explained · Beautiful markets across equities, crypto & futures). Use
the brand violet→blue gradient as an accent glow. One self-contained HTML file.
```

---

## Tips for great results

- **Order matters.** Do `0 · FOUNDATION` → `14 · style guide` → screens 1→10 → iOS 11→13. Designing the
  style guide early locks the visual language so every later screen matches.
- **Force consistency.** Start later prompts with *"Match the exact visual language, spacing, and colors
  of the previous screens."*
- **Iterate in place.** After a mockup, ask for specific changes rather than regenerating from scratch:
  *"hero chart +60px; move the regime card above ideas; try the accent as teal."*
- **Variations.** *"Give me a second version with a denser table"* or *"…a light-theme variant."*
- **Export.** Ask for *"a single self-contained HTML file I can open in a browser"* (already requested) —
  then screenshot at the canvas size for your design library, or feed back into the prototype.
- **Ground further (optional).** Attaching `docs/aperture_app_design.md` and a couple of prototype
  screenshots makes the output even more faithful, but isn't required.
```
