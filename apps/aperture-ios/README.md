# Aperture — iOS (SwiftUI)

The native iOS app for the `wang_trading` engine — same design language and the **same data-honesty
contract** as the [web app](../aperture-web), in a platform-native container (bottom tab bar +
`NavigationStack` pushes, large titles, native candlestick + Swift Charts). Built to
[`docs/aperture_v1_design.md`](../../docs/aperture_v1_design.md).

## Run

```bash
open apps/aperture-ios/Aperture.xcodeproj   # Xcode 16+ (built/tested on Xcode 26)
# select an iOS 17+ simulator → ⌘R
```

Or from the command line:

```bash
cd apps/aperture-ios
xcodebuild -scheme Aperture -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  CODE_SIGNING_ALLOWED=NO build
```

The project uses Xcode's **file-system-synchronized group**, so new files under `Aperture/` are
picked up automatically — no `.pbxproj` surgery.

## Screens (parity with web)

- **Tabs:** Home (= Overview) · Markets · Ideas · Strategies · More — all **LIVE**.
- **Pushes:** Symbol detail (candles + bar microstructure), Idea detail (the full decision chain),
  Strategy detail — all **LIVE**.
- **More:** Model & Features (LIVE, model-gated) + every **COMING** destination as a grouped list,
  each opening a dignified `ComingView` with its verbatim unlock condition.

Same honesty rules as web: meta/cal probabilities are model-gated (`ComingRing`); regime-fit, sizing
cascade, SHAP, cost, track record, per-strategy performance, NAV/regime are `ComingCard` /
`DataUnavailableChip`, never faked. Bar-microstructure columns (the real `bars` hypertable) and live
idea/decision data render fully.

## Architecture

```
Aperture/
├── ApertureApp.swift          @main · navigation destinations · tab-bar chrome
├── Shell/RootTabView.swift    TabView + TrustStrip (3 honest pills)
├── DesignSystem/
│   ├── ApertureTokens.generated.swift   ← generated from ../../aperture-web/tokens.json
│   ├── Components.swift · Honesty.swift · Charts.swift · Format.swift · FlexWrap.swift
├── Model/                     Models · MockData (seeded mulberry32) · Readiness
└── Screens/                   Home · Markets · Ideas · IdeaDetail · SymbolDetail ·
                               Strategies · StrategyDetail · Model · More · Coming
```

## Tokens

`ApertureTokens.generated.swift` (the `Tok`, `Radius`, `Space` enums + `Color(hex:)`) is **generated**
from the web app's `tokens.json` — run `pnpm -C apps/aperture-web tokens` to regenerate after a token
change so web and iOS stay in lockstep. Don't edit the generated file by hand.

## Notes

- Targets iOS 17+. Consumes the same FastAPI BFF as web (mock data today; replace `MockData` calls
  with a networking layer returning the same model types).
- `RootTabView` reads optional env vars (`APERTURE_TAB`, `APERTURE_PUSH`) to open a tab / seed a
  detail push — inert in normal use, handy for UI tests and screenshots.
- Roadmap (per the design doc): WidgetKit widgets, Lock-screen + Dynamic Island Live Activity, push
  (mirroring the engine's Telegram alerts), Face ID gating, pull-to-refresh against the publisher
  cadence.
```
