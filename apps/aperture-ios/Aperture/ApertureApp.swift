// Aperture — native SwiftUI app for the wang_trading quant engine.
// Decision support + monitoring, data-honest v1. Shares tokens.json with the web
// app (ApertureTokens.generated.swift) and consumes the same FastAPI BFF.
//
// Appearance FOLLOWS THE SYSTEM: every color is an adaptive token (Tok.*), so the
// app is light by day and dark at night automatically (Settings → Display). We do
// NOT pin `.preferredColorScheme`, and we let the nav/tab bars use native adaptive
// chrome (a custom opaque nav appearance also suppressed large titles on iOS 26).
import SwiftUI

@main
struct ApertureApp: App {
    var body: some Scene {
        WindowGroup {
            RootTabView()
                .tint(Tok.accent)
        }
    }
}

// MARK: - Navigation destinations (registered on every NavigationStack)
extension View {
    func apertureDestinations() -> some View {
        self
            .navigationDestination(for: Sym.self) { SymbolDetailView(sym: $0) }
            .navigationDestination(for: TradeIdea.self) { IdeaDetailView(idea: $0) }
            .navigationDestination(for: Strategy.self) { StrategyDetailView(strategy: $0) }
            .navigationDestination(for: ScreenSpec.self) { spec in
                if spec.id == "model" { ModelView() } else { ComingView(spec: spec) }
            }
    }

    /// The app backdrop behind a scroll view.
    func apertureBackground() -> some View {
        self.background(Tok.bg0.ignoresSafeArea())
    }
}
