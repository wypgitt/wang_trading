import SwiftUI

struct RootTabView: View {
    // Inert in normal use; an env var lets UI tests / screenshots open a tab and
    // optionally seed a detail push. (APERTURE_TAB=0..4, APERTURE_PUSH=idea|symbol)
    @State private var selection = Int(ProcessInfo.processInfo.environment["APERTURE_TAB"] ?? "0") ?? 0
    @State private var ideasPath = NavigationPath()
    @State private var marketsPath = NavigationPath()
    @State private var morePath = NavigationPath()

    var body: some View {
        TabView(selection: $selection) {
            NavigationStack { HomeView().apertureDestinations() }
                .tabItem { Label("Home", systemImage: "house.fill") }.tag(0)
            NavigationStack(path: $marketsPath) { MarketsView().apertureDestinations() }
                .tabItem { Label("Markets", systemImage: "chart.line.uptrend.xyaxis") }.tag(1)
            NavigationStack(path: $ideasPath) { IdeasView().apertureDestinations() }
                .tabItem { Label("Ideas", systemImage: "bolt.fill") }.tag(2)
            NavigationStack { StrategiesView().apertureDestinations() }
                .tabItem { Label("Strategies", systemImage: "square.grid.2x2.fill") }.tag(3)
            NavigationStack(path: $morePath) { MoreView().apertureDestinations() }
                .tabItem { Label("More", systemImage: "ellipsis.circle.fill") }.tag(4)
        }
        .onAppear {
            switch ProcessInfo.processInfo.environment["APERTURE_PUSH"] {
            case "idea": ideasPath.append(MockData.ideas[0])
            case "symbol": marketsPath.append(MockData.sym("NVDA"))
            case "model": morePath.append(Readiness.model)
            default: break
            }
        }
    }
}

/// The three honest trust pills — Mode · Model · Freshness. Bound only to fields
/// the ApiEnvelope actually carries today.
struct TrustStrip: View {
    var trust = MockData.trust
    var body: some View {
        HStack(spacing: 8) {
            pill("PAPER", color: Tok.info, dot: Tok.info)
            pill(trust.modelLoaded ? "MODEL LOADED" : "MODEL REQUIRED",
                 color: trust.modelLoaded ? Tok.pos : Tok.warn,
                 dot: trust.modelLoaded ? Tok.pos : Tok.warn)
            pill(trust.stale ? "STALE" : Fmt.timeAgo(trust.stalenessSeconds),
                 color: trust.stale ? Tok.warn : Tok.text2,
                 dot: trust.stale ? Tok.warn : Tok.pos, live: !trust.stale)
            Spacer(minLength: 0)
        }
    }

    private func pill(_ text: String, color: Color, dot: Color, live: Bool = false) -> some View {
        HStack(spacing: 6) {
            PulseDot(color: dot, live: live)
            Text(text).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(color)
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(Tok.surface2, in: Capsule())
        .overlay(Capsule().stroke(color.opacity(0.35), lineWidth: 1))
    }
}

/// A 7pt freshness dot that pulses its opacity when live (mirrors StatusDot's
/// idiom but keeps the caller-supplied color).
private struct PulseDot: View {
    var color: Color
    var live: Bool = false
    @State private var pulse = false
    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 7, height: 7)
            .opacity(live && pulse ? 0.45 : 1)
            .animation(live ? .easeInOut(duration: 0.9).repeatForever(autoreverses: true) : .default, value: pulse)
            .onAppear { if live { pulse = true } }
    }
}
