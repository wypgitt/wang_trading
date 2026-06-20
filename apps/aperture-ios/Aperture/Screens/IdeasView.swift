import SwiftUI

/// Trade Ideas — the live decision list. Tap a row → the full decision chain.
struct IdeasView: View {
    enum Filter: Hashable { case all, buy, sell, watch, needsModel }
    @State private var filter: Filter = .all
    private let counts = MockData.actionCounts

    private var rows: [TradeIdea] {
        MockData.ideas.filter { i in
            switch filter {
            case .all: return true
            case .buy: return i.action == .BUY
            case .sell: return i.action == .SELL
            case .watch: return i.action == .WATCH
            case .needsModel: return i.action == .MODEL_REQUIRED
            }
        }
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                TrustStrip()

                // Summary tiles — action counts + a freshness tile (parity with web).
                // Gross/Net target dropped: they read on NAV, which the BFF returns null.
                HStack(spacing: 10) {
                    tile("Buy", "\(counts.buy)", Tok.buy)
                    tile("Sell", "\(counts.sell)", Tok.sell)
                    tile("Watch", "\(counts.watch)", Tok.watch)
                }
                HStack(spacing: 10) {
                    tile("Model?", "\(counts.modelReq)", counts.modelReq > 0 ? Tok.warn : Tok.text3)
                    tile("Freshness", Fmt.timeAgo(MockData.trust.stalenessSeconds), Tok.text1)
                }

                SegPicker(options: [(.all, "All"), (.buy, "Buy"), (.sell, "Sell"),
                                    (.watch, "Watch"), (.needsModel, "Model?")], selection: $filter)

                Card(padding: 0) {
                    VStack(spacing: 0) {
                        ForEach(rows) { idea in
                            NavigationLink(value: idea) {
                                IdeaRow(idea: idea).padding(.horizontal, 14)
                            }.buttonStyle(.plain)
                            .accessibilityElement(children: .combine)
                            .accessibilityLabel(rowLabel(idea))
                            if idea.id != rows.last?.id { Divider().overlay(Tok.border) }
                        }
                    }
                }
            }
            .padding(16)
        }
        .refreshable { try? await Task.sleep(for: .seconds(0.4)) }
        .apertureBackground()
        .navigationTitle("Trade Ideas")
    }

    /// Honest a11y summary for a row: only fields the engine actually produces.
    /// Calibrated probability is read out only when present; otherwise it is
    /// announced as model-required (mirrors the ComingRing in the row).
    private func rowLabel(_ idea: TradeIdea) -> String {
        var parts = ["\(idea.symbol), \(idea.action.label), target weight \(Fmt.pctSigned(idea.targetWeight, 1))"]
        if let cal = idea.calibratedProbability {
            parts.append("calibrated probability \(Fmt.pct(cal, 0))")
        } else {
            parts.append("calibrated probability requires a loaded model")
        }
        return parts.joined(separator: ", ")
    }

    private func tile(_ label: String, _ value: String, _ color: Color) -> some View {
        Card(padding: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Eyebrow(text: label)
                Text(value).font(.system(size: 22, weight: .bold)).monospacedDigit().foregroundStyle(color)
            }
        }
    }
}
