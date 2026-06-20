import SwiftUI

/// Home (= Overview). The L0 "what now?" glance — a decision dashboard, not a
/// money dashboard, since the engine produces a decision-flow snapshot. NAV /
/// equity / regime are dignified-coming, never faked.
struct HomeView: View {
    private let counts = MockData.actionCounts
    private let top = Array(MockData.topActionable.prefix(5))
    private let pulse = MockData.enginePulse

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                TrustStrip()

                // 1 — decision headline
                Card {
                    VStack(alignment: .leading, spacing: 8) {
                        Eyebrow(text: "This cycle")
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            Text("\(counts.buy + counts.sell + counts.watch)")
                                .font(.system(size: 38, weight: .bold)).monospacedDigit()
                            Text("actionable ideas").font(.system(size: 16, weight: .semibold))
                                .foregroundStyle(Tok.text2)
                        }
                        HStack(spacing: 24) {
                            countStat("Buy", counts.buy, Tok.buy, Tok.buySoft)
                            countStat("Sell", counts.sell, Tok.sell, Tok.sellSoft)
                            countStat("Watch", counts.watch, Tok.watch, Tok.watchSoft)
                        }
                        .padding(.top, 6)
                        if counts.modelReq > 0 {
                            Text("\(counts.modelReq) need a model")
                                .font(.system(size: 12)).foregroundStyle(Tok.text3).padding(.top, 4)
                        }
                    }
                }

                // 4 — engine pulse
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Engine pulse", subtitle: "Summed stage latency — the one honest health signal today")
                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            Text(String(format: "%.2f", pulse.total)).font(.system(size: 24, weight: .bold)).monospacedDigit()
                            Text("s total cycle").font(.system(size: 13)).foregroundStyle(Tok.text2)
                        }
                        ForEach(pulse.stages, id: \.0) { stage, sec in
                            HStack(spacing: 10) {
                                Text(stage).font(.system(size: 11, design: .monospaced)).foregroundStyle(Tok.text3)
                                    .frame(width: 128, alignment: .leading)
                                Bar(value: sec, max: pulse.total, color: Tok.accent2, height: 6)
                            }
                        }
                    }
                }

                // 3 — top trade ideas
                Card {
                    VStack(alignment: .leading, spacing: 4) {
                        PanelHeader(title: "Top trade ideas", subtitle: "Tap for the full decision chain")
                        ForEach(top) { idea in
                            NavigationLink(value: idea) { IdeaRow(idea: idea) }.buttonStyle(.plain)
                            if idea.id != top.last?.id { Divider().overlay(Tok.border) }
                        }
                    }
                }

                // 5 + 7 — coming
                ComingPanel(title: "Portfolio value", subtitle: "NAV + equity curve",
                            unlock: "Unlocks when the engine persists positions + a NAV series. Today live_orders_sent=0 and src/portfolio has zero production callers.", wave: 5)
                ComingPanel(title: "Market regime", subtitle: "LSTM 4-class detector",
                            unlock: "Unlocks when the regime detector is wired into the live cycle. RegimeDetector has zero runtime callers today.", wave: 6)
                ComingPanel(title: "Risk & performance", subtitle: "Sharpe · Max DD · Vol · Win",
                            unlock: "Unlocks with the persisted-portfolio gate; backtest metrics are never persisted (retrain gate broken).", wave: 5)
            }
            .padding(16)
        }
        .apertureBackground()
        .navigationTitle("Overview")
        .navigationBarTitleDisplayMode(.large)
    }

    private func countStat(_ label: String, _ value: Int, _ color: Color, _ soft: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(color)
                .padding(.horizontal, 9).padding(.vertical, 3).background(soft, in: Capsule())
            Text("\(value)").font(.system(size: 28, weight: .bold)).monospacedDigit()
                .foregroundStyle(value > 0 ? color : Tok.text3)
        }
    }
}

/// One idea row (shared by Home + Ideas).
struct IdeaRow: View {
    let idea: TradeIdea
    var body: some View {
        HStack(spacing: 12) {
            AssetGlyph(symbol: idea.symbol, type: idea.type, size: 34)
            VStack(alignment: .leading, spacing: 2) {
                Text(idea.symbol).font(.system(size: 14, weight: .semibold, design: .monospaced))
                Text(idea.strategy ?? "").font(.system(size: 11.5)).foregroundStyle(Tok.text3)
            }
            ActionPillView(action: idea.action)
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Eyebrow(text: "Target wt")
                Text(Fmt.pctSigned(idea.targetWeight, 1)).font(.system(size: 14, weight: .semibold)).monospacedDigit()
            }
            if let cal = idea.calibratedProbability {
                ProbRing(value: cal, size: 44, label: "Cal p")
            } else {
                ComingRing(size: 44, label: "Cal p")
            }
            Image(systemName: "chevron.right")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Tok.text3)
                .accessibilityHidden(true)
        }
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(idea.symbol), \(idea.action.label)")
        .accessibilityHint("Opens the full decision chain")
        .accessibilityAddTraits(.isButton)
    }
}

/// A titled card whose body is a ComingCard.
struct ComingPanel: View {
    let title: String
    var subtitle: String? = nil
    let unlock: String
    var wave: Int? = nil
    var variant: LockKind = .gated
    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                PanelHeader(title: title, subtitle: subtitle)
                ComingCard(title: "\(title) — coming as the engine lands it", unlock: unlock, wave: wave, variant: variant)
            }
        }
    }
}
