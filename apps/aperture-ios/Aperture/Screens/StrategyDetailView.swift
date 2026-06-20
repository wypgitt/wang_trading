import SwiftUI

struct StrategyDetailView: View {
    let strategy: Strategy
    private var ideas: [TradeIdea] { MockData.activeIdeas(for: strategy.id) }
    private var family: FamilyReadiness { Families.of(strategy.id) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // header
                HStack(spacing: 10) {
                    Circle().fill(Tok.category[strategy.category] ?? Tok.accent).frame(width: 10, height: 10)
                    Text(strategy.name).font(.system(size: 20, weight: .bold))
                    if family.active {
                        StatusDot(ok: true, live: true)
                        Text("Active").font(.system(size: 12)).foregroundStyle(Tok.text3)
                    } else {
                        StatusDot(ok: false)
                        Text("Inactive — \(family.reason ?? "no live feed wired")")
                            .font(.system(size: 12)).foregroundStyle(Tok.text3)
                    }
                    Spacer()
                }
                HStack(spacing: 8) {
                    Chip(tint: Tok.category[strategy.category] ?? Tok.accent) { Text(strategy.category) }
                    Text("source · \(strategy.source)").font(.system(size: 12)).foregroundStyle(Tok.text3)
                }
                FlowChips(items: strategy.assetClasses.map { $0.rawValue.capitalized })

                Card { Text(strategy.thesis).font(.system(size: 14)).foregroundStyle(Tok.text2).lineSpacing(3) }

                ComingPanel(title: "Performance", subtitle: "Sharpe · Win · P&L share · YTD · Allocation · Avg hold",
                            unlock: "Per-strategy performance — coming when backtest metrics are persisted (retrain gate broken, retrain_pipeline.py:265).", wave: 5)

                ComingPanel(title: "Strategy equity", subtitle: "Cumulative sleeve P&L",
                            unlock: "Strategy equity — coming when per-strategy backtest runs are persisted.", wave: 5)

                // regime fit — unfilled rails so the layout reads intentional (no numbers fabricated)
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Regime fit", subtitle: "Expected edge by market regime")
                        VStack(spacing: 10) {
                            ForEach(["Trending ↑", "Trending ↓", "Mean-revert", "High vol"], id: \.self) { label in
                                HStack(spacing: 10) {
                                    Text(label).font(.system(size: 12.5)).foregroundStyle(Tok.text3)
                                        .frame(width: 92, alignment: .leading)
                                    Bar(value: 0)
                                }
                            }
                        }
                        ComingCard(title: "Regime fit — coming as the engine lands it",
                                   unlock: "Regime fit — coming when the regime detector is wired into the live cycle. RegimeDetector has zero runtime callers.",
                                   wave: 6, variant: .wireable, compact: true)
                    }
                }

                // parameters (LIVE)
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        PanelHeader(title: "Parameters", subtitle: "Live configuration")
                        ForEach(Array(strategy.params.enumerated()), id: \.offset) { _, kv in
                            HStack {
                                Text(kv.0).font(.system(size: 13, design: .monospaced)).foregroundStyle(Tok.text3)
                                Spacer()
                                Text(kv.1).font(.system(size: 13, weight: .medium)).monospacedDigit().foregroundStyle(Tok.text1)
                            }
                            if kv.0 != strategy.params.last?.0 { Divider().overlay(Tok.border) }
                        }
                    }
                }

                // active ideas (LIVE)
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        PanelHeader(title: "Active ideas from this family", subtitle: "Tap a row for the full decision chain")
                        if ideas.isEmpty {
                            Text(family.active
                                 ? "No live ideas above the entry gate this cycle."
                                 : "No live ideas — context feed not wired.")
                                .font(.system(size: 13)).foregroundStyle(Tok.text3).padding(.vertical, 6)
                        } else {
                            ForEach(ideas) { idea in
                                NavigationLink(value: idea) {
                                    HStack(spacing: 10) {
                                        AssetGlyph(symbol: idea.symbol, type: idea.type, size: 30)
                                        Text(idea.symbol).font(.system(size: 13.5, weight: .semibold, design: .monospaced))
                                        ActionPillView(action: idea.action)
                                        Spacer()
                                        Text(Fmt.pctSigned(idea.targetWeight, 1)).font(.system(size: 13, weight: .semibold)).monospacedDigit()
                                        Text(Fmt.prob(idea.calibratedProbability)).font(.system(size: 13)).monospacedDigit().foregroundStyle(Tok.text2)
                                        Image(systemName: "chevron.right").font(.system(size: 11)).foregroundStyle(Tok.text3)
                                    }
                                    .padding(.vertical, 6).contentShape(Rectangle())
                                }.buttonStyle(.plain)
                                if idea.id != ideas.last?.id { Divider().overlay(Tok.border) }
                            }
                        }
                    }
                }
            }
            .padding(16)
        }
        .refreshable {
            try? await Task.sleep(nanoseconds: 600_000_000)
        }
        .apertureBackground()
        .navigationTitle("Strategy")
        .navigationBarTitleDisplayMode(.inline)
    }
}
