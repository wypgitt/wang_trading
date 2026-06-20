import SwiftUI

/// Idea detail — the decision drawer-on-web becomes a native push here. Same
/// payload, platform-native container. Honest: regime-fit, cascade, SHAP, cost,
/// track-record hold their slots as "coming", never faked.
struct IdeaDetailView: View {
    let idea: TradeIdea

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // header
                HStack(spacing: 12) {
                    AssetGlyph(symbol: idea.symbol, type: idea.type, size: 40)
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 8) {
                            Text(idea.symbol).font(.system(size: 18, weight: .bold, design: .monospaced))
                            ActionPillView(action: idea.action)
                        }
                        Text("\(idea.strategy ?? "") · \(idea.barType) bars")
                            .font(.system(size: 12)).foregroundStyle(Tok.text3)
                    }
                    Spacer()
                }

                // 1 — reason
                Text(idea.reason).font(.system(size: 14)).foregroundStyle(Tok.text2).lineSpacing(3)

                // 2 — decision chain
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Eyebrow(text: "Decision chain")
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                chainStep("Signals", "\(idea.signalCount)", Tok.text1)
                                chevron
                                chainStep("Meta p", Fmt.prob(idea.metaProbability), idea.metaProbability == nil ? Tok.text3 : Tok.text1, gated: idea.metaProbability == nil)
                                chevron
                                chainStep("Calibrated", Fmt.prob(idea.calibratedProbability), idea.calibratedProbability == nil ? Tok.text3 : Tok.text1, gated: idea.calibratedProbability == nil)
                                chevron
                                chainStepComing("Regime fit")
                                chevron
                                chainStep("Bet size", idea.betSize.map { Fmt.pct($0, 1) } ?? "—", Tok.text1)
                                chevron
                                chainStep("Target", Fmt.pctSigned(idea.targetWeight, 1), idea.targetWeight >= 0 ? Tok.pos : Tok.neg)
                            }
                        }
                    }
                }

                // 3 — conviction
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Model conviction")
                        HStack(spacing: 22) {
                            ringOrGated(idea.metaProbability, "Meta")
                            ringOrGated(idea.calibratedProbability, "Calibrated")
                            VStack(alignment: .leading, spacing: 6) {
                                Eyebrow(text: "Track record")
                                DataUnavailableChip(modelGated: false)
                                Text("Snapshot is overwritten each cycle — no call history exists yet.")
                                    .font(.system(size: 11)).foregroundStyle(Tok.text3)
                            }
                            Spacer()
                        }
                        if idea.metaProbability != nil {
                            Text("raw == calibrated — the pipeline exposes one calibrated scalar today.")
                                .font(.system(size: 11)).foregroundStyle(Tok.text3)
                        }
                    }
                }

                // 4 — signals
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Signals (\(idea.signals.count))")
                        ForEach(Array(idea.signals.enumerated()), id: \.offset) { _, sig in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text(sig.family).font(.system(size: 13, weight: .semibold, design: .monospaced))
                                    Text(Fmt.sideLabel(sig.side))
                                        .font(.system(size: 10.5, weight: .semibold))
                                        .foregroundStyle(sig.side > 0 ? Tok.pos : sig.side < 0 ? Tok.neg : Tok.text3)
                                        .padding(.horizontal, 7).padding(.vertical, 2)
                                        .background(Tok.surface2, in: Capsule())
                                    Spacer()
                                    Text("conf \(String(format: "%.2f", sig.confidence))")
                                        .font(.system(size: 12)).monospacedDigit().foregroundStyle(Tok.text2)
                                }
                                Bar(value: sig.confidence, color: Tok.accent2, height: 5)
                                FlowChips(items: sig.meta.map { "\($0.0) \($0.1)" })
                            }
                            if sig.family != idea.signals.last?.family { Divider().overlay(Tok.border) }
                        }
                    }
                }

                // 5 — sizing cascade (coming)
                ComingPanel(title: "Bet-sizing cascade", subtitle: "5-layer cascade · binding constraint",
                            unlock: "Coming when the engine surfaces constraints_applied at the idea boundary.", wave: 4)

                // 6 — SHAP (coming)
                ComingPanel(title: "Why the model leaned this way", subtitle: "Top feature contributions (SHAP)",
                            unlock: "Coming when shap_importance (TreeSHAP) is persisted — implemented but never called by the live cycle.", wave: 4)

                // 7 — cost & constraints
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        PanelHeader(title: "Pre-trade cost & constraints")
                        HStack {
                            Eyebrow(text: "Expected cost")
                            Spacer()
                            DataUnavailableChip()
                        }
                    }
                }

                // 8 — latency
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        PanelHeader(title: "Pipeline latency", subtitle: "Per-stage stage_latency_seconds")
                        let total = idea.stageLatency.reduce(0) { $0 + $1.1 }
                        ForEach(idea.stageLatency, id: \.0) { stage, sec in
                            HStack(spacing: 10) {
                                Text(stage).font(.system(size: 11, design: .monospaced)).foregroundStyle(Tok.text3)
                                    .frame(width: 132, alignment: .leading)
                                Bar(value: sec, max: total, color: Tok.accent, height: 5)
                                Text(String(format: "%.3fs", sec)).font(.system(size: 11)).monospacedDigit()
                                    .foregroundStyle(Tok.text2).frame(width: 52, alignment: .trailing)
                            }
                        }
                    }
                }

                // 9 — open symbol (read-first; v1 stages no orders)
                VStack(spacing: 10) {
                    NavigationLink(value: MockData.sym(idea.symbol)) {
                        HStack { Image(systemName: "chart.bar.xaxis"); Text("Open \(idea.symbol)") }
                            .font(.system(size: 14, weight: .semibold)).foregroundStyle(Tok.text1)
                            .frame(maxWidth: .infinity).padding(.vertical, 11)
                            .background(Tok.surface2, in: RoundedRectangle(cornerRadius: Radius.md))
                            .overlay(RoundedRectangle(cornerRadius: Radius.md).stroke(Tok.border, lineWidth: 1))
                    }.buttonStyle(.plain)
                    Text("Live actions gated — read-only in v1 (live_orders_sent=0).")
                        .font(.system(size: 11)).foregroundStyle(Tok.text3)
                }
            }
            .padding(16)
        }
        .apertureBackground()
        .navigationTitle(idea.symbol)
        .navigationBarTitleDisplayMode(.inline)
    }

    private var chevron: some View {
        Image(systemName: "chevron.right").font(.system(size: 10)).foregroundStyle(Tok.text3)
            .accessibilityLabel("then")
    }

    private func chainStep(_ label: String, _ value: String, _ color: Color, gated: Bool = false) -> some View {
        VStack(spacing: 3) {
            Text(value).font(.system(size: 14, weight: .bold)).monospacedDigit().foregroundStyle(color)
            Eyebrow(text: label)
        }
        .padding(.horizontal, 10).padding(.vertical, 8)
        .background(Tok.surfaceInset, in: RoundedRectangle(cornerRadius: 9))
    }
    private func chainStepComing(_ label: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: "lock.fill").font(.system(size: 11)).foregroundStyle(Tok.text3)
                .accessibilityLabel("\(label) — coming")
            Eyebrow(text: label)
        }
        .padding(.horizontal, 10).padding(.vertical, 8)
        .background(Tok.surfaceInset, in: RoundedRectangle(cornerRadius: 9))
        .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(Tok.borderStrong, style: StrokeStyle(lineWidth: 1, dash: [3, 3])))
    }
    private func ringOrGated(_ v: Double?, _ label: String) -> some View {
        Group { if let v { ProbRing(value: v, size: 56, label: label) } else { ComingRing(size: 56, label: label) } }
    }
}

/// A simple wrapping chip flow.
struct FlowChips: View {
    let items: [String]
    var body: some View {
        FlexWrap(spacing: 6, lineSpacing: 6) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, t in
                Text(t).font(.system(size: 11, design: .monospaced)).foregroundStyle(Tok.text2)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Tok.surface2, in: Capsule())
            }
        }
    }
}
