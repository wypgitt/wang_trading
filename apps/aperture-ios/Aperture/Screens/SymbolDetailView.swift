import SwiftUI

/// Symbol detail — bridges "a stock/coin" ↔ "what the model thinks". Candles +
/// bar microstructure are LIVE (bars hypertable); SHAP + computed features are
/// coming (ephemeral / never persisted).
struct SymbolDetailView: View {
    let sym: Sym
    @State private var chartMode: ChartMode = .candles
    enum ChartMode: Hashable { case candles, area }

    private var idea: TradeIdea? { MockData.idea(for: sym.symbol) }
    private let trust = MockData.trust

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // hero
                Card {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack(spacing: 12) {
                            AssetGlyph(symbol: sym.symbol, type: sym.type, size: 40)
                            VStack(alignment: .leading, spacing: 3) {
                                HStack(spacing: 8) {
                                    Text(sym.symbol).font(.system(size: 18, weight: .bold, design: .monospaced))
                                    Chip { Text(sym.type.rawValue.capitalized) }
                                    Chip(tint: Tok.text3) { Text("bar \(sym.bar.barType)") }
                                }
                                Text(sym.name).font(.system(size: 13)).foregroundStyle(Tok.text3)
                            }
                            Spacer()
                            freshnessChip
                        }
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            Text(Fmt.price(sym.price)).font(.system(size: 30, weight: .bold)).monospacedDigit()
                            Delta(value: sym.change1d, dp: 2, size: 15)
                        }
                        SegPicker(options: [(.candles, "Candles"), (.area, "Area")], selection: $chartMode)
                        Group {
                            if chartMode == .candles {
                                CandleChartView(candles: Array(sym.candles.suffix(70)))
                            } else {
                                AreaChartView(data: sym.candles.suffix(70).map(\.c),
                                              color: sym.change1m >= 0 ? Tok.pos : Tok.neg)
                            }
                        }
                        .frame(height: 240)
                        Text("x-axis is bar-indexed — \(sym.bar.barType) bars are event-driven and unevenly spaced in time.")
                            .font(.system(size: 11)).foregroundStyle(Tok.text3)
                        // change strip
                        HStack {
                            changeCell("1W", sym.change1w)
                            changeCell("1M", sym.change1m)
                            changeCell("YTD", sym.changeYtd)
                            statCell("Mkt cap", sym.marketCap > 0 ? Fmt.compact(sym.marketCap) : "—")
                            statCell("Volume", Fmt.compact(sym.volume))
                        }
                    }
                }

                // what the engine sees + why
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "What the engine sees", subtitle: "The model's current read on this instrument")
                        if let idea {
                            HStack(spacing: 10) {
                                ActionPillView(action: idea.action)
                                Text("\(idea.strategy ?? "") · \(Fmt.sideLabel(idea.topSignalSide))")
                                    .font(.system(size: 12)).foregroundStyle(Tok.text2)
                            }
                            Text(idea.reason).font(.system(size: 13)).foregroundStyle(Tok.text2).lineSpacing(3)
                            HStack(spacing: 22) {
                                if let m = idea.metaProbability { ProbRing(value: m, size: 50, label: "Meta") } else { ComingRing(size: 50, label: "Meta") }
                                if let c = idea.calibratedProbability { ProbRing(value: c, size: 50, label: "Cal") } else { ComingRing(size: 50, label: "Cal") }
                                VStack(alignment: .leading, spacing: 8) {
                                    labeledComing("Regime fit")
                                    labeledComing("Pre-trade cost")
                                }
                                Spacer()
                            }
                        } else {
                            Text("No active idea for \(sym.symbol) this cycle.")
                                .font(.system(size: 13)).foregroundStyle(Tok.text3).padding(.vertical, 8)
                        }
                    }
                }

                ComingPanel(title: "Why", subtitle: "Top model feature contributions",
                            unlock: "Feature contributions — coming when shap_importance is persisted. TreeSHAP is implemented but never called by the live cycle.", wave: 4)

                // bar microstructure (LIVE)
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Bar microstructure", subtitle: "Persisted bar columns from the bars hypertable — live")
                        let b = sym.bar
                        let cells: [(String, String)] = [
                            ("VWAP", Fmt.price(b.vwap)), ("Dollar volume", Fmt.compact(b.dollarVolume)),
                            ("Tick count", "\(b.tickCount)"), ("Buy / sell vol", "\(Fmt.compact(b.buyVolume, prefix: "")) / \(Fmt.compact(b.sellVolume, prefix: ""))"),
                            ("Volume imbalance", Fmt.compact(b.volumeImbalance, prefix: "")), ("Tick imbalance", Fmt.signed(b.tickImbalanceRatio, 3)),
                            ("Imbalance", Fmt.num(b.imbalance, 0)), ("Threshold", Fmt.num(b.threshold, 0)),
                            ("Bar duration", "\(b.barDurationSeconds / 60)m \(b.barDurationSeconds % 60)s"), ("Bar type", b.barType),
                        ]
                        LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)], spacing: 14) {
                            ForEach(cells, id: \.0) { c in
                                VStack(alignment: .leading, spacing: 2) {
                                    Eyebrow(text: c.0)
                                    Text(c.1).font(.system(size: 15, weight: .semibold)).monospacedDigit()
                                }
                            }
                        }
                    }
                }

                ComingPanel(title: "Live feature grid", subtitle: "GARCH vol · RSI · order-flow · VPIN · Kyle λ",
                            unlock: "Live feature values — coming when the Feature Factory persists features (FeatureStore.save_features has zero callers today).", wave: 4)
            }
            .padding(16)
        }
        .refreshable {
            // Re-pull this symbol's read from the engine. Mock data is static, so
            // this just yields a refresh gesture; the live BFF will repopulate here.
            try? await Task.sleep(nanoseconds: 600_000_000)
        }
        .apertureBackground()
        .navigationTitle(sym.symbol)
        .navigationBarTitleDisplayMode(.inline)
    }

    /// Compact freshness signal — mirrors the TrustStrip's third pill so the trust
    /// state is present on this pushed screen. Bound to the same staleness field.
    private var freshnessChip: some View {
        HStack(spacing: 6) {
            StatusDot(ok: !trust.stale, live: !trust.stale)
            Text(trust.stale ? "STALE" : Fmt.timeAgo(trust.stalenessSeconds))
                .font(.system(size: 11, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(trust.stale ? Tok.warn : Tok.text2)
        }
        .padding(.horizontal, 9).padding(.vertical, 4)
        .background(Tok.surface2, in: Capsule())
        .overlay(Capsule().stroke((trust.stale ? Tok.warn : Tok.pos).opacity(0.3), lineWidth: 1))
    }

    private func changeCell(_ label: String, _ v: Double) -> some View {
        VStack(alignment: .leading, spacing: 3) { Eyebrow(text: label); Delta(value: v, dp: 1, arrow: false, size: 13) }
            .frame(maxWidth: .infinity, alignment: .leading)
    }
    private func statCell(_ label: String, _ v: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Eyebrow(text: label)
            Text(v).font(.system(size: 13, weight: .semibold)).monospacedDigit().foregroundStyle(Tok.text1)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private func labeledComing(_ label: String) -> some View {
        HStack(spacing: 8) { Eyebrow(text: label); DataUnavailableChip() }
    }
}
