import SwiftUI

/// Strategies — identity, config, and live signal counts are real; per-family
/// performance is coming (backtest metrics never persisted).
struct StrategiesView: View {
    @State private var category = "All"
    private let cats = ["All", "Momentum", "Mean Reversion", "Trend", "Volatility", "Carry", "Arbitrage"]

    private var rows: [Strategy] {
        category == "All" ? MockData.strategies : MockData.strategies.filter { $0.category == category }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                TrustStrip()

                Text("\(Families.counts.total) families · \(Families.counts.active) active · \(Families.counts.inactive) inactive")
                    .font(.system(size: 12.5)).foregroundStyle(Tok.text3)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(cats, id: \.self) { c in
                            Button { category = c } label: {
                                Text(c).font(.system(size: 12.5, weight: .medium))
                                    .foregroundStyle(category == c ? Tok.text1 : Tok.text2)
                                    .padding(.horizontal, 11).padding(.vertical, 6)
                                    .background(category == c ? Tok.accentSoft : Tok.surface2, in: Capsule())
                                    .overlay(Capsule().stroke(category == c ? Tok.accent.opacity(0.4) : Tok.border, lineWidth: 1))
                            }.buttonStyle(.plain)
                        }
                    }
                }

                Card(padding: 12) {
                    HStack(spacing: 8) {
                        Image(systemName: "lock.fill").font(.system(size: 12)).foregroundStyle(Tok.text3)
                        Text("Strategy identity, config and live signal counts are real. Per-family Sharpe, win rate, P&L share and allocation unlock with backtest metrics (Wave 5).")
                            .font(.system(size: 11.5)).foregroundStyle(Tok.text3)
                    }
                }

                ForEach(rows) { s in
                    NavigationLink(value: s) { StrategyCard(strategy: s) }.buttonStyle(.plain)
                }
            }
            .padding(16)
        }
        .apertureBackground()
        .navigationTitle("Strategies")
        .refreshable { }
    }
}

struct StrategyCard: View {
    let strategy: Strategy
    private var activeCount: Int { MockData.activeIdeas(for: strategy.id).filter { $0.action == .BUY || $0.action == .SELL }.count }
    // Active / dormant is data-driven from the engine's generator dispatch
    // (Families.of), NOT from the prototype live/shadow status flag.
    private var family: FamilyReadiness { Families.of(strategy.id) }

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Circle().fill(Tok.category[strategy.category] ?? Tok.accent).frame(width: 8, height: 8)
                    Text(strategy.name).font(.system(size: 15, weight: .semibold)).foregroundStyle(Tok.text1)
                    Spacer()
                    statusBadge
                }
                Text("\(strategy.category) · \(strategy.source)").font(.system(size: 12)).foregroundStyle(Tok.text3)
                FlowChips(items: strategy.assetClasses.map { $0.rawValue.capitalized })
                Text(strategy.thesis).font(.system(size: 12.5)).foregroundStyle(Tok.text2).lineLimit(3).lineSpacing(2)
                Divider().overlay(Tok.border)
                HStack(spacing: 16) {
                    comingMetric("Sharpe"); comingMetric("Win"); comingMetric("P&L share"); comingMetric("Alloc")
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Eyebrow(text: "Active ideas")
                        Text("\(activeCount)").font(.system(size: 18, weight: .bold)).monospacedDigit()
                            .foregroundStyle(activeCount > 0 ? Tok.accent : Tok.text3)
                    }
                }
            }
        }
    }

    @ViewBuilder private var statusBadge: some View {
        if family.active {
            HStack(spacing: 6) {
                StatusDot(ok: true, live: true)
                Text("Active").font(.system(size: 11)).foregroundStyle(Tok.text2)
            }
        } else {
            VStack(alignment: .trailing, spacing: 2) {
                Chip(tint: Tok.text3, border: Tok.border) {
                    HStack(spacing: 6) {
                        Circle().fill(Tok.text3.opacity(0.7)).frame(width: 7, height: 7)
                        Text("Inactive — no live feed")
                    }
                }
                if let reason = family.reason {
                    Text(reason).font(.system(size: 10)).foregroundStyle(Tok.text3)
                        .multilineTextAlignment(.trailing).lineLimit(2)
                }
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Inactive — no live feed. \(family.reason ?? "no live feed wired")")
        }
    }

    private func comingMetric(_ label: String) -> some View {
        VStack(spacing: 4) { Eyebrow(text: label); DataUnavailableChip() }
    }
}
