import SwiftUI
import UIKit

/// More — everything heavier than the four cockpit tabs, as a grouped list.
/// Preserves web↔iOS parity without a 13-item tab bar.
struct MoreView: View {
    /// Live destinations = the 4 cockpit tabs + the model-gated Engine row.
    /// The coming count is derived so it can't drift from Readiness.coming.
    private static let liveDestinations = 5
    private static var comingDestinations: Int { Readiness.coming.flatMap { $0.items }.count }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                // Engine (LIVE, model-gated)
                section("Engine") {
                    NavigationLink(value: Readiness.model) {
                        MoreRow(label: Readiness.model.label, symbol: Readiness.model.sfSymbol,
                                purpose: Readiness.model.purpose, live: true, modelGated: true)
                    }.buttonStyle(.plain)
                    .simultaneousGesture(TapGesture().onEnded { rowHaptic() })
                }

                // Coming groups
                ForEach(Readiness.coming, id: \.group) { group in
                    section(group.group) {
                        ForEach(group.items) { spec in
                            NavigationLink(value: spec) {
                                MoreRow(label: spec.label, symbol: spec.sfSymbol, purpose: spec.purpose,
                                        live: false, lock: spec.lock)
                            }.buttonStyle(.plain)
                            .simultaneousGesture(TapGesture().onEnded { rowHaptic() })
                            if spec.id != group.items.last?.id { Divider().overlay(Tok.border).padding(.leading, 44) }
                        }
                    }
                }

                // Settings
                section("Settings") {
                    settingRow("Mode", "Paper · read-only")
                    Divider().overlay(Tok.border).padding(.leading, 44)
                    settingRow("Theme", "Dark")
                    Divider().overlay(Tok.border).padding(.leading, 44)
                    settingRow("Density", "Comfortable")
                }

                Text("Aperture · wang quant engine · data-honest v1\n\(Self.liveDestinations) live destinations · \(Self.comingDestinations) coming")
                    .font(.system(size: 11)).foregroundStyle(Tok.text3).multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity).padding(.top, 4)
            }
            .padding(16)
        }
        .apertureBackground()
        .navigationTitle("More")
    }

    @ViewBuilder private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Eyebrow(text: title)
            Card(padding: 0) { VStack(spacing: 0) { content() } }
        }
    }

    /// Light tap feedback when a destination row is selected.
    private func rowHaptic() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    private func settingRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.system(size: 14)).foregroundStyle(Tok.text1)
            Spacer()
            Text(value).font(.system(size: 13)).foregroundStyle(Tok.text3)
        }.padding(.horizontal, 14).padding(.vertical, 12)
    }
}

struct MoreRow: View {
    let label: String
    let symbol: String
    let purpose: String
    var live: Bool
    var modelGated: Bool = false
    var lock: LockKind? = nil
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol).font(.system(size: 16)).foregroundStyle(live ? Tok.accent2 : Tok.text3)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(.system(size: 14, weight: .medium)).foregroundStyle(live ? Tok.text1 : Tok.text2)
                Text(purpose).font(.system(size: 11.5)).foregroundStyle(Tok.text3).lineLimit(1)
            }
            Spacer()
            if live && !modelGated { Circle().fill(Tok.pos).frame(width: 7, height: 7) }
            else if modelGated { Circle().fill(Tok.info).frame(width: 7, height: 7) }
            else if let lock { Image(systemName: lock.symbol).font(.system(size: 12)).foregroundStyle(Tok.text3) }
            Image(systemName: "chevron.right").font(.system(size: 12)).foregroundStyle(Tok.text3)
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .contentShape(Rectangle())
    }
}
