import SwiftUI

/// A dignified "Coming" destination — not a dead link, not a 404. Title + purpose
/// + a muted wireframe ghost + the exact unlock condition from data_readiness.md.
struct ComingView: View {
    let spec: ScreenSpec

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 12) {
                    Image(systemName: spec.sfSymbol).font(.system(size: 20))
                        .foregroundStyle(Tok.accent).frame(width: 40, height: 40)
                        .background(Tok.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(spec.label).font(.system(size: 20, weight: .bold))
                        Text(spec.purpose).font(.system(size: 13)).foregroundStyle(Tok.text3)
                    }
                    Spacer()
                }

                if let lock = spec.lock {
                    HStack(spacing: 7) {
                        Image(systemName: lock.symbol).font(.system(size: 12))
                        Text(lock.label).font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(Tok.text3)
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(Tok.surface2, in: Capsule())
                    .overlay(Capsule().stroke(Tok.border, lineWidth: 1))
                }

                // wireframe ghost
                Card {
                    VStack(spacing: 12) {
                        HStack(spacing: 12) { ForEach(0..<4, id: \.self) { _ in ghostBlock(height: 56) } }
                        ghostBlock(height: 150)
                        HStack(spacing: 12) { ghostBlock(height: 100); ghostBlock(height: 100) }
                    }
                    .opacity(0.25)
                    .overlay(
                        ComingCard(title: "\(spec.label) — coming as the engine lands it",
                                   unlock: spec.unlock, wave: spec.wave, variant: spec.lock ?? .gated)
                            .padding(.horizontal, 4)
                    )
                }

                if spec.lock == .wireable {
                    Card(padding: 14) {
                        HStack(spacing: 10) {
                            Image(systemName: "wrench.and.screwdriver.fill").foregroundStyle(Tok.warn)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Wire this next").font(.system(size: 13.5, weight: .semibold))
                                Text("Needs no new persistence — only a BFF stub rewire. A fast-follow after v1.")
                                    .font(.system(size: 12)).foregroundStyle(Tok.text3)
                            }
                        }
                    }
                }
            }
            .padding(16)
        }
        .apertureBackground()
        .navigationTitle(spec.label)
        .navigationBarTitleDisplayMode(.inline)
    }

    private func ghostBlock(height: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 12).fill(Tok.surface2).frame(maxWidth: .infinity).frame(height: height)
    }
}
