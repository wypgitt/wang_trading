// Honesty primitives — iOS twin of src/components/ui/honesty.tsx. The dignified
// visual language for engine outputs that are real in code but NOT yet persisted,
// so the UI never fabricates a number. See docs/data_readiness.md.
import SwiftUI

/// Block "Not yet available" state with a concrete unlock condition.
struct ComingCard: View {
    var title: String = "Coming as the engine lands it"
    let unlock: String
    var wave: Int? = nil
    var variant: LockKind = .gated
    var compact: Bool = false
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: variant.symbol)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Tok.accent)
                .frame(width: 38, height: 38)
                .background(Tok.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 11))
            Text(title).font(.system(size: 14, weight: .semibold)).foregroundStyle(Tok.text1)
                .multilineTextAlignment(.center)
            Text(unlock).font(.system(size: 12.5)).foregroundStyle(Tok.text2)
                .multilineTextAlignment(.center).lineSpacing(2)
            if let wave { Eyebrow(text: "Wave \(wave)").foregroundStyle(Tok.accent) }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, compact ? 18 : 32).padding(.horizontal, 20)
        .background(Tok.surfaceInset, in: RoundedRectangle(cornerRadius: Radius.md))
        .overlay(
            RoundedRectangle(cornerRadius: Radius.md)
                .strokeBorder(Tok.borderStrong, style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
        )
    }
}

/// Inline marker for a single absent metric — a small "Coming" / "Model?" chip.
struct DataUnavailableChip: View {
    var label: String? = nil
    var modelGated: Bool = false
    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 5) {
                Image(systemName: modelGated ? "cpu" : "lock.fill").font(.system(size: 10))
                Text(modelGated ? "Model?" : "Coming").font(.system(size: 11.5, weight: .medium))
            }
            .foregroundStyle(Tok.text3)
            .padding(.horizontal, 9).padding(.vertical, 4)
            .background(Tok.surface2, in: Capsule())
            .overlay(Capsule().stroke(Tok.border, lineWidth: 1))
            if let label { Eyebrow(text: label) }
        }
    }
}

/// A dashed ring that slots beside real ProbRings when a value is absent.
struct ComingRing: View {
    var size: CGFloat = 52
    var label: String? = nil
    var modelGated: Bool = true
    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle().strokeBorder(Tok.border, style: StrokeStyle(lineWidth: 5, dash: [2, 6]))
                Image(systemName: modelGated ? "cpu" : "lock.fill")
                    .font(.system(size: size * 0.28)).foregroundStyle(Tok.text3)
            }
            .frame(width: size, height: size)
            if let label { Eyebrow(text: label) }
        }
    }
}
