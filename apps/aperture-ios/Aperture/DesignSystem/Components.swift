// Shared SwiftUI components — the iOS twin of the web component catalog
// (src/components/ui/primitives.tsx). Same design language, native container.
import SwiftUI

// MARK: - Card
struct Card<Content: View>: View {
    var padding: CGFloat = Space.s4
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Tok.surface1)
            .overlay(RoundedRectangle(cornerRadius: Radius.lg).stroke(Tok.border, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: Radius.lg))
    }
}

// MARK: - Section + eyebrow text
struct Eyebrow: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 10.5, weight: .semibold))
            .tracking(1.1)
            .foregroundStyle(Tok.text3)
    }
}

struct PanelHeader: View {
    let title: String
    var subtitle: String? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.system(size: 15.5, weight: .semibold)).foregroundStyle(Tok.text1)
            if let subtitle { Text(subtitle).font(.system(size: 12)).foregroundStyle(Tok.text3) }
        }
    }
}

// MARK: - Action pill
struct ActionPillView: View {
    let action: Action
    var body: some View {
        let c = action.pillColors
        Text(action.label)
            .font(.system(size: 11.5, weight: .semibold))
            .foregroundStyle(c.fg)
            .padding(.horizontal, 9).padding(.vertical, 3)
            .background(c.bg, in: Capsule())
    }
}

// MARK: - Chip
struct Chip<Label: View>: View {
    var tint: Color = Tok.text2
    var border: Color = Tok.border
    @ViewBuilder var label: Label
    var body: some View {
        label
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(tint)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(Tok.surface2, in: Capsule())
            .overlay(Capsule().stroke(border, lineWidth: 1))
    }
}

// MARK: - Asset glyph
struct AssetGlyph: View {
    let symbol: String
    let type: AssetType
    var size: CGFloat = 34
    var body: some View {
        let tint = Tok.assetTint[type.rawValue] ?? Tok.accent
        let txt = String(symbol.prefix(type == .crypto ? 3 : 4))
        Text(txt)
            .font(.system(size: size * (txt.count > 3 ? 0.28 : 0.34), weight: .bold, design: .rounded))
            .foregroundStyle(tint)
            .frame(width: size, height: size)
            .background(tint.opacity(0.13), in: RoundedRectangle(cornerRadius: size * 0.3))
            .overlay(RoundedRectangle(cornerRadius: size * 0.3).stroke(tint.opacity(0.2), lineWidth: 1))
    }
}

// MARK: - Signed delta
struct Delta: View {
    let value: Double
    var dp: Int = 2
    var arrow: Bool = true
    var size: CGFloat = 13
    var body: some View {
        let color = value > 0 ? Tok.pos : value < 0 ? Tok.neg : Tok.text3
        let prefix = (arrow && value != 0) ? (value > 0 ? "▲ " : "▼ ") : ""
        Text("\(prefix)\(Fmt.pctSigned(value, dp))")
            .font(.system(size: size, weight: .semibold))
            .monospacedDigit()
            .foregroundStyle(color)
    }
}

// MARK: - Stat block
struct StatBlock: View {
    let label: String
    let value: String
    var valueColor: Color = Tok.text1
    var sub: String? = nil
    var valueSize: CGFloat = 22
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Eyebrow(text: label)
            Text(value).font(.system(size: valueSize, weight: .bold)).monospacedDigit().foregroundStyle(valueColor)
            if let sub { Text(sub).font(.system(size: 12)).foregroundStyle(Tok.text2) }
        }
    }
}

// MARK: - Progress bar
struct Bar: View {
    let value: Double
    var max: Double = 1
    var color: Color = Tok.accent
    var height: CGFloat = 6
    var body: some View {
        GeometryReader { geo in
            let frac = Swift.max(0, Swift.min(1, value / Swift.max(max, .ulpOfOne)))
            ZStack(alignment: .leading) {
                Capsule().fill(Tok.surfaceInset)
                Capsule().fill(color).frame(width: geo.size.width * frac)
            }
        }
        .frame(height: height)
    }
}

// MARK: - Probability ring
struct ProbRing: View {
    let value: Double?
    var size: CGFloat = 52
    var label: String? = nil
    var body: some View {
        let v = value ?? 0
        let color = v >= 0.65 ? Tok.pos : v >= 0.55 ? Tok.warn : Tok.text3
        VStack(spacing: 4) {
            ZStack {
                Circle().stroke(Tok.surfaceInset, lineWidth: 5)
                Circle().trim(from: 0, to: v)
                    .stroke(color, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text(value == nil ? "—" : String(format: "%.2f", v))
                    .font(.system(size: size * 0.27, weight: .bold)).monospacedDigit()
                    .foregroundStyle(color)
            }
            .frame(width: size, height: size)
            if let label { Eyebrow(text: label) }
        }
    }
}

// MARK: - Status dot
struct StatusDot: View {
    var ok: Bool
    var live: Bool = false
    @State private var pulse = false
    var body: some View {
        Circle()
            .fill(ok ? Tok.pos : Tok.neg)
            .frame(width: 7, height: 7)
            .opacity(live && pulse ? 0.45 : 1)
            .animation(live ? .easeInOut(duration: 0.9).repeatForever(autoreverses: true) : .default, value: pulse)
            .onAppear { if live { pulse = true } }
    }
}

// MARK: - Segmented control (thin wrapper for the Aperture look)
struct SegPicker<T: Hashable>: View {
    let options: [(T, String)]
    @Binding var selection: T
    var body: some View {
        HStack(spacing: 2) {
            ForEach(options, id: \.0) { opt in
                let active = opt.0 == selection
                Button { selection = opt.0 } label: {
                    Text(opt.1)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(active ? Tok.text1 : Tok.text2)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(active ? Tok.surface3 : .clear, in: RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(Tok.surfaceInset, in: RoundedRectangle(cornerRadius: 11))
        .overlay(RoundedRectangle(cornerRadius: 11).stroke(Tok.border, lineWidth: 1))
    }
}

extension View {
    /// Tabular numeric styling shorthand.
    func numeric() -> some View { self.monospacedDigit() }
}
