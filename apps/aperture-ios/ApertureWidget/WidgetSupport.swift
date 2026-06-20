import SwiftUI
import UIKit

// Self-contained tokens for the widget extension (extensions can't share the
// app's synchronized source group). Values mirror ApertureTokens.generated.swift
// — both derive from apps/aperture-web/tokens.json. ADAPTIVE: the widget follows
// the system appearance like the app (dark at night, light by day).
enum WTok {
    static func hex(_ h: UInt) -> Color {
        Color(.sRGB, red: Double((h >> 16) & 0xFF) / 255, green: Double((h >> 8) & 0xFF) / 255, blue: Double(h & 0xFF) / 255, opacity: 1)
    }
    static func adaptive(_ dark: UInt, _ light: UInt) -> Color {
        Color(uiColor: UIColor { $0.userInterfaceStyle == .dark ? UIColor(hex(dark)) : UIColor(hex(light)) })
    }
    static let bg0 = adaptive(0x0A0C10, 0xF5F6F8)
    static let surface2 = adaptive(0x1A1F27, 0xF1F3F6)
    static let text1 = adaptive(0xEEF1F6, 0x161A21)
    static let text2 = adaptive(0xA3ADBB, 0x4F5763)
    static let text3 = adaptive(0x6C7787, 0x6C7682)
    static let buy = adaptive(0x1ECB8B, 0x0A8F5E)
    static let sell = adaptive(0xF6465D, 0xCC2436)
    static let watch = adaptive(0x4D9FFF, 0x1F74E0)
    static let warn = adaptive(0xF0A93B, 0xB9760F)
    static let accent = adaptive(0x7C5CFF, 0x6A45E8)
    static let grad = LinearGradient(colors: [hex(0x7C5CFF), hex(0x4D9FFF)], startPoint: .topLeading, endPoint: .bottomTrailing)
}

/// The honest snapshot the widget renders. In production this is read from the
/// App Group shared cache the publisher writes on its cadence; today it mirrors
/// the engine's decision counts. A widget NEVER shows NAV — there is no persisted
/// portfolio, so it surfaces the decision verdict + freshness, nothing faked.
struct ApertureSnapshot {
    let buy: Int
    let sell: Int
    let watch: Int
    let topSymbol: String
    let topAction: String
    let topActionColor: Color
    let topWeight: Double
    let freshnessSeconds: Int

    var total: Int { buy + sell + watch }
    var freshness: String { freshnessSeconds < 60 ? "\(freshnessSeconds)s ago" : "\(freshnessSeconds / 60)m ago" }

    static let sample = ApertureSnapshot(
        buy: 7, sell: 3, watch: 2,
        topSymbol: "NVDA", topAction: "Buy", topActionColor: WTok.buy, topWeight: 0.092,
        freshnessSeconds: 4)
}
