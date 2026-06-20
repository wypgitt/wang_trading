// Formatting helpers — iOS twin of src/lib/format.ts. Consistent number/percent
// rendering with tabular figures.
import Foundation
import SwiftUI

enum Fmt {
    static func pct(_ v: Double, _ dp: Int = 2) -> String { String(format: "%.\(dp)f%%", v * 100) }
    static func pctSigned(_ v: Double, _ dp: Int = 2) -> String {
        (v >= 0 ? "+" : "") + String(format: "%.\(dp)f%%", v * 100)
    }
    static func signed(_ v: Double, _ dp: Int = 2) -> String {
        (v >= 0 ? "+" : "") + String(format: "%.\(dp)f", v)
    }
    static func prob(_ v: Double?) -> String { v == nil ? "—" : String(format: "%.2f", v!) }
    static func bps(_ v: Double) -> String { (v >= 0 ? "+" : "") + String(format: "%.1f bps", v) }

    static func price(_ v: Double) -> String {
        v >= 1 ? num(v, 2) : num(v, 4)
    }
    static func num(_ v: Double, _ dp: Int = 2) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.minimumFractionDigits = dp
        f.maximumFractionDigits = dp
        return f.string(from: NSNumber(value: v)) ?? String(format: "%.\(dp)f", v)
    }
    static func compact(_ v: Double, prefix: String = "$") -> String {
        let abs = Swift.abs(v), sign = v < 0 ? "-" : ""
        if abs >= 1e12 { return "\(sign)\(prefix)\(String(format: "%.2f", abs / 1e12))T" }
        if abs >= 1e9 { return "\(sign)\(prefix)\(String(format: "%.2f", abs / 1e9))B" }
        if abs >= 1e6 { return "\(sign)\(prefix)\(String(format: "%.2f", abs / 1e6))M" }
        if abs >= 1e3 { return "\(sign)\(prefix)\(String(format: "%.1f", abs / 1e3))K" }
        return "\(sign)\(prefix)\(String(format: "%.0f", abs))"
    }
    static func timeAgo(_ seconds: Int) -> String {
        if seconds < 60 { return "\(seconds)s ago" }
        if seconds < 3600 { return "\(seconds / 60)m ago" }
        return "\(seconds / 3600)h ago"
    }
    static func sideLabel(_ side: Int) -> String { side > 0 ? "LONG" : side < 0 ? "SHORT" : "FLAT" }
}

extension Action {
    /// (fill color, text color) for the pill.
    var pillColors: (bg: Color, fg: Color) {
        switch self {
        case .BUY: return (Tok.buySoft, Tok.buy)
        case .SELL: return (Tok.sellSoft, Tok.sell)
        case .WATCH: return (Tok.watchSoft, Tok.watch)
        case .MODEL_REQUIRED, .NO_DATA: return (Tok.neutralSoft, Tok.neutral)
        }
    }
}
