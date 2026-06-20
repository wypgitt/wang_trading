// AUTO-GENERATED from tokens.json by scripts/gen-tokens.mjs — DO NOT EDIT.
import SwiftUI
import UIKit

extension Color {
  /// Hex initializer used by the generated token palette.
  init(hex: UInt) {
    self.init(
      .sRGB,
      red: Double((hex >> 16) & 0xFF) / 255,
      green: Double((hex >> 8) & 0xFF) / 255,
      blue: Double(hex & 0xFF) / 255,
      opacity: 1
    )
  }
}

/// An adaptive color: resolves to the dark or light value per the system
/// appearance (Settings → Display) — so the app is light by day, dark at night.
func apertureAdaptive(_ dark: Color, _ light: Color) -> Color {
  Color(uiColor: UIColor { trait in trait.userInterfaceStyle == .dark ? UIColor(dark) : UIColor(light) })
}

/// Aperture color tokens (generated from tokens.json). All adaptive.
enum Tok {
  static let bg0 = apertureAdaptive(Color(hex: 0x0A0C10), Color(hex: 0xF5F6F8))
  static let bg1 = apertureAdaptive(Color(hex: 0x0E1116), Color(hex: 0xEBEDF1))
  static let surface1 = apertureAdaptive(Color(hex: 0x14181F), Color(hex: 0xFFFFFF))
  static let surface2 = apertureAdaptive(Color(hex: 0x1A1F27), Color(hex: 0xF1F3F6))
  static let surface3 = apertureAdaptive(Color(hex: 0x222933), Color(hex: 0xE6EAEF))
  static let surfaceInset = apertureAdaptive(Color(hex: 0x0C0F14), Color(hex: 0xECEFF3))
  static let border = apertureAdaptive(Color(.sRGB, red: 1.0000, green: 1.0000, blue: 1.0000, opacity: 0.07), Color(.sRGB, red: 0.0627, green: 0.0863, blue: 0.1333, opacity: 0.10))
  static let borderStrong = apertureAdaptive(Color(.sRGB, red: 1.0000, green: 1.0000, blue: 1.0000, opacity: 0.13), Color(.sRGB, red: 0.0627, green: 0.0863, blue: 0.1333, opacity: 0.20))
  static let grid = apertureAdaptive(Color(.sRGB, red: 1.0000, green: 1.0000, blue: 1.0000, opacity: 0.055), Color(.sRGB, red: 0.0627, green: 0.0863, blue: 0.1333, opacity: 0.09))
  static let text1 = apertureAdaptive(Color(hex: 0xEEF1F6), Color(hex: 0x161A21))
  static let text2 = apertureAdaptive(Color(hex: 0xA3ADBB), Color(hex: 0x4F5763))
  static let text3 = apertureAdaptive(Color(hex: 0x6C7787), Color(hex: 0x6C7682))
  static let textInverse = apertureAdaptive(Color(hex: 0x0A0C10), Color(hex: 0xFFFFFF))
  static let pos = apertureAdaptive(Color(hex: 0x1ECB8B), Color(hex: 0x0A8F5E))
  static let posDim = apertureAdaptive(Color(hex: 0x15A673), Color(hex: 0x097C52))
  static let posSoft = apertureAdaptive(Color(.sRGB, red: 0.1176, green: 0.7961, blue: 0.5451, opacity: 0.13), Color(.sRGB, red: 0.0392, green: 0.5608, blue: 0.3686, opacity: 0.12))
  static let neg = apertureAdaptive(Color(hex: 0xF6465D), Color(hex: 0xCC2436))
  static let negDim = apertureAdaptive(Color(hex: 0xD23A4E), Color(hex: 0xB11F30))
  static let negSoft = apertureAdaptive(Color(.sRGB, red: 0.9647, green: 0.2745, blue: 0.3647, opacity: 0.13), Color(.sRGB, red: 0.8000, green: 0.1412, blue: 0.2118, opacity: 0.10))
  static let accent = apertureAdaptive(Color(hex: 0x7C5CFF), Color(hex: 0x6A45E8))
  static let accent2 = apertureAdaptive(Color(hex: 0x4D9FFF), Color(hex: 0x1F74E0))
  static let accentSoft = apertureAdaptive(Color(.sRGB, red: 0.4863, green: 0.3608, blue: 1.0000, opacity: 0.15), Color(.sRGB, red: 0.4157, green: 0.2706, blue: 0.9098, opacity: 0.12))
  static let warn = apertureAdaptive(Color(hex: 0xF0A93B), Color(hex: 0xB9760F))
  static let warnSoft = apertureAdaptive(Color(.sRGB, red: 0.9412, green: 0.6627, blue: 0.2314, opacity: 0.14), Color(.sRGB, red: 0.7255, green: 0.4627, blue: 0.0588, opacity: 0.13))
  static let info = apertureAdaptive(Color(hex: 0x4D9FFF), Color(hex: 0x1F74E0))
  static let infoSoft = apertureAdaptive(Color(.sRGB, red: 0.3020, green: 0.6235, blue: 1.0000, opacity: 0.14), Color(.sRGB, red: 0.1216, green: 0.4549, blue: 0.8784, opacity: 0.13))
  static let regimeUp = apertureAdaptive(Color(hex: 0x1ECB8B), Color(hex: 0x0A8F5E))
  static let regimeDown = apertureAdaptive(Color(hex: 0xF6465D), Color(hex: 0xCC2436))
  static let regimeMr = apertureAdaptive(Color(hex: 0xB07CFF), Color(hex: 0x7C4FD6))
  static let regimeHv = apertureAdaptive(Color(hex: 0xF0A93B), Color(hex: 0xB9760F))
  static let buy = apertureAdaptive(Color(hex: 0x1ECB8B), Color(hex: 0x0A8F5E))
  static let buySoft = apertureAdaptive(Color(.sRGB, red: 0.1176, green: 0.7961, blue: 0.5451, opacity: 0.14), Color(.sRGB, red: 0.0392, green: 0.5608, blue: 0.3686, opacity: 0.13))
  static let sell = apertureAdaptive(Color(hex: 0xF6465D), Color(hex: 0xCC2436))
  static let sellSoft = apertureAdaptive(Color(.sRGB, red: 0.9647, green: 0.2745, blue: 0.3647, opacity: 0.14), Color(.sRGB, red: 0.8000, green: 0.1412, blue: 0.2118, opacity: 0.11))
  static let watch = apertureAdaptive(Color(hex: 0x4D9FFF), Color(hex: 0x1F74E0))
  static let watchSoft = apertureAdaptive(Color(.sRGB, red: 0.3020, green: 0.6235, blue: 1.0000, opacity: 0.14), Color(.sRGB, red: 0.1216, green: 0.4549, blue: 0.8784, opacity: 0.13))
  static let neutral = apertureAdaptive(Color(hex: 0x8A93A3), Color(hex: 0x5B6470))
  static let neutralSoft = apertureAdaptive(Color(.sRGB, red: 0.5412, green: 0.5765, blue: 0.6392, opacity: 0.14), Color(.sRGB, red: 0.3569, green: 0.3922, blue: 0.4392, opacity: 0.13))

  static let accentGrad = LinearGradient(colors: [Color(hex: 0x7C5CFF), Color(hex: 0x4D9FFF)], startPoint: .topLeading, endPoint: .bottomTrailing)
  static let posGrad = LinearGradient(colors: [Color(hex: 0x1ECB8B), Color(hex: 0x4D9FFF)], startPoint: .topLeading, endPoint: .bottomTrailing)

  static let category: [String: Color] = [
    "Momentum": Color(hex: 0x4D9FFF),
    "Mean Reversion": Color(hex: 0xB07CFF),
    "Trend": Color(hex: 0x1ECB8B),
    "Volatility": Color(hex: 0xF0A93B),
    "Carry": Color(hex: 0x22D3EE),
    "Arbitrage": Color(hex: 0xF6679A)
  ]
  static let assetTint: [String: Color] = [
    "equity": Color(hex: 0x4D9FFF),
    "index": Color(hex: 0xB07CFF),
    "crypto": Color(hex: 0xF0A93B),
    "future": Color(hex: 0x22D3EE)
  ]
  static let regimeHex: [String: Color] = [
    "trending_up": apertureAdaptive(Color(hex: 0x1ECB8B), Color(hex: 0x0A8F5E)),
    "trending_down": apertureAdaptive(Color(hex: 0xF6465D), Color(hex: 0xCC2436)),
    "mean_reverting": apertureAdaptive(Color(hex: 0xB07CFF), Color(hex: 0x7C4FD6)),
    "high_volatility": apertureAdaptive(Color(hex: 0xF0A93B), Color(hex: 0xB9760F))
  ]
  static let staleThresholdSeconds: Double = 90
}

/// Corner radii (generated).
enum Radius {
  static let xs: CGFloat = 6
  static let sm: CGFloat = 9
  static let md: CGFloat = 13
  static let lg: CGFloat = 18
  static let xl: CGFloat = 24
  static let pill: CGFloat = 999
}

/// Spacing scale on a 4-pt grid (generated).
enum Space {
  static let s1: CGFloat = 4
  static let s2: CGFloat = 8
  static let s3: CGFloat = 12
  static let s4: CGFloat = 16
  static let s5: CGFloat = 20
  static let s6: CGFloat = 24
  static let s8: CGFloat = 32
  static let s10: CGFloat = 40
  static let s12: CGFloat = 48
}
