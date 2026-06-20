#!/usr/bin/env node
// ---------------------------------------------------------------------------
// Aperture token generator — single source of truth → platform files.
//
//   tokens.json  →  src/app/tokens.generated.css        (web CSS vars)
//                →  src/lib/tokens.generated.ts          (chart hex objects)
//                →  ../aperture-ios/Aperture/DesignSystem/ApertureTokens.generated.swift
//
// Idempotent. Run in predev / prebuild. CI can run it and `git diff --exit-code`
// to guarantee the committed generated files never drift from tokens.json.
// ---------------------------------------------------------------------------
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, '..');
const tokens = JSON.parse(readFileSync(resolve(root, 'tokens.json'), 'utf8'));

const BANNER = 'AUTO-GENERATED from tokens.json by scripts/gen-tokens.mjs — DO NOT EDIT.';

// ---- helpers ---------------------------------------------------------------
const kebab = (s) => s.replace(/([a-z0-9])([A-Z])/g, '$1-$2').replace(/\s+/g, '-').toLowerCase();

// Flatten color.* into kebab CSS var names: surface.1 -> --surface-1, border._ -> --border
function cssVars() {
  const lines = [];
  const push = (name, val) => lines.push(`  --${name}: ${val};`);
  for (const [group, val] of Object.entries(tokens.color)) {
    for (const [k, v] of Object.entries(val)) {
      const base = group === 'category' || group === 'asset' ? `${group}-${kebab(k)}` : group;
      const name = k === '_' ? base : `${base}-${kebab(k)}`;
      push(name, v);
    }
  }
  push('accent-grad', tokens.gradient.accent);
  push('pos-grad', tokens.gradient.pos);
  for (const [k, v] of Object.entries(tokens.radius)) push(`r-${k}`, `${v}px`);
  for (const [k, v] of Object.entries(tokens.space)) push(`sp-${k}`, `${v}px`);
  for (const [k, v] of Object.entries(tokens.shadow)) push(`shadow-${k}`, v);
  push('font-sans', tokens.font.sans);
  push('font-mono', tokens.font.mono);
  push('sidebar-w', `${tokens.layout.sidebarW}px`);
  push('content-max', `${tokens.layout.contentMax}px`);
  return `/* ${BANNER} */\n:root {\n  color-scheme: dark;\n${lines.join('\n')}\n}\n`;
}

// Light-theme overrides under [data-theme="light"] (dark-first product; light is the alternate).
function cssLight() {
  if (!tokens.light) return '';
  const lines = [];
  const push = (name, val) => lines.push(`  --${name}: ${val};`);
  for (const [group, val] of Object.entries(tokens.light)) {
    for (const [k, v] of Object.entries(val)) {
      const name = k === '_' ? group : `${group}-${kebab(k)}`;
      push(name, v);
    }
  }
  for (const [k, v] of Object.entries(tokens.shadowLight ?? {})) push(`shadow-${k}`, v);
  return `\n/* ${BANNER} */\n[data-theme="light"] {\n  color-scheme: light;\n${lines.join('\n')}\n}\n`;
}

// ---- chart hex objects (CSS vars don't resolve inside SVG attributes) -------
const cc = tokens.color;
// Build the flat chart-hex object from a color source (CSS vars don't resolve in
// SVG attributes, so charts read these). buildC(dark) → C; buildC(light) → C_LIGHT.
function buildC(s) {
  return {
    bg0: s.bg['0'], bg1: s.bg['1'],
    surface1: s.surface['1'], surface2: s.surface['2'], surface3: s.surface['3'], surfaceInset: s.surface.inset,
    border: s.border._, borderStrong: s.border.strong, grid: s.border.grid,
    text1: s.text['1'], text2: s.text['2'], text3: s.text['3'], textInverse: s.text.inverse,
    pos: s.pos._, posDim: s.pos.dim, neg: s.neg._, negDim: s.neg.dim,
    warn: s.warn._, info: s.info._, accent: s.accent._, accent2: s.accent['2'],
    regimeUp: s.regime.up, regimeDown: s.regime.down, regimeMr: s.regime.mr, regimeHv: s.regime.hv,
    buy: s.action.buy, sell: s.action.sell, watch: s.action.watch, neutral: s.action.neutral,
  };
}
// merge light overrides over the dark base, per group
function mergedLight() {
  const out = {};
  for (const g of Object.keys(cc)) out[g] = { ...cc[g], ...(tokens.light?.[g] ?? {}) };
  return out;
}

function tsObjects() {
  const C = buildC(cc);
  const C_LIGHT = buildC(mergedLight());
  const j = (o) => JSON.stringify(o, null, 2).replace(/"([a-zA-Z0-9_]+)":/g, '$1:');
  return `// ${BANNER}
export const C = ${j(C)} as const;

export const C_LIGHT = ${j(C_LIGHT)} as const;

export const CAT: Record<string, string> = ${j(cc.category)};

export const REGIME_HEX: Record<string, string> = ${j({
    trending_up: cc.regime.up, trending_down: cc.regime.down,
    mean_reverting: cc.regime.mr, high_volatility: cc.regime.hv,
  })};

export const REGIME_LABEL: Record<string, string> = ${j({
    trending_up: 'Trending ↑', trending_down: 'Trending ↓',
    mean_reverting: 'Mean-revert', high_volatility: 'High vol',
  })};

export const ASSET_TINT: Record<string, string> = ${j(cc.asset)};

export const STALE_THRESHOLD_SECONDS = ${tokens.config.staleThresholdSeconds};
`;
}

// ---- iOS SwiftUI -----------------------------------------------------------
function hexToUInt(hex) {
  const m = /^#([0-9a-fA-F]{6})$/.exec(hex);
  return m ? `0x${m[1].toUpperCase()}` : null;
}
function rgbaToSwift(v) {
  const m = /rgba\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/.exec(v);
  if (!m) return null;
  const [r, g, b, a] = [m[1], m[2], m[3], m[4]];
  return `Color(.sRGB, red: ${(+r / 255).toFixed(4)}, green: ${(+g / 255).toFixed(4)}, blue: ${(+b / 255).toFixed(4)}, opacity: ${a})`;
}
function swiftColor(v) {
  const h = hexToUInt(v);
  if (h) return `Color(hex: ${h})`;
  return rgbaToSwift(v) ?? `Color.clear`;
}
function swift() {
  const ml = mergedLight(); // dark base + light overrides
  const lines = [];
  // Each token is ADAPTIVE: dark value in dark mode, light value in light mode,
  // resolved by the system appearance. So the whole app follows the OS automatically.
  const add = (name, darkV, lightV) =>
    lines.push(`  static let ${name} = apertureAdaptive(${swiftColor(darkV)}, ${swiftColor(lightV)})`);
  add('bg0', cc.bg['0'], ml.bg['0']); add('bg1', cc.bg['1'], ml.bg['1']);
  add('surface1', cc.surface['1'], ml.surface['1']); add('surface2', cc.surface['2'], ml.surface['2']);
  add('surface3', cc.surface['3'], ml.surface['3']); add('surfaceInset', cc.surface.inset, ml.surface.inset);
  add('border', cc.border._, ml.border._); add('borderStrong', cc.border.strong, ml.border.strong); add('grid', cc.border.grid, ml.border.grid);
  add('text1', cc.text['1'], ml.text['1']); add('text2', cc.text['2'], ml.text['2']); add('text3', cc.text['3'], ml.text['3']); add('textInverse', cc.text.inverse, ml.text.inverse);
  add('pos', cc.pos._, ml.pos._); add('posDim', cc.pos.dim, ml.pos.dim); add('posSoft', cc.pos.soft, ml.pos.soft);
  add('neg', cc.neg._, ml.neg._); add('negDim', cc.neg.dim, ml.neg.dim); add('negSoft', cc.neg.soft, ml.neg.soft);
  add('accent', cc.accent._, ml.accent._); add('accent2', cc.accent['2'], ml.accent['2']); add('accentSoft', cc.accent.soft, ml.accent.soft);
  add('warn', cc.warn._, ml.warn._); add('warnSoft', cc.warn.soft, ml.warn.soft); add('info', cc.info._, ml.info._); add('infoSoft', cc.info.soft, ml.info.soft);
  add('regimeUp', cc.regime.up, ml.regime.up); add('regimeDown', cc.regime.down, ml.regime.down); add('regimeMr', cc.regime.mr, ml.regime.mr); add('regimeHv', cc.regime.hv, ml.regime.hv);
  add('buy', cc.action.buy, ml.action.buy); add('buySoft', cc.action.buySoft, ml.action.buySoft);
  add('sell', cc.action.sell, ml.action.sell); add('sellSoft', cc.action.sellSoft, ml.action.sellSoft);
  add('watch', cc.action.watch, ml.action.watch); add('watchSoft', cc.action.watchSoft, ml.action.watchSoft);
  add('neutral', cc.action.neutral, ml.action.neutral); add('neutralSoft', cc.action.neutralSoft, ml.action.neutralSoft);

  const radius = Object.entries(tokens.radius).map(([k, v]) => `  static let ${k}: CGFloat = ${v}`).join('\n');
  const space = Object.entries(tokens.space).map(([k, v]) => `  static let s${k}: CGFloat = ${v}`).join('\n');
  const adaptiveEntry = (k, d, l) => `    "${k}": apertureAdaptive(${swiftColor(d)}, ${swiftColor(l)})`;
  return `// ${BANNER}
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
${lines.join('\n')}

  static let accentGrad = LinearGradient(colors: [Color(hex: 0x7C5CFF), Color(hex: 0x4D9FFF)], startPoint: .topLeading, endPoint: .bottomTrailing)
  static let posGrad = LinearGradient(colors: [Color(hex: 0x1ECB8B), Color(hex: 0x4D9FFF)], startPoint: .topLeading, endPoint: .bottomTrailing)

  static let category: [String: Color] = [
${Object.entries(cc.category).map(([k, v]) => `    "${k}": ${swiftColor(v)}`).join(',\n')}
  ]
  static let assetTint: [String: Color] = [
${Object.entries(cc.asset).map(([k, v]) => `    "${k}": ${swiftColor(v)}`).join(',\n')}
  ]
  static let regimeHex: [String: Color] = [
${adaptiveEntry('trending_up', cc.regime.up, ml.regime.up)},
${adaptiveEntry('trending_down', cc.regime.down, ml.regime.down)},
${adaptiveEntry('mean_reverting', cc.regime.mr, ml.regime.mr)},
${adaptiveEntry('high_volatility', cc.regime.hv, ml.regime.hv)}
  ]
  static let staleThresholdSeconds: Double = ${tokens.config.staleThresholdSeconds}
}

/// Corner radii (generated).
enum Radius {
${radius}
}

/// Spacing scale on a 4-pt grid (generated).
enum Space {
${space}
}
`;
}

// ---- write -----------------------------------------------------------------
function write(rel, content) {
  const path = resolve(root, rel);
  if (!existsSync(dirname(path))) mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content);
  console.log(`  ✓ ${rel}`);
}

console.log('gen-tokens: emitting platform files from tokens.json');
write('src/app/tokens.generated.css', cssVars() + cssLight());
write('src/lib/tokens.generated.ts', tsObjects());
write('../aperture-ios/Aperture/DesignSystem/ApertureTokens.generated.swift', swift());
console.log('gen-tokens: done');
