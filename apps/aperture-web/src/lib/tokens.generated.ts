// AUTO-GENERATED from tokens.json by scripts/gen-tokens.mjs — DO NOT EDIT.
export const C = {
  bg0: "#0a0c10",
  bg1: "#0e1116",
  surface1: "#14181f",
  surface2: "#1a1f27",
  surface3: "#222933",
  surfaceInset: "#0c0f14",
  border: "rgba(255,255,255,0.07)",
  borderStrong: "rgba(255,255,255,0.13)",
  grid: "rgba(255,255,255,0.055)",
  text1: "#eef1f6",
  text2: "#a3adbb",
  text3: "#6c7787",
  textInverse: "#0a0c10",
  pos: "#1ecb8b",
  posDim: "#15a673",
  neg: "#f6465d",
  negDim: "#d23a4e",
  warn: "#f0a93b",
  info: "#4d9fff",
  accent: "#7c5cff",
  accent2: "#4d9fff",
  regimeUp: "#1ecb8b",
  regimeDown: "#f6465d",
  regimeMr: "#b07cff",
  regimeHv: "#f0a93b",
  buy: "#1ecb8b",
  sell: "#f6465d",
  watch: "#4d9fff",
  neutral: "#8a93a3"
} as const;

export const C_LIGHT = {
  bg0: "#f5f6f8",
  bg1: "#ebedf1",
  surface1: "#ffffff",
  surface2: "#f1f3f6",
  surface3: "#e6eaef",
  surfaceInset: "#eceff3",
  border: "rgba(16,22,34,0.10)",
  borderStrong: "rgba(16,22,34,0.20)",
  grid: "rgba(16,22,34,0.09)",
  text1: "#161a21",
  text2: "#4f5763",
  text3: "#6c7682",
  textInverse: "#ffffff",
  pos: "#0a8f5e",
  posDim: "#097c52",
  neg: "#cc2436",
  negDim: "#b11f30",
  warn: "#b9760f",
  info: "#1f74e0",
  accent: "#6a45e8",
  accent2: "#1f74e0",
  regimeUp: "#0a8f5e",
  regimeDown: "#cc2436",
  regimeMr: "#7c4fd6",
  regimeHv: "#b9760f",
  buy: "#0a8f5e",
  sell: "#cc2436",
  watch: "#1f74e0",
  neutral: "#5b6470"
} as const;

export const CAT: Record<string, string> = {
  Momentum: "#4d9fff",
  "Mean Reversion": "#b07cff",
  Trend: "#1ecb8b",
  Volatility: "#f0a93b",
  Carry: "#22d3ee",
  Arbitrage: "#f6679a"
};

export const REGIME_HEX: Record<string, string> = {
  trending_up: "#1ecb8b",
  trending_down: "#f6465d",
  mean_reverting: "#b07cff",
  high_volatility: "#f0a93b"
};

export const REGIME_LABEL: Record<string, string> = {
  trending_up: "Trending ↑",
  trending_down: "Trending ↓",
  mean_reverting: "Mean-revert",
  high_volatility: "High vol"
};

export const ASSET_TINT: Record<string, string> = {
  equity: "#4d9fff",
  index: "#b07cff",
  crypto: "#f0a93b",
  future: "#22d3ee"
};

export const STALE_THRESHOLD_SECONDS = 90;
