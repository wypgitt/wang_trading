// Hex mirrors of theme.css tokens. Recharts/SVG need real colors (CSS vars do
// not resolve inside SVG presentation attributes), so charts import from here.
export const C = {
  bg0: '#0a0c10',
  bg1: '#0e1116',
  surface1: '#14181f',
  surface2: '#1a1f27',
  surface3: '#222933',
  surfaceInset: '#0c0f14',
  border: 'rgba(255,255,255,0.07)',
  borderStrong: 'rgba(255,255,255,0.13)',
  text1: '#eef1f6',
  text2: '#a3adbb',
  text3: '#6c7787',
  pos: '#1ecb8b',
  neg: '#f6465d',
  warn: '#f0a93b',
  info: '#4d9fff',
  accent: '#7c5cff',
  accent2: '#4d9fff',
  grid: 'rgba(255,255,255,0.055)',
  regimeUp: '#1ecb8b',
  regimeDown: '#f6465d',
  regimeMr: '#b07cff',
  regimeHv: '#f0a93b',
};

export const CAT: Record<string, string> = {
  Momentum: '#4d9fff',
  'Mean Reversion': '#b07cff',
  Trend: '#1ecb8b',
  Volatility: '#f0a93b',
  Carry: '#22d3ee',
  Arbitrage: '#f6679a',
};

export const REGIME_HEX: Record<string, string> = {
  trending_up: '#1ecb8b',
  trending_down: '#f6465d',
  mean_reverting: '#b07cff',
  high_volatility: '#f0a93b',
};

export const REGIME_LABEL: Record<string, string> = {
  trending_up: 'Trending ↑',
  trending_down: 'Trending ↓',
  mean_reverting: 'Mean-revert',
  high_volatility: 'High vol',
};

export const ASSET_TINT: Record<string, string> = {
  equity: '#4d9fff',
  index: '#b07cff',
  crypto: '#f0a93b',
  future: '#22d3ee',
};
