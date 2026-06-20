// ---------------------------------------------------------------------------
// Mock dataset modeled on the real engine schemas:
//   - TradeIdea / TradeIdeaReport      (src/ui/trade_ideas.py)
//   - Signal + 10 families             (src/signal_battery/*)
//   - Backtest metrics + 3 gates       (src/backtesting/*)
//   - Meta-labeler / calibration / drift (src/ml_layer/*)
//   - Factor risk model                (src/portfolio/factor_risk.py)
// All series are seeded so charts are stable across reloads.
// ---------------------------------------------------------------------------
import { Candle, genCandles, mulberry32, Point } from '../lib/rng';

export type AssetType = 'equity' | 'index' | 'crypto' | 'future';
export type Action = 'BUY' | 'SELL' | 'WATCH' | 'MODEL_REQUIRED' | 'NO_DATA';

export interface Sym {
  symbol: string;
  name: string;
  type: AssetType;
  price: number;
  change1d: number;
  change1w: number;
  change1m: number;
  changeYtd: number;
  spark: number[];
  line: Point[];
  candles: Candle[];
  marketCap: number;
  volume: number;
  hasIdea: boolean;
  bar: BarMicro;
}

// Bar microstructure — mirrors the persisted columns of the `bars` hypertable
// (the Bar dataclass in src/data_engine/models.py). These are real, persisted
// outputs of the Data Engine, unlike the ephemeral Feature Factory values.
export interface BarMicro {
  barType: string;            // BarType.value — e.g. 'tib' | 'dollar'
  vwap: number;
  dollarVolume: number;
  tickCount: number;
  buyVolume: number;
  sellVolume: number;
  volumeImbalance: number;    // buy_volume - sell_volume
  tickImbalanceRatio: number; // (buy_ticks - sell_ticks) / tick_count
  imbalance: number;          // cumulative imbalance that triggered the bar
  threshold: number;          // the threshold that was exceeded
  barDurationSeconds: number;
}

const N = 130;

interface SymDef {
  symbol: string;
  name: string;
  type: AssetType;
  base: number;
  drift: number;
  vol: number;
  seed: number;
  cap: number;
}

const DEFS: SymDef[] = [
  // Equities
  { symbol: 'NVDA', name: 'NVIDIA Corp.', type: 'equity', base: 118, drift: 0.0019, vol: 0.026, seed: 7, cap: 2.9e12 },
  { symbol: 'AAPL', name: 'Apple Inc.', type: 'equity', base: 196, drift: 0.0005, vol: 0.013, seed: 11, cap: 3.0e12 },
  { symbol: 'MSFT', name: 'Microsoft Corp.', type: 'equity', base: 421, drift: 0.0007, vol: 0.013, seed: 13, cap: 3.1e12 },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', type: 'equity', base: 176, drift: 0.0006, vol: 0.016, seed: 17, cap: 2.1e12 },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', type: 'equity', base: 184, drift: 0.0008, vol: 0.018, seed: 19, cap: 1.9e12 },
  { symbol: 'TSLA', name: 'Tesla Inc.', type: 'equity', base: 248, drift: -0.0003, vol: 0.034, seed: 23, cap: 0.79e12 },
  { symbol: 'META', name: 'Meta Platforms', type: 'equity', base: 504, drift: 0.0011, vol: 0.02, seed: 29, cap: 1.3e12 },
  { symbol: 'JPM', name: 'JPMorgan Chase', type: 'equity', base: 205, drift: 0.0004, vol: 0.012, seed: 31, cap: 0.59e12 },
  // Indexes
  { symbol: 'SPX', name: 'S&P 500 Index', type: 'index', base: 5430, drift: 0.0005, vol: 0.0085, seed: 41, cap: 0 },
  { symbol: 'NDX', name: 'Nasdaq 100', type: 'index', base: 19200, drift: 0.0007, vol: 0.011, seed: 43, cap: 0 },
  { symbol: 'RUT', name: 'Russell 2000', type: 'index', base: 2030, drift: 0.0002, vol: 0.013, seed: 47, cap: 0 },
  { symbol: 'VIX', name: 'CBOE Volatility', type: 'index', base: 14.2, drift: -0.0006, vol: 0.05, seed: 53, cap: 0 },
  // Crypto
  { symbol: 'BTC', name: 'Bitcoin', type: 'crypto', base: 61000, drift: 0.0014, vol: 0.028, seed: 61, cap: 1.2e12 },
  { symbol: 'ETH', name: 'Ethereum', type: 'crypto', base: 3380, drift: 0.0011, vol: 0.032, seed: 67, cap: 0.41e12 },
  { symbol: 'SOL', name: 'Solana', type: 'crypto', base: 142, drift: 0.0022, vol: 0.045, seed: 71, cap: 0.066e12 },
  { symbol: 'AVAX', name: 'Avalanche', type: 'crypto', base: 27.4, drift: 0.0006, vol: 0.05, seed: 73, cap: 0.011e12 },
  // Futures
  { symbol: 'ES', name: 'E-mini S&P 500', type: 'future', base: 5442, drift: 0.0005, vol: 0.0088, seed: 79, cap: 0 },
  { symbol: 'CL', name: 'Crude Oil WTI', type: 'future', base: 78.6, drift: -0.0004, vol: 0.02, seed: 83, cap: 0 },
  { symbol: 'GC', name: 'Gold', type: 'future', base: 2338, drift: 0.0009, vol: 0.011, seed: 89, cap: 0 },
];

const IDEA_SYMBOLS = new Set(['NVDA', 'TSLA', 'BTC', 'ETH', 'SOL', 'META', 'AMZN', 'GOOGL', 'CL', 'AVAX', 'JPM', 'AAPL']);

// Derive the latest persisted bar's microstructure from the seeded candle, so
// these columns are stable across reloads and consistent with the price series.
function buildBar(d: SymDef, candles: Candle[]): BarMicro {
  const last = candles[candles.length - 1];
  const rand = mulberry32(d.seed * 131 + 7);
  const isDollar = d.type === 'crypto';

  const volume = last.v;
  // VWAP sits inside the bar's range, weighted toward the close.
  const vwap = Math.min(last.h, Math.max(last.l, (last.h + last.l + last.c * 2) / 4));
  const dollarVolume = vwap * volume;
  const tickCount = Math.round(800 + rand() * 4200);

  // Buy/sell flow leans with the bar's direction (the engine classifies by tick rule).
  const ret = last.o ? (last.c - last.o) / last.o : 0;
  const buyFrac = Math.min(0.62, Math.max(0.38, 0.5 + ret * 6 + (rand() - 0.5) * 0.06));
  const buyVolume = Math.round(volume * buyFrac);
  const sellVolume = volume - buyVolume;
  const buyTicks = Math.round(tickCount * buyFrac);
  const sellTicks = tickCount - buyTicks;
  const sign = buyFrac >= 0.5 ? 1 : -1;

  let imbalance: number;
  let threshold: number;
  if (isDollar) {
    // Dollar bars close on cumulative dollar volume; no imbalance trigger.
    threshold = Math.round(dollarVolume * (0.96 + rand() * 0.03));
    imbalance = 0;
  } else {
    // TIB bars close when |cumulative signed tick imbalance| exceeds a threshold.
    threshold = Math.round(1800 + rand() * 3200);
    imbalance = Math.round(sign * threshold * (1 + rand() * 0.04));
  }

  return {
    barType: isDollar ? 'dollar' : 'tib',
    vwap: Math.round(vwap * 100) / 100,
    dollarVolume: Math.round(dollarVolume),
    tickCount,
    buyVolume,
    sellVolume,
    volumeImbalance: buyVolume - sellVolume,
    tickImbalanceRatio: Math.round(((buyTicks - sellTicks) / tickCount) * 1000) / 1000,
    imbalance,
    threshold,
    barDurationSeconds: Math.round(30 + rand() * 600),
  };
}

function buildSym(d: SymDef): Sym {
  const candles = genCandles(d.seed, N, d.base, d.drift, d.vol);
  const closes = candles.map((c) => c.c);
  const last = closes[closes.length - 1];
  const at = (back: number) => closes[Math.max(0, closes.length - 1 - back)];
  return {
    symbol: d.symbol,
    name: d.name,
    type: d.type,
    price: last,
    change1d: last / at(1) - 1,
    change1w: last / at(5) - 1,
    change1m: last / at(21) - 1,
    changeYtd: last / closes[0] - 1,
    spark: closes.slice(-46),
    line: closes.slice(-46).map((v, i) => ({ t: i, v })),
    candles,
    marketCap: d.cap,
    volume: candles[candles.length - 1].v * last,
    hasIdea: IDEA_SYMBOLS.has(d.symbol),
    bar: buildBar(d, candles),
  };
}

export const SYMBOLS: Sym[] = DEFS.map(buildSym);
export const symBy = (s: string): Sym => SYMBOLS.find((x) => x.symbol === s) ?? SYMBOLS[0];

// ---------------------------------------------------------------------------
// Regime (LSTM detector)
// ---------------------------------------------------------------------------
export interface Regime {
  label: string;
  probabilities: { trending_up: number; trending_down: number; mean_reverting: number; high_volatility: number };
}
export const REGIME: Regime = {
  label: 'trending_up',
  probabilities: { trending_up: 0.71, trending_down: 0.07, mean_reverting: 0.15, high_volatility: 0.07 },
};

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------
export interface Position {
  symbol: string;
  type: AssetType;
  side: number;
  qty: number;
  entryPrice: number;
  markPrice: number;
  weight: number;
  notional: number;
  unrealizedPnl: number;
  unrealizedPct: number;
  strategy: string;
  dayPnl: number;
}

const navCandles = genCandles(900, 180, 1_000_000, 0.0011, 0.006);
const navHistory: Point[] = navCandles.map((c, i) => ({ t: i, v: Math.round(c.c) }));
const NAV = navHistory[navHistory.length - 1].v;

// running drawdown from peak
let peak = -Infinity;
const drawdownHistory: Point[] = navHistory.map((p) => {
  peak = Math.max(peak, p.v);
  return { t: p.t, v: (p.v - peak) / peak };
});

export interface Portfolio {
  nav: number;
  navHistory: Point[];
  drawdownHistory: Point[];
  dailyPnl: number;
  dailyPnlPct: number;
  cumPnl: number;
  cumPnlPct: number;
  drawdown: number;
  grossExposure: number;
  netExposure: number;
  longExposure: number;
  shortExposure: number;
  cashPct: number;
  positions: Position[];
  sharpe: number;
  sortino: number;
  calmar: number;
  volAnn: number;
  maxDd: number;
  winRate: number;
  profitFactor: number;
  beta: number;
}

const RAW_POS: Array<[string, number, number, string]> = [
  // symbol, side, weight, strategy
  ['NVDA', 1, 0.092, 'ts_momentum'],
  ['META', 1, 0.071, 'cs_momentum'],
  ['BTC', 1, 0.083, 'ts_momentum'],
  ['ETH', 1, 0.057, 'donchian_breakout'],
  ['GOOGL', 1, 0.048, 'cs_momentum'],
  ['AMZN', 1, 0.041, 'ma_crossover'],
  ['SOL', 1, 0.039, 'funding_rate_arb'],
  ['TSLA', -1, 0.052, 'mean_reversion'],
  ['CL', -1, 0.036, 'futures_carry'],
  ['JPM', -1, 0.028, 'stat_arb'],
];

const positions: Position[] = RAW_POS.map(([symbol, side, weight, strategy], i) => {
  const s = symBy(symbol);
  const notional = weight * NAV;
  const qty = Math.round((notional / s.price) * (side as number));
  const entry = s.price * (1 - side * (0.02 + (i % 5) * 0.012));
  const uPct = side * (s.price / entry - 1);
  return {
    symbol,
    type: s.type,
    side,
    qty,
    entryPrice: entry,
    markPrice: s.price,
    weight: weight * side,
    notional,
    unrealizedPnl: uPct * notional,
    unrealizedPct: uPct,
    strategy,
    dayPnl: s.change1d * notional * side,
  };
});

const longExp = positions.filter((p) => p.side > 0).reduce((a, p) => a + Math.abs(p.weight), 0);
const shortExp = positions.filter((p) => p.side < 0).reduce((a, p) => a + Math.abs(p.weight), 0);
const dayPnl = positions.reduce((a, p) => a + p.dayPnl, 0);

export const PORTFOLIO: Portfolio = {
  nav: NAV,
  navHistory,
  drawdownHistory,
  dailyPnl: dayPnl,
  dailyPnlPct: dayPnl / NAV,
  cumPnl: NAV - navHistory[0].v,
  cumPnlPct: NAV / navHistory[0].v - 1,
  drawdown: drawdownHistory[drawdownHistory.length - 1].v,
  grossExposure: longExp + shortExp,
  netExposure: longExp - shortExp,
  longExposure: longExp,
  shortExposure: shortExp,
  cashPct: 1 - (longExp + shortExp) + 0.5,
  positions,
  sharpe: 1.84,
  sortino: 2.61,
  calmar: 2.12,
  volAnn: 0.114,
  maxDd: -0.083,
  winRate: 0.563,
  profitFactor: 1.71,
  beta: 0.21,
};

// ---------------------------------------------------------------------------
// Trade Ideas (matches TradeIdea dataclass)
// ---------------------------------------------------------------------------
export interface ShapItem {
  feature: string;
  value: number;
  contribution: number;
  percentile: number;
}
export interface IdeaSignal {
  family: string;
  side: number;
  confidence: number;
  meta: Record<string, string | number>;
}
export interface CascadeStage {
  stage: string;
  value: number;
  binding: boolean;
}
export interface TradeIdea {
  symbol: string;
  type: AssetType;
  action: Action;
  targetWeight: number;
  targetNotional: number;
  estimatedQuantity: number | null;
  latestPrice: number | null;
  barType: string;
  barsLoaded: number;
  featureRows: number;
  signalCount: number;
  topSignalFamily: string | null;
  topSignalSide: number;
  topSignalConfidence: number | null;
  avgSignalConfidence: number | null;
  metaProbability: number | null;
  calibratedProbability: number | null;
  regimeFitScore: number | null;
  betSize: number | null;
  sizingConstraints: string[];
  strategy: string | null;
  reason: string;
  expectedCostBps: number | null;
  topShap: ShapItem | null;
  shap: ShapItem[];
  trackRecordWinRate: number | null;
  trackRecordN: number | null;
  stageLatency: Record<string, number>;
  cascade: CascadeStage[];
  signals: IdeaSignal[];
}

function shapSet(seed: number): ShapItem[] {
  const feats: Array<[string, number]> = [
    ['ts_mom_63', 0.42],
    ['garch_vol', -0.21],
    ['regime_trending_up', 0.33],
    ['rsi_14', -0.14],
    ['order_flow_imbalance', 0.19],
    ['ffd_close_d04', 0.11],
    ['vpin', -0.08],
    ['sentiment_24h', 0.07],
  ];
  return feats
    .map(([feature, base], i) => ({
      feature,
      value: Math.round((base * 3 + ((seed + i) % 7) * 0.1) * 100) / 100,
      contribution: Math.round((base + (((seed + i) % 5) - 2) * 0.02) * 1000) / 1000,
      percentile: Math.min(99, 30 + ((seed * (i + 3)) % 70)),
    }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

interface IdeaDef {
  symbol: string;
  action: Action;
  weight: number;
  family: string;
  side: number;
  meta: number;
  cal: number;
  fit: number;
  reason: string;
  cost: number;
  constraints: string[];
}

const IDEA_DEFS: IdeaDef[] = [
  { symbol: 'NVDA', action: 'BUY', weight: 0.092, family: 'ts_momentum', side: 1, meta: 0.74, cal: 0.71, fit: 0.86, cost: 4.2, constraints: ['vol_target'], reason: 'Strong 3M time-series momentum (z=1.9), confirmed by MA crossover. Meta-labeler 0.74 / calibrated 0.71 in a trending-up regime (fit 0.86).' },
  { symbol: 'BTC', action: 'BUY', weight: 0.083, family: 'ts_momentum', side: 1, meta: 0.7, cal: 0.68, fit: 0.81, cost: 6.1, constraints: ['crypto_cap'], reason: 'Donchian 55-day breakout with positive funding carry. Trend + carry agree; sizing clipped by 30% crypto cap.' },
  { symbol: 'META', action: 'BUY', weight: 0.071, family: 'ts_momentum', side: 1, meta: 0.69, cal: 0.66, fit: 0.78, cost: 3.4, constraints: [], reason: 'Strong 12-month time-series momentum, vol-normalized (z=1.6). Trend intact; calibrated 0.66 in a supportive regime.' },
  { symbol: 'ETH', action: 'BUY', weight: 0.057, family: 'donchian_breakout', side: 1, meta: 0.64, cal: 0.62, fit: 0.74, cost: 5.8, constraints: ['crypto_cap'], reason: 'Channel breakout above 55-bar high; ATR-normalized size. Crypto sleeve near cap.' },
  { symbol: 'TSLA', action: 'SELL', weight: -0.052, family: 'mean_reversion', side: -1, meta: 0.67, cal: 0.64, fit: 0.71, cost: 4.9, constraints: ['kelly_cap'], reason: 'O-U z-score +2.3 above mean (half-life 14 bars, ADF p=0.03). Fade the deviation; Kelly cap binding.' },
  { symbol: 'GOOGL', action: 'BUY', weight: 0.048, family: 'ma_crossover', side: 1, meta: 0.61, cal: 0.59, fit: 0.69, cost: 3.1, constraints: [], reason: '20/50 EMA bullish crossover with triple-MA confirmation. Moderate conviction.' },
  { symbol: 'AMZN', action: 'BUY', weight: 0.041, family: 'ma_crossover', side: 1, meta: 0.6, cal: 0.58, fit: 0.66, cost: 3.3, constraints: [], reason: '20/50 EMA bullish crossover with triple-MA confirmation.' },
  { symbol: 'SOL', action: 'BUY', weight: 0.039, family: 'donchian_breakout', side: 1, meta: 0.66, cal: 0.63, fit: 0.7, cost: 7.4, constraints: ['crypto_cap'], reason: 'Donchian 55-bar channel breakout; ATR-normalized size. Crypto sleeve near the cap.' },
  { symbol: 'CL', action: 'SELL', weight: -0.036, family: 'mean_reversion', side: -1, meta: 0.58, cal: 0.56, fit: 0.61, cost: 2.8, constraints: [], reason: 'O-U z-score +2.1 above mean (half-life 18 bars, ADF p=0.04). Fade the deviation — short.' },
  { symbol: 'JPM', action: 'SELL', weight: -0.028, family: 'mean_reversion', side: -1, meta: 0.57, cal: 0.55, fit: 0.6, cost: 2.2, constraints: [], reason: 'O-U mean-reversion short: z-score +2.0 above mean on a stationary series.' },
  { symbol: 'AVAX', action: 'WATCH', weight: 0, family: 'donchian_breakout', side: 1, meta: 0.52, cal: 0.5, fit: 0.55, cost: 8.1, constraints: ['below_threshold'], reason: 'Approaching breakout but meta-prob 0.52 below 0.55 entry gate. Watching.' },
  { symbol: 'AAPL', action: 'WATCH', weight: 0, family: 'mean_reversion', side: 0, meta: 0.49, cal: 0.48, fit: 0.52, cost: 2.6, constraints: ['below_threshold'], reason: 'Z-score near band but not at entry threshold. No actionable edge yet.' },
  { symbol: 'RUT', action: 'MODEL_REQUIRED', weight: 0, family: 'ts_momentum', side: 0, meta: null as unknown as number, cal: null as unknown as number, fit: 0.4, cost: 0, constraints: ['insufficient_history'], reason: 'Insufficient bar history for the meta-labeler this cycle — inference skipped.' },
];

export const TRADE_IDEAS: TradeIdea[] = IDEA_DEFS.map((d, i) => {
  const s = symBy(d.symbol);
  const notional = Math.abs(d.weight) * NAV;
  const cascade: CascadeStage[] = [
    { stage: 'AFML size', value: Math.abs(d.weight) * 1.7, binding: false },
    { stage: 'Kelly cap', value: Math.abs(d.weight) * 1.35, binding: d.constraints.includes('kelly_cap') },
    { stage: 'Vol target', value: Math.abs(d.weight) * 1.12, binding: d.constraints.includes('vol_target') },
    { stage: 'ATR cap', value: Math.abs(d.weight) * 1.04, binding: false },
    { stage: 'Risk budget', value: Math.abs(d.weight), binding: d.constraints.includes('crypto_cap') },
  ];
  const signals: IdeaSignal[] = [
    { family: d.family, side: d.side, confidence: d.meta ? d.meta + 0.05 : 0.5, meta: famMeta(d.family) },
  ];
  if (i % 2 === 0 && d.action !== 'MODEL_REQUIRED') {
    signals.push({ family: 'ma_crossover', side: d.side, confidence: 0.41, meta: { fast_ema: s.price * 0.99, slow_ema: s.price * 0.97 } });
  }
  const shap = shapSet(i + 3);
  return {
    symbol: d.symbol,
    type: s.type,
    action: d.action,
    targetWeight: d.weight,
    targetNotional: d.weight * NAV,
    estimatedQuantity: d.weight ? Math.round((notional / s.price) * d.side) : 0,
    latestPrice: s.price,
    barType: s.type === 'crypto' ? 'dollar' : 'tib',
    barsLoaded: 980 + i * 7,
    featureRows: 940 + i * 6,
    signalCount: signals.length,
    topSignalFamily: d.family,
    topSignalSide: d.side,
    topSignalConfidence: d.meta ? d.meta + 0.05 : null,
    avgSignalConfidence: d.meta ? d.meta - 0.02 : null,
    metaProbability: d.meta,
    calibratedProbability: d.cal,
    regimeFitScore: d.fit,
    betSize: d.weight ? Math.abs(d.weight) : 0,
    sizingConstraints: d.constraints,
    strategy: d.family,
    reason: d.reason,
    expectedCostBps: d.cost,
    topShap: shap[0],
    shap,
    trackRecordWinRate: d.meta ? 0.5 + (i % 5) * 0.03 : null,
    trackRecordN: d.meta ? 40 + i * 9 : null,
    stageLatency: {
      data_fetch: 0.04 + (i % 3) * 0.01,
      feature_compute: 0.11 + (i % 4) * 0.02,
      signal_generation: 0.03,
      meta_inference: 0.02,
      sizing: 0.01,
      target_generation: 0.01,
    },
    cascade,
    signals,
  };
});

function famMeta(family: string): Record<string, string | number> {
  switch (family) {
    case 'ts_momentum':
      return { lookbacks: '21/63/126/252', z_63: 1.9, aggregate: 0.72 };
    case 'cs_momentum':
      return { decile_rank: 9, lookback_return: 0.34, skip: '21d' };
    case 'mean_reversion':
      return { half_life: 14, adf_pvalue: 0.03, z_score: 2.3 };
    case 'donchian_breakout':
      return { entry_period: 55, exit_period: 20, event: 'entry' };
    case 'ma_crossover':
      return { fast_period: 20, slow_period: 50 };
    case 'funding_rate_arb':
      return { annualized_funding: 0.18, payments_per_day: 3 };
    case 'futures_carry':
      return { carry: -0.06, days_to_expiry: 28 };
    case 'stat_arb':
      return { hedge_ratio: 0.84, spread_z: 2.1, half_life: 9 };
    default:
      return {};
  }
}

export const IDEA_TOTALS = {
  buy: TRADE_IDEAS.filter((i) => i.action === 'BUY').length,
  sell: TRADE_IDEAS.filter((i) => i.action === 'SELL').length,
  watch: TRADE_IDEAS.filter((i) => i.action === 'WATCH').length,
  modelRequired: TRADE_IDEAS.filter((i) => i.action === 'MODEL_REQUIRED').length,
  grossTargetWeight: TRADE_IDEAS.reduce((a, i) => a + Math.abs(i.targetWeight), 0),
  netTargetWeight: TRADE_IDEAS.reduce((a, i) => a + i.targetWeight, 0),
};

// ---------------------------------------------------------------------------
// Strategies (10 signal families)
// ---------------------------------------------------------------------------
export type StratStatus = 'live' | 'shadow' | 'paused';
export interface Strategy {
  id: string;
  name: string;
  category: string;
  source: string;
  thesis: string;
  status: StratStatus;
  sharpe: number;
  winRate: number;
  trades: number;
  contributionPct: number;
  pnlYtd: number;
  allocation: number;
  regimeFit: Record<string, number>;
  params: Array<{ key: string; value: string }>;
  equityCurve: Point[];
  activeSignals: number;
  avgHoldBars: number;
  assetClasses: AssetType[];
}

interface StratDef extends Omit<Strategy, 'equityCurve'> {
  seed: number;
  drift: number;
}

const STRAT_DEFS: StratDef[] = [
  { id: 'ts_momentum', name: 'Time-Series Momentum', category: 'Momentum', source: 'Clenow · Chan', status: 'live', sharpe: 1.12, winRate: 0.54, trades: 318, contributionPct: 0.26, pnlYtd: 0.082, allocation: 0.22, activeSignals: 6, avgHoldBars: 34, assetClasses: ['equity', 'crypto', 'future'], seed: 201, drift: 0.0016, regimeFit: { trending_up: 0.86, trending_down: 0.62, mean_reverting: 0.21, high_volatility: 0.4 }, params: [{ key: 'lookbacks', value: '21 / 63 / 126 / 252' }, { key: 'history_window', value: '252' }, { key: 'vol_normalize', value: 'true' }], thesis: 'Per-asset momentum across multiple lookbacks, volatility-normalized to z-scores and weighted into a single conviction. Goes long winners / short losers. Best in persistent trends.' },
  { id: 'cs_momentum', name: 'Cross-Sectional Momentum', category: 'Momentum', source: 'Jansen', status: 'live', sharpe: 0.94, winRate: 0.52, trades: 142, contributionPct: 0.17, pnlYtd: 0.051, allocation: 0.15, activeSignals: 4, avgHoldBars: 42, assetClasses: ['equity'], seed: 211, drift: 0.0012, regimeFit: { trending_up: 0.78, trending_down: 0.55, mean_reverting: 0.3, high_volatility: 0.34 }, params: [{ key: 'lookback', value: '252 bars' }, { key: 'skip', value: '21 bars' }, { key: 'deciles', value: 'top 0.9 / bottom 0.1' }], thesis: '12-month momentum with a 1-month skip to dodge short-term reversal. Ranks the cross-section, longs the top decile and shorts the bottom. Panel-relative alpha.' },
  { id: 'mean_reversion', name: 'Mean Reversion (O-U)', category: 'Mean Reversion', source: 'Chan', status: 'live', sharpe: 1.03, winRate: 0.61, trades: 287, contributionPct: 0.14, pnlYtd: 0.044, allocation: 0.12, activeSignals: 3, avgHoldBars: 11, assetClasses: ['equity', 'index'], seed: 221, drift: 0.0011, regimeFit: { trending_up: 0.24, trending_down: 0.28, mean_reverting: 0.88, high_volatility: 0.46 }, params: [{ key: 'entry_z', value: '2.0' }, { key: 'exit_z', value: '0.5' }, { key: 'half_life', value: '1–100 bars' }, { key: 'adf_pvalue', value: '≤ 0.05' }], thesis: 'Fits an Ornstein-Uhlenbeck process; trades stationary series back toward the mean when the z-score breaches ±2σ. Half-life and ADF gate which names are tradeable.' },
  { id: 'ma_crossover', name: 'MA Crossover', category: 'Trend', source: 'Clenow', status: 'live', sharpe: 0.71, winRate: 0.46, trades: 96, contributionPct: 0.07, pnlYtd: 0.021, allocation: 0.08, activeSignals: 2, avgHoldBars: 51, assetClasses: ['equity', 'future'], seed: 231, drift: 0.0008, regimeFit: { trending_up: 0.82, trending_down: 0.7, mean_reverting: 0.18, high_volatility: 0.3 }, params: [{ key: 'fast', value: '20 EMA' }, { key: 'slow', value: '50 EMA' }, { key: 'triple_ma', value: 'on' }], thesis: 'Classic EMA crossover with optional triple-MA 2-of-3 voting. Slow, robust trend capture that complements the faster momentum sleeves.' },
  { id: 'donchian_breakout', name: 'Donchian Breakout', category: 'Trend', source: 'Clenow', status: 'live', sharpe: 0.88, winRate: 0.43, trades: 124, contributionPct: 0.1, pnlYtd: 0.033, allocation: 0.1, activeSignals: 3, avgHoldBars: 47, assetClasses: ['crypto', 'future'], seed: 241, drift: 0.001, regimeFit: { trending_up: 0.84, trending_down: 0.66, mean_reverting: 0.16, high_volatility: 0.5 }, params: [{ key: 'entry_channel', value: '55 bars' }, { key: 'exit_channel', value: '20 bars' }, { key: 'sizing', value: 'ATR units' }], thesis: 'Turtle-style channel breakout: enter on a new 55-bar extreme, exit on the opposing 20-bar channel. ATR-normalized position sizing. Captures fat-tailed trends.' },
  { id: 'vrp', name: 'Volatility Risk Premium', category: 'Volatility', source: 'Sinclair', status: 'live', sharpe: 1.21, winRate: 0.64, trades: 73, contributionPct: 0.09, pnlYtd: 0.03, allocation: 0.07, activeSignals: 1, avgHoldBars: 18, assetClasses: ['index'], seed: 251, drift: 0.0013, regimeFit: { trending_up: 0.5, trending_down: 0.4, mean_reverting: 0.6, high_volatility: 0.86 }, params: [{ key: 'vrp_lookback', value: '30 bars' }, { key: 'high_pct', value: '75' }, { key: 'low_pct', value: '25' }], thesis: 'Trades the gap between implied and realized vol. Sells vol when VRP is rich, buys when cheap, and emits a regime modifier that scales the other families.' },
  { id: 'futures_carry', name: 'Futures Carry', category: 'Carry', source: 'Clenow', status: 'live', sharpe: 0.79, winRate: 0.57, trades: 64, contributionPct: 0.05, pnlYtd: 0.017, allocation: 0.06, activeSignals: 2, avgHoldBars: 63, assetClasses: ['future'], seed: 261, drift: 0.0007, regimeFit: { trending_up: 0.55, trending_down: 0.5, mean_reverting: 0.52, high_volatility: 0.38 }, params: [{ key: 'annualize', value: 'true' }, { key: 'conf_window', value: '252' }], thesis: 'Roll-yield harvest from the front/back futures spread. Long backwardation (positive roll), short contango (roll drag). Slow, diversifying carry.' },
  { id: 'funding_rate_arb', name: 'Funding-Rate Arb', category: 'Carry', source: 'Crypto', status: 'live', sharpe: 1.42, winRate: 0.72, trades: 51, contributionPct: 0.08, pnlYtd: 0.027, allocation: 0.06, activeSignals: 2, avgHoldBars: 28, assetClasses: ['crypto'], seed: 271, drift: 0.0014, regimeFit: { trending_up: 0.6, trending_down: 0.58, mean_reverting: 0.55, high_volatility: 0.42 }, params: [{ key: 'entry', value: '10% annualized' }, { key: 'exit', value: '2% annualized' }, { key: 'cadence', value: '3×/day' }], thesis: 'Delta-neutral crypto carry: long spot, short perpetual, collect funding. Fires when annualized funding clears the entry threshold. High Sharpe, capacity-limited.' },
  { id: 'stat_arb', name: 'Statistical Arbitrage', category: 'Arbitrage', source: 'Chan', status: 'live', sharpe: 0.97, winRate: 0.59, trades: 203, contributionPct: 0.06, pnlYtd: 0.02, allocation: 0.05, activeSignals: 2, avgHoldBars: 13, assetClasses: ['equity'], seed: 281, drift: 0.0011, regimeFit: { trending_up: 0.4, trending_down: 0.42, mean_reverting: 0.82, high_volatility: 0.5 }, params: [{ key: 'entry_z', value: '2.0' }, { key: 'exit_z', value: '0.5' }, { key: 'hedge', value: 'Kalman dynamic' }], thesis: 'Cointegration pairs trading with a Kalman-filtered dynamic hedge ratio. Trades the mean-reverting spread; Engle-Granger / Johansen gate the pair selection.' },
  { id: 'cross_exchange_arb', name: 'Cross-Exchange Arb', category: 'Arbitrage', source: 'Crypto', status: 'shadow', sharpe: 1.55, winRate: 0.81, trades: 38, contributionPct: 0.03, pnlYtd: 0.009, allocation: 0.02, activeSignals: 0, avgHoldBars: 1, assetClasses: ['crypto'], seed: 291, drift: 0.0009, regimeFit: { trending_up: 0.5, trending_down: 0.5, mean_reverting: 0.5, high_volatility: 0.7 }, params: [{ key: 'min_spread', value: '10 bps' }, { key: 'fee_estimate', value: '20 bps' }], thesis: 'Bar-level spot arbitrage across venues. Buys the cheap book, sells the rich one when the spread clears fees. Delta-neutral; currently in shadow pending live venue keys.' },
];

export const STRATEGIES: Strategy[] = STRAT_DEFS.map((d) => {
  const { seed, drift, ...rest } = d;
  return { ...rest, equityCurve: genCandles(seed, 130, 100, drift, 0.01).map((c, i) => ({ t: i, v: c.c })) };
});
export const stratBy = (id: string): Strategy => STRATEGIES.find((s) => s.id === id) ?? STRATEGIES[0];

// ---------------------------------------------------------------------------
// Backtest / validation
// ---------------------------------------------------------------------------
const eqCandles = genCandles(500, 250, 100, 0.0013, 0.0075);
const benchCandles = genCandles(501, 250, 100, 0.0006, 0.0095);
let bpeak = -Infinity;
const btDrawdown: Point[] = eqCandles.map((c) => {
  bpeak = Math.max(bpeak, c.c);
  return { t: c.t, v: (c.c - bpeak) / bpeak };
});

export interface BacktestGate {
  name: string;
  value: number;
  threshold: number;
  pass: boolean;
  detail: string;
}
export const BACKTEST = {
  name: 'Ensemble · walk-forward 2019–2026',
  equityCurve: eqCandles.map((c) => ({ t: c.t, v: c.c })),
  benchmark: benchCandles.map((c) => ({ t: c.t, v: c.c })),
  drawdownCurve: btDrawdown,
  metrics: {
    totalReturn: 1.94,
    annReturn: 0.213,
    annVol: 0.116,
    sharpe: 1.84,
    sortino: 2.61,
    calmar: 2.12,
    maxDd: -0.101,
    maxDdDur: 47,
    winRate: 0.563,
    profitFactor: 1.71,
    avgTrade: 0.0021,
    turnover: 8.4,
    costDragBps: 38,
    skew: 0.31,
    kurtosis: 4.2,
    tailRatio: 1.18,
    trades: 1843,
  },
  gates: [
    { name: 'CPCV', value: 0.71, threshold: 0.6, pass: true, detail: '32 / 45 combinatorial paths positive (≥ 60% required)' },
    { name: 'Deflated Sharpe', value: 0.011, threshold: 0.05, pass: true, detail: 'p = 0.011 after deflating for trials (< 0.05 required)' },
    { name: 'PBO', value: 0.28, threshold: 0.4, pass: true, detail: 'Probability of backtest overfitting 0.28 (< 0.40 required)' },
  ] as BacktestGate[],
  // 45 combinatorial-purged CV paths; ~71% positive to match the CPCV gate.
  cpcvPaths: Array.from({ length: 45 }, (_, i) => {
    const r = mulberry32(700 + i * 17)();
    return Math.round((r - 0.29) * 0.9 * 100) / 100;
  }),
  monthlyReturns: [
    { year: 2024, months: [0.021, -0.008, 0.034, 0.012, 0.019, -0.015, 0.027, 0.008, 0.031, 0.014, 0.022, 0.018] },
    { year: 2025, months: [0.018, 0.025, -0.012, 0.029, 0.016, 0.021, -0.006, 0.033, 0.011, 0.024, 0.019, 0.027] },
    { year: 2026, months: [0.023, 0.017, 0.031, -0.009, 0.026, null, null, null, null, null, null, null] },
  ] as Array<{ year: number; months: (number | null)[] }>,
  regimeBreakdown: [
    { regime: 'trending_up', sharpe: 2.31, ret: 0.118, trades: 642 },
    { regime: 'trending_down', sharpe: 1.12, ret: 0.041, trades: 388 },
    { regime: 'mean_reverting', sharpe: 1.74, ret: 0.067, trades: 561 },
    { regime: 'high_volatility', sharpe: 0.82, ret: 0.022, trades: 252 },
  ],
  tradeLog: TRADE_IDEAS.slice(0, 9).map((idea, i) => ({
    symbol: idea.symbol,
    side: idea.topSignalSide,
    family: idea.topSignalFamily ?? 'ts_momentum',
    entry: idea.latestPrice ?? 100,
    exit: (idea.latestPrice ?? 100) * (1 + idea.topSignalSide * (0.01 + (i % 4) * 0.012)),
    netPnl: (i % 3 === 0 ? -1 : 1) * (520 + i * 180),
    holdBars: 8 + i * 3,
    metaProb: idea.metaProbability ?? 0.6,
    returnPct: (i % 3 === 0 ? -1 : 1) * (0.012 + (i % 4) * 0.006),
  })),
};

// ---------------------------------------------------------------------------
// Model & features
// ---------------------------------------------------------------------------
export const MODEL = {
  version: 'meta_v1.7.2',
  trainedAt: '2026-05-12',
  lastRetrainHours: 62,
  runId: 'a1b2c3d4e5f64789',
  type: 'LightGBM meta-labeler + isotonic calibration',
  // Real MLflow run metrics/params (these ARE logged). AUC/Brier/ECE below are
  // NOT logged by the engine — kept only so legacy refs don't break; the Model
  // screen renders cvScore/trainAcc/trainingEvents instead.
  cvScore: 0.582, // mean of 5-fold CV
  trainAcc: 0.631, // in-sample accuracy
  trainingEvents: 4820, // n_training_events
  // Promotion gate flags from the production run. Uniformly false because the
  // retrain gate is hard-broken (retrain_pipeline.py:265) → shown as "not run",
  // never green-pass / red-fail.
  gates: { cpcv: false, dsr: false, pbo: false },
  auc: 0.673,
  brier: 0.214,
  ece: 0.031,
  calibration: Array.from({ length: 10 }, (_, i) => {
    const lo = i / 10;
    const predicted = lo + 0.05;
    const observed = Math.max(0, Math.min(1, predicted + (i < 5 ? 0.03 : -0.04) + ((i % 3) - 1) * 0.015));
    return { lo, hi: lo + 0.1, predicted, observed, count: 40 + ((i * 37) % 120) };
  }),
  featureImportance: [
    { feature: 'ts_mom_63', importance: 0.142, family: 'Momentum' },
    { feature: 'garch_vol', importance: 0.118, family: 'Volatility' },
    { feature: 'regime_prob_up', importance: 0.097, family: 'Regime' },
    { feature: 'order_flow_imbalance', importance: 0.083, family: 'Microstructure' },
    { feature: 'ffd_close_d04', importance: 0.071, family: 'FracDiff' },
    { feature: 'rsi_14', importance: 0.064, family: 'Classical' },
    { feature: 'vpin', importance: 0.058, family: 'Microstructure' },
    { feature: 'cs_mom_decile', importance: 0.052, family: 'Momentum' },
    { feature: 'kyle_lambda', importance: 0.047, family: 'Microstructure' },
    { feature: 'sentiment_24h', importance: 0.039, family: 'Sentiment' },
    { feature: 'realized_vol_short', importance: 0.034, family: 'Volatility' },
    { feature: 'sadf', importance: 0.028, family: 'StructuralBreak' },
  ],
  drift: [
    { feature: 'ts_mom_63', kl: 0.04, severity: 'ok' },
    { feature: 'garch_vol', kl: 0.11, severity: 'ok' },
    { feature: 'order_flow_imbalance', kl: 0.27, severity: 'warn' },
    { feature: 'vpin', kl: 0.42, severity: 'alert' },
    { feature: 'sentiment_24h', kl: 0.09, severity: 'ok' },
    { feature: 'rsi_14', kl: 0.06, severity: 'ok' },
  ] as Array<{ feature: string; kl: number; severity: 'ok' | 'warn' | 'alert' }>,
  metaProbHist: [
    { bucket: '0.0', count: 18 },
    { bucket: '0.1', count: 42 },
    { bucket: '0.2', count: 88 },
    { bucket: '0.3', count: 141 },
    { bucket: '0.4', count: 196 },
    { bucket: '0.5', count: 173 },
    { bucket: '0.6', count: 121 },
    { bucket: '0.7', count: 74 },
    { bucket: '0.8', count: 38 },
    { bucket: '0.9', count: 12 },
  ],
  rlShadow: { agreement: 0.78, shadowSharpe: 1.91, liveSharpe: 1.84, status: 'shadow · auto-revert armed' },
  retrainTimeline: [
    { date: '2026-05-12', event: 'Promoted meta_v1.7.2', sharpe: 1.84, promoted: true },
    { date: '2026-04-28', event: 'Promoted meta_v1.7.1', sharpe: 1.79, promoted: true },
    { date: '2026-04-14', event: 'Rejected — PBO 0.46', sharpe: 1.71, promoted: false },
    { date: '2026-03-31', event: 'Promoted meta_v1.7.0', sharpe: 1.77, promoted: true },
  ],
};

// ---------------------------------------------------------------------------
// Factor risk model (PCA)
// ---------------------------------------------------------------------------
export const FACTOR_MODEL = {
  totalRisk: 0.114,
  systematicPct: 0.68,
  idiosyncraticPct: 0.32,
  factors: [
    { name: 'Market (PC1)', exposure: 0.31, contribution: 0.041, explainedVar: 0.44 },
    { name: 'Momentum (PC2)', exposure: 0.58, contribution: 0.029, explainedVar: 0.19 },
    { name: 'Size (PC3)', exposure: -0.18, contribution: 0.012, explainedVar: 0.11 },
    { name: 'Crypto-beta (PC4)', exposure: 0.42, contribution: 0.018, explainedVar: 0.09 },
    { name: 'Vol (PC5)', exposure: -0.22, contribution: 0.008, explainedVar: 0.06 },
  ],
};

// ---------------------------------------------------------------------------
// System status (status bar / health)
// ---------------------------------------------------------------------------
export const SYSTEM = {
  mode: 'Paper',
  dataFreshnessSec: 2.4,
  modelAgeHours: 62,
  brokerOk: true,
  breakers: 0,
  alerts: 1,
  lastRefreshSec: 8,
  sourceFreshness: [
    { source: 'Bars', sec: 2.1 },
    { source: 'Features', sec: 4.7 },
    { source: 'Signals', sec: 5.2 },
    { source: 'Sentiment', sec: 51 },
    { source: 'On-chain', sec: 88 },
    { source: 'Funding', sec: 12 },
  ],
};

export const REGIME_COLORS: Record<string, string> = {
  trending_up: 'var(--regime-up)',
  trending_down: 'var(--regime-down)',
  mean_reverting: 'var(--regime-mr)',
  high_volatility: 'var(--regime-hv)',
};

export const CATEGORY_COLORS: Record<string, string> = {
  Momentum: '#4d9fff',
  'Mean Reversion': '#b07cff',
  Trend: '#1ecb8b',
  Volatility: '#f0a93b',
  Carry: '#22d3ee',
  Arbitrage: '#f6679a',
};
