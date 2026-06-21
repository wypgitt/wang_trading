// ---------------------------------------------------------------------------
// Data access layer. Every accessor is async and returns an ApiEnvelope so the
// screens bind to the same trust contract the FastAPI BFF (src/web) ships.
//
//   - Mock-first by default: when NEXT_PUBLIC_APERTURE_BFF_URL is unset the
//     accessors resolve instantly from the seeded mock dataset (the demo).
//   - Live: when the BFF URL is set, each accessor fetches the real endpoint
//     and MAPS the BFF's honest (camelCase) response onto the frontend shape.
//     Fields the engine does not produce yet (marketCap, calendar changes,
//     SHAP / cascade / per-signal rows, family performance, model AUC/drift/
//     calibration, …) are mapped to null/[] — never fabricated — so the UI's
//     Coming / ModelGated / DataUnavailable states light up honestly.
//
// Accessors NEVER reject: a transport failure is normalized into an error
// envelope (valid-empty data + errors[] + request id) so a screen always
// renders its Error state rather than crashing.
// ---------------------------------------------------------------------------
import {
  Action,
  AssetType,
  MODEL,
  STRATEGIES,
  SYMBOLS,
  Strategy,
  Sym,
  TRADE_IDEAS,
  TradeIdea,
  ShapItem,
  stratBy,
  symBy,
} from './mock';
import { BarMicro } from './mock';
import { ApiEnvelope, envelope, nextRequestId } from './envelope';
import { BffError, USE_BFF, fetchEnvelope } from './bff';

export * from './mock';
export type { ApiEnvelope } from './envelope';

// ---- live plumbing ---------------------------------------------------------

function transportErrorEnvelope<F>(empty: F, err: BffError): ApiEnvelope<F> {
  return {
    data: empty,
    as_of: new Date().toISOString(),
    source: 'bff',
    staleness_seconds: null,
    source_freshness: null,
    model_version: null,
    regime: null,
    warnings: [],
    errors: [{ code: 'INTERNAL', message: err?.message ?? 'request failed' }],
    request_id: err?.requestId ?? nextRequestId(),
  };
}

// Fetch + map one endpoint. `empty` is the valid-empty data used when the BFF
// returns data:null (honest error envelope) or the transport fails.
async function live<B, F>(
  path: string,
  map: (b: B) => F,
  empty: F,
): Promise<ApiEnvelope<F>> {
  try {
    const env = await fetchEnvelope<B>(path);
    return { ...env, data: env.data == null ? empty : map(env.data) };
  } catch (e) {
    return transportErrorEnvelope<F>(empty, e as BffError);
  }
}

// Asset-class lookup from the static universe (mock mirrors the BFF reference
// map). Used to tag a bare TradeIdea with its glyph type.
const _SYM_TYPE: Record<string, AssetType> = Object.fromEntries(
  SYMBOLS.map((s) => [s.symbol, s.type]),
);
const symType = (symbol: string): AssetType => _SYM_TYPE[symbol] ?? 'equity';

// ---- BFF response shapes (camelCase; exclude_none ⇒ optional fields) --------

interface BffBarMicro {
  barType: string;
  vwap?: number; dollarVolume?: number; tickCount?: number;
  buyVolume?: number; sellVolume?: number; volumeImbalance?: number;
  tickImbalanceRatio?: number; imbalance?: number; threshold?: number;
  barDurationSeconds?: number;
}
interface BffShap { feature: string; value: number; contribution: number; absContribution?: number; percentile: number; }
interface BffIdea {
  symbol: string; action: Action; targetWeight?: number; targetNotional?: number;
  estimatedQuantity?: number; latestPrice?: number; barType?: string;
  barsLoaded?: number; featureRows?: number; signalCount?: number;
  topSignalFamily?: string; topSignalSide?: number; topSignalConfidence?: number;
  avgSignalConfidence?: number; metaProbability?: number; calibratedProbability?: number;
  regimeFitScore?: number; betSize?: number; sizingConstraintsApplied?: string[];
  strategy?: string; reason?: string; expectedCostBps?: number; topShapFeature?: BffShap;
  trackRecordWinRate?: number; trackRecordN?: number; stageLatencySeconds?: Record<string, number>;
}
interface BffMarketRow {
  symbol: string; name: string; type: AssetType; price?: number; spark?: number[];
  changeWindowPct?: number; volume?: number; barsLoaded?: number; latestBarAt?: string;
  hasIdea?: boolean; action?: Action; targetWeight?: number; bar?: BffBarMicro;
  dataAvailable?: boolean; marketCap?: number;
}
interface BffSymbolView extends BffMarketRow {
  barType?: string; candles?: { t: number; o: number; h: number; l: number; c: number; v: number }[];
  line?: { t: number; v: number }[];
}
interface BffFamily {
  id: string; name: string; category: string; source: string; thesis: string;
  status: 'live' | 'shadow' | 'paused'; assetClasses?: AssetType[];
  params?: { key: string; value: string }[]; activeSignals?: number;
  sharpe?: number; winRate?: number; trades?: number; contributionPct?: number;
  pnlYtd?: number; allocation?: number; avgHoldBars?: number;
  regimeFit?: Record<string, number>; equityCurve?: { t: number; v: number }[];
}

// ---- mappers (BFF → frontend) ----------------------------------------------

function mapBar(b?: BffBarMicro): BarMicro | null {
  if (!b) return null;
  return {
    barType: b.barType,
    vwap: b.vwap ?? 0,
    dollarVolume: b.dollarVolume ?? 0,
    tickCount: b.tickCount ?? 0,
    buyVolume: b.buyVolume ?? 0,
    sellVolume: b.sellVolume ?? 0,
    volumeImbalance: b.volumeImbalance ?? 0,
    tickImbalanceRatio: b.tickImbalanceRatio ?? 0,
    imbalance: b.imbalance ?? 0,
    threshold: b.threshold ?? 0,
    barDurationSeconds: b.barDurationSeconds ?? 0,
  };
}

function mapShap(s: BffShap): ShapItem {
  return { feature: s.feature, value: s.value, contribution: s.contribution, percentile: s.percentile };
}

function mapIdea(b: BffIdea): TradeIdea {
  return {
    symbol: b.symbol,
    type: symType(b.symbol),
    action: b.action,
    targetWeight: b.targetWeight ?? 0,
    targetNotional: b.targetNotional ?? 0,
    estimatedQuantity: b.estimatedQuantity ?? null,
    latestPrice: b.latestPrice ?? null,
    barType: b.barType ?? '',
    barsLoaded: b.barsLoaded ?? 0,
    featureRows: b.featureRows ?? 0,
    signalCount: b.signalCount ?? 0,
    topSignalFamily: b.topSignalFamily ?? null,
    topSignalSide: b.topSignalSide ?? 0,
    topSignalConfidence: b.topSignalConfidence ?? null,
    avgSignalConfidence: b.avgSignalConfidence ?? null,
    metaProbability: b.metaProbability ?? null,
    calibratedProbability: b.calibratedProbability ?? null,
    regimeFitScore: b.regimeFitScore ?? null,
    betSize: b.betSize ?? null,
    sizingConstraints: b.sizingConstraintsApplied ?? [],
    strategy: b.strategy ?? null,
    reason: b.reason ?? '',
    expectedCostBps: b.expectedCostBps ?? null,
    topShap: b.topShapFeature ? mapShap(b.topShapFeature) : null,
    shap: [], // COMING — no persisted SHAP / detail store yet
    trackRecordWinRate: b.trackRecordWinRate ?? null,
    trackRecordN: b.trackRecordN ?? null,
    stageLatency: b.stageLatencySeconds ?? {},
    cascade: [], // COMING — sizing cascade not surfaced yet
    signals: [], // COMING — per-signal rows not persisted yet
  };
}

function mapSym(b: BffMarketRow | BffSymbolView): Sym {
  const spark = b.spark ?? [];
  const line = (b as BffSymbolView).line ?? spark.map((v, i) => ({ t: i, v }));
  const candles = (b as BffSymbolView).candles ?? [];
  return {
    symbol: b.symbol,
    name: b.name,
    type: b.type,
    price: b.price ?? null,
    change1d: b.changeWindowPct ?? null, // the real session/window change
    change1w: null, // COMING — no calendar-anchored bars
    change1m: null, // COMING
    changeYtd: null, // COMING
    spark,
    line,
    candles,
    marketCap: b.marketCap ?? null, // COMING — no source
    volume: b.volume ?? null,
    hasIdea: b.hasIdea ?? false,
    bar: mapBar(b.bar),
  };
}

function mapStrategy(b: BffFamily): Strategy {
  return {
    id: b.id,
    name: b.name,
    category: b.category,
    source: b.source,
    thesis: b.thesis,
    status: b.status,
    sharpe: b.sharpe ?? null,
    winRate: b.winRate ?? null,
    trades: b.trades ?? null,
    contributionPct: b.contributionPct ?? null,
    pnlYtd: b.pnlYtd ?? null,
    allocation: b.allocation ?? null,
    regimeFit: b.regimeFit ?? {},
    params: b.params ?? [],
    equityCurve: b.equityCurve ?? [],
    activeSignals: b.activeSignals ?? 0,
    avgHoldBars: b.avgHoldBars ?? null,
    assetClasses: b.assetClasses ?? [],
  };
}

// ---- empty shapes (valid-empty data for error / data:null) -----------------

const EMPTY_OVERVIEW: OverviewData = {
  actionCounts: { buy: 0, sell: 0, watch: 0, modelRequired: 0, noData: 0 },
  topActionable: [],
  enginePulse: { stages: [], totalSeconds: 0 },
  nav: null,
  navHistory: null,
};
const emptySym = (symbol: string): Sym => ({
  symbol, name: symbol, type: symType(symbol), price: null,
  change1d: null, change1w: null, change1m: null, changeYtd: null,
  spark: [], line: [], candles: [], marketCap: null, volume: null,
  hasIdea: false, bar: null,
});
const emptyStrategy = (id: string): Strategy => ({
  id, name: id, category: '', source: '', thesis: '', status: 'shadow',
  sharpe: null, winRate: null, trades: null, contributionPct: null, pnlYtd: null,
  allocation: null, regimeFit: {}, params: [], equityCurve: [], activeSignals: 0,
  avgHoldBars: null, assetClasses: [],
});

// ---- Overview --------------------------------------------------------------
export interface ActionCounts {
  buy: number;
  sell: number;
  watch: number;
  modelRequired: number;
  noData: number;
}
export interface EnginePulse {
  stages: { stage: string; seconds: number }[];
  totalSeconds: number;
}
export interface OverviewData {
  actionCounts: ActionCounts;
  /** top-N actionable, server pre-sorted by abs(target_weight) */
  topActionable: TradeIdea[];
  enginePulse: EnginePulse;
  // COMING — null today, with unlock copy carried in envelope.warnings
  nav: null;
  navHistory: null;
}

// stage_latency_seconds summed across the top ideas — the one honest system-health
// signal today (per data_readiness.md: stage_latency is PRODUCED).
function aggregatePulse(ideas: TradeIdea[]): EnginePulse {
  const acc: Record<string, number> = {};
  for (const idea of ideas) {
    for (const [stage, sec] of Object.entries(idea.stageLatency)) {
      acc[stage] = (acc[stage] ?? 0) + sec;
    }
  }
  const order = ['data_fetch', 'feature_compute', 'signal_generation', 'meta_inference', 'sizing', 'target_generation'];
  const stages = order
    .filter((s) => acc[s] != null)
    .map((stage) => ({ stage, seconds: Math.round(acc[stage] * 1000) / 1000 }));
  const totalSeconds = Math.round(stages.reduce((a, s) => a + s.seconds, 0) * 1000) / 1000;
  return { stages, totalSeconds };
}

interface BffOverview {
  actionCounts: ActionCounts;
  topActionable: BffIdea[];
  enginePulse: EnginePulse;
  nav: null;
  navHistory: null;
}

export async function getOverview(): Promise<ApiEnvelope<OverviewData>> {
  if (USE_BFF) {
    return live<BffOverview, OverviewData>(
      '/overview',
      (b) => ({
        actionCounts: b.actionCounts,
        topActionable: (b.topActionable ?? []).map(mapIdea),
        enginePulse: b.enginePulse ?? { stages: [], totalSeconds: 0 },
        nav: null,
        navHistory: null,
      }),
      EMPTY_OVERVIEW,
    );
  }
  const actionable = TRADE_IDEAS.filter((i) => i.action === 'BUY' || i.action === 'SELL');
  const topActionable = [...actionable]
    .sort((a, b) => Math.abs(b.targetWeight) - Math.abs(a.targetWeight))
    .slice(0, 5);
  const data: OverviewData = {
    actionCounts: {
      buy: TRADE_IDEAS.filter((i) => i.action === 'BUY').length,
      sell: TRADE_IDEAS.filter((i) => i.action === 'SELL').length,
      watch: TRADE_IDEAS.filter((i) => i.action === 'WATCH').length,
      modelRequired: TRADE_IDEAS.filter((i) => i.action === 'MODEL_REQUIRED').length,
      noData: TRADE_IDEAS.filter((i) => i.action === 'NO_DATA').length,
    },
    topActionable,
    enginePulse: aggregatePulse(TRADE_IDEAS),
    nav: null,
    navHistory: null,
  };
  return envelope(data, {
    source: 'GET /api/v1/overview',
    warnings: [
      'portfolio metrics (nav/pnl/drawdown/exposure) unavailable: no persisted portfolio',
    ],
  });
}

// ---- Markets ---------------------------------------------------------------
export async function getMarkets(): Promise<ApiEnvelope<Sym[]>> {
  if (USE_BFF) {
    return live<BffMarketRow[], Sym[]>('/markets', (rows) => rows.map(mapSym), []);
  }
  return envelope(SYMBOLS, { source: 'GET /api/v1/markets' });
}

// ---- Trade ideas -----------------------------------------------------------
interface BffTradeIdeas { ideaCount: number; totals: unknown; ideas: BffIdea[]; }

export async function getTradeIdeas(): Promise<ApiEnvelope<TradeIdea[]>> {
  if (USE_BFF) {
    // The BFF returns { ideaCount, totals, ideas }; the screens bind the array.
    return live<BffTradeIdeas, TradeIdea[]>('/trade-ideas', (b) => (b.ideas ?? []).map(mapIdea), []);
  }
  return envelope(TRADE_IDEAS, { source: 'GET /api/v1/trade-ideas' });
}

export async function getIdea(symbol: string): Promise<ApiEnvelope<TradeIdea | null>> {
  if (USE_BFF) {
    return live<BffIdea | null, TradeIdea | null>(
      `/trade-ideas/${encodeURIComponent(symbol)}`,
      (b) => (b ? mapIdea(b) : null),
      null,
    );
  }
  const idea = TRADE_IDEAS.find((i) => i.symbol === symbol) ?? null;
  return envelope(idea, { source: `GET /api/v1/trade-ideas/${symbol}` });
}

// ---- Symbol ----------------------------------------------------------------
export interface SymbolDetail {
  sym: Sym;
  idea: TradeIdea | null;
}
interface BffSymbolDetail { sym: BffSymbolView; idea: BffIdea | null; }

export async function getSymbol(symbol: string): Promise<ApiEnvelope<SymbolDetail>> {
  if (USE_BFF) {
    return live<BffSymbolDetail, SymbolDetail>(
      `/symbols/${encodeURIComponent(symbol)}`,
      (b) => ({ sym: mapSym(b.sym), idea: b.idea ? mapIdea(b.idea) : null }),
      { sym: emptySym(symbol), idea: null },
    );
  }
  const sym = symBy(symbol);
  const idea = TRADE_IDEAS.find((i) => i.symbol === symbol) ?? null;
  return envelope({ sym, idea }, { source: `GET /api/v1/symbols/${symbol}` });
}

// ---- Strategies ------------------------------------------------------------
export async function getStrategies(): Promise<ApiEnvelope<Strategy[]>> {
  if (USE_BFF) {
    return live<BffFamily[], Strategy[]>('/signals/families', (rows) => rows.map(mapStrategy), []);
  }
  return envelope(STRATEGIES, { source: 'GET /api/v1/signals/families' });
}

interface BffFamilyDetail { strategy: BffFamily; ideas: BffIdea[]; }

export async function getStrategy(
  id: string,
): Promise<ApiEnvelope<{ strategy: Strategy; ideas: TradeIdea[] }>> {
  if (USE_BFF) {
    return live<BffFamilyDetail, { strategy: Strategy; ideas: TradeIdea[] }>(
      `/signals/family-${encodeURIComponent(id)}`,
      (b) => ({ strategy: mapStrategy(b.strategy), ideas: (b.ideas ?? []).map(mapIdea) }),
      { strategy: emptyStrategy(id), ideas: [] },
    );
  }
  const strategy = stratBy(id);
  const ideas = TRADE_IDEAS.filter((i) => i.strategy === id);
  return envelope({ strategy, ideas }, { source: `GET /api/v1/signals/family-${id}` });
}

// ---- Model & features ------------------------------------------------------
type ModelData = typeof MODEL;

interface BffModel {
  version?: string; trainedAt?: string; lastRetrainHours?: number; runId?: string;
  type?: string; cvScore?: number; trainAcc?: number; trainingEvents?: number;
  gates?: { cpcv: boolean | null; dsr: boolean | null; pbo: boolean | null };
  retrainTimeline?: unknown[]; metaProbHist?: { bucket: string; count: number }[];
  calibration?: unknown[]; featureImportance?: unknown[]; drift?: unknown[]; rlShadow?: unknown;
}

export async function getModel(): Promise<ApiEnvelope<ModelData>> {
  if (USE_BFF) {
    // Map field-for-field. Crucially, the COMING fields (auc/brier/ece,
    // calibration, featureImportance, drift, rlShadow, retrainTimeline) are
    // forced null/[] — NOT merged from the mock — so a missing field can never
    // surface a fabricated value. The Model screen renders these as Coming.
    return live<BffModel, ModelData>(
      '/model',
      (b) =>
        ({
          version: b.version ?? null,
          trainedAt: b.trainedAt ?? null,
          lastRetrainHours: b.lastRetrainHours ?? null,
          runId: b.runId ?? null,
          type: b.type ?? null,
          cvScore: b.cvScore ?? null,
          trainAcc: b.trainAcc ?? null,
          trainingEvents: b.trainingEvents ?? null,
          gates: b.gates ?? { cpcv: null, dsr: null, pbo: null },
          auc: null, // not logged by the engine
          brier: null,
          ece: null,
          calibration: b.calibration ?? [],
          featureImportance: b.featureImportance ?? [],
          drift: b.drift ?? [],
          metaProbHist: b.metaProbHist ?? [],
          rlShadow: b.rlShadow ?? null,
          retrainTimeline: b.retrainTimeline ?? [],
        }) as unknown as ModelData,
      MODEL,
    );
  }
  return envelope(MODEL, { source: 'GET /api/v1/model' });
}
