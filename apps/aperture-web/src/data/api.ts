// ---------------------------------------------------------------------------
// Data access layer. Mock-first today; every accessor returns an ApiEnvelope so
// the screens bind to the same trust contract the FastAPI BFF (src/web) ships.
// Swapping to live data = replacing the bodies here with fetch() calls — the
// screens don't change. v1 honesty (what's null/coming) is encoded HERE, once.
// ---------------------------------------------------------------------------
import {
  MODEL,
  STRATEGIES,
  SYMBOLS,
  Strategy,
  Sym,
  TRADE_IDEAS,
  TradeIdea,
  stratBy,
  symBy,
} from './mock';
import { ApiEnvelope, envelope } from './envelope';

export * from './mock';
export type { ApiEnvelope } from './envelope';

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

export function getOverview(): ApiEnvelope<OverviewData> {
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
export function getMarkets(): ApiEnvelope<Sym[]> {
  return envelope(SYMBOLS, { source: 'GET /api/v1/markets' });
}

// ---- Trade ideas -----------------------------------------------------------
export function getTradeIdeas(): ApiEnvelope<TradeIdea[]> {
  return envelope(TRADE_IDEAS, { source: 'GET /api/v1/trade-ideas' });
}
export function getIdea(symbol: string): ApiEnvelope<TradeIdea | null> {
  const idea = TRADE_IDEAS.find((i) => i.symbol === symbol) ?? null;
  return envelope(idea, { source: `GET /api/v1/trade-ideas/${symbol}` });
}

// ---- Symbol ----------------------------------------------------------------
export interface SymbolDetail {
  sym: Sym;
  idea: TradeIdea | null;
}
export function getSymbol(symbol: string): ApiEnvelope<SymbolDetail> {
  const sym = symBy(symbol);
  const idea = TRADE_IDEAS.find((i) => i.symbol === symbol) ?? null;
  return envelope({ sym, idea }, { source: `GET /api/v1/symbols/${symbol}` });
}

// ---- Strategies ------------------------------------------------------------
export function getStrategies(): ApiEnvelope<Strategy[]> {
  return envelope(STRATEGIES, { source: 'GET /api/v1/signals/families' });
}
export function getStrategy(id: string): ApiEnvelope<{ strategy: Strategy; ideas: TradeIdea[] }> {
  const strategy = stratBy(id);
  const ideas = TRADE_IDEAS.filter((i) => i.strategy === id);
  return envelope({ strategy, ideas }, { source: `GET /api/v1/signals/family-${id}` });
}

// ---- Model & features ------------------------------------------------------
export function getModel(): ApiEnvelope<typeof MODEL> {
  return envelope(MODEL, { source: 'GET /api/v1/model' });
}
