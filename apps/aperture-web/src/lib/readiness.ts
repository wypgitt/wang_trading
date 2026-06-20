// ---------------------------------------------------------------------------
// Screen readiness — the machine-readable twin of docs/data_readiness.md.
//
// A screen is LIVE (real engine data today) or COMING (visible, dignified, with
// its exact unlock condition shown on open). When an engine wave lands, flip one
// entry here — never edit nav code. This is what makes "huge but simple" honest.
// ---------------------------------------------------------------------------

export type Readiness = 'live' | 'coming';
export type LockKind = 'gated' | 'wireable' | 'deferred';

export interface ScreenSpec {
  id: string;
  label: string;
  href: string;
  icon: string;
  readiness: Readiness;
  /** model-gated: real today only when an MLflow production model is loaded */
  modelGated?: boolean;
  lock?: LockKind;
  /** one-line purpose, lifted from the v2 page spec */
  purpose?: string;
  /** verbatim unlock condition from data_readiness.md */
  unlock?: string;
  /** which data_readiness.md wave unlocks it */
  wave?: number;
}

export interface NavGroup {
  group: string;
  items: ScreenSpec[];
}

export const NAV: NavGroup[] = [
  {
    group: 'Cockpit',
    items: [
      { id: 'overview', label: 'Overview', href: '/overview', icon: 'overview', readiness: 'live', purpose: '“What now” in one glance' },
      { id: 'markets', label: 'Markets', href: '/markets', icon: 'markets', readiness: 'live', purpose: 'The universe, real bars' },
      { id: 'ideas', label: 'Trade Ideas', href: '/ideas', icon: 'ideas', readiness: 'live', purpose: 'Decision table + decision drawer' },
      { id: 'strategies', label: 'Strategies', href: '/strategies', icon: 'strategies', readiness: 'live', purpose: 'Live ideas per family' },
    ],
  },
  {
    group: 'Engine',
    items: [
      { id: 'model', label: 'Model & Features', href: '/model', icon: 'model', readiness: 'live', modelGated: true, purpose: 'Meta-prob histogram + retrain timeline' },
    ],
  },
  {
    group: 'Portfolio',
    items: [
      {
        id: 'portfolio', label: 'Portfolio & Risk', href: '/portfolio', icon: 'portfolio', readiness: 'coming', lock: 'gated', wave: 5,
        purpose: 'Positions, exposure, factor risk, drawdown',
        unlock: 'Unlocks when the engine persists positions + a NAV series. Today no orders are routed (live_orders_sent=0) and src/portfolio has zero production callers.',
      },
      {
        id: 'execution', label: 'Execution & TCA', href: '/execution', icon: 'research', readiness: 'coming', lock: 'gated', wave: 5,
        purpose: 'Orders, fills, transaction-cost analysis',
        unlock: 'Unlocks when the order-routing path writes ExecutionStorage (orders/fills/TCA). The deployed path stops before routing.',
      },
    ],
  },
  {
    group: 'Research',
    items: [
      {
        id: 'backtests', label: 'Backtests', href: '/backtests', icon: 'research', readiness: 'coming', lock: 'gated', wave: 5,
        purpose: 'Walk-forward equity, the 3 promotion gates, trade log',
        unlock: 'Unlocks when backtest runs are persisted and the retrain gate is fixed (retrain_pipeline.py:265 falls through to gate_unavailable).',
      },
      {
        id: 'scenarios', label: 'Scenarios & Stress', href: '/scenarios', icon: 'layers', readiness: 'coming', lock: 'gated', wave: 6,
        purpose: 'Factor shocks, stress paths, what-if',
        unlock: 'Unlocks when ScenarioService calls the real factor_risk engine instead of returning mock numbers.',
      },
      {
        id: 'track-record', label: 'Track Record', href: '/track-record', icon: 'research', readiness: 'coming', lock: 'gated', wave: 6,
        purpose: 'Realized calls, hit rate, attribution over time',
        unlock: 'Unlocks when a call-history store exists (the trade-ideas snapshot is overwritten each publish).',
      },
    ],
  },
  {
    group: 'Operate',
    items: [
      {
        id: 'monitoring', label: 'Monitoring & Alerts', href: '/monitoring', icon: 'bell', readiness: 'coming', lock: 'gated', wave: 6,
        purpose: 'Freshness heatmap, breaker state, alert feed',
        unlock: 'Unlocks when pipeline metrics are scraped over HTTP (the registry is currently unscraped; /metrics serves only bff_* self-metrics).',
      },
      {
        id: 'preflight', label: 'Preflight & Go-Live', href: '/preflight', icon: 'shield', readiness: 'coming', lock: 'wireable', wave: 3,
        purpose: 'Live blocker checks + infrastructure probes',
        unlock: 'Wireable now — point PreflightService at the real PreflightChecker / InfrastructureProbe. Runnable engine code exists; only a BFF stub to rewire. Fast-follow, not v1.',
      },
      {
        id: 'replay', label: 'Replay / Time Travel', href: '/replay', icon: 'refresh', readiness: 'coming', lock: 'deferred', wave: 6,
        purpose: 'Reconstruct any past cycle from the audit chain',
        unlock: 'Unlocks when the audit chain is written (ComplianceAuditLogger is never instantiated).',
      },
    ],
  },
];

const ALL = NAV.flatMap((g) => g.items);
export const screenById = (id: string): ScreenSpec | undefined => ALL.find((s) => s.id === id);
export const screenByHref = (href: string): ScreenSpec | undefined => ALL.find((s) => s.href === href);

// ---------------------------------------------------------------------------
// Strategy family readiness — the machine-readable twin of the data_readiness.md
// family table (aperture_v1_design.md §"The active / dormant truth"). Status is
// derived from each generator's dispatch `kind` in src/signal_battery/orchestrator.py:
// the deployed cycle supplies only single-symbol `bars`, so families needing extra
// context never fire. Flip one entry when a context feed is wired — never edit a screen.
// ---------------------------------------------------------------------------
export interface FamilyReadiness {
  active: boolean;
  kind: string;
  /** why dormant (for inactive families) */
  reason?: string;
}
export const FAMILY_READINESS: Record<string, FamilyReadiness> = {
  ts_momentum: { active: true, kind: 'bars' },
  mean_reversion: { active: true, kind: 'bars' },
  ma_crossover: { active: true, kind: 'bars' },
  donchian_breakout: { active: true, kind: 'bars' },
  cs_momentum: { active: false, kind: 'panel', reason: 'needs a multi_asset_prices panel feed' },
  stat_arb: { active: false, kind: 'pair', reason: 'needs a cointegrated stat_arb_pair' },
  futures_carry: { active: false, kind: 'bars_extra', reason: 'needs futures_curve context' },
  vrp: { active: false, kind: 'bars_extra', reason: 'needs vol_features context' },
  funding_rate_arb: { active: false, kind: 'bars_extra', reason: 'needs a funding_rates feed' },
  cross_exchange_arb: { active: false, kind: 'exchange_prices', reason: 'needs multi-venue prices' },
};
export const familyReadiness = (id: string): FamilyReadiness =>
  FAMILY_READINESS[id] ?? { active: false, kind: 'unknown', reason: 'no live feed wired' };
export const familyCounts = () => {
  const vals = Object.values(FAMILY_READINESS);
  return { active: vals.filter((v) => v.active).length, inactive: vals.filter((v) => !v.active).length, total: vals.length };
};

export const LOCK_GLYPH: Record<LockKind, string> = { gated: 'lock', wireable: 'settings', deferred: 'refresh' };
export const LOCK_LABEL: Record<LockKind, string> = {
  gated: 'Gated · needs engine persistence',
  wireable: 'Wireable · BFF stub to rewire',
  deferred: 'Deferred · needs the audit chain',
};
