'use client';
import { Suspense, useCallback, useMemo } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { Panel } from '@/components/ui/Panel';
import { AssetGlyph, ActionPill, ProbRing, Segmented } from '@/components/ui/primitives';
import { DataUnavailable } from '@/components/ui/honesty';
import { Icon } from '@/components/Icon';
import { IdeaDrawer } from '@/components/ideas/IdeaDrawer';
import { getTradeIdeas } from '@/data/api';
import { TradeIdea } from '@/data/mock';
import { deriveTrust } from '@/data/envelope';
import { useDensity } from '@/lib/density';
import { useChartColors } from '@/lib/theme';
import { fmtCompact, fmtPctSigned, fmtProb, fmtTimeAgo } from '@/lib/format';

type Filter = 'all' | 'BUY' | 'SELL' | 'WATCH' | 'MODEL_REQUIRED';
const FILTERS: { value: Filter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'BUY', label: 'Buy' },
  { value: 'SELL', label: 'Sell' },
  { value: 'WATCH', label: 'Watch' },
  { value: 'MODEL_REQUIRED', label: 'Model?' },
];

const UNLOCK = {
  meta: 'Meta probability — load an MLflow production model. Currently MODEL_REQUIRED.',
  calibrated: 'Calibrated probability — load an MLflow production model. Currently MODEL_REQUIRED.',
  regimeFit:
    'Regime fit — coming when the regime detector is wired into the live cycle. RegimeDetector has zero runtime callers.',
  cost: 'Pre-trade cost — coming when a cost service is wired.',
} as const;

function Tile({ label, value, sub, color }: { label: string; value: string; sub: string; color?: string }) {
  return (
    <div className="card" style={{ padding: '14px 16px' }}>
      <div className="eyebrow">{label}</div>
      <div className="num" style={{ fontSize: 24, fontWeight: 700, marginTop: 5, letterSpacing: '-0.02em', color }}>{value}</div>
      <div className="muted" style={{ fontSize: 11.5, marginTop: 2 }}>{sub}</div>
    </div>
  );
}

const FILTER_VALUES: Filter[] = ['all', 'BUY', 'SELL', 'WATCH', 'MODEL_REQUIRED'];

function IdeasInner() {
  const C = useChartColors();
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const env = getTradeIdeas();
  const all = env.data;
  const trust = deriveTrust(env);
  const { density } = useDensity();
  // Pro density reveals the ABSENT coming columns (Regime fit, Cost) + diagnostic
  // raw counts. Comfort hides them entirely (the "ABSENT columns hidden, not
  // blank" contract) — it never turns a coming slot into a number.
  const pro = density === 'pro';

  // The URL is the source of truth: filter + open drawer are both shareable and
  // back/forward-button friendly. (Overview deep-links here with ?idea=SYMBOL.)
  const rawFilter = params.get('filter') as Filter | null;
  const filter: Filter = rawFilter && FILTER_VALUES.includes(rawFilter) ? rawFilter : 'all';
  const selected = useMemo(() => all.find((i) => i.symbol === params.get('idea')) ?? null, [all, params]);

  const setParam = useCallback(
    (key: string, value: string | null) => {
      const next = new URLSearchParams(params.toString());
      if (value == null || value === 'all') next.delete(key);
      else next.set(key, value);
      const qs = next.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [params, pathname, router],
  );
  const setFilter = useCallback((f: Filter) => setParam('filter', f), [setParam]);
  const openIdea = useCallback((idea: TradeIdea) => setParam('idea', idea.symbol), [setParam]);
  const closeIdea = useCallback(() => setParam('idea', null), [setParam]);

  const rows = useMemo(() => (filter === 'all' ? all : all.filter((i) => i.action === filter)), [all, filter]);

  const totals = useMemo(() => {
    const buy = all.filter((i) => i.action === 'BUY').length;
    const sell = all.filter((i) => i.action === 'SELL').length;
    const watch = all.filter((i) => i.action === 'WATCH').length;
    const needsModel = all.filter((i) => i.action === 'MODEL_REQUIRED').length;
    return { buy, sell, watch, needsModel };
  }, [all]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Summary tiles */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
        <Tile label="Buy" value={String(totals.buy)} sub="long candidates" color={totals.buy ? C.buy : C.text3} />
        <Tile label="Sell" value={String(totals.sell)} sub="short candidates" color={totals.sell ? C.sell : C.text3} />
        <Tile label="Watch" value={String(totals.watch)} sub="below entry gate" color={totals.watch ? C.watch : C.text3} />
        <Tile
          label="Model?"
          value={String(totals.needsModel)}
          sub="awaiting a model"
          color={totals.needsModel ? C.warn : C.text3}
        />
        <Tile
          label="Freshness"
          value={fmtTimeAgo(trust.stalenessSeconds)}
          sub={trust.stale ? 'stale snapshot' : 'snapshot age'}
          color={trust.stale ? C.warn : undefined}
        />
      </div>

      {/* Filter tabs */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <Segmented options={FILTERS} value={filter} onChange={setFilter} />
        <div className="muted" style={{ fontSize: 13 }}>
          <span className="num">{rows.length}</span> {rows.length === 1 ? 'idea' : 'ideas'}
          {totals.needsModel > 0 && <> · <span className="num">{totals.needsModel}</span> need a model</>}
        </div>
      </div>

      {/* Dense ideas table */}
      <Panel pad={0}>
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ paddingLeft: 18 }}>Symbol</th>
              <th style={{ textAlign: 'left' }}>Action</th>
              <th style={{ textAlign: 'left' }}>Strategy</th>
              <th>Top conf</th>
              <th>Meta p</th>
              <th>Cal p</th>
              {pro && <th title={UNLOCK.regimeFit}>Regime fit</th>}
              <th>Target wt</th>
              <th>Notional</th>
              {pro && <th title={UNLOCK.cost}>Cost</th>}
              {pro && <th title="bars loaded for this idea (diagnostic)">Bars</th>}
              {pro && <th title="feature rows computed (diagnostic)">Features</th>}
              <th style={{ width: 28, paddingRight: 14 }} />
            </tr>
          </thead>
          <tbody>
            {rows.map((idea) => (
              <tr
                key={idea.symbol}
                className="clickable"
                role="button"
                tabIndex={0}
                aria-label={`Open ${idea.symbol} decision chain`}
                onClick={() => openIdea(idea)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openIdea(idea);
                  }
                }}
              >
                <td style={{ paddingLeft: 18 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <AssetGlyph sym={idea} size={32} />
                    <div>
                      <div className="mono" style={{ fontWeight: 650, fontSize: 13.5 }}>{idea.symbol}</div>
                      <div className="dim" style={{ fontSize: 11.5 }}>{idea.strategy ?? '—'}</div>
                    </div>
                  </div>
                </td>
                <td style={{ textAlign: 'left' }}><ActionPill action={idea.action} /></td>
                <td className="mono muted" style={{ textAlign: 'left', fontSize: 12 }}>{idea.strategy ?? '—'}</td>
                <td className="num">{idea.topSignalConfidence != null ? idea.topSignalConfidence.toFixed(2) : '—'}</td>
                <td className="num">
                  {idea.metaProbability != null ? (
                    fmtProb(idea.metaProbability)
                  ) : (
                    <span style={{ display: 'inline-flex', justifyContent: 'flex-end' }}>
                      <DataUnavailable modelGated unlock={UNLOCK.meta} />
                    </span>
                  )}
                </td>
                <td>
                  {idea.calibratedProbability != null ? (
                    <span style={{ display: 'inline-flex', justifyContent: 'flex-end', width: '100%' }}>
                      <ProbRing value={idea.calibratedProbability} size={34} />
                    </span>
                  ) : (
                    <span style={{ display: 'inline-flex', justifyContent: 'flex-end', width: '100%' }}>
                      <DataUnavailable size={34} modelGated unlock={UNLOCK.calibrated} />
                    </span>
                  )}
                </td>
                {/* Regime fit — ABSENT today; only surfaced in Pro density as a coming chip. */}
                {pro && (
                  <td>
                    <span style={{ display: 'inline-flex', justifyContent: 'flex-end', width: '100%' }}>
                      <DataUnavailable unlock={UNLOCK.regimeFit} />
                    </span>
                  </td>
                )}
                <td className={`num ${idea.targetWeight > 0 ? 'pos' : idea.targetWeight < 0 ? 'neg' : 'dim'}`} style={{ fontWeight: 600 }}>
                  {idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}
                </td>
                <td className="num muted">{idea.targetNotional ? fmtCompact(idea.targetNotional) : '—'}</td>
                {/* Cost — ABSENT today; Pro-only coming chip. */}
                {pro && (
                  <td>
                    <span style={{ display: 'inline-flex', justifyContent: 'flex-end', width: '100%' }}>
                      <DataUnavailable unlock={UNLOCK.cost} />
                    </span>
                  </td>
                )}
                {/* Diagnostic raw counts — Pro density only (real, from the snapshot). */}
                {pro && <td className="num dim">{idea.barsLoaded}</td>}
                {pro && <td className="num dim">{idea.featureRows}</td>}
                <td style={{ paddingRight: 14, color: C.text3 }}>
                  <span style={{ display: 'inline-flex' }}><Icon name="chevronRight" size={15} /></span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <div className="dim" style={{ padding: '28px 0', textAlign: 'center', fontSize: 13 }}>No ideas match this filter.</div>
        )}
      </Panel>

      {selected && <IdeaDrawer idea={selected} onClose={closeIdea} />}
    </div>
  );
}

export default function IdeasPage() {
  return (
    <Suspense fallback={<div className="dim" style={{ padding: 40 }}>Loading ideas…</div>}>
      <IdeasInner />
    </Suspense>
  );
}
