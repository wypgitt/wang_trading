'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Panel } from '@/components/ui/Panel';
import { ActionPill, ProbRing, Delta, Bar } from '@/components/ui/primitives';
import { AssetGlyph } from '@/components/ui/primitives';
import { ComingState } from '@/components/ui/honesty';
import { DataUnavailable } from '@/components/ui/honesty';
import { Icon } from '@/components/Icon';
import { useDensity } from '@/lib/density';
import { getOverview, OverviewData } from '@/data/api';
import { TradeIdea } from '@/data/mock';
import { deriveTrust } from '@/data/envelope';
import { Loaded, ViewProps } from '@/data/useEnvelope';
import { useChartColors } from '@/lib/theme';
import { fmtPctSigned } from '@/lib/format';

// Humanized engine-pulse stage labels (Comfort density). Pro density keeps the
// raw snake_case id; either way the raw id is exposed via a title tooltip.
const STAGE_LABELS: Record<string, string> = {
  data_fetch: 'Data fetch',
  feature_compute: 'Feature compute',
  signal_generation: 'Signal gen',
  meta_inference: 'Meta inference',
  sizing: 'Sizing',
  target_generation: 'Targets',
};

// Human-readable kind label for change chips — never color-only (a11y).
const CHANGE_KIND_LABEL: Record<Change['kind'], string> = {
  new: 'new idea',
  flipped: 'flipped',
  up: 'weight up',
  down: 'weight down',
};

// Client-side snapshot diff (the v1-feasible "show the diff"). Compares the
// current snapshot to the previous one in localStorage, keyed by symbol, scoped
// to fields PRODUCED today (action, target_weight, top_signal_side).
type Change = { symbol: string; kind: 'new' | 'flipped' | 'up' | 'down'; detail: string };
function diffSnapshot(ideas: TradeIdea[]): Change[] {
  if (typeof localStorage === 'undefined') return [];
  const KEY = 'aperture:lastIdeas';
  const prevRaw = localStorage.getItem(KEY);
  const cur = ideas.map((i) => ({ symbol: i.symbol, action: i.action, w: i.targetWeight, side: i.topSignalSide }));
  localStorage.setItem(KEY, JSON.stringify(cur));
  if (!prevRaw) return []; // cold launch — honestly absent
  const prev = JSON.parse(prevRaw) as typeof cur;
  const byPrev = new Map(prev.map((p) => [p.symbol, p]));
  const out: Change[] = [];
  for (const c of cur) {
    const p = byPrev.get(c.symbol);
    if (!p) { out.push({ symbol: c.symbol, kind: 'new', detail: 'new idea' }); continue; }
    if (p.action !== c.action && ((p.action === 'BUY' && c.action === 'SELL') || (p.action === 'SELL' && c.action === 'BUY'))) {
      out.push({ symbol: c.symbol, kind: 'flipped', detail: `${p.action} → ${c.action}` });
    }
    const dw = c.w - p.w;
    if (Math.abs(dw) >= 0.0025) out.push({ symbol: c.symbol, kind: dw > 0 ? 'up' : 'down', detail: `${dw > 0 ? '+' : ''}${(dw * 100).toFixed(0)}bps` });
  }
  return out;
}

export default function OverviewPage() {
  return <Loaded fetcher={getOverview} View={OverviewView} />;
}

function OverviewView({ data, env }: ViewProps<OverviewData>) {
  const C = useChartColors();
  const { density } = useDensity();
  const trust = deriveTrust(env);
  const { actionCounts: ac, topActionable, enginePulse } = data;
  const total = ac.buy + ac.sell + ac.watch;

  // Trust state derived from the envelope. Stale = staleness > 90 (last-good
  // values dim + amber --warn top-border + amber chip on the ideas panel).
  // Mock never trips this, but the structure is real so the live BFF lights it.
  const stale = trust.stale;
  const hasErrors = env.errors.length > 0;

  const [changes, setChanges] = useState<Change[] | null>(null);
  const [dismissed, setDismissed] = useState(false);
  // Persist dismissal keyed to env.as_of so it survives navigation but reappears
  // on a new cycle (a new snapshot = a new as_of = a fresh strip).
  const dismissKey = `aperture:whatChangedDismissed:${env.as_of}`;
  useEffect(() => {
    setChanges(diffSnapshot([...topActionable]));
    if (typeof sessionStorage !== 'undefined') {
      setDismissed(sessionStorage.getItem(dismissKey) === '1');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const dismiss = () => {
    setDismissed(true);
    if (typeof sessionStorage !== 'undefined') sessionStorage.setItem(dismissKey, '1');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* 1 + 4 — decision headline (hero) + engine pulse */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.55fr) minmax(0,1fr)', gap: 16 }}>
        <Panel pad={22}>
          <div className="eyebrow">This cycle</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginTop: 8 }}>
            <span className="num" style={{ fontSize: 40, fontWeight: 750, letterSpacing: '-0.03em' }}>{total}</span>
            <span style={{ fontSize: 17, color: C.text2, fontWeight: 600 }}>actionable {total === 1 ? 'idea' : 'ideas'}</span>
          </div>
          <div style={{ display: 'flex', gap: 26, marginTop: 20 }}>
            <CountStat label="Buy" value={ac.buy} cls="pill-buy" color={C.buy} />
            <CountStat label="Sell" value={ac.sell} cls="pill-sell" color={C.sell} />
            <CountStat label="Watch" value={ac.watch} cls="pill-watch" color={C.watch} />
          </div>
          {(ac.modelRequired > 0 || ac.noData > 0) && (
            <div className="muted" style={{ fontSize: 12.5, marginTop: 16 }}>
              {ac.modelRequired > 0 && <>{ac.modelRequired} need a model</>}
              {ac.modelRequired > 0 && ac.noData > 0 && ' · '}
              {ac.noData > 0 && <>{ac.noData} no data</>}
            </div>
          )}
        </Panel>

        <Panel title="Engine pulse" subtitle="Summed stage latency — the one honest health signal today">
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 12 }}>
            <span className="num" style={{ fontSize: 26, fontWeight: 700 }}>{enginePulse.totalSeconds.toFixed(2)}</span>
            <span className="muted" style={{ fontSize: 13 }}>s total cycle</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {enginePulse.stages.map((s) => (
              <div key={s.stage} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span
                  className="mono"
                  title={s.stage}
                  style={{ width: 132, flex: 'none', fontSize: 11.5, color: C.text3 }}
                >
                  {density === 'pro' ? s.stage : STAGE_LABELS[s.stage] ?? s.stage}
                </span>
                <div style={{ flex: 1 }}><Bar value={s.seconds} max={enginePulse.totalSeconds} color={C.accent2} height={6} /></div>
                {density === 'pro' && <span className="num" style={{ width: 48, textAlign: 'right', fontSize: 11.5, color: C.text2 }}>{s.seconds.toFixed(3)}s</span>}
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* 2 — what-changed strip (client-side diff) */}
      {changes != null && !dismissed && (
        <div
          className="card"
          role="status"
          aria-live="polite"
          aria-label={changes.length ? `What changed: ${changes.length} update${changes.length === 1 ? '' : 's'} since last refresh` : 'What changed: no changes since last refresh'}
          style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 12, borderColor: changes.length ? 'rgba(124,92,255,0.3)' : 'var(--border)' }}
        >
          <span style={{ color: C.accent, display: 'inline-flex' }} aria-hidden><Icon name="refresh" size={15} /></span>
          <span className="eyebrow" style={{ color: C.text2 }}>What changed</span>
          {changes.length === 0 ? (
            <span className="muted" style={{ fontSize: 12.5 }}>No changes since last refresh.</span>
          ) : (
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', flex: 1 }}>
              {changes.slice(0, 8).map((c, i) => (
                <span
                  key={i}
                  className="chip"
                  aria-label={`${c.symbol} ${CHANGE_KIND_LABEL[c.kind]}: ${c.detail}`}
                  style={{ height: 24, fontSize: 11.5 }}
                >
                  <span className="mono">{c.symbol}</span>
                  <span style={{ color: c.kind === 'flipped' ? C.warn : c.kind === 'new' ? C.accent : c.kind === 'up' ? C.pos : C.neg }}>{c.detail}</span>
                </span>
              ))}
            </div>
          )}
          <div style={{ flex: changes.length ? 'none' : 1 }} />
          <button className="chip" style={{ height: 24, fontSize: 11.5 }} onClick={dismiss}>Dismiss</button>
        </div>
      )}

      {/* 3 — top trade ideas (the core gesture) */}
      {/* Stale (staleness>90): --warn top-border + amber chip + dimmed last-good
          values. Error (errors[]): inline retry + copyable req:{id} INSIDE this
          slot only — other panels are never blanked. */}
      <Panel
        title="Top trade ideas"
        subtitle="Highest-conviction actions this cycle — tap for the full decision chain"
        style={stale ? { borderTop: `2px solid ${C.warn}` } : undefined}
        action={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {stale && (
              <span
                className="chip"
                aria-label={`Stale data — feed is ${Math.round(trust.stalenessSeconds)}s old`}
                style={{ height: 28, color: C.warn, borderColor: 'rgba(240,169,59,0.4)', background: 'rgba(240,169,59,0.08)' }}
              >
                STALE · {Math.round(trust.stalenessSeconds)}s
              </span>
            )}
            <Link href="/ideas" className="chip" style={{ height: 28 }}>View all <Icon name="arrowUpRight" size={13} /></Link>
          </div>
        }
      >
        {hasErrors ? (
          <div
            role="alert"
            style={{ padding: '20px 0', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, textAlign: 'center' }}
          >
            <div style={{ color: C.neg, fontSize: 13, fontWeight: 600 }}>Overview trade-ideas aggregation unavailable.</div>
            {env.errors[0]?.message && (
              <div className="muted" style={{ fontSize: 12 }}>{env.errors[0].message}</div>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button className="chip" style={{ height: 28 }} onClick={() => window.location.reload()}>Retry</button>
              <button
                className="chip mono"
                title="Copy request id"
                style={{ height: 28, fontSize: 11.5, color: C.text3 }}
                onClick={() => navigator.clipboard?.writeText(env.request_id)}
              >
                req: {env.request_id}
              </button>
            </div>
          </div>
        ) : topActionable.length === 0 ? (
          <div className="dim" style={{ padding: '28px 0', textAlign: 'center', fontSize: 13 }}>No ideas above the entry gate this cycle.</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', opacity: stale ? 0.55 : 1 }}>
            {topActionable.map((idea, i) => (
              <Link
                key={idea.symbol}
                href={`/ideas?idea=${idea.symbol}`}
                className="row-hover"
                style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 8px', borderTop: i ? '1px solid var(--border)' : 'none', borderRadius: 8 }}
              >
                <AssetGlyph sym={idea} size={34} />
                <div style={{ minWidth: 92 }}>
                  <div className="mono" style={{ fontWeight: 650, fontSize: 14 }}>{idea.symbol}</div>
                  <div className="dim" style={{ fontSize: 11.5 }}>{idea.strategy}</div>
                </div>
                <ActionPill action={idea.action} />
                <div style={{ flex: 1 }} />
                <div style={{ textAlign: 'right', minWidth: 84 }}>
                  <div className="eyebrow">Target wt</div>
                  <div className="num" style={{ fontSize: 14, fontWeight: 600, marginTop: 2, color: stale ? C.text2 : undefined }}>{fmtPctSigned(idea.targetWeight, 1)}</div>
                </div>
                {idea.calibratedProbability != null ? (
                  <ProbRing value={idea.calibratedProbability} size={46} label="Cal p" />
                ) : (
                  <DataUnavailable size={46} label="Cal p" modelGated unlock="Calibrated probability — load an MLflow production model. Currently MODEL_REQUIRED." />
                )}
              </Link>
            ))}
          </div>
        )}
      </Panel>

      {/* 5 + 7 — portfolio value/equity (COMING) + regime (COMING) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.55fr) minmax(0,1fr)', gap: 16 }}>
        <Panel title="Portfolio value" subtitle="NAV + equity curve">
          <ComingState
            title="Portfolio value — coming as the engine lands it"
            unlock="Unlocks when the engine persists positions + a NAV series. Today live_orders_sent=0 and src/portfolio has zero production callers."
            wave={5}
            ghost={<GhostCurve />}
          />
        </Panel>
        <Panel title="Market regime" subtitle="LSTM 4-class detector">
          <ComingState
            title="Regime — coming when wired into the live cycle"
            unlock="Unlocks when the regime detector is invoked + persisted. RegimeDetector has zero runtime callers today."
            wave={6}
            ghost={<GhostBars />}
          />
        </Panel>
      </div>

      {/* 6 — risk & performance band (COMING) */}
      <Panel title="Risk & performance" subtitle="Sharpe · Max DD · Vol · Win · Exposure">
        <ComingState
          title="Risk & performance — coming with persisted portfolio + backtest metrics"
          unlock="Unlocks with the same portfolio gate; backtest metrics are never persisted (retrain gate broken, retrain_pipeline.py:265)."
          wave={5}
          compact
        />
      </Panel>
    </div>
  );
}

function CountStat({ label, value, cls, color }: { label: string; value: number; cls: string; color: string }) {
  const C = useChartColors();
  return (
    <div>
      <span className={`pill ${cls}`} style={{ marginBottom: 8 }}>{label}</span>
      <div className="num" style={{ fontSize: 30, fontWeight: 720, marginTop: 8, color: value ? color : C.text3 }}>{value}</div>
    </div>
  );
}

function GhostCurve() {
  const C = useChartColors();
  return (
    <svg width={420} height={120} viewBox="0 0 420 120" fill="none" aria-hidden>
      <path d="M0 96 L60 80 L120 88 L180 56 L240 64 L300 36 L360 44 L420 18" stroke={C.text2} strokeWidth={2} />
    </svg>
  );
}
function GhostBars() {
  const C = useChartColors();
  return (
    <div style={{ display: 'flex', gap: 6, height: 80, alignItems: 'flex-end' }} aria-hidden>
      {[64, 20, 44, 30].map((h, i) => (
        <div key={i} style={{ width: 26, height: h, borderRadius: 4, background: C.text2 }} />
      ))}
    </div>
  );
}
