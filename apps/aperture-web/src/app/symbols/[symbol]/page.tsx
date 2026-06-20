'use client';
import { useMemo, useState } from 'react';
import Link from 'next/link';
import { Panel } from '@/components/ui/Panel';
import { ActionPill, AssetGlyph, Delta, ProbRing, Segmented } from '@/components/ui/primitives';
import { ComingState, DataUnavailable } from '@/components/ui/honesty';
import { CandleChart } from '@/components/charts/CandleChart';
import { AreaChart } from '@/components/charts/AreaChart';
import { Icon } from '@/components/Icon';
import { getSymbol, SYMBOLS } from '@/data/api';
import { Point } from '@/lib/rng';
import { STALE_THRESHOLD_SECONDS } from '@/lib/colors';
import { useChartColors } from '@/lib/theme';
import { fmtCompact, fmtNum, fmtPctSigned, fmtPrice, sideLabel } from '@/lib/format';

// Timeframe windows over the persisted candle series (last N bars).
const TF: Record<string, number> = { '1M': 22, '3M': 66, '6M': 130 };

// Honesty unlock copy for engine outputs that are real in code but NOT yet
// persisted / surfaced today (see docs/data_readiness.md). The UI shows these
// states instead of fabricating a number.
const SHAP_UNLOCK =
  'Feature contributions — coming when shap_importance is persisted. TreeSHAP is implemented but never called by the live cycle. Wave 4.';
const FEATURES_UNLOCK =
  'Live feature values — coming when the Feature Factory persists features (FeatureStore.save_features has zero callers today). GARCH vol, realized vol, RSI-14, order-flow imbalance, Kyle λ, VPIN, Amihud and Roll spread land here. Wave 4.';
const REGIME_UNLOCK =
  'Regime fit — coming when the regime detector is wired into the live cycle. RegimeDetector has zero runtime callers today, so the BFF returns null. Wave 6.';
const COST_UNLOCK =
  'Pre-trade cost — coming when a cost model is wired. There is no cost service today, so the value is null at the BFF boundary. Wave 5.';
const TRACK_UNLOCK =
  'Track record — coming when realized fills are persisted. live_orders_sent=0 today, so per-strategy win-rate is null at the boundary. Wave 5.';

// Bar-type info affordance copy. Bars are event-driven (tib / vib / dollar /
// volume / tick / tick_run / time): each row closes when a sampling threshold is
// hit, so bars are unevenly spaced in wall-clock time. The chart x-axis is
// therefore bar-indexed, not a clock — honest for imbalance/volume sampling.
const BAR_TYPE_INFO =
  'Bar sampling scheme (tib / vib / dollar / volume / tick / tick_run / time). Each bar closes when its threshold is met, so bars are unevenly spaced in wall-clock time — the chart x-axis is bar-indexed, not a clock.';

type View = 'candles' | 'area';

// Light per-section error fallback (spec §States "Error — per-element only").
// One section failing never blanks the screen; we show the copy + a copyable
// request id and leave the rest of the page live. Inert with mock today.
function SectionError({ message, requestId }: { message: string; requestId: string }) {
  const C = useChartColors();
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        padding: '14px 16px',
        border: `1px dashed ${C.negDim}66`,
        borderRadius: 12,
        background: `${C.neg}0d`,
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 600, color: C.text1 }}>{message}</div>
      <div className="dim mono" style={{ fontSize: 11 }}>req:{requestId}</div>
    </div>
  );
}

export default function SymbolDetailPage({ params }: { params: { symbol: string } }) {
  const C = useChartColors();
  const symbol = decodeURIComponent(params.symbol);

  // Honest not-found: symBy falls back to SYMBOLS[0]; detect a genuine miss so
  // we render a calm "not found" state instead of silently showing another row.
  // (Hooks below run unconditionally on the fallback Sym; we gate only the JSX.)
  const notFound = !SYMBOLS.some((s) => s.symbol === symbol);

  const env = getSymbol(symbol);
  const { sym, idea } = env.data;
  const stale = env.staleness_seconds > STALE_THRESHOLD_SECONDS;
  const symErr = env.errors.find((e) => e.field == null || e.field === 'symbol' || e.field === 'bars');
  const ideaErr = env.errors.find((e) => e.field === 'idea' || e.field === 'trade_idea');

  const [view, setView] = useState<View>('candles');
  const [tf, setTf] = useState('3M');

  // Timeframe slices the persisted candle series; the area line is derived from
  // the same window so both views agree. (sym.line is the full short series.)
  const candles = useMemo(() => sym.candles.slice(-TF[tf]), [sym.candles, tf]);
  const line: Point[] = useMemo(() => candles.map((c, i) => ({ t: i, v: c.c })), [candles]);
  const net = line.length ? line[line.length - 1].v / line[0].v - 1 : 0;

  // Real, persisted bar-level columns from the `bars` hypertable [LIVE].
  const b = sym.bar;
  const isDollar = b.barType === 'dollar';
  const durSec = Math.round(b.barDurationSeconds);
  const durLabel = durSec >= 60 ? `${Math.floor(durSec / 60)}m ${durSec % 60}s` : `${durSec}s`;
  const micro: { label: string; value: string; hint: string }[] = [
    { label: 'VWAP', value: fmtPrice(b.vwap), hint: 'volume-weighted' },
    { label: 'Dollar volume', value: fmtCompact(b.dollarVolume), hint: 'Σ price × size' },
    { label: 'Tick count', value: fmtNum(b.tickCount, 0), hint: 'trades in bar' },
    { label: 'Buy / sell vol', value: `${fmtCompact(b.buyVolume, '')} / ${fmtCompact(b.sellVolume, '')}`, hint: 'classified flow' },
    { label: 'Volume imbalance', value: fmtCompact(b.volumeImbalance, ''), hint: 'buy − sell' },
    { label: 'Tick imbalance', value: fmtPctSigned(b.tickImbalanceRatio, 1), hint: '(buy − sell ticks) / n' },
    { label: 'Imbalance', value: fmtNum(b.imbalance, 0), hint: 'cumulative signed' },
    { label: 'Threshold', value: isDollar ? fmtCompact(b.threshold) : fmtNum(b.threshold, 0), hint: isDollar ? '$ volume to close bar' : 'imbalance to close bar' },
    { label: 'Bar duration', value: durLabel, hint: 'open → close' },
    { label: 'Bar type', value: b.barType, hint: 'sampling scheme' },
  ];

  if (notFound) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <Link href="/markets" className="dim" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 13, alignSelf: 'flex-start' }}>
          <Icon name="arrowLeft" size={15} /> Markets
        </Link>
        <div className="card" style={{ padding: 38, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <span style={{ display: 'grid', placeItems: 'center', width: 38, height: 38, borderRadius: 11, background: `${C.text3}1f`, color: C.text2 }}>
            <Icon name="search" size={18} />
          </span>
          <div style={{ fontSize: 15, fontWeight: 650, color: C.text1 }}>Symbol not found</div>
          <div className="muted" style={{ fontSize: 13, maxWidth: 380, lineHeight: 1.55 }}>
            No instrument <span className="mono">{symbol}</span> in the universe this cycle. It may not be tracked, or the ticker may be mistyped.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        ...(stale ? { borderTop: `2px solid ${C.warn}`, paddingTop: 14 } : {}),
      }}
    >
      {/* Back link */}
      <Link href="/markets" className="dim" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 13, alignSelf: 'flex-start' }}>
        <Icon name="arrowLeft" size={15} /> Markets
      </Link>

      {/* 1 + 2 — price hero + chart */}
      <div className="card" style={{ padding: 22 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
            <AssetGlyph sym={sym} size={46} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 9, flexWrap: 'wrap' }}>
                <span className="mono" style={{ fontWeight: 700, fontSize: 17, letterSpacing: '-0.01em' }}>{sym.symbol}</span>
                <span className="chip" style={{ height: 22, fontSize: 11, textTransform: 'capitalize' }}>{sym.type}</span>
                <span className="chip" style={{ height: 22, fontSize: 11 }}>
                  <span className="dim">bar</span>&nbsp;<span className="mono">{b.barType}</span>
                </span>
                {stale && (
                  <span
                    className="chip"
                    title={`Snapshot is ${Math.round(env.staleness_seconds)}s old (> ${STALE_THRESHOLD_SECONDS}s). Showing last-good values while the engine refreshes.`}
                    style={{ height: 22, fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 5, color: C.warn, borderColor: `${C.warn}55` }}
                  >
                    <Icon name="refresh" size={11} /> Stale · refreshing
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginTop: 8 }}>
                <span className="num" style={{ fontSize: 34, fontWeight: 720, letterSpacing: '-0.03em' }}>{fmtPrice(sym.price)}</span>
                <Delta value={sym.change1d} size={16} />
              </div>
              <div className="muted" style={{ fontSize: 13, marginTop: 2 }}>{sym.name}</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <Segmented<View> value={view} onChange={setView} size="sm" options={[{ value: 'candles', label: 'Candles' }, { value: 'area', label: 'Area' }]} />
            <Segmented value={tf} onChange={setTf} size="sm" options={Object.keys(TF).map((k) => ({ value: k, label: k }))} />
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          {symErr ? (
            <SectionError message="Couldn't load bars. Retry." requestId={env.request_id} />
          ) : view === 'candles' ? (
            <CandleChart candles={candles} height={340} />
          ) : (
            <AreaChart data={line} height={340} color={net >= 0 ? C.pos : C.neg} showTooltip showYAxis valueFmt={fmtPrice} />
          )}
        </div>

        {/* Change strip */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
          {([
            ['1 week', <Delta key="1w" value={sym.change1w} dp={2} arrow={false} size={15} />],
            ['1 month', <Delta key="1m" value={sym.change1m} dp={2} arrow={false} size={15} />],
            ['YTD', <Delta key="ytd" value={sym.changeYtd} dp={2} arrow={false} size={15} />],
            [
              'Bar type',
              <span key="bartype" className="mono" style={{ fontSize: 15, fontWeight: 650, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                {sym.bar.barType}
                <span
                  title={BAR_TYPE_INFO}
                  aria-label={BAR_TYPE_INFO}
                  style={{
                    cursor: 'help',
                    display: 'grid',
                    placeItems: 'center',
                    width: 15,
                    height: 15,
                    borderRadius: '50%',
                    border: `1px solid ${C.borderStrong}`,
                    color: C.text3,
                    fontSize: 10,
                    fontWeight: 700,
                    fontFamily: 'serif',
                    fontStyle: 'italic',
                    lineHeight: 1,
                  }}
                >
                  i
                </span>
              </span>,
            ],
            ['Market cap', <span key="cap" className="num" style={{ fontSize: 15, fontWeight: 650 }}>{sym.marketCap ? fmtCompact(sym.marketCap) : '—'}</span>],
            ['Volume', <span key="vol" className="num" style={{ fontSize: 15, fontWeight: 650 }}>{fmtCompact(sym.volume)}</span>],
          ] as const).map(([label, node], i) => (
            <div key={label} style={{ paddingLeft: i ? 18 : 0, borderLeft: i ? '1px solid var(--border)' : 'none' }}>
              <div className="eyebrow">{label}</div>
              <div style={{ marginTop: 4 }}>{node}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 3 — what the engine sees (idea) + why (SHAP) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.5fr) minmax(0,1fr)', gap: 16, alignItems: 'start' }}>
        <Panel
          title="What the engine sees"
          subtitle="The model's current read on this instrument"
          action={
            idea ? (
              <Link
                href={`/ideas?idea=${sym.symbol}`}
                className="dim"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12.5, whiteSpace: 'nowrap', flexShrink: 0 }}
              >
                Full decision chain ›
              </Link>
            ) : undefined
          }
        >
          {ideaErr ? (
            <SectionError message="Couldn't load the engine read. Retry." requestId={env.request_id} />
          ) : idea ? (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
                <ActionPill action={idea.action} />
                <span className="muted" style={{ fontSize: 13 }}>via</span>
                <span className="mono" style={{ fontWeight: 600, fontSize: 13 }}>{idea.strategy ?? '—'}</span>
                <span className="muted" style={{ fontSize: 13 }}>· {sideLabel(idea.topSignalSide)}</span>
              </div>
              <p style={{ fontSize: 13.5, lineHeight: 1.55, color: C.text1, marginBottom: 18 }}>{idea.reason}</p>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 20, flexWrap: 'wrap' }}>
                {idea.metaProbability != null ? (
                  <ProbRing value={idea.metaProbability} label="Meta p" size={58} />
                ) : (
                  <DataUnavailable size={58} label="Meta p" modelGated unlock="Meta probability — load an MLflow production model. Currently MODEL_REQUIRED." />
                )}
                {idea.calibratedProbability != null ? (
                  <ProbRing value={idea.calibratedProbability} label="Cal p" size={58} />
                ) : (
                  <DataUnavailable size={58} label="Cal p" modelGated unlock="Calibrated probability — load an MLflow production model. Currently MODEL_REQUIRED." />
                )}
                <DataUnavailable size={58} label="Regime fit" unlock={REGIME_UNLOCK} />
                <div style={{ flex: 1, minWidth: 150, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                  <div>
                    <div className="eyebrow">Target weight</div>
                    <div className={`num ${idea.targetWeight > 0 ? 'pos' : idea.targetWeight < 0 ? 'neg' : 'dim'}`} style={{ fontSize: 18, fontWeight: 700, marginTop: 4 }}>
                      {idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="eyebrow">Pre-trade cost</div>
                    <div style={{ marginTop: 6 }}>
                      <DataUnavailable unlock={COST_UNLOCK} />
                    </div>
                  </div>
                  <div>
                    <div className="eyebrow">Track record</div>
                    <div style={{ marginTop: 6 }}>
                      <DataUnavailable unlock={TRACK_UNLOCK} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="muted" style={{ fontSize: 13.5, padding: '14px 0' }}>
              No active idea for {sym.symbol} this cycle. The signal battery produced no actionable edge above the entry gate.
            </div>
          )}
        </Panel>

        <Panel title="Why" subtitle="Top model feature contributions">
          <ComingState title="Feature contributions — coming when shap_importance is persisted" unlock={SHAP_UNLOCK} wave={4} variant="deferred" />
        </Panel>
      </div>

      {/* 4 — bar microstructure [LIVE] — real persisted columns */}
      <Panel title="Bar microstructure" subtitle={`Persisted ${b.barType}-bar columns from the bars hypertable — live`}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12 }}>
          {micro.map((f) => (
            <div key={f.label} style={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 12, padding: 14 }}>
              <div className="eyebrow">{f.label}</div>
              <div className="num" style={{ fontSize: 19, fontWeight: 680, marginTop: 6 }}>{f.value}</div>
              <div className="dim" style={{ fontSize: 11, marginTop: 3 }}>{f.hint}</div>
            </div>
          ))}
        </div>
      </Panel>

      {/* 5 — live features (ephemeral Feature Factory) — coming */}
      <Panel title="Live feature grid" subtitle="Feature Factory · microstructure, volatility & classical signals">
        <ComingState title="Live feature values — coming as the Feature Factory persists" unlock={FEATURES_UNLOCK} wave={4} />
      </Panel>
    </div>
  );
}
