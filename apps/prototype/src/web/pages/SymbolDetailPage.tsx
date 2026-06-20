import { useState } from 'react';
import { Panel } from '../../components/ui/Panel';
import { AreaChart } from '../../components/charts/AreaChart';
import { CandleChart } from '../../components/charts/CandleChart';
import { IdeaDrawer } from '../../components/IdeaDrawer';
import { ActionPill, AssetGlyph, Delta, ProbRing, Segmented } from '../../components/ui/primitives';
import { ComingSoon, ComingState, DataUnavailable } from '../../components/ui/honesty';
import { Icon } from '../../components/Icon';
import { useNav } from '../../nav';
import { symBy, TRADE_IDEAS } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtCompact, fmtNum, fmtPctSigned, fmtPrice, sideLabel } from '../../lib/format';

const TF: Record<string, number> = { '1M': 22, '3M': 66, '6M': 130 };

// Honesty unlock conditions for engine outputs that are real in code but never
// persisted / surfaced today (see docs/data_readiness.md). The UI shows these
// instead of fabricating a number.
const REGIME_UNLOCK =
  'Regime fit unlocks when RegimeDetector is invoked and persisted — it has zero runtime callers today, so the BFF returns null. Wave 6.';
const COST_UNLOCK =
  'Pre-trade cost unlocks when a cost model is wired — there is no cost service today, so the value is null at the BFF boundary. Wave 5.';
const SHAP_UNLOCK =
  'Top feature contributions unlock when the engine persists shap_importance — TreeSHAP is implemented but never called. Wave 4.';
const FEATURES_UNLOCK =
  'Unlocks when the engine persists computed features (a save_features caller) — Wave 4.';

export function SymbolDetailPage({ symbol }: { symbol: string }) {
  const { go } = useNav();
  const sym = symBy(symbol);
  const idea = TRADE_IDEAS.find((i) => i.symbol === symbol) ?? null;
  const [view, setView] = useState<'candles' | 'area'>('candles');
  const [tf, setTf] = useState('3M');
  const [drawer, setDrawer] = useState(false);

  const cs = sym.candles.slice(-TF[tf]);
  const line = cs.map((c, i) => ({ t: i, v: c.c }));
  const net = line[line.length - 1].v / line[0].v - 1;

  // Real, persisted bar-level columns from the bars hypertable.
  const b = sym.bar;
  const isDollar = b.barType === 'dollar';
  const durSec = Math.round(b.barDurationSeconds);
  const durLabel = durSec >= 60 ? `${Math.floor(durSec / 60)}m ${durSec % 60}s` : `${durSec}s`;
  const micro = [
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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Hero + chart */}
      <div className="card" style={{ padding: 22 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
            <AssetGlyph sym={sym} size={46} />
            <div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                <span className="num" style={{ fontSize: 34, fontWeight: 720, letterSpacing: '-0.03em' }}>{fmtPrice(sym.price)}</span>
                <Delta value={sym.change1d} size={16} />
              </div>
              <div className="muted" style={{ fontSize: 13, marginTop: 2 }}>
                {sym.name} · <span style={{ textTransform: 'capitalize' }}>{sym.type}</span>
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <Segmented value={view} onChange={setView} size="sm" options={[{ value: 'candles', label: 'Candles' }, { value: 'area', label: 'Area' }]} />
            <Segmented value={tf} onChange={setTf} size="sm" options={Object.keys(TF).map((k) => ({ value: k, label: k }))} />
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          {view === 'candles' ? <CandleChart candles={cs} height={340} /> : <AreaChart data={line} height={340} color={net >= 0 ? C.pos : C.neg} showTooltip showYAxis valueFmt={fmtPrice} />}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 0, marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
          {[
            ['1 week', fmtPctSigned(sym.change1w)],
            ['1 month', fmtPctSigned(sym.change1m)],
            ['YTD', fmtPctSigned(sym.changeYtd)],
            ['Mkt cap / Vol', sym.marketCap ? fmtCompact(sym.marketCap) : fmtCompact(sym.volume)],
            ['Bar type', sym.type === 'crypto' ? 'dollar' : 'tib'],
          ].map(([l, v], i) => (
            <div key={l} style={{ paddingLeft: i ? 18 : 0, borderLeft: i ? '1px solid var(--border)' : 'none' }}>
              <div className="eyebrow">{l}</div>
              <div className={`num ${i < 3 ? (String(v).startsWith('-') ? 'neg' : 'pos') : ''}`} style={{ fontSize: 15, fontWeight: 650, marginTop: 4 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Engine view */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 18, alignItems: 'start' }}>
        <Panel
          title="What the engine sees"
          subtitle="The model's current read on this instrument"
          action={idea && <button className="chip" onClick={() => setDrawer(true)}>Full decision chain <Icon name="chevronRight" size={14} /></button>}
        >
          {idea ? (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
                <ActionPill action={idea.action} />
                <span className="muted" style={{ fontSize: 13 }}>via</span>
                <span className="mono" style={{ fontWeight: 600, fontSize: 13 }}>{idea.strategy}</span>
                <span className="muted" style={{ fontSize: 13 }}>· {sideLabel(idea.topSignalSide)}</span>
              </div>
              <p style={{ fontSize: 13.5, lineHeight: 1.55, color: C.text1, marginBottom: 16 }}>{idea.reason}</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
                <ProbRing value={idea.metaProbability} label="Meta" size={58} />
                <ProbRing value={idea.calibratedProbability} label="Calibrated" size={58} />
                <DataUnavailable size={58} label="Regime fit" unlock={REGIME_UNLOCK} />
                <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div>
                    <div className="eyebrow">Target weight</div>
                    <div className={`num ${idea.targetWeight >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 18, fontWeight: 700, marginTop: 3 }}>{idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}</div>
                  </div>
                  <div>
                    <div className="eyebrow">Pre-trade cost</div>
                    <div style={{ marginTop: 6 }}>
                      <DataUnavailable unlock={COST_UNLOCK} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="muted" style={{ fontSize: 13.5, padding: '12px 0' }}>
              No active idea for {symbol} this cycle. The signal battery produced no actionable edge above the entry gate.
            </div>
          )}
        </Panel>

        <Panel title="Why" subtitle="Top model feature contributions">
          <ComingState unlock={SHAP_UNLOCK} />
        </Panel>
      </div>

      {/* Bar microstructure — real, persisted columns from the bars hypertable */}
      <Panel title="Bar microstructure" subtitle={`Persisted ${b.barType}-bar columns from the bars hypertable`}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {micro.map((f) => (
            <div key={f.label} style={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 12, padding: 14 }}>
              <div className="eyebrow">{f.label}</div>
              <div className="num" style={{ fontSize: 19, fontWeight: 680, marginTop: 6 }}>{f.value}</div>
              <div className="dim" style={{ fontSize: 11, marginTop: 3 }}>{f.hint}</div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Live feature grid — gated on a feature-persistence bridge */}
      <ComingSoon
        title="Live feature grid"
        subtitle="Feature Factory · microstructure, volatility & classical signals"
        unlock={FEATURES_UNLOCK}
      />

      {drawer && idea && <IdeaDrawer idea={idea} onClose={() => setDrawer(false)} onOpenSymbol={() => setDrawer(false)} />}
    </div>
  );
}
