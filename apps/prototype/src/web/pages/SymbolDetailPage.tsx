import { useState } from 'react';
import { Panel } from '../../components/ui/Panel';
import { AreaChart } from '../../components/charts/AreaChart';
import { CandleChart } from '../../components/charts/CandleChart';
import { MiniBars } from '../../components/charts/MiniBars';
import { IdeaDrawer } from '../../components/IdeaDrawer';
import { ActionPill, AssetGlyph, Delta, ProbRing, Segmented } from '../../components/ui/primitives';
import { Icon } from '../../components/Icon';
import { useNav } from '../../nav';
import { symBy, TRADE_IDEAS } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtBps, fmtCompact, fmtPct, fmtPctSigned, fmtPrice, sideLabel } from '../../lib/format';

const TF: Record<string, number> = { '1M': 22, '3M': 66, '6M': 130 };

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

function symbolFeatures(symbol: string, closes: number[]) {
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) rets.push(closes[i] / closes[i - 1] - 1);
  const std = (a: number[]) => {
    if (!a.length) return 0;
    const m = a.reduce((x, y) => x + y, 0) / a.length;
    return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
  };
  const rvShort = std(rets.slice(-5)) * Math.sqrt(252);
  const rvLong = std(rets.slice(-30)) * Math.sqrt(252);
  const last14 = rets.slice(-14);
  const gains = last14.filter((r) => r > 0).reduce((a, b) => a + b, 0) / 14;
  const losses = -last14.filter((r) => r < 0).reduce((a, b) => a + b, 0) / 14 || 1e-6;
  const rsi = 100 - 100 / (1 + gains / losses);
  const h = hash(symbol);
  return [
    { label: 'GARCH vol', value: fmtPct(rvLong * 1.08, 1), hint: 'conditional, annualized' },
    { label: 'Realized vol (5)', value: fmtPct(rvShort, 1), hint: 'short window' },
    { label: 'RSI-14', value: rsi.toFixed(0), hint: rsi > 70 ? 'overbought' : rsi < 30 ? 'oversold' : 'neutral' },
    { label: 'Order-flow imb.', value: fmtPctSigned(((h % 22) - 11) / 100), hint: 'signed volume' },
    { label: 'Kyle λ', value: (0.4 + (h % 50) / 100).toFixed(2) + 'e-6', hint: 'price impact' },
    { label: 'VPIN', value: (0.18 + (h % 40) / 100).toFixed(2), hint: 'informed trading' },
    { label: 'Amihud illiq.', value: (0.1 + (h % 30) / 100).toFixed(2), hint: 'illiquidity' },
    { label: 'Roll spread', value: (2 + (h % 8)).toFixed(1) + ' bps', hint: 'effective spread' },
  ];
}

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
  const feats = symbolFeatures(symbol, sym.candles.map((c) => c.c));

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
                <ProbRing value={idea.regimeFitScore} label="Regime fit" size={58} />
                <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div>
                    <div className="eyebrow">Target weight</div>
                    <div className={`num ${idea.targetWeight >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 18, fontWeight: 700, marginTop: 3 }}>{idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}</div>
                  </div>
                  <div>
                    <div className="eyebrow">Pre-trade cost</div>
                    <div className="num" style={{ fontSize: 18, fontWeight: 700, marginTop: 3 }}>{idea.expectedCostBps ? fmtBps(idea.expectedCostBps) : '—'}</div>
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
          {idea ? (
            <MiniBars items={idea.shap.slice(0, 6).map((s) => ({ label: s.feature, value: s.contribution, sub: (s.contribution >= 0 ? '+' : '') + s.contribution.toFixed(3) }))} signed labelWidth={130} />
          ) : (
            <div className="muted" style={{ fontSize: 13 }}>No SHAP attribution without an active prediction.</div>
          )}
        </Panel>
      </div>

      {/* Features */}
      <Panel title="Live features" subtitle="Feature Factory · microstructure, volatility & classical signals">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {feats.map((f) => (
            <div key={f.label} style={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 12, padding: 14 }}>
              <div className="eyebrow">{f.label}</div>
              <div className="num" style={{ fontSize: 19, fontWeight: 680, marginTop: 6 }}>{f.value}</div>
              <div className="dim" style={{ fontSize: 11, marginTop: 3 }}>{f.hint}</div>
            </div>
          ))}
        </div>
      </Panel>

      {drawer && idea && <IdeaDrawer idea={idea} onClose={() => setDrawer(false)} onOpenSymbol={() => setDrawer(false)} />}
    </div>
  );
}
