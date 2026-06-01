import { useState } from 'react';
import { AreaChart } from '../../components/charts/AreaChart';
import { ActionPill, AssetGlyph, Delta, ProbRing } from '../../components/ui/primitives';
import { IOSCard, NavHeader, Screen } from '../iosUi';
import { useIOS } from '../iosNav';
import { symBy, TRADE_IDEAS } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtBps, fmtCompact, fmtPctSigned, fmtPrice, sideLabel } from '../../lib/format';

const TF: Record<string, number> = { '1M': 22, '3M': 66, '6M': 130 };

export function SymbolScreen({ id }: { id: string }) {
  const { pop } = useIOS();
  const sym = symBy(id);
  const idea = TRADE_IDEAS.find((i) => i.symbol === id) ?? null;
  const [tf, setTf] = useState('3M');
  const line = sym.candles.slice(-TF[tf]).map((c, i) => ({ t: i, v: c.c }));
  const net = line[line.length - 1].v / line[0].v - 1;

  return (
    <>
      <NavHeader title={sym.symbol} subtitle={sym.name} onBack={pop} />
      <div className="scroll-y" style={{ flex: 1, minHeight: 0 }}>
        <Screen>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 2px 14px' }}>
            <AssetGlyph sym={sym} size={44} />
            <div>
              <div className="num" style={{ fontSize: 30, fontWeight: 740, letterSpacing: '-0.03em' }}>{fmtPrice(sym.price)}</div>
              <Delta value={sym.change1d} size={14} />
            </div>
          </div>

          <IOSCard pad={14}>
            <AreaChart data={line} height={180} color={net >= 0 ? C.pos : C.neg} showTooltip valueFmt={fmtPrice} />
            <div style={{ display: 'inline-flex', gap: 3, background: C.surfaceInset, borderRadius: 10, padding: 3, marginTop: 8 }}>
              {Object.keys(TF).map((k) => (
                <button key={k} onClick={() => setTf(k)} className="num" style={{ height: 28, padding: '0 16px', borderRadius: 7, fontSize: 12.5, fontWeight: 600, color: k === tf ? '#fff' : C.text2, background: k === tf ? C.surface3 : 'transparent' }}>{k}</button>
              ))}
            </div>
          </IOSCard>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 12 }}>
            {[
              ['1W', fmtPctSigned(sym.change1w)],
              ['1M', fmtPctSigned(sym.change1m)],
              ['YTD', fmtPctSigned(sym.changeYtd)],
            ].map(([l, v]) => (
              <IOSCard key={l} pad={12}>
                <div className="eyebrow">{l}</div>
                <div className={`num ${String(v).startsWith('-') ? 'neg' : 'pos'}`} style={{ fontSize: 15, fontWeight: 680, marginTop: 3 }}>{v}</div>
              </IOSCard>
            ))}
          </div>

          {idea && (
            <IOSCard pad={16} style={{ marginTop: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <span style={{ fontWeight: 650, fontSize: 15 }}>What the engine sees</span>
                <ActionPill action={idea.action} />
              </div>
              <p style={{ fontSize: 13.5, lineHeight: 1.55, color: C.text1, marginBottom: 14 }}>{idea.reason}</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <ProbRing value={idea.metaProbability} label="Meta" size={52} />
                <ProbRing value={idea.calibratedProbability} label="Cal." size={52} />
                <ProbRing value={idea.regimeFitScore} label="Regime" size={52} />
                <div style={{ flex: 1 }}>
                  <div className="eyebrow">Target · cost</div>
                  <div className={`num ${idea.targetWeight >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 18, fontWeight: 700, marginTop: 3 }}>
                    {idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}
                  </div>
                  <div className="num dim" style={{ fontSize: 12 }}>{idea.expectedCostBps ? fmtBps(idea.expectedCostBps) : '—'} · {sideLabel(idea.topSignalSide)}</div>
                </div>
              </div>
            </IOSCard>
          )}

          <IOSCard pad={16} style={{ marginTop: 12 }}>
            <div style={{ fontWeight: 650, fontSize: 15, marginBottom: 12 }}>Snapshot</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
              {[
                ['Market cap / Vol', sym.marketCap ? fmtCompact(sym.marketCap) : fmtCompact(sym.volume)],
                ['Bar type', sym.type === 'crypto' ? 'dollar' : 'tib'],
                ['Asset class', sym.type],
                ['Active idea', idea ? idea.action : 'none'],
              ].map(([l, v]) => (
                <div key={l}>
                  <div className="eyebrow">{l}</div>
                  <div className="num" style={{ fontSize: 15, fontWeight: 600, marginTop: 3, textTransform: 'capitalize' }}>{v}</div>
                </div>
              ))}
            </div>
          </IOSCard>
        </Screen>
      </div>
    </>
  );
}
