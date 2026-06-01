import { MiniBars } from '../../components/charts/MiniBars';
import { ActionPill, ProbRing } from '../../components/ui/primitives';
import { IOSCard, NavHeader, Screen, SectionHeader } from '../iosUi';
import { useIOS } from '../iosNav';
import { TRADE_IDEAS } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtBps, fmtPct, fmtPctSigned, sideLabel } from '../../lib/format';

export function IdeaScreen({ id }: { id: string }) {
  const { pop, push } = useIOS();
  const idea = TRADE_IDEAS.find((i) => i.symbol === id) ?? TRADE_IDEAS[0];

  const chain = [
    { label: 'Signals', value: String(idea.signalCount), tone: C.text1 },
    { label: 'Meta p', value: idea.metaProbability?.toFixed(2) ?? '—', tone: C.accent2 },
    { label: 'Calibrated', value: idea.calibratedProbability?.toFixed(2) ?? '—', tone: C.accent },
    { label: 'Regime fit', value: idea.regimeFitScore?.toFixed(2) ?? '—', tone: C.regimeUp },
    { label: 'Bet size', value: idea.betSize ? fmtPct(idea.betSize, 1) : '—', tone: C.warn },
    { label: 'Target', value: fmtPctSigned(idea.targetWeight), tone: idea.targetWeight >= 0 ? C.pos : C.neg },
  ];
  const cascadeMax = Math.max(...idea.cascade.map((c) => c.value), 1e-6);

  return (
    <>
      <NavHeader title={idea.symbol} subtitle={idea.strategy ?? undefined} onBack={pop} right={<ActionPill action={idea.action} />} />
      <div className="scroll-y" style={{ flex: 1, minHeight: 0 }}>
        <Screen>
          <IOSCard pad={15} style={{ marginTop: 12 }}>
            <p style={{ fontSize: 14, lineHeight: 1.6, color: C.text1 }}>{idea.reason}</p>
          </IOSCard>

          <SectionHeader title="Decision chain" />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 9 }}>
            {chain.map((s) => (
              <IOSCard key={s.label} pad={12} style={{ textAlign: 'center' }}>
                <div className="num" style={{ fontSize: 18, fontWeight: 720, color: s.tone }}>{s.value}</div>
                <div className="dim" style={{ fontSize: 10, marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.03em' }}>{s.label}</div>
              </IOSCard>
            ))}
          </div>

          <SectionHeader title="Model conviction" />
          <IOSCard pad={16}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around' }}>
              <ProbRing value={idea.metaProbability} label="Meta" size={62} />
              <ProbRing value={idea.calibratedProbability} label="Calibrated" size={62} />
              <div style={{ textAlign: 'center' }}>
                <div className="num" style={{ fontSize: 22, fontWeight: 720 }}>{idea.trackRecordWinRate ? fmtPct(idea.trackRecordWinRate, 0) : '—'}</div>
                <div className="eyebrow" style={{ marginTop: 4 }}>win · {idea.trackRecordN ?? 0}n</div>
              </div>
            </div>
          </IOSCard>

          <SectionHeader title={`Signals (${idea.signals.length})`} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
            {idea.signals.map((s, i) => (
              <IOSCard key={i} pad={13}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="mono" style={{ fontWeight: 600, fontSize: 13 }}>{s.family}</span>
                  <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <span className={`pill ${s.side > 0 ? 'pill-buy' : s.side < 0 ? 'pill-sell' : 'pill-neutral'}`}>{sideLabel(s.side)}</span>
                    <span className="num muted" style={{ fontSize: 12 }}>{s.confidence.toFixed(2)}</span>
                  </span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 9 }}>
                  {Object.entries(s.meta).map(([k, v]) => (
                    <span key={k} style={{ fontSize: 10.5, color: C.text3, background: C.surfaceInset, borderRadius: 6, padding: '3px 7px' }} className="mono">
                      {k} <span style={{ color: C.text2 }}>{String(v)}</span>
                    </span>
                  ))}
                </div>
              </IOSCard>
            ))}
          </div>

          <SectionHeader title="Bet-sizing cascade" />
          <IOSCard pad={15}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
              {idea.cascade.map((c) => (
                <div key={c.stage} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ width: 78, fontSize: 11.5, color: c.binding ? C.warn : C.text2 }}>{c.stage}</span>
                  <div style={{ flex: 1, height: 20, background: C.surfaceInset, borderRadius: 6, position: 'relative', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', inset: 0, width: `${(c.value / cascadeMax) * 100}%`, background: c.binding ? 'var(--warn-soft)' : 'var(--accent-soft)', borderRight: `2px solid ${c.binding ? C.warn : C.accent}` }} />
                    <span className="num" style={{ position: 'absolute', right: 7, top: 2, fontSize: 11.5, fontWeight: 600 }}>{fmtPct(c.value, 1)}</span>
                  </div>
                </div>
              ))}
            </div>
          </IOSCard>

          <SectionHeader title="Why · SHAP" />
          <IOSCard pad={15}>
            <MiniBars items={idea.shap.slice(0, 6).map((s) => ({ label: s.feature, value: s.contribution, sub: (s.contribution >= 0 ? '+' : '') + s.contribution.toFixed(3) }))} signed labelWidth={120} />
          </IOSCard>

          <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
            <button className="btn btn-primary" style={{ flex: 1 }} onClick={() => push({ type: 'symbol', id: idea.symbol })}>
              Open {idea.symbol}
            </button>
            <button className="btn" style={{ flex: 1 }}>Stage for execution</button>
          </div>
        </Screen>
      </div>
    </>
  );
}
