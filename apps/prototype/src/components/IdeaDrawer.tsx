import { Icon } from './Icon';
import { ActionPill, ProbRing } from './ui/primitives';
import { MiniBars } from './charts/MiniBars';
import { TradeIdea } from '../data/mock';
import { C } from '../lib/colors';
import { fmtBps, fmtPct, fmtPctSigned, fmtPrice, sideLabel } from '../lib/format';

function Section({ label, children, action }: { label: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div style={{ padding: '18px 22px', borderTop: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div className="eyebrow">{label}</div>
        {action}
      </div>
      {children}
    </div>
  );
}

export function IdeaDrawer({ idea, onClose, onOpenSymbol }: { idea: TradeIdea; onClose: () => void; onOpenSymbol: (s: string) => void }) {
  const chain = [
    { label: 'Signals', value: String(idea.signalCount), tone: C.text1 },
    { label: 'Meta p', value: idea.metaProbability?.toFixed(2) ?? '—', tone: C.accent2 },
    { label: 'Calibrated', value: idea.calibratedProbability?.toFixed(2) ?? '—', tone: C.accent },
    { label: 'Regime fit', value: idea.regimeFitScore?.toFixed(2) ?? '—', tone: C.regimeUp },
    { label: 'Bet size', value: idea.betSize ? fmtPct(idea.betSize, 1) : '—', tone: C.warn },
    { label: 'Target', value: fmtPctSigned(idea.targetWeight), tone: idea.targetWeight >= 0 ? C.pos : C.neg },
  ];
  const cascadeMax = Math.max(...idea.cascade.map((c) => c.value), 1e-6);
  const latencyTotal = Object.values(idea.stageLatency).reduce((a, b) => a + b, 0);

  return (
    <div className="backdrop" style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', justifyContent: 'flex-end', background: 'rgba(4,6,9,0.58)' }} onClick={onClose}>
      <div
        className="drawer-panel scroll-y"
        style={{ width: 500, maxWidth: '94vw', height: '100%', background: 'var(--bg-1)', borderLeft: '1px solid var(--border)', boxShadow: 'var(--shadow-pop)' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ padding: '18px 22px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', position: 'sticky', top: 0, background: 'var(--bg-1)', zIndex: 2 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 22, fontWeight: 720, letterSpacing: '-0.02em' }}>{idea.symbol}</span>
              <ActionPill action={idea.action} />
            </div>
            <div className="muted" style={{ fontSize: 12.5, marginTop: 3 }}>
              {idea.strategy} · {sideLabel(idea.topSignalSide)} · {idea.barType} bars
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" style={{ height: 34, padding: '0 12px', fontSize: 12.5 }} onClick={() => onOpenSymbol(idea.symbol)}>
              Open <Icon name="arrowUpRight" size={14} />
            </button>
            <button className="btn" style={{ width: 34, height: 34, padding: 0 }} onClick={onClose}>
              <Icon name="close" size={16} />
            </button>
          </div>
        </div>

        {/* Reason */}
        <div style={{ padding: '0 22px 18px' }}>
          <div style={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 12, padding: 14, fontSize: 13.5, lineHeight: 1.55, color: C.text1 }}>
            {idea.reason}
          </div>
        </div>

        {/* Decision chain */}
        <Section label="Decision chain">
          <div style={{ display: 'flex', alignItems: 'stretch', gap: 6 }}>
            {chain.map((step, i) => (
              <div key={step.label} style={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}>
                <div style={{ flex: 1, background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 10, padding: '9px 6px', textAlign: 'center' }}>
                  <div className="num" style={{ fontSize: 14.5, fontWeight: 700, color: step.tone }}>
                    {step.value}
                  </div>
                  <div className="dim" style={{ fontSize: 9.5, marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                    {step.label}
                  </div>
                </div>
                {i < chain.length - 1 && <span style={{ color: C.text3, fontSize: 12, padding: '0 1px' }}>›</span>}
              </div>
            ))}
          </div>
        </Section>

        {/* Probabilities + track record */}
        <Section label="Model conviction">
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <ProbRing value={idea.metaProbability} label="Meta" size={64} />
            <ProbRing value={idea.calibratedProbability} label="Calibrated" size={64} />
            <div style={{ flex: 1 }}>
              {idea.trackRecordWinRate != null ? (
                <>
                  <div className="muted" style={{ fontSize: 12 }}>Historical track record</div>
                  <div className="num" style={{ fontSize: 20, fontWeight: 700, marginTop: 3 }}>
                    {fmtPct(idea.trackRecordWinRate, 0)} <span className="muted" style={{ fontSize: 13, fontWeight: 500 }}>win</span>
                  </div>
                  <div className="dim" style={{ fontSize: 11.5 }}>over {idea.trackRecordN} similar calls</div>
                </>
              ) : (
                <div className="muted" style={{ fontSize: 12.5 }}>No track record — model inference skipped this cycle.</div>
              )}
            </div>
          </div>
        </Section>

        {/* Signals */}
        <Section label={`Signals (${idea.signals.length})`}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {idea.signals.map((s, i) => (
              <div key={i} style={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 11, padding: '11px 13px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600, fontSize: 13 }} className="mono">
                    {s.family}
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`pill ${s.side > 0 ? 'pill-buy' : s.side < 0 ? 'pill-sell' : 'pill-neutral'}`}>{sideLabel(s.side)}</span>
                    <span className="num muted" style={{ fontSize: 12 }}>conf {s.confidence.toFixed(2)}</span>
                  </span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 9 }}>
                  {Object.entries(s.meta).map(([k, v]) => (
                    <span key={k} style={{ fontSize: 11, color: C.text3, background: 'var(--surface-inset)', borderRadius: 6, padding: '3px 7px' }}>
                      <span className="mono">{k}</span> <span className="mono" style={{ color: C.text2 }}>{String(v)}</span>
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* Sizing cascade */}
        <Section label="Bet-sizing cascade" action={<span className="dim" style={{ fontSize: 11 }}>5-layer · binding highlighted</span>}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {idea.cascade.map((c) => (
              <div key={c.stage} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ width: 92, fontSize: 12, color: c.binding ? C.warn : C.text2 }}>{c.stage}</span>
                <div style={{ flex: 1, height: 22, background: 'var(--surface-inset)', borderRadius: 7, position: 'relative', overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', inset: 0, width: `${(c.value / cascadeMax) * 100}%`, background: c.binding ? 'var(--warn-soft)' : 'var(--accent-soft)', borderRight: `2px solid ${c.binding ? C.warn : C.accent}` }} />
                  <span className="num" style={{ position: 'absolute', right: 8, top: 3, fontSize: 12, fontWeight: 600 }}>{fmtPct(c.value, 1)}</span>
                </div>
                {c.binding && <span className="pill" style={{ background: 'var(--warn-soft)', color: C.warn }}>binding</span>}
              </div>
            ))}
          </div>
        </Section>

        {/* SHAP */}
        <Section label="Why the model leaned this way" action={<span className="dim" style={{ fontSize: 11 }}>SHAP contributions</span>}>
          <MiniBars
            items={idea.shap.slice(0, 6).map((s) => ({ label: s.feature, value: s.contribution, sub: (s.contribution >= 0 ? '+' : '') + s.contribution.toFixed(3) }))}
            signed
            labelWidth={150}
          />
        </Section>

        {/* Cost + latency */}
        <Section label="Execution & timing">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div className="muted" style={{ fontSize: 12 }}>Pre-trade cost</div>
              <div className="num" style={{ fontSize: 20, fontWeight: 700, marginTop: 3 }}>{idea.expectedCostBps ? fmtBps(idea.expectedCostBps) : '—'}</div>
              <div className="dim" style={{ fontSize: 11.5, marginTop: 7 }}>Constraints</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 5 }}>
                {idea.sizingConstraints.length ? (
                  idea.sizingConstraints.map((c) => (
                    <span key={c} className="pill pill-neutral mono" style={{ fontSize: 10.5 }}>
                      {c}
                    </span>
                  ))
                ) : (
                  <span className="dim" style={{ fontSize: 12 }}>none</span>
                )}
              </div>
            </div>
            <div>
              <div className="muted" style={{ fontSize: 12 }}>Pipeline latency · {(latencyTotal * 1000).toFixed(0)}ms</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
                {Object.entries(idea.stageLatency).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ width: 96, fontSize: 11, color: C.text3 }} className="mono">{k}</span>
                    <div style={{ flex: 1, height: 5, background: 'var(--surface-inset)', borderRadius: 99 }}>
                      <div style={{ width: `${(v / latencyTotal) * 100}%`, height: '100%', background: C.accent2, borderRadius: 99 }} />
                    </div>
                    <span className="num dim" style={{ fontSize: 10.5, width: 40, textAlign: 'right' }}>{(v * 1000).toFixed(0)}ms</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Section>

        <div style={{ padding: 22, display: 'flex', gap: 10 }}>
          <button className="btn btn-primary" style={{ flex: 1 }}>
            <Icon name="check" size={16} /> Stage for execution
          </button>
          <button className="btn" onClick={() => onOpenSymbol(idea.symbol)}>
            Symbol detail
          </button>
        </div>
      </div>
    </div>
  );
}
