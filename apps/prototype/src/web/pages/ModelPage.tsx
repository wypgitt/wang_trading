import { Bar as RBar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Panel } from '../../components/ui/Panel';
import { MiniBars } from '../../components/charts/MiniBars';
import { Icon } from '../../components/Icon';
import { MODEL } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtPct } from '../../lib/format';

const FAM_COLOR: Record<string, string> = {
  Momentum: '#4d9fff',
  Volatility: '#f0a93b',
  Regime: '#1ecb8b',
  Microstructure: '#f6679a',
  FracDiff: '#b07cff',
  Classical: '#22d3ee',
  Sentiment: '#7c5cff',
  StructuralBreak: '#9aa4b2',
};
const SEV_COLOR = { ok: C.pos, warn: C.warn, alert: C.neg } as const;

export function ModelPage() {
  const calData = MODEL.calibration.map((b) => ({ predicted: b.predicted, observed: b.observed, ref: b.predicted }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Header */}
      <div className="card" style={{ padding: 22, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 18 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span className="mono" style={{ fontSize: 22, fontWeight: 720 }}>{MODEL.version}</span>
            <span className="pill pill-buy"><span className="dot" style={{ background: C.pos }} /> Promoted</span>
          </div>
          <div className="muted" style={{ fontSize: 13, marginTop: 5 }}>
            {MODEL.type} · trained {MODEL.trainedAt} · {MODEL.lastRetrainHours}h ago
          </div>
        </div>
        <div style={{ display: 'flex', gap: 28 }}>
          {[
            ['AUC', MODEL.auc.toFixed(3)],
            ['Brier', MODEL.brier.toFixed(3)],
            ['ECE', MODEL.ece.toFixed(3)],
          ].map(([l, v]) => (
            <div key={l} style={{ textAlign: 'right' }}>
              <div className="eyebrow">{l}</div>
              <div className="num" style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Calibration + histogram */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, alignItems: 'start' }}>
        <Panel title="Calibration reliability" subtitle="Predicted vs observed win rate · diagonal = perfect">
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={calData} margin={{ top: 6, right: 10, bottom: 4, left: -8 }}>
              <CartesianGrid stroke={C.grid} />
              <XAxis dataKey="predicted" type="number" domain={[0, 1]} tickCount={6} tick={{ fill: C.text3, fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(1)} />
              <YAxis domain={[0, 1]} tickCount={6} tick={{ fill: C.text3, fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip contentStyle={{ background: C.surface3, border: `1px solid ${C.borderStrong}`, borderRadius: 9, fontSize: 12.5 }} />
              <Line type="linear" dataKey="ref" stroke={C.text3} strokeWidth={1.4} strokeDasharray="4 4" dot={false} isAnimationActive={false} name="Perfect" />
              <Line type="monotone" dataKey="observed" stroke={C.accent} strokeWidth={2.4} dot={{ r: 3, fill: C.accent }} isAnimationActive={false} name="Observed" />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Meta-probability distribution" subtitle="Predictions this cycle by confidence bucket">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={MODEL.metaProbHist} margin={{ top: 6, right: 8, bottom: 4, left: -16 }}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="bucket" tick={{ fill: C.text3, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: C.text3, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} contentStyle={{ background: C.surface3, border: `1px solid ${C.borderStrong}`, borderRadius: 9, fontSize: 12.5 }} />
              <RBar dataKey="count" fill={C.accent} radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Feature importance + drift */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.25fr 1fr', gap: 18, alignItems: 'start' }}>
        <Panel title="Feature importance" subtitle="Top drivers of the meta-labeler · colored by family">
          <MiniBars
            items={MODEL.featureImportance.map((f) => ({ label: f.feature, value: f.importance, color: FAM_COLOR[f.family] ?? C.accent, sub: fmtPct(f.importance, 1) }))}
            labelWidth={170}
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 16 }}>
            {[...new Set(MODEL.featureImportance.map((f) => f.family))].map((fam) => (
              <span key={fam} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11.5, color: C.text2 }}>
                <span className="dot" style={{ background: FAM_COLOR[fam] ?? C.accent }} /> {fam}
              </span>
            ))}
          </div>
        </Panel>

        <Panel title="Feature drift" subtitle="KL divergence vs training distribution">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {MODEL.drift.map((d) => (
              <div key={d.feature}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 5 }}>
                  <span className="mono" style={{ color: C.text2 }}>{d.feature}</span>
                  <span className="num" style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontWeight: 600 }}>
                    <span className="dot" style={{ background: SEV_COLOR[d.severity] }} />
                    {d.kl.toFixed(2)}
                  </span>
                </div>
                <div style={{ height: 7, background: C.surfaceInset, borderRadius: 99, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, (d.kl / 0.5) * 100)}%`, height: '100%', background: SEV_COLOR[d.severity], borderRadius: 99 }} />
                </div>
              </div>
            ))}
          </div>
          <div className="muted" style={{ fontSize: 11.5, marginTop: 14 }}>
            <span style={{ color: C.neg }}>vpin</span> breached the alert threshold — retrain trigger armed.
          </div>
        </Panel>
      </div>

      {/* RL shadow + retrain timeline */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.3fr', gap: 18, alignItems: 'start' }}>
        <Panel title="RL agent · shadow mode" subtitle="PPO policy running in parallel, not trading">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 16 }}>
            {[
              ['Agreement', fmtPct(MODEL.rlShadow.agreement, 0), C.accent],
              ['Shadow Sharpe', MODEL.rlShadow.shadowSharpe.toFixed(2), C.pos],
              ['Live Sharpe', MODEL.rlShadow.liveSharpe.toFixed(2), C.text1],
            ].map(([l, v, c]) => (
              <div key={l}>
                <div className="eyebrow">{l}</div>
                <div className="num" style={{ fontSize: 22, fontWeight: 700, marginTop: 4, color: c as string }}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 12.5, color: C.text2, background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 10, padding: '11px 13px' }}>
            <Icon name="shield" size={16} /> {MODEL.rlShadow.status}
          </div>
        </Panel>

        <Panel title="Retrain timeline" subtitle="Weekly + drift-triggered, gated by the 3 promotion checks">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {MODEL.retrainTimeline.map((r, i) => (
              <div key={i} style={{ display: 'flex', gap: 14, padding: '12px 0', borderTop: i ? '1px solid var(--border)' : 'none' }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 3 }}>
                  <span className="dot" style={{ width: 9, height: 9, background: r.promoted ? C.pos : C.neg }} />
                  {i < MODEL.retrainTimeline.length - 1 && <span style={{ width: 1, flex: 1, background: C.border, marginTop: 4 }} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: 13.5, fontWeight: 550 }}>{r.event}</span>
                    <span className="num muted" style={{ fontSize: 12.5 }}>Sharpe {r.sharpe.toFixed(2)}</span>
                  </div>
                  <div className="dim mono" style={{ fontSize: 11.5, marginTop: 2 }}>{r.date}</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
