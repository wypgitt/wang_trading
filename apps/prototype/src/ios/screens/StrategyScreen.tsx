import { AreaChart } from '../../components/charts/AreaChart';
import { Bar } from '../../components/ui/primitives';
import { IOSCard, NavHeader, Screen, SectionHeader } from '../iosUi';
import { useIOS } from '../iosNav';
import { stratBy } from '../../data/mock';
import { CAT, C, REGIME_HEX, REGIME_LABEL } from '../../lib/colors';
import { fmtPct, fmtPctSigned } from '../../lib/format';

export function StrategyScreen({ id }: { id: string }) {
  const { pop } = useIOS();
  const s = stratBy(id);
  const up = s.equityCurve[s.equityCurve.length - 1].v >= s.equityCurve[0].v;

  return (
    <>
      <NavHeader title={s.name} subtitle={s.category} onBack={pop} />
      <div className="scroll-y" style={{ flex: 1, minHeight: 0 }}>
        <Screen>
          <IOSCard pad={16} style={{ marginTop: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <span className="chip" style={{ borderColor: `${CAT[s.category]}55` }}>
                <span className="dot" style={{ background: CAT[s.category] }} />
                {s.category}
              </span>
              <span className={`pill ${s.status === 'live' ? 'pill-buy' : 'pill-watch'}`}>{s.status}</span>
              <span className="dim mono" style={{ fontSize: 11.5, marginLeft: 'auto' }}>{s.source}</span>
            </div>
            <p style={{ fontSize: 14, lineHeight: 1.6, color: C.text1 }}>{s.thesis}</p>
          </IOSCard>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 12 }}>
            {[
              ['Sharpe', s.sharpe.toFixed(2)],
              ['Win rate', fmtPct(s.winRate, 0)],
              ['YTD', fmtPctSigned(s.pnlYtd)],
              ['P&L share', fmtPct(s.contributionPct, 0)],
              ['Allocation', fmtPct(s.allocation, 0)],
              ['Avg hold', `${s.avgHoldBars}b`],
            ].map(([l, v]) => (
              <IOSCard key={l} pad={12}>
                <div className="eyebrow">{l}</div>
                <div className="num" style={{ fontSize: 16, fontWeight: 680, marginTop: 3 }}>{v}</div>
              </IOSCard>
            ))}
          </div>

          <SectionHeader title="Equity curve" />
          <IOSCard pad={14}>
            <AreaChart data={s.equityCurve} height={170} color={up ? C.pos : C.neg} showTooltip baseline={100} valueFmt={(v) => v.toFixed(0)} />
          </IOSCard>

          <SectionHeader title="Regime fit" />
          <IOSCard pad={16}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {(Object.keys(s.regimeFit) as (keyof typeof s.regimeFit)[]).map((k) => (
                <div key={k}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 5 }}>
                    <span style={{ color: C.text2 }}>{REGIME_LABEL[k]}</span>
                    <span className="num" style={{ fontWeight: 600 }}>{fmtPct(s.regimeFit[k], 0)}</span>
                  </div>
                  <Bar value={s.regimeFit[k]} color={REGIME_HEX[k]} height={7} />
                </div>
              ))}
            </div>
          </IOSCard>

          <SectionHeader title="Parameters" />
          <IOSCard pad={4}>
            {s.params.map((p, i) => (
              <div key={p.key} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 12px', borderTop: i ? '1px solid var(--border)' : 'none' }}>
                <span className="mono" style={{ fontSize: 12.5, color: C.text2 }}>{p.key}</span>
                <span className="mono" style={{ fontSize: 12.5, fontWeight: 600 }}>{p.value}</span>
              </div>
            ))}
          </IOSCard>
        </Screen>
      </div>
    </>
  );
}
