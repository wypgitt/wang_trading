import { Panel } from '../../components/ui/Panel';
import { AreaChart } from '../../components/charts/AreaChart';
import { ActionPill, Bar } from '../../components/ui/primitives';
import { Icon } from '../../components/Icon';
import { useNav } from '../../nav';
import { stratBy, TRADE_IDEAS } from '../../data/mock';
import { CAT, C, REGIME_HEX, REGIME_LABEL } from '../../lib/colors';
import { fmtPct, fmtPctSigned } from '../../lib/format';

export function StrategyDetailPage({ id }: { id: string }) {
  const { go } = useNav();
  const s = stratBy(id);
  const ideas = TRADE_IDEAS.filter((i) => i.strategy === id);
  const eqUp = s.equityCurve[s.equityCurve.length - 1].v >= s.equityCurve[0].v;

  const stats: [string, string, string][] = [
    ['Sharpe', s.sharpe.toFixed(2), 'risk-adjusted'],
    ['Win rate', fmtPct(s.winRate, 1), `${s.trades} trades`],
    ['P&L share', fmtPct(s.contributionPct, 0), 'of portfolio'],
    ['YTD return', fmtPctSigned(s.pnlYtd), 'this sleeve'],
    ['Allocation', fmtPct(s.allocation, 0), 'of gross'],
    ['Avg hold', `${s.avgHoldBars} bars`, `${s.activeSignals} active`],
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Thesis hero */}
      <div className="card" style={{ padding: 22 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <span className="chip" style={{ borderColor: `${CAT[s.category]}55` }}>
            <span className="dot" style={{ background: CAT[s.category] }} />
            {s.category}
          </span>
          <span className="muted" style={{ fontSize: 12.5 }}>source · {s.source}</span>
          <span className="muted mono" style={{ fontSize: 12 }}>{s.id}</span>
          <div style={{ flex: 1 }} />
          {s.assetClasses.map((a) => (
            <span key={a} className="pill pill-neutral" style={{ textTransform: 'capitalize' }}>{a}</span>
          ))}
        </div>
        <p style={{ fontSize: 15, lineHeight: 1.6, color: C.text1, maxWidth: 880 }}>{s.thesis}</p>
      </div>

      {/* Stat strip */}
      <div className="card" style={{ padding: 0, display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)' }}>
        {stats.map(([l, v, sub], i) => (
          <div key={l} style={{ padding: '15px 18px', borderLeft: i ? '1px solid var(--border)' : 'none' }}>
            <div className="eyebrow">{l}</div>
            <div className="num" style={{ fontSize: 21, fontWeight: 680, marginTop: 5 }}>{v}</div>
            <div className="muted" style={{ fontSize: 11.5, marginTop: 2 }}>{sub}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 18, alignItems: 'start' }}>
        <Panel title="Strategy equity curve" subtitle="Cumulative sleeve P&L, rebased to 100">
          <AreaChart data={s.equityCurve} height={250} color={eqUp ? C.pos : C.neg} showTooltip showYAxis valueFmt={(v) => v.toFixed(0)} baseline={100} />
        </Panel>

        <Panel title="Regime fit" subtitle="Expected edge by market regime">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 13, marginTop: 4 }}>
            {(Object.keys(s.regimeFit) as (keyof typeof s.regimeFit)[]).map((k) => (
              <div key={k}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 6 }}>
                  <span style={{ color: C.text2 }}>{REGIME_LABEL[k]}</span>
                  <span className="num" style={{ fontWeight: 600 }}>{fmtPct(s.regimeFit[k], 0)}</span>
                </div>
                <Bar value={s.regimeFit[k]} color={REGIME_HEX[k]} height={8} />
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: 18, alignItems: 'start' }}>
        <Panel title="Parameters" subtitle="Live configuration">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {s.params.map((p, i) => (
              <div key={p.key} style={{ display: 'flex', justifyContent: 'space-between', padding: '11px 2px', borderTop: i ? '1px solid var(--border)' : 'none' }}>
                <span className="mono" style={{ fontSize: 12.5, color: C.text2 }}>{p.key}</span>
                <span className="mono" style={{ fontSize: 12.5, fontWeight: 600 }}>{p.value}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title={`Active ideas from this strategy (${ideas.length})`} subtitle="Current cycle">
          {ideas.length ? (
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {ideas.map((idea, i) => (
                <button
                  key={idea.symbol}
                  onClick={() => go('symbol', idea.symbol)}
                  className="row-hover"
                  style={{ display: 'grid', gridTemplateColumns: '1fr 88px 100px 90px 22px', alignItems: 'center', gap: 12, padding: '12px 8px', borderTop: i ? '1px solid var(--border)' : 'none', textAlign: 'left' }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <ActionPill action={idea.action} />
                    <span style={{ fontWeight: 650 }}>{idea.symbol}</span>
                  </div>
                  <div className="num" style={{ textAlign: 'right' }}>{idea.calibratedProbability?.toFixed(2) ?? '—'} <span className="dim" style={{ fontSize: 10 }}>cal</span></div>
                  <div className={`num ${idea.targetWeight >= 0 ? 'pos' : 'neg'}`} style={{ textAlign: 'right', fontWeight: 600 }}>{idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}</div>
                  <div className="num muted" style={{ textAlign: 'right' }}>{idea.regimeFitScore?.toFixed(2)}</div>
                  <span style={{ color: C.text3, display: 'flex', justifyContent: 'flex-end' }}><Icon name="chevronRight" size={15} /></span>
                </button>
              ))}
            </div>
          ) : (
            <div className="muted" style={{ fontSize: 13, padding: '8px 0' }}>No active ideas from this strategy this cycle.</div>
          )}
        </Panel>
      </div>
    </div>
  );
}
