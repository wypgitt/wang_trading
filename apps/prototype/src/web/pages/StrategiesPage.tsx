import { useState } from 'react';
import { Sparkline } from '../../components/charts/Sparkline';
import { useNav } from '../../nav';
import { STRATEGIES, Strategy } from '../../data/mock';
import { CAT, C } from '../../lib/colors';
import { fmtPct } from '../../lib/format';

function StatusPill({ status }: { status: Strategy['status'] }) {
  const map = { live: ['pill-buy', 'Live'], shadow: ['pill-watch', 'Shadow'], paused: ['pill-neutral', 'Paused'] } as const;
  const [cls, label] = map[status];
  return <span className={`pill ${cls}`}>{label}</span>;
}

const CATS = ['All', 'Momentum', 'Mean Reversion', 'Trend', 'Volatility', 'Carry', 'Arbitrage'];

export function StrategiesPage() {
  const { go } = useNav();
  const [cat, setCat] = useState('All');
  const visible = cat === 'All' ? STRATEGIES : STRATEGIES.filter((s) => s.category === cat);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {CATS.map((c) => (
            <button key={c} className={`chip ${cat === c ? 'active' : ''}`} onClick={() => setCat(c)}>
              {c !== 'All' && <span className="dot" style={{ background: CAT[c] }} />}
              {c}
            </button>
          ))}
        </div>
        <span className="muted" style={{ fontSize: 12.5 }}>
          {STRATEGIES.filter((s) => s.status === 'live').length} live · {STRATEGIES.filter((s) => s.status === 'shadow').length} shadow
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
        {visible.map((s) => {
          const eq = s.equityCurve.map((p) => p.v);
          const up = eq[eq.length - 1] >= eq[0];
          return (
            <button
              key={s.id}
              onClick={() => go('strategy', s.id)}
              className="card lift"
              style={{ padding: 17, textAlign: 'left', display: 'flex', flexDirection: 'column', gap: 13 }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ fontWeight: 660, fontSize: 15 }}>{s.name}</div>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginTop: 5, fontSize: 12, color: C.text2 }}>
                    <span className="dot" style={{ background: CAT[s.category] }} />
                    {s.category}
                  </div>
                </div>
                <StatusPill status={s.status} />
              </div>

              <Sparkline data={eq} width={300} height={40} color={up ? C.pos : C.neg} />

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                {[
                  ['Sharpe', s.sharpe.toFixed(2)],
                  ['Win rate', fmtPct(s.winRate, 0)],
                  ['P&L share', fmtPct(s.contributionPct, 0)],
                ].map(([l, v]) => (
                  <div key={l}>
                    <div className="dim" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{l}</div>
                    <div className="num" style={{ fontSize: 16, fontWeight: 680, marginTop: 2 }}>{v}</div>
                  </div>
                ))}
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: C.text3, marginBottom: 5 }}>
                  <span>Allocation</span>
                  <span className="num">{fmtPct(s.allocation, 0)} of gross</span>
                </div>
                <div style={{ height: 6, background: C.surfaceInset, borderRadius: 99, overflow: 'hidden' }}>
                  <div style={{ width: `${(s.allocation / 0.25) * 100}%`, height: '100%', background: CAT[s.category], borderRadius: 99 }} />
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
