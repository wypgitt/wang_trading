import { useState } from 'react';
import { Sparkline } from '../../components/charts/Sparkline';
import { Icon } from '../../components/Icon';
import { IOSCard, LargeTitle, Screen } from '../iosUi';
import { useIOS } from '../iosNav';
import { STRATEGIES } from '../../data/mock';
import { CAT, C } from '../../lib/colors';
import { fmtPct } from '../../lib/format';

const CATS = ['All', 'Momentum', 'Mean Reversion', 'Trend', 'Volatility', 'Carry', 'Arbitrage'];

export function Strategies() {
  const { push } = useIOS();
  const [cat, setCat] = useState('All');
  const visible = cat === 'All' ? STRATEGIES : STRATEGIES.filter((s) => s.category === cat);

  return (
    <Screen>
      <LargeTitle title="Strategies" subtitle="10 signal families" />

      <div style={{ display: 'flex', gap: 8, overflowX: 'auto', margin: '0 -16px 14px', padding: '0 16px' }}>
        {CATS.map((c) => (
          <button key={c} className={`chip ${cat === c ? 'active' : ''}`} style={{ flex: 'none' }} onClick={() => setCat(c)}>
            {c !== 'All' && <span className="dot" style={{ background: CAT[c] }} />}
            {c}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {visible.map((s) => {
          const eq = s.equityCurve.map((p) => p.v);
          const up = eq[eq.length - 1] >= eq[0];
          return (
            <button key={s.id} onClick={() => push({ type: 'strategy', id: s.id })} style={{ textAlign: 'left' }}>
              <IOSCard pad={15}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 660, fontSize: 16 }}>{s.name}</div>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginTop: 3, fontSize: 12, color: C.text2 }}>
                      <span className="dot" style={{ background: CAT[s.category] }} />
                      {s.category}
                      <span className={`pill ${s.status === 'live' ? 'pill-buy' : 'pill-watch'}`} style={{ marginLeft: 4 }}>{s.status}</span>
                    </div>
                  </div>
                  <Sparkline data={eq} width={92} height={34} color={up ? C.pos : C.neg} />
                  <Icon name="chevronRight" size={16} />
                </div>
                <div style={{ display: 'flex', gap: 22, marginTop: 13 }}>
                  {[
                    ['Sharpe', s.sharpe.toFixed(2)],
                    ['Win', fmtPct(s.winRate, 0)],
                    ['P&L share', fmtPct(s.contributionPct, 0)],
                    ['Alloc', fmtPct(s.allocation, 0)],
                  ].map(([l, v]) => (
                    <div key={l}>
                      <div className="dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{l}</div>
                      <div className="num" style={{ fontSize: 15, fontWeight: 680, marginTop: 2 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </IOSCard>
            </button>
          );
        })}
      </div>
    </Screen>
  );
}
