import { useState } from 'react';
import { AreaChart } from '../../components/charts/AreaChart';
import { Sparkline } from '../../components/charts/Sparkline';
import { ActionPill, Delta, RegimeBar } from '../../components/ui/primitives';
import { Icon } from '../../components/Icon';
import { IOSCard, LargeTitle, Screen, SectionHeader } from '../iosUi';
import { useIOS } from '../iosNav';
import { PORTFOLIO, REGIME, SYMBOLS, TRADE_IDEAS } from '../../data/mock';
import { C, REGIME_LABEL } from '../../lib/colors';
import { fmtCompact, fmtPct, fmtPctSigned, fmtPrice } from '../../lib/format';

const TF: Record<string, number> = { '1W': 7, '1M': 22, '3M': 66, All: 180 };

export function Home() {
  const { setTab, push } = useIOS();
  const [tf, setTf] = useState('3M');
  const slice = PORTFOLIO.navHistory.slice(-TF[tf]).map((p, i) => ({ t: i, v: p.v }));
  const net = slice[slice.length - 1].v / slice[0].v - 1;
  const col = net >= 0 ? C.pos : C.neg;
  const ideas = TRADE_IDEAS.filter((i) => i.action === 'BUY' || i.action === 'SELL').slice(0, 4);
  const movers = [...SYMBOLS].sort((a, b) => Math.abs(b.change1d) - Math.abs(a.change1d)).slice(0, 6);

  return (
    <Screen>
      <LargeTitle title="Overview" subtitle="Paper · live" right={<span className="dot live-dot" style={{ background: C.pos, width: 9, height: 9 }} />} />

      <IOSCard pad={18}>
        <div className="eyebrow">Net asset value</div>
        <div className="num" style={{ fontSize: 38, fontWeight: 740, letterSpacing: '-0.03em', marginTop: 4 }}>{fmtCompact(PORTFOLIO.nav)}</div>
        <div className={`num ${PORTFOLIO.dailyPnl >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 14.5, fontWeight: 600, marginTop: 2 }}>
          {PORTFOLIO.dailyPnl >= 0 ? '▲' : '▼'} {fmtCompact(Math.abs(PORTFOLIO.dailyPnl))} ({fmtPctSigned(PORTFOLIO.dailyPnlPct)}) today
        </div>
        <div style={{ margin: '6px -4px 10px' }}>
          <AreaChart data={slice} height={150} color={col} showTooltip valueFmt={fmtCompact} />
        </div>
        <div style={{ display: 'inline-flex', gap: 3, background: C.surfaceInset, borderRadius: 10, padding: 3 }}>
          {Object.keys(TF).map((k) => (
            <button key={k} onClick={() => setTf(k)} className="num" style={{ height: 28, padding: '0 14px', borderRadius: 7, fontSize: 12.5, fontWeight: 600, color: k === tf ? '#fff' : C.text2, background: k === tf ? C.surface3 : 'transparent' }}>
              {k}
            </button>
          ))}
        </div>
      </IOSCard>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 12 }}>
        {[
          ['Sharpe', PORTFOLIO.sharpe.toFixed(2)],
          ['Max DD', fmtPct(PORTFOLIO.maxDd)],
          ['Win rate', fmtPct(PORTFOLIO.winRate, 0)],
        ].map(([l, v]) => (
          <IOSCard key={l} pad={13}>
            <div className="eyebrow">{l}</div>
            <div className="num" style={{ fontSize: 19, fontWeight: 700, marginTop: 4 }}>{v}</div>
          </IOSCard>
        ))}
      </div>

      <IOSCard pad={16} style={{ marginTop: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ fontWeight: 650, fontSize: 15 }}>Market regime</span>
          <span style={{ fontSize: 13, color: C.regimeUp, fontWeight: 600 }}>{REGIME_LABEL[REGIME.label]} · {fmtPct(REGIME.probabilities.trending_up, 0)}</span>
        </div>
        <RegimeBar probs={REGIME.probabilities} height={10} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 11.5, color: C.text3 }}>
          {(Object.keys(REGIME.probabilities) as (keyof typeof REGIME.probabilities)[]).map((k) => (
            <span key={k}>{REGIME_LABEL[k].replace(' ', ' ')} {fmtPct(REGIME.probabilities[k], 0)}</span>
          ))}
        </div>
      </IOSCard>

      <SectionHeader title="Top ideas" action="All" onAction={() => setTab('ideas')} />
      <IOSCard pad={0}>
        {ideas.map((idea, i) => (
          <button
            key={idea.symbol}
            onClick={() => push({ type: 'idea', id: idea.symbol })}
            style={{ display: 'flex', alignItems: 'center', gap: 12, width: '100%', padding: '13px 15px', borderTop: i ? '1px solid var(--border)' : 'none', textAlign: 'left' }}
          >
            <ActionPill action={idea.action} />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 650, fontSize: 15 }}>{idea.symbol}</div>
              <div className="mono dim" style={{ fontSize: 11.5 }}>{idea.strategy}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className={`num ${idea.targetWeight >= 0 ? 'pos' : 'neg'}`} style={{ fontWeight: 650, fontSize: 14 }}>{fmtPctSigned(idea.targetWeight)}</div>
              <div className="dim" style={{ fontSize: 11 }}>cal {idea.calibratedProbability?.toFixed(2)}</div>
            </div>
            <Icon name="chevronRight" size={16} />
          </button>
        ))}
      </IOSCard>

      <SectionHeader title="Movers" action="Markets" onAction={() => setTab('markets')} />
      <div style={{ display: 'flex', gap: 10, overflowX: 'auto', padding: '2px 2px 6px', margin: '0 -16px', paddingLeft: 16, paddingRight: 16 }}>
        {movers.map((s) => (
          <button key={s.symbol} onClick={() => push({ type: 'symbol', id: s.symbol })} style={{ flex: 'none', width: 138 }}>
            <IOSCard pad={13}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 650, fontSize: 14 }}>{s.symbol}</span>
              </div>
              <div style={{ margin: '8px 0 6px' }}>
                <Sparkline data={s.spark} width={112} height={32} />
              </div>
              <div className="num" style={{ fontSize: 13, fontWeight: 600 }}>{fmtPrice(s.price)}</div>
              <Delta value={s.change1d} size={12} arrow={false} />
            </IOSCard>
          </button>
        ))}
      </div>
    </Screen>
  );
}
