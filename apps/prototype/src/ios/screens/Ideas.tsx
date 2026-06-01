import { useState } from 'react';
import { ActionPill, Bar } from '../../components/ui/primitives';
import { Icon } from '../../components/Icon';
import { IOSCard, LargeTitle, Screen } from '../iosUi';
import { useIOS } from '../iosNav';
import { Action, IDEA_TOTALS, TRADE_IDEAS } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtBps, fmtPctSigned } from '../../lib/format';

const TABS: { id: Action | 'all'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'BUY', label: 'Buy' },
  { id: 'SELL', label: 'Sell' },
  { id: 'WATCH', label: 'Watch' },
];

export function Ideas() {
  const { push } = useIOS();
  const [tab, setTab] = useState<Action | 'all'>('all');
  const ideas = tab === 'all' ? TRADE_IDEAS : TRADE_IDEAS.filter((i) => i.action === tab);

  return (
    <Screen>
      <LargeTitle title="Trade Ideas" subtitle={`${TRADE_IDEAS.length} this cycle · ${fmtPctSigned(IDEA_TOTALS.netTargetWeight)} net`} />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 14 }}>
        {[
          ['Buy', IDEA_TOTALS.buy, C.pos],
          ['Sell', IDEA_TOTALS.sell, C.neg],
          ['Watch', IDEA_TOTALS.watch, C.info],
        ].map(([l, v, c]) => (
          <IOSCard key={l as string} pad={12}>
            <div className="eyebrow">{l as string}</div>
            <div className="num" style={{ fontSize: 22, fontWeight: 720, marginTop: 3, color: c as string }}>{v as number}</div>
          </IOSCard>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 8, overflowX: 'auto', margin: '0 -16px 14px', padding: '0 16px' }}>
        {TABS.map((t) => (
          <button key={t.id} className={`chip ${tab === t.id ? 'active' : ''}`} style={{ flex: 'none' }} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {ideas.map((idea) => (
          <button key={idea.symbol} className="ios-idea" onClick={() => push({ type: 'idea', id: idea.symbol })} style={{ textAlign: 'left' }}>
            <IOSCard pad={15}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <ActionPill action={idea.action} />
                <span style={{ fontWeight: 680, fontSize: 17 }}>{idea.symbol}</span>
                <span className="mono dim" style={{ fontSize: 12 }}>{idea.strategy}</span>
                <div style={{ flex: 1 }} />
                <Icon name="chevronRight" size={16} />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 13 }}>
                <div>
                  <div className="dim" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Cal. p</div>
                  <div className="num" style={{ fontSize: 16, fontWeight: 680, color: (idea.calibratedProbability ?? 0) >= 0.6 ? C.pos : C.text1 }}>{idea.calibratedProbability?.toFixed(2) ?? '—'}</div>
                </div>
                <div>
                  <div className="dim" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Target</div>
                  <div className={`num ${idea.targetWeight >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 16, fontWeight: 680 }}>{idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}</div>
                </div>
                <div>
                  <div className="dim" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Cost</div>
                  <div className="num" style={{ fontSize: 16, fontWeight: 680 }}>{idea.expectedCostBps ? fmtBps(idea.expectedCostBps) : '—'}</div>
                </div>
                <div style={{ flex: 1, minWidth: 40 }}>
                  <div className="dim" style={{ fontSize: 10.5, textAlign: 'right', marginBottom: 4 }}>regime fit {idea.regimeFitScore?.toFixed(2)}</div>
                  <Bar value={idea.regimeFitScore ?? 0} color={C.regimeUp} height={6} />
                </div>
              </div>
            </IOSCard>
          </button>
        ))}
      </div>
    </Screen>
  );
}
