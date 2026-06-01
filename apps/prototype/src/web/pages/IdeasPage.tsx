import { useState } from 'react';
import { Icon } from '../../components/Icon';
import { IdeaDrawer } from '../../components/IdeaDrawer';
import { ActionPill } from '../../components/ui/primitives';
import { useNav } from '../../nav';
import { Action, IDEA_TOTALS, PORTFOLIO, TRADE_IDEAS, TradeIdea } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtBps, fmtCompact, fmtPct, fmtPctSigned } from '../../lib/format';

const TABS: { id: Action | 'all'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'BUY', label: 'Buy' },
  { id: 'SELL', label: 'Sell' },
  { id: 'WATCH', label: 'Watch' },
  { id: 'MODEL_REQUIRED', label: 'Model?' },
];

function Tile({ label, value, sub, color }: { label: string; value: string; sub: string; color?: string }) {
  return (
    <div className="card" style={{ padding: '14px 16px' }}>
      <div className="eyebrow">{label}</div>
      <div className="num" style={{ fontSize: 24, fontWeight: 700, marginTop: 5, color }}>
        {value}
      </div>
      <div className="muted" style={{ fontSize: 11.5, marginTop: 2 }}>
        {sub}
      </div>
    </div>
  );
}

export function IdeasPage() {
  const { go } = useNav();
  const [tab, setTab] = useState<Action | 'all'>('all');
  const [selected, setSelected] = useState<TradeIdea | null>(null);

  const ideas = tab === 'all' ? TRADE_IDEAS : TRADE_IDEAS.filter((i) => i.action === tab);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Summary tiles */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
        <Tile label="Buy" value={String(IDEA_TOTALS.buy)} sub="long candidates" color={C.pos} />
        <Tile label="Sell" value={String(IDEA_TOTALS.sell)} sub="short candidates" color={C.neg} />
        <Tile label="Watch" value={String(IDEA_TOTALS.watch)} sub="below entry gate" color={C.info} />
        <Tile label="Gross target" value={fmtPct(IDEA_TOTALS.grossTargetWeight, 0)} sub="of NAV" />
        <Tile label="Net target" value={fmtPctSigned(IDEA_TOTALS.netTargetWeight)} sub={`on ${fmtCompact(PORTFOLIO.nav)}`} />
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8 }}>
        {TABS.map((t) => (
          <button key={t.id} className={`chip ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="card" style={{ padding: '6px 6px 8px', overflowX: 'auto' }}>
        <table className="tbl">
          <thead>
            <tr>
              <th>Symbol</th>
              <th style={{ textAlign: 'left' }}>Action</th>
              <th style={{ textAlign: 'left' }}>Strategy</th>
              <th>Top conf</th>
              <th>Meta p</th>
              <th>Cal. p</th>
              <th>Regime fit</th>
              <th>Target wt</th>
              <th>Notional</th>
              <th>Cost</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {ideas.map((idea) => (
              <tr key={idea.symbol} className="clickable" onClick={() => setSelected(idea)}>
                <td style={{ fontWeight: 650 }}>{idea.symbol}</td>
                <td style={{ textAlign: 'left' }}>
                  <ActionPill action={idea.action} />
                </td>
                <td className="mono muted" style={{ textAlign: 'left', fontSize: 12 }}>
                  {idea.strategy ?? '—'}
                </td>
                <td className="num">{idea.topSignalConfidence?.toFixed(2) ?? '—'}</td>
                <td className="num">{idea.metaProbability?.toFixed(2) ?? '—'}</td>
                <td className="num" style={{ fontWeight: 600, color: (idea.calibratedProbability ?? 0) >= 0.6 ? C.pos : C.text1 }}>
                  {idea.calibratedProbability?.toFixed(2) ?? '—'}
                </td>
                <td className="num">{idea.regimeFitScore?.toFixed(2) ?? '—'}</td>
                <td className={`num ${idea.targetWeight > 0 ? 'pos' : idea.targetWeight < 0 ? 'neg' : 'dim'}`} style={{ fontWeight: 600 }}>
                  {idea.targetWeight ? fmtPctSigned(idea.targetWeight) : '—'}
                </td>
                <td className="num muted">{idea.targetNotional ? fmtCompact(idea.targetNotional) : '—'}</td>
                <td className="num muted">{idea.expectedCostBps ? fmtBps(idea.expectedCostBps) : '—'}</td>
                <td style={{ color: C.text3 }}>
                  <Icon name="chevronRight" size={15} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && <IdeaDrawer idea={selected} onClose={() => setSelected(null)} onOpenSymbol={(s) => { setSelected(null); go('symbol', s); }} />}
    </div>
  );
}
