import { useState } from 'react';
import { Panel } from '../../components/ui/Panel';
import { AreaChart } from '../../components/charts/AreaChart';
import { Sparkline } from '../../components/charts/Sparkline';
import { MiniBars } from '../../components/charts/MiniBars';
import { ActionPill, Bar, Delta, RegimeBar } from '../../components/ui/primitives';
import { Icon } from '../../components/Icon';
import { useNav } from '../../nav';
import {
  IDEA_TOTALS,
  PORTFOLIO,
  REGIME,
  STRATEGIES,
  SYMBOLS,
  TRADE_IDEAS,
} from '../../data/mock';
import { CAT, C, REGIME_HEX, REGIME_LABEL } from '../../lib/colors';
import { fmtCompact, fmtPct, fmtPctSigned, fmtPrice } from '../../lib/format';

const TF: Record<string, number> = { '1W': 7, '1M': 22, '3M': 66, '6M': 132, All: 180 };

export function OverviewPage() {
  const { go } = useNav();
  const [tf, setTf] = useState('3M');
  const slice = PORTFOLIO.navHistory.slice(-TF[tf]);
  const eq = slice.map((p, i) => ({ t: i, v: p.v }));
  const net = eq[eq.length - 1].v / eq[0].v - 1;
  const col = net >= 0 ? C.pos : C.neg;

  const ideas = TRADE_IDEAS.filter((i) => i.action === 'BUY' || i.action === 'SELL').slice(0, 5);
  const movers = [...SYMBOLS].sort((a, b) => Math.abs(b.change1d) - Math.abs(a.change1d)).slice(0, 5);
  const contrib = [...STRATEGIES]
    .sort((a, b) => b.contributionPct - a.contributionPct)
    .slice(0, 6)
    .map((s) => ({ label: s.name, value: s.contributionPct, color: CAT[s.category], sub: fmtPct(s.contributionPct, 0) }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Hero */}
      <div className="card" style={{ padding: 22, position: 'relative', overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div className="eyebrow">Net asset value · paper</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginTop: 6 }}>
              <span className="num" style={{ fontSize: 40, fontWeight: 720, letterSpacing: '-0.03em' }}>
                {fmtCompact(PORTFOLIO.nav)}
              </span>
              <span className={`num ${PORTFOLIO.dailyPnl >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 16, fontWeight: 650 }}>
                {PORTFOLIO.dailyPnl >= 0 ? '▲' : '▼'} {fmtCompact(Math.abs(PORTFOLIO.dailyPnl))} ({fmtPctSigned(PORTFOLIO.dailyPnlPct)}) today
              </span>
            </div>
            <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>
              {fmtPctSigned(net)} over {tf} · since inception {fmtPctSigned(PORTFOLIO.cumPnlPct)}
            </div>
          </div>
          <div style={{ display: 'inline-flex', gap: 4, background: C.surfaceInset, borderRadius: 11, padding: 3 }}>
            {Object.keys(TF).map((k) => (
              <button
                key={k}
                onClick={() => setTf(k)}
                className="num"
                style={{
                  height: 30,
                  padding: '0 13px',
                  borderRadius: 8,
                  fontSize: 12.5,
                  fontWeight: 600,
                  color: k === tf ? '#fff' : 'var(--text-2)',
                  background: k === tf ? 'var(--surface-3)' : 'transparent',
                }}
              >
                {k}
              </button>
            ))}
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <AreaChart data={eq} color={col} height={216} showTooltip valueFmt={(v) => fmtCompact(v)} />
        </div>
      </div>

      {/* Stat strip */}
      <div className="card" style={{ padding: 0, display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)' }}>
        {[
          ['Sharpe', PORTFOLIO.sharpe.toFixed(2), 'Sortino ' + PORTFOLIO.sortino.toFixed(2)],
          ['Max drawdown', fmtPct(PORTFOLIO.maxDd), 'Calmar ' + PORTFOLIO.calmar.toFixed(2)],
          ['Ann. volatility', fmtPct(PORTFOLIO.volAnn), 'Beta ' + PORTFOLIO.beta.toFixed(2)],
          ['Win rate', fmtPct(PORTFOLIO.winRate, 1), 'PF ' + PORTFOLIO.profitFactor.toFixed(2)],
          ['Gross exposure', fmtPct(PORTFOLIO.grossExposure, 0), 'Net ' + fmtPct(PORTFOLIO.netExposure, 0)],
          ['Open positions', String(PORTFOLIO.positions.length), IDEA_TOTALS.buy + IDEA_TOTALS.sell + ' new ideas'],
        ].map(([label, value, sub], i) => (
          <div key={label} style={{ padding: '15px 18px', borderLeft: i ? '1px solid var(--border)' : 'none' }}>
            <div className="eyebrow">{label}</div>
            <div className="num" style={{ fontSize: 21, fontWeight: 680, marginTop: 5 }}>
              {value}
            </div>
            <div className="muted" style={{ fontSize: 11.5, marginTop: 2 }}>
              {sub}
            </div>
          </div>
        ))}
      </div>

      {/* Two-column */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.55fr 1fr', gap: 18, alignItems: 'start' }}>
        <Panel
          title="Top trade ideas"
          subtitle="Highest-conviction actions this cycle"
          action={
            <button className="chip" onClick={() => go('ideas')}>
              View all {IDEA_TOTALS.buy + IDEA_TOTALS.sell + IDEA_TOTALS.watch} <Icon name="chevronRight" size={14} />
            </button>
          }
        >
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {ideas.map((idea) => {
              const sym = SYMBOLS.find((s) => s.symbol === idea.symbol)!;
              return (
                <button
                  key={idea.symbol}
                  onClick={() => go('symbol', idea.symbol)}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '120px 1fr 84px 96px 70px',
                    alignItems: 'center',
                    gap: 12,
                    padding: '11px 8px',
                    borderTop: '1px solid var(--border)',
                    textAlign: 'left',
                  }}
                  className="row-hover"
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                    <ActionPill action={idea.action} />
                    <span style={{ fontWeight: 650, fontSize: 13.5 }}>{idea.symbol}</span>
                  </div>
                  <div className="muted" style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {idea.strategy}
                  </div>
                  <Sparkline data={sym.spark} width={84} height={26} />
                  <div style={{ textAlign: 'right' }}>
                    <div className="num" style={{ fontSize: 13, fontWeight: 600 }}>
                      {fmtPctSigned(idea.targetWeight)}
                    </div>
                    <div className="dim" style={{ fontSize: 11 }}>
                      target
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div className="num" style={{ fontSize: 13, fontWeight: 600, color: (idea.calibratedProbability ?? 0) >= 0.6 ? C.pos : C.text1 }}>
                      {idea.calibratedProbability?.toFixed(2)}
                    </div>
                    <div className="dim" style={{ fontSize: 11 }}>
                      cal. p
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </Panel>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <Panel title="Market regime" subtitle="LSTM detector">
            <RegimeBar probs={REGIME.probabilities} height={10} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9, marginTop: 14 }}>
              {(Object.keys(REGIME.probabilities) as (keyof typeof REGIME.probabilities)[]).map((k) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ width: 96, fontSize: 12.5, color: C.text2 }}>{REGIME_LABEL[k]}</span>
                  <div style={{ flex: 1 }}>
                    <Bar value={REGIME.probabilities[k]} color={REGIME_HEX[k]} />
                  </div>
                  <span className="num" style={{ width: 38, textAlign: 'right', fontSize: 12.5, fontWeight: 600 }}>
                    {fmtPct(REGIME.probabilities[k], 0)}
                  </span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="P&L contribution" subtitle="Share of portfolio return by strategy">
            <MiniBars items={contrib} labelWidth={150} />
          </Panel>
        </div>
      </div>

      {/* Movers */}
      <Panel title="Market movers" action={<button className="chip" onClick={() => go('markets')}>Markets <Icon name="chevronRight" size={14} /></button>}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
          {movers.map((s) => (
            <button
              key={s.symbol}
              onClick={() => go('symbol', s.symbol)}
              className="row-hover"
              style={{ textAlign: 'left', padding: 13, borderRadius: 13, border: '1px solid var(--border)', background: 'var(--surface-1)' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 650, fontSize: 13.5 }}>{s.symbol}</span>
                <Delta value={s.change1d} size={12.5} arrow={false} />
              </div>
              <div style={{ margin: '8px 0' }}>
                <Sparkline data={s.spark} width={150} height={34} />
              </div>
              <div className="num muted" style={{ fontSize: 12 }}>
                {fmtPrice(s.price)}
              </div>
            </button>
          ))}
        </div>
      </Panel>
    </div>
  );
}
