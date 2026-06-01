import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Panel } from '../../components/ui/Panel';
import { Icon } from '../../components/Icon';
import { BACKTEST } from '../../data/mock';
import { C, REGIME_HEX, REGIME_LABEL } from '../../lib/colors';
import { fmtPct, fmtPctSigned } from '../../lib/format';

function heatColor(v: number | null): string {
  if (v == null) return 'transparent';
  const a = Math.min(0.55, Math.abs(v) * 13);
  return v >= 0 ? `rgba(30,203,139,${a})` : `rgba(246,70,93,${a})`;
}

const MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];

export function ResearchPage() {
  const bt = BACKTEST;
  const m = bt.metrics;
  const chartData = bt.equityCurve.map((p, i) => ({ t: i, strat: p.v, bench: bt.benchmark[i]?.v }));

  const metrics: [string, string][] = [
    ['Total return', fmtPctSigned(m.totalReturn)],
    ['Ann. return', fmtPctSigned(m.annReturn)],
    ['Sharpe', m.sharpe.toFixed(2)],
    ['Sortino', m.sortino.toFixed(2)],
    ['Max DD', fmtPct(m.maxDd)],
    ['Win rate', fmtPct(m.winRate, 1)],
    ['Profit factor', m.profitFactor.toFixed(2)],
    ['Turnover', m.turnover.toFixed(1) + '×'],
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Metrics strip */}
      <div className="card" style={{ padding: 0, display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)' }}>
        {metrics.map(([l, v], i) => (
          <div key={l} style={{ padding: '14px 16px', borderLeft: i ? '1px solid var(--border)' : 'none' }}>
            <div className="eyebrow">{l}</div>
            <div className="num" style={{ fontSize: 18, fontWeight: 680, marginTop: 5 }}>{v}</div>
          </div>
        ))}
      </div>

      {/* Equity vs benchmark */}
      <Panel title="Walk-forward equity curve" subtitle={bt.name + ' · vs SPX benchmark'}>
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={C.accent} stopOpacity={0.32} />
                <stop offset="92%" stopColor={C.accent} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="t" hide />
            <YAxis orientation="right" width={48} tick={{ fill: C.text3, fontSize: 11 }} axisLine={false} tickLine={false} domain={['dataMin', 'dataMax']} />
            <Tooltip
              contentStyle={{ background: C.surface3, border: `1px solid ${C.borderStrong}`, borderRadius: 9, fontSize: 12.5 }}
              labelStyle={{ display: 'none' }}
              itemStyle={{ padding: 0 }}
            />
            <Area type="monotone" dataKey="strat" name="Strategy" stroke={C.accent} strokeWidth={2} fill="url(#eqg)" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="bench" name="Benchmark" stroke={C.text3} strokeWidth={1.6} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: 18, marginTop: 6 }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 12.5 }}><span style={{ width: 14, height: 3, background: C.accent, borderRadius: 2 }} /> Strategy</span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 12.5, color: C.text2 }}><span style={{ width: 14, height: 3, background: C.text3, borderRadius: 2 }} /> SPX benchmark</span>
        </div>
      </Panel>

      {/* Validation gates */}
      <div>
        <div className="eyebrow" style={{ marginBottom: 12 }}>Promotion gates · all three must pass</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
          {bt.gates.map((g) => (
            <div key={g.name} className="card" style={{ padding: 18 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 650, fontSize: 14.5 }}>{g.name}</span>
                <span className={`pill ${g.pass ? 'pill-buy' : 'pill-sell'}`}>
                  <Icon name={g.pass ? 'check' : 'close'} size={12} /> {g.pass ? 'Pass' : 'Fail'}
                </span>
              </div>
              <div className="num" style={{ fontSize: 28, fontWeight: 720, marginTop: 12, color: g.pass ? C.pos : C.neg }}>
                {g.name === 'CPCV' ? fmtPct(g.value, 0) : g.value < 0.1 ? g.value.toFixed(3) : g.value.toFixed(2)}
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 7, lineHeight: 1.5, minHeight: 36 }}>{g.detail}</div>
              {g.name === 'CPCV' && (
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 38, marginTop: 8 }}>
                  {bt.cpcvPaths.map((v, i) => (
                    <div key={i} style={{ flex: 1, height: `${Math.min(100, Math.abs(v) * 120 + 8)}%`, background: v >= 0 ? C.pos : C.neg, opacity: 0.75, borderRadius: 1 }} title={v.toFixed(2)} />
                  ))}
                </div>
              )}
              {g.name !== 'CPCV' && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ height: 8, background: C.surfaceInset, borderRadius: 99, position: 'relative', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${(g.value / g.threshold) * 100}%`, background: g.pass ? C.pos : C.neg, borderRadius: 99 }} />
                    <div style={{ position: 'absolute', left: '100%', top: -3, bottom: -3, width: 2, background: C.text2, transform: 'translateX(-2px)' }} />
                  </div>
                  <div className="dim" style={{ fontSize: 11, marginTop: 5 }}>threshold {g.threshold}</div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Monthly heatmap + regime */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 18, alignItems: 'start' }}>
        <Panel title="Monthly returns" subtitle="Net of costs">
          <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 3 }}>
            <thead>
              <tr>
                <th className="eyebrow" style={{ textAlign: 'left', paddingBottom: 6 }}>Year</th>
                {MONTHS.map((mo, i) => (
                  <th key={i} className="eyebrow" style={{ textAlign: 'center', fontSize: 10 }}>{mo}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bt.monthlyReturns.map((row) => (
                <tr key={row.year}>
                  <td className="num" style={{ fontSize: 12.5, fontWeight: 600, paddingRight: 8 }}>{row.year}</td>
                  {row.months.map((v, i) => (
                    <td
                      key={i}
                      className="num"
                      title={v == null ? '' : fmtPctSigned(v)}
                      style={{ textAlign: 'center', fontSize: 10.5, height: 30, borderRadius: 6, background: heatColor(v), color: v == null ? C.text3 : C.text1 }}
                    >
                      {v == null ? '' : (v * 100).toFixed(1)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel title="Performance by regime" subtitle="Conditional Sharpe">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 2 }}>
            {bt.regimeBreakdown.map((r) => (
              <div key={r.regime}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 6 }}>
                  <span style={{ color: C.text2, display: 'inline-flex', alignItems: 'center', gap: 7 }}>
                    <span className="dot" style={{ background: REGIME_HEX[r.regime] }} />
                    {REGIME_LABEL[r.regime]}
                  </span>
                  <span className="num">Sharpe {r.sharpe.toFixed(2)} · {r.trades} trades</span>
                </div>
                <div style={{ height: 8, background: C.surfaceInset, borderRadius: 99, overflow: 'hidden' }}>
                  <div style={{ width: `${(r.sharpe / 2.5) * 100}%`, height: '100%', background: REGIME_HEX[r.regime], borderRadius: 99 }} />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* Trade log */}
      <Panel title="Recent trade log" subtitle="Sampled from the walk-forward" pad={14}>
        <div style={{ overflowX: 'auto' }}>
          <table className="tbl">
            <thead>
              <tr>
                <th>Symbol</th>
                <th style={{ textAlign: 'left' }}>Strategy</th>
                <th>Side</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Hold</th>
                <th>Meta p</th>
                <th>Return</th>
                <th>Net P&L</th>
              </tr>
            </thead>
            <tbody>
              {bt.tradeLog.map((t, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 650 }}>{t.symbol}</td>
                  <td className="mono muted" style={{ textAlign: 'left', fontSize: 12 }}>{t.family}</td>
                  <td className="num">{t.side > 0 ? 'Long' : 'Short'}</td>
                  <td className="num muted">{t.entry.toFixed(2)}</td>
                  <td className="num muted">{t.exit.toFixed(2)}</td>
                  <td className="num muted">{t.holdBars} bars</td>
                  <td className="num">{t.metaProb.toFixed(2)}</td>
                  <td className={`num ${t.returnPct >= 0 ? 'pos' : 'neg'}`}>{fmtPctSigned(t.returnPct)}</td>
                  <td className={`num ${t.netPnl >= 0 ? 'pos' : 'neg'}`} style={{ fontWeight: 600 }}>{t.netPnl >= 0 ? '+' : ''}${Math.abs(t.netPnl).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
