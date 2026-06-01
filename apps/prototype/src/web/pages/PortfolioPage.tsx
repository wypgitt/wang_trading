import { Panel } from '../../components/ui/Panel';
import { AreaChart } from '../../components/charts/AreaChart';
import { MiniBars } from '../../components/charts/MiniBars';
import { Donut } from '../../components/ui/primitives';
import { useNav } from '../../nav';
import { AssetType, FACTOR_MODEL, PORTFOLIO } from '../../data/mock';
import { ASSET_TINT, C } from '../../lib/colors';
import { fmtCompact, fmtPct, fmtPctSigned, fmtPrice, sideLabel } from '../../lib/format';

export function PortfolioPage() {
  const { go } = useNav();
  const p = PORTFOLIO;

  const byClass = new Map<AssetType, number>();
  for (const pos of p.positions) byClass.set(pos.type, (byClass.get(pos.type) ?? 0) + Math.abs(pos.weight));
  const allocSegments = [...byClass.entries()].map(([k, v]) => ({ label: k, value: v, color: ASSET_TINT[k] }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Exposure + allocation */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 18, alignItems: 'stretch' }}>
        <Panel title="Exposure" subtitle="Gross, net, and the long/short split">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 18 }}>
            {[
              ['Gross', fmtPct(p.grossExposure, 0), C.text1],
              ['Net', fmtPct(p.netExposure, 0), p.netExposure >= 0 ? C.pos : C.neg],
              ['Long', fmtPct(p.longExposure, 0), C.pos],
              ['Short', fmtPct(p.shortExposure, 0), C.neg],
            ].map(([l, v, c]) => (
              <div key={l}>
                <div className="eyebrow">{l}</div>
                <div className="num" style={{ fontSize: 22, fontWeight: 700, marginTop: 4, color: c as string }}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', height: 14, borderRadius: 99, overflow: 'hidden', gap: 3 }}>
            <div style={{ width: `${(p.longExposure / p.grossExposure) * 100}%`, background: C.pos }} title="Long" />
            <div style={{ width: `${(p.shortExposure / p.grossExposure) * 100}%`, background: C.neg }} title="Short" />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 9, fontSize: 12, color: C.text3 }}>
            <span>◀ {fmtPct(p.longExposure / p.grossExposure, 0)} long</span>
            <span>{fmtPct(p.shortExposure / p.grossExposure, 0)} short ▶</span>
          </div>
          <div className="divider" style={{ margin: '18px 0' }} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
            {[
              ['Sharpe', p.sharpe.toFixed(2)],
              ['Sortino', p.sortino.toFixed(2)],
              ['Max DD', fmtPct(p.maxDd)],
            ].map(([l, v]) => (
              <div key={l}>
                <div className="eyebrow">{l}</div>
                <div className="num" style={{ fontSize: 18, fontWeight: 680, marginTop: 4 }}>{v}</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Allocation by asset class" subtitle="Share of gross exposure">
          <div style={{ display: 'flex', gap: 22, alignItems: 'center' }}>
            <Donut
              segments={allocSegments}
              size={150}
              thickness={18}
              center={
                <div>
                  <div className="num" style={{ fontSize: 22, fontWeight: 720 }}>{fmtPct(p.grossExposure, 0)}</div>
                  <div className="eyebrow">gross</div>
                </div>
              }
            />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 11 }}>
              {allocSegments.map((s) => (
                <div key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span className="dot" style={{ background: s.color, width: 9, height: 9 }} />
                  <span style={{ fontSize: 13, textTransform: 'capitalize', flex: 1 }}>{s.label}</span>
                  <span className="num" style={{ fontWeight: 600, fontSize: 13 }}>{fmtPct(s.value, 0)}</span>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </div>

      {/* Positions */}
      <Panel title={`Positions (${p.positions.length})`} subtitle="Live book · marked to last price" pad={14}>
        <div style={{ overflowX: 'auto' }}>
          <table className="tbl">
            <thead>
              <tr>
                <th>Symbol</th>
                <th style={{ textAlign: 'left' }}>Side</th>
                <th style={{ textAlign: 'left' }}>Strategy</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Mark</th>
                <th>Weight</th>
                <th>Notional</th>
                <th>Unreal. P&L</th>
                <th>Day</th>
              </tr>
            </thead>
            <tbody>
              {p.positions.map((pos) => (
                <tr key={pos.symbol} className="clickable" onClick={() => go('symbol', pos.symbol)}>
                  <td style={{ fontWeight: 650 }}>{pos.symbol}</td>
                  <td style={{ textAlign: 'left' }}>
                    <span className={`pill ${pos.side > 0 ? 'pill-buy' : 'pill-sell'}`}>{sideLabel(pos.side)}</span>
                  </td>
                  <td className="mono muted" style={{ textAlign: 'left', fontSize: 12 }}>{pos.strategy}</td>
                  <td className="num">{Math.abs(pos.qty).toLocaleString()}</td>
                  <td className="num muted">{fmtPrice(pos.entryPrice)}</td>
                  <td className="num">{fmtPrice(pos.markPrice)}</td>
                  <td className="num">{fmtPctSigned(pos.weight)}</td>
                  <td className="num muted">{fmtCompact(pos.notional)}</td>
                  <td className={`num ${pos.unrealizedPnl >= 0 ? 'pos' : 'neg'}`} style={{ fontWeight: 600 }}>
                    {pos.unrealizedPnl >= 0 ? '+' : ''}{fmtCompact(pos.unrealizedPnl)} <span style={{ fontSize: 11, opacity: 0.8 }}>({fmtPctSigned(pos.unrealizedPct)})</span>
                  </td>
                  <td className={`num ${pos.dayPnl >= 0 ? 'pos' : 'neg'}`}>{pos.dayPnl >= 0 ? '+' : ''}{fmtCompact(pos.dayPnl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Factor risk + drawdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, alignItems: 'start' }}>
        <Panel title="Factor risk model" subtitle="PCA decomposition of portfolio variance">
          <div style={{ display: 'flex', gap: 22, alignItems: 'center', marginBottom: 18 }}>
            <Donut
              segments={[
                { label: 'Systematic', value: FACTOR_MODEL.systematicPct, color: C.accent },
                { label: 'Idiosyncratic', value: FACTOR_MODEL.idiosyncraticPct, color: C.surface3 },
              ]}
              size={132}
              thickness={16}
              center={
                <div>
                  <div className="num" style={{ fontSize: 18, fontWeight: 720 }}>{fmtPct(FACTOR_MODEL.totalRisk, 1)}</div>
                  <div className="eyebrow">ann. risk</div>
                </div>
              }
            />
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 8 }}>
                <span className="dot" style={{ background: C.accent }} /> Systematic <span className="num" style={{ marginLeft: 'auto', fontWeight: 600 }}>{fmtPct(FACTOR_MODEL.systematicPct, 0)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <span className="dot" style={{ background: C.surface3 }} /> Idiosyncratic <span className="num" style={{ marginLeft: 'auto', fontWeight: 600 }}>{fmtPct(FACTOR_MODEL.idiosyncraticPct, 0)}</span>
              </div>
            </div>
          </div>
          <div className="eyebrow" style={{ marginBottom: 12 }}>Factor exposures</div>
          <MiniBars items={FACTOR_MODEL.factors.map((f) => ({ label: f.name, value: f.exposure, sub: f.exposure.toFixed(2) }))} signed labelWidth={130} />
        </Panel>

        <Panel title="Drawdown" subtitle="Underwater equity from running peak">
          <AreaChart data={p.drawdownHistory.map((d) => ({ t: d.t, v: d.v }))} height={236} color={C.neg} showTooltip showYAxis valueFmt={(v) => fmtPct(v, 0)} baseline={0} />
        </Panel>
      </div>
    </div>
  );
}
