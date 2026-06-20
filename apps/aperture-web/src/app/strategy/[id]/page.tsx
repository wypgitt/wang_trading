'use client';
import Link from 'next/link';
import { Panel } from '@/components/ui/Panel';
import { ActionPill, AssetGlyph, StatusDot } from '@/components/ui/primitives';
import { ComingState } from '@/components/ui/honesty';
import { Icon } from '@/components/Icon';
import { getStrategy } from '@/data/api';
import { TradeIdea } from '@/data/mock';
import { familyReadiness } from '@/lib/readiness';
import { CAT, REGIME_HEX, REGIME_LABEL } from '@/lib/colors';
import { useChartColors } from '@/lib/theme';
import { fmtPctSigned, fmtProb } from '@/lib/format';

export default function Page({ params }: { params: { id: string } }) {
  const C = useChartColors();
  const { strategy: s, ideas } = getStrategy(params.id).data;
  const catColor = CAT[s.category] ?? C.accent;
  // Data-driven status: a family is Active only when the deployed single-symbol
  // bars path can fire its generator (src/signal_battery/orchestrator.py). Never
  // the fabricated live/shadow/paused pill — that lied about every family.
  const fam = familyReadiness(s.id);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* 1 — Header */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Link href="/strategies" className="dim" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5, width: 'fit-content' }}>
          <Icon name="arrowLeft" size={14} /> Strategies
        </Link>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <span className="dot" style={{ background: catColor, width: 9, height: 9 }} title={s.category} />
          <h2 style={{ fontSize: 22, fontWeight: 720, letterSpacing: '-0.02em' }}>{s.name}</h2>
          <span className="chip" style={{ height: 24, gap: 6 }} title={fam.active ? undefined : fam.reason}>
            <StatusDot ok={fam.active} live={fam.active} />
            {fam.active ? 'Active' : `Inactive — ${fam.reason ?? 'no live feed wired'}`}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span className="chip" style={{ height: 24, borderColor: `${catColor}40` }}>
            <span className="dot" style={{ background: catColor }} />
            {s.category}
          </span>
          <span className="muted" style={{ fontSize: 12.5 }}>source · {s.source}</span>
          <span className="muted mono" style={{ fontSize: 12 }}>{s.id}</span>
          <div style={{ flex: 1 }} />
          {s.assetClasses.map((a) => (
            <span key={a} className="pill pill-neutral" style={{ textTransform: 'capitalize' }}>{a}</span>
          ))}
        </div>
      </div>

      {/* 2 — Thesis [LIVE] */}
      <Panel>
        <div className="eyebrow" style={{ marginBottom: 8 }}>Thesis</div>
        <p style={{ fontSize: 15, lineHeight: 1.62, color: C.text1, maxWidth: 880 }}>{s.thesis}</p>
      </Panel>

      {/* 3 — Stat strip → COMING (wave 5) */}
      <Panel title="Performance" subtitle="Sharpe · Win rate · P&L share · YTD · Allocation · Avg hold">
        <ComingState
          title="Per-strategy performance — coming when backtest metrics are persisted"
          unlock="Per-strategy performance — coming when backtest metrics are persisted (retrain gate broken, retrain_pipeline.py:265)."
          wave={5}
          compact
          ghost={<GhostStats />}
        />
      </Panel>

      {/* 4 — Equity (COMING ghost, wave 5) | Regime-fit (COMING, wave 6) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.55fr) minmax(0,1fr)', gap: 16, alignItems: 'start' }}>
        <Panel title="Strategy equity" subtitle="Cumulative sleeve P&L, rebased to 100">
          <ComingState
            title="Strategy equity — coming with persisted backtest runs"
            unlock="Strategy equity — coming when per-strategy backtest runs are persisted."
            wave={5}
            ghost={<GhostCurve />}
          />
        </Panel>
        <Panel title="Regime fit" subtitle="Expected edge by market regime">
          <RegimeFitColumn />
        </Panel>
      </div>

      {/* 5 — Parameters [LIVE] | 6 — Active ideas [LIVE] */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1.55fr)', gap: 16, alignItems: 'start' }}>
        <Panel title="Parameters" subtitle="Live configuration">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {s.params.map((p, i) => (
              <div
                key={p.key}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 16, padding: '11px 2px', borderTop: i ? '1px solid var(--border)' : 'none' }}
              >
                <span className="mono" style={{ fontSize: 12.5, color: C.text2 }}>{p.key}</span>
                <span className="mono" style={{ fontSize: 12.5, fontWeight: 600, textAlign: 'right' }}>{p.value}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel
          title="Active ideas from this family"
          subtitle="Current cycle — tap a row for the full decision chain"
          action={ideas.length > 0 ? <span className="chip" style={{ height: 24 }}><span className="num">{ideas.length}</span></span> : undefined}
        >
          {ideas.length === 0 ? (
            <div className="dim" style={{ padding: '24px 0', textAlign: 'center', fontSize: 13 }}>
              {fam.active
                ? 'No live ideas above the entry gate this cycle.'
                : 'No live ideas — context feed not wired.'}
            </div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>Instrument</th>
                  <th>Action</th>
                  <th style={{ textAlign: 'right' }}>Target wt</th>
                  <th style={{ textAlign: 'right' }}>Cal p</th>
                  <th style={{ width: 24 }}></th>
                </tr>
              </thead>
              <tbody>
                {ideas.map((idea) => (
                  <IdeaRow key={idea.symbol} idea={idea} />
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
    </div>
  );
}

function IdeaRow({ idea }: { idea: TradeIdea }) {
  const cellLink = `/symbols/${idea.symbol}`;
  const weighted = idea.targetWeight !== 0;
  return (
    <tr className="clickable">
      <td>
        <Link href={cellLink} style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
          <AssetGlyph sym={idea} size={30} />
          <div>
            <div className="mono" style={{ fontWeight: 650, fontSize: 13.5 }}>{idea.symbol}</div>
            <div className="dim" style={{ fontSize: 11 }}>{idea.reason.split('.')[0]}</div>
          </div>
        </Link>
      </td>
      <td><Link href={cellLink}><ActionPill action={idea.action} /></Link></td>
      <td style={{ textAlign: 'right' }}>
        <Link href={cellLink} className={`num ${weighted ? (idea.targetWeight > 0 ? 'pos' : 'neg') : 'dim'}`} style={{ fontWeight: 600 }}>
          {weighted ? fmtPctSigned(idea.targetWeight, 1) : '—'}
        </Link>
      </td>
      <td style={{ textAlign: 'right' }}>
        {/* calibratedProbability is a real persisted engine output per family */}
        <Link href={cellLink} className="num muted">{fmtProb(idea.calibratedProbability)}</Link>
      </td>
      <td style={{ textAlign: 'right' }}>
        <Link href={cellLink} className="dim" style={{ display: 'inline-flex' }}>
          <Icon name="chevronRight" size={15} />
        </Link>
      </td>
    </tr>
  );
}

// ---- Wireframe ghosts (low-opacity hints of the eventual layout) -----------
function GhostStats() {
  const C = useChartColors();
  return (
    <div style={{ display: 'flex', gap: 24 }}>
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ width: 38, height: 8, borderRadius: 3, background: C.text2 }} />
          <div style={{ width: 54, height: 18, borderRadius: 4, background: C.text2 }} />
        </div>
      ))}
    </div>
  );
}
function GhostCurve() {
  const C = useChartColors();
  return (
    <svg width={420} height={120} viewBox="0 0 420 120" fill="none">
      <path d="M0 100 L60 84 L120 92 L180 58 L240 70 L300 38 L360 50 L420 20" stroke={C.text2} strokeWidth={2} />
    </svg>
  );
}
// Regime-fit (COMING · Wave 6). The four regime labels stay so the layout reads
// intentional, but each rail is UNFILLED — never a fabricated probability. The
// RegimeDetector has zero runtime callers, so no per-regime edge exists today.
const REGIME_KEYS = ['trending_up', 'trending_down', 'mean_reverting', 'high_volatility'] as const;
function RegimeFitColumn() {
  const C = useChartColors();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 11 }}>
        {REGIME_KEYS.map((k) => (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="dot" style={{ background: REGIME_HEX[k], opacity: 0.5 }} />
            <span style={{ fontSize: 12.5, color: C.text2, width: 92 }}>{REGIME_LABEL[k]}</span>
            {/* Unfilled rail — empty on purpose; a value would be invented. */}
            <div style={{ flex: 1, height: 8, borderRadius: 99, background: C.surfaceInset, border: `1px solid ${C.border}` }} />
          </div>
        ))}
      </div>
      <div
        className="muted"
        style={{ fontSize: 12, lineHeight: 1.55, display: 'flex', alignItems: 'center', gap: 7, color: C.text3 }}
      >
        <Icon name="settings" size={14} />
        coming when the regime detector is wired into the live cycle · Wave 6
      </div>
    </div>
  );
}
