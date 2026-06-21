'use client';
import { useMemo, useState } from 'react';
import Link from 'next/link';
import { Panel } from '@/components/ui/Panel';
import { StatusDot, Segmented } from '@/components/ui/primitives';
import { DataUnavailable } from '@/components/ui/honesty';
import { Icon } from '@/components/Icon';
import { getStrategies, getTradeIdeas, AssetType, Strategy } from '@/data/api';
import { Loaded, ViewProps, useEnvelope } from '@/data/useEnvelope';
import { CAT, ASSET_TINT } from '@/lib/colors';
import { useChartColors } from '@/lib/theme';
import { familyReadiness, familyCounts } from '@/lib/readiness';

// ---------------------------------------------------------------------------
// Strategies grid — PARTIAL per docs/data_readiness.md.
//
// LIVE today: the strategy identity/config (name, category, source, thesis,
// status, asset classes) + the live count of active ideas for this family,
// derived from getTradeIdeas() filtered by idea.strategy === strategy.id.
//
// NOT-YET-AVAILABLE: per-strategy Sharpe / win rate / P&L share / allocation /
// trades / avg-hold are NEVER persisted (backtest metrics aren't persisted; the
// retrain gate is broken — retrain_pipeline.py:265). Those slots render as
// DataUnavailable chips, never as fabricated numbers. We deliberately do NOT
// draw the equity sparkline as a real track record.
// ---------------------------------------------------------------------------

type CatFilter = 'All' | 'Momentum' | 'Mean Reversion' | 'Trend' | 'Volatility' | 'Carry' | 'Arbitrage';

const FILTERS: { value: CatFilter; label: string }[] = [
  { value: 'All', label: 'All' },
  { value: 'Momentum', label: 'Momentum' },
  { value: 'Mean Reversion', label: 'Mean Rev' },
  { value: 'Trend', label: 'Trend' },
  { value: 'Volatility', label: 'Volatility' },
  { value: 'Carry', label: 'Carry' },
  { value: 'Arbitrage', label: 'Arbitrage' },
];

const ASSET_LABEL: Record<AssetType, string> = {
  equity: 'Equity',
  index: 'Index',
  crypto: 'Crypto',
  future: 'Futures',
};

// The single shared unlock string for every not-yet-persisted performance metric.
const PERF_UNLOCK =
  'Per-strategy performance — coming when backtest metrics are persisted (retrain gate broken, retrain_pipeline.py:265).';

export default function StrategiesPage() {
  return <Loaded fetcher={getStrategies} View={StrategiesView} />;
}

function StrategiesView({ data }: ViewProps<Strategy[]>) {
  const C = useChartColors();
  const [cat, setCat] = useState<CatFilter>('All');
  const strategies = data;
  // Secondary accessor — the live active-idea counts. Fetched independently so
  // a slow/failed ideas call never blocks the strategy roster; guard the null
  // env (still loading) by treating ideas as empty until it resolves.
  const ideasEnv = useEnvelope(getTradeIdeas);
  const ideas = ideasEnv.env?.data ?? [];

  // LIVE: count of actionable (BUY/SELL) live ideas per family this cycle.
  const activeByFamily = useMemo(() => {
    const m: Record<string, number> = {};
    for (const idea of ideas) {
      if (idea.strategy == null) continue;
      if (idea.action !== 'BUY' && idea.action !== 'SELL') continue;
      m[idea.strategy] = (m[idea.strategy] ?? 0) + 1;
    }
    return m;
  }, [ideas]);

  const visible = useMemo(
    () => (cat === 'All' ? strategies : strategies.filter((s) => s.category === cat)),
    [strategies, cat],
  );

  // Active / inactive split is DATA-DRIVEN from FAMILY_READINESS (derived from the
  // engine's generator dispatch), not from a prototype live/shadow flag.
  const counts = familyCounts();
  const totalActive = Object.values(activeByFamily).reduce((a, n) => a + n, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        {/* a11y: the Segmented buttons live in a shared primitive we don't own here,
            so we expose group semantics + an accessible name on the wrapper. */}
        <div role="group" aria-label="Filter strategies by category">
          <Segmented options={FILTERS} value={cat} onChange={setCat} />
        </div>
        <div className="muted" style={{ fontSize: 13 }}>
          <span className="num">{counts.total}</span> families · <span className="num">{counts.active}</span> active ·{' '}
          <span className="num">{counts.inactive}</span> inactive · <span className="num">{totalActive}</span> active ideas
        </div>
      </div>

      {/* Honest framing: identity is real; realized performance lands with backtest persistence. */}
      <div
        className="card"
        style={{ padding: '11px 15px', display: 'flex', alignItems: 'center', gap: 11, borderColor: 'rgba(124,92,255,0.22)' }}
      >
        <span style={{ color: C.accent, display: 'inline-flex' }}>
          <Icon name="lock" size={14} />
        </span>
        <span className="muted" style={{ fontSize: 12.5, lineHeight: 1.5 }}>
          Strategy identity, config and live signal counts are real today. Per-family Sharpe, win rate, P&amp;L share and
          allocation are not yet persisted — they unlock with backtest metrics (Wave 5).
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(330px, 1fr))', gap: 14 }}>
        {visible.map((s) => (
          <StrategyCard key={s.id} s={s} activeIdeas={activeByFamily[s.id] ?? 0} />
        ))}
      </div>
    </div>
  );
}

function StrategyCard({ s, activeIdeas }: { s: Strategy; activeIdeas: number }) {
  const C = useChartColors();
  const catColor = CAT[s.category] ?? C.accent;
  // Active / dormant is DATA-DRIVEN from the engine's generator dispatch, NOT
  // from the prototype live/shadow flag. Dormant families never fire, so their
  // live idea count is naturally 0.
  const family = familyReadiness(s.id);
  const dormant = !family.active;

  return (
    <Link
      href={`/strategy/${s.id}`}
      className="card lift"
      style={{ padding: 17, display: 'flex', flexDirection: 'column', gap: 12 }}
    >
      {/* Header — category dot + name + status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="dot" style={{ background: catColor, flex: 'none' }} />
            <span style={{ fontWeight: 660, fontSize: 15, letterSpacing: '-0.01em' }}>{s.name}</span>
          </div>
          <div className="dim" style={{ fontSize: 11.5, marginTop: 4, marginLeft: 15 }}>
            {s.category} · {s.source}
          </div>
        </div>
        {family.active ? (
          <span
            style={{ display: 'inline-flex', alignItems: 'center', gap: 6, flex: 'none', fontSize: 12, color: C.text2 }}
          >
            <StatusDot ok live />
            Active
          </span>
        ) : (
          <span
            title={family.reason ?? 'no live feed wired'}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              flex: 'none',
              fontSize: 11,
              fontWeight: 600,
              color: C.text3,
              background: C.surfaceInset,
              border: `1px solid ${C.border}`,
              borderRadius: 99,
              padding: '3px 9px',
              cursor: 'help',
            }}
          >
            <span className="dot" style={{ background: C.text3, opacity: 0.7 }} />
            Inactive — no live feed
          </span>
        )}
      </div>

      {/* Asset-class chips — LIVE config */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {s.assetClasses.map((a) => {
          const tint = ASSET_TINT[a] ?? C.accent;
          return (
            <span
              key={a}
              className="num"
              style={{
                height: 21,
                padding: '0 8px',
                display: 'inline-flex',
                alignItems: 'center',
                borderRadius: 99,
                fontSize: 10.5,
                fontWeight: 600,
                letterSpacing: '0.02em',
                color: tint,
                background: `${tint}1a`,
                border: `1px solid ${tint}33`,
              }}
            >
              {ASSET_LABEL[a]}
            </span>
          );
        })}
      </div>

      {/* Thesis — clamped to ~3 lines */}
      <p
        className="muted"
        style={{
          fontSize: 12.5,
          lineHeight: 1.5,
          margin: 0,
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}
      >
        {s.thesis}
      </p>

      {/* Metric row — Active ideas is LIVE; the rest are honestly gated. */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, flexWrap: 'wrap', marginTop: 2 }}>
        <div>
          <div className="eyebrow">Active ideas</div>
          <div
            className="num"
            style={{
              fontSize: 18,
              fontWeight: 700,
              marginTop: 3,
              color: dormant || activeIdeas === 0 ? C.text3 : catColor,
              letterSpacing: '-0.02em',
            }}
          >
            {activeIdeas}
          </div>
        </div>
        <div style={{ flex: 1 }} />
        {/* The chip variant of DataUnavailable renders only a lock + "Coming"; we
            pair each with its metric label so the gated slots stay identifiable. */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {(['Sharpe', 'Win', 'P&L share', 'Alloc'] as const).map((label) => (
            <span key={label} style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
              <span className="eyebrow" style={{ color: C.text3 }}>{label}</span>
              <DataUnavailable unlock={PERF_UNLOCK} label={label} />
            </span>
          ))}
        </div>
      </div>

      {/* Calm footer — dormant families have no feed wired; active families show
          their real live idea count. */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 11,
          color: dormant ? C.text3 : C.text2,
          borderTop: `1px solid ${C.border}`,
          paddingTop: 9,
          marginTop: 1,
        }}
      >
        <span className="dot" style={{ background: dormant ? C.text3 : catColor, opacity: dormant ? 0.6 : 1 }} />
        {dormant
          ? 'No live ideas — feed not wired'
          : `${activeIdeas} active ${activeIdeas === 1 ? 'idea' : 'ideas'} this cycle`}
      </div>
    </Link>
  );
}
