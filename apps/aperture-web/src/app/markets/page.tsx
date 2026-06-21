'use client';
import { useMemo, useState } from 'react';
import Link from 'next/link';
import { Panel } from '@/components/ui/Panel';
import { AssetGlyph, Delta, Segmented } from '@/components/ui/primitives';
import { Sparkline } from '@/components/charts/Sparkline';
import { Icon } from '@/components/Icon';
import { getMarkets, AssetType, Sym } from '@/data/api';
import { deriveTrust } from '@/data/envelope';
import { Loaded, ViewProps } from '@/data/useEnvelope';
import { useChartColors } from '@/lib/theme';
import { fmtCompact, fmtPrice, fmtTimeAgo } from '@/lib/format';

type Filter = 'all' | AssetType;
const FILTERS: { value: Filter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'equity', label: 'Equities' },
  { value: 'index', label: 'Indexes' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'future', label: 'Futures' },
];

// Filter labels for the empty-state copy ("No {label} in the universe yet.").
const FILTER_LABEL: Record<Filter, string> = {
  all: 'instruments',
  equity: 'equities',
  index: 'indexes',
  crypto: 'crypto',
  future: 'futures',
};

// Sortable numeric columns. The default (key === null) is the honest baseline:
// active-idea rows pinned first, then abs(1D change) desc.
type SortKey = 'price' | 'change1d' | 'change1w' | 'change1m' | 'volume';
type SortDir = 'asc' | 'desc';
interface SortState {
  key: SortKey | null;
  dir: SortDir;
}

// Null-safe accessors: a missing value sorts to the bottom (live BFF nulls
// calendar changes / marketCap, and prices/volume for priceless rows). Never a
// fake 0 that would rank a no-data row alongside a true zero.
const SORT_GET: Record<SortKey, (s: Sym) => number> = {
  price: (s) => s.price ?? -Infinity,
  change1d: (s) => s.change1d ?? -Infinity,
  change1w: (s) => s.change1w ?? -Infinity,
  change1m: (s) => s.change1m ?? -Infinity,
  volume: (s) => s.volume ?? -Infinity,
};

function ariaSortFor(state: SortState, key: SortKey): 'ascending' | 'descending' | 'none' {
  if (state.key !== key) return 'none';
  return state.dir === 'asc' ? 'ascending' : 'descending';
}

// A sortable numeric header: a button that toggles direction (default desc) and
// carries aria-sort on its <th>. Numerics are right-aligned to match the cells.
function SortHeader({
  label,
  sortKey,
  sort,
  onSort,
  style,
}: {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (k: SortKey) => void;
  style?: React.CSSProperties;
}) {
  const C = useChartColors();
  const active = sort.key === sortKey;
  return (
    <th aria-sort={ariaSortFor(sort, sortKey)} style={style}>
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        title={`Sort by ${label}`}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'flex-end',
          gap: 4,
          width: '100%',
          font: 'inherit',
          letterSpacing: 'inherit',
          textTransform: 'inherit',
          color: active ? C.text1 : 'inherit',
          cursor: 'pointer',
        }}
      >
        {label}
        <span style={{ width: 9, display: 'inline-flex', justifyContent: 'center', color: active ? C.accent : C.text3 }}>
          {active ? (sort.dir === 'asc' ? '▲' : '▼') : ''}
        </span>
      </button>
    </th>
  );
}

export default function MarketsPage() {
  return <Loaded fetcher={getMarkets} View={MarketsView} />;
}

function MarketsView({ data, env }: ViewProps<Sym[]>) {
  const C = useChartColors();
  const [filter, setFilter] = useState<Filter>('all');
  const [sort, setSort] = useState<SortState>({ key: null, dir: 'desc' });

  // Full envelope rides the real trust contract for freshness/staleness.
  const all = data;
  const trust = deriveTrust(env);

  const onSort = (key: SortKey) =>
    setSort((prev) => (prev.key === key ? { key, dir: prev.dir === 'desc' ? 'asc' : 'desc' } : { key, dir: 'desc' }));

  const filtered = useMemo(
    () => (filter === 'all' ? all : all.filter((s) => s.type === filter)),
    [all, filter],
  );

  // Pure client sort over loaded rows. Default: idea rows pinned, then |1D| desc.
  const rows = useMemo(() => {
    const list = [...filtered];
    if (sort.key === null) {
      list.sort((a, b) => {
        if (a.hasIdea !== b.hasIdea) return a.hasIdea ? -1 : 1;
        // Null 1D change (no producer / priceless row) sinks below real movers.
        return Math.abs(b.change1d ?? 0) - Math.abs(a.change1d ?? 0);
      });
      return list;
    }
    const get = SORT_GET[sort.key];
    const mul = sort.dir === 'desc' ? -1 : 1;
    list.sort((a, b) => mul * (get(a) - get(b)));
    return list;
  }, [filtered, sort]);

  const withIdeas = all.filter((s) => s.hasIdea).length;
  const filterLabel = FILTER_LABEL[filter];
  const rel = fmtTimeAgo(trust.stalenessSeconds);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <Segmented options={FILTERS} value={filter} onChange={setFilter} />
        <div className="muted" style={{ fontSize: 13 }}>
          <span className="num">{rows.length}</span> instruments ·{' '}
          {withIdeas === 0 ? (
            'no active ideas'
          ) : (
            <>
              <span className="num">{withIdeas}</span> with active ideas
            </>
          )}
        </div>

        {/* Freshness indicator — uses only envelope fields (as_of / staleness). */}
        <span
          className="chip"
          title={`Snapshot ${trust.stalenessSeconds}s old · staleness threshold 90s · ${trust.asOf}`}
          style={{
            marginLeft: 'auto',
            borderColor: trust.stale ? 'rgba(240,169,59,0.4)' : C.border,
            color: trust.stale ? C.warn : C.text2,
          }}
        >
          <span className={`dot ${trust.stale ? '' : 'live-dot'}`} style={{ background: trust.stale ? C.warn : C.pos }} />
          {trust.stale ? 'STALE' : rel}
        </span>
      </div>

      {/* Stale strip — last-good prices, surfaced honestly (no faked recovery). */}
      {trust.stale && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontSize: 12.5,
            color: C.warn,
            background: 'rgba(240,169,59,0.08)',
            border: `1px solid rgba(240,169,59,0.3)`,
            borderRadius: 10,
            padding: '8px 12px',
          }}
        >
          <span className="dot" style={{ background: C.warn }} />
          prices as of {rel} — feed is {fmtTimeAgo(trust.stalenessSeconds)} old
        </div>
      )}

      <Panel pad={0} style={trust.stale ? { borderTop: `2px solid ${C.warn}` } : undefined}>
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ paddingLeft: 18 }}>Instrument</th>
              <th>30d trend</th>
              <SortHeader label="Price" sortKey="price" sort={sort} onSort={onSort} />
              <SortHeader label="1D" sortKey="change1d" sort={sort} onSort={onSort} />
              <SortHeader label="1W" sortKey="change1w" sort={sort} onSort={onSort} />
              <SortHeader label="1M" sortKey="change1m" sort={sort} onSort={onSort} />
              <SortHeader label="Volume" sortKey="volume" sort={sort} onSort={onSort} style={{ paddingRight: 18 }} />
              <th style={{ width: 28 }}></th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ paddingLeft: 18, paddingRight: 18 }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 12,
                      padding: '34px 16px',
                      color: C.text2,
                    }}
                  >
                    <span style={{ fontSize: 13.5 }}>No {filterLabel} in the universe yet.</span>
                    <button className="btn" style={{ height: 28, padding: '0 12px', fontSize: 12.5 }} onClick={() => setFilter('all')}>
                      Clear filter
                    </button>
                  </div>
                </td>
              </tr>
            ) : (
              rows.map((s) => (
                <tr key={s.symbol} className="clickable">
                  <td style={{ paddingLeft: 18 }}>
                    <Link href={`/symbols/${s.symbol}`} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <AssetGlyph sym={s} size={32} />
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                          <span className="mono" style={{ fontWeight: 650, fontSize: 13.5 }}>{s.symbol}</span>
                          {s.hasIdea && <span className="dot" style={{ background: C.accent }} title="Active idea" />}
                        </div>
                        <div className="dim" style={{ fontSize: 11.5 }}>{s.name}</div>
                      </div>
                    </Link>
                  </td>
                  <td>
                    <Link href={`/symbols/${s.symbol}`} style={{ display: 'inline-flex', justifyContent: 'flex-end', width: '100%' }}>
                      {/* Sparkline self-guards (<2 pts renders blank); empty/absent spark = no trend, not a crash. */}
                      <Sparkline data={s.spark ?? []} width={92} height={28} />
                    </Link>
                  </td>
                  {/* Nullable bar-derived cells: live BFF nulls calendar changes (1W/1M),
                      marketCap, and price/volume for priceless rows. Render "—", never a fake 0. */}
                  <td className="num"><Link href={`/symbols/${s.symbol}`}>{s.price == null ? <Dash /> : fmtPrice(s.price)}</Link></td>
                  <td>{s.change1d == null ? <Dash /> : <Delta value={s.change1d} dp={1} />}</td>
                  <td>{s.change1w == null ? <Dash /> : <Delta value={s.change1w} dp={1} arrow={false} />}</td>
                  <td>{s.change1m == null ? <Dash /> : <Delta value={s.change1m} dp={1} arrow={false} />}</td>
                  <td className="num muted" style={{ paddingRight: 18 }}>{s.volume == null ? <Dash /> : fmtCompact(s.volume)}</td>
                  <td style={{ paddingRight: 14 }}>
                    <Link href={`/symbols/${s.symbol}`} className="dim" style={{ display: 'inline-flex' }}>
                      <Icon name="chevronRight" size={15} />
                    </Link>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

// Honest em-dash for an absent numeric cell (live BFF nulls; priceless rows).
// dim + the "no value" aria-label keep it out of the screen-reader number stream.
function Dash() {
  const C = useChartColors();
  return (
    <span className="dim" aria-label="no value" style={{ color: C.text3 }}>
      —
    </span>
  );
}
