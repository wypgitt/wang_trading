import { useState } from 'react';
import { Icon } from '../../components/Icon';
import { Sparkline } from '../../components/charts/Sparkline';
import { AssetGlyph, Delta } from '../../components/ui/primitives';
import { useNav } from '../../nav';
import { AssetType, SYMBOLS, Sym } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtCompact, fmtPrice } from '../../lib/format';

const FILTERS: { id: AssetType | 'all'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'equity', label: 'Equities' },
  { id: 'index', label: 'Indexes' },
  { id: 'crypto', label: 'Crypto' },
  { id: 'future', label: 'Futures' },
];

function MarketRow({ s, onClick }: { s: Sym; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="row-hover"
      style={{
        display: 'grid',
        gridTemplateColumns: '1.7fr 130px 110px 84px 84px 84px 130px 22px',
        alignItems: 'center',
        gap: 14,
        padding: '12px 14px',
        borderTop: '1px solid var(--border)',
        textAlign: 'left',
        width: '100%',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
        <AssetGlyph sym={s} />
        <div style={{ minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontWeight: 650, fontSize: 14 }}>{s.symbol}</span>
            {s.hasIdea && <span className="dot" style={{ background: C.accent }} title="Active trade idea" />}
          </div>
          <div className="muted" style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {s.name}
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <Sparkline data={s.spark} width={120} height={32} />
      </div>
      <div className="num" style={{ textAlign: 'right', fontWeight: 600, fontSize: 13.5 }}>
        {fmtPrice(s.price)}
      </div>
      <div style={{ textAlign: 'right' }}>
        <Delta value={s.change1d} size={13} arrow={false} />
      </div>
      <div style={{ textAlign: 'right' }}>
        <Delta value={s.change1w} size={13} arrow={false} />
      </div>
      <div style={{ textAlign: 'right' }}>
        <Delta value={s.change1m} size={13} arrow={false} />
      </div>
      <div className="num muted" style={{ textAlign: 'right', fontSize: 12.5 }}>
        {s.marketCap ? fmtCompact(s.marketCap) : fmtCompact(s.volume)}
      </div>
      <span style={{ color: C.text3, display: 'flex', justifyContent: 'flex-end' }}>
        <Icon name="chevronRight" size={16} />
      </span>
    </button>
  );
}

export function MarketsPage() {
  const { go } = useNav();
  const [filter, setFilter] = useState<AssetType | 'all'>('all');
  const visible = filter === 'all' ? SYMBOLS : SYMBOLS.filter((s) => s.type === filter);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 8 }}>
          {FILTERS.map((f) => (
            <button key={f.id} className={`chip ${filter === f.id ? 'active' : ''}`} onClick={() => setFilter(f.id)}>
              {f.label}
            </button>
          ))}
        </div>
        <span className="muted" style={{ fontSize: 12.5 }}>
          {visible.length} instruments · {SYMBOLS.filter((s) => s.hasIdea).length} with active ideas
        </span>
      </div>

      <div className="card" style={{ padding: '4px 8px 8px' }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1.7fr 130px 110px 84px 84px 84px 130px 22px',
            gap: 14,
            padding: '12px 14px 8px',
          }}
          className="eyebrow"
        >
          <span>Instrument</span>
          <span style={{ textAlign: 'center' }}>30d trend</span>
          <span style={{ textAlign: 'right' }}>Price</span>
          <span style={{ textAlign: 'right' }}>1D</span>
          <span style={{ textAlign: 'right' }}>1W</span>
          <span style={{ textAlign: 'right' }}>1M</span>
          <span style={{ textAlign: 'right' }}>Mkt cap / Vol</span>
          <span />
        </div>
        {visible.map((s) => (
          <MarketRow key={s.symbol} s={s} onClick={() => go('symbol', s.symbol)} />
        ))}
      </div>
    </div>
  );
}
