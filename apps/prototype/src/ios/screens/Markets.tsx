import { useState } from 'react';
import { Sparkline } from '../../components/charts/Sparkline';
import { AssetGlyph, Delta } from '../../components/ui/primitives';
import { Icon } from '../../components/Icon';
import { IOSCard, LargeTitle, Screen } from '../iosUi';
import { useIOS } from '../iosNav';
import { AssetType, SYMBOLS } from '../../data/mock';
import { C } from '../../lib/colors';
import { fmtPrice } from '../../lib/format';

const FILTERS: { id: AssetType | 'all'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'equity', label: 'Equities' },
  { id: 'index', label: 'Indexes' },
  { id: 'crypto', label: 'Crypto' },
  { id: 'future', label: 'Futures' },
];

export function Markets() {
  const { push } = useIOS();
  const [filter, setFilter] = useState<AssetType | 'all'>('all');
  const visible = filter === 'all' ? SYMBOLS : SYMBOLS.filter((s) => s.type === filter);

  return (
    <Screen>
      <LargeTitle title="Markets" subtitle={`${SYMBOLS.length} instruments`} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 9, background: C.surface3, borderRadius: 12, padding: '10px 13px', marginBottom: 14, color: C.text3 }}>
        <Icon name="search" size={16} />
        <span style={{ fontSize: 14 }}>Search</span>
      </div>

      <div style={{ display: 'flex', gap: 8, overflowX: 'auto', margin: '0 -16px 14px', padding: '0 16px' }}>
        {FILTERS.map((f) => (
          <button key={f.id} className={`chip ${filter === f.id ? 'active' : ''}`} style={{ flex: 'none' }} onClick={() => setFilter(f.id)}>
            {f.label}
          </button>
        ))}
      </div>

      <IOSCard pad={0}>
        {visible.map((s, i) => (
          <button
            key={s.symbol}
            onClick={() => push({ type: 'symbol', id: s.symbol })}
            style={{ display: 'flex', alignItems: 'center', gap: 12, width: '100%', padding: '11px 14px', borderTop: i ? '1px solid var(--border)' : 'none', textAlign: 'left' }}
          >
            <AssetGlyph sym={s} size={36} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 650, fontSize: 14.5 }}>
                {s.symbol}
                {s.hasIdea && <span className="dot" style={{ background: C.accent, width: 6, height: 6 }} />}
              </div>
              <div className="muted" style={{ fontSize: 11.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.name}</div>
            </div>
            <Sparkline data={s.spark} width={62} height={28} />
            <div style={{ textAlign: 'right', minWidth: 76 }}>
              <div className="num" style={{ fontWeight: 600, fontSize: 14 }}>{fmtPrice(s.price)}</div>
              <Delta value={s.change1d} size={12} arrow={false} />
            </div>
          </button>
        ))}
      </IOSCard>
    </Screen>
  );
}
