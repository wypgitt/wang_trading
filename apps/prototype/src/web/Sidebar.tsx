import { Icon } from '../components/Icon';
import { PageId, useNav } from '../nav';
import { SYSTEM } from '../data/mock';

const GROUPS: { group: string; items: { id: PageId; label: string; icon: string }[] }[] = [
  {
    group: 'Cockpit',
    items: [
      { id: 'overview', label: 'Overview', icon: 'overview' },
      { id: 'markets', label: 'Markets', icon: 'markets' },
      { id: 'ideas', label: 'Trade Ideas', icon: 'ideas' },
      { id: 'strategies', label: 'Strategies', icon: 'strategies' },
    ],
  },
  {
    group: 'Portfolio',
    items: [
      { id: 'portfolio', label: 'Portfolio & Risk', icon: 'portfolio' },
      { id: 'research', label: 'Research & Backtests', icon: 'research' },
      { id: 'model', label: 'Model & Features', icon: 'model' },
    ],
  },
];

const PARENT: Partial<Record<PageId, PageId>> = { symbol: 'markets', strategy: 'strategies' };

export function Sidebar() {
  const { route, go } = useNav();
  const active = PARENT[route.page] ?? route.page;
  return (
    <aside
      style={{
        background: 'var(--bg-1)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        padding: '16px 12px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 8px 18px' }}>
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 9,
            background: 'var(--accent-grad)',
            display: 'grid',
            placeItems: 'center',
            boxShadow: 'var(--shadow-glow)',
          }}
        >
          <span style={{ fontWeight: 800, fontSize: 16, color: '#fff' }}>✦</span>
        </div>
        <div>
          <div style={{ fontWeight: 750, fontSize: 15.5, letterSpacing: '-0.02em' }}>Aperture</div>
          <div className="dim" style={{ fontSize: 11 }}>
            wang · quant engine
          </div>
        </div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: 18, flex: 1, overflowY: 'auto' }}>
        {GROUPS.map((g) => (
          <div key={g.group}>
            <div className="eyebrow" style={{ padding: '0 8px 8px' }}>
              {g.group}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {g.items.map((it) => {
                const on = active === it.id;
                return (
                  <button
                    key={it.id}
                    data-page={it.id}
                    onClick={() => go(it.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 11,
                      height: 38,
                      padding: '0 10px',
                      borderRadius: 10,
                      color: on ? 'var(--text-1)' : 'var(--text-2)',
                      background: on ? 'var(--accent-soft)' : 'transparent',
                      fontWeight: on ? 600 : 500,
                      fontSize: 13.5,
                      position: 'relative',
                      transition: 'all 0.12s ease',
                    }}
                    onMouseEnter={(e) => {
                      if (!on) e.currentTarget.style.background = 'var(--surface-1)';
                    }}
                    onMouseLeave={(e) => {
                      if (!on) e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <span style={{ color: on ? 'var(--accent-2)' : 'var(--text-3)' }}>
                      <Icon name={it.icon} size={18} />
                    </span>
                    {it.label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="card" style={{ padding: 12, marginTop: 12, background: 'var(--surface-1)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 12.5, fontWeight: 600 }}>
            <span className="dot live-dot" style={{ background: 'var(--pos)' }} />
            {SYSTEM.mode} trading
          </span>
          <span className="dim" style={{ fontSize: 11.5 }}>live</span>
        </div>
        <div className="dim" style={{ fontSize: 11.5, marginTop: 7 }}>
          {SYSTEM.breakers} breakers · {SYSTEM.alerts} alert · data {SYSTEM.dataFreshnessSec}s
        </div>
      </div>
    </aside>
  );
}
