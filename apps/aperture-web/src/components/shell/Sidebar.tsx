'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Icon } from '../Icon';
import { NAV, LOCK_GLYPH, LOCK_LABEL } from '../../lib/readiness';

// Symbol & Strategy detail are non-nav detail routes — they highlight their parent.
function activeHref(pathname: string): string {
  if (pathname.startsWith('/symbols')) return '/markets';
  if (pathname.startsWith('/strategy/')) return '/strategies';
  return pathname;
}

// Live vs coming counts, derived from the NAV map (never hardcoded) so the
// footer stays honest when a screen flips readiness in '@/lib/readiness'.
const NAV_ITEMS = NAV.flatMap((g) => g.items);
const LIVE_COUNT = NAV_ITEMS.filter((it) => it.readiness === 'live').length;
const COMING_COUNT = NAV_ITEMS.filter((it) => it.readiness === 'coming').length;

export function Sidebar() {
  const pathname = usePathname() || '/overview';
  const active = activeHref(pathname);

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
      <Link href="/overview" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 8px 18px' }}>
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
          <div className="dim" style={{ fontSize: 11 }}>wang · quant engine</div>
        </div>
      </Link>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: 16, flex: 1, overflowY: 'auto' }}>
        {NAV.map((g) => (
          <div key={g.group}>
            <div className="eyebrow" style={{ padding: '0 8px 8px' }}>{g.group}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {g.items.map((it) => {
                const on = active === it.href;
                const coming = it.readiness === 'coming';
                return (
                  <Link
                    key={it.id}
                    href={it.href}
                    title={coming ? LOCK_LABEL[it.lock ?? 'gated'] : undefined}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 11,
                      height: 38,
                      padding: '0 10px',
                      borderRadius: 10,
                      color: on ? 'var(--text-1)' : coming ? 'var(--text-3)' : 'var(--text-2)',
                      background: on ? 'var(--accent-soft)' : 'transparent',
                      fontWeight: on ? 600 : 500,
                      fontSize: 13.5,
                      transition: 'all 0.12s ease',
                    }}
                  >
                    <span style={{ color: on ? 'var(--accent-2)' : 'var(--text-3)', display: 'inline-flex' }}>
                      <Icon name={it.icon} size={18} />
                    </span>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {it.label}
                    </span>
                    {it.readiness === 'live' && !on && it.modelGated && (
                      <span aria-hidden="true" className="dot" style={{ background: 'var(--info)', opacity: 0.7 }} />
                    )}
                    {it.readiness === 'live' && !it.modelGated && (
                      <span aria-hidden="true" className="dot live-dot" style={{ background: 'var(--pos)', opacity: on ? 1 : 0.6 }} />
                    )}
                    {coming && (
                      <span style={{ color: 'var(--text-3)', display: 'inline-flex', opacity: 0.7 }}>
                        <Icon name={LOCK_GLYPH[it.lock ?? 'gated']} size={13} />
                      </span>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="card" style={{ padding: 12, marginTop: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 12.5, fontWeight: 600 }}>
            <span aria-hidden="true" className="dot live-dot" style={{ background: 'var(--info)' }} />
            Paper trading
          </span>
          <span className="dim" style={{ fontSize: 11.5 }}>read-only</span>
        </div>
        <div className="dim" style={{ fontSize: 11.5, marginTop: 7, lineHeight: 1.5 }}>
          Decision support · live actions gated. {LIVE_COUNT} live · {COMING_COUNT} coming.
        </div>
      </div>
    </aside>
  );
}
