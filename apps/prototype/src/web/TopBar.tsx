import { Icon } from '../components/Icon';
import { RegimeChip } from '../components/ui/primitives';
import { PageId, Route, useNav } from '../nav';
import { PORTFOLIO, REGIME, stratBy, symBy, SYSTEM } from '../data/mock';
import { fmtCompact, fmtPctSigned } from '../lib/format';

const PARENT: Partial<Record<PageId, PageId>> = { symbol: 'markets', strategy: 'strategies' };

function titleFor(route: Route): { title: string; sub: string } {
  switch (route.page) {
    case 'overview':
      return { title: 'Overview', sub: 'Your engine at a glance' };
    case 'markets':
      return { title: 'Markets', sub: 'Equities · Indexes · Crypto · Futures' };
    case 'ideas':
      return { title: 'Trade Ideas', sub: 'The full decision chain, live' };
    case 'strategies':
      return { title: 'Strategies', sub: '10 signal families across 6 categories' };
    case 'portfolio':
      return { title: 'Portfolio & Risk', sub: 'Positions · exposure · factor risk' };
    case 'research':
      return { title: 'Research & Backtests', sub: 'Walk-forward · validation gates' };
    case 'model':
      return { title: 'Model & Features', sub: 'Meta-labeler · calibration · drift' };
    case 'symbol': {
      const s = symBy(route.param ?? '');
      return { title: s.symbol, sub: s.name };
    }
    case 'strategy': {
      const st = stratBy(route.param ?? '');
      return { title: st.name, sub: `${st.category} · ${st.source}` };
    }
  }
}

export function TopBar() {
  const { route, go } = useNav();
  const { title, sub } = titleFor(route);
  const back = PARENT[route.page];
  const pnlPct = PORTFOLIO.dailyPnlPct;

  return (
    <header
      style={{
        height: 68,
        flex: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        padding: '0 30px',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(10,12,16,0.72)',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 5,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
        {back && (
          <button
            onClick={() => go(back)}
            className="btn"
            style={{ width: 38, padding: 0, height: 36, borderRadius: 10 }}
            title="Back"
          >
            <Icon name="arrowLeft" size={18} />
          </button>
        )}
        <div style={{ minWidth: 0 }}>
          <h1 style={{ fontSize: 21, lineHeight: 1.1 }}>{title}</h1>
          <div className="muted" style={{ fontSize: 12.5, marginTop: 2 }}>
            {sub}
          </div>
        </div>
      </div>

      <div style={{ flex: 1 }} />

      <button
        className="chip"
        style={{ height: 36, color: 'var(--text-3)', gap: 9, padding: '0 13px', width: 230, justifyContent: 'flex-start' }}
      >
        <Icon name="search" size={15} />
        <span style={{ fontSize: 12.5 }}>Search symbols, strategies…</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, opacity: 0.7 }}>⌘K</span>
      </button>

      <RegimeChip regime={REGIME} />

      <span className="chip" style={{ gap: 7 }}>
        <span className="dot live-dot" style={{ background: 'var(--pos)' }} />
        <span style={{ fontSize: 12.5 }}>{SYSTEM.lastRefreshSec}s ago</span>
      </span>

      <div style={{ textAlign: 'right' }}>
        <div className="num" style={{ fontWeight: 700, fontSize: 14.5 }}>
          {fmtCompact(PORTFOLIO.nav)}
        </div>
        <div className={`num ${pnlPct >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 12, fontWeight: 600 }}>
          {fmtPctSigned(pnlPct)} today
        </div>
      </div>

      <button className="btn" style={{ width: 38, padding: 0, height: 38, borderRadius: 11 }}>
        <Icon name="bell" size={18} />
      </button>
      <div
        style={{
          width: 38,
          height: 38,
          borderRadius: 11,
          background: 'var(--surface-2)',
          border: '1px solid var(--border)',
          display: 'grid',
          placeItems: 'center',
          fontWeight: 700,
          fontSize: 13,
        }}
      >
        YW
      </div>
    </header>
  );
}
