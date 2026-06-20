'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Icon } from '../Icon';
import { useDensity } from '../../lib/density';
import { useTheme } from '../../lib/theme';
import { screenByHref } from '../../lib/readiness';
import { getTradeIdeas } from '../../data/api';
import { deriveTrust } from '../../data/envelope';
import { fmtTimeAgo } from '../../lib/format';

// Page title + sub, derived from the route (detail routes resolve their own title).
function titleFor(pathname: string): { title: string; sub: string } {
  if (pathname.startsWith('/symbols/')) {
    const s = decodeURIComponent(pathname.split('/')[2] ?? '');
    return { title: s, sub: 'Symbol detail — price + what the engine sees' };
  }
  if (pathname.startsWith('/strategy/')) {
    return { title: 'Strategy', sub: 'Thesis · parameters · live ideas' };
  }
  const spec = screenByHref(pathname);
  if (spec) return { title: spec.label, sub: spec.purpose ?? '' };
  return { title: 'Overview', sub: '“What now” in one glance' };
}

export function TrustBar() {
  const pathname = usePathname() || '/overview';
  const { title, sub } = titleFor(pathname);
  const { density, setDensity } = useDensity();
  const { mode: themeMode, resolved: resolvedTheme, cycle: cycleTheme } = useTheme();
  const themeGlyph = themeMode === 'system' ? '◐' : themeMode === 'light' ? '☀' : '☾';
  const themeLabel =
    themeMode === 'system' ? `Theme: System (${resolvedTheme})` : themeMode === 'light' ? 'Theme: Light' : 'Theme: Dark';

  // Three honest pills only — bound to fields the envelope actually carries.
  // The Model pill reflects whether a production model is LOADED (model_version
  // present). A per-idea MODEL_REQUIRED (e.g. insufficient history) is handled
  // inline on that idea — it does not flip the global pill.
  const env = getTradeIdeas();
  const trust = deriveTrust(env);
  const modelOk = trust.modelLoaded;

  return (
    <header className="trust-bar">
      <div style={{ minWidth: 0 }}>
        <h1 style={{ fontSize: 19, lineHeight: 1.1 }}>{title}</h1>
        <div className="dim" style={{ fontSize: 12, marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {sub}
        </div>
      </div>

      <div style={{ flex: 1 }} />

      {/* ⌘K command palette — jump to a symbol / strategy / screen */}
      <button
        className="chip"
        onClick={() => window.dispatchEvent(new CustomEvent('aperture:open-search'))}
        style={{ height: 34, gap: 8, color: 'var(--text-3)', minWidth: 200, justifyContent: 'flex-start' }}
        title="Command palette — jump to a symbol, strategy, or screen"
      >
        <Icon name="search" size={14} />
        <span style={{ flex: 1, textAlign: 'left' }}>Search symbols…</span>
        <span className="mono" style={{ fontSize: 11, opacity: 0.7 }}>⌘K</span>
      </button>

      {/* Mode */}
      <span className="chip" style={{ borderColor: 'rgba(77,159,255,0.4)', color: 'var(--info)' }} title="Engine is read-only (live_orders_sent=0). Live mode is coming.">
        <span aria-hidden="true" className="dot" style={{ background: 'var(--info)' }} />
        PAPER
      </span>

      {/* Model — drives whether meta/cal probabilities render */}
      <Link
        href="/model"
        className="chip"
        style={{ borderColor: modelOk ? 'rgba(30,203,139,0.4)' : 'rgba(240,169,59,0.4)', color: modelOk ? 'var(--pos)' : 'var(--warn)' }}
        title={modelOk ? `Production model loaded: ${trust.modelVersion}` : 'Some ideas return MODEL_REQUIRED — probabilities gated'}
      >
        <span aria-hidden="true" className="dot" style={{ background: modelOk ? 'var(--pos)' : 'var(--warn)' }} />
        {modelOk ? 'MODEL LOADED' : 'MODEL REQUIRED'}
      </Link>

      {/* Freshness — the single most important honesty signal */}
      <Link
        href="/overview"
        className="chip"
        style={{ borderColor: trust.stale ? 'rgba(240,169,59,0.4)' : 'var(--border)', color: trust.stale ? 'var(--warn)' : 'var(--text-2)' }}
        title={`Snapshot ${trust.stalenessSeconds}s old · staleness threshold 90s`}
      >
        <span aria-hidden="true" className={`dot ${trust.stale ? '' : 'live-dot'}`} style={{ background: trust.stale ? 'var(--warn)' : 'var(--pos)' }} />
        {trust.stale ? 'STALE' : fmtTimeAgo(trust.stalenessSeconds)}
      </Link>

      {/* Density toggle */}
      <button
        className="chip"
        style={{ height: 34, gap: 7 }}
        onClick={() => setDensity(density === 'pro' ? 'comfort' : 'pro')}
        title="Pro density reveals more real fields and tighter layout — it never unlocks coming data."
      >
        <Icon name="layers" size={14} />
        {density === 'pro' ? 'Pro' : 'Comfort'}
      </button>

      {/* Theme cycle: System (follows OS) → Light → Dark. Default System. */}
      <button
        className="chip"
        style={{ height: 34, gap: 7 }}
        onClick={cycleTheme}
        aria-label={`${themeLabel}. Click to change.`}
        title={`${themeLabel} — click to cycle System / Light / Dark`}
      >
        <span style={{ fontSize: 14 }}>{themeGlyph}</span>
        {themeMode === 'system' ? 'Auto' : themeMode === 'light' ? 'Light' : 'Dark'}
      </button>
    </header>
  );
}
