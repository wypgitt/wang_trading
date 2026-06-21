'use client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Icon } from '../Icon';
import { AssetGlyph } from '../ui/primitives';
import { getMarkets, getStrategies } from '../../data/api';
import { NAV } from '../../lib/readiness';
import { useChartColors } from '../../lib/theme';

// ⌘K command palette — jump to any symbol, strategy, or screen. Client-side over
// REAL entities only (the bars universe, the strategy roster, the nav map). The
// deep "replay 2026-…" grammar stays COMING; this is the honest, shippable v1.
type Kind = 'Symbol' | 'Strategy' | 'Screen';
interface Item {
  kind: Kind;
  key: string;
  label: string;
  sub?: string;
  href: string;
  coming?: boolean;
  glyph?: { symbol: string; type: 'equity' | 'index' | 'crypto' | 'future' };
}

function useItems(): Item[] {
  // Screens are static (instant). Symbols + strategies come from the async data
  // layer (mock resolves instantly; the live BFF fetches), so we load them into
  // state — the palette is usable immediately and the entities fill in.
  const screens: Item[] = useMemo(
    () =>
      NAV.flatMap((grp) =>
        grp.items.map((it) => ({
          kind: 'Screen' as const, key: `screen:${it.id}`, label: it.label, sub: grp.group,
          href: it.href, coming: it.readiness === 'coming',
        })),
      ),
    [],
  );
  const [dynamic, setDynamic] = useState<Item[]>([]);
  useEffect(() => {
    let alive = true;
    Promise.all([getMarkets(), getStrategies()]).then(([m, s]) => {
      if (!alive) return;
      const symbols: Item[] = (m.data ?? []).map((sym) => ({
        kind: 'Symbol', key: `sym:${sym.symbol}`, label: sym.symbol, sub: sym.name,
        href: `/symbols/${sym.symbol}`, glyph: { symbol: sym.symbol, type: sym.type },
      }));
      const strategies: Item[] = (s.data ?? []).map((st) => ({
        kind: 'Strategy', key: `strat:${st.id}`, label: st.name, sub: st.category, href: `/strategy/${st.id}`,
      }));
      setDynamic([...symbols, ...strategies]);
    });
    return () => { alive = false; };
  }, []);
  return useMemo(() => [...screens, ...dynamic], [screens, dynamic]);
}

function score(q: string, it: Item): number {
  const hay = `${it.label} ${it.sub ?? ''}`.toLowerCase();
  const needle = q.toLowerCase().trim();
  if (!needle) return 1;
  if (it.label.toLowerCase().startsWith(needle)) return 100;
  if (hay.startsWith(needle)) return 80;
  if (hay.includes(needle)) return 50;
  // subsequence (fuzzy)
  let i = 0;
  for (const ch of hay) if (ch === needle[i]) i++;
  return i === needle.length ? 20 : 0;
}

export function CommandPalette() {
  const C = useChartColors();
  const router = useRouter();
  const items = useItems();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const close = useCallback(() => { setOpen(false); setQuery(''); setActive(0); }, []);

  const results = useMemo(() => {
    const scored = items
      .map((it) => ({ it, s: score(query, it) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s)
      .slice(0, 24)
      .map((x) => x.it);
    return scored;
  }, [items, query]);

  useEffect(() => { setActive(0); }, [query]);

  // global ⌘K / Ctrl-K, plus the TrustBar search chip dispatches this event.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    const onOpen = () => setOpen(true);
    window.addEventListener('keydown', onKey);
    window.addEventListener('aperture:open-search', onOpen as EventListener);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('aperture:open-search', onOpen as EventListener);
    };
  }, []);

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 0);
  }, [open]);

  const go = useCallback((it: Item) => { router.push(it.href); close(); }, [router, close]);

  const onInputKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, results.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
    else if (e.key === 'Enter') { e.preventDefault(); if (results[active]) go(results[active]); }
  };

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${active}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }, [active]);

  if (!open) return null;

  return (
    <div
      className="backdrop"
      onMouseDown={close}
      style={{ position: 'fixed', inset: 0, zIndex: 100, background: 'rgba(5,7,10,0.62)', backdropFilter: 'blur(3px)', display: 'flex', justifyContent: 'center', alignItems: 'flex-start', paddingTop: '12vh' }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
        className="drawer-panel"
        style={{ width: 'min(620px, 92vw)', background: C.surface1, border: `1px solid ${C.borderStrong}`, borderRadius: 16, boxShadow: '0 18px 50px rgba(0,0,0,0.55)', overflow: 'hidden' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 16px', borderBottom: `1px solid ${C.border}` }}>
          <span style={{ color: C.text3, display: 'inline-flex' }}><Icon name="search" size={18} /></span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKey}
            placeholder="Jump to a symbol, strategy, or screen…"
            aria-label="Search"
            style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: C.text1, fontSize: 15 }}
          />
          <span className="mono" style={{ fontSize: 11, color: C.text3 }}>esc</span>
        </div>
        <div ref={listRef} className="scroll-y" style={{ maxHeight: 380, padding: 8 }}>
          {results.length === 0 ? (
            <div className="dim" style={{ padding: '28px 12px', textAlign: 'center', fontSize: 13 }}>
              No matches. Try a ticker (NVDA), a strategy, or a screen.
            </div>
          ) : (
            results.map((it, i) => (
              <button
                key={it.key}
                data-idx={i}
                onMouseEnter={() => setActive(i)}
                onClick={() => go(it)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 12, width: '100%', textAlign: 'left',
                  padding: '9px 12px', borderRadius: 10,
                  background: i === active ? C.surface2 : 'transparent',
                }}
              >
                {it.glyph ? (
                  <AssetGlyph sym={it.glyph} size={28} />
                ) : (
                  <span style={{ width: 28, height: 28, borderRadius: 8, display: 'grid', placeItems: 'center', background: C.surfaceInset, color: C.text3 }}>
                    <Icon name={it.kind === 'Strategy' ? 'strategies' : 'overview'} size={15} />
                  </span>
                )}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13.5, fontWeight: 600, color: C.text1 }} className={it.kind === 'Symbol' ? 'mono' : ''}>
                    {it.label}
                  </div>
                  {it.sub && <div className="dim" style={{ fontSize: 11.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.sub}</div>}
                </div>
                {it.coming && <span style={{ color: C.text3, display: 'inline-flex' }}><Icon name="lock" size={13} /></span>}
                <span className="eyebrow" style={{ fontSize: 10 }}>{it.kind}</span>
              </button>
            ))
          )}
        </div>
        <div style={{ display: 'flex', gap: 16, padding: '9px 16px', borderTop: `1px solid ${C.border}`, color: C.text3, fontSize: 11.5 }}>
          <span><span className="mono">↑↓</span> navigate</span>
          <span><span className="mono">↵</span> open</span>
          <span><span className="mono">esc</span> close</span>
        </div>
      </div>
    </div>
  );
}
