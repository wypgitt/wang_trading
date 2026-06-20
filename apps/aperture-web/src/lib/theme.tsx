'use client';
// ---------------------------------------------------------------------------
// Theme. Defaults to FOLLOW SYSTEM (prefers-color-scheme) so the app is light by
// day and dark at night automatically; the user can still force light or dark.
// CSS-var-driven UI flips instantly via the [data-theme] attribute. Charts read
// resolved hex (CSS vars don't resolve in SVG attributes), so they use
// useChartColors() — which returns the active palette and re-renders on change.
// ---------------------------------------------------------------------------
import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { C, C_LIGHT } from './colors';

export type ThemeMode = 'system' | 'light' | 'dark';
export type ResolvedTheme = 'light' | 'dark';

const KEY = 'aperture:theme';
const MQ = '(prefers-color-scheme: dark)';

function systemTheme(): ResolvedTheme {
  if (typeof window === 'undefined' || !window.matchMedia) return 'dark';
  return window.matchMedia(MQ).matches ? 'dark' : 'light';
}
function resolve(mode: ThemeMode): ResolvedTheme {
  return mode === 'system' ? systemTheme() : mode;
}

const ThemeCtx = createContext<{
  mode: ThemeMode;
  resolved: ResolvedTheme;
  setMode: (m: ThemeMode) => void;
  cycle: () => void;
}>({ mode: 'system', resolved: 'dark', setMode: () => {}, cycle: () => {} });

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>('system');
  const [resolved, setResolved] = useState<ResolvedTheme>('dark');

  const apply = useCallback((m: ThemeMode) => {
    const r = resolve(m);
    setResolved(r);
    document.documentElement.dataset.theme = r;
  }, []);

  // Initial mount: read saved mode (default system) and resolve.
  useEffect(() => {
    const saved = (typeof localStorage !== 'undefined' && localStorage.getItem(KEY)) as ThemeMode | null;
    const m: ThemeMode = saved === 'light' || saved === 'dark' || saved === 'system' ? saved : 'system';
    setModeState(m);
    apply(m);
  }, [apply]);

  // Live OS changes: only re-resolve while in 'system' mode.
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia(MQ);
    const onChange = () => {
      if ((localStorage.getItem(KEY) ?? 'system') === 'system') apply('system');
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [apply]);

  const setMode = useCallback(
    (m: ThemeMode) => {
      setModeState(m);
      apply(m);
      try {
        localStorage.setItem(KEY, m);
      } catch {
        /* private mode */
      }
    },
    [apply],
  );

  // System → Light → Dark → System
  const cycle = useCallback(() => {
    setMode(mode === 'system' ? 'light' : mode === 'light' ? 'dark' : 'system');
  }, [mode, setMode]);

  return <ThemeCtx.Provider value={{ mode, resolved, setMode, cycle }}>{children}</ThemeCtx.Provider>;
}

export const useTheme = () => useContext(ThemeCtx);

/** The active chart hex palette — re-evaluated (and re-rendered) on theme change. */
export function useChartColors() {
  return useTheme().resolved === 'light' ? C_LIGHT : C;
}
