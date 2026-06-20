'use client';
// ---------------------------------------------------------------------------
// Pro-density toggle. ONE global preference (persisted per device). Not a second
// app, not a second theme — it only re-points spacing/size aliases and disclosure
// defaults via a `data-density` attribute on <html>. It NEVER unlocks COMING data
// or changes colors/semantics. The v2 "operator cockpit" is Aperture-at-pro.
// ---------------------------------------------------------------------------
import { createContext, useCallback, useContext, useEffect, useState } from 'react';

export type Density = 'comfort' | 'pro';

const DensityCtx = createContext<{ density: Density; setDensity: (d: Density) => void }>({
  density: 'comfort',
  setDensity: () => {},
});

const KEY = 'aperture:density';

export function DensityProvider({ children }: { children: React.ReactNode }) {
  const [density, setDensityState] = useState<Density>('comfort');

  useEffect(() => {
    const saved = (typeof localStorage !== 'undefined' && localStorage.getItem(KEY)) as Density | null;
    if (saved === 'pro' || saved === 'comfort') {
      setDensityState(saved);
      document.documentElement.dataset.density = saved;
    } else {
      document.documentElement.dataset.density = 'comfort';
    }
  }, []);

  const setDensity = useCallback((d: Density) => {
    setDensityState(d);
    document.documentElement.dataset.density = d;
    try {
      localStorage.setItem(KEY, d);
    } catch {
      /* private mode */
    }
  }, []);

  return <DensityCtx.Provider value={{ density, setDensity }}>{children}</DensityCtx.Provider>;
}

export const useDensity = () => useContext(DensityCtx);
