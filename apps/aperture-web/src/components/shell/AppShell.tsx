'use client';
import { usePathname } from 'next/navigation';
import { DensityProvider } from '../../lib/density';
import { ThemeProvider } from '../../lib/theme';
import { Sidebar } from './Sidebar';
import { TrustBar } from './TrustBar';
import { CommandPalette } from './CommandPalette';

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <ThemeProvider>
      <DensityProvider>
        <CommandPalette />
        <div className="app-shell">
          <Sidebar />
          <div className="app-main">
            <TrustBar />
            <main className="page-scroll">
              {/* key on route → re-trigger the fade-up on navigation */}
              <div className="page-inner fade-up" key={pathname}>
                {children}
              </div>
            </main>
          </div>
        </div>
      </DensityProvider>
    </ThemeProvider>
  );
}
