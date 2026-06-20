import type { Metadata, Viewport } from 'next';
import './globals.css';
import { AppShell } from '../components/shell/AppShell';

export const metadata: Metadata = {
  title: 'Aperture — wang quant engine',
  description: 'Your quant engine, in focus. Decision support + monitoring for the wang_trading engine.',
};

export const viewport: Viewport = {
  colorScheme: 'light dark',
};

// Set the resolved theme before first paint (no flash). Follows the OS by default
// (mode 'system'); honours a saved 'light'/'dark' override.
const NO_FLASH = `(function(){try{
var m=localStorage.getItem('aperture:theme')||'system';
var dark=m==='dark'||(m==='system'&&matchMedia('(prefers-color-scheme: dark)').matches);
document.documentElement.dataset.theme=dark?'dark':'light';
}catch(e){document.documentElement.dataset.theme='dark';}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-density="comfort" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: NO_FLASH }} />
      </head>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
