import { CSSProperties, ReactNode } from 'react';
import { Icon } from '../components/Icon';
import { C } from '../lib/colors';

export function Screen({ children }: { children: ReactNode }) {
  return <div style={{ padding: '2px 16px 108px' }}>{children}</div>;
}

export function LargeTitle({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', padding: '10px 2px 16px' }}>
      <div>
        <h1 style={{ fontSize: 30, fontWeight: 760, letterSpacing: '-0.03em' }}>{title}</h1>
        {subtitle && <div className="muted" style={{ fontSize: 13.5, marginTop: 3 }}>{subtitle}</div>}
      </div>
      {right}
    </div>
  );
}

export function IOSCard({ children, style, pad = 16 }: { children: ReactNode; style?: CSSProperties; pad?: number }) {
  return (
    <div style={{ background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 20, padding: pad, ...style }}>
      {children}
    </div>
  );
}

export function SectionHeader({ title, action, onAction }: { title: string; action?: string; onAction?: () => void }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 4px 10px' }}>
      <h3 style={{ fontSize: 18, fontWeight: 680, letterSpacing: '-0.02em' }}>{title}</h3>
      {action && (
        <button onClick={onAction} style={{ color: C.accent2, fontSize: 14, fontWeight: 600 }}>
          {action}
        </button>
      )}
    </div>
  );
}

export function NavHeader({ title, subtitle, onBack, right }: { title: string; subtitle?: string; onBack: () => void; right?: ReactNode }) {
  return (
    <div
      style={{
        height: 50,
        flex: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '0 10px',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(10,12,16,0.8)',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 4,
      }}
    >
      <button onClick={onBack} style={{ display: 'flex', alignItems: 'center', color: C.accent2, fontSize: 16, fontWeight: 600, gap: 1 }}>
        <Icon name="arrowLeft" size={22} />
      </button>
      <div style={{ flex: 1, textAlign: 'center' }}>
        <div style={{ fontWeight: 680, fontSize: 16 }}>{title}</div>
        {subtitle && <div className="dim" style={{ fontSize: 11 }}>{subtitle}</div>}
      </div>
      <div style={{ width: 32, display: 'flex', justifyContent: 'flex-end' }}>{right}</div>
    </div>
  );
}
