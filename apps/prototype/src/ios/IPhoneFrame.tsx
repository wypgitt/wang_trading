import { ReactNode } from 'react';

function StatusBarIOS() {
  return (
    <div
      style={{
        height: 54,
        flex: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 28px',
        paddingTop: 8,
        fontSize: 15,
        fontWeight: 650,
        color: 'var(--text-1)',
        position: 'relative',
        zIndex: 3,
      }}
    >
      <span style={{ letterSpacing: '-0.02em' }}>9:41</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        {/* signal */}
        <svg width="18" height="12" viewBox="0 0 18 12" fill="var(--text-1)">
          <rect x="0" y="8" width="3" height="4" rx="1" />
          <rect x="5" y="5" width="3" height="7" rx="1" />
          <rect x="10" y="2.5" width="3" height="9.5" rx="1" />
          <rect x="15" y="0" width="3" height="12" rx="1" opacity="0.35" />
        </svg>
        {/* wifi */}
        <svg width="16" height="12" viewBox="0 0 16 12" fill="none" stroke="var(--text-1)" strokeWidth="1.6">
          <path d="M1 4.2a10 10 0 0 1 14 0" />
          <path d="M3.5 6.8a6 6 0 0 1 9 0" />
          <path d="M6 9.3a2.5 2.5 0 0 1 4 0" />
        </svg>
        {/* battery */}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2 }}>
          <span style={{ width: 24, height: 12, borderRadius: 3, border: '1.4px solid var(--text-2)', padding: 1.5, display: 'inline-block' }}>
            <span style={{ display: 'block', width: '78%', height: '100%', background: 'var(--text-1)', borderRadius: 1 }} />
          </span>
          <span style={{ width: 1.6, height: 4, background: 'var(--text-2)', borderRadius: 1 }} />
        </span>
      </span>
    </div>
  );
}

export function IPhoneFrame({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        width: 412,
        height: 850,
        borderRadius: 56,
        background: 'linear-gradient(160deg, #2a2d34, #15171c)',
        padding: 12,
        boxShadow: '0 40px 90px rgba(0,0,0,0.6), inset 0 0 2px rgba(255,255,255,0.3)',
        flex: 'none',
      }}
    >
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          borderRadius: 45,
          overflow: 'hidden',
          background: 'var(--bg-0)',
          display: 'flex',
          flexDirection: 'column',
          fontFamily: '-apple-system, "SF Pro Text", "SF Pro Display", system-ui, sans-serif',
        }}
      >
        <StatusBarIOS />
        {/* Dynamic Island */}
        <div
          style={{
            position: 'absolute',
            top: 12,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 122,
            height: 34,
            background: '#000',
            borderRadius: 20,
            zIndex: 5,
          }}
        />
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', position: 'relative' }}>{children}</div>
        {/* Home indicator */}
        <div
          style={{
            position: 'absolute',
            bottom: 8,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 138,
            height: 5,
            borderRadius: 99,
            background: 'rgba(255,255,255,0.55)',
            zIndex: 20,
          }}
        />
      </div>
    </div>
  );
}
