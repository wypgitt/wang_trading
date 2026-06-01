import { Segmented } from './ui/primitives';

export function PrototypeBar({
  platform,
  setPlatform,
}: {
  platform: 'web' | 'ios';
  setPlatform: (p: 'web' | 'ios') => void;
}) {
  return (
    <div
      style={{
        height: 48,
        flex: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 18px',
        background: 'var(--bg-1)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
        <div
          style={{
            width: 26,
            height: 26,
            borderRadius: 8,
            background: 'var(--accent-grad)',
            display: 'grid',
            placeItems: 'center',
            boxShadow: 'var(--shadow-glow)',
          }}
        >
          <span style={{ fontWeight: 800, fontSize: 15, color: '#fff' }}>✦</span>
        </div>
        <div style={{ fontWeight: 700, fontSize: 14.5, letterSpacing: '-0.02em' }}>Aperture</div>
        <span className="dim" style={{ fontSize: 12.5 }}>
          quant engine cockpit
        </span>
        <span className="pill pill-neutral" style={{ marginLeft: 2 }}>
          Prototype
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span className="dim" style={{ fontSize: 12.5 }}>
          Preview as
        </span>
        <Segmented
          value={platform}
          onChange={setPlatform}
          size="sm"
          options={[
            { value: 'web', label: 'Web' },
            { value: 'ios', label: 'iOS' },
          ]}
        />
      </div>
    </div>
  );
}
