'use client';
import { ReactNode } from 'react';
import { Icon } from './Icon';
import { ComingState } from './ui/honesty';
import { ScreenSpec, LOCK_LABEL } from '../lib/readiness';
import { useChartColors } from '../lib/theme';

// A dignified "Coming" destination — not a toast, not a 404. Standard template
// reused for every locked screen: title + purpose, a muted wireframe ghost of the
// eventual layout, and the exact unlock condition verbatim from data_readiness.md.
export function ComingScreen({ spec, ghost }: { spec: ScreenSpec; ghost?: ReactNode }) {
  const C = useChartColors();
  const variant = spec.lock ?? 'gated';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{ display: 'grid', placeItems: 'center', width: 40, height: 40, borderRadius: 12, background: `${C.accent}14`, color: C.accent }}>
          <Icon name={spec.icon} size={20} />
        </span>
        <div>
          <h2 style={{ fontSize: 22 }}>{spec.label}</h2>
          {spec.purpose && <div className="muted" style={{ fontSize: 13.5, marginTop: 2 }}>{spec.purpose}</div>}
        </div>
        <div style={{ flex: 1 }} />
        <span className="chip" style={{ color: C.text3 }}>
          <Icon name={variant === 'wireable' ? 'settings' : variant === 'deferred' ? 'refresh' : 'lock'} size={13} />
          {LOCK_LABEL[variant]}
        </span>
      </div>

      <ComingState
        title={`${spec.label} — coming as the engine lands it`}
        unlock={spec.unlock ?? 'Coming as the engine lands it.'}
        wave={spec.wave}
        variant={variant}
        ghost={ghost ?? <DefaultGhost />}
      />

      {variant === 'wireable' && (
        <div className="card" style={{ padding: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: C.warn, display: 'inline-flex' }}><Icon name="settings" size={18} /></span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, fontSize: 13.5 }}>Wire this next</div>
            <div className="muted" style={{ fontSize: 12.5, marginTop: 2 }}>
              Needs no new persistence — only a BFF stub rewire. A fast-follow candidate after v1.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// A low-opacity skeleton of "real panels" so the depth is legible without numbers.
function DefaultGhost() {
  const C = useChartColors();
  return (
    <div style={{ width: '88%', maxWidth: 880, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14 }}>
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} style={{ height: 64, borderRadius: 12, background: C.surface2 }} />
        ))}
      </div>
      <div style={{ height: 180, borderRadius: 14, background: C.surface2 }} />
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 14 }}>
        <div style={{ height: 120, borderRadius: 14, background: C.surface2 }} />
        <div style={{ height: 120, borderRadius: 14, background: C.surface2 }} />
      </div>
    </div>
  );
}
