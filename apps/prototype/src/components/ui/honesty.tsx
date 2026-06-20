import { ReactNode } from 'react';
import { Panel } from './Panel';
import { Icon } from '../Icon';
import { C } from '../../lib/colors';

// Honesty primitives — a shared, dignified visual language for engine outputs
// that are real in the codebase but NOT yet persisted or surfaced over HTTP, so
// the UI must never fabricate a number for them. See docs/data_readiness.md.
//
//   ComingState     — block treatment for an empty panel body
//   ComingSoon      — a full titled Panel wrapping ComingState
//   DataUnavailable — inline slot marker (ring or stat) for a single metric

/** Block "coming as the engine lands it" state with a concrete unlock condition. */
export function ComingState({
  title = 'Coming as the engine lands it',
  unlock,
  compact = false,
}: {
  title?: string;
  unlock: string;
  compact?: boolean;
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        gap: 9,
        padding: compact ? '20px 16px' : '34px 20px',
        border: `1px dashed ${C.borderStrong}`,
        borderRadius: 14,
        background: C.surfaceInset,
      }}
    >
      <span
        style={{
          display: 'grid',
          placeItems: 'center',
          width: 36,
          height: 36,
          borderRadius: 11,
          background: `${C.accent}14`,
          color: C.accent,
        }}
      >
        <Icon name="lock" size={18} />
      </span>
      <div style={{ fontSize: 13.5, fontWeight: 600, color: C.text1 }}>{title}</div>
      <div className="muted" style={{ fontSize: 12, lineHeight: 1.5, maxWidth: 340 }}>
        {unlock}
      </div>
    </div>
  );
}

/** A full titled Panel whose body is a ComingState. */
export function ComingSoon({
  title,
  subtitle,
  unlock,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  unlock: string;
}) {
  return (
    <Panel title={title} subtitle={subtitle}>
      <ComingState unlock={unlock} />
    </Panel>
  );
}

/**
 * Inline marker for a single absent metric. The unlock condition is always
 * carried on the tooltip so the slot never shows a bare '—' or a fake number.
 *   size set   → a ProbRing-sized dashed ring (slots beside real rings)
 *   size unset → a compact "Coming" chip (replaces a stat value)
 */
export function DataUnavailable({
  unlock,
  label,
  size,
}: {
  unlock: string;
  label?: string;
  size?: number;
}) {
  if (size) {
    const r = (size - 6) / 2;
    const cx = size / 2;
    return (
      <div title={unlock} style={{ cursor: 'help' }}>
        <div style={{ position: 'relative', width: size, height: size }}>
          <svg width={size} height={size}>
            <circle
              cx={cx}
              cy={cx}
              r={r}
              fill="none"
              stroke={C.border}
              strokeWidth={5}
              strokeDasharray="2 6"
              strokeLinecap="round"
            />
          </svg>
          <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>
            <Icon name="lock" size={Math.round(size * 0.3)} className="dim" />
          </div>
        </div>
        {label && (
          <div className="eyebrow" style={{ textAlign: 'center', marginTop: 4 }}>
            {label}
          </div>
        )}
      </div>
    );
  }
  return (
    <span
      title={unlock}
      className="chip"
      style={{ cursor: 'help', height: 24, color: C.text3, gap: 5 }}
    >
      <Icon name="lock" size={11} />
      Coming
    </span>
  );
}
