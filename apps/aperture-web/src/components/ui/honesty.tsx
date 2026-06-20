'use client';

import { CSSProperties, ReactNode } from 'react';
import { Panel } from './Panel';
import { Icon } from '../Icon';
import { useChartColors } from '../../lib/theme';
import { DataStateKind } from '../../data/envelope';

// ---------------------------------------------------------------------------
// Honesty primitives — the shared, dignified visual language for engine outputs
// that are real in the codebase but NOT yet persisted/surfaced, so the UI must
// never fabricate a number for them. See docs/data_readiness.md + the v1 design
// doc §3 "The canonical honest data-state system".
//
//   Skeleton        — shimmer placeholder (Loading)
//   ComingState     — block "Not yet available" with a concrete unlock condition
//   ComingSoon      — a full titled Panel wrapping ComingState
//   DataUnavailable — inline slot marker (ring or chip) for a single absent metric
//   DataState       — the wrapper that enforces the seven-state switch
// ---------------------------------------------------------------------------

const ICON_FOR: Record<string, string> = { gated: 'lock', wireable: 'settings', deferred: 'refresh', model: 'model' };

/** Shimmer placeholder matching a value's footprint. */
export function Skeleton({ w = '100%', h = 14, r = 7, style }: { w?: number | string; h?: number | string; r?: number; style?: CSSProperties }) {
  return <div className="skeleton" style={{ width: w, height: h, borderRadius: r, ...style }} />;
}

/** Block "coming as the engine lands it" state with a concrete unlock condition. */
export function ComingState({
  title = 'Coming as the engine lands it',
  unlock,
  wave,
  variant = 'gated',
  compact = false,
  ghost,
}: {
  title?: ReactNode;
  unlock: string;
  wave?: number;
  variant?: 'gated' | 'wireable' | 'deferred' | 'model';
  compact?: boolean;
  /** optional low-opacity wireframe ghost of the eventual layout */
  ghost?: ReactNode;
}) {
  const C = useChartColors();
  const icon = ICON_FOR[variant] ?? 'lock';
  return (
    <div
      style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        gap: 10,
        padding: compact ? '20px 16px' : '38px 24px',
        border: `1px dashed ${C.borderStrong}`,
        borderRadius: 14,
        background: C.surfaceInset,
        overflow: 'hidden',
      }}
    >
      {ghost != null && (
        <div aria-hidden style={{ position: 'absolute', inset: 0, opacity: 0.16, pointerEvents: 'none', display: 'grid', placeItems: 'center' }}>
          {ghost}
        </div>
      )}
      <span style={{ position: 'relative', display: 'grid', placeItems: 'center', width: 38, height: 38, borderRadius: 11, background: `${C.accent}14`, color: C.accent }}>
        <Icon name={icon} size={18} />
      </span>
      <div style={{ position: 'relative', fontSize: 14, fontWeight: 650, color: C.text1 }}>{title}</div>
      <div className="muted" style={{ position: 'relative', fontSize: 12.5, lineHeight: 1.55, maxWidth: 420 }}>
        {unlock}
      </div>
      {wave != null && (
        <span className="eyebrow" style={{ position: 'relative', color: C.accent, letterSpacing: '0.06em' }}>
          Wave {wave}
        </span>
      )}
    </div>
  );
}

/** A full titled Panel whose body is a ComingState. */
export function ComingSoon({
  title,
  subtitle,
  unlock,
  wave,
  variant,
  ghost,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  unlock: string;
  wave?: number;
  variant?: 'gated' | 'wireable' | 'deferred' | 'model';
  ghost?: ReactNode;
}) {
  return (
    <Panel title={title} subtitle={subtitle}>
      <ComingState unlock={unlock} wave={wave} variant={variant} ghost={ghost} />
    </Panel>
  );
}

/**
 * Inline marker for a single absent metric. The unlock condition rides on the
 * tooltip so the slot never shows a bare '—' or a fake number.
 *   size set   → a ProbRing-sized dashed ring (slots beside real rings)
 *   size unset → a compact "Coming" chip (replaces a stat value)
 */
export function DataUnavailable({
  unlock,
  label,
  size,
  modelGated = false,
}: {
  unlock: string;
  label?: string;
  size?: number;
  modelGated?: boolean;
}) {
  const C = useChartColors();
  const text = modelGated ? 'Model?' : 'Coming';
  if (size) {
    const r = (size - 6) / 2;
    const cx = size / 2;
    return (
      <div title={unlock} style={{ cursor: 'help' }}>
        <div style={{ position: 'relative', width: size, height: size }}>
          <svg width={size} height={size} aria-hidden="true">
            <circle cx={cx} cy={cx} r={r} fill="none" stroke={C.border} strokeWidth={5} strokeDasharray="2 6" strokeLinecap="round" />
          </svg>
          <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>
            <Icon name={modelGated ? 'model' : 'lock'} size={Math.round(size * 0.3)} className="dim" />
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
    <span title={unlock} className="chip" style={{ cursor: 'help', height: 24, color: C.text3, gap: 5 }}>
      <Icon name={modelGated ? 'model' : 'lock'} size={11} />
      {text}
    </span>
  );
}

/**
 * The wrapper every data element should pass through. Centralizes the seven-state
 * switch so the prohibition ("never render 0/—/N/A for an ABSENT field") is
 * structurally enforced, not hoped for.
 */
export function DataState({
  state,
  children,
  unlock,
  label,
  requestId,
  skeleton,
  onRetry,
}: {
  state: DataStateKind;
  children?: ReactNode; // the live value
  unlock?: string;
  label?: string;
  requestId?: string;
  skeleton?: ReactNode;
  onRetry?: () => void;
}) {
  const C = useChartColors();
  switch (state) {
    case 'live':
    case 'stale':
      return <>{children}</>;
    case 'loading':
      return <>{skeleton ?? <Skeleton />}</>;
    case 'empty':
      return <span className="dim">—</span>;
    case 'coming':
      return <DataUnavailable unlock={unlock ?? 'Coming as the engine lands it.'} label={label} />;
    case 'modelGated':
      return <DataUnavailable unlock={unlock ?? 'Load an MLflow production model. Currently MODEL_REQUIRED.'} label={label} modelGated />;
    case 'error':
      return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: C.neg, fontSize: 12.5 }}>
          Couldn’t load{label ? ` ${label}` : ''}.
          {onRetry && (
            <button className="btn" style={{ height: 24, padding: '0 10px', fontSize: 12 }} onClick={onRetry}>
              Retry
            </button>
          )}
          {requestId && <span className="mono dim" style={{ fontSize: 11 }}>req: {requestId}</span>}
        </span>
      );
    default:
      return null;
  }
}
