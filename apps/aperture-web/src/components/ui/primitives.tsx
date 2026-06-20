'use client';
import { ReactNode } from 'react';
import { Action, Regime, Sym } from '../../data/mock';
import { ASSET_TINT, REGIME_HEX, REGIME_LABEL } from '../../lib/colors';
import { useChartColors } from '../../lib/theme';
import { fmtPctSigned, fmtSigned } from '../../lib/format';

// ---- Asset glyph -----------------------------------------------------------
export function AssetGlyph({ sym, size = 34 }: { sym: Pick<Sym, 'symbol' | 'type'>; size?: number }) {
  const C = useChartColors();
  const tint = ASSET_TINT[sym.type] ?? C.accent;
  const txt = sym.symbol.slice(0, sym.type === 'crypto' ? 3 : 4);
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: size * 0.3,
        flex: 'none',
        display: 'grid',
        placeItems: 'center',
        background: `${tint}22`,
        color: tint,
        fontWeight: 700,
        fontSize: size * (txt.length > 3 ? 0.28 : 0.34),
        letterSpacing: '-0.02em',
        border: `1px solid ${tint}33`,
      }}
    >
      {txt}
    </div>
  );
}

// ---- Action pill -----------------------------------------------------------
const ACTION_MAP: Record<Action, [string, string]> = {
  BUY: ['pill-buy', 'Buy'],
  SELL: ['pill-sell', 'Sell'],
  WATCH: ['pill-watch', 'Watch'],
  MODEL_REQUIRED: ['pill-neutral', 'Model?'],
  NO_DATA: ['pill-neutral', 'No data'],
};
export function ActionPill({ action }: { action: Action }) {
  const [cls, label] = ACTION_MAP[action];
  return <span className={`pill ${cls}`}>{label}</span>;
}

// ---- Signed delta ----------------------------------------------------------
export function Delta({
  value,
  kind = 'pct',
  dp = 2,
  arrow = true,
  size,
  weight = 600,
}: {
  value: number;
  kind?: 'pct' | 'num';
  dp?: number;
  arrow?: boolean;
  size?: number;
  weight?: number;
}) {
  const cls = value > 0 ? 'pos' : value < 0 ? 'neg' : 'dim';
  const txt = kind === 'pct' ? fmtPctSigned(value, dp) : fmtSigned(value, dp);
  return (
    <span className={`num ${cls}`} style={{ fontSize: size, fontWeight: weight }}>
      {arrow && value !== 0 ? (value > 0 ? '▲ ' : '▼ ') : ''}
      {txt}
    </span>
  );
}

// ---- Stat block ------------------------------------------------------------
export function Stat({
  label,
  value,
  sub,
  subColor,
  valueSize = 22,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  subColor?: string;
  valueSize?: number;
}) {
  const C = useChartColors();
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div className="num" style={{ fontSize: valueSize, fontWeight: 680, marginTop: 5, letterSpacing: '-0.02em' }}>
        {value}
      </div>
      {sub != null && <div style={{ fontSize: 12.5, marginTop: 3, color: subColor ?? C.text2 }}>{sub}</div>}
    </div>
  );
}

// ---- Progress bar ----------------------------------------------------------
export function Bar({ value, max = 1, color, height = 6 }: { value: number; max?: number; color?: string; height?: number }) {
  const C = useChartColors();
  const barColor = color ?? C.accent;
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div style={{ height, background: C.surfaceInset, borderRadius: 99, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: barColor, borderRadius: 99 }} />
    </div>
  );
}

// ---- Regime stacked bar ----------------------------------------------------
const REGIME_ORDER = ['trending_up', 'mean_reverting', 'high_volatility', 'trending_down'] as const;
export function RegimeBar({ probs, height = 8 }: { probs: Regime['probabilities']; height?: number }) {
  return (
    <div style={{ display: 'flex', height, borderRadius: 99, overflow: 'hidden', gap: 2 }}>
      {REGIME_ORDER.map((k) => {
        const v = probs[k];
        if (v <= 0) return null;
        return <div key={k} style={{ width: `${v * 100}%`, background: REGIME_HEX[k] }} title={`${REGIME_LABEL[k]} ${(v * 100).toFixed(0)}%`} />;
      })}
    </div>
  );
}

export function RegimeChip({ regime }: { regime: Regime }) {
  const top = regime.label;
  const col = REGIME_HEX[top];
  const p = regime.probabilities[top as keyof Regime['probabilities']];
  return (
    <span className="chip" style={{ borderColor: `${col}40` }}>
      <span className="dot" style={{ background: col }} />
      {REGIME_LABEL[top]} · <span className="num">{(p * 100).toFixed(0)}%</span>
    </span>
  );
}

// ---- Donut -----------------------------------------------------------------
export function Donut({
  segments,
  size = 128,
  thickness = 15,
  center,
}: {
  segments: { label: string; value: number; color: string }[];
  size?: number;
  thickness?: number;
  center?: ReactNode;
}) {
  const C = useChartColors();
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  const r = (size - thickness) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circ = 2 * Math.PI * r;
  let off = 0;
  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.surfaceInset} strokeWidth={thickness} />
        <g transform={`rotate(-90 ${cx} ${cy})`}>
          {segments.map((s) => {
            const frac = s.value / total;
            const dash = frac * circ;
            const el = (
              <circle
                key={s.label}
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke={s.color}
                strokeWidth={thickness}
                strokeDasharray={`${dash} ${circ - dash}`}
                strokeDashoffset={-off}
                strokeLinecap="butt"
              />
            );
            off += dash;
            return el;
          })}
        </g>
      </svg>
      {center != null && (
        <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', textAlign: 'center' }}>{center}</div>
      )}
    </div>
  );
}

// ---- Probability ring ------------------------------------------------------
export function ProbRing({ value, size = 52, label }: { value: number | null; size?: number; label?: string }) {
  const C = useChartColors();
  const v = value ?? 0;
  const r = (size - 6) / 2;
  const cx = size / 2;
  const circ = 2 * Math.PI * r;
  const col = v >= 0.65 ? C.pos : v >= 0.55 ? C.warn : C.text3;
  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={cx} cy={cx} r={r} fill="none" stroke={C.surfaceInset} strokeWidth={5} />
        <g transform={`rotate(-90 ${cx} ${cx})`}>
          <circle cx={cx} cy={cx} r={r} fill="none" stroke={col} strokeWidth={5} strokeDasharray={`${v * circ} ${circ}`} strokeLinecap="round" />
        </g>
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>
        <span className="num" style={{ fontSize: size * 0.27, fontWeight: 700, color: col }}>
          {value == null ? '—' : value.toFixed(2)}
        </span>
      </div>
      {label && <div className="eyebrow" style={{ textAlign: 'center', marginTop: 4 }}>{label}</div>}
    </div>
  );
}

// ---- Segmented control -----------------------------------------------------
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  size?: 'sm' | 'md';
}) {
  const C = useChartColors();
  return (
    <div style={{ display: 'inline-flex', background: C.surfaceInset, border: `1px solid ${C.border}`, borderRadius: 11, padding: 3, gap: 2 }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            data-value={o.value}
            onClick={() => onChange(o.value)}
            className="num"
            style={{
              height: size === 'sm' ? 26 : 32,
              padding: size === 'sm' ? '0 11px' : '0 14px',
              borderRadius: 8,
              fontSize: size === 'sm' ? 12 : 13,
              fontWeight: 600,
              color: active ? '#fff' : C.text2,
              background: active ? C.surface3 : 'transparent',
              transition: 'all 0.12s ease',
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

// ---- Status dot ------------------------------------------------------------
export function StatusDot({ ok, live }: { ok: boolean; live?: boolean }) {
  const C = useChartColors();
  return <span className={`dot ${live ? 'live-dot' : ''}`} style={{ background: ok ? C.pos : C.neg }} />;
}

// ---- Section header --------------------------------------------------------
export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
      <h3 style={{ fontSize: 15.5 }}>{children}</h3>
      {right}
    </div>
  );
}
