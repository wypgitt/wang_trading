'use client';
import { useChartColors } from '../../lib/theme';

export interface BarItem {
  label: string;
  value: number;
  color?: string;
  sub?: string;
}

interface Props {
  items: BarItem[];
  max?: number;
  fmt?: (v: number) => string;
  signed?: boolean;
  labelWidth?: number;
}

/** Horizontal labeled bars — feature importance, contributions, exposures. */
export function MiniBars({ items, max, fmt, signed = false, labelWidth = 140 }: Props) {
  const C = useChartColors();
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
      {items.map((it) => {
        const frac = Math.min(1, Math.abs(it.value) / m);
        const col = it.color ?? (signed ? (it.value >= 0 ? C.pos : C.neg) : C.accent);
        return (
          <div key={it.label} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{ width: labelWidth, flex: 'none', fontSize: 12.5, color: C.text2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
              title={it.label}
            >
              <span className="mono">{it.label}</span>
            </div>
            <div style={{ flex: 1, height: 9, background: C.surfaceInset, borderRadius: 99, position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${frac * 100}%`, background: col, borderRadius: 99 }} />
            </div>
            <div className="num" style={{ width: 62, textAlign: 'right', fontSize: 12.5, color: C.text1, flex: 'none' }}>
              {it.sub ?? (fmt ? fmt(it.value) : it.value.toFixed(2))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
