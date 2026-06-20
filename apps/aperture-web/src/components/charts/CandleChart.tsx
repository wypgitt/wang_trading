'use client';
import { useWidth } from '../../lib/useWidth';
import { Candle } from '../../lib/rng';
import { useChartColors } from '../../lib/theme';
import { fmtPrice } from '../../lib/format';

interface Props {
  candles: Candle[];
  height?: number;
}

/** Custom SVG candlestick + volume — full styling control, no extra deps. */
export function CandleChart({ candles, height = 340 }: Props) {
  const C = useChartColors();
  const [ref, w] = useWidth<HTMLDivElement>();
  const padL = 6;
  const padR = 54;
  const padT = 8;
  const padB = 20;
  const volH = 44;
  const gap = 10;
  const priceH = height - volH - padT - padB - gap;

  const highs = candles.map((c) => c.h);
  const lows = candles.map((c) => c.l);
  const max = Math.max(...highs);
  const min = Math.min(...lows);
  const range = max - min || 1;
  const plotW = Math.max(0, w - padL - padR);
  const n = candles.length;
  const cw = n ? plotW / n : 0;
  const bodyW = Math.max(1.5, cw * 0.62);
  const y = (p: number) => padT + (1 - (p - min) / range) * priceH;
  const volMax = Math.max(...candles.map((c) => c.v)) || 1;
  const volTop = padT + priceH + gap;

  const gridY = [0, 0.25, 0.5, 0.75, 1].map((f) => min + range * f);
  const last = candles[candles.length - 1];
  const lastUp = last && last.c >= last.o;

  return (
    <div ref={ref} style={{ width: '100%' }}>
      {w > 0 && (
        <svg width={w} height={height} style={{ display: 'block' }}>
          {gridY.map((gv, i) => (
            <g key={i}>
              <line x1={padL} x2={padL + plotW} y1={y(gv)} y2={y(gv)} stroke={C.grid} strokeWidth={1} />
              <text x={w - padR + 8} y={y(gv) + 3.5} fill={C.text3} fontSize={11} fontFamily="var(--font-mono)">
                {fmtPrice(gv)}
              </text>
            </g>
          ))}
          {candles.map((c, i) => {
            const x = padL + i * cw + cw / 2;
            const up = c.c >= c.o;
            const col = up ? C.pos : C.neg;
            const bodyTop = Math.min(y(c.o), y(c.c));
            const bodyH = Math.max(1, Math.abs(y(c.o) - y(c.c)));
            const vh = (c.v / volMax) * volH;
            return (
              <g key={i}>
                <line x1={x} x2={x} y1={y(c.h)} y2={y(c.l)} stroke={col} strokeWidth={1} opacity={0.9} />
                <rect x={x - bodyW / 2} y={bodyTop} width={bodyW} height={bodyH} fill={col} rx={1} />
                <rect x={x - bodyW / 2} y={volTop + (volH - vh)} width={bodyW} height={vh} fill={col} opacity={0.32} rx={0.5} />
              </g>
            );
          })}
          {last && (
            <g>
              <line
                x1={padL}
                x2={padL + plotW}
                y1={y(last.c)}
                y2={y(last.c)}
                stroke={lastUp ? C.pos : C.neg}
                strokeWidth={1}
                strokeDasharray="2 3"
                opacity={0.7}
              />
              <rect x={w - padR + 2} y={y(last.c) - 9} width={padR - 4} height={18} rx={4} fill={lastUp ? C.pos : C.neg} />
              <text
                x={w - padR + (padR - 4) / 2 + 1}
                y={y(last.c) + 3.5}
                fill="#06120d"
                fontSize={11}
                fontWeight={700}
                textAnchor="middle"
                fontFamily="var(--font-mono)"
              >
                {fmtPrice(last.c)}
              </text>
            </g>
          )}
        </svg>
      )}
    </div>
  );
}
