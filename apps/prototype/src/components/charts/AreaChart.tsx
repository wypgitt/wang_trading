import { useId } from 'react';
import {
  Area,
  AreaChart as RAreaChart,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Point } from '../../lib/rng';
import { C } from '../../lib/colors';

interface Props {
  data: Point[];
  color?: string;
  height?: number;
  showTooltip?: boolean;
  showYAxis?: boolean;
  valueFmt?: (v: number) => string;
  baseline?: number | null;
}

function ChartTip({ active, payload, valueFmt }: any) {
  if (!active || !payload?.length) return null;
  const v = payload[0].value as number;
  return (
    <div
      style={{
        background: C.surface3,
        border: `1px solid ${C.borderStrong}`,
        borderRadius: 9,
        padding: '6px 11px',
        fontSize: 12.5,
        fontVariantNumeric: 'tabular-nums',
        boxShadow: '0 10px 30px rgba(0,0,0,0.55)',
      }}
    >
      {valueFmt ? valueFmt(v) : v.toFixed(2)}
    </div>
  );
}

export function AreaChart({
  data,
  color = C.accent,
  height = 80,
  showTooltip = false,
  showYAxis = false,
  valueFmt,
  baseline = null,
}: Props) {
  const gid = 'ag' + useId().replace(/:/g, '');
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RAreaChart data={data} margin={{ top: 6, right: 2, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.34} />
            <stop offset="92%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <YAxis
          hide={!showYAxis}
          domain={['dataMin', 'dataMax']}
          width={showYAxis ? 52 : 0}
          tick={{ fill: C.text3, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={valueFmt}
          orientation="right"
        />
        <XAxis dataKey="t" hide />
        {baseline != null && (
          <ReferenceLine y={baseline} stroke={C.text3} strokeDasharray="3 4" strokeOpacity={0.5} />
        )}
        {showTooltip && (
          <Tooltip
            content={<ChartTip valueFmt={valueFmt} />}
            cursor={{ stroke: C.text2, strokeOpacity: 0.4, strokeWidth: 1 }}
          />
        )}
        <Area
          type="monotone"
          dataKey="v"
          stroke={color}
          strokeWidth={2}
          fill={`url(#${gid})`}
          dot={false}
          isAnimationActive={false}
          activeDot={{ r: 3.5, fill: color, stroke: C.surface1, strokeWidth: 2 }}
        />
      </RAreaChart>
    </ResponsiveContainer>
  );
}
