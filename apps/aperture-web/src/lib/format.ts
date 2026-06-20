// Formatting helpers — consistent number/currency/percent rendering.

export const fmtUsd = (v: number, dp = 2): string =>
  v.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: dp,
    maximumFractionDigits: dp,
  });

export const fmtCompact = (v: number, prefix = '$'): string => {
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e12) return `${sign}${prefix}${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}${prefix}${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}${prefix}${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}${prefix}${(abs / 1e3).toFixed(1)}K`;
  return `${sign}${prefix}${abs.toFixed(0)}`;
};

export const fmtNum = (v: number, dp = 2): string =>
  v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });

export const fmtPrice = (v: number): string => {
  if (v >= 1000) return fmtNum(v, 2);
  if (v >= 1) return fmtNum(v, 2);
  return fmtNum(v, 4);
};

export const fmtPct = (v: number, dp = 2): string => `${(v * 100).toFixed(dp)}%`;

export const fmtPctSigned = (v: number, dp = 2): string =>
  `${v >= 0 ? '+' : ''}${(v * 100).toFixed(dp)}%`;

export const fmtSigned = (v: number, dp = 2): string => `${v >= 0 ? '+' : ''}${v.toFixed(dp)}`;

export const fmtBps = (v: number): string => `${v >= 0 ? '+' : ''}${v.toFixed(1)} bps`;

export const fmtProb = (v: number | null): string => (v == null ? '—' : v.toFixed(2));

export const signClass = (v: number): string => (v > 0 ? 'pos' : v < 0 ? 'neg' : '');

export const sideLabel = (side: number): string =>
  side > 0 ? 'LONG' : side < 0 ? 'SHORT' : 'FLAT';

export const fmtTimeAgo = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
};
