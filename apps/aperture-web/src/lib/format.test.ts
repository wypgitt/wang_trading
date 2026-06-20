import { describe, it, expect } from 'vitest';
import { fmtPctSigned, fmtCompact, fmtProb, fmtTimeAgo, sideLabel, signClass } from './format';

describe('format helpers', () => {
  it('signs percentages and keeps tabular precision', () => {
    expect(fmtPctSigned(0.092, 1)).toBe('+9.2%');
    expect(fmtPctSigned(-0.052, 1)).toBe('-5.2%');
    expect(fmtPctSigned(0)).toBe('+0.00%');
  });

  it('compacts large numbers with a currency prefix', () => {
    expect(fmtCompact(2.9e12)).toBe('$2.90T');
    expect(fmtCompact(3.4e9)).toBe('$3.40B');
    expect(fmtCompact(142_300_000)).toBe('$142.30M');
    expect(fmtCompact(-40000)).toBe('-$40.0K');
  });

  it('reserves the em-dash for a legitimately null probability', () => {
    expect(fmtProb(null)).toBe('—');
    expect(fmtProb(0.71)).toBe('0.71');
  });

  it('renders relative freshness', () => {
    expect(fmtTimeAgo(4)).toBe('4s ago');
    expect(fmtTimeAgo(120)).toBe('2m ago');
    expect(fmtTimeAgo(7200)).toBe('2h ago');
  });

  it('maps side + sign to semantic labels', () => {
    expect(sideLabel(1)).toBe('LONG');
    expect(sideLabel(-1)).toBe('SHORT');
    expect(signClass(1)).toBe('pos');
    expect(signClass(-1)).toBe('neg');
  });
});
