import { describe, it, expect } from 'vitest';
import { NAV, FAMILY_READINESS, familyCounts, familyReadiness, screenById } from './readiness';

describe('screen readiness map', () => {
  it('has exactly 5 LIVE destinations and 8 COMING (the honest v1 IA)', () => {
    const all = NAV.flatMap((g) => g.items);
    expect(all.filter((s) => s.readiness === 'live')).toHaveLength(5);
    expect(all.filter((s) => s.readiness === 'coming')).toHaveLength(8);
  });

  it('gives every COMING screen a verbatim unlock condition + wave', () => {
    for (const s of NAV.flatMap((g) => g.items)) {
      if (s.readiness === 'coming') {
        expect(s.unlock, `${s.id} unlock`).toBeTruthy();
        expect(s.wave, `${s.id} wave`).toBeGreaterThan(0);
      }
    }
  });

  it('resolves screens by id', () => {
    expect(screenById('overview')?.readiness).toBe('live');
    expect(screenById('portfolio')?.readiness).toBe('coming');
  });
});

describe('strategy family readiness', () => {
  it('is 4 active · 6 inactive (single-symbol bars path)', () => {
    const c = familyCounts();
    expect(c).toEqual({ active: 4, inactive: 6, total: 10 });
  });

  it('marks only the four bars-families active', () => {
    const active = Object.entries(FAMILY_READINESS).filter(([, v]) => v.active).map(([k]) => k);
    expect(active.sort()).toEqual(['donchian_breakout', 'ma_crossover', 'mean_reversion', 'ts_momentum']);
  });

  it('gives every dormant family a reason', () => {
    for (const [id, v] of Object.entries(FAMILY_READINESS)) {
      if (!v.active) expect(v.reason, `${id} reason`).toBeTruthy();
    }
  });

  it('falls back to inactive for an unknown family', () => {
    expect(familyReadiness('nope').active).toBe(false);
  });
});
