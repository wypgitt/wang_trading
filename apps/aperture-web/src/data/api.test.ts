import { describe, it, expect } from 'vitest';
import { getOverview, getMarkets, getModel, getStrategy, TRADE_IDEAS, MODEL } from './api';
import { deriveTrust, envelope } from './envelope';
import { FAMILY_READINESS } from '../lib/readiness';

const ACTIVE = new Set(Object.entries(FAMILY_READINESS).filter(([, v]) => v.active).map(([k]) => k));

describe('data honesty — live ideas only come from ACTIVE families', () => {
  it('every trade idea is attributed to one of the 4 active families', () => {
    for (const idea of TRADE_IDEAS) {
      expect(ACTIVE.has(idea.strategy ?? ''), `${idea.symbol} → ${idea.strategy}`).toBe(true);
    }
  });

  it('no dormant family has any live idea', () => {
    for (const [fam, v] of Object.entries(FAMILY_READINESS)) {
      if (!v.active) {
        expect(getStrategy(fam).data.ideas, `${fam} should be empty`).toHaveLength(0);
      }
    }
  });
});

describe('Overview — decision dashboard, NAV is coming not faked', () => {
  const env = getOverview();
  it('returns null NAV with the unlock warning (never $0)', () => {
    expect(env.data.nav).toBeNull();
    expect(env.data.navHistory).toBeNull();
    expect(env.warnings.join(' ')).toMatch(/no persisted portfolio/i);
  });
  it('top actionable are BUY/SELL only, pre-sorted by |target weight|', () => {
    const top = env.data.topActionable;
    expect(top.every((i) => i.action === 'BUY' || i.action === 'SELL')).toBe(true);
    const w = top.map((i) => Math.abs(i.targetWeight));
    expect(w).toEqual([...w].sort((a, b) => b - a));
  });
  it('engine pulse sums real per-stage latency', () => {
    expect(env.data.enginePulse.totalSeconds).toBeGreaterThan(0);
    expect(env.data.enginePulse.stages.length).toBeGreaterThan(0);
  });
});

describe('Model — gate flags are false ("not run"), not faked metrics', () => {
  it('CPCV/DSR/PBO are uniformly false (retrain gate broken)', () => {
    expect(getModel().data.gates).toEqual({ cpcv: false, dsr: false, pbo: false });
  });
  it('exposes real MLflow run stats', () => {
    expect(MODEL.cvScore).toBeGreaterThan(0);
    expect(MODEL.trainingEvents).toBeGreaterThan(0);
    expect(MODEL.runId).toBeTruthy();
  });
});

describe('envelope trust state', () => {
  it('flags stale only past the threshold', () => {
    expect(deriveTrust(envelope(null, { staleness_seconds: 4 })).stale).toBe(false);
    expect(deriveTrust(envelope(null, { staleness_seconds: 120 })).stale).toBe(true);
  });
  it('mode is always PAPER (engine is read-only)', () => {
    expect(deriveTrust(getMarkets()).mode).toBe('PAPER');
  });
});
