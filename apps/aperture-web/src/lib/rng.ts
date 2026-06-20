// Deterministic, seeded series generators so charts are stable across reloads
// (and screenshots reproducible). Plain Math.random is intentionally avoided.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller standard normal from a uniform generator.
function gauss(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export interface Point {
  t: number; // index / timestamp surrogate
  v: number;
}

/** Geometric-ish random walk with drift + mean pull, returns price points. */
export function genWalk(
  seed: number,
  n: number,
  start: number,
  driftPerStep: number,
  vol: number,
): Point[] {
  const rand = mulberry32(seed);
  const out: Point[] = [];
  let v = start;
  for (let i = 0; i < n; i++) {
    const shock = gauss(rand) * vol;
    v = v * (1 + driftPerStep + shock);
    if (v < start * 0.04) v = start * 0.04; // floor
    out.push({ t: i, v: round(v) });
  }
  return out;
}

export interface Candle {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export function genCandles(
  seed: number,
  n: number,
  start: number,
  driftPerStep: number,
  vol: number,
): Candle[] {
  const rand = mulberry32(seed);
  const out: Candle[] = [];
  let close = start;
  for (let i = 0; i < n; i++) {
    const o = close;
    const drift = driftPerStep + gauss(rand) * vol;
    const c = Math.max(start * 0.04, o * (1 + drift));
    const wick = Math.abs(gauss(rand)) * vol * o;
    const h = Math.max(o, c) + wick * 0.6;
    const l = Math.min(o, c) - wick * 0.6;
    const vlm = (0.6 + rand()) * 1_000_000;
    out.push({ t: i, o: round(o), h: round(h), l: round(l), c: round(c), v: Math.round(vlm) });
    close = c;
  }
  return out;
}

function round(v: number): number {
  if (v >= 1000) return Math.round(v * 100) / 100;
  if (v >= 1) return Math.round(v * 100) / 100;
  return Math.round(v * 10000) / 10000;
}

/** Pick a deterministic subset/jitter helper. */
export function jitter(seed: number, base: number, spread: number): number {
  const r = mulberry32(seed)();
  return base + (r - 0.5) * 2 * spread;
}
