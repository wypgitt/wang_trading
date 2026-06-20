// ---------------------------------------------------------------------------
// ApiEnvelope — mirrors src/web/envelope.py. Every BFF response carries this
// trust metadata; the UI's persistent trust state binds directly to it. Today
// the data is mock, but the SHAPE is the real contract so swapping in the BFF
// (data/api.ts) is a one-file change.
// ---------------------------------------------------------------------------
import { STALE_THRESHOLD_SECONDS } from '@/lib/colors';

export interface ApiEnvelope<T> {
  data: T;
  as_of: string; // ISO timestamp of the snapshot
  source: string;
  staleness_seconds: number;
  source_freshness: Record<string, number>;
  model_version: string | null;
  regime: Regime | null; // null today — RegimeDetector has zero runtime callers
  warnings: string[];
  errors: EnvelopeError[];
  request_id: string;
}

export interface EnvelopeError {
  field?: string;
  code: string;
  message: string;
}

export interface Regime {
  label: string;
  probabilities: {
    trending_up: number;
    trending_down: number;
    mean_reverting: number;
    high_volatility: number;
  };
}

// The seven canonical data states. Every data-bearing slot resolves to exactly
// one — "Not yet available" is first-class, never a fake zero. See the design
// doc §3 "The canonical honest data-state system".
export type DataStateKind =
  | 'live'
  | 'loading'
  | 'empty'
  | 'stale'
  | 'error'
  | 'coming' // Not yet available (ABSENT field — has a real unlock condition)
  | 'modelGated'; // specialization of coming: needs an MLflow production model

export interface TrustState {
  mode: 'PAPER' | 'LIVE';
  modelLoaded: boolean;
  modelVersion: string | null;
  asOf: string;
  stalenessSeconds: number;
  stale: boolean;
  requestId: string;
}

export function deriveTrust<T>(env: ApiEnvelope<T>): TrustState {
  return {
    mode: 'PAPER', // engine is read-only; live_orders_sent=0
    modelLoaded: env.model_version != null,
    modelVersion: env.model_version,
    asOf: env.as_of,
    stalenessSeconds: env.staleness_seconds,
    stale: env.staleness_seconds > STALE_THRESHOLD_SECONDS,
    requestId: env.request_id,
  };
}

let _seq = 1000;
/** Deterministic request-id surrogate (real BFF assigns these). */
export function nextRequestId(): string {
  _seq += 1;
  return `req_${_seq.toString(36)}f${(_seq * 7).toString(36)}`;
}

export function envelope<T>(data: T, over: Partial<ApiEnvelope<T>> = {}): ApiEnvelope<T> {
  return {
    data,
    as_of: new Date(0).toISOString(),
    source: 'mock',
    staleness_seconds: 4,
    source_freshness: { bars: 2.1, signals: 5.2, snapshot: 4.0 },
    model_version: 'meta_v1.7.2',
    regime: null,
    warnings: [],
    errors: [],
    request_id: nextRequestId(),
    ...over,
  };
}
