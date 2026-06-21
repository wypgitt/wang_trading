// ---------------------------------------------------------------------------
// BFF client — the swap point between mock-first and the live FastAPI BFF.
//
// Today the data layer (data/api.ts) returns mock envelopes synchronously, which
// keeps every screen render-pure. When APERTURE_BFF_URL is set, an accessor can
// instead `await fetchEnvelope<T>(path)` to hit the real BFF — the response is
// already an ApiEnvelope<T> (src/web/envelope.py), so the trust contract is
// identical and screens bind to the same shape.
//
// Migration path (per the design doc): introduce React Query, turn each accessor
// into a query hook backed by fetchEnvelope, and the existing Loading/Stale/Error
// data-states light up from real envelope metadata (staleness_seconds, errors[],
// request_id) instead of staying inert.
// ---------------------------------------------------------------------------
import { ApiEnvelope, nextRequestId } from './envelope';

// NEXT_PUBLIC_ so the URL is inlined into the client bundle (screens are client
// components and fetch the BFF directly; the BFF CORS-allows the origin).
// APERTURE_BFF_URL is also accepted for server-side / non-Next callers.
export const BFF_URL =
  process.env.NEXT_PUBLIC_APERTURE_BFF_URL ?? process.env.APERTURE_BFF_URL ?? '';
export const USE_BFF = BFF_URL.length > 0;

export class BffError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly requestId: string,
  ) {
    super(message);
    this.name = 'BffError';
  }
}

/**
 * Fetch + unwrap one ApiEnvelope<T> from the BFF. Throws BffError (carrying the
 * copyable request id) on a transport/HTTP failure so the caller's Error state
 * can render it. Never invents data — a failure surfaces, it does not fall back
 * to a fake value.
 */
export async function fetchEnvelope<T>(path: string, init?: RequestInit): Promise<ApiEnvelope<T>> {
  if (!USE_BFF) {
    throw new BffError('APERTURE_BFF_URL is not configured (mock-first mode).', 0, nextRequestId());
  }
  const url = `${BFF_URL.replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;
  const res = await fetch(url, {
    ...init,
    headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
  });
  const reqId = res.headers.get('x-request-id') ?? nextRequestId();
  const body = (await res.json().catch(() => null)) as ApiEnvelope<T> | null;
  // The BFF returns a full envelope even on honest errors (e.g. 503
  // SNAPSHOT_UNAVAILABLE, 404 NOT_FOUND): surface it so the screen's Error
  // state renders the real code + request id instead of a synthetic message.
  // Only a transport failure / non-envelope body throws.
  if (!res.ok) {
    if (body && typeof body === 'object' && Array.isArray(body.errors)) {
      return { ...body, request_id: body.request_id ?? reqId };
    }
    throw new BffError(`BFF ${res.status} for ${path}`, res.status, reqId);
  }
  if (!body) {
    throw new BffError(`BFF ${res.status} returned no body for ${path}`, res.status, reqId);
  }
  return { ...body, request_id: body.request_id ?? reqId };
}
