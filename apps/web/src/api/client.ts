/**
 * Typed fetch wrapper for the BFF.
 *
 * Every BFF response is wrapped in the envelope defined in
 * docs/api_contracts_v2.md §0.1. The wrapper preserves that shape and
 * surfaces non-2xx responses as ApiError.
 */

export interface RegimeProbabilities {
  trending_up: number;
  trending_down: number;
  mean_reverting: number;
  high_volatility: number;
}

export interface RegimeSnapshot {
  label: string;
  probabilities: RegimeProbabilities;
  as_of: string;
}

export interface SourceFreshness {
  [source: string]: number;
}

export interface ApiErrorItem {
  code: string;
  message: string;
  field: string | null;
}

export interface ApiEnvelope<T> {
  as_of: string;
  source: string;
  staleness_seconds?: number;
  source_freshness?: SourceFreshness;
  model_version?: string | null;
  regime?: RegimeSnapshot | null;
  warnings: string[];
  errors: ApiErrorItem[];
  data: T;
}

export class ApiError extends Error {
  readonly status: number;
  readonly envelope: ApiEnvelope<unknown> | null;

  constructor(message: string, status: number, envelope: ApiEnvelope<unknown> | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.envelope = envelope;
  }
}

const DEFAULT_HEADERS: HeadersInit = {
  Accept: "application/json",
};

/**
 * Fetch a JSON envelope from the BFF.
 *
 * Throws ApiError on non-2xx. Tries to parse the body either way so the
 * caller can render envelope.errors[].message when available.
 */
export async function apiFetch<T>(path: string, init?: RequestInit): Promise<ApiEnvelope<T>> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...DEFAULT_HEADERS,
      ...(init?.headers ?? {}),
    },
  });

  let parsed: ApiEnvelope<T> | null = null;
  try {
    parsed = (await response.json()) as ApiEnvelope<T>;
  } catch {
    parsed = null;
  }

  if (!response.ok) {
    const message = parsed?.errors?.[0]?.message ?? `Request failed: ${response.status}`;
    throw new ApiError(message, response.status, parsed as ApiEnvelope<unknown> | null);
  }

  if (!parsed) {
    throw new ApiError("Empty response body", response.status, null);
  }

  return parsed;
}
