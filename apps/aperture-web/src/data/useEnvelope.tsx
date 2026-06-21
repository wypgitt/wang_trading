'use client';
// ---------------------------------------------------------------------------
// Async data-loading boundary. The data layer (data/api.ts) is async (mock
// resolves instantly; the live BFF fetches). `useEnvelope` runs the fetch and
// exposes {env, isLoading, reload}; `Loaded` renders a skeleton while loading
// and then hands the View a guaranteed-present `data` plus the full `env` so
// the screen's own Stale / Error / Empty / Coming states light up from real
// envelope metadata.
//
// Accessors NEVER reject — a transport/HTTP failure is normalized into an
// error envelope with a valid-empty `data` shape + errors[] — so a screen can
// always render (its error branch shows the copyable request id), never crash.
// ---------------------------------------------------------------------------
import { useEffect, useState } from 'react';
import { ApiEnvelope } from './envelope';

export interface EnvelopeState<T> {
  env: ApiEnvelope<T> | null;
  isLoading: boolean;
  reload: () => void;
}

export function useEnvelope<T>(
  fetcher: () => Promise<ApiEnvelope<T>>,
  deps: unknown[] = [],
): EnvelopeState<T> {
  const [env, setEnv] = useState<ApiEnvelope<T> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    setIsLoading(true);
    fetcher()
      .then((e) => {
        if (alive) {
          setEnv(e);
          setIsLoading(false);
        }
      })
      .catch(() => {
        // Defensive: accessors envelope their own errors, so this is unexpected.
        if (alive) setIsLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { env, isLoading, reload: () => setNonce((n) => n + 1) };
}

export interface ViewProps<T> {
  data: T;
  env: ApiEnvelope<T>;
  reload: () => void;
}

export function Loaded<T>({
  fetcher,
  deps = [],
  View,
}: {
  fetcher: () => Promise<ApiEnvelope<T>>;
  deps?: unknown[];
  View: React.ComponentType<ViewProps<T>>;
}) {
  const { env, isLoading, reload } = useEnvelope(fetcher, deps);
  if (isLoading || env == null) return <PageLoading />;
  // Accessors normalize to a valid-empty `data` even on error, so `data` is
  // present here; the View still reads env.errors / staleness for its states.
  const data = env.data as T;
  return <View data={data} env={env} reload={reload} />;
}

export function PageLoading() {
  return (
    <div
      role="status"
      aria-busy="true"
      aria-label="Loading"
      style={{ display: 'flex', flexDirection: 'column', gap: 16 }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.55fr) minmax(0,1fr)', gap: 16 }}>
        <SkeletonCard h={150} />
        <SkeletonCard h={150} />
      </div>
      <SkeletonCard h={220} />
    </div>
  );
}

function SkeletonCard({ h }: { h: number }) {
  // Reuses the global .skeleton shimmer (globals.css).
  return (
    <div
      className="skeleton"
      aria-hidden
      style={{ height: h, borderRadius: 13 }}
    />
  );
}
