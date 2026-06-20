'use client';
import { useEffect, useRef } from 'react';
import Link from 'next/link';
import { Icon } from '@/components/Icon';
import { AssetGlyph, ActionPill, ProbRing, Bar } from '@/components/ui/primitives';
import { ComingState, DataUnavailable } from '@/components/ui/honesty';
import { MiniBars } from '@/components/charts/MiniBars';
import { TradeIdea, symBy } from '@/data/mock';
import { useChartColors } from '@/lib/theme';
import { fmtPct, fmtPctSigned, fmtProb, sideLabel } from '@/lib/format';

// ---------------------------------------------------------------------------
// The decision drawer — the flagship gesture. Renders the full decision chain
// for a single idea, but obeys the v1 data-honesty contract: regime fit, the
// sizing cascade, SHAP, pre-trade cost and track record are PRESENT in the mock
// yet NOT persisted by the engine today, so they render as honest coming-states
// instead of fabricated numbers. See docs/data_readiness.md.
// ---------------------------------------------------------------------------

const UNLOCK = {
  meta: 'Meta probability — load an MLflow production model. Currently MODEL_REQUIRED.',
  calibrated: 'Calibrated probability — load an MLflow production model. Currently MODEL_REQUIRED.',
  regimeFit:
    'Regime fit — coming when the regime detector is wired into the live cycle. RegimeDetector has zero runtime callers.',
  cascade:
    'Sizing cascade & binding constraint — coming when the engine surfaces constraints_applied at the idea boundary.',
  shap: 'Feature contributions — coming when shap_importance (TreeSHAP) is persisted.',
  cost: 'Pre-trade cost — coming when a cost service is wired.',
  trackRecord:
    'Track record — coming when a call-history store exists (the snapshot is overwritten each publish).',
} as const;

function Section({
  label,
  children,
  action,
}: {
  label: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  const C = useChartColors();
  return (
    <div style={{ padding: '18px 22px', borderTop: `1px solid ${C.border}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div className="eyebrow">{label}</div>
        {action}
      </div>
      {children}
    </div>
  );
}

export function IdeaDrawer({ idea, onClose }: { idea: TradeIdea; onClose: () => void }) {
  const C = useChartColors();
  const closeRef = useRef<HTMLButtonElement>(null);

  // Escape-to-close + body-scroll-lock + focus management while the drawer is open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Move focus into the dialog on mount; restore to the trigger on close.
    const prevFocus = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
      prevFocus?.focus?.();
    };
  }, [onClose]);

  const modelLoaded = idea.metaProbability != null && idea.calibratedProbability != null;

  const name = symBy(idea.symbol).name;
  const latencyTotal = Object.values(idea.stageLatency).reduce((a, b) => a + b, 0) || 1e-9;

  // Decision chain — Signals › Meta › Calibrated › Regime-fit › Bet › Target.
  // Most steps are LIVE; meta/calibrated may be model-gated, and regime-fit is a
  // not-yet-available coming-state (never the mock number).
  const chain: Array<{ label: string; value: string; tone: string; coming?: boolean; gated?: boolean }> = [
    { label: 'Signals', value: String(idea.signalCount), tone: C.text1 },
    {
      label: 'Meta p',
      value: idea.metaProbability != null ? fmtProb(idea.metaProbability) : 'Model?',
      tone: C.accent2,
      gated: idea.metaProbability == null,
    },
    {
      label: 'Calibrated',
      value: idea.calibratedProbability != null ? fmtProb(idea.calibratedProbability) : 'Model?',
      tone: C.accent,
      gated: idea.calibratedProbability == null,
    },
    { label: 'Regime fit', value: 'Coming', tone: C.regimeUp, coming: true },
    {
      label: 'Bet size',
      value: idea.betSize ? fmtPct(idea.betSize, 1) : '—',
      tone: C.warn,
    },
    {
      label: 'Target',
      value: fmtPctSigned(idea.targetWeight),
      tone: idea.targetWeight > 0 ? C.pos : idea.targetWeight < 0 ? C.neg : C.text3,
    },
  ];

  return (
    <div
      className="backdrop"
      style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', justifyContent: 'flex-end', background: 'rgba(4,6,9,0.58)' }}
      onClick={onClose}
    >
      <div
        className="drawer-panel scroll-y"
        role="dialog"
        aria-modal="true"
        aria-label={`${idea.symbol} decision chain`}
        style={{
          width: 460,
          maxWidth: '94vw',
          height: '100%',
          background: 'var(--surface-1)',
          borderLeft: `1px solid ${C.border}`,
          boxShadow: 'var(--shadow-pop)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '18px 22px',
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 12,
            position: 'sticky',
            top: 0,
            background: 'var(--surface-1)',
            borderBottom: `1px solid ${C.border}`,
            zIndex: 2,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
            <AssetGlyph sym={idea} size={40} />
            <div style={{ minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="mono" style={{ fontSize: 19, fontWeight: 720, letterSpacing: '-0.02em' }}>{idea.symbol}</span>
                <ActionPill action={idea.action} />
              </div>
              <div className="muted" style={{ fontSize: 12.5, marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {name} · {idea.strategy ?? '—'} · {idea.barType} bars
              </div>
            </div>
          </div>
          <button ref={closeRef} className="btn" style={{ width: 34, height: 34, padding: 0, flex: 'none' }} onClick={onClose} aria-label="Close">
            <Icon name="close" size={16} />
          </button>
        </div>

        {/* (1) NL reason [LIVE] */}
        <div style={{ padding: '18px 22px 4px' }}>
          <div style={{ background: C.surfaceInset, border: `1px solid ${C.border}`, borderRadius: 12, padding: 14, fontSize: 13.5, lineHeight: 1.55, color: C.text1 }}>
            {idea.reason}
          </div>
        </div>

        {/* (2) Decision-chain stepper [LIVE, fit coming] */}
        <Section label="Decision chain">
          <div style={{ display: 'flex', alignItems: 'stretch', gap: 4 }}>
            {chain.map((step, i) => (
              <div key={step.label} style={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}>
                <div
                  title={step.coming ? UNLOCK.regimeFit : step.gated ? UNLOCK.meta : undefined}
                  style={{
                    flex: 1,
                    minWidth: 0,
                    background: C.surfaceInset,
                    border: `1px solid ${step.coming || step.gated ? C.borderStrong : C.border}`,
                    borderStyle: step.coming || step.gated ? 'dashed' : 'solid',
                    borderRadius: 10,
                    padding: '9px 4px',
                    textAlign: 'center',
                    cursor: step.coming || step.gated ? 'help' : 'default',
                  }}
                >
                  <div
                    className={step.coming || step.gated ? 'mono' : 'num'}
                    style={{ fontSize: step.coming || step.gated ? 10.5 : 14, fontWeight: 700, color: step.coming || step.gated ? C.text3 : step.tone, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}
                  >
                    {(step.coming || step.gated) && <Icon name="lock" size={10} />}
                    {step.value}
                  </div>
                  <div className="dim" style={{ fontSize: 9, marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    {step.label}
                  </div>
                </div>
                {i < chain.length - 1 && <span style={{ color: C.text3, fontSize: 12, padding: '0 1px', flex: 'none' }}>›</span>}
              </div>
            ))}
          </div>
        </Section>

        {/* (3) Conviction rings + track record */}
        <Section label="Model conviction">
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 22 }}>
            {idea.metaProbability != null ? (
              <ProbRing value={idea.metaProbability} label="Meta" size={64} />
            ) : (
              <DataUnavailable size={64} label="Meta" modelGated unlock={UNLOCK.meta} />
            )}
            {idea.calibratedProbability != null ? (
              <ProbRing value={idea.calibratedProbability} label="Calibrated" size={64} />
            ) : (
              <DataUnavailable size={64} label="Calibrated" modelGated unlock={UNLOCK.calibrated} />
            )}
            <div style={{ flex: 1 }}>
              <div className="muted" style={{ fontSize: 12 }}>Historical track record</div>
              <div style={{ marginTop: 7 }}>
                <DataUnavailable unlock={UNLOCK.trackRecord} />
              </div>
              <div className="dim" style={{ fontSize: 11.5, marginTop: 7, lineHeight: 1.5 }}>
                The published snapshot is overwritten each cycle — no call history exists yet.
              </div>
            </div>
          </div>
          {modelLoaded && (
            <div className="dim" style={{ fontSize: 11, marginTop: 12, lineHeight: 1.5 }}>
              raw == calibrated — the pipeline exposes one calibrated scalar today.
            </div>
          )}
        </Section>

        {/* (4) Signals [LIVE] */}
        <Section label={`Signals (${idea.signals.length})`}>
          {idea.signals.length === 0 ? (
            <div className="dim" style={{ fontSize: 12.5 }}>No signals fired this cycle.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {idea.signals.map((s, i) => (
                <div key={i} style={{ background: C.surfaceInset, border: `1px solid ${C.border}`, borderRadius: 11, padding: '11px 13px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                    <span className="mono" style={{ fontWeight: 600, fontSize: 13 }}>{s.family}</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className={`pill ${s.side > 0 ? 'pill-buy' : s.side < 0 ? 'pill-sell' : 'pill-neutral'}`}>{sideLabel(s.side)}</span>
                      <span className="num muted" style={{ fontSize: 12 }}>conf {s.confidence.toFixed(2)}</span>
                    </span>
                  </div>
                  <div style={{ marginTop: 9 }}>
                    <Bar value={s.confidence} max={1} color={s.side > 0 ? C.buy : s.side < 0 ? C.sell : C.neutral} height={5} />
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
                    {Object.entries(s.meta).map(([k, v]) => (
                      <span key={k} style={{ fontSize: 11, color: C.text3, background: C.surface2, borderRadius: 6, padding: '3px 7px' }}>
                        <span className="mono">{k}</span> <span className="mono" style={{ color: C.text2 }}>{String(v)}</span>
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* (5) Sizing cascade [COMING · wave 4] */}
        <Section label="Bet-sizing cascade">
          <ComingState
            title="Sizing cascade — coming as the engine lands it"
            unlock={UNLOCK.cascade}
            wave={4}
            variant="wireable"
            ghost={<GhostCascade />}
          />
        </Section>

        {/* (6) Why / SHAP [COMING · wave 4] */}
        <Section label="Why the model leaned this way">
          <ComingState
            title="Feature contributions — coming as the engine lands it"
            unlock={UNLOCK.shap}
            wave={4}
            variant="wireable"
            ghost={<GhostShap />}
          />
        </Section>

        {/* (7) Pre-trade cost & constraints
            The binding constraint is part of the Wave-4 sizing cascade (constraints_applied),
            covered by the cascade ComingState in section (5). Only the gated cost reads here —
            sizing_constraints_applied is ABSENT, never a fabricated chip list. */}
        <Section label="Pre-trade cost & constraints">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span className="muted" style={{ fontSize: 12.5 }}>Expected cost</span>
            <DataUnavailable unlock={UNLOCK.cost} />
          </div>
        </Section>

        {/* (8) Pipeline latency [LIVE] */}
        <Section label={`Pipeline latency · ${(latencyTotal * 1000).toFixed(0)}ms`}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(idea.stageLatency).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="mono" style={{ width: 128, flex: 'none', fontSize: 11.5, color: C.text3 }}>{k}</span>
                <div style={{ flex: 1 }}>
                  <Bar value={v} max={latencyTotal} color={C.accent2} height={6} />
                </div>
                <span className="num dim" style={{ width: 46, textAlign: 'right', fontSize: 11.5 }}>{(v * 1000).toFixed(0)}ms</span>
              </div>
            ))}
          </div>
        </Section>

        {/* (9) Open symbol — v1 is read-first; bridge to the real candle route. */}
        <div style={{ padding: 22, borderTop: `1px solid ${C.border}` }}>
          <Link
            href={`/symbols/${idea.symbol}`}
            className="btn"
            onClick={onClose}
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
          >
            <Icon name="chevronRight" size={15} /> Open symbol
          </Link>
        </div>
      </div>
    </div>
  );
}

// Faint wireframe ghosts (rendered at ~16% opacity behind the ComingState copy).
function GhostCascade() {
  const C = useChartColors();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: 320 }}>
      {[100, 80, 66, 61, 58].map((w, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 60, height: 8, borderRadius: 4, background: C.text2 }} />
          <div style={{ flex: 1, height: 18, borderRadius: 6, background: C.surface3, position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', inset: 0, width: `${w}%`, background: C.text2, borderRadius: 6 }} />
          </div>
        </div>
      ))}
    </div>
  );
}
function GhostShap() {
  const ghost = [0.42, -0.33, 0.27, -0.19, 0.14, -0.09];
  return (
    <div style={{ width: 320 }}>
      <MiniBars
        items={ghost.map((v, i) => ({ label: `feature_${i}`, value: v, sub: ' ' }))}
        signed
        labelWidth={110}
      />
    </div>
  );
}
