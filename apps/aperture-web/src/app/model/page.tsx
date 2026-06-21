'use client';
import { Panel } from '@/components/ui/Panel';
import { Stat, StatusDot } from '@/components/ui/primitives';
import { ComingState } from '@/components/ui/honesty';
import { Icon } from '@/components/Icon';
import { getModel, MODEL } from '@/data/api';
import { Loaded, ViewProps } from '@/data/useEnvelope';
import { useChartColors } from '@/lib/theme';

// `getModel` resolves ApiEnvelope<typeof MODEL>; mirror that here since the data
// type isn't separately exported from the accessor module.
type ModelData = typeof MODEL;

// Entry-gate threshold — the meta-prob below which the engine does NOT act.
// Used to color histogram buckets and place a reference marker.
const ENTRY_GATE = 0.55;

// Family tints reused from the prototype for the (ghosted) feature-importance layout.
const FAM_COLOR: Record<string, string> = {
  Momentum: '#4d9fff',
  Volatility: '#f0a93b',
  Regime: '#1ecb8b',
  Microstructure: '#f6679a',
  FracDiff: '#b07cff',
  Classical: '#22d3ee',
  Sentiment: '#7c5cff',
  StructuralBreak: '#9aa4b2',
};

// "—" em-dash for an absent scalar — never a fake 0.
const DASH = '—';

export default function ModelPage() {
  return <Loaded fetcher={getModel} View={ModelView} />;
}

function ModelView({ data }: ViewProps<ModelData>) {
  const C = useChartColors();
  const m = data;

  // State: no production model loaded → one calm MODEL_REQUIRED panel (spec §States).
  if (m.version == null) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <Panel pad={22}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
            <span
              style={{ display: 'grid', placeItems: 'center', width: 40, height: 40, borderRadius: 11, background: `${C.neutral}1e`, color: C.neutral, flex: 'none' }}
            >
              <Icon name="model" size={20} />
            </span>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="mono" style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-0.01em' }}>MODEL_REQUIRED</span>
                <span className="pill" style={{ height: 22, fontSize: 11, color: C.neutral, borderColor: `${C.neutral}44` }}>not loaded</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 650, color: C.text1, marginTop: 10 }}>No production model loaded</div>
              <div className="muted" style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 6, maxWidth: 520 }}>
                The engine has no registered MLflow production model, so it emits <span className="mono">MODEL_REQUIRED</span> and meta / calibrated probabilities are unavailable. Register a production model to light up this screen.
              </div>
            </div>
          </div>
        </Panel>
      </div>
    );
  }

  // Meta-probability histogram — LIVE (model-gated). Proportional bars from raw counts.
  const metaProbHist = m.metaProbHist ?? [];
  const histMax = Math.max(...metaProbHist.map((b) => b.count ?? 0), 1);

  // Retrain timeline — LIVE but may be empty when no MLflow history is persisted.
  const retrainTimeline = m.retrainTimeline ?? [];

  // Promotion gate — real booleans (or null when the run didn't emit them).
  // Uniformly false/null because the retrain gate is hard-broken. Rendered as
  // NEUTRAL "not run", never green-pass / red-fail.
  const GATE_TIP = 'did not run ≠ failed — the retrain gate is hard-broken (retrain_pipeline.py:265)';
  const g = m.gates ?? { cpcv: null, dsr: null, pbo: null };
  const gates: [string, boolean | null][] = [
    ['CPCV', g.cpcv ?? null],
    ['DSR', g.dsr ?? null],
    ['PBO', g.pbo ?? null],
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* 1 — Model header [LIVE: version/type/run id/age] + real MLflow Stat strip */}
      <Panel pad={22}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 18 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span className="mono" style={{ fontSize: 22, fontWeight: 720, letterSpacing: '-0.02em' }}>{m.version}</span>
              <span className="pill pill-buy"><StatusDot ok live /> Production</span>
            </div>
            <div className="muted" style={{ fontSize: 13, marginTop: 6 }}>
              {m.type ?? DASH} · trained {m.trainedAt ?? DASH} · run <span className="mono" style={{ color: C.text2 }}>{m.runId ?? DASH}</span> · {m.lastRetrainHours == null ? <span>{DASH}</span> : <><span className="num">{m.lastRetrainHours}</span>h ago</>}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 30 }}>
            <Stat label="CV score" value={m.cvScore == null ? DASH : m.cvScore.toFixed(3)} sub="mean · 5-fold" valueSize={20} />
            <Stat label="Train acc" value={m.trainAcc == null ? DASH : m.trainAcc.toFixed(3)} sub="in-sample" valueSize={20} />
            <Stat label="Events" value={m.trainingEvents == null ? DASH : <span className="num">{m.trainingEvents.toLocaleString()}</span>} sub="training labels" valueSize={20} />
          </div>
        </div>
        <div
          className="muted"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 9,
            fontSize: 12,
            lineHeight: 1.5,
            marginTop: 16,
            paddingTop: 14,
            borderTop: '1px solid var(--border)',
          }}
        >
          <span style={{ color: C.accent, display: 'inline-flex', flex: 'none' }}><Icon name="model" size={14} /></span>
          This screen — meta &amp; calibrated probabilities included — depends on an MLflow production model. A model is loaded, so the live parts render; calibration (ECE / Brier / reliability) stays gated until calibration history is persisted.
        </div>
      </Panel>

      {/* 1b — Promotion gate [LIVE flags, honestly "not run"] */}
      <Panel
        title="Promotion gate"
        subtitle="CPCV / DSR / PBO from the production run — neutral until the retrain gate is fixed"
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
          {gates.map(([label]) => (
            <div
              key={label}
              title={GATE_TIP}
              style={{
                cursor: 'help',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '10px 14px',
                borderRadius: 12,
                border: `1px solid ${C.neutral}33`,
                background: `${C.neutral}12`,
              }}
            >
              <span style={{ display: 'inline-flex', color: C.neutral }}><Icon name="lock" size={14} /></span>
              <span style={{ fontSize: 13, fontWeight: 600, color: C.text1 }}>{label}</span>
              <span className="pill" style={{ height: 20, fontSize: 10.5, color: C.neutral, borderColor: `${C.neutral}44`, background: 'transparent' }}>not run</span>
            </div>
          ))}
        </div>
        <div
          className="muted"
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 9,
            fontSize: 11.5,
            lineHeight: 1.55,
            marginTop: 14,
            paddingTop: 14,
            borderTop: '1px solid var(--border)',
          }}
        >
          <span style={{ color: C.warn, display: 'inline-flex', flex: 'none', marginTop: 1 }}><Icon name="lock" size={13} /></span>
          {GATE_TIP}. The flags are real but uniformly <span className="mono">false</span> — so promote / reject reasons below are placeholders, not validated verdicts.
        </div>
      </Panel>

      {/* 2 — Meta-prob histogram [LIVE bars] | Calibration reliability [COMING wave 6] */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: 16, alignItems: 'stretch' }}>
        <Panel title="Meta-probability histogram" subtitle="Predictions this cycle by confidence bucket — live, model-gated">
          {metaProbHist.length === 0 ? (
            // Empty, NOT ComingState — the producer (meta_labels) exists; it's just
            // legitimately empty for this model_version (spec §States).
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                minHeight: 180,
                textAlign: 'center',
              }}
            >
              <span style={{ display: 'grid', placeItems: 'center', width: 36, height: 36, borderRadius: 11, background: C.surfaceInset, color: C.text3 }}>
                <Icon name="model" size={17} />
              </span>
              <div className="dim" style={{ fontSize: 13 }}>No predictions recorded yet for this model.</div>
            </div>
          ) : (
            <>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, height: 180, paddingTop: 6 }}>
                {metaProbHist.map((b) => {
                  const count = b.count ?? 0;
                  const frac = count / histMax;
                  // Color below the entry gate vs at/above it distinctly: below-gate
                  // predictions don't act (muted), at/above-gate ones do (accent2).
                  const atGate = parseFloat(b.bucket) >= ENTRY_GATE;
                  const color = atGate ? C.accent2 : C.accent;
                  return (
                    <div key={b.bucket} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, height: '100%', justifyContent: 'flex-end' }}>
                      <span className="num" style={{ fontSize: 10.5, color: C.text3 }}>{count}</span>
                      <div
                        title={`p ${b.bucket} · ${count} predictions · ${atGate ? 'at/above' : 'below'} entry gate (${ENTRY_GATE})`}
                        style={{ width: '100%', height: `${Math.max(2, frac * 100)}%`, background: color, borderRadius: '4px 4px 0 0', minHeight: 3, opacity: atGate ? 1 : 0.55 }}
                      />
                      <span className="mono" style={{ fontSize: 10.5, color: C.text3 }}>{b.bucket}</span>
                    </div>
                  );
                })}
              </div>
              <div
                className="muted"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontSize: 11,
                  marginTop: 12,
                  paddingTop: 10,
                  borderTop: '1px solid var(--border)',
                }}
              >
                <span style={{ width: 18, height: 0, borderTop: `2px dashed ${C.warn}`, display: 'inline-block', flex: 'none' }} />
                Entry gate <span className="num" style={{ color: C.text2 }}>{ENTRY_GATE.toFixed(2)}</span> — buckets at/above act (<span style={{ color: C.accent2 }}>blue</span>); below the gate are skipped (<span style={{ color: C.accent }}>dim</span>).
              </div>
            </>
          )}
        </Panel>

        <Panel title="Calibration reliability" subtitle="Predicted vs observed win rate · diagonal = perfect">
          <ComingState
            title="Calibration reliability"
            unlock="Calibration reliability — coming when calibration buckets are persisted."
            wave={6}
            variant="deferred"
            ghost={<GhostCalibration />}
          />
        </Panel>
      </div>

      {/* 3 — Feature importance [COMING wave 4] */}
      <Panel title="Feature importance" subtitle="Top drivers of the meta-labeler · TreeSHAP">
        <ComingState
          title="Feature importance"
          unlock="Feature importance — coming when shap_importance (TreeSHAP) is persisted (never called today)."
          wave={4}
          variant="wireable"
          ghost={<GhostMiniBars />}
        />
      </Panel>

      {/* 4 — Feature drift [COMING wave 6] | RL shadow [COMING wave 6] */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: 16, alignItems: 'stretch' }}>
        <Panel title="Feature drift" subtitle="Divergence vs the training distribution">
          <ComingState
            title="Feature drift"
            unlock="Feature drift — coming when a baseline is set; drift currently emits a hardcoded 1.0 and get_drifted_features() returns []."
            wave={6}
            variant="deferred"
          />
        </Panel>
        <Panel title="RL shadow" subtitle="PPO policy running in parallel, not trading">
          <ComingState
            title="RL shadow"
            unlock="RL shadow — coming when the shadow agent is wired and compared in production."
            wave={6}
            variant="wireable"
          />
        </Panel>
      </div>

      {/* 5 — Retrain timeline [LIVE] with honest gate caption */}
      <Panel
        title="Retrain timeline"
        subtitle="Promote / reject events from MLflow — live"
      >
        {retrainTimeline.length === 0 ? (
          // Empty, NOT ComingState — the MLflow run-history producer exists; it's
          // just legitimately empty (no promote/reject events recorded yet).
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              minHeight: 120,
              textAlign: 'center',
            }}
          >
            <span style={{ display: 'grid', placeItems: 'center', width: 36, height: 36, borderRadius: 11, background: C.surfaceInset, color: C.text3 }}>
              <Icon name="model" size={17} />
            </span>
            <div className="dim" style={{ fontSize: 13 }}>No retrain events recorded yet.</div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {retrainTimeline.map((r, i) => (
              <div key={`${r.date}-${i}`} style={{ display: 'flex', gap: 14, padding: '12px 0', borderTop: i ? '1px solid var(--border)' : 'none' }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 3 }}>
                  <StatusDot ok={r.promoted} />
                  {i < retrainTimeline.length - 1 && <span style={{ width: 1, flex: 1, background: C.border, marginTop: 4 }} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ fontSize: 13.5, fontWeight: 550 }}>{r.event ?? DASH}</span>
                      <span
                        className="pill"
                        style={{
                          height: 20,
                          fontSize: 10.5,
                          color: r.promoted ? C.pos : C.neg,
                          borderColor: `${r.promoted ? C.pos : C.neg}40`,
                          background: `${r.promoted ? C.pos : C.neg}14`,
                        }}
                      >
                        {r.promoted ? 'Promoted' : 'Rejected'}
                      </span>
                    </div>
                    <span className="num muted" style={{ fontSize: 12.5 }}>Sharpe {r.sharpe == null ? DASH : r.sharpe.toFixed(2)}</span>
                  </div>
                  <div className="dim mono" style={{ fontSize: 11.5, marginTop: 3 }}>{r.date ?? DASH}</div>
                </div>
              </div>
            ))}
          </div>
        )}
        <div
          className="muted"
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 9,
            fontSize: 11.5,
            lineHeight: 1.55,
            marginTop: 14,
            paddingTop: 14,
            borderTop: '1px solid var(--border)',
          }}
        >
          <span style={{ color: C.warn, display: 'inline-flex', flex: 'none', marginTop: 1 }}><Icon name="lock" size={13} /></span>
          Same upstream cause as the Promotion gate above: CPCV / DSR / PBO didn&apos;t run (retrain gate broken at retrain_pipeline.py:265), so promote / reject reasons here are placeholders, not validated verdicts.
        </div>
      </Panel>
    </div>
  );
}

// Faint y=x reference for the eventual calibration plot. Fluid-width: the SVG
// scales to its container via viewBox (no fixed pixel width).
function GhostCalibration() {
  const C = useChartColors();
  return (
    <svg width="100%" height={160} viewBox="0 0 240 160" preserveAspectRatio="none" fill="none">
      <line x1={8} y1={152} x2={232} y2={8} stroke={C.text2} strokeWidth={1.4} strokeDasharray="4 4" />
      <path d="M8 152 L60 120 L112 96 L164 56 L208 30 L232 16" stroke={C.text2} strokeWidth={2} />
    </svg>
  );
}

// Faint MiniBars-shaped ghost for the eventual feature-importance layout. Fluid-width.
function GhostMiniBars() {
  const C = useChartColors();
  const fams =['Momentum', 'Volatility', 'Regime', 'Microstructure', 'FracDiff'];
  return (
    <div style={{ width: '100%', maxWidth: 360, display: 'flex', flexDirection: 'column', gap: 9 }}>
      {[0.92, 0.74, 0.61, 0.5, 0.38].map((w, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 70, height: 8, borderRadius: 99, background: C.text2 }} />
          <div style={{ flex: 1, height: 9, borderRadius: 99, background: C.surface3, overflow: 'hidden' }}>
            <div style={{ width: `${w * 100}%`, height: '100%', borderRadius: 99, background: FAM_COLOR[fams[i]] }} />
          </div>
        </div>
      ))}
    </div>
  );
}
