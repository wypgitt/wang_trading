import SwiftUI

/// Model & Features — model-gated. Header + meta-prob histogram + retrain
/// timeline are LIVE (MLflow). Calibration / importance / drift / RL are coming.
struct ModelView: View {
    private let m = MockData.model

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // header
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 10) {
                            Text(m.version).font(.system(size: 20, weight: .bold, design: .monospaced))
                            Text("Production").font(.system(size: 11, weight: .semibold)).foregroundStyle(Tok.pos)
                                .padding(.horizontal, 8).padding(.vertical, 3).background(Tok.posSoft, in: Capsule())
                            Spacer()
                        }
                        Text("\(m.type) · trained \(m.trainedAt) · \(m.lastRetrainHours)h ago")
                            .font(.system(size: 12.5)).foregroundStyle(Tok.text2)
                        HStack(spacing: 6) {
                            Text("run").font(.system(size: 11, weight: .semibold)).foregroundStyle(Tok.text3)
                            Text(m.runId).font(.system(size: 11.5, design: .monospaced)).foregroundStyle(Tok.text2)
                        }
                        HStack(spacing: 24) {
                            StatBlock(label: "CV score", value: String(format: "%.3f", m.cvScore), sub: "mean · 5-fold", valueSize: 18)
                            StatBlock(label: "Train acc", value: String(format: "%.3f", m.trainAcc), sub: "in-sample", valueSize: 18)
                            StatBlock(label: "Events", value: m.trainingEvents.formatted(.number.grouping(.automatic)), sub: "training", valueSize: 18)
                            Spacer()
                        }
                        Text("Meta & calibrated probabilities — and this screen — depend on an MLflow production model. A model is loaded, so the live parts render; quality metrics stay gated until calibration history is persisted.")
                            .font(.system(size: 11.5)).foregroundStyle(Tok.text3).lineSpacing(2)
                    }
                }

                // histogram (LIVE)
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Meta-probability histogram", subtitle: "Predictions this cycle by confidence bucket — live, model-gated")
                        let maxC = m.metaProbHist.map(\.1).max() ?? 1
                        HStack(alignment: .bottom, spacing: 6) {
                            ForEach(Array(m.metaProbHist.enumerated()), id: \.offset) { i, b in
                                VStack(spacing: 5) {
                                    Text("\(b.1)").font(.system(size: 9)).monospacedDigit().foregroundStyle(Tok.text3)
                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(i < 5 ? Tok.accent : Tok.accent2)
                                        .frame(height: CGFloat(b.1) / CGFloat(maxC) * 120 + 4)
                                    Text(b.0).font(.system(size: 9, design: .monospaced)).foregroundStyle(Tok.text3)
                                }.frame(maxWidth: .infinity)
                            }
                        }
                    }
                }

                // promotion gate (neutral "not run" — gate is hard-broken)
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Promotion gate", subtitle: "CPCV / DSR / PBO — the checks that guard a promote")
                        HStack(spacing: 8) {
                            gateChip("CPCV", ran: m.gates.cpcv)
                            gateChip("DSR", ran: m.gates.dsr)
                            gateChip("PBO", ran: m.gates.pbo)
                            Spacer()
                        }
                        Text("did not run ≠ failed — retrain gate hard-broken (retrain_pipeline.py:265)")
                            .font(.system(size: 11.5)).foregroundStyle(Tok.text3).lineSpacing(2)
                    }
                }

                ComingPanel(title: "Calibration reliability", subtitle: "Predicted vs observed win rate",
                            unlock: "Calibration reliability — coming when calibration buckets are persisted.", wave: 6)
                ComingPanel(title: "Feature importance", subtitle: "Top drivers of the meta-labeler",
                            unlock: "Feature importance — coming when shap_importance (TreeSHAP) is persisted (never called today).", wave: 4)
                ComingPanel(title: "Feature drift", subtitle: "Divergence vs the training distribution",
                            unlock: "Feature drift — coming when a baseline is set; drift currently emits a hardcoded 1.0 and get_drifted_features() returns [].", wave: 6)
                ComingPanel(title: "RL shadow", subtitle: "PPO policy running in parallel, not trading",
                            unlock: "RL shadow — coming when the shadow agent is wired and compared in production.", wave: 6)

                // retrain timeline (LIVE)
                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        PanelHeader(title: "Retrain timeline", subtitle: "Promote / reject events from MLflow — live")
                        ForEach(Array(m.retrainTimeline.enumerated()), id: \.offset) { _, e in
                            HStack(alignment: .top, spacing: 10) {
                                Circle().fill(e.promoted ? Tok.pos : Tok.neg).frame(width: 8, height: 8).padding(.top, 5)
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack {
                                        Text(e.event).font(.system(size: 13, weight: .semibold))
                                        Text(e.promoted ? "Promoted" : "Rejected")
                                            .font(.system(size: 10, weight: .semibold))
                                            .foregroundStyle(e.promoted ? Tok.pos : Tok.neg)
                                            .padding(.horizontal, 7).padding(.vertical, 2)
                                            .background(e.promoted ? Tok.posSoft : Tok.negSoft, in: Capsule())
                                        Spacer()
                                        Text("Sharpe \(String(format: "%.2f", e.sharpe))").font(.system(size: 12)).monospacedDigit().foregroundStyle(Tok.text2)
                                    }
                                    Text(e.date).font(.system(size: 11)).foregroundStyle(Tok.text3)
                                }
                            }
                        }
                        Text("Gate scores (CPCV / PBO) aren't computed in any runnable entrypoint (retrain_pipeline.py:265), so promote/reject reasons are thin today.")
                            .font(.system(size: 11)).foregroundStyle(Tok.text3).padding(.top, 4)
                    }
                }
            }
            .padding(16)
        }
        .refreshable { try? await Task.sleep(nanoseconds: 600_000_000) }
        .apertureBackground()
        .navigationTitle("Model & Features")
        .navigationBarTitleDisplayMode(.inline)
    }

    /// Neutral "not run" gate chip — never green-pass / red-fail.
    private func gateChip(_ label: String, ran: Bool) -> some View {
        Chip(tint: Tok.text3, border: Tok.border) {
            HStack(spacing: 6) {
                Image(systemName: "minus.circle").font(.system(size: 10))
                Text(label).font(.system(size: 12, weight: .semibold))
                Text(ran ? "ran" : "not run").font(.system(size: 11, weight: .medium)).foregroundStyle(Tok.text3)
            }
        }
    }
}
