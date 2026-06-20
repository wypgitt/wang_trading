import ActivityKit
import WidgetKit
import SwiftUI

// The "trading session" Live Activity — Lock Screen banner + Dynamic Island.
// Honest: decision counts + the top idea + freshness. Never NAV.
struct SessionLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: ApertureSessionAttributes.self) { context in
            LockScreenView(state: context.state, label: context.attributes.sessionLabel, isStale: context.isStale)
                .activityBackgroundTint(WTok.bg0)
                .activitySystemActionForegroundColor(WTok.text1)
        } dynamicIsland: { context in
            let s = context.state
            let stale = context.isStale
            return DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(s.total)").font(.system(size: 22, weight: .bold)).foregroundStyle(WTok.text1)
                        Text("ideas").font(.system(size: 10)).foregroundStyle(WTok.text3)
                    }
                }
                DynamicIslandExpandedRegion(.trailing) {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(s.topSymbol).font(.system(size: 15, weight: .bold, design: .monospaced)).foregroundStyle(WTok.text1)
                        Text(s.topAction).font(.system(size: 11, weight: .semibold)).foregroundStyle(actionColor(s.topAction))
                    }
                }
                DynamicIslandExpandedRegion(.bottom) {
                    HStack(spacing: 12) {
                        pill("\(s.buy)", "Buy", WTok.buy)
                        pill("\(s.sell)", "Sell", WTok.sell)
                        pill("\(s.watch)", "Watch", WTok.watch)
                        Spacer()
                        if s.breakerTripped {
                            Label("Breaker", systemImage: "exclamationmark.triangle.fill")
                                .font(.system(size: 11, weight: .semibold)).foregroundStyle(WTok.sell)
                        } else if stale {
                            Label("Stale", systemImage: "clock.badge.exclamationmark")
                                .font(.system(size: 11, weight: .semibold)).foregroundStyle(WTok.warn)
                        } else {
                            Text(s.capturedAt, style: .relative).font(.system(size: 11)).foregroundStyle(WTok.text3)
                        }
                    }
                }
            } compactLeading: {
                Image(systemName: "bolt.fill").foregroundStyle(WTok.accent)
            } compactTrailing: {
                Text("\(s.total)").font(.system(size: 13, weight: .bold)).foregroundStyle(WTok.text1)
            } minimal: {
                Text("\(s.total)").font(.system(size: 12, weight: .bold)).foregroundStyle(WTok.accent)
            }
            .widgetURL(URL(string: "aperture://ideas"))
        }
    }

    private func pill(_ n: String, _ label: String, _ color: Color) -> some View {
        HStack(spacing: 4) {
            Text(n).font(.system(size: 13, weight: .bold)).foregroundStyle(color)
            Text(label).font(.system(size: 10)).foregroundStyle(WTok.text3)
        }
    }
    private func actionColor(_ a: String) -> Color {
        a == "Buy" ? WTok.buy : a == "Sell" ? WTok.sell : WTok.watch
    }
}

private struct LockScreenView: View {
    let state: ApertureSessionAttributes.ContentState
    let label: String
    let isStale: Bool
    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 5).fill(WTok.grad).frame(width: 16, height: 16)
                        .overlay(Text("✦").font(.system(size: 10, weight: .bold)).foregroundStyle(.white))
                    Text(label).font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.text2)
                    Spacer()
                    if isStale {
                        Label("Stale", systemImage: "clock.badge.exclamationmark")
                            .font(.system(size: 10, weight: .semibold)).foregroundStyle(WTok.warn)
                    } else {
                        Text(state.capturedAt, style: .relative).font(.system(size: 10)).foregroundStyle(WTok.text3)
                    }
                }
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text("\(state.total)").font(.system(size: 26, weight: .bold)).foregroundStyle(WTok.text1)
                    Text("actionable").font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.text2)
                }
                HStack(spacing: 12) {
                    Text("\(state.buy) buy").font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.buy)
                    Text("\(state.sell) sell").font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.sell)
                    Text("\(state.watch) watch").font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.watch)
                }
            }
            Divider().overlay(WTok.surface2)
            VStack(alignment: .leading, spacing: 3) {
                Text("TOP IDEA").font(.system(size: 9, weight: .semibold)).tracking(0.6).foregroundStyle(WTok.text3)
                Text(state.topSymbol).font(.system(size: 16, weight: .bold, design: .monospaced)).foregroundStyle(WTok.text1)
                Text(state.topAction).font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(state.topAction == "Buy" ? WTok.buy : state.topAction == "Sell" ? WTok.sell : WTok.watch)
            }
            .frame(width: 92, alignment: .leading)
        }
        .padding(14)
    }
}
