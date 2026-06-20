import ActivityKit
import Foundation

// Drives the "trading session" Live Activity. start() requests it, update() pushes
// a fresh ContentState, stop() ends it. The content is the honest decision snapshot
// (counts + top idea + captured timestamp) — never NAV.
@MainActor
final class SessionActivityController: ObservableObject {
    @Published private(set) var isRunning = false
    private var activity: Activity<ApertureSessionAttributes>?

    /// Reconnect to an activity that survived a relaunch (ActivityKit persists them
    /// across launches), and end any extras so >1 can't accumulate.
    init() {
        let live = Activity<ApertureSessionAttributes>.activities
        activity = live.first
        isRunning = activity != nil
        if live.count > 1 {
            for extra in live.dropFirst() {
                Task { await extra.end(nil, dismissalPolicy: .immediate) }
            }
        }
    }

    /// The user must allow Live Activities (Settings → app, or Face ID & Passcode).
    var areEnabled: Bool { ActivityAuthorizationInfo().areActivitiesEnabled }

    func start() {
        guard areEnabled, activity == nil else { return }
        let attrs = ApertureSessionAttributes(sessionLabel: "Trading session")
        do {
            activity = try Activity.request(attributes: attrs, content: Self.content(), pushType: nil)
            isRunning = true
        } catch {
            isRunning = false
        }
    }

    /// Call on the publisher cadence (or App Group cache write) to refresh the
    /// surfaced snapshot + extend the staleDate.
    func update() {
        guard let activity else { return }
        Task { await activity.update(Self.content()) }
    }

    func stop() {
        guard let activity else { return }
        Task {
            await activity.end(Self.content(), dismissalPolicy: .immediate)
            self.activity = nil
            self.isRunning = false
        }
    }

    // staleDate = capture time + the app's stale threshold, so the system dims the
    // activity (context.isStale) once the snapshot ages past the freshness window.
    private static func content() -> ActivityContent<ApertureSessionAttributes.ContentState> {
        let state = snapshot()
        return ActivityContent(state: state, staleDate: state.capturedAt.addingTimeInterval(Tok.staleThresholdSeconds))
    }

    /// Today: derived from the honest mock. Production: read from the App Group
    /// shared cache the TradeIdeaPublisher writes each cycle. Pure (no actor state)
    /// → nonisolated so it's callable from tests / background contexts.
    nonisolated static func snapshot() -> ApertureSessionAttributes.ContentState {
        let c = MockData.actionCounts
        let top = MockData.topActionable.first
        return .init(
            buy: c.buy, sell: c.sell, watch: c.watch,
            topSymbol: top?.symbol ?? "—",
            topAction: top?.action.label ?? "—",
            capturedAt: Date().addingTimeInterval(-Double(MockData.trust.stalenessSeconds)),
            breakerTripped: false)
    }
}
