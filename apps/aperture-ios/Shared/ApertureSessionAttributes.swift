import ActivityKit
import Foundation

// SHARED between the app (which starts / updates / ends the Activity) and the
// widget extension (which renders it on the Lock Screen + Dynamic Island). This
// one file is a member of BOTH targets — ActivityKit matches the activity across
// processes by this attributes type.
//
// Honest by construction: the Live Activity surfaces the engine's DECISION state
// (actionable counts + the top idea + freshness), never a portfolio value — there
// is no persisted NAV, so a Live Activity never shows money.
struct ApertureSessionAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var buy: Int
        var sell: Int
        var watch: Int
        var topSymbol: String
        var topAction: String      // "Buy" / "Sell" / "Watch"
        /// When the engine produced this snapshot. The Live Activity renders
        /// freshness as live-updating relative time from this (counts up
        /// honestly), and the staleDate is derived from it — so the surface can
        /// never imply a freshness it isn't tracking.
        var capturedAt: Date
        var breakerTripped: Bool

        var total: Int { buy + sell + watch }
    }

    /// Static for the life of the activity.
    var sessionLabel: String
}
