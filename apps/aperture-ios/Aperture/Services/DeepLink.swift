import Foundation

// A deep link carried by a push/local notification → routed into the app's
// navigation. Mirrors the engine's alert taxonomy (new high-conviction idea,
// regime flip, drift alert, breaker trip), each landing on the relevant screen.
enum DeepLink: Equatable {
    case ideasList          // "N new ideas this cycle"
    case idea(String)       // a specific idea → decision chain
    case symbol(String)     // a specific instrument
    case strategy(String)   // a strategy family
    case tab(String)        // "home" | "markets" | "ideas" | "strategies" | "more"

    /// Build from a notification payload's userInfo (APNs `aps` siblings or a
    /// local notification). Shape: { "screen": "...", "symbol"?: "...", "id"?: "..." }.
    static func from(_ userInfo: [AnyHashable: Any]) -> DeepLink? {
        guard let screen = (userInfo["screen"] ?? userInfo["aperture_screen"]) as? String else { return nil }
        let symbol = userInfo["symbol"] as? String
        switch screen {
        case "idea": return symbol.map(DeepLink.idea) ?? .ideasList
        case "ideas": return .ideasList
        case "symbol": return symbol.map(DeepLink.symbol)
        case "strategy": return (userInfo["id"] as? String).map(DeepLink.strategy)
        case "home", "markets", "strategies", "more": return .tab(screen)
        default: return nil
        }
    }

    /// Build from an `aperture://<host>?symbol=…&id=…` URL (widget / Live Activity
    /// taps). Reuses the same parser as the notification payload path.
    static func from(_ url: URL) -> DeepLink? {
        guard url.scheme == "aperture", let host = url.host else { return nil }
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        var info: [AnyHashable: Any] = ["screen": host]
        for item in items where item.value != nil { info[item.name] = item.value! }
        return from(info)
    }
}

/// Shared navigation bus. The notification layer publishes a `DeepLink`; the tab
/// shell consumes it and drives `selection` + the relevant `NavigationPath`.
final class Router: ObservableObject {
    @Published var pending: DeepLink?
    func handle(_ link: DeepLink) { pending = link }
}
