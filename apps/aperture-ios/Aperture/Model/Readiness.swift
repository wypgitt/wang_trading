// Screen readiness — iOS twin of apps/aperture-web/src/lib/readiness.ts and the
// machine-readable form of docs/data_readiness.md. Drives the "More" list and the
// Coming detail screens. Flip one entry when an engine wave lands.
import Foundation

enum LockKind { case gated, wireable, deferred
    var label: String {
        switch self {
        case .gated: return "Gated · needs engine persistence"
        case .wireable: return "Wireable · BFF stub to rewire"
        case .deferred: return "Deferred · needs the audit chain"
        }
    }
    var symbol: String {
        switch self {
        case .gated: return "lock.fill"
        case .wireable: return "wrench.and.screwdriver.fill"
        case .deferred: return "clock.arrow.circlepath"
        }
    }
}

struct ScreenSpec: Identifiable, Hashable {
    let id: String
    let label: String
    let sfSymbol: String
    let modelGated: Bool
    let lock: LockKind?
    let purpose: String
    let unlock: String
    let wave: Int?
    var isLive: Bool { lock == nil }

    static func == (lhs: ScreenSpec, rhs: ScreenSpec) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

/// Strategy family readiness — iOS twin of FAMILY_READINESS in the web app's
/// lib/readiness.ts and the data_readiness.md family table. Status is derived
/// from each generator's dispatch `kind`: the deployed cycle supplies only
/// single-symbol `bars`, so families needing extra context never fire.
struct FamilyReadiness {
    let active: Bool
    let kind: String
    let reason: String? // why dormant
}

enum Families {
    static let map: [String: FamilyReadiness] = [
        "ts_momentum": .init(active: true, kind: "bars", reason: nil),
        "mean_reversion": .init(active: true, kind: "bars", reason: nil),
        "ma_crossover": .init(active: true, kind: "bars", reason: nil),
        "donchian_breakout": .init(active: true, kind: "bars", reason: nil),
        "cs_momentum": .init(active: false, kind: "panel", reason: "needs a multi_asset_prices panel feed"),
        "stat_arb": .init(active: false, kind: "pair", reason: "needs a cointegrated stat_arb_pair"),
        "futures_carry": .init(active: false, kind: "bars_extra", reason: "needs futures_curve context"),
        "vrp": .init(active: false, kind: "bars_extra", reason: "needs vol_features context"),
        "funding_rate_arb": .init(active: false, kind: "bars_extra", reason: "needs a funding_rates feed"),
        "cross_exchange_arb": .init(active: false, kind: "exchange_prices", reason: "needs multi-venue prices"),
    ]
    static func of(_ id: String) -> FamilyReadiness { map[id] ?? .init(active: false, kind: "unknown", reason: "no live feed wired") }
    static var counts: (active: Int, inactive: Int, total: Int) {
        let v = map.values
        return (v.filter { $0.active }.count, v.filter { !$0.active }.count, v.count)
    }
}

enum Readiness {
    // Live, model-gated (reached from the More tab on iOS)
    static let model = ScreenSpec(
        id: "model", label: "Model & Features", sfSymbol: "cpu", modelGated: true, lock: nil,
        purpose: "Meta-prob histogram + retrain timeline",
        unlock: "", wave: nil)

    static let coming: [(group: String, items: [ScreenSpec])] = [
        ("Portfolio", [
            ScreenSpec(id: "portfolio", label: "Portfolio & Risk", sfSymbol: "chart.pie.fill", modelGated: false, lock: .gated,
                       purpose: "Positions, exposure, factor risk, drawdown",
                       unlock: "Unlocks when the engine persists positions + a NAV series. Today no orders are routed (live_orders_sent=0) and src/portfolio has zero production callers.", wave: 5),
            ScreenSpec(id: "execution", label: "Execution & TCA", sfSymbol: "arrow.left.arrow.right", modelGated: false, lock: .gated,
                       purpose: "Orders, fills, transaction-cost analysis",
                       unlock: "Unlocks when the order-routing path writes ExecutionStorage (orders/fills/TCA). The deployed path stops before routing.", wave: 5),
        ]),
        ("Research", [
            ScreenSpec(id: "backtests", label: "Backtests", sfSymbol: "chart.xyaxis.line", modelGated: false, lock: .gated,
                       purpose: "Walk-forward equity, the 3 promotion gates, trade log",
                       unlock: "Unlocks when backtest runs are persisted and the retrain gate is fixed (retrain_pipeline.py:265 falls through to gate_unavailable).", wave: 5),
            ScreenSpec(id: "scenarios", label: "Scenarios & Stress", sfSymbol: "square.stack.3d.up", modelGated: false, lock: .gated,
                       purpose: "Factor shocks, stress paths, what-if",
                       unlock: "Unlocks when ScenarioService calls the real factor_risk engine instead of returning mock numbers.", wave: 6),
            ScreenSpec(id: "track-record", label: "Track Record", sfSymbol: "checklist", modelGated: false, lock: .gated,
                       purpose: "Realized calls, hit rate, attribution over time",
                       unlock: "Unlocks when a call-history store exists (the trade-ideas snapshot is overwritten each publish).", wave: 6),
        ]),
        ("Operate", [
            ScreenSpec(id: "monitoring", label: "Monitoring & Alerts", sfSymbol: "bell.badge", modelGated: false, lock: .gated,
                       purpose: "Freshness heatmap, breaker state, alert feed",
                       unlock: "Unlocks when pipeline metrics are scraped over HTTP (the registry is currently unscraped; /metrics serves only bff_* self-metrics).", wave: 6),
            ScreenSpec(id: "preflight", label: "Preflight & Go-Live", sfSymbol: "checkmark.shield", modelGated: false, lock: .wireable,
                       purpose: "Live blocker checks + infrastructure probes",
                       unlock: "Wireable now — point PreflightService at the real PreflightChecker / InfrastructureProbe. Runnable engine code exists; only a BFF stub to rewire.", wave: 3),
            ScreenSpec(id: "replay", label: "Replay / Time Travel", sfSymbol: "clock.arrow.circlepath", modelGated: false, lock: .deferred,
                       purpose: "Reconstruct any past cycle from the audit chain",
                       unlock: "Unlocks when the audit chain is written (ComplianceAuditLogger is never instantiated).", wave: 6),
        ]),
    ]
}
