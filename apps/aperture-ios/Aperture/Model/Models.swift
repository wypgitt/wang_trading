// Domain models — mirror the web app's data layer (src/data/mock.ts), which is
// itself modeled field-for-field on the real engine schemas (TradeIdea, Signal,
// Bar microstructure, Strategy, MetaLabeler). The iOS app consumes the same
// FastAPI BFF in production; today it renders the same honest mock.
import Foundation

enum AssetType: String, Codable, CaseIterable {
    case equity, index, crypto, future
}

enum Action: String, Codable {
    case BUY, SELL, WATCH, MODEL_REQUIRED, NO_DATA

    var label: String {
        switch self {
        case .BUY: return "Buy"
        case .SELL: return "Sell"
        case .WATCH: return "Watch"
        case .MODEL_REQUIRED: return "Model?"
        case .NO_DATA: return "No data"
        }
    }
}

/// Persisted bar microstructure — the `bars` hypertable columns (REAL today).
struct BarMicro: Hashable {
    var barType: String
    var vwap: Double
    var dollarVolume: Double
    var tickCount: Int
    var buyVolume: Double
    var sellVolume: Double
    var volumeImbalance: Double
    var tickImbalanceRatio: Double
    var imbalance: Double
    var threshold: Double
    var barDurationSeconds: Int
}

struct Candle: Hashable, Identifiable {
    let id: Int
    var o, h, l, c: Double
    var v: Double
}

struct Sym: Identifiable, Hashable {
    var symbol: String
    var name: String
    var type: AssetType
    var price: Double
    var change1d: Double
    var change1w: Double
    var change1m: Double
    var changeYtd: Double
    var spark: [Double]
    var candles: [Candle]
    var marketCap: Double
    var volume: Double
    var hasIdea: Bool
    var bar: BarMicro
    var id: String { symbol }
}

struct IdeaSignal: Hashable {
    var family: String
    var side: Int
    var confidence: Double
    var meta: [(String, String)]  // ordered key/value for display

    static func == (lhs: IdeaSignal, rhs: IdeaSignal) -> Bool {
        lhs.family == rhs.family && lhs.side == rhs.side && lhs.confidence == rhs.confidence
            && lhs.meta.map(\.0) == rhs.meta.map(\.0) && lhs.meta.map(\.1) == rhs.meta.map(\.1)
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(family); hasher.combine(side); hasher.combine(confidence)
    }
}

struct CascadeStage: Hashable {
    var stage: String
    var value: Double
    var binding: Bool
}

struct TradeIdea: Identifiable, Hashable {
    var symbol: String
    var type: AssetType
    var action: Action
    var targetWeight: Double
    var targetNotional: Double
    var latestPrice: Double?
    var barType: String
    var barsLoaded: Int
    var featureRows: Int
    var signalCount: Int
    var topSignalFamily: String?
    var topSignalSide: Int
    var topSignalConfidence: Double?
    var avgSignalConfidence: Double?
    // model-gated — real GBM output only when an MLflow production model is loaded
    var metaProbability: Double?
    var calibratedProbability: Double?
    // ABSENT today (hardcoded null at the BFF) — never rendered as a number
    var regimeFitScore: Double?
    var betSize: Double?
    var sizingConstraints: [String]
    var strategy: String?
    var reason: String
    var expectedCostBps: Double?
    var trackRecordWinRate: Double?
    var trackRecordN: Int?
    var stageLatency: [(String, Double)]  // ordered
    var cascade: [CascadeStage]
    var signals: [IdeaSignal]
    var id: String { symbol }

    static func == (lhs: TradeIdea, rhs: TradeIdea) -> Bool { lhs.symbol == rhs.symbol }
    func hash(into hasher: inout Hasher) { hasher.combine(symbol) }
}

enum StratStatus: String { case live, shadow, paused }

struct Strategy: Identifiable, Hashable {
    var id: String
    var name: String
    var category: String
    var source: String
    var thesis: String
    var status: StratStatus
    // The following performance fields are mock in the data — gated as COMING on
    // screen (per data_readiness.md, never persisted).
    var sharpe: Double
    var winRate: Double
    var pnlYtd: Double
    var allocation: Double
    var contributionPct: Double
    var avgHoldBars: Int
    var regimeFit: [String: Double]
    var params: [(String, String)]
    var equityCurve: [Double]
    var assetClasses: [AssetType]

    static func == (lhs: Strategy, rhs: Strategy) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

struct RetrainEvent: Hashable {
    var date: String
    var event: String
    var sharpe: Double
    var promoted: Bool
}

struct ModelInfo {
    var version: String
    var trainedAt: String
    var lastRetrainHours: Int
    var runId: String
    var type: String
    // Real MLflow run metrics/params (these ARE logged).
    var cvScore: Double // mean of 5-fold CV
    var trainAcc: Double // in-sample accuracy
    var trainingEvents: Int
    // Promotion gate flags — uniformly false because the retrain gate is hard-broken
    // (retrain_pipeline.py:265). Rendered "not run", never green-pass / red-fail.
    var gates: (cpcv: Bool, dsr: Bool, pbo: Bool)
    var metaProbHist: [(String, Int)]
    var retrainTimeline: [RetrainEvent]
}

/// Trust state — derived from the ApiEnvelope (src/web/envelope.py). Only the
/// three honest pills are bound today: Mode · Model · Freshness.
struct TrustState {
    var mode = "PAPER"
    var modelLoaded = true
    var modelVersion = "meta_v1.7.2"
    var stalenessSeconds = 4
    var stale = false
}
