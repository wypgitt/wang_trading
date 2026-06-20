// Seeded mock dataset — the iOS twin of apps/aperture-web/src/data/mock.ts.
// Series are deterministic (mulberry32) so charts are stable across launches.
import Foundation

// MARK: - Seeded RNG (mulberry32, matches the web rng.ts)
struct Mulberry32 {
    private var a: UInt32
    init(_ seed: Int) { a = UInt32(truncatingIfNeeded: seed) }
    mutating func next() -> Double {
        a = a &+ 0x6D2B79F5
        var t = a
        t = (t ^ (t >> 15)) &* (t | 1)
        t ^= t &+ ((t ^ (t >> 7)) &* (t | 61))
        return Double((t ^ (t >> 14)) & 0xFFFF_FFFF) / 4294967296.0
    }
    mutating func gauss() -> Double {
        var u = 0.0, v = 0.0
        while u == 0 { u = next() }
        while v == 0 { v = next() }
        return (-2.0 * log(u)).squareRoot() * cos(2 * Double.pi * v)
    }
}

func genCandles(seed: Int, n: Int, start: Double, drift: Double, vol: Double) -> [Candle] {
    var rng = Mulberry32(seed)
    var out: [Candle] = []
    var close = start
    for i in 0..<n {
        let o = close
        let d = drift + rng.gauss() * vol
        let c = max(start * 0.04, o * (1 + d))
        let wick = abs(rng.gauss()) * vol * o
        let h = max(o, c) + wick * 0.6
        let l = min(o, c) - wick * 0.6
        let vlm = (0.6 + rng.next()) * 1_000_000
        out.append(Candle(id: i, o: o, h: h, l: l, c: c, v: vlm))
        close = c
    }
    return out
}

// MARK: - Universe definitions
private struct SymDef {
    let symbol: String, name: String; let type: AssetType
    let base: Double, drift: Double, vol: Double; let seed: Int; let cap: Double
}

private let DEFS: [SymDef] = [
    .init(symbol: "NVDA", name: "NVIDIA Corp.", type: .equity, base: 118, drift: 0.0019, vol: 0.026, seed: 7, cap: 2.9e12),
    .init(symbol: "AAPL", name: "Apple Inc.", type: .equity, base: 196, drift: 0.0005, vol: 0.013, seed: 11, cap: 3.0e12),
    .init(symbol: "MSFT", name: "Microsoft Corp.", type: .equity, base: 421, drift: 0.0007, vol: 0.013, seed: 13, cap: 3.1e12),
    .init(symbol: "GOOGL", name: "Alphabet Inc.", type: .equity, base: 176, drift: 0.0006, vol: 0.016, seed: 17, cap: 2.1e12),
    .init(symbol: "AMZN", name: "Amazon.com Inc.", type: .equity, base: 184, drift: 0.0008, vol: 0.018, seed: 19, cap: 1.9e12),
    .init(symbol: "TSLA", name: "Tesla Inc.", type: .equity, base: 248, drift: -0.0003, vol: 0.034, seed: 23, cap: 0.79e12),
    .init(symbol: "META", name: "Meta Platforms", type: .equity, base: 504, drift: 0.0011, vol: 0.02, seed: 29, cap: 1.3e12),
    .init(symbol: "JPM", name: "JPMorgan Chase", type: .equity, base: 205, drift: 0.0004, vol: 0.012, seed: 31, cap: 0.59e12),
    .init(symbol: "SPX", name: "S&P 500 Index", type: .index, base: 5430, drift: 0.0005, vol: 0.0085, seed: 41, cap: 0),
    .init(symbol: "NDX", name: "Nasdaq 100", type: .index, base: 19200, drift: 0.0007, vol: 0.011, seed: 43, cap: 0),
    .init(symbol: "RUT", name: "Russell 2000", type: .index, base: 2030, drift: 0.0002, vol: 0.013, seed: 47, cap: 0),
    .init(symbol: "VIX", name: "CBOE Volatility", type: .index, base: 14.2, drift: -0.0006, vol: 0.05, seed: 53, cap: 0),
    .init(symbol: "BTC", name: "Bitcoin", type: .crypto, base: 61000, drift: 0.0014, vol: 0.028, seed: 61, cap: 1.2e12),
    .init(symbol: "ETH", name: "Ethereum", type: .crypto, base: 3380, drift: 0.0011, vol: 0.032, seed: 67, cap: 0.41e12),
    .init(symbol: "SOL", name: "Solana", type: .crypto, base: 142, drift: 0.0022, vol: 0.045, seed: 71, cap: 0.066e12),
    .init(symbol: "AVAX", name: "Avalanche", type: .crypto, base: 27.4, drift: 0.0006, vol: 0.05, seed: 73, cap: 0.011e12),
    .init(symbol: "ES", name: "E-mini S&P 500", type: .future, base: 5442, drift: 0.0005, vol: 0.0088, seed: 79, cap: 0),
    .init(symbol: "CL", name: "Crude Oil WTI", type: .future, base: 78.6, drift: -0.0004, vol: 0.02, seed: 83, cap: 0),
    .init(symbol: "GC", name: "Gold", type: .future, base: 2338, drift: 0.0009, vol: 0.011, seed: 89, cap: 0),
]

private let IDEA_SYMBOLS: Set<String> = ["NVDA", "TSLA", "BTC", "ETH", "SOL", "META", "AMZN", "GOOGL", "CL", "AVAX", "JPM", "AAPL"]

private func buildBar(_ d: SymDef, _ candles: [Candle]) -> BarMicro {
    let last = candles.last!
    var rng = Mulberry32(d.seed * 131 + 7)
    let isDollar = d.type == .crypto
    let volume = last.v
    let vwap = min(last.h, max(last.l, (last.h + last.l + last.c * 2) / 4))
    let dollarVolume = vwap * volume
    let tickCount = Int(800 + rng.next() * 4200)
    let ret = last.o != 0 ? (last.c - last.o) / last.o : 0
    let buyFrac = min(0.62, max(0.38, 0.5 + ret * 6 + (rng.next() - 0.5) * 0.06))
    let buyVolume = (volume * buyFrac).rounded()
    let sellVolume = volume - buyVolume
    let buyTicks = Int(Double(tickCount) * buyFrac)
    let sellTicks = tickCount - buyTicks
    let sign: Double = buyFrac >= 0.5 ? 1 : -1
    var imbalance = 0.0, threshold = 0.0
    if isDollar {
        threshold = (dollarVolume * (0.96 + rng.next() * 0.03)).rounded()
    } else {
        threshold = (1800 + rng.next() * 3200).rounded()
        imbalance = (sign * threshold * (1 + rng.next() * 0.04)).rounded()
    }
    return BarMicro(
        barType: isDollar ? "dollar" : "tib",
        vwap: (vwap * 100).rounded() / 100, dollarVolume: dollarVolume.rounded(),
        tickCount: tickCount, buyVolume: buyVolume, sellVolume: sellVolume,
        volumeImbalance: buyVolume - sellVolume,
        tickImbalanceRatio: (Double(buyTicks - sellTicks) / Double(tickCount) * 1000).rounded() / 1000,
        imbalance: imbalance, threshold: threshold,
        barDurationSeconds: Int(30 + rng.next() * 600))
}

private func buildSym(_ d: SymDef) -> Sym {
    let candles = genCandles(seed: d.seed, n: 130, start: d.base, drift: d.drift, vol: d.vol)
    let closes = candles.map(\.c)
    let last = closes.last!
    func at(_ back: Int) -> Double { closes[max(0, closes.count - 1 - back)] }
    return Sym(
        symbol: d.symbol, name: d.name, type: d.type, price: last,
        change1d: last / at(1) - 1, change1w: last / at(5) - 1,
        change1m: last / at(21) - 1, changeYtd: last / closes.first! - 1,
        spark: Array(closes.suffix(46)), candles: candles,
        marketCap: d.cap, volume: candles.last!.v * last,
        hasIdea: IDEA_SYMBOLS.contains(d.symbol), bar: buildBar(d, candles))
}

enum MockData {
    static let symbols: [Sym] = DEFS.map(buildSym)
    static func sym(_ s: String) -> Sym { symbols.first { $0.symbol == s } ?? symbols[0] }

    static let trust = TrustState()

    // MARK: Trade ideas
    private struct IdeaDef {
        let symbol: String; let action: Action; let weight: Double
        let family: String; let side: Int
        let meta: Double?; let cal: Double?; let fit: Double
        let cost: Double; let constraints: [String]; let reason: String
    }
    private static let ideaDefs: [IdeaDef] = [
        .init(symbol: "NVDA", action: .BUY, weight: 0.092, family: "ts_momentum", side: 1, meta: 0.74, cal: 0.71, fit: 0.86, cost: 4.2, constraints: ["vol_target"], reason: "Strong 3M time-series momentum (z=1.9), confirmed by MA crossover. Meta-labeler 0.74 / calibrated 0.71 in a trending-up regime (fit 0.86)."),
        .init(symbol: "BTC", action: .BUY, weight: 0.083, family: "ts_momentum", side: 1, meta: 0.70, cal: 0.68, fit: 0.81, cost: 6.1, constraints: ["crypto_cap"], reason: "Donchian 55-day breakout with positive funding carry. Trend + carry agree; sizing clipped by 30% crypto cap."),
        .init(symbol: "META", action: .BUY, weight: 0.071, family: "ts_momentum", side: 1, meta: 0.69, cal: 0.66, fit: 0.78, cost: 3.4, constraints: [], reason: "Strong 12-month time-series momentum, vol-normalized (z=1.6). Trend intact; calibrated 0.66 in a supportive regime."),
        .init(symbol: "ETH", action: .BUY, weight: 0.057, family: "donchian_breakout", side: 1, meta: 0.64, cal: 0.62, fit: 0.74, cost: 5.8, constraints: ["crypto_cap"], reason: "Channel breakout above 55-bar high; ATR-normalized size. Crypto sleeve near cap."),
        .init(symbol: "TSLA", action: .SELL, weight: -0.052, family: "mean_reversion", side: -1, meta: 0.67, cal: 0.64, fit: 0.71, cost: 4.9, constraints: ["kelly_cap"], reason: "O-U z-score +2.3 above mean (half-life 14 bars, ADF p=0.03). Fade the deviation; Kelly cap binding."),
        .init(symbol: "GOOGL", action: .BUY, weight: 0.048, family: "ma_crossover", side: 1, meta: 0.61, cal: 0.59, fit: 0.69, cost: 3.1, constraints: [], reason: "20/50 EMA bullish crossover with triple-MA confirmation. Moderate conviction."),
        .init(symbol: "AMZN", action: .BUY, weight: 0.041, family: "ma_crossover", side: 1, meta: 0.60, cal: 0.58, fit: 0.66, cost: 3.3, constraints: [], reason: "20/50 EMA bullish crossover with triple-MA confirmation."),
        .init(symbol: "SOL", action: .BUY, weight: 0.039, family: "donchian_breakout", side: 1, meta: 0.66, cal: 0.63, fit: 0.70, cost: 7.4, constraints: ["crypto_cap"], reason: "Donchian 55-bar channel breakout; ATR-normalized size. Crypto sleeve near the cap."),
        .init(symbol: "CL", action: .SELL, weight: -0.036, family: "mean_reversion", side: -1, meta: 0.58, cal: 0.56, fit: 0.61, cost: 2.8, constraints: [], reason: "O-U z-score +2.1 above mean (half-life 18 bars, ADF p=0.04). Fade the deviation — short."),
        .init(symbol: "JPM", action: .SELL, weight: -0.028, family: "mean_reversion", side: -1, meta: 0.57, cal: 0.55, fit: 0.60, cost: 2.2, constraints: [], reason: "O-U mean-reversion short: z-score +2.0 above mean on a stationary series."),
        .init(symbol: "AVAX", action: .WATCH, weight: 0, family: "donchian_breakout", side: 1, meta: 0.52, cal: 0.50, fit: 0.55, cost: 8.1, constraints: ["below_threshold"], reason: "Approaching breakout but meta-prob 0.52 below 0.55 entry gate. Watching."),
        .init(symbol: "AAPL", action: .WATCH, weight: 0, family: "mean_reversion", side: 0, meta: 0.49, cal: 0.48, fit: 0.52, cost: 2.6, constraints: ["below_threshold"], reason: "Z-score near band but not at entry threshold. No actionable edge yet."),
        .init(symbol: "RUT", action: .MODEL_REQUIRED, weight: 0, family: "ts_momentum", side: 0, meta: nil, cal: nil, fit: 0.40, cost: 0, constraints: ["insufficient_history"], reason: "Insufficient bar history for the meta-labeler this cycle — inference skipped."),
    ]

    private static func famMeta(_ family: String) -> [(String, String)] {
        switch family {
        case "ts_momentum": return [("lookbacks", "21/63/126/252"), ("z_63", "1.9"), ("aggregate", "0.72")]
        case "cs_momentum": return [("decile_rank", "9"), ("lookback_return", "0.34"), ("skip", "21d")]
        case "mean_reversion": return [("half_life", "14"), ("adf_pvalue", "0.03"), ("z_score", "2.3")]
        case "donchian_breakout": return [("entry_period", "55"), ("exit_period", "20"), ("event", "entry")]
        case "ma_crossover": return [("fast_period", "20"), ("slow_period", "50")]
        case "funding_rate_arb": return [("annualized_funding", "0.18"), ("payments_per_day", "3")]
        case "futures_carry": return [("carry", "-0.06"), ("days_to_expiry", "28")]
        case "stat_arb": return [("hedge_ratio", "0.84"), ("spread_z", "2.1"), ("half_life", "9")]
        default: return []
        }
    }

    static let ideas: [TradeIdea] = ideaDefs.enumerated().map { (i, d) in
        let s = sym(d.symbol)
        let cascade: [CascadeStage] = [
            .init(stage: "AFML size", value: abs(d.weight) * 1.7, binding: false),
            .init(stage: "Kelly cap", value: abs(d.weight) * 1.35, binding: d.constraints.contains("kelly_cap")),
            .init(stage: "Vol target", value: abs(d.weight) * 1.12, binding: d.constraints.contains("vol_target")),
            .init(stage: "ATR cap", value: abs(d.weight) * 1.04, binding: false),
            .init(stage: "Risk budget", value: abs(d.weight), binding: d.constraints.contains("crypto_cap")),
        ]
        var signals: [IdeaSignal] = [
            .init(family: d.family, side: d.side, confidence: (d.meta ?? 0.5) + 0.05, meta: famMeta(d.family))
        ]
        if i % 2 == 0 && d.action != .MODEL_REQUIRED {
            signals.append(.init(family: "ma_crossover", side: d.side, confidence: 0.41,
                                 meta: [("fast_ema", String(format: "%.2f", s.price * 0.99)),
                                        ("slow_ema", String(format: "%.2f", s.price * 0.97))]))
        }
        return TradeIdea(
            symbol: d.symbol, type: s.type, action: d.action, targetWeight: d.weight,
            targetNotional: d.weight * 1_000_000, latestPrice: s.price,
            barType: s.type == .crypto ? "dollar" : "tib", barsLoaded: 980 + i * 7,
            featureRows: 940 + i * 6, signalCount: signals.count, topSignalFamily: d.family,
            topSignalSide: d.side, topSignalConfidence: d.meta.map { $0 + 0.05 },
            avgSignalConfidence: d.meta.map { $0 - 0.02 }, metaProbability: d.meta,
            calibratedProbability: d.cal, regimeFitScore: d.fit,
            betSize: d.weight != 0 ? abs(d.weight) : 0, sizingConstraints: d.constraints,
            strategy: d.family, reason: d.reason, expectedCostBps: d.cost,
            trackRecordWinRate: d.meta != nil ? 0.5 + Double(i % 5) * 0.03 : nil,
            trackRecordN: d.meta != nil ? 40 + i * 9 : nil,
            stageLatency: [("data_fetch", 0.04 + Double(i % 3) * 0.01),
                           ("feature_compute", 0.11 + Double(i % 4) * 0.02),
                           ("signal_generation", 0.03), ("meta_inference", 0.02),
                           ("sizing", 0.01), ("target_generation", 0.01)],
            cascade: cascade, signals: signals)
    }

    static var actionCounts: (buy: Int, sell: Int, watch: Int, modelReq: Int) {
        (ideas.filter { $0.action == .BUY }.count,
         ideas.filter { $0.action == .SELL }.count,
         ideas.filter { $0.action == .WATCH }.count,
         ideas.filter { $0.action == .MODEL_REQUIRED }.count)
    }
    static var topActionable: [TradeIdea] {
        ideas.filter { $0.action == .BUY || $0.action == .SELL }
            .sorted { abs($0.targetWeight) > abs($1.targetWeight) }
    }
    static func idea(for symbol: String) -> TradeIdea? { ideas.first { $0.symbol == symbol } }

    static var enginePulse: (stages: [(String, Double)], total: Double) {
        var acc: [String: Double] = [:]
        let order = ["data_fetch", "feature_compute", "signal_generation", "meta_inference", "sizing", "target_generation"]
        for idea in ideas { for (k, v) in idea.stageLatency { acc[k, default: 0] += v } }
        let stages = order.compactMap { k in acc[k].map { (k, ($0 * 1000).rounded() / 1000) } }
        let total = (stages.reduce(0) { $0 + $1.1 } * 1000).rounded() / 1000
        return (stages, total)
    }

    // MARK: Strategies
    private struct StratDef {
        let id: String, name: String, category: String, source: String, thesis: String
        let status: StratStatus; let sharpe: Double, winRate: Double, pnlYtd: Double
        let allocation: Double, contributionPct: Double; let avgHoldBars: Int
        let regimeFit: [String: Double]; let params: [(String, String)]
        let assetClasses: [AssetType]; let seed: Int; let drift: Double
    }
    private static let stratDefs: [StratDef] = [
        .init(id: "ts_momentum", name: "Time-Series Momentum", category: "Momentum", source: "Clenow · Chan", thesis: "Per-asset momentum across multiple lookbacks, volatility-normalized to z-scores and weighted into a single conviction. Goes long winners / short losers. Best in persistent trends.", status: .live, sharpe: 1.12, winRate: 0.54, pnlYtd: 0.082, allocation: 0.22, contributionPct: 0.26, avgHoldBars: 34, regimeFit: ["trending_up": 0.86, "trending_down": 0.62, "mean_reverting": 0.21, "high_volatility": 0.4], params: [("lookbacks", "21 / 63 / 126 / 252"), ("history_window", "252"), ("vol_normalize", "true")], assetClasses: [.equity, .crypto, .future], seed: 201, drift: 0.0016),
        .init(id: "cs_momentum", name: "Cross-Sectional Momentum", category: "Momentum", source: "Jansen", thesis: "12-month momentum with a 1-month skip to dodge short-term reversal. Ranks the cross-section, longs the top decile and shorts the bottom. Panel-relative alpha.", status: .live, sharpe: 0.94, winRate: 0.52, pnlYtd: 0.051, allocation: 0.15, contributionPct: 0.17, avgHoldBars: 42, regimeFit: ["trending_up": 0.78, "trending_down": 0.55, "mean_reverting": 0.3, "high_volatility": 0.34], params: [("lookback", "252 bars"), ("skip", "21 bars"), ("deciles", "top 0.9 / bottom 0.1")], assetClasses: [.equity], seed: 211, drift: 0.0012),
        .init(id: "mean_reversion", name: "Mean Reversion (O-U)", category: "Mean Reversion", source: "Chan", thesis: "Fits an Ornstein-Uhlenbeck process; trades stationary series back toward the mean when the z-score breaches ±2σ. Half-life and ADF gate which names are tradeable.", status: .live, sharpe: 1.03, winRate: 0.61, pnlYtd: 0.044, allocation: 0.12, contributionPct: 0.14, avgHoldBars: 11, regimeFit: ["trending_up": 0.24, "trending_down": 0.28, "mean_reverting": 0.88, "high_volatility": 0.46], params: [("entry_z", "2.0"), ("exit_z", "0.5"), ("half_life", "1–100 bars"), ("adf_pvalue", "≤ 0.05")], assetClasses: [.equity, .index], seed: 221, drift: 0.0011),
        .init(id: "ma_crossover", name: "MA Crossover", category: "Trend", source: "Clenow", thesis: "Classic EMA crossover with optional triple-MA 2-of-3 voting. Slow, robust trend capture that complements the faster momentum sleeves.", status: .live, sharpe: 0.71, winRate: 0.46, pnlYtd: 0.021, allocation: 0.08, contributionPct: 0.07, avgHoldBars: 51, regimeFit: ["trending_up": 0.82, "trending_down": 0.7, "mean_reverting": 0.18, "high_volatility": 0.3], params: [("fast", "20 EMA"), ("slow", "50 EMA"), ("triple_ma", "on")], assetClasses: [.equity, .future], seed: 231, drift: 0.0008),
        .init(id: "donchian_breakout", name: "Donchian Breakout", category: "Trend", source: "Clenow", thesis: "Turtle-style channel breakout: enter on a new 55-bar extreme, exit on the opposing 20-bar channel. ATR-normalized position sizing. Captures fat-tailed trends.", status: .live, sharpe: 0.88, winRate: 0.43, pnlYtd: 0.033, allocation: 0.1, contributionPct: 0.1, avgHoldBars: 47, regimeFit: ["trending_up": 0.84, "trending_down": 0.66, "mean_reverting": 0.16, "high_volatility": 0.5], params: [("entry_channel", "55 bars"), ("exit_channel", "20 bars"), ("sizing", "ATR units")], assetClasses: [.crypto, .future], seed: 241, drift: 0.001),
        .init(id: "vrp", name: "Volatility Risk Premium", category: "Volatility", source: "Sinclair", thesis: "Trades the gap between implied and realized vol. Sells vol when VRP is rich, buys when cheap, and emits a regime modifier that scales the other families.", status: .live, sharpe: 1.21, winRate: 0.64, pnlYtd: 0.03, allocation: 0.07, contributionPct: 0.09, avgHoldBars: 18, regimeFit: ["trending_up": 0.5, "trending_down": 0.4, "mean_reverting": 0.6, "high_volatility": 0.86], params: [("vrp_lookback", "30 bars"), ("high_pct", "75"), ("low_pct", "25")], assetClasses: [.index], seed: 251, drift: 0.0013),
        .init(id: "futures_carry", name: "Futures Carry", category: "Carry", source: "Clenow", thesis: "Roll-yield harvest from the front/back futures spread. Long backwardation (positive roll), short contango (roll drag). Slow, diversifying carry.", status: .live, sharpe: 0.79, winRate: 0.57, pnlYtd: 0.017, allocation: 0.06, contributionPct: 0.05, avgHoldBars: 63, regimeFit: ["trending_up": 0.55, "trending_down": 0.5, "mean_reverting": 0.52, "high_volatility": 0.38], params: [("annualize", "true"), ("conf_window", "252")], assetClasses: [.future], seed: 261, drift: 0.0007),
        .init(id: "funding_rate_arb", name: "Funding-Rate Arb", category: "Carry", source: "Crypto", thesis: "Delta-neutral crypto carry: long spot, short perpetual, collect funding. Fires when annualized funding clears the entry threshold. High Sharpe, capacity-limited.", status: .live, sharpe: 1.42, winRate: 0.72, pnlYtd: 0.027, allocation: 0.06, contributionPct: 0.08, avgHoldBars: 28, regimeFit: ["trending_up": 0.6, "trending_down": 0.58, "mean_reverting": 0.55, "high_volatility": 0.42], params: [("entry", "10% annualized"), ("exit", "2% annualized"), ("cadence", "3×/day")], assetClasses: [.crypto], seed: 271, drift: 0.0014),
        .init(id: "stat_arb", name: "Statistical Arbitrage", category: "Arbitrage", source: "Chan", thesis: "Cointegration pairs trading with a Kalman-filtered dynamic hedge ratio. Trades the mean-reverting spread; Engle-Granger / Johansen gate the pair selection.", status: .live, sharpe: 0.97, winRate: 0.59, pnlYtd: 0.02, allocation: 0.05, contributionPct: 0.06, avgHoldBars: 13, regimeFit: ["trending_up": 0.4, "trending_down": 0.42, "mean_reverting": 0.82, "high_volatility": 0.5], params: [("entry_z", "2.0"), ("exit_z", "0.5"), ("hedge", "Kalman dynamic")], assetClasses: [.equity], seed: 281, drift: 0.0011),
        .init(id: "cross_exchange_arb", name: "Cross-Exchange Arb", category: "Arbitrage", source: "Crypto", thesis: "Bar-level spot arbitrage across venues. Buys the cheap book, sells the rich one when the spread clears fees. Delta-neutral; currently in shadow pending live venue keys.", status: .shadow, sharpe: 1.55, winRate: 0.81, pnlYtd: 0.009, allocation: 0.02, contributionPct: 0.03, avgHoldBars: 1, regimeFit: ["trending_up": 0.5, "trending_down": 0.5, "mean_reverting": 0.5, "high_volatility": 0.7], params: [("min_spread", "10 bps"), ("fee_estimate", "20 bps")], assetClasses: [.crypto], seed: 291, drift: 0.0009),
    ]
    static let strategies: [Strategy] = stratDefs.map { d in
        Strategy(id: d.id, name: d.name, category: d.category, source: d.source, thesis: d.thesis,
                 status: d.status, sharpe: d.sharpe, winRate: d.winRate, pnlYtd: d.pnlYtd,
                 allocation: d.allocation, contributionPct: d.contributionPct, avgHoldBars: d.avgHoldBars,
                 regimeFit: d.regimeFit, params: d.params,
                 equityCurve: genCandles(seed: d.seed, n: 90, start: 100, drift: d.drift, vol: 0.01).map(\.c),
                 assetClasses: d.assetClasses)
    }
    static func strategy(_ id: String) -> Strategy { strategies.first { $0.id == id } ?? strategies[0] }
    static func activeIdeas(for stratId: String) -> [TradeIdea] { ideas.filter { $0.strategy == stratId } }

    // MARK: Model
    static let model = ModelInfo(
        version: "meta_v1.7.2", trainedAt: "2026-05-12", lastRetrainHours: 62,
        runId: "a1b2c3d4e5f64789",
        type: "LightGBM meta-labeler + isotonic calibration",
        cvScore: 0.582, trainAcc: 0.631, trainingEvents: 4820,
        gates: (cpcv: false, dsr: false, pbo: false),
        metaProbHist: [("0.0", 18), ("0.1", 42), ("0.2", 88), ("0.3", 141), ("0.4", 196),
                       ("0.5", 173), ("0.6", 121), ("0.7", 74), ("0.8", 38), ("0.9", 12)],
        retrainTimeline: [
            .init(date: "2026-05-12", event: "Promoted meta_v1.7.2", sharpe: 1.84, promoted: true),
            .init(date: "2026-04-28", event: "Promoted meta_v1.7.1", sharpe: 1.79, promoted: true),
            .init(date: "2026-04-14", event: "Rejected — PBO 0.46", sharpe: 1.71, promoted: false),
            .init(date: "2026-03-31", event: "Promoted meta_v1.7.0", sharpe: 1.77, promoted: true),
        ])
}
