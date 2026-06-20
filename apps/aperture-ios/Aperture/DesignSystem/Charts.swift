// Charts — iOS twin of src/components/charts. Sparkline + candles are custom
// (full control), the area hero uses Swift Charts.
import SwiftUI
import Charts

// MARK: - Sparkline
struct Sparkline: View {
    let data: [Double]
    var color: Color? = nil
    var fill: Bool = true
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let lo = data.min() ?? 0
            let hi = data.max() ?? 1
            let range = (hi - lo) == 0 ? 1 : (hi - lo)
            let stroke = color ?? ((data.last ?? 0) >= (data.first ?? 0) ? Tok.pos : Tok.neg)
            let pt: (Int) -> CGPoint = { i in
                let x = data.count <= 1 ? 0 : w * CGFloat(i) / CGFloat(data.count - 1)
                let y = h - CGFloat((data[i] - lo) / range) * (h - 3) - 1.5
                return CGPoint(x: x, y: y)
            }
            ZStack {
                if fill {
                    Path { p in
                        p.move(to: CGPoint(x: 0, y: h))
                        for i in data.indices { p.addLine(to: pt(i)) }
                        p.addLine(to: CGPoint(x: w, y: h)); p.closeSubpath()
                    }.fill(stroke.opacity(0.1))
                }
                Path { p in
                    for i in data.indices { i == 0 ? p.move(to: pt(i)) : p.addLine(to: pt(i)) }
                }.stroke(stroke, style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
            }
        }
    }
}

// MARK: - Area hero chart (Swift Charts)
struct AreaChartView: View {
    let data: [Double]
    var color: Color = Tok.accent
    var body: some View {
        let lo = data.min() ?? 0
        let hi = data.max() ?? 1
        Chart(Array(data.enumerated()), id: \.offset) { item in
            AreaMark(x: .value("t", item.offset), yStart: .value("lo", lo), yEnd: .value("v", item.element))
                .foregroundStyle(LinearGradient(colors: [color.opacity(0.34), color.opacity(0)],
                                                startPoint: .top, endPoint: .bottom))
                .interpolationMethod(.monotone)
            LineMark(x: .value("t", item.offset), y: .value("v", item.element))
                .foregroundStyle(color)
                .lineStyle(StrokeStyle(lineWidth: 2))
                .interpolationMethod(.monotone)
        }
        .chartYScale(domain: lo...(hi == lo ? lo + 1 : hi))
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
    }
}

// MARK: - Candlestick + volume (custom)
struct CandleChartView: View {
    let candles: [Candle]
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let padR: CGFloat = 50, padT: CGFloat = 6, padB: CGFloat = 6
            let volH: CGFloat = 40, gap: CGFloat = 8
            let priceH = h - volH - padT - padB - gap
            let hi = candles.map(\.h).max() ?? 1
            let lo = candles.map(\.l).min() ?? 0
            let range = (hi - lo) == 0 ? 1 : (hi - lo)
            let plotW = Swift.max(0, w - padR)
            let n = candles.count
            let cw = n == 0 ? 0 : plotW / CGFloat(n)
            let bodyW = Swift.max(1.5, cw * 0.62)
            let volMax = candles.map(\.v).max() ?? 1
            let y: (Double) -> CGFloat = { p in padT + (1 - CGFloat((p - lo) / range)) * priceH }
            let last = candles.last
            let lastUp = (last?.c ?? 0) >= (last?.o ?? 0)

            ZStack(alignment: .topLeading) {
                ForEach(0..<5, id: \.self) { i in
                    let gv = lo + range * (Double(i) / 4)
                    let yy = y(gv)
                    Rectangle().fill(Tok.grid).frame(height: 1).offset(y: yy)
                    Text(Fmt.price(gv)).font(.system(size: 9.5, design: .monospaced))
                        .foregroundStyle(Tok.text3).offset(x: w - padR + 6, y: yy - 6)
                }
                ForEach(candles) { c in
                    let x = CGFloat(c.id) * cw + cw / 2
                    let up = c.c >= c.o
                    let col = up ? Tok.pos : Tok.neg
                    let bodyTop = Swift.min(y(c.o), y(c.c))
                    let bodyH = Swift.max(1, Swift.abs(y(c.o) - y(c.c)))
                    let vh = CGFloat(c.v / volMax) * volH
                    Rectangle().fill(col.opacity(0.9)).frame(width: 1, height: Swift.abs(y(c.l) - y(c.h)))
                        .offset(x: x, y: y(c.h))
                    RoundedRectangle(cornerRadius: 1).fill(col).frame(width: bodyW, height: bodyH)
                        .offset(x: x - bodyW / 2, y: bodyTop)
                    RoundedRectangle(cornerRadius: 0.5).fill(col.opacity(0.32)).frame(width: bodyW, height: vh)
                        .offset(x: x - bodyW / 2, y: padT + priceH + gap + (volH - vh))
                }
                if let last {
                    let yy = y(last.c)
                    Rectangle().fill((lastUp ? Tok.pos : Tok.neg).opacity(0.6)).frame(height: 1).offset(y: yy)
                    Text(Fmt.price(last.c))
                        .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                        .foregroundStyle(Color(hex: 0x06120D))
                        .padding(.horizontal, 4).padding(.vertical, 1.5)
                        .background(lastUp ? Tok.pos : Tok.neg, in: RoundedRectangle(cornerRadius: 4))
                        .offset(x: w - padR + 2, y: yy - 9)
                }
            }
        }
    }
}

// MARK: - Horizontal labeled bars (importance, contributions)
struct MiniBar: View {
    let label: String
    let value: Double
    let max: Double
    var color: Color = Tok.accent
    var valueText: String
    var body: some View {
        HStack(spacing: 12) {
            Text(label).font(.system(size: 12, design: .monospaced)).foregroundStyle(Tok.text2)
                .frame(width: 130, alignment: .leading).lineLimit(1)
            Bar(value: value, max: max, color: color, height: 9)
            Text(valueText).font(.system(size: 12)).monospacedDigit().foregroundStyle(Tok.text1)
                .frame(width: 56, alignment: .trailing)
        }
    }
}
