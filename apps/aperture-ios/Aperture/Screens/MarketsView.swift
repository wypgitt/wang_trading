import SwiftUI

/// Markets — the most finished v1 screen (real bars hypertable). Search + class
/// filter; rows push to Symbol detail.
struct MarketsView: View {
    @State private var filter: FilterOpt = .all
    @State private var query = ""

    enum FilterOpt: Hashable { case all, type(AssetType) }

    private var filtered: [Sym] {
        MockData.symbols.filter { s in
            (filter == .all || filter == .type(s.type)) &&
            (query.isEmpty || s.symbol.localizedCaseInsensitiveContains(query) || s.name.localizedCaseInsensitiveContains(query))
        }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                TrustStrip()

                SegPicker(options: [(.all, "All"), (.type(.equity), "Equities"),
                                    (.type(.index), "Indexes"), (.type(.crypto), "Crypto"),
                                    (.type(.future), "Futures")], selection: $filter)
                    .padding(.bottom, 2)

                Card(padding: 0) {
                    VStack(spacing: 0) {
                        ForEach(filtered) { s in
                            NavigationLink(value: s) { MarketRow(sym: s) }.buttonStyle(.plain)
                            if s.id != filtered.last?.id { Divider().overlay(Tok.border) }
                        }
                    }
                }
            }
            .padding(16)
        }
        .refreshable { try? await Task.sleep(for: .seconds(0.4)) }
        .apertureBackground()
        .navigationTitle("Markets")
        .searchable(text: $query, prompt: "Search symbols")
    }
}

struct MarketRow: View {
    let sym: Sym
    var body: some View {
        HStack(spacing: 12) {
            AssetGlyph(symbol: sym.symbol, type: sym.type, size: 32)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(sym.symbol).font(.system(size: 13.5, weight: .semibold, design: .monospaced))
                    if sym.hasIdea {
                        Circle().fill(Tok.accent).frame(width: 6, height: 6)
                            .accessibilityLabel("Active idea")
                    }
                }
                Text(sym.name).font(.system(size: 11.5)).foregroundStyle(Tok.text3).lineLimit(1)
            }
            Spacer()
            Sparkline(data: sym.spark).frame(width: 64, height: 26)
                .accessibilityLabel("\(sym.symbol) trend, \(sym.change1d >= 0 ? "up" : "down") \(Fmt.pct(abs(sym.change1d), 1)) today\(sym.hasIdea ? ", active idea" : "")")
            VStack(alignment: .trailing, spacing: 2) {
                Text(Fmt.price(sym.price)).font(.system(size: 13.5, weight: .semibold)).monospacedDigit()
                Delta(value: sym.change1d, dp: 1, size: 11.5)
            }
            .frame(width: 84, alignment: .trailing)
            Image(systemName: "chevron.right").font(.system(size: 12, weight: .semibold)).foregroundStyle(Tok.text3)
        }
        .padding(.horizontal, 14).padding(.vertical, 11)
        .contentShape(Rectangle())
    }
}
