import WidgetKit
import SwiftUI

struct Entry: TimelineEntry {
    let date: Date
    let snap: ApertureSnapshot
}

struct Provider: TimelineProvider {
    func placeholder(in context: Context) -> Entry { Entry(date: Date(), snap: .sample) }
    func getSnapshot(in context: Context, completion: @escaping (Entry) -> Void) {
        completion(Entry(date: Date(), snap: .sample))
    }
    func getTimeline(in context: Context, completion: @escaping (Timeline<Entry>) -> Void) {
        // Production: read the App Group shared cache the TradeIdeaPublisher writes
        // on its cadence, then refresh on that cadence. Today: the honest sample.
        let entry = Entry(date: Date(), snap: .sample)
        completion(Timeline(entries: [entry], policy: .after(Date().addingTimeInterval(300))))
    }
}

// MARK: - Views
struct ApertureWidgetView: View {
    @Environment(\.widgetFamily) private var family
    let entry: Entry

    var body: some View {
        switch family {
        case .accessoryRectangular: rectangular
        case .accessoryInline: inline
        case .systemMedium: medium
        default: small
        }
    }

    private var header: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 6).fill(WTok.grad).frame(width: 18, height: 18)
                .overlay(Text("✦").font(.system(size: 11, weight: .bold)).foregroundStyle(.white))
            Text("Aperture").font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.text2)
            Spacer()
            Text(entry.snap.freshness).font(.system(size: 10)).foregroundStyle(WTok.text3)
        }
    }

    private var counts: some View {
        HStack(spacing: 10) {
            countPill("\(entry.snap.buy)", "Buy", WTok.buy)
            countPill("\(entry.snap.sell)", "Sell", WTok.sell)
            countPill("\(entry.snap.watch)", "Watch", WTok.watch)
        }
    }

    private func countPill(_ n: String, _ label: String, _ color: Color) -> some View {
        VStack(spacing: 1) {
            Text(n).font(.system(size: 17, weight: .bold)).monospacedDigit().foregroundStyle(color)
            Text(label).font(.system(size: 8.5, weight: .semibold)).foregroundStyle(WTok.text3)
        }
    }

    private var small: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            Spacer(minLength: 0)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text("\(entry.snap.total)").font(.system(size: 30, weight: .bold)).monospacedDigit().foregroundStyle(WTok.text1)
                Text("ideas").font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.text2)
            }
            counts
        }
        .containerBackground(WTok.bg0, for: .widget)
    }

    private var medium: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                header
                Spacer(minLength: 0)
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text("\(entry.snap.total)").font(.system(size: 30, weight: .bold)).monospacedDigit().foregroundStyle(WTok.text1)
                    Text("actionable").font(.system(size: 12, weight: .semibold)).foregroundStyle(WTok.text2)
                }
                counts
            }
            Divider().overlay(WTok.surface2)
            VStack(alignment: .leading, spacing: 4) {
                Text("TOP IDEA").font(.system(size: 9, weight: .semibold)).tracking(0.8).foregroundStyle(WTok.text3)
                Text(entry.snap.topSymbol).font(.system(size: 17, weight: .bold, design: .monospaced)).foregroundStyle(WTok.text1)
                Text(entry.snap.topAction).font(.system(size: 11, weight: .semibold)).foregroundStyle(entry.snap.topActionColor)
                    .padding(.horizontal, 7).padding(.vertical, 2)
                    .background(entry.snap.topActionColor.opacity(0.16), in: Capsule())
                Text("target " + signed(entry.snap.topWeight)).font(.system(size: 11)).monospacedDigit().foregroundStyle(WTok.text2)
                Spacer(minLength: 0)
            }
            .frame(maxWidth: 120, alignment: .leading)
        }
        .containerBackground(WTok.bg0, for: .widget)
    }

    private var rectangular: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text("Aperture · \(entry.snap.total) ideas").font(.system(size: 13, weight: .semibold))
            Text("\(entry.snap.buy) buy · \(entry.snap.sell) sell · \(entry.snap.freshness)").font(.system(size: 11))
        }
    }

    private var inline: some View {
        Text("Aperture · \(entry.snap.total) ideas · \(entry.snap.freshness)")
    }

    private func signed(_ v: Double) -> String { (v >= 0 ? "+" : "") + String(format: "%.1f%%", v * 100) }
}

struct ApertureWidget: Widget {
    let kind = "ApertureWidget"
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: Provider()) { entry in
            ApertureWidgetView(entry: entry)
                .widgetURL(URL(string: "aperture://ideas"))
        }
        .configurationDisplayName("Aperture")
        .description("This cycle's actionable ideas + freshness.")
        .supportedFamilies([.systemSmall, .systemMedium, .accessoryRectangular, .accessoryInline])
    }
}

@main
struct ApertureWidgetBundle: WidgetBundle {
    var body: some Widget {
        ApertureWidget()
        SessionLiveActivity()
    }
}
