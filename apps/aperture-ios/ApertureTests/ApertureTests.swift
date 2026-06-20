import XCTest
@testable import Aperture

/// Unit tests mirroring the web Vitest suite — they lock in the data-honesty
/// invariants so a regression that fakes data (or attributes a live idea to a
/// dormant family) fails CI.
final class ApertureTests: XCTestCase {

    // MARK: Family readiness
    func testFamilyCountsAreFourActiveSixInactive() {
        XCTAssertEqual(Families.counts.active, 4)
        XCTAssertEqual(Families.counts.inactive, 6)
        XCTAssertEqual(Families.counts.total, 10)
    }

    func testOnlyBarsFamiliesActive() {
        let active = Families.map.filter { $0.value.active }.keys.sorted()
        XCTAssertEqual(active, ["donchian_breakout", "ma_crossover", "mean_reversion", "ts_momentum"])
    }

    func testEveryDormantFamilyHasAReason() {
        for (id, fr) in Families.map where !fr.active {
            XCTAssertNotNil(fr.reason, "\(id) should carry a dormancy reason")
        }
    }

    // MARK: Data honesty
    func testEveryLiveIdeaComesFromAnActiveFamily() {
        for idea in MockData.ideas {
            let fam = idea.strategy ?? ""
            XCTAssertTrue(Families.of(fam).active, "\(idea.symbol) → \(fam) must be an active family")
        }
    }

    func testNoDormantFamilyHasLiveIdeas() {
        for (id, fr) in Families.map where !fr.active {
            XCTAssertTrue(MockData.activeIdeas(for: id).isEmpty, "\(id) must have no live ideas")
        }
    }

    func testModelGatesAreAllFalseNotRun() {
        XCTAssertFalse(MockData.model.gates.cpcv)
        XCTAssertFalse(MockData.model.gates.dsr)
        XCTAssertFalse(MockData.model.gates.pbo)
    }

    func testModelExposesRealMlflowStats() {
        XCTAssertGreaterThan(MockData.model.cvScore, 0)
        XCTAssertGreaterThan(MockData.model.trainingEvents, 0)
        XCTAssertFalse(MockData.model.runId.isEmpty)
    }

    // MARK: Action counts + ordering
    func testActionCountsAreStable() {
        let c = MockData.actionCounts
        XCTAssertEqual(c.buy, 7)
        XCTAssertEqual(c.sell, 3)
        XCTAssertEqual(c.watch, 2)
        XCTAssertEqual(c.modelReq, 1)
    }

    func testTopActionableSortedByWeightDesc() {
        let w = MockData.topActionable.map { Swift.abs($0.targetWeight) }
        XCTAssertEqual(w, w.sorted(by: >))
        XCTAssertTrue(MockData.topActionable.allSatisfy { $0.action == .BUY || $0.action == .SELL })
    }

    // MARK: Formatting (parity with web)
    func testCompactAddsTrillionTier() {
        XCTAssertEqual(Fmt.compact(2.9e12), "$2.90T")
        XCTAssertEqual(Fmt.compact(3.4e9), "$3.40B")
    }

    func testPctSignedAndProb() {
        XCTAssertEqual(Fmt.pctSigned(0.092, 1), "+9.2%")
        XCTAssertEqual(Fmt.pctSigned(-0.052, 1), "-5.2%")
        XCTAssertEqual(Fmt.prob(nil), "—")
        XCTAssertEqual(Fmt.prob(0.71), "0.71")
    }

    // MARK: Seeded RNG determinism
    func testMulberry32IsDeterministic() {
        var a = Mulberry32(42)
        var b = Mulberry32(42)
        XCTAssertEqual(a.next(), b.next())
        XCTAssertEqual(MockData.symbols.count, 19)
    }

    // MARK: Notification deep-link parsing
    func testDeepLinkParsing() {
        XCTAssertEqual(DeepLink.from(["screen": "idea", "symbol": "NVDA"]), .idea("NVDA"))
        XCTAssertEqual(DeepLink.from(["screen": "idea"]), .ideasList)
        XCTAssertEqual(DeepLink.from(["screen": "ideas"]), .ideasList)
        XCTAssertEqual(DeepLink.from(["screen": "symbol", "symbol": "BTC"]), .symbol("BTC"))
        XCTAssertEqual(DeepLink.from(["screen": "strategy", "id": "ts_momentum"]), .strategy("ts_momentum"))
        XCTAssertEqual(DeepLink.from(["screen": "markets"]), .tab("markets"))
        XCTAssertNil(DeepLink.from(["foo": "bar"]))
        // Required-field contract: screen=symbol needs symbol, screen=strategy needs id.
        XCTAssertNil(DeepLink.from(["screen": "symbol"]))
        XCTAssertNil(DeepLink.from(["screen": "strategy"]))
    }

    func testDeepLinkFromURL() {
        XCTAssertEqual(DeepLink.from(URL(string: "aperture://idea?symbol=NVDA")!), .idea("NVDA"))
        XCTAssertEqual(DeepLink.from(URL(string: "aperture://ideas")!), .ideasList)
        XCTAssertEqual(DeepLink.from(URL(string: "aperture://symbol?symbol=BTC")!), .symbol("BTC"))
        XCTAssertNil(DeepLink.from(URL(string: "https://example.com")!))
    }

    // MARK: Live Activity content is the honest decision snapshot (never NAV)
    func testSessionSnapshotIsHonest() {
        let s = SessionActivityController.snapshot()
        XCTAssertEqual(s.buy, 7)
        XCTAssertEqual(s.sell, 3)
        XCTAssertEqual(s.watch, 2)
        XCTAssertEqual(s.total, 12)
        XCTAssertEqual(s.topSymbol, "NVDA")   // top by |target weight|
        XCTAssertEqual(s.topAction, "Buy")
    }
}
