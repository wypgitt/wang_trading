import { useState, type ReactNode } from "react";
import { Layout } from "@/components/Layout";
import type { PageId } from "@/components/Sidebar";
import { TradeIdeasPage } from "@/pages/TradeIdeasPage";
import { CommandCenterPage } from "@/pages/CommandCenterPage";
import { useTradeIdeas } from "@/api/trade_ideas";

function PlaceholderPage({ title }: { title: string }): JSX.Element {
  return (
    <div style={{ padding: "32px 4px", display: "flex", flexDirection: "column", gap: 8 }}>
      <h1 style={{ margin: 0, fontSize: "var(--font-size-page-title)", fontWeight: 720 }}>{title}</h1>
      <p style={{ color: "var(--muted)", margin: 0 }}>This page is not implemented in Sprint 1.</p>
    </div>
  );
}

const PAGE_TITLES: Record<PageId, string> = {
  command_center: "Command Center",
  trade_ideas: "Trade Ideas",
  symbol_detail: "Symbol Detail",
  portfolio_risk: "Portfolio & Risk",
  execution_tca: "Execution & TCA",
  signals: "Signals",
  model_features: "Model & Features",
  backtests: "Backtests & Research",
  scenarios: "Scenarios & Stress",
  track_record: "Track Record",
  replay: "Replay",
  preflight: "Preflight & Go-Live",
  monitoring: "Monitoring & Alerts",
  audit: "Audit & Compliance",
  settings: "Settings",
};

export function App(): JSX.Element {
  const [active, setActive] = useState<PageId>("trade_ideas");

  // The status bar's "Last refresh" tracks the trade-ideas envelope as a
  // pragmatic Sprint-1 stand-in for /api/v1/system/overview.
  const statusQuery = useTradeIdeas();

  let page: ReactNode;
  switch (active) {
    case "trade_ideas":
      page = <TradeIdeasPage />;
      break;
    case "command_center":
      page = <CommandCenterPage />;
      break;
    default:
      page = <PlaceholderPage title={PAGE_TITLES[active]} />;
  }

  return (
    <Layout active={active} onSelect={setActive} statusEnvelope={statusQuery.data}>
      {page}
    </Layout>
  );
}
