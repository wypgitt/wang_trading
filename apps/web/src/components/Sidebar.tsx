/**
 * Left sidebar — 15 nav items per design v2 §7.2.
 * Active page is lifted to App; clicking a nav item dispatches onSelect.
 */

import "./Sidebar.css";

export type PageId =
  | "command_center"
  | "trade_ideas"
  | "symbol_detail"
  | "portfolio_risk"
  | "execution_tca"
  | "signals"
  | "model_features"
  | "backtests"
  | "scenarios"
  | "track_record"
  | "replay"
  | "preflight"
  | "monitoring"
  | "audit"
  | "settings";

interface NavItem {
  id: PageId;
  label: string;
}

const NAV_ITEMS: readonly NavItem[] = [
  { id: "command_center", label: "Command Center" },
  { id: "trade_ideas", label: "Trade Ideas" },
  { id: "symbol_detail", label: "Symbol Detail" },
  { id: "portfolio_risk", label: "Portfolio & Risk" },
  { id: "execution_tca", label: "Execution & TCA" },
  { id: "signals", label: "Signals" },
  { id: "model_features", label: "Model & Features" },
  { id: "backtests", label: "Backtests & Research" },
  { id: "scenarios", label: "Scenarios & Stress" },
  { id: "track_record", label: "Track Record" },
  { id: "replay", label: "Replay" },
  { id: "preflight", label: "Preflight & Go-Live" },
  { id: "monitoring", label: "Monitoring & Alerts" },
  { id: "audit", label: "Audit & Compliance" },
  { id: "settings", label: "Settings" },
];

interface SidebarProps {
  active: PageId;
  onSelect: (page: PageId) => void;
}

export function Sidebar({ active, onSelect }: SidebarProps): JSX.Element {
  return (
    <nav className="sidebar" aria-label="Primary navigation">
      <div className="sidebar-brand">Wang Trading</div>
      <ul className="sidebar-list">
        {NAV_ITEMS.map((item) => {
          const isActive = item.id === active;
          return (
            <li key={item.id}>
              <button
                type="button"
                className={`sidebar-item${isActive ? " sidebar-item-active" : ""}`}
                aria-current={isActive ? "page" : undefined}
                onClick={() => onSelect(item.id)}
              >
                <span className="sidebar-marker" aria-hidden="true" />
                <span className="sidebar-label">{item.label}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
