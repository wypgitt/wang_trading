/**
 * Trade Ideas page — Sprint 1.
 * Calls the BFF, renders the v2 default-visible columns, supports manual
 * refresh. The toolbar matches the design v2 page-toolbar pattern.
 */

import { useTradeIdeas } from "@/api/trade_ideas";
import { TradeIdeasTable } from "@/components/TradeIdeasTable";
import "./TradeIdeasPage.css";

function formatStaleness(seconds: number | undefined): string {
  if (seconds === undefined || seconds === null) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

export function TradeIdeasPage(): JSX.Element {
  const query = useTradeIdeas();
  const env = query.data;
  const totals = env?.data?.totals;

  return (
    <div className="ti-page">
      <div className="ti-toolbar">
        <div className="ti-toolbar-left">
          <h1 className="ti-title">Trade Ideas</h1>
          {env ? (
            <span className="ti-meta">
              {env.data?.idea_count ?? 0} ideas
              {totals
                ? ` · gross ${(totals.gross_target_weight * 100).toFixed(2)}% · net ${
                    totals.net_target_weight >= 0 ? "+" : ""
                  }${(totals.net_target_weight * 100).toFixed(2)}%`
                : ""}
              {" · "}staleness {formatStaleness(env.staleness_seconds)}
              {env.model_version ? ` · model ${env.model_version}` : ""}
            </span>
          ) : null}
        </div>
        <div className="ti-toolbar-actions">
          <button
            type="button"
            className="btn"
            disabled={query.isFetching}
            onClick={() => query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {env && env.warnings.length > 0 ? (
        <div className="ti-warnings" role="status">
          {env.warnings.map((w, i) => (
            <span key={i} className="ti-warning">
              {w}
            </span>
          ))}
        </div>
      ) : null}

      <TradeIdeasTable
        envelope={env}
        isLoading={query.isLoading}
        error={query.error}
      />
    </div>
  );
}
