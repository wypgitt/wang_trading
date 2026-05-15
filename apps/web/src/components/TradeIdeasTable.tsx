/**
 * Trade Ideas table — default visible columns per design v2 §13.2:
 *   Symbol, Action, Target Weight, Notional, Est Qty, Price,
 *   Meta, Top Confidence, Signals, Top Family, Bar time, Reason.
 *
 * Loading: 5-row skeleton. Empty: centered message. Error: renders the
 * envelope errors[].message.
 */

import { ApiError, type ApiEnvelope, type ApiErrorItem } from "@/api/client";
import type { TradeAction, TradeIdea, TradeIdeasResponse } from "@/api/trade_ideas";
import "./TradeIdeasTable.css";

interface TradeIdeasTableProps {
  envelope: ApiEnvelope<TradeIdeasResponse> | null | undefined;
  isLoading: boolean;
  error: Error | null;
}

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const QTY = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const PRICE = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function fmtUsd(v: number | null): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return USD.format(v);
}

function fmtPct(v: number | null): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(2)}%`;
}

function fmtProb(v: number | null): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(2);
}

function fmtQty(v: number | null): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return QTY.format(v);
}

function fmtPrice(v: number | null): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return PRICE.format(v);
}

function fmtBarTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${hh}:${mm}Z`;
}

const ACTION_CLASS: Record<TradeAction, string> = {
  BUY: "pill pill-buy",
  SELL: "pill pill-sell",
  WATCH: "pill pill-watch",
  MODEL_REQUIRED: "pill pill-model",
  NO_DATA: "pill pill-nodata",
  ERROR: "pill pill-error",
};

const ACTION_LABEL: Record<TradeAction, string> = {
  BUY: "BUY",
  SELL: "SELL",
  WATCH: "WATCH",
  MODEL_REQUIRED: "MODEL?",
  NO_DATA: "NO DATA",
  ERROR: "ERROR",
};

function ActionPill({ action }: { action: TradeAction }): JSX.Element {
  return <span className={ACTION_CLASS[action] ?? "pill"}>{ACTION_LABEL[action] ?? action}</span>;
}

function weightClass(w: number | null): string {
  if (w === null || w === undefined || w === 0) return "num";
  return w > 0 ? "num num-pos" : "num num-neg";
}

function TableHeader(): JSX.Element {
  return (
    <thead>
      <tr>
        <th className="col-text">Symbol</th>
        <th className="col-text">Action</th>
        <th>Target Weight</th>
        <th>Notional</th>
        <th>Est Qty</th>
        <th>Price</th>
        <th>Meta</th>
        <th>Top Conf.</th>
        <th>Signals</th>
        <th className="col-text">Top Family</th>
        <th>Bar Time</th>
        <th className="col-text col-wide">Reason</th>
      </tr>
    </thead>
  );
}

function IdeaRow({ idea }: { idea: TradeIdea }): JSX.Element {
  return (
    <tr>
      <td className="col-text col-sym">{idea.symbol}</td>
      <td className="col-text">
        <ActionPill action={idea.action} />
      </td>
      <td className={weightClass(idea.target_weight)}>{fmtPct(idea.target_weight)}</td>
      <td className="num">{fmtUsd(idea.target_notional)}</td>
      <td className="num">{fmtQty(idea.estimated_quantity)}</td>
      <td className="num">{fmtPrice(idea.latest_price)}</td>
      <td className="num">{fmtProb(idea.meta_probability)}</td>
      <td className="num">{fmtProb(idea.top_signal_confidence)}</td>
      <td className="num">{idea.signal_count ?? "—"}</td>
      <td className="col-text">{idea.top_signal_family ?? "—"}</td>
      <td className="num">{fmtBarTime(idea.latest_bar_at)}</td>
      <td className="col-text col-wide col-reason">{idea.reason ?? "—"}</td>
    </tr>
  );
}

function SkeletonRows(): JSX.Element {
  return (
    <tbody>
      {Array.from({ length: 5 }, (_, i) => (
        <tr key={i} className="skel-row">
          {Array.from({ length: 12 }, (__, j) => (
            <td key={j}>
              <span className="skel-block" />
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  );
}

function ErrorBlock({ errors, fallback }: { errors: ApiErrorItem[]; fallback: string }): JSX.Element {
  if (errors.length === 0) {
    return (
      <div className="ideas-error" role="alert">
        <span className="ideas-error-title">Error loading trade ideas</span>
        <span className="ideas-error-msg">{fallback}</span>
      </div>
    );
  }
  return (
    <div className="ideas-error" role="alert">
      <span className="ideas-error-title">
        {errors.length === 1 ? "Error loading trade ideas" : `${errors.length} errors loading trade ideas`}
      </span>
      <ul className="ideas-error-list">
        {errors.map((e, i) => (
          <li key={`${e.code}-${i}`}>
            <code>{e.code}</code> — {e.message}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function TradeIdeasTable({ envelope, isLoading, error }: TradeIdeasTableProps): JSX.Element {
  if (error) {
    // ApiError instances carry the envelope through; non-ApiError errors don't.
    const errs: ApiErrorItem[] =
      envelope?.errors ??
      (error instanceof ApiError ? error.envelope?.errors ?? [] : []);
    return <ErrorBlock errors={errs} fallback={error.message} />;
  }

  if (isLoading) {
    return (
      <div className="ideas-card">
        <table className="ideas-table">
          <TableHeader />
          <SkeletonRows />
        </table>
      </div>
    );
  }

  const ideas = envelope?.data?.ideas ?? [];

  if (ideas.length === 0) {
    return (
      <div className="ideas-card">
        <table className="ideas-table">
          <TableHeader />
        </table>
        <div className="ideas-empty">No trade ideas at this time.</div>
      </div>
    );
  }

  return (
    <div className="ideas-card">
      <table className="ideas-table">
        <TableHeader />
        <tbody>
          {ideas.map((idea) => (
            <IdeaRow key={idea.symbol} idea={idea} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
