/**
 * Sticky global status bar — matches the wireframe `.status-bar` layout.
 * 14 fields per design v2 §7.3. Sprint 1 wires only "Last refresh" from
 * the live trade-ideas envelope; the rest use mock placeholders that
 * will be replaced by /api/v1/system/overview in a later sprint.
 */

import type { ReactNode } from "react";
import type { ApiEnvelope } from "@/api/client";
import type { TradeIdeasResponse } from "@/api/trade_ideas";
import "./StatusBar.css";

interface StatusBarProps {
  envelope: ApiEnvelope<TradeIdeasResponse> | null | undefined;
}

interface BadgeProps {
  tone: "ok" | "warn" | "crit" | "regime-up";
  text: string;
}

function Badge({ tone, text }: BadgeProps): JSX.Element {
  return <span className={`sb-badge sb-badge-${tone}`}>{text}</span>;
}

function formatRefresh(iso: string | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}Z`;
}

interface StatusCellProps {
  label: string;
  children: ReactNode;
  sub?: string;
}

function StatusCell({ label, children, sub }: StatusCellProps): JSX.Element {
  return (
    <div className="sb-cell" role="button" tabIndex={0}>
      <span className="sb-label">{label}</span>
      <span className="sb-value">{children}</span>
      {sub ? <span className="sb-sub">{sub}</span> : null}
    </div>
  );
}

export function StatusBar({ envelope }: StatusBarProps): JSX.Element {
  // Mock values for everything except Last Refresh (Sprint 1 placeholder).
  const lastRefresh = formatRefresh(envelope?.as_of);

  return (
    <div className="sb" role="banner" aria-label="Global status bar">
      <StatusCell label="NAV">
        <span className="sb-num">$10.12M</span>
      </StatusCell>
      <StatusCell label="Daily P&L" sub="+0.24%">
        <span className="sb-num sb-pos">+$24.1k</span>
      </StatusCell>
      <StatusCell label="Drawdown">
        <span className="sb-num">-1.41%</span>
      </StatusCell>
      <StatusCell label="Gross">
        <span className="sb-num">24.6%</span>
      </StatusCell>
      <StatusCell label="Net">
        <span className="sb-num">+6.2%</span>
      </StatusCell>
      <StatusCell label="Model" sub="3d old">
        <span className="sb-num">meta_v1.7.2</span>
      </StatusCell>
      <StatusCell label="Data" sub="worst: sentiment 84s">
        <Badge tone="ok" text="OK 2s" />
      </StatusCell>
      <StatusCell label="Regime">
        <Badge tone="regime-up" text="TREND UP 0.72" />
      </StatusCell>
      <StatusCell label="Drift">
        <Badge tone="warn" text="3 feat." />
      </StatusCell>
      <StatusCell label="Cal">
        <Badge tone="ok" text="3d" />
      </StatusCell>
      <StatusCell label="Broker">
        <Badge tone="ok" text="41ms" />
      </StatusCell>
      <StatusCell label="Breakers">
        <Badge tone="ok" text="0/8" />
      </StatusCell>
      <StatusCell label="Alerts">
        <Badge tone="warn" text="2 warn" />
      </StatusCell>
      <StatusCell label="Last refresh" sub="auto 30s">
        <span className="sb-num">{lastRefresh}</span>
      </StatusCell>
    </div>
  );
}
