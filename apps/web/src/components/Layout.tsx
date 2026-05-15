/**
 * Application shell — matches design v2 §7.1.
 *
 * Layout:
 *   Status bar (sticky)
 *   Diff rail placeholder (Sprint 1 — empty until SSE wired)
 *   Body: Sidebar | Page content
 */

import type { ReactNode } from "react";
import type { ApiEnvelope } from "@/api/client";
import type { TradeIdeasResponse } from "@/api/trade_ideas";
import { Sidebar, type PageId } from "./Sidebar";
import { StatusBar } from "./StatusBar";
import "./Layout.css";

interface LayoutProps {
  active: PageId;
  onSelect: (page: PageId) => void;
  statusEnvelope: ApiEnvelope<TradeIdeasResponse> | null | undefined;
  children: ReactNode;
}

export function Layout({ active, onSelect, statusEnvelope, children }: LayoutProps): JSX.Element {
  return (
    <div className="app">
      <StatusBar envelope={statusEnvelope} />
      <DiffRailPlaceholder />
      <div className="app-body">
        <Sidebar active={active} onSelect={onSelect} />
        <main className="app-main">{children}</main>
      </div>
    </div>
  );
}

function DiffRailPlaceholder(): JSX.Element {
  return (
    <div className="diff-rail" role="region" aria-label="Diff since last refresh">
      <span className="diff-rail-label">DIFF</span>
      <span className="diff-rail-empty">No mutations since last refresh</span>
    </div>
  );
}
