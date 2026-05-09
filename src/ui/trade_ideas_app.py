"""Small local HTTP UI for read-only daily trade ideas."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


class TradeIdeasHTTPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # http.server.HTTPServer does a reverse DNS lookup for server_name,
        # which can hang on some local macOS resolvers for 127.0.0.1.
        TCPServer.server_bind(self)
        self.server_name = str(self.server_address[0])
        self.server_port = int(self.server_address[1])

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        config_path: str | None,
        allow_confidence_fallback: bool,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.config_path = config_path
        self.allow_confidence_fallback = allow_confidence_fallback


class TradeIdeasHandler(BaseHTTPRequestHandler):
    server: TradeIdeasHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._send_html(_HTML)
            return
        if parsed.path == "/healthz":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/ideas":
            self._handle_ideas(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep refreshes from flooding the operator terminal.
        return

    def _handle_ideas(self, query: str) -> None:
        params = parse_qs(query)
        symbols = _symbols_arg(params.get("symbols", [""])[0])
        bar_limit = _int_arg(params.get("bar_limit", ["500"])[0], default=500)
        min_abs_weight = _float_arg(
            params.get("min_abs_weight", ["0.0025"])[0],
            default=0.0025,
        )
        allow_fallback = self.server.allow_confidence_fallback or _bool_arg(
            params.get("allow_confidence_fallback", ["0"])[0]
        )
        try:
            from src.ui.trade_ideas import generate_trade_idea_report_sync

            report = generate_trade_idea_report_sync(
                config_path=self.server.config_path,
                symbols=symbols,
                bar_limit=bar_limit,
                min_abs_weight=min_abs_weight,
                allow_confidence_fallback=allow_fallback,
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {
                    "generated_at": None,
                    "mode": "error",
                    "errors": [str(exc)],
                    "warnings": [],
                    "ideas": [],
                    "totals": {},
                    "live_orders_sent": 0,
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(report.to_dict())

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(
    *,
    host: str,
    port: int,
    config_path: str | None,
    allow_confidence_fallback: bool,
) -> None:
    server = TradeIdeasHTTPServer(
        (host, port),
        TradeIdeasHandler,
        config_path=config_path,
        allow_confidence_fallback=allow_confidence_fallback,
    )
    print(f"Trade ideas UI listening on http://{host}:{port}")
    print(f"Config: {config_path or 'auto'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down trade ideas UI")
    finally:
        server.server_close()


def main() -> int:
    args = _parse_args()
    if args.once:
        from src.ui.trade_ideas import generate_trade_idea_report_sync

        report = generate_trade_idea_report_sync(
            config_path=args.config,
            symbols=_symbols_arg(args.symbols),
            bar_limit=args.bar_limit,
            min_abs_weight=args.min_abs_weight,
            allow_confidence_fallback=args.allow_confidence_fallback,
        )
        print(json.dumps(report.to_dict(), default=str, indent=2, sort_keys=True))
        return 0
    run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        allow_confidence_fallback=args.allow_confidence_fallback,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only trade ideas UI")
    parser.add_argument(
        "--config",
        default="config/live_trading.yaml",
        help="Live trading YAML to load in paper-rehearsal mode",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol override",
    )
    parser.add_argument("--bar-limit", type=int, default=500)
    parser.add_argument("--min-abs-weight", type=float, default=0.0025)
    parser.add_argument(
        "--allow-confidence-fallback",
        action="store_true",
        help="Use signal confidence as paper-only meta probability when no production model is loaded",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one JSON report and exit",
    )
    return parser.parse_args()


def _symbols_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    symbols = [s.strip().upper() for s in value.split(",") if s.strip()]
    return symbols or None


def _int_arg(value: str, *, default: int) -> int:
    try:
        return max(1, int(value))
    except Exception:  # noqa: BLE001
        return default


def _float_arg(value: str, *, default: float) -> float:
    try:
        return max(0.0, float(value))
    except Exception:  # noqa: BLE001
        return default


def _bool_arg(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trade Ideas</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #f6f8fb;
      --surface: #ffffff;
      --buy: #0f7a54;
      --buy-bg: #e7f6ef;
      --sell: #b42318;
      --sell-bg: #fdeceb;
      --watch: #7a5a00;
      --watch-bg: #fff7d6;
      --model: #285dad;
      --model-bg: #eaf2ff;
      --error: #7f1d1d;
      --error-bg: #fff1f1;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #eef2f7;
      font-size: 14px;
    }
    header {
      min-height: 84px;
      padding: 20px 28px 18px;
      background: #132238;
      color: #f8fbff;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      border-bottom: 4px solid #36a37d;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.05;
      font-weight: 720;
      letter-spacing: 0;
    }
    .header-meta {
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: #d7e2f2;
    }
    .metric {
      display: grid;
      gap: 3px;
      min-width: 92px;
    }
    .metric span { font-size: 11px; text-transform: uppercase; color: #a9bbd3; }
    .metric strong { font-size: 16px; font-weight: 680; }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px 24px 32px;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(210px, 1fr) 120px 140px auto auto;
      gap: 10px;
      align-items: end;
      padding: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 620;
    }
    input[type="text"], input[type="number"] {
      width: 100%;
      height: 36px;
      border: 1px solid #c9d2df;
      border-radius: 5px;
      padding: 0 10px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    .check {
      height: 36px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
      white-space: nowrap;
    }
    button {
      height: 36px;
      border: 0;
      border-radius: 5px;
      padding: 0 16px;
      font: inherit;
      font-weight: 680;
      color: #fff;
      background: #186b58;
      cursor: pointer;
    }
    button:disabled { opacity: 0.65; cursor: wait; }
    .status {
      min-height: 34px;
      margin: 12px 0;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border-radius: 999px;
      padding: 3px 10px;
      background: #fff;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .pill.warn { background: var(--watch-bg); color: var(--watch); border-color: #f0d870; }
    .pill.err { background: var(--error-bg); color: var(--error); border-color: #f2b8b5; }
    .table-wrap {
      overflow: auto;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }
    table {
      width: 100%;
      min-width: 1120px;
      border-collapse: collapse;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #edf0f5;
      text-align: right;
      vertical-align: middle;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      color: #526071;
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 760;
    }
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:last-child, td:last-child {
      text-align: left;
    }
    td.reason {
      min-width: 260px;
      white-space: normal;
      color: #38465a;
      line-height: 1.35;
    }
    .symbol {
      font-weight: 760;
      font-size: 15px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 78px;
      height: 26px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
    }
    .badge.buy { color: var(--buy); background: var(--buy-bg); }
    .badge.sell { color: var(--sell); background: var(--sell-bg); }
    .badge.watch, .badge.no_data { color: var(--watch); background: var(--watch-bg); }
    .badge.model_required { color: var(--model); background: var(--model-bg); }
    .badge.error { color: var(--error); background: var(--error-bg); }
    .empty {
      padding: 36px;
      color: var(--muted);
      text-align: center;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    @media (max-width: 860px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .header-meta {
        justify-content: flex-start;
      }
      main {
        padding: 14px;
      }
      .controls {
        grid-template-columns: 1fr 1fr;
      }
      .controls label:first-child {
        grid-column: 1 / -1;
      }
      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Trade Ideas</h1>
    </div>
    <div class="header-meta">
      <div class="metric"><span>Generated</span><strong id="generated">-</strong></div>
      <div class="metric"><span>Mode</span><strong id="mode">-</strong></div>
      <div class="metric"><span>NAV</span><strong id="nav">-</strong></div>
      <div class="metric"><span>Live orders</span><strong id="orders">0</strong></div>
    </div>
  </header>

  <main>
    <form class="controls" id="controls">
      <label>Symbols
        <input id="symbols" type="text" autocomplete="off" placeholder="AAPL,MSFT,NVDA">
      </label>
      <label>Bars
        <input id="barLimit" type="number" min="1" step="1" value="500">
      </label>
      <label>Min weight
        <input id="minWeight" type="number" min="0" step="0.001" value="0.0025">
      </label>
      <label class="check">
        <input id="fallback" type="checkbox">
        Paper fallback
      </label>
      <button id="refresh" type="submit">Refresh</button>
    </form>

    <div class="status" id="status"></div>

    <div class="table-wrap" id="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Action</th>
            <th>Target</th>
            <th>Notional</th>
            <th>Est qty</th>
            <th>Price</th>
            <th>Meta</th>
            <th>Confidence</th>
            <th>Signals</th>
            <th>Top family</th>
            <th>Bar time</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody id="ideas"></tbody>
      </table>
    </div>
    <div class="empty" id="empty" hidden>No trade ideas loaded.</div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const els = {
      generated: $("generated"),
      mode: $("mode"),
      nav: $("nav"),
      orders: $("orders"),
      status: $("status"),
      tableWrap: $("tableWrap"),
      empty: $("empty"),
      ideas: $("ideas"),
      refresh: $("refresh"),
      controls: $("controls"),
      symbols: $("symbols"),
      barLimit: $("barLimit"),
      minWeight: $("minWeight"),
      fallback: $("fallback"),
    };

    function formatPct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return `${(Number(value) * 100).toFixed(2)}%`;
    }
    function formatUsd(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return new Intl.NumberFormat(undefined, {
        style: "currency", currency: "USD", maximumFractionDigits: 0
      }).format(Number(value));
    }
    function formatPrice(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(2);
    }
    function formatQty(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    function formatTime(value) {
      if (!value) return "-";
      const d = new Date(value);
      return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
    }
    function setText(node, value) {
      node.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
    }
    function addCell(row, value, className) {
      const cell = document.createElement("td");
      if (className) cell.className = className;
      setText(cell, value);
      row.appendChild(cell);
      return cell;
    }
    function renderStatus(report) {
      els.status.replaceChildren();
      const base = [
        `Model: ${report.model_source || "-"}`,
        `Bar type: ${report.bar_type || "-"}`,
        `Gross: ${formatPct(report.totals?.gross_target_weight || 0)}`,
        `Net: ${formatPct(report.totals?.net_target_weight || 0)}`,
      ];
      base.forEach((text) => {
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = text;
        els.status.appendChild(pill);
      });
      (report.warnings || []).forEach((text) => {
        const pill = document.createElement("span");
        pill.className = "pill warn";
        pill.textContent = text;
        els.status.appendChild(pill);
      });
      (report.errors || []).forEach((text) => {
        const pill = document.createElement("span");
        pill.className = "pill err";
        pill.textContent = text;
        els.status.appendChild(pill);
      });
    }
    function renderIdeas(report) {
      els.generated.textContent = formatTime(report.generated_at);
      els.mode.textContent = report.mode || "-";
      els.nav.textContent = formatUsd(report.nav);
      els.orders.textContent = String(report.live_orders_sent || 0);
      renderStatus(report);
      els.ideas.replaceChildren();
      const ideas = report.ideas || [];
      els.tableWrap.hidden = ideas.length === 0;
      els.empty.hidden = ideas.length !== 0;
      ideas.forEach((idea) => {
        const row = document.createElement("tr");
        addCell(row, idea.symbol, "symbol");
        const actionCell = addCell(row, "", "");
        const badge = document.createElement("span");
        const action = String(idea.action || "WATCH").toLowerCase();
        badge.className = `badge ${action}`;
        badge.textContent = idea.action || "WATCH";
        actionCell.appendChild(badge);
        addCell(row, formatPct(idea.target_weight));
        addCell(row, formatUsd(idea.target_notional));
        addCell(row, formatQty(idea.estimated_quantity));
        addCell(row, formatPrice(idea.latest_price));
        addCell(row, formatPct(idea.meta_probability));
        addCell(row, formatPct(idea.top_signal_confidence));
        addCell(row, idea.signal_count);
        addCell(row, idea.top_signal_family || "-");
        addCell(row, formatTime(idea.latest_bar_at));
        addCell(row, idea.reason || "-", "reason");
        els.ideas.appendChild(row);
      });
    }
    async function loadIdeas() {
      els.refresh.disabled = true;
      const params = new URLSearchParams();
      if (els.symbols.value.trim()) params.set("symbols", els.symbols.value.trim());
      params.set("bar_limit", els.barLimit.value || "500");
      params.set("min_abs_weight", els.minWeight.value || "0.0025");
      if (els.fallback.checked) params.set("allow_confidence_fallback", "1");
      try {
        const response = await fetch(`/api/ideas?${params.toString()}`, { cache: "no-store" });
        const payload = await response.json();
        renderIdeas(payload);
      } catch (error) {
        renderIdeas({
          generated_at: null,
          mode: "error",
          nav: null,
          live_orders_sent: 0,
          warnings: [],
          errors: [String(error)],
          ideas: [],
          totals: {},
        });
      } finally {
        els.refresh.disabled = false;
      }
    }
    els.controls.addEventListener("submit", (event) => {
      event.preventDefault();
      loadIdeas();
    });
    loadIdeas();
    setInterval(loadIdeas, 300000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
