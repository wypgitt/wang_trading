# wang-web

React SPA for Wang Trading. Read-only operator console; talks to the
FastAPI BFF (`src/web/`) over `/api/v1/*`.

## Stack

- React 18 + TypeScript (strict) + Vite 5
- TanStack Query v5 for server state
- Plain CSS with design tokens — no component library
- pnpm 9 as the package manager (Node 20+)

Locked in `docs/backend_design.md` §24 and §24a.

## Layout

```
apps/web/
  index.html             # Vite entry; mounts #root, loads /src/main.tsx
  vite.config.ts         # Dev server proxies /api → http://127.0.0.1:8080
  tsconfig.json          # strict + path alias "@/* → src/*"
  src/
    main.tsx             # createRoot + QueryClientProvider
    App.tsx              # Route shell (state-based, router lands later)
    api/
      client.ts          # apiFetch + ApiEnvelope<T> (api_contracts §0.1)
      trade_ideas.ts     # TradeIdea types + useTradeIdeas() hook
    components/
      Layout.tsx         # StatusBar + Diff rail + Sidebar | main
      Sidebar.tsx        # 15 nav items (design v2 §7.2)
      StatusBar.tsx      # 14 fields (design v2 §7.3)
      TradeIdeasTable.tsx# v2 default-visible columns (design v2 §13.2)
    pages/
      TradeIdeasPage.tsx
      CommandCenterPage.tsx
    styles/
      tokens.css         # Color/typography/spacing tokens (design v2 §8)
      global.css         # Resets, helper classes, pill styles
```

## Running

The BFF must be up first. From the repo root:

```
uvicorn src.web.app:app --host 127.0.0.1 --port 8080 --reload
```

Then in this directory:

```
pnpm install
pnpm dev
```

Vite serves at <http://localhost:5173> and proxies `/api/*` to
`127.0.0.1:8080`, so the trade-ideas table hits a real BFF response with
the envelope from `docs/api_contracts_v2.md` §0.1 and §1.1.

## Scripts

| Script | Purpose |
| --- | --- |
| `pnpm dev` | Vite dev server with HMR + API proxy |
| `pnpm build` | `tsc --noEmit` then `vite build` → `dist/` |
| `pnpm preview` | Serve the built bundle from `dist/` |
| `pnpm type-check` | `tsc --noEmit` standalone |
| `pnpm lint` | ESLint over `src/**/*.{ts,tsx}` |
| `pnpm test` | Placeholder; tests land in a later sprint |

## Production

`pnpm build` produces a static bundle in `dist/`. In production it is
served by nginx (see `docs/backend_design.md` §17). nginx reverse-proxies
`/api/*` to the BFF, so the relative `apiFetch` paths used here work in
both dev and prod with no code changes.

## Sprint 1 scope

- Global shell (status bar, diff rail placeholder, sidebar).
- `Trade Ideas` page wired to `GET /api/v1/trade-ideas`.
- All other sidebar pages render placeholder copy.
- The status bar pulls "Last refresh" from the trade-ideas envelope and
  mocks the rest until `/api/v1/system/overview` lands.

## Conventions

- No component libraries. No Tailwind. Plain CSS with the tokens in
  `src/styles/tokens.css`.
- Server state is owned by TanStack Query (default `staleTime` 30s).
  Local UI state uses `useState` / `useReducer`.
- All API code uses the `ApiEnvelope<T>` wrapper and surfaces
  `envelope.errors[].message` in the UI on failure.
- Numeric cells use tabular numerals and right-align.
