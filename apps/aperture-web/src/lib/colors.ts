// Stable import path for chart/SVG hex colors. The values are generated from
// tokens.json (the single source of truth) — see scripts/gen-tokens.mjs.
// CSS vars don't resolve inside SVG presentation attributes, so charts import
// these hex mirrors instead.
export * from './tokens.generated';
