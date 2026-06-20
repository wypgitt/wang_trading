"""Pydantic DTOs for the BFF.

These mirror docs/web_app_design_v2.md §27 and docs/api_contracts_v2.md.
Update both when shape changes; the design doc is the source of truth.

The DTOs are intentionally all in one file for the initial scaffold. Split
into a package when individual sections grow past ~150 lines.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from .envelope import RegimeSnapshot


class CamelModel(BaseModel):
    """Base DTO for the data payload — serialize camelCase, accept snake_case.

    The clients bind camelCase (``targetWeight``, ``metaProbability``); the
    :func:`envelope` helper dumps the ``data`` payload with ``by_alias=True``,
    so every response field lands camelCased in one place (the locked casing
    decision, docs/aperture_backend_design.md §10.4). ``populate_by_name`` lets
    routes/tests keep constructing with Python snake_case kwargs;
    ``protected_namespaces=()`` allows the ``model_*`` fields the engine emits.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        protected_namespaces=(),
    )


Action = Literal[
    "BUY", "SELL", "WATCH", "MODEL_REQUIRED", "NO_DATA", "ERROR"
]

Side = Literal[-1, 0, 1]


# ---------- Shared ----------

class ShapContributor(CamelModel):
    feature: str
    value: float
    contribution: float
    abs_contribution: float
    percentile: float


class SizingLayer(CamelModel):
    name: Literal["afml", "kelly", "vol", "atr", "final"]
    value: float
    capped: bool
    cap_reason: str | None = None


class SizingWaterfall(CamelModel):
    layers: list[SizingLayer]
    constraints_applied: list[str]
    side: Side
    final: float


class DecisionChainStep(CamelModel):
    name: Literal[
        "bars", "features", "signals", "regime", "model",
        "calibration", "sizing", "target", "cost", "risk", "execution",
    ]
    status: Literal["ok", "warning", "error", "skipped", "unknown"]
    value: str | float | None = None
    count: int | None = None
    timestamp: datetime | None = None
    latency_seconds: float | None = None
    message: str | None = None


# ---------- Trade Ideas ----------

class TradeIdea(CamelModel):
    symbol: str
    action: Action
    target_weight: float
    target_notional: float
    estimated_quantity: float | None = None
    latest_price: float | None = None
    latest_bar_at: datetime | None = None
    bar_type: str | None = None
    bars_loaded: int = 0
    feature_rows: int = 0
    signal_count: int = 0
    top_signal_family: str | None = None
    top_signal_side: Side | None = None
    top_signal_confidence: float | None = None
    avg_signal_confidence: float | None = None
    meta_probability: float | None = None
    calibrated_probability: float | None = None
    regime: RegimeSnapshot | None = None
    regime_fit_score: float | None = None
    bet_size: float | None = None
    sizing_constraints_applied: list[str] = Field(default_factory=list)
    strategy: str | None = None
    reason: str = ""
    expected_cost_bps: float | None = None
    top_shap_feature: ShapContributor | None = None
    track_record_win_rate: float | None = None
    track_record_n: int | None = None
    stage_latency_seconds: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class TradeIdeasTotals(CamelModel):
    buy: int = 0
    sell: int = 0
    watch: int = 0
    model_required: int = 0
    no_data: int = 0
    error: int = 0
    gross_target_weight: float = 0.0
    net_target_weight: float = 0.0


class TradeIdeasResponse(CamelModel):
    idea_count: int
    totals: TradeIdeasTotals
    ideas: list[TradeIdea]


# ---------- Per-family signal metadata (typed union) ----------

class TsMomentumMetadata(CamelModel):
    family: Literal["ts_momentum"] = "ts_momentum"
    lookbacks: list[int]
    weights: list[float]
    z_scores: dict[str, float]
    aggregate: float


class CsMomentumMetadata(CamelModel):
    family: Literal["cs_momentum"] = "cs_momentum"
    decile_rank: float
    lookback_return: float
    z_score: float
    skip_periods: int


class MeanReversionMetadata(CamelModel):
    family: Literal["mean_reversion"] = "mean_reversion"
    half_life: float | None = None
    adf_pvalue: float | None = None
    z_score: float | None = None
    entry_threshold: float | None = None
    exit_threshold: float | None = None


class StatArbMetadata(CamelModel):
    family: Literal["stat_arb"] = "stat_arb"
    pair_symbol: str
    cointegration_pvalue: float | None = None
    hedge_ratio: float | None = None
    spread_z_score: float | None = None
    spread_halflife: float | None = None


class FuturesCarryMetadata(CamelModel):
    family: Literal["futures_carry"] = "futures_carry"
    front_price: float
    back_price: float
    days_to_expiry: int
    carry: float


class VrpMetadata(CamelModel):
    family: Literal["vrp"] = "vrp"
    iv: float
    rv: float
    vrp: float
    vrp_percentile_rank: float
    regime_modifier: dict[str, float] = Field(default_factory=dict)


FamilyMetadata = (
    TsMomentumMetadata
    | CsMomentumMetadata
    | MeanReversionMetadata
    | StatArbMetadata
    | FuturesCarryMetadata
    | VrpMetadata
)


# ---------- Trade Idea Detail ----------

class SignalRow(CamelModel):
    timestamp: datetime
    family: str
    side: Side
    confidence: float


class ModelInferenceDetail(CamelModel):
    source: str
    version: str | None = None
    run_id: str | None = None
    alias: str | None = None
    trained_at: datetime | None = None
    n_training_events: int | None = None
    calibration: str | None = None
    calibration_age_days: float | None = None
    feature_hash: str | None = None


class BarLatest(CamelModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    tick_count: int | None = None
    buy_volume: float | None = None
    sell_volume: float | None = None
    imbalance: float | None = None
    threshold: float | None = None
    bar_duration_seconds: float | None = None


class BarSummary(CamelModel):
    bar_type: str | None = None
    n: int
    latest: BarLatest | None = None


class MicrostructureSnapshot(CamelModel):
    kyle_lambda: float | None = None
    amihud_lambda: float | None = None
    roll_spread: float | None = None
    vpin: float | None = None
    hasbrouck_lambda: float | None = None
    order_flow_imbalance: float | None = None
    trade_intensity: float | None = None


class StructuralBreakSnapshot(CamelModel):
    cusum: float | None = None
    sadf: float | None = None
    gsadf: float | None = None
    chow_p_value: float | None = None
    regime_break_detected: bool | None = None


class OnchainSnapshot(CamelModel):
    exchange_inflow: float | None = None
    exchange_outflow: float | None = None
    net_flow: float | None = None
    whale_transactions: int | None = None
    active_addresses: int | None = None
    stablecoin_supply_change_24h: float | None = None


class SentimentSnapshot(CamelModel):
    score: float
    momentum_24h: float | None = None
    momentum_7d: float | None = None
    article_count_24h: int | None = None
    article_count_7d: int | None = None


class CostForecast(CamelModel):
    expected_total_bps: float
    expected_slippage_bps: float | None = None
    expected_market_impact_bps: float | None = None
    expected_commission_bps: float | None = None
    algo: str | None = None
    window_days: int | None = None
    n_observations: int | None = None
    vs_twap_bps: float | None = None
    vs_vwap_bps: float | None = None


class TrackRecordBucket(CamelModel):
    n: int
    win_rate: float
    avg_return: float
    median_holding_bars: float | None = None


class TrackRecordSummary(CamelModel):
    symbol: str
    family: str | None = None
    trailing_90d: TrackRecordBucket | None = None
    trailing_180d: TrackRecordBucket | None = None
    all_time: TrackRecordBucket | None = None


class FeatureSnapshot(CamelModel):
    rows: int
    latest: dict[str, float] = Field(default_factory=dict)


class AlertSummary(CamelModel):
    alert_id: str
    severity: Literal["info", "warning", "critical"]
    timestamp: datetime
    title: str
    source: str
    message: str
    acknowledged: bool = False


class AuditSummary(CamelModel):
    entry_id: str
    event_type: str
    timestamp: datetime
    symbol: str | None = None


class TradeIdeaDetail(CamelModel):
    idea: TradeIdea
    chain: list[DecisionChainStep] = Field(default_factory=list)
    signals: list[SignalRow] = Field(default_factory=list)
    signal_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    model: ModelInferenceDetail | None = None
    shap: list[ShapContributor] = Field(default_factory=list)
    sizing: SizingWaterfall | None = None
    features: FeatureSnapshot | None = None
    microstructure: MicrostructureSnapshot | None = None
    structural_breaks: StructuralBreakSnapshot | None = None
    onchain: OnchainSnapshot | None = None
    sentiment: SentimentSnapshot | None = None
    bars: BarSummary | None = None
    cost_forecast: CostForecast | None = None
    track_record: TrackRecordSummary | None = None
    nl_explanation: str | None = None
    related_alerts: list[AlertSummary] = Field(default_factory=list)
    related_audit_entries: list[AuditSummary] = Field(default_factory=list)


# ---------- Portfolio simulator ----------

class SimulateRequest(CamelModel):
    apply: Literal["current_portfolio", "current_targets", "both"] = "current_targets"
    filter: dict[str, Any] | None = None


class SimulateResponse(CamelModel):
    projected: dict[str, Any]
    deltas: dict[str, float]
    breaker_headroom_after: dict[str, float]
    expected_cost_usd: float
    expected_cost_bps: float
    estimated_duration_seconds: float
    affected_ideas: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


# ---------- Scenarios ----------

class ScenarioShocks(CamelModel):
    symbol_pct: dict[str, float] | None = None
    vol_multiplier: float | None = None
    correlation_target: float | None = None
    factor_shift: dict[str, float] | None = None
    liquidity_multiplier: float | None = None


class ScenarioRequest(CamelModel):
    shocks: ScenarioShocks
    apply_to: Literal["current_portfolio", "current_targets", "both"] = "current_targets"


class AffectedIdea(CamelModel):
    symbol: str
    old_target: float
    new_target: float
    flipped: bool


class SuggestedHedge(CamelModel):
    long: str
    short: str
    ratio: float


class ScenarioResult(CamelModel):
    pnl_impact_usd: float
    drawdown_impact: float
    factor_exposure_deltas: dict[str, float] = Field(default_factory=dict)
    breaker_headroom: dict[str, float] = Field(default_factory=dict)
    affected_ideas: list[AffectedIdea] = Field(default_factory=list)
    suggested_hedges: list[SuggestedHedge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------- Replay ----------

class ReplaySnapshot(CamelModel):
    ts: datetime
    model_version: str | None = None
    regime: RegimeSnapshot | None = None
    ideas: list[TradeIdea]
    audit_chain_verified_to: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------- Preflight ----------

class PreflightCheck(CamelModel):
    name: str
    status: Literal["PASS", "FAIL", "SKIPPED", "UNKNOWN"]
    evaluated_at: datetime | None = None
    reason: str | None = None
    runbook: str | None = None


class PreflightStatus(CamelModel):
    overall: Literal["READY", "BLOCKED", "UNKNOWN"]
    checks: list[PreflightCheck]
    operator_sentinel: dict[str, Any] | None = None
    capital_deployment: dict[str, Any] | None = None
    infra_probes: dict[str, str] = Field(default_factory=dict)
    reconciliation: dict[str, Any] | None = None


# ---------- Calibration ----------

class CalibrationBucket(CamelModel):
    predicted_bucket_lo: float
    predicted_bucket_hi: float
    predicted_mean: float
    observed_rate: float
    count: int


class CalibrationReport(CamelModel):
    window: str
    buckets: list[CalibrationBucket]
    brier_score: float | None = None
    ece: float | None = None
    as_of: datetime


# ---------- RL Shadow ----------

class RlShadowMetrics(CamelModel):
    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int


class RlShadowPromotion(CamelModel):
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    months_span: float | None = None


class RlShadowDecision(CamelModel):
    ts: datetime
    hrp_target: dict[str, float]
    rl_target: dict[str, float]
    executed_target: dict[str, float]


class RlShadowReport(CamelModel):
    window_days: int
    n_observations: int
    hrp_metrics: RlShadowMetrics
    rl_metrics: RlShadowMetrics
    paired_t_stat: float | None = None
    p_value: float | None = None
    rl_is_better: bool | None = None
    significance: Literal["significant", "trending", "insignificant"] = "insignificant"
    promotion_eligibility: RlShadowPromotion
    recent_decisions: list[RlShadowDecision] = Field(default_factory=list)


# ---------- Freshness Heatmap ----------

class FreshnessThresholds(CamelModel):
    ok: float
    warn: float


class FreshnessHeatmap(CamelModel):
    as_of: datetime
    rows: list[str]
    columns: list[str]
    values: dict[str, dict[str, float | None]]
    thresholds: dict[str, FreshnessThresholds]


# ======================================================================
# v1 RESPONSE DTOs — the 8-endpoint surface (aperture_backend_design §2)
# ======================================================================
#
# Every field below is either (a) PRODUCED by the engine today (bars
# hypertable / trade-ideas snapshot / MLflow registry), or (b) held
# null/empty because its producer does not exist yet — never synthesised.
# The COMING fields carry their unlock gate in a trailing comment.


class PricePoint(CamelModel):
    """A single (index, value) point — mirrors the client ``Point``."""

    t: float
    v: float


class Candle(CamelModel):
    """OHLCV candle from the bars hypertable — mirrors the client ``Candle``."""

    t: float
    o: float
    h: float
    l: float
    c: float
    v: float


# ---------- /overview ----------

class ActionCounts(CamelModel):
    buy: int = 0
    sell: int = 0
    watch: int = 0
    model_required: int = 0
    no_data: int = 0


class EnginePulseStage(CamelModel):
    stage: str
    seconds: float


class EnginePulse(CamelModel):
    """Per-stage wall-clock summed across the snapshot's ideas.

    ``stage_latency_seconds`` is the one honest system-health signal today
    (data_readiness: PRODUCED). Ordered by the engine's pipeline stages.
    """

    stages: list[EnginePulseStage] = Field(default_factory=list)
    total_seconds: float = 0.0


class OverviewResponse(CamelModel):
    action_counts: ActionCounts
    top_actionable: list[TradeIdea] = Field(default_factory=list)
    engine_pulse: EnginePulse
    nav: float | None = None          # COMING: no persisted portfolio (Wave 5)
    nav_history: list[PricePoint] | None = None  # COMING: no persisted portfolio (Wave 5)
    regime: RegimeSnapshot | None = None  # COMING: RegimeDetector has no runtime caller


# ---------- /markets ----------

class BarMicro(CamelModel):
    """Latest-bar microstructure — persisted columns of the bars hypertable."""

    bar_type: str
    vwap: float | None = None
    dollar_volume: float | None = None
    tick_count: int | None = None
    buy_volume: float | None = None
    sell_volume: float | None = None
    volume_imbalance: float | None = None
    tick_imbalance_ratio: float | None = None
    imbalance: float | None = None
    threshold: float | None = None
    bar_duration_seconds: float | None = None


class MarketRow(CamelModel):
    """One instrument on the markets grid — bar-derived, honestly partial.

    ``data_available=False`` means the bars table was unreachable or held no
    rows for this symbol; the row still renders (name/type from the static
    instrument map) but price/spark/bar are null.
    """

    symbol: str
    name: str
    type: str                          # equity | index | crypto | future
    price: float | None = None         # latest close
    spark: list[float] = Field(default_factory=list)  # recent closes
    change_window_pct: float | None = None  # (last-first)/first over the spark window
    volume: float | None = None        # latest bar volume
    bars_loaded: int = 0
    latest_bar_at: datetime | None = None
    has_idea: bool = False             # joined from the trade-ideas snapshot
    action: Action | None = None       # from the snapshot, when has_idea
    target_weight: float | None = None
    bar: BarMicro | None = None
    data_available: bool = True
    market_cap: float | None = None    # COMING: no source in bars or the universe map


# ---------- /symbols/{symbol} ----------

class SymbolView(CamelModel):
    """Full single-instrument view: real OHLC series + latest microstructure."""

    symbol: str
    name: str
    type: str
    price: float | None = None
    spark: list[float] = Field(default_factory=list)
    change_window_pct: float | None = None
    volume: float | None = None
    bars_loaded: int = 0
    latest_bar_at: datetime | None = None
    bar_type: str | None = None
    candles: list[Candle] = Field(default_factory=list)   # real OHLCV from bars
    line: list[PricePoint] = Field(default_factory=list)  # close series
    bar: BarMicro | None = None
    data_available: bool = True
    market_cap: float | None = None    # COMING: no source


class SymbolDetailResponse(CamelModel):
    sym: SymbolView
    idea: TradeIdea | None = None      # joined from the snapshot, when present


# ---------- /signals/families and /signals/family-{id} ----------

class FamilyParam(CamelModel):
    key: str
    value: str


class SignalFamilyCard(CamelModel):
    """A signal family: static metadata (real) + live-snapshot activity (real).

    All *performance* fields (sharpe, win-rate, P&L, allocation, regime fit,
    equity curve) are COMING — they need backtest-run + track-record
    persistence the engine does not write yet — so they stay null/empty.
    """

    id: str
    name: str
    category: str
    source: str
    thesis: str
    status: Literal["live", "shadow", "paused"]
    asset_classes: list[str] = Field(default_factory=list)
    params: list[FamilyParam] = Field(default_factory=list)
    active_signals: int = 0            # live ideas from this family in the snapshot
    # COMING — no backtest/track-record persistence:
    sharpe: float | None = None
    win_rate: float | None = None
    trades: int | None = None
    contribution_pct: float | None = None
    pnl_ytd: float | None = None
    allocation: float | None = None
    avg_hold_bars: float | None = None
    regime_fit: dict[str, float] = Field(default_factory=dict)
    equity_curve: list[PricePoint] = Field(default_factory=list)


class SignalFamilyDetailResponse(CamelModel):
    strategy: SignalFamilyCard
    ideas: list[TradeIdea] = Field(default_factory=list)


# ---------- /model ----------

class ModelGates(CamelModel):
    """Promotion-gate verdicts from the production run (now real: the retrain
    gate calls ``StrategyGate.evaluate_candidate``). ``null`` = gate not run."""

    cpcv: bool | None = None
    dsr: bool | None = None
    pbo: bool | None = None


class MetaProbBucket(CamelModel):
    bucket: str
    count: int


class ModelFeatureImportance(CamelModel):
    feature: str
    importance: float
    family: str | None = None


class ModelDriftItem(CamelModel):
    feature: str
    kl: float
    severity: Literal["ok", "warn", "alert"]


class RetrainEvent(CamelModel):
    run_id: str
    trained_at: datetime | None = None
    cv_score: float | None = None
    promoted: bool | None = None


class ModelResponse(CamelModel):
    """Production meta-labeler card. Returned even with no model registered
    (all-null + a warning ⇒ the client's MODEL_REQUIRED state)."""

    version: str | None = None
    trained_at: datetime | None = None
    last_retrain_hours: float | None = None
    run_id: str | None = None
    type: str | None = None
    cv_score: float | None = None          # REAL: MLflow metric mean_cv_score
    train_acc: float | None = None         # REAL: MLflow metric train_accuracy
    training_events: int | None = None     # REAL: param n_training_events
    gates: ModelGates | None = None        # REAL: production-run gate verdicts
    retrain_timeline: list[RetrainEvent] = Field(default_factory=list)  # REAL: MLflow run history
    meta_prob_hist: list[MetaProbBucket] = Field(default_factory=list)  # Empty until db_manager wired into predict_proba
    # COMING / not-logged by the engine (stay null/empty, never synthesised):
    auc: float | None = None               # not logged
    brier: float | None = None             # not logged
    ece: float | None = None               # not logged
    calibration: list[CalibrationBucket] = Field(default_factory=list)        # no calibration persistence
    feature_importance: list[ModelFeatureImportance] = Field(default_factory=list)  # FeatureStore.save_features has no callers
    drift: list[ModelDriftItem] = Field(default_factory=list)                 # DriftDetector.set_baseline has no callers
    rl_shadow: dict[str, Any] | None = None  # RL shadow has no persistence
