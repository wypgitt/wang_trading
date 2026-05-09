"""Runtime YAML validation for production entry points.

The pydantic ``Settings`` model covers global ``settings.yaml``. These
validators cover the smaller service-specific runtime files used by live
trading, retraining, and monitoring. They intentionally validate structure
before ``env:KEY`` expansion so example files can be checked without secrets
present in the shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RuntimeConfigError(ValueError):
    """Raised when a service runtime YAML is malformed."""


@dataclass(frozen=True)
class _RuleSet:
    required: tuple[str, ...]
    allowed_top_level: set[str]


_ASSET_CLASSES = {"equities", "crypto", "futures"}

_RULES: dict[str, _RuleSet] = {
    "live_trading": _RuleSet(
        required=("asset_class", "symbols", "pipeline"),
        allowed_top_level={
            "asset_class",
            "symbols",
            "deployment",
            "operator",
            "operator_checkin_path",
            "operator_checkin_max_age_h",
            "halt_file",
            "crash_file",
            "compliance_log_path",
            "alpaca",
            "broker",
            "binance",
            "coinbase",
            "kraken",
            "bybit",
            "ibkr",
            "storage",
            "feature_store",
            "mlflow",
            "model_registry",
            "circuit_breakers",
            "risk",
            "costs",
            "bars",
            "bar_type",
            "features",
            "signals",
            "pipeline",
            "preflight",
            "paper_stats",
            "infra",
            "monitoring",
            "snapshots",
            "initial_cash",
            "allow_confidence_meta_fallback",
            "dry_run",
            "paper_prices",
            "paper_rehearsal",
        },
    ),
    "retrain": _RuleSet(
        required=("asset_class", "symbols", "bar_type", "model"),
        allowed_top_level={
            "asset_class",
            "symbols",
            "bar_type",
            "limit",
            "storage",
            "feature_store",
            "mlflow",
            "model_registry",
            "model",
            "meta_labeling",
            "features",
            "signals",
            "costs",
            "monitoring",
            "retrain_interval_days",
            "min_new_bars",
            "scheduler_interval_s",
            "min_improvement_pct",
        },
    ),
    "monitoring": _RuleSet(
        required=("monitoring",),
        allowed_top_level={"monitoring"},
    ),
}


def validate_runtime_config(
    data: dict[str, Any],
    kind: str,
    *,
    source: str | None = None,
) -> None:
    """Validate a runtime config mapping.

    Parameters
    ----------
    data:
        Raw YAML mapping before env expansion.
    kind:
        One of ``live_trading``, ``retrain``, or ``monitoring``. Unknown kinds
        are ignored so ad-hoc research configs remain flexible.
    source:
        Optional path included in the error message.
    """

    rules = _RULES.get(kind)
    if rules is None:
        return
    errors: list[str] = []

    for key in rules.required:
        if key not in data:
            errors.append(f"missing required key: {key}")

    unknown = sorted(set(data) - rules.allowed_top_level)
    for key in unknown:
        errors.append(f"unknown top-level key: {key}")

    _validate_asset_class(data, errors)
    _validate_symbols(data, errors)

    if kind == "live_trading":
        _validate_mapping(data, "pipeline", errors)
        _validate_mapping(data, "preflight", errors, required=False)
        _validate_storage(data, errors, required=False)
        _validate_mlflow(data, errors, required=False)
        _validate_circuit_breakers(data, errors)
        _validate_live_broker(data, errors)
        rehearsal = _validate_mapping(data, "paper_rehearsal", errors, required=False)
        if isinstance(rehearsal, dict):
            if "enabled" in rehearsal and not isinstance(rehearsal["enabled"], bool):
                errors.append("paper_rehearsal.enabled must be a boolean")
            if "record_path" in rehearsal and not isinstance(rehearsal["record_path"], str):
                errors.append("paper_rehearsal.record_path must be a string")
    elif kind == "retrain":
        _validate_scalar(data, "bar_type", errors, allowed={
            "tick", "volume", "dollar", "tib", "vib", "time",
        })
        _validate_positive_int(data, "limit", errors, required=False)
        _validate_positive_int(data, "retrain_interval_days", errors, required=False)
        _validate_positive_int(data, "min_new_bars", errors, required=False)
        _validate_positive_int(data, "scheduler_interval_s", errors, required=False)
        _validate_model(data, errors)
        _validate_storage(data, errors, required=False)
        _validate_mlflow(data, errors, required=False)
    elif kind == "monitoring":
        monitoring = _validate_mapping(data, "monitoring", errors)
        if isinstance(monitoring, dict):
            _validate_positive_int(monitoring, "metrics_port", errors, prefix="monitoring")

    if errors:
        location = f" in {source}" if source else ""
        bullet_list = "\n  - ".join(errors)
        raise RuntimeConfigError(
            f"Invalid {kind} runtime config{location}:\n  - {bullet_list}"
        )


def _validate_asset_class(data: dict[str, Any], errors: list[str]) -> None:
    if "asset_class" not in data:
        return
    value = data.get("asset_class")
    if value not in _ASSET_CLASSES:
        errors.append(
            "asset_class must be one of "
            f"{sorted(_ASSET_CLASSES)} (got {value!r})"
        )


def _validate_symbols(data: dict[str, Any], errors: list[str]) -> None:
    if "symbols" not in data:
        return
    symbols = data.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        errors.append("symbols must be a non-empty list")
        return
    bad = [s for s in symbols if not isinstance(s, str) or not s.strip()]
    if bad:
        errors.append("symbols must contain only non-empty strings")


def _validate_mapping(
    data: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    required: bool = True,
) -> Any:
    if key not in data:
        if required:
            errors.append(f"{key} must be a mapping")
        return None
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be a mapping")
        return None
    return value


def _validate_scalar(
    data: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    allowed: set[str] | None = None,
) -> None:
    if key not in data:
        return
    value = data.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{key} must be a non-empty string")
        return
    if allowed is not None and value not in allowed:
        errors.append(f"{key} must be one of {sorted(allowed)} (got {value!r})")


def _validate_positive_int(
    data: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    required: bool = True,
    prefix: str | None = None,
) -> None:
    label = f"{prefix}.{key}" if prefix else key
    if key not in data:
        if required:
            errors.append(f"{label} is required")
        return
    value = data.get(key)
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{label} must be a positive integer")


def _validate_storage(
    data: dict[str, Any],
    errors: list[str],
    *,
    required: bool,
) -> None:
    storage = _validate_mapping(data, "storage", errors, required=required)
    if isinstance(storage, dict) and "database_url" in storage:
        value = storage["database_url"]
        if not isinstance(value, str):
            errors.append("storage.database_url must be a string")


def _validate_mlflow(
    data: dict[str, Any],
    errors: list[str],
    *,
    required: bool,
) -> None:
    mlflow = data.get("mlflow") or data.get("model_registry")
    if mlflow is None:
        if required:
            errors.append("mlflow must be a mapping")
        return
    if not isinstance(mlflow, dict):
        errors.append("mlflow/model_registry must be a mapping")
        return
    for key in ("tracking_uri", "experiment_name"):
        if key in mlflow and not isinstance(mlflow[key], str):
            errors.append(f"mlflow.{key} must be a string")


def _validate_circuit_breakers(data: dict[str, Any], errors: list[str]) -> None:
    breakers = data.get("circuit_breakers")
    if breakers is None:
        return
    if not isinstance(breakers, dict):
        errors.append("circuit_breakers must be a mapping")
        return
    for key in (
        "max_order_pct",
        "daily_loss_limit_pct",
        "max_single_position_pct",
        "max_gross_exposure",
    ):
        if key in breakers and not isinstance(breakers[key], (int, float)):
            errors.append(f"circuit_breakers.{key} must be numeric")


def _validate_live_broker(data: dict[str, Any], errors: list[str]) -> None:
    asset = data.get("asset_class")
    broker_key = {
        "equities": "alpaca",
        "crypto": "binance",
        "futures": "ibkr",
    }.get(asset)
    if broker_key is None or broker_key not in data:
        return
    broker = data.get(broker_key)
    if not isinstance(broker, dict):
        errors.append(f"{broker_key} must be a mapping")


def _validate_model(data: dict[str, Any], errors: list[str]) -> None:
    model = _validate_mapping(data, "model", errors)
    if isinstance(model, dict):
        if "type" not in model:
            errors.append("model.type is required")
        _validate_scalar(model, "type", errors, allowed={
            "lightgbm", "xgboost", "random_forest",
        })
