"""Compatibility shim — the canonical abstract broker interface now lives in
`src.execution.broker_adapter`. Existing imports continue to work."""

from src.execution.broker_adapter import BaseBrokerAdapter

__all__ = ["BaseBrokerAdapter"]
