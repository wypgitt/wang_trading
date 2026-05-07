"""Concrete infrastructure probes used by live preflight.

The preflight checker still accepts static ``infra`` values for tests and
offline rehearsals, but production bootstrap injects this probe so DB,
MLflow, Prometheus, feature-store freshness, and alert-channel checks reflect
the actual environment.
"""

from __future__ import annotations

import asyncio
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class InfrastructureProbe:
    db_url: str | None = None
    mlflow_tracking_uri: str | None = None
    prometheus_url: str | None = None
    grafana_url: str | None = None
    feature_store_path: str | None = None
    db_disk_path: str | None = None
    alert_manager: Any | None = None
    alert_ping_enabled: bool = True
    timeout_s: float = 3.0

    async def collect(self) -> dict[str, Any]:
        checks = await asyncio.gather(
            self._probe_db(),
            self._probe_mlflow(),
            self._probe_http("prometheus", self.prometheus_url),
            self._probe_http("grafana", self.grafana_url),
            self._probe_feature_freshness(),
            self._probe_alerts(),
            return_exceptions=True,
        )
        out: dict[str, Any] = {}
        for result in checks:
            if isinstance(result, dict):
                out.update(result)
        return out

    async def _probe_db(self) -> dict[str, Any]:
        if not self.db_url:
            return {"db_reachable": False, "db_error": "db_url not configured"}
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._probe_db_sync),
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            return {"db_reachable": False, "db_error": str(exc)}

    def _probe_db_sync(self) -> dict[str, Any]:
        from sqlalchemy import create_engine, text

        engine = create_engine(self.db_url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                freshness_h = self._feature_freshness_from_db(conn)
        finally:
            engine.dispose()
        out: dict[str, Any] = {"db_reachable": True}
        if freshness_h is not None:
            out["feature_freshness_h"] = freshness_h
        disk_path = self.db_disk_path or self.feature_store_path or "."
        out["db_disk_pct"] = _disk_used_pct(disk_path)
        return out

    @staticmethod
    def _feature_freshness_from_db(conn: Any) -> float | None:
        from sqlalchemy import text

        try:
            row = conn.execute(
                text("SELECT MAX(timestamp) FROM features")
            ).first()
        except Exception:
            return None
        ts = row[0] if row else None
        if ts is None:
            return None
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                return None
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0

    async def _probe_mlflow(self) -> dict[str, Any]:
        uri = self.mlflow_tracking_uri
        if not uri:
            return {"mlflow_up": False, "mlflow_error": "tracking URI not configured"}
        parsed = urlparse(uri)
        if parsed.scheme in {"http", "https"}:
            result = await self._probe_http("mlflow", uri)
            return {
                "mlflow_up": bool(result.get("mlflow_up")),
                "mlflow_error": result.get("mlflow_error", ""),
            }
        if parsed.scheme == "sqlite":
            db_path = Path(parsed.path)
            parent = db_path.parent if str(db_path.parent) else Path(".")
            return {
                "mlflow_up": parent.exists(),
                "mlflow_error": "" if parent.exists() else f"missing parent: {parent}",
            }
        if parsed.scheme in {"", "file"}:
            path = Path(parsed.path or uri)
            exists = path.exists() or path.parent.exists()
            return {
                "mlflow_up": exists,
                "mlflow_error": "" if exists else f"missing path: {path}",
            }
        return {"mlflow_up": True}

    async def _probe_http(self, name: str, url: str | None) -> dict[str, Any]:
        key = f"{name}_up"
        err_key = f"{name}_error"
        if not url:
            return {key: False, err_key: "url not configured"}
        try:
            ok = await asyncio.wait_for(
                asyncio.to_thread(_http_ok, url, self.timeout_s),
                timeout=self.timeout_s + 1.0,
            )
            return {key: ok, err_key: "" if ok else "non-2xx response"}
        except Exception as exc:  # noqa: BLE001
            return {key: False, err_key: str(exc)}

    async def _probe_feature_freshness(self) -> dict[str, Any]:
        if not self.feature_store_path:
            return {}
        age = await asyncio.to_thread(_feature_store_freshness_h, self.feature_store_path)
        return {} if age is None else {"feature_freshness_h": age}

    async def _probe_alerts(self) -> dict[str, Any]:
        if not self.alert_ping_enabled:
            return {"alerts_ok": True, "alerts_error": "alert ping disabled"}
        if self.alert_manager is None:
            return {"alerts_ok": False, "alerts_error": "alert_manager not configured"}
        try:
            from src.monitoring.alerting import Alert, AlertSeverity

            result = await self.alert_manager.send_alert(
                Alert(
                    severity=AlertSeverity.INFO,
                    title="Preflight Alert Ping",
                    message="Alert channel preflight probe",
                    source="preflight",
                ),
                cooldown_seconds=0,
            )
            return {"alerts_ok": bool(result) and all(bool(x) for x in result)}
        except Exception as exc:  # noqa: BLE001
            return {"alerts_ok": False, "alerts_error": str(exc)}


def _http_ok(url: str, timeout_s: float) -> bool:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 300
    except urllib.error.HTTPError as exc:
        return 200 <= int(exc.code) < 300


def _disk_used_pct(path: str | Path) -> float:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return float(usage.used / usage.total * 100.0)


def _feature_store_freshness_h(path: str | Path) -> float | None:
    root = Path(path)
    if not root.exists():
        return None
    files = [p for p in root.rglob("*.parquet") if p.is_file()]
    if not files:
        return None
    latest = max(p.stat().st_mtime for p in files)
    return (datetime.now(timezone.utc).timestamp() - latest) / 3600.0
