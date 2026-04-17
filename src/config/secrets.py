"""Secrets management (P6.15).

Pluggable backends so dev/staging/prod can share the same config surface:

* :class:`EnvSecretsManager` — env vars, for dev.
* :class:`EncryptedFileSecretsManager` — Fernet-encrypted YAML, master key
  supplied via ``WANG_MASTER_KEY``.
* :class:`AWSSecretsManager` / :class:`GCPSecretManager` — cloud-backed with
  a 5-minute TTL cache so we don't hammer the provider.

``get_secrets_manager()`` picks a backend via the ``WANG_SECRETS_BACKEND``
env var; pass it through to every subsystem that needs a key.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────

class SecretsManager(ABC):
    """Read/write/rotate secrets. Backend-agnostic."""

    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str) -> None: ...

    @abstractmethod
    def rotate(self, key: str, new_value: str | None = None) -> str: ...

    @abstractmethod
    def list_keys(self) -> list[str]: ...


# ── Env-var backend ───────────────────────────────────────────────────────

class EnvSecretsManager(SecretsManager):
    """Reads/writes process environment variables. Dev only."""

    def __init__(self, *, prefix: str = "WANG_SECRET_") -> None:
        self.prefix = prefix

    def _env_key(self, key: str) -> str:
        return self.prefix + key.upper().replace(".", "_")

    def get(self, key: str) -> str | None:
        return os.environ.get(self._env_key(key))

    def set(self, key: str, value: str) -> None:
        os.environ[self._env_key(key)] = value

    def rotate(self, key: str, new_value: str | None = None) -> str:
        if new_value is None:
            new_value = os.urandom(24).hex()
        self.set(key, new_value)
        return new_value

    def list_keys(self) -> list[str]:
        return sorted(
            k[len(self.prefix):].lower()
            for k in os.environ
            if k.startswith(self.prefix)
        )


# ── Encrypted-file backend ────────────────────────────────────────────────

class EncryptedFileSecretsManager(SecretsManager):
    """Fernet-encrypted JSON blob on disk.

    The master key is loaded from ``master_key_env`` (default
    ``WANG_MASTER_KEY``). Keys are stored as plaintext key names → encrypted
    values. The full file is re-encrypted under a new master key by
    ``rotate_master_key``.
    """

    MASTER_KEY_ENV = "WANG_MASTER_KEY"

    def __init__(
        self,
        path: str | Path,
        *,
        master_key: str | None = None,
        master_key_env: str | None = None,
    ) -> None:
        try:
            from cryptography.fernet import Fernet  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "cryptography is required for EncryptedFileSecretsManager"
            ) from exc
        self._Fernet = Fernet
        self.path = Path(path)
        key = master_key or os.environ.get(master_key_env or self.MASTER_KEY_ENV)
        if not key:
            raise ValueError(
                f"master key missing: set {master_key_env or self.MASTER_KEY_ENV}"
            )
        self.master_key = key
        self._fernet = Fernet(key.encode("utf-8"))

    # ── internal helpers ──────────────────────────────────────────────

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        raw = self.path.read_bytes()
        if not raw:
            return {}
        try:
            decrypted = self._fernet.decrypt(raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"failed to decrypt {self.path}: wrong master key or tampering"
            ) from exc
        return json.loads(decrypted.decode("utf-8"))

    def _write(self, data: dict[str, str]) -> None:
        payload = json.dumps(data, sort_keys=True).encode("utf-8")
        encrypted = self._fernet.encrypt(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_bytes(encrypted)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover - permissions
            pass

    # ── API ───────────────────────────────────────────────────────────

    def get(self, key: str) -> str | None:
        return self._read().get(key)

    def set(self, key: str, value: str) -> None:
        data = self._read()
        data[key] = value
        self._write(data)

    def rotate(self, key: str, new_value: str | None = None) -> str:
        if new_value is None:
            new_value = os.urandom(24).hex()
        self.set(key, new_value)
        return new_value

    def list_keys(self) -> list[str]:
        return sorted(self._read().keys())

    def rotate_master_key(self, new_master_key: str) -> None:
        """Re-encrypt the whole blob under a new master key."""
        data = self._read()
        self.master_key = new_master_key
        self._fernet = self._Fernet(new_master_key.encode("utf-8"))
        self._write(data)

    @classmethod
    def generate_master_key(cls) -> str:
        from cryptography.fernet import Fernet  # type: ignore
        return Fernet.generate_key().decode("utf-8")


# ── Cloud backends with TTL cache ─────────────────────────────────────────

class _CachingSecretsManager(SecretsManager):
    """Base class that adds a per-key TTL cache."""

    def __init__(self, *, cache_ttl_s: float = 300.0) -> None:
        self._cache: dict[str, tuple[float, str | None]] = {}
        self._ttl = float(cache_ttl_s)

    def _cached(self, key: str) -> tuple[bool, str | None]:
        hit = self._cache.get(key)
        if hit is None:
            return False, None
        ts, value = hit
        if (time.time() - ts) > self._ttl:
            self._cache.pop(key, None)
            return False, None
        return True, value

    def _put(self, key: str, value: str | None) -> None:
        self._cache[key] = (time.time(), value)

    def _invalidate(self, key: str) -> None:
        self._cache.pop(key, None)


class AWSSecretsManager(_CachingSecretsManager):
    """AWS Secrets Manager backend via boto3."""

    def __init__(
        self,
        *,
        region_name: str = "us-east-1",
        client: Any | None = None,
        cache_ttl_s: float = 300.0,
    ) -> None:
        super().__init__(cache_ttl_s=cache_ttl_s)
        if client is not None:
            self._client = client
        else:
            try:
                import boto3  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("boto3 is required for AWSSecretsManager") from exc
            self._client = boto3.client("secretsmanager", region_name=region_name)

    def get(self, key: str) -> str | None:
        ok, cached = self._cached(key)
        if ok:
            return cached
        try:
            resp = self._client.get_secret_value(SecretId=key)
        except Exception as exc:
            log.warning("AWS get_secret_value failed for %s: %s", key, exc)
            self._put(key, None)
            return None
        value = resp.get("SecretString") or (
            resp.get("SecretBinary") or b""
        ).decode("utf-8", errors="replace") or None
        self._put(key, value)
        return value

    def set(self, key: str, value: str) -> None:
        try:
            self._client.create_secret(Name=key, SecretString=value)
        except Exception:
            self._client.put_secret_value(SecretId=key, SecretString=value)
        self._invalidate(key)

    def rotate(self, key: str, new_value: str | None = None) -> str:
        if new_value is None:
            new_value = os.urandom(24).hex()
        self._client.put_secret_value(SecretId=key, SecretString=new_value)
        self._invalidate(key)
        return new_value

    def list_keys(self) -> list[str]:
        try:
            resp = self._client.list_secrets()
        except Exception as exc:
            log.warning("AWS list_secrets failed: %s", exc)
            return []
        return sorted(s.get("Name", "") for s in resp.get("SecretList", []))


class GCPSecretManager(_CachingSecretsManager):
    """Google Secret Manager backend."""

    def __init__(
        self,
        *,
        project_id: str,
        client: Any | None = None,
        cache_ttl_s: float = 300.0,
    ) -> None:
        super().__init__(cache_ttl_s=cache_ttl_s)
        self.project_id = project_id
        if client is not None:
            self._client = client
        else:
            try:
                from google.cloud import secretmanager  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "google-cloud-secret-manager is required for GCPSecretManager"
                ) from exc
            self._client = secretmanager.SecretManagerServiceClient()

    def _name(self, key: str, version: str = "latest") -> str:
        return f"projects/{self.project_id}/secrets/{key}/versions/{version}"

    def get(self, key: str) -> str | None:
        ok, cached = self._cached(key)
        if ok:
            return cached
        try:
            resp = self._client.access_secret_version(name=self._name(key))
            value = resp.payload.data.decode("utf-8")
        except Exception as exc:
            log.warning("GCP access_secret_version failed for %s: %s", key, exc)
            value = None
        self._put(key, value)
        return value

    def set(self, key: str, value: str) -> None:
        parent = f"projects/{self.project_id}"
        try:
            self._client.create_secret(
                parent=parent, secret_id=key,
                secret={"replication": {"automatic": {}}},
            )
        except Exception:
            pass
        self._client.add_secret_version(
            parent=f"{parent}/secrets/{key}",
            payload={"data": value.encode("utf-8")},
        )
        self._invalidate(key)

    def rotate(self, key: str, new_value: str | None = None) -> str:
        if new_value is None:
            new_value = os.urandom(24).hex()
        self.set(key, new_value)
        return new_value

    def list_keys(self) -> list[str]:
        try:
            resp = self._client.list_secrets(
                parent=f"projects/{self.project_id}",
            )
        except Exception as exc:
            log.warning("GCP list_secrets failed: %s", exc)
            return []
        return sorted(s.name.split("/")[-1] for s in resp)


# ── Factory ───────────────────────────────────────────────────────────────

def get_secrets_manager(
    backend: str | None = None,
    **kwargs: Any,
) -> SecretsManager:
    """Pick a backend from ``WANG_SECRETS_BACKEND`` (or the ``backend`` arg)."""
    backend = (backend or os.environ.get("WANG_SECRETS_BACKEND", "env")).lower()
    if backend == "env":
        return EnvSecretsManager(**kwargs)
    if backend == "file":
        path = kwargs.pop("path", None) or os.environ.get(
            "WANG_SECRETS_PATH", "config/secrets.enc",
        )
        return EncryptedFileSecretsManager(path, **kwargs)
    if backend == "aws":
        return AWSSecretsManager(**kwargs)
    if backend == "gcp":
        return GCPSecretManager(**kwargs)
    raise ValueError(f"unknown secrets backend: {backend!r}")
