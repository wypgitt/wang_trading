"""Tests for the secrets management layer (P6.15)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.config.secrets import (
    AWSSecretsManager,
    EncryptedFileSecretsManager,
    EnvSecretsManager,
    GCPSecretManager,
    get_secrets_manager,
)


# ── Env backend ────────────────────────────────────────────────────────

class TestEnvSecretsManager:
    def test_get_and_set_roundtrip(self, monkeypatch):
        monkeypatch.delenv("WANG_SECRET_ALPACA_API_KEY", raising=False)
        mgr = EnvSecretsManager()
        assert mgr.get("alpaca.api_key") is None
        mgr.set("alpaca.api_key", "secret-value")
        assert mgr.get("alpaca.api_key") == "secret-value"

    def test_list_keys(self, monkeypatch):
        monkeypatch.setenv("WANG_SECRET_FOO", "x")
        monkeypatch.setenv("WANG_SECRET_BAR", "y")
        mgr = EnvSecretsManager()
        assert "foo" in mgr.list_keys()
        assert "bar" in mgr.list_keys()

    def test_rotate_generates_value(self):
        mgr = EnvSecretsManager()
        new = mgr.rotate("rotating_key")
        assert mgr.get("rotating_key") == new
        assert len(new) >= 24

    def test_prefix_isolates_keys(self, monkeypatch):
        monkeypatch.setenv("OTHER_PREFIX_X", "y")
        mgr = EnvSecretsManager(prefix="WANG_SECRET_")
        assert "other_prefix_x" not in mgr.list_keys()


# ── Encrypted file backend ─────────────────────────────────────────────

class TestEncryptedFileSecretsManager:
    def _make(self, tmp_path):
        key = EncryptedFileSecretsManager.generate_master_key()
        return EncryptedFileSecretsManager(
            tmp_path / "secrets.enc", master_key=key,
        ), key

    def test_roundtrip(self, tmp_path):
        mgr, _ = self._make(tmp_path)
        mgr.set("alpaca.api_key", "AKIA-test-value")
        mgr.set("binance.api_key", "binance-test")
        assert mgr.get("alpaca.api_key") == "AKIA-test-value"
        assert sorted(mgr.list_keys()) == ["alpaca.api_key", "binance.api_key"]

    def test_persisted_file_is_encrypted(self, tmp_path):
        mgr, _ = self._make(tmp_path)
        mgr.set("alpaca.api_key", "VERY_SECRET_VALUE")
        raw = (tmp_path / "secrets.enc").read_bytes()
        assert b"VERY_SECRET_VALUE" not in raw  # encrypted at rest

    def test_rotate_updates_value(self, tmp_path):
        mgr, _ = self._make(tmp_path)
        mgr.set("api_key", "v1")
        new = mgr.rotate("api_key")
        assert mgr.get("api_key") == new
        assert new != "v1"

    def test_tamper_detection_wrong_key_fails(self, tmp_path):
        mgr, _ = self._make(tmp_path)
        mgr.set("api_key", "v1")

        # Same file, different master key → decrypt fails.
        other_key = EncryptedFileSecretsManager.generate_master_key()
        other = EncryptedFileSecretsManager(
            tmp_path / "secrets.enc", master_key=other_key,
        )
        with pytest.raises(ValueError):
            other.get("api_key")

    def test_rotate_master_key_reencrypts(self, tmp_path):
        mgr, _ = self._make(tmp_path)
        mgr.set("api_key", "v1")

        new_master = EncryptedFileSecretsManager.generate_master_key()
        mgr.rotate_master_key(new_master)

        # Using the new master works.
        again = EncryptedFileSecretsManager(
            tmp_path / "secrets.enc", master_key=new_master,
        )
        assert again.get("api_key") == "v1"

    def test_missing_master_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WANG_MASTER_KEY", raising=False)
        with pytest.raises(ValueError, match="master key"):
            EncryptedFileSecretsManager(tmp_path / "x.enc")

    def test_tampered_ciphertext_fails_to_decrypt(self, tmp_path):
        mgr, _ = self._make(tmp_path)
        mgr.set("api_key", "v1")
        path = tmp_path / "secrets.enc"
        data = path.read_bytes()
        # Flip a byte in the middle
        flipped = bytearray(data)
        flipped[len(flipped) // 2] ^= 0x01
        path.write_bytes(bytes(flipped))
        with pytest.raises(ValueError):
            mgr.get("api_key")


# ── Cloud backends (mocked) ────────────────────────────────────────────

class TestAWSSecretsManager:
    def test_get_caches_value(self):
        client = MagicMock()
        client.get_secret_value.return_value = {"SecretString": "v1"}
        mgr = AWSSecretsManager(client=client)
        assert mgr.get("k1") == "v1"
        # Second get pulls from cache, not client.
        assert mgr.get("k1") == "v1"
        client.get_secret_value.assert_called_once()

    def test_cache_expires(self):
        client = MagicMock()
        client.get_secret_value.return_value = {"SecretString": "v1"}
        mgr = AWSSecretsManager(client=client, cache_ttl_s=0.01)
        mgr.get("k1")
        time.sleep(0.05)
        mgr.get("k1")
        assert client.get_secret_value.call_count == 2

    def test_rotate_invalidates_cache(self):
        client = MagicMock()
        client.get_secret_value.return_value = {"SecretString": "v1"}
        mgr = AWSSecretsManager(client=client)
        mgr.get("k1")
        mgr.rotate("k1", "v2")
        client.put_secret_value.assert_called_once()
        # After invalidation, next get should re-fetch.
        client.get_secret_value.return_value = {"SecretString": "v2"}
        assert mgr.get("k1") == "v2"


class TestGCPSecretManager:
    def test_get_and_cache(self):
        payload = MagicMock()
        payload.data = b"v1"
        resp = MagicMock(payload=payload)
        client = MagicMock()
        client.access_secret_version.return_value = resp
        mgr = GCPSecretManager(project_id="proj", client=client)
        assert mgr.get("k1") == "v1"
        mgr.get("k1")  # cached
        client.access_secret_version.assert_called_once()


# ── Factory ────────────────────────────────────────────────────────────

class TestFactory:
    def test_env_backend(self):
        mgr = get_secrets_manager("env")
        assert isinstance(mgr, EnvSecretsManager)

    def test_file_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "WANG_MASTER_KEY",
            EncryptedFileSecretsManager.generate_master_key(),
        )
        mgr = get_secrets_manager("file", path=str(tmp_path / "x.enc"))
        assert isinstance(mgr, EncryptedFileSecretsManager)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown secrets backend"):
            get_secrets_manager("martian-vault")

    def test_reads_env_var_default(self, monkeypatch):
        monkeypatch.setenv("WANG_SECRETS_BACKEND", "env")
        mgr = get_secrets_manager()
        assert isinstance(mgr, EnvSecretsManager)
