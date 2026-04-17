#!/usr/bin/env python3
"""Interactive helper to bootstrap secrets storage (P6.15).

Run:  python scripts/setup_secrets.py
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.secrets import (  # noqa: E402
    EncryptedFileSecretsManager,
    get_secrets_manager,
)


KEYS_TO_SEED = [
    "alpaca.api_key",
    "alpaca.secret_key",
    "binance.api_key",
    "binance.secret_key",
    "coinbase.api_key",
    "coinbase.secret_key",
    "coinbase.passphrase",
    "ibkr.account_id",
    "telegram.bot_token",
    "telegram.chat_id",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("setup_secrets")
    p.add_argument("--backend", default=os.environ.get("WANG_SECRETS_BACKEND", "file"),
                   choices=["file", "env", "aws", "gcp"])
    p.add_argument("--path", default="config/secrets.enc",
                   help="Encrypted secrets file path (file backend only)")
    p.add_argument("--generate-master-key", action="store_true",
                   help="Print a fresh Fernet master key and exit")
    p.add_argument("--rotate-master-key", action="store_true",
                   help="Re-encrypt the secrets file under a new master key")
    return p.parse_args()


def _interactive_seed(mgr) -> None:
    print("Enter secret values (leave blank to skip):")
    for key in KEYS_TO_SEED:
        value = getpass.getpass(f"  {key}: ")
        if value:
            mgr.set(key, value)
            print(f"    stored {key}")
    print(f"Done. Stored keys: {mgr.list_keys()}")


def main() -> int:
    args = _parse_args()

    if args.generate_master_key:
        print(EncryptedFileSecretsManager.generate_master_key())
        return 0

    if args.backend == "file" and not os.environ.get("WANG_MASTER_KEY"):
        print("ERROR: WANG_MASTER_KEY is not set. Generate one with:")
        print("  python scripts/setup_secrets.py --generate-master-key")
        return 1

    kwargs = {"path": args.path} if args.backend == "file" else {}
    mgr = get_secrets_manager(args.backend, **kwargs)

    if args.rotate_master_key and args.backend == "file":
        new_key = EncryptedFileSecretsManager.generate_master_key()
        mgr.rotate_master_key(new_key)  # type: ignore[attr-defined]
        print("Secrets file re-encrypted. New master key:")
        print(new_key)
        return 0

    _interactive_seed(mgr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
