"""Durable ledger snapshots for :class:`PaymentRegistry` (registry.py).

Credits, spent credits, processed-transfer ids and the chain scan cursor must
survive a validator restart. A crash mid-write must never leave a half-applied
ledger behind (that would drop balances or re-credit an already-spent deposit).

Every write is a full JSON snapshot: serialize to a sibling ``*.tmp`` file,
``fsync`` it, ``os.replace`` onto the live path, then ``fsync`` the directory
so the rename itself is durable. Readers treat a missing file as an empty
ledger; a present but unreadable/corrupt file is a HARD error — a validator
must never silently start from zero on top of real paid balances.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

LEDGER_VERSION = 1
LEDGER_ENV = "SN125_PAYMENT_LEDGER"


class LedgerStoreError(Exception):
    """The ledger file exists but cannot be trusted."""


def default_ledger_path(audit_dir: str | Path | None = None,
                        rounds_dir: str | Path | None = None) -> Path:
    """``<audit_dir>/payments/ledger.json``.

    The audit directory is the validator's durable, operator-owned state
    volume (compose.yaml mounts it); the ``payments/`` subdirectory keeps the
    ledger out of the artifact publisher's non-recursive ``*.jsonl`` glob.
    Falls back to the sibling ``audit`` dir of ``rounds_dir``, then to the
    package default ``sn125/audit``.
    """
    if audit_dir:
        base = Path(audit_dir).expanduser()
    elif rounds_dir:
        base = Path(rounds_dir).expanduser().resolve().parent / "audit"
    else:
        base = Path(__file__).resolve().parents[1] / "audit"
    return base / "payments" / "ledger.json"


def resolve_ledger_path(explicit: str | Path | None = None, *,
                        audit_dir: str | Path | None = None,
                        rounds_dir: str | Path | None = None,
                        env: dict[str, str] | None = None) -> Path:
    """CLI flag > ``SN125_PAYMENT_LEDGER`` env > :func:`default_ledger_path`."""
    if explicit:
        return Path(explicit).expanduser()
    src = os.environ if env is None else env
    from_env = (src.get(LEDGER_ENV) or "").strip()
    if from_env:
        return Path(from_env).expanduser()
    return default_ledger_path(audit_dir, rounds_dir)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` with ``payload`` durably (tmp + fsync + rename + dir fsync)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=1, sort_keys=True).encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def load_ledger(path: str | Path) -> dict[str, Any] | None:
    """Decoded snapshot, or ``None`` when ``path`` does not exist.

    Raises :class:`LedgerStoreError` for a present-but-invalid file so the
    caller cannot rebuild an empty registry over paid state by accident.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise LedgerStoreError(f"unreadable payment ledger {path}: {e}") from e
    if not isinstance(data, dict):
        raise LedgerStoreError(f"payment ledger {path} is not a JSON object")
    if data.get("version") != LEDGER_VERSION:
        raise LedgerStoreError(
            f"payment ledger {path} has unsupported version {data.get('version')!r} "
            f"(expected {LEDGER_VERSION})")
    if not isinstance(data.get("events"), list):
        raise LedgerStoreError(f"payment ledger {path} is missing its events list")
    return data
