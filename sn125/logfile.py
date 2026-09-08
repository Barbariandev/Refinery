"""Rotating on-disk copy of the process log, kept under the audit directory.

The validator's stdout/stderr normally lands only in journald / a Docker log
driver on the box. Writing the same records to ``<audit_dir>/validator.log``
(and ``autoupdate.log`` for the supervisor) puts the full logbook next to the
audit JSONL, where the continuous publisher mirrors it to the private R2
bucket on every cycle. Rotation caps disk use; the publisher re-uploads the
active file when it grows and picks up the ``.N`` backups by pattern.

Attaching is idempotent per path and never raises: a read-only audit dir
degrades to a warning, not a validator that refuses to start.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_BACKUPS = 5
MAX_BYTES_ENV = "SN125_PROCESS_LOG_MAX_BYTES"
BACKUPS_ENV = "SN125_PROCESS_LOG_BACKUPS"
DISABLE_ENV = "SN125_PROCESS_LOG"

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def attach_rotating_log(path: str | Path, *, logger: logging.Logger | None = None,
                        max_bytes: int | None = None, backups: int | None = None,
                        level: int = logging.INFO
                        ) -> logging.handlers.RotatingFileHandler | None:
    """Add a RotatingFileHandler for ``path`` to ``logger`` (root by default).

    Returns the handler, the already-attached handler for the same path, or
    ``None`` when disabled via ``SN125_PROCESS_LOG=0`` or the file cannot be
    opened.
    """
    if (os.environ.get(DISABLE_ENV) or "").strip().lower() in {"0", "false", "no", "off"}:
        return None
    target = logging.getLogger() if logger is None else logger
    path = Path(path).expanduser()
    for h in target.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler) and \
                Path(getattr(h, "baseFilename", "")) == path.resolve():
            return h
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            str(path), maxBytes=int(max_bytes if max_bytes is not None
                                    else _env_int(MAX_BYTES_ENV, DEFAULT_MAX_BYTES, minimum=1 << 20)),
            backupCount=int(backups if backups is not None
                            else _env_int(BACKUPS_ENV, DEFAULT_BACKUPS)),
            encoding="utf-8")
    except OSError as exc:
        logging.getLogger("sn125.logfile").warning(
            "process log %s not written (%s); stdout/journal only", path, exc)
        return None
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    target.addHandler(handler)
    if target.level == logging.NOTSET or target.level > level:
        target.setLevel(level)
    logging.getLogger("sn125.logfile").info("process log -> %s", path)
    return handler
