"""Small environment helpers for local operator scripts.

This intentionally handles only simple dotenv lines:

    KEY=value
    export KEY=value

It avoids a runtime dependency on python-dotenv while making the repo's scripts
work with the operator's local .env file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load simple KEY=value entries from a .env file into os.environ.

    Existing environment variables win unless ``override`` is true. The returned
    dict contains only variables loaded or overwritten by this call.
    """
    env_path = Path(path) if path else Path.cwd() / ".env"
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_"):
            continue
        if not all(c.isalnum() or c == "_" for c in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
