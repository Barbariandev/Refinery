"""Portable dashboard feed, generated with the same contract as the web server."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .data import build_dashboard_data

SNAPSHOT_SCHEMA = "refinery.dashboard-snapshot.v1"


def build_snapshot(rounds_dir: Path, cloud_status_path: Path | None, *,
                   validator_hotkey: str, version: str) -> dict:
    dashboard = build_dashboard_data(
        rounds_dir, cloud_status_path, validator_hotkey=validator_hotkey, version=version,
    )
    dashboard["network"]["status"] = "validator-artifacts"
    return {
        "schema": SNAPSHOT_SCHEMA,
        "generated": dashboard["generated"],
        "dashboard": dashboard,
        "cloud": dashboard["cloud"],
        "cloud_status_available": bool(dashboard["cloud"]),
    }


def write_snapshot(directory: Path, snapshot: dict) -> None:
    """Atomic local replacement; snapshot.json contains both feeds consistently."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("dashboard.json", snapshot["dashboard"]),
        ("cloud-status.json", snapshot["cloud"]),
        ("snapshot.json", snapshot),
    ):
        descriptor, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=directory)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(value, output, allow_nan=False, default=str)
            os.replace(temporary, directory / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
