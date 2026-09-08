"""Durable inter-round loop state for ``run_fsm_validator`` (live.py).

Between rounds the loop carries state that lives nowhere else:

- ``carryover``      DEFERRED submissions (credit already debited, preimage
                     already revealed) waiting for FIFO re-admission next round
- ``recent_rounds``  the fairness window: hotkeys evaluated in the last N rounds
- ``round_num``      the round counter (part of every round id)
- ``last_weights``   the last computed weight map the epoch refresh thread
                     keeps re-setting on chain between rounds

Losing any of it on a restart is a real loss: a deferred miner paid and never
gets evaluated, a fresh process starts at round 0 again, and vtrust decays
until the next round computes weights. This module writes all of it as ONE
atomic JSON snapshot at every round boundary (the only point where a restart
is lossless) and restores it at start-up, which is what makes the auto-update
supervisor (sn125/autoupdate.py) safe.

The snapshot is derived state — the durable ledger (payments/store.py) and the
round JSONs stay the source of truth for money and scores — so a corrupt or
missing state file degrades to "start at round 0 with an empty carryover",
loudly, instead of refusing to start.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

from ..payments.store import atomic_write_json
from .round_fsm import Submission, SubStatus

log = logging.getLogger("sn125.roundsm.state")

STATE_VERSION = 1
STATE_FILENAME = "validator_state.json"


def default_state_path(audit_dir: str | Path) -> Path:
    return Path(audit_dir) / STATE_FILENAME


def _submission_to_json(sub: Submission) -> dict[str, Any]:
    return {
        "commit_hash": sub.commit_hash,
        "hotkey": sub.hotkey,
        "coldkey": sub.coldkey,
        "payload_b64": (base64.b64encode(sub.payload).decode("ascii")
                        if sub.payload is not None else None),
        "accepted_at": float(sub.accepted_at),
        "accept_index": int(sub.accept_index),
        "deferrals": int(sub.deferrals),
        "forensics": list(sub.forensics),
    }


def _submission_from_json(raw: dict[str, Any]) -> Submission:
    payload = raw.get("payload_b64")
    return Submission(
        commit_hash=str(raw["commit_hash"]),
        hotkey=str(raw["hotkey"]),
        coldkey=str(raw["coldkey"]),
        status=SubStatus.DEFERRED,
        payload=base64.b64decode(payload) if payload is not None else None,
        accepted_at=float(raw.get("accepted_at", 0.0)),
        accept_index=int(raw.get("accept_index", 0)),
        deferrals=int(raw.get("deferrals", 0)),
        forensics=list(raw.get("forensics") or []),
    )


def save_loop_state(path: str | Path, *, round_num: int,
                    carryover: list[Submission],
                    recent_rounds: list[set[str]],
                    last_weights: dict[str, float] | None,
                    last_round_id: str | None = None) -> None:
    """Atomically write the inter-round snapshot (tmp + fsync + rename)."""
    payload = {
        "version": STATE_VERSION,
        "saved_at": int(time.time()),
        "round_num": int(round_num),
        "last_round_id": last_round_id,
        "carryover": [_submission_to_json(s) for s in carryover],
        "recent_rounds": [sorted(r) for r in recent_rounds],
        "last_weights": {str(k): float(v) for k, v in (last_weights or {}).items()},
    }
    atomic_write_json(path, payload)


def load_loop_state(path: str | Path) -> dict[str, Any] | None:
    """Decoded snapshot or ``None`` when absent/unusable (logged, never raised).

    Returns ``{"round_num", "carryover", "recent_rounds", "last_weights",
    "last_round_id", "saved_at"}`` with the carryover already rebuilt as
    ``Submission`` objects in DEFERRED status.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("version", 0)) != STATE_VERSION:
            raise ValueError(f"unsupported state version {data.get('version')!r}")
        carryover = [_submission_from_json(s) for s in (data.get("carryover") or [])]
        for s in carryover:
            if s.payload is None:
                raise ValueError(f"carryover {s.commit_hash[:12]} has no payload")
        recent = [set(str(h) for h in r) for r in (data.get("recent_rounds") or [])]
        weights = {str(k): float(v) for k, v in (data.get("last_weights") or {}).items()}
        return {
            "round_num": int(data.get("round_num", 0)),
            "last_round_id": data.get("last_round_id"),
            "saved_at": int(data.get("saved_at", 0) or 0),
            "carryover": carryover,
            "recent_rounds": recent,
            "last_weights": weights,
        }
    except Exception as e:
        log.error("validator state %s is unusable (%s); starting from round 0 with "
                  "an empty carryover. Money and scores are unaffected (ledger and "
                  "round files are the source of truth).", path, e)
        return None
