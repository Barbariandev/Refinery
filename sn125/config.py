"""Launch-time constants for the SN125 production path.

Values here are public launch configuration, not secrets. Replace placeholders
with production values before launch. Keep reward-economics constants here so
the validator, dashboard, and dry-run path do not drift.
"""
from __future__ import annotations

import hashlib

TREASURY_COLDKEY = "5DLu5XrMV8Wt7aSmutxwAT1tNdwXtLxPDHvd5JAiY6WDnnX7"

AUTHORIZED_VALIDATOR_HOTKEY = "5D2tFvu5gyX3gKbNH1MLoFKYpbtQmKgJ3DQURaK6jhSehMUV"

BURN_UID = 0

NETUID = 125

LAUNCH_BURN_SCHEDULE = (
    {
        "phase": "full_incentive",
        "days": "1+",
        "duration_days": None,
        "burn_fraction": 0.0,
        "payable_fraction": 1.0,
        "summary": "Full miner emission available from day one, earned by verified progress. No automatic research pause.",
    },
)
LAUNCH_BURN_FRACTION_FLOOR = LAUNCH_BURN_SCHEDULE[0]["burn_fraction"]
LAUNCH_FIRST_CYCLE_DAYS = 42


def launch_burn_phase_for_day(day: int) -> dict:
    """Return the configured burn phase for a 1-indexed operation day."""
    try:
        day_i = max(1, int(day))
    except (TypeError, ValueError):
        day_i = 1
    elapsed = 0
    for phase in LAUNCH_BURN_SCHEDULE:
        duration = phase.get("duration_days")
        if duration is None:
            return dict(phase)
        elapsed += int(duration)
        if day_i <= elapsed:
            return dict(phase)
    return dict(LAUNCH_BURN_SCHEDULE[-1])


def launch_burn_fraction_for_day(day: int) -> float:
    return float(launch_burn_phase_for_day(day).get("burn_fraction", 0.0))

PROGRESS_HALF_LIFE_DAYS = 14.0
FULL_PAY_PROGRESS_NATS = 0.10
FULL_PAY_PROGRESS_NATS_LAUNCH = 0.03
FULL_PAY_RAMP_DAYS = 56.0
LAUNCH_TIMESTAMP = 0
FRONTIER_HALF_LIFE_DAYS = PROGRESS_HALF_LIFE_DAYS


def full_pay_reference_nats(now: float, anchor_ts: float | None) -> float:
    """Decayed-improvement level that buys 100% payable at time ``now``."""
    if not anchor_ts or anchor_ts <= 0 or FULL_PAY_RAMP_DAYS <= 0:
        return FULL_PAY_PROGRESS_NATS if anchor_ts is None else FULL_PAY_PROGRESS_NATS_LAUNCH
    frac = max(0.0, min(1.0, (float(now) - float(anchor_ts)) / (FULL_PAY_RAMP_DAYS * 86400.0)))
    return FULL_PAY_PROGRESS_NATS_LAUNCH + frac * (FULL_PAY_PROGRESS_NATS - FULL_PAY_PROGRESS_NATS_LAUNCH)


def progress_payable_share(progress_nats: float, now: float, anchor_ts: float | None) -> float:
    """min(1, P / full_pay_reference): the fraction of emission progress buys."""
    ref = full_pay_reference_nats(now, anchor_ts)
    if ref <= 0 or progress_nats <= 0:
        return 0.0
    return max(0.0, min(1.0, float(progress_nats) / ref))


def launch_day(now: float, launch_ts: float | None = None) -> int | None:
    """1-indexed operation day for the burn schedule, or None before launch /
    when LAUNCH_TIMESTAMP is unset."""
    ts = LAUNCH_TIMESTAMP if launch_ts is None else launch_ts
    if not ts or ts <= 0:
        return None
    return max(1, int((float(now) - float(ts)) // 86400) + 1)


MIN_FRONTIER_IMPROVEMENT = 0.03
FRONTIER_IMPROVEMENT_UNIT = "nats"
FRONTIER_THRESHOLD_FRACTION = 0.50
FRONTIER_THRESHOLD_WINDOW = 3
FRONTIER_THRESHOLD_DECAY_HALF_LIFE_DAYS = 7.0

FRONTIER_CONFIRMATION_RUNS = 1
FRONTIER_CONFIRMATION_MAX_ATTEMPTS = 3

INITIAL_OPERATION_DAYS = 40
PUBLIC_RANDOMNESS_ANCHOR = (
    "sn125-forge-initial-40d-public-seeds-v1:"
    "lean-llama-360m:fineweb-edu:seq2048:batch32"
)


def validator_hotkey_configured() -> bool:
    return (
        bool(AUTHORIZED_VALIDATOR_HOTKEY)
        and AUTHORIZED_VALIDATOR_HOTKEY
        != "CHANGE_ME_SN125_PRODUCTION_VALIDATOR_HOTKEY"
    )


def public_seed(namespace: str, index: int = 0) -> int:
    material = f"{PUBLIC_RANDOMNESS_ANCHOR}:{namespace}:{index}".encode()
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def public_trial_seeds(k: int, namespace: str = "main") -> list[int]:
    return [public_seed(namespace, i) for i in range(k)]


def public_seed_manifest(k: int) -> dict:
    return {
        "scheme": "sn125-public-seeds-v1",
        "initial_operation_days": INITIAL_OPERATION_DAYS,
        "anchor": PUBLIC_RANDOMNESS_ANCHOR,
        "main": public_trial_seeds(k, "main"),
        "budget_transfer": public_trial_seeds(1, "budget_transfer"),
        "transfer_audit": public_trial_seeds(1, "transfer_audit"),
        "transfer_task_index": public_seed("transfer_task", 0),
    }
