"""Launch-time constants for the SN125 production path.

Values here are public launch configuration, not secrets. Replace placeholders
with production values before launch. Keep reward-economics constants here so
the validator, dashboard, and dry-run path do not drift.
"""
from __future__ import annotations

import hashlib

# Public destination for prepaid evaluation credits. The per-round fee is
# pinned separately; this address is only the receiving treasury account.
TREASURY_COLDKEY = "5DLu5XrMV8Wt7aSmutxwAT1tNdwXtLxPDHvd5JAiY6WDnnX7"

# The production validator hotkey (as published on the release branch). Miners
# refuse every CommitHash/GetSubmission request from any other hotkey instead
# of trusting Bittensor's default Axon verifier.
AUTHORIZED_VALIDATOR_HOTKEY = "5HNaCr7KPjZ16fGaqLL9jnyNLrDTwv49hqCzF8XZGShjK125"

# Unearned emission is represented internally by Validator.BURN_KEY and routed
# to this UID on chain. UID 0 is the subnet-owner/burn convention.
BURN_UID = 0

# Refinery lives on Bittensor netuid 125 (finney). Hard-coded on purpose: the
# validator, the attestation bundle and the lite attestation validator refuse
# any other netuid so a mis-typed flag can never set weights elsewhere.
NETUID = 125

# Full incentives from day one: no scheduled withholding of miner emission.
# Payment still depends on verified progress; unearned emission burns.
# After six weeks, discuss results and whether a research pause is useful.
# That discussion is not an automatic pause or a scheduled burn increase.
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
LAUNCH_FIRST_CYCLE_DAYS = 42  # Review milestone, not a stop condition.


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
# Unix timestamp of the first mainnet round, or 0 (unset, as on the release
# branch): the payout ramp then anchors on the first confirmed frontier event.
LAUNCH_TIMESTAMP = 0

# ── Round schedule (2026-10-17) ───────────────────────────────────────────
# Rounds open on a 24 h grid anchored on Monday 2026-10-19 00:00:00 UTC — the
# launch round. The validator (sn125/roundsm/live.py) applies it as:
#   * a round that finishes before the next grid line waits for it, so the
#     commit window opens at 00:00 UTC every day it can; a round that overran
#     its grid line opens the next one immediately (GPU time is never idled),
#     and the grid is picked up again by the next round that finishes early;
#   * before the anchor, a round opens only if its full period fits before it,
#     so the round that opens AT the anchor is the launch round;
#   * Challenge 2 (REFINER_START_TIMESTAMP) is live from the round that opens
#     at or after its start instant: earlier rounds carry no refiner block,
#     reject refiner envelopes at the gate, and keep the Challenge-1-only
#     weight vector.
# 0 disables the grid (rounds open period-to-period from process start, the
# pre-2026-10-17 behaviour); the dry run and the tests inject 0 with their
# synthetic clocks.
ROUND_GRID_ANCHOR_TIMESTAMP = 1792368000      # 2026-10-19T00:00:00Z
REFINER_START_TIMESTAMP = ROUND_GRID_ANCHOR_TIMESTAMP


def next_round_start(round_start: float, now: float, period_s: float,
                     anchor_ts: float | int | None) -> float:
    """When the round after one that opened at ``round_start`` opens: the first
    grid line after ``round_start`` (``anchor_ts`` + k × ``period_s``), or
    ``round_start + period_s`` without a grid — never earlier than ``now``."""
    period = float(period_s)
    if anchor_ts and anchor_ts > 0 and period > 0:
        k = (float(round_start) - float(anchor_ts)) // period + 1
        target = float(anchor_ts) + k * period
    else:
        target = float(round_start) + period
    return max(float(now), target)


def round_hold_until(now: float, period_s: float, anchor_ts: float | int | None) -> float:
    """Earliest instant a round may open at ``now``: ``now`` itself, except that
    before the grid anchor a round whose full period would straddle the anchor
    waits for it (the launch round opens exactly on the anchor)."""
    if anchor_ts and anchor_ts > 0 and float(now) < float(anchor_ts) \
            and float(now) + float(period_s) > float(anchor_ts):
        return float(anchor_ts)
    return float(now)
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


# Frontier improvement is measured in NATS of held-out loss (absolute delta
# `L_baseline - L_submission`, min over tasks), not as a relative fraction.
# Run-to-run noise is a nats quantity (it does not shrink as the frontier
# improves), so the bar is expressed in the same unit. Before 2026-09-03 the
# validator compared this constant against the RELATIVE improvement
# `(L_b - L_s) / L_b` while the docs described it as 0.03 nats — a 3% relative
# bar is ~0.083 nats at the genesis loss (2.7764), i.e. ~19 sigma: a bar that
# real optimizer progress (a ~1.35x compute-efficiency gain is ~0.03 nats
# here) could essentially never clear. `Validator._improvement_and_significance`
# now derives nats from the recorded final losses and falls back to the
# relative number only when a score record carries no losses (stricter, never
# looser).
# Absolute floor: a new frontier event must beat the rolling best by more than
# MIN_FRONTIER_IMPROVEMENT nats. On top of this floor an ADAPTIVE threshold
# applies (Validator._min_improvement_threshold): 50% of the mean improvement
# of the last 3 frontier events, halving every 7 stagnant days. That keeps
# "resubmit the leader's source and win on run-to-run noise" negative-EV.
# Replication-noise basis (2026-07-04, twelve B200 runs, torch 2.8.0+cu128):
#   - DECAYED schedule (genesis_adamw_wsd, WSD fully decayed inside the budget;
#     the regime every competitive submission occupies): 3 replicas landed
#     final held-out losses std 0.0044 / spread 0.0084 nats
#     (experiments/data/replication_floor_20260704_145738.json). Against a
#     copycat of a decayed leader the relevant noise is the paired difference
#     (both the leader's realized loss and the copy are noisy): std ~0.0062,
#     so the floor is ~4.8 sigma single-run — and n=3 leaves ~40% uncertainty
#     on that sigma, which is why frontier events additionally require an
#     independent CONFIRMATION run (FRONTIER_CONFIRMATION_RUNS) before they
#     pay or move the bar.
#   - CONSTANT LR (no schedule): heavy-tailed — nine runs across three batches
#     showed matched-step spreads of 0.05-0.36 nats (pooled std ~0.12). An
#     un-decayed LR amplifies kernel nondeterminism chaotically. Constant-LR
#     submissions also finish ~0.6 nats WORSE than decayed ones at this
#     budget, so this tail cannot reach the frontier; it is priced out on
#     merit, not on the threshold.
# The eval itself is deterministic given the checkpoint.
MIN_FRONTIER_IMPROVEMENT = 0.03                # nats of held-out loss
FRONTIER_IMPROVEMENT_UNIT = "nats"
FRONTIER_THRESHOLD_FRACTION = 0.50          # of trailing mean improvement
FRONTIER_THRESHOLD_WINDOW = 3               # trailing frontier events averaged
FRONTIER_THRESHOLD_DECAY_HALF_LIFE_DAYS = 7.0  # bar halves per stagnant week

# Confirmation before payment (2026-09-03). A submission that clears the bar
# is NOT paid and does NOT move the rolling baseline on its first run. The
# validator re-runs the same source (validator-funded, same task, same public
# seed) in the next round; the frontier event is created only if the
# confirmation ALSO clears the bar, and the recorded frontier loss is the
# WORSE of the two runs (a lucky draw can never set the bar). Cost: one extra
# evaluation per bar-clearing candidate — rare by construction — against a
# false frontier that would otherwise collect the leader share until beaten
# and ratchet the bar out of reach of real improvements. 0 disables (pay on
# the single run, legacy behaviour).
FRONTIER_CONFIRMATION_RUNS = 1
# A confirmation that cannot be RUN (provider flake / infra) is retried on
# later rounds up to this many attempts; a confirmation run that the miner's
# code fails (DQ) rejects the candidate outright.
FRONTIER_CONFIRMATION_MAX_ATTEMPTS = 3

REFINER_EMISSION_SHARE = 0.5
MIN_REFINER_IMPROVEMENT = 0.02                 # suite points
REFINER_IMPROVEMENT_UNIT = "suite_points"
# Full-pay reference for the refiner pool: the first floor-clearing event pays
# the whole pool at launch; in steady state one floor-clearing event a week
# keeps it fully paid (0.02 / (1 - 0.5 ** 0.5) = 0.068). Same 56-day ramp,
# same 14-day half-life, same adaptive bar (50% of the trailing three, halving
# per stagnant week) and the same confirmation rule as the optimizer frontier.
REFINER_FULL_PAY_PROGRESS = 0.07
REFINER_FULL_PAY_PROGRESS_LAUNCH = 0.02


def refiner_full_pay_reference(now: float, anchor_ts: float | None) -> float:
    """Decayed-improvement level (suite points) that buys 100% of the refiner pool."""
    if not anchor_ts or anchor_ts <= 0 or FULL_PAY_RAMP_DAYS <= 0:
        return REFINER_FULL_PAY_PROGRESS if anchor_ts is None else REFINER_FULL_PAY_PROGRESS_LAUNCH
    frac = max(0.0, min(1.0, (float(now) - float(anchor_ts)) / (FULL_PAY_RAMP_DAYS * 86400.0)))
    return REFINER_FULL_PAY_PROGRESS_LAUNCH + frac * (REFINER_FULL_PAY_PROGRESS - REFINER_FULL_PAY_PROGRESS_LAUNCH)


def refiner_progress_payable_share(progress: float, now: float, anchor_ts: float | None) -> float:
    """min(1, P / reference): the fraction of the refiner pool progress buys."""
    ref = refiner_full_pay_reference(now, anchor_ts)
    if ref <= 0 or progress <= 0:
        return 0.0
    return max(0.0, min(1.0, float(progress) / ref))


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
