"""Refiner task (Challenge 2) frontier: bar, confirmation, progress-scaled pool.

The optimizer task's mechanism (``Validator.compute_weights``,
docs/INCENTIVE_MECHANISM.md) applied to the refiner task in its own unit —
suite points, the benchmark-suite score of the student a refiner produced —
and its own pool:

  * a submission whose suite score beats the refiner frontier (the pinned
    identity baseline until an event confirms) by more than the adaptive bar
    becomes a PENDING candidate; the validator re-runs its envelope next
    round at its own cost; the event is created only when the confirmation
    also clears the bar, with the WORSE of the two runs as the event's score;
  * every confirmed event ``i`` carries ``credit_i(t) = g_i * 0.5 ** (age / 14 d)``;
    the pool pays ``min(1, sum credit / reference(t)) * (1 - burn_floor)`` of
    itself pro rata to credit and burns the rest;
  * :func:`blend_weights` folds the pool into the round's weight vector:
    ``share`` of miner emission is this pool, ``1 - share`` the optimizer
    frontier (``config.REFINER_EMISSION_SHARE``).

State lives in the public round records only (``refiner_task.frontier`` of
every round: ``new_event`` and the full ``pending_confirmation`` list), so
``python -m sn125.verify`` replays every settlement from prior records
exactly as the validator computed it. Events and pending candidates are keyed
by the task ``signature`` (roundsm.refiner_round.refiner_signature of the
pinned baseline) so a change of suite, student, budget, data or seed starts a
new frontier instead of ranking incomparable scores.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

from ..config import (FRONTIER_CONFIRMATION_MAX_ATTEMPTS, FRONTIER_CONFIRMATION_RUNS,
                      FRONTIER_THRESHOLD_DECAY_HALF_LIFE_DAYS, FRONTIER_THRESHOLD_FRACTION,
                      FRONTIER_THRESHOLD_WINDOW, MIN_REFINER_IMPROVEMENT, PROGRESS_HALF_LIFE_DAYS,
                      REFINER_IMPROVEMENT_UNIT, refiner_full_pay_reference, refiner_progress_payable_share)

log = logging.getLogger(__name__)

SCHEME = "refiner-frontier-progress-scaled-v1"
BURN_KEY = "__BURN__"          # same sentinel as Validator.BURN_KEY
CONFIRM_PREFIX = "rfconfirm-"  # round-JSON key / cloud sub_uid stem of a refiner confirmation run


def confirmation_key(commit_hash: str) -> str:
    return f"{CONFIRM_PREFIX}{str(commit_hash)[:16]}"


# ── reading prior rounds ─────────────────────────────────────────────────────
def _round_files(rounds_dir) -> list[Path]:
    rdir = Path(rounds_dir) if rounds_dir else None
    if rdir is None or not rdir.is_dir():
        return []
    return sorted(rdir.glob("*.json"))


def _frontier_of(rd: dict) -> dict | None:
    blk = rd.get("refiner_task")
    if not isinstance(blk, dict):
        return None
    fr = blk.get("frontier")
    return fr if isinstance(fr, dict) else None


def event_id(ev: dict) -> str:
    material = json.dumps({
        "hotkey": ev.get("hotkey", ""), "commit_hash": ev.get("commit_hash", ""),
        "timestamp": int(ev.get("timestamp", 0) or 0),
        "improvement": round(float(ev.get("improvement", 0.0) or 0.0), 12),
        "signature": ev.get("signature", ""),
    }, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def load_events(rounds_dir, signature: str) -> list[dict]:
    """Confirmed refiner frontier events of prior rounds for this task signature,
    oldest first (one ``new_event`` per round at most)."""
    events, seen = [], set()
    for fp in _round_files(rounds_dir):
        try:
            rd = json.loads(fp.read_text())
        except Exception:
            continue
        fr = _frontier_of(rd)
        ev = fr.get("new_event") if fr else None
        if not isinstance(ev, dict) or ev.get("signature", "") != signature:
            continue
        try:
            imp = float(ev.get("improvement", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(imp) or imp <= 0.0:
            continue
        ev = dict(ev)
        ev["improvement"] = imp
        ev.setdefault("timestamp", int(rd.get("timestamp", 0) or 0))
        ev.setdefault("event_id", event_id(ev))
        if ev["event_id"] in seen:
            continue
        seen.add(ev["event_id"])
        events.append(ev)
    events.sort(key=lambda e: (int(e.get("timestamp", 0) or 0), e.get("event_id", "")))
    return events


def load_pending(rounds_dir, signature: str) -> list[dict]:
    """Bar-clearing refiner candidates awaiting confirmation: the newest round
    that carries ``refiner_task.frontier.pending_confirmation`` is authoritative."""
    newest: tuple[int, str] | None = None
    pending: list[dict] = []
    for fp in _round_files(rounds_dir):
        try:
            rd = json.loads(fp.read_text())
        except Exception:
            continue
        fr = _frontier_of(rd)
        if fr is None or "pending_confirmation" not in fr:
            continue
        key = (int(rd.get("timestamp", 0) or 0), fp.name)
        if newest is not None and key <= newest:
            continue
        newest = key
        pending = [dict(p) for p in (fr.get("pending_confirmation") or [])
                   if isinstance(p, dict) and p.get("commit_hash") and p.get("signature", "") == signature]
    return pending


def recover_envelope(rounds_dir, commit_hash: str, source_round_id: str = "") -> str:
    """The published envelope text of a refiner commit (``sources`` is keyed by
    the sha256 of the text, which for an envelope IS the commit hash)."""
    files = _round_files(rounds_dir)
    if not files or not commit_hash:
        return ""
    ordered = ([f for f in files if f.stem == source_round_id] if source_round_id else []) + files[::-1]
    for fp in ordered:
        try:
            rd = json.loads(fp.read_text())
        except Exception:
            continue
        text = (rd.get("sources") or {}).get(commit_hash)
        if isinstance(text, str) and text:
            return text
    return ""


def pending_jobs(rounds_dir, cfg: dict | None) -> list[tuple[str, bytes, dict]]:
    """``(confirm-key, envelope bytes, pending record)`` for every refiner
    candidate awaiting its validator-funded confirmation run."""
    if not cfg or FRONTIER_CONFIRMATION_RUNS <= 0:
        return []
    jobs = []
    for p in load_pending(rounds_dir, cfg.get("signature", "")):
        ch = str(p.get("commit_hash", ""))
        text = recover_envelope(rounds_dir, ch, str(p.get("source_round_id", "") or ""))
        if not text:
            log.error("  refiner confirmation: no envelope on disk for %s (%s)", ch[:16], p.get("hotkey", ""))
            continue
        jobs.append((confirmation_key(ch), text.encode("utf-8"), p))
    return jobs


# ── the bar ──────────────────────────────────────────────────────────────────
def bar(events: list[dict], now: float, floor: float = MIN_REFINER_IMPROVEMENT) -> float:
    """Adaptive bar in suite points: ``max(floor, 0.5 × mean(last 3 events) ×
    0.5 ** (stagnant days / 7))`` — the optimizer frontier's rule."""
    recent = [float(ev.get("improvement", 0.0) or 0.0) for ev in events[-FRONTIER_THRESHOLD_WINDOW:]]
    recent = [g for g in recent if g > 0.0]
    if not recent:
        return floor
    b = FRONTIER_THRESHOLD_FRACTION * (sum(recent) / len(recent))
    last_ts = max(int(ev.get("timestamp", 0) or 0) for ev in events)
    stagnant_days = max(0.0, (now - last_ts) / 86400.0)
    b *= 0.5 ** (stagnant_days / FRONTIER_THRESHOLD_DECAY_HALF_LIFE_DAYS)
    return max(floor, b)


# ── reading a result ─────────────────────────────────────────────────────────
def suite_score(result: dict | None) -> float | None:
    """The absolute suite score of a scored refiner result (block entry or raw
    box result); ``None`` for a failed run or one whose held-out gate tripped
    (its score is 0 by rule and it cannot set a frontier)."""
    if not isinstance(result, dict) or result.get("failed"):
        return None
    for g in (result.get("gates") or {}).values():
        if isinstance(g, dict) and g.get("enabled") and g.get("passed") is False:
            return None
    bench = result.get("bench")
    if isinstance(bench, dict) and "submission" in bench:            # block entry
        val = bench.get("submission")
    else:                                                            # raw box result
        val = ((result.get("submission") or {}).get("bench") or {}).get("score")
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


# ── settlement ───────────────────────────────────────────────────────────────
def settle(cfg: dict, entries: dict, confirmations: dict | None, *, rounds_dir, round_id: str,
           now: float, burn_floor: float = 0.0, launch_timestamp: int = 0) -> dict:
    """One round of the refiner frontier. ``entries`` are this round's block
    entries (``build_refiner_block``), ``confirmations`` maps a pending commit
    hash to ``{"outcome", "reason", "result"}`` from its confirmation run.
    Returns the ``frontier`` section of the block; ``weights`` inside it is the
    POOL's weight map (hotkeys + BURN_KEY, summing to 1)."""
    now = int(now)
    signature = str(cfg.get("signature", "") or "")
    share = float(cfg.get("emission_share", 0.0) or 0.0)
    events = load_events(rounds_dir, signature)
    pending = load_pending(rounds_dir, signature) if FRONTIER_CONFIRMATION_RUNS > 0 else []
    confirmations = confirmations or {}
    the_bar = bar(events, now)
    try:
        baseline_score = float(cfg.get("baseline_score"))
    except (TypeError, ValueError):
        baseline_score = None
    frontier_score = max([baseline_score if baseline_score is not None else -math.inf] +
                         [float(e.get("suite_score", -math.inf)) for e in events])

    # 1. resolve pending confirmations
    outcomes, confirmed, still_pending = [], [], []
    for p in pending:
        ch = str(p.get("commit_hash", ""))
        rec = {"hotkey": p.get("hotkey", ""), "commit_hash": ch, "submission_round_id": p.get("source_round_id", "")}
        c = confirmations.get(ch)
        if c is None:
            still_pending.append(dict(p))
            outcomes.append({**rec, "outcome": "pending", "reason": "no confirmation result this round"})
            continue
        conf_score = suite_score(c.get("result")) if c.get("outcome") == "scored" else None
        if conf_score is not None:
            run1 = float(p.get("suite_score"))
            base = float(p.get("frontier_score"))
            worst = min(run1, conf_score)
            g_run1, g_conf, g = run1 - base, conf_score - base, worst - base
            if g > the_bar and g_run1 > the_bar and g_conf > the_bar:
                done = int(p.get("confirmations_done", 0) or 0) + 1
                p2 = dict(p, suite_score=worst, confirmations_done=done,
                          confirmation_scores=list(p.get("confirmation_scores") or []) + [conf_score])
                if done >= FRONTIER_CONFIRMATION_RUNS:
                    confirmed.append((float(g), p2, run1, conf_score, worst, base))
                else:
                    still_pending.append(p2)
                    outcomes.append({**rec, "outcome": "confirmed_partial", "improvement": float(g),
                                     "confirmations_done": done, "confirmations_required": FRONTIER_CONFIRMATION_RUNS})
            else:
                outcomes.append({**rec, "outcome": "rejected", "reason": "confirmation run did not clear the bar",
                                 "improvement_submission": g_run1, "improvement_confirmation": g_conf,
                                 "bar": the_bar, "frontier_score": base, "confirmation_score": conf_score})
            continue
        verdict = str(c.get("outcome", "infra_dq"))
        reason = str(c.get("reason", ""))
        if verdict == "scored":
            verdict, reason = "dq", reason or "confirmation run produced no suite score"
        if verdict == "dq":
            outcomes.append({**rec, "outcome": "rejected", "reason": f"confirmation run failed: {reason}"})
            continue
        attempts = int(p.get("attempts", 0) or 0) + 1
        if attempts >= FRONTIER_CONFIRMATION_MAX_ATTEMPTS:
            outcomes.append({**rec, "outcome": "rejected",
                             "reason": f"confirmation could not be run after {attempts} attempts: {reason}"})
        else:
            still_pending.append(dict(p, attempts=attempts))
            outcomes.append({**rec, "outcome": "retry", "attempts": attempts, "reason": reason})

    # 2. best confirmed candidate becomes the frontier event
    new_event = None
    if confirmed:
        confirmed.sort(key=lambda x: (-x[0], x[1].get("commit_hash", "")))
        g, p, run1, conf_score, worst, base = confirmed[0]
        new_event = {
            "event_id": "", "hotkey": p.get("hotkey", ""), "commit_hash": p.get("commit_hash", ""),
            "timestamp": now, "improvement": float(g), "improvement_unit": REFINER_IMPROVEMENT_UNIT,
            "suite_score": worst, "frontier_score_before": base, "baseline_score": baseline_score,
            "signature": signature, "weights": p.get("weights"),
            "confirmation": {"submission_round_id": p.get("source_round_id", ""), "confirmation_round_id": round_id,
                             "runs": {"submission": run1, "confirmations": list(p.get("confirmation_scores") or [conf_score])}},
        }
        new_event["event_id"] = event_id(new_event)
        events.append(new_event)
        frontier_score = max(frontier_score, worst)
        outcomes.append({"hotkey": new_event["hotkey"], "commit_hash": new_event["commit_hash"],
                         "submission_round_id": p.get("source_round_id", ""), "outcome": "confirmed",
                         "improvement": float(g), "event_id": new_event["event_id"]})
        for g2, p2, *_ in confirmed[1:]:
            outcomes.append({"hotkey": p2.get("hotkey", ""), "commit_hash": p2.get("commit_hash", ""),
                             "submission_round_id": p2.get("source_round_id", ""), "outcome": "superseded",
                             "improvement": float(g2), "reason": "a better candidate confirmed in the same round"})

    # 3. this round's candidates: the best bar-clearing entry goes to confirmation
    candidates = []
    if frontier_score > -math.inf:
        for ch, e in (entries or {}).items():
            if not isinstance(e, dict) or str(e.get("signature", "") or "") != signature:
                continue
            s = suite_score(e)
            if s is None:
                continue
            g = s - frontier_score
            if g > the_bar:
                candidates.append((g, str(ch), e, s))
    if candidates:
        g, ch, e, s = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]   # best improvement, ties → lowest hash
        if any(p.get("commit_hash") == ch for p in still_pending):
            outcomes.append({"hotkey": e.get("hotkey", ""), "commit_hash": ch, "outcome": "duplicate",
                             "reason": "same envelope already awaiting confirmation"})
        else:
            still_pending.append({
                "hotkey": e.get("hotkey", ""), "commit_hash": ch, "source_round_id": round_id, "timestamp": now,
                "improvement": float(g), "improvement_unit": REFINER_IMPROVEMENT_UNIT, "suite_score": s,
                "frontier_score": frontier_score, "baseline_score": baseline_score, "signature": signature,
                "weights": e.get("weights"), "attempts": 0, "confirmations_done": 0, "confirmation_scores": [],
            })
            outcomes.append({"hotkey": e.get("hotkey", ""), "commit_hash": ch, "submission_round_id": round_id,
                             "outcome": "pending_new", "improvement": float(g),
                             "reason": "cleared the bar; awaiting confirmation run"})

    # 4. pay the pool: progress-scaled, pro rata to decayed credit
    half_life_s = max(1.0, PROGRESS_HALF_LIFE_DAYS * 86400.0)
    active, total_credit, first_ts = [], 0.0, None
    for ev in events:
        try:
            imp, ts = float(ev.get("improvement", 0.0)), int(ev.get("timestamp", now) or now)
        except (TypeError, ValueError):
            continue
        first_ts = ts if first_ts is None else min(first_ts, ts)
        credit = imp * (0.5 ** (max(0.0, now - ts) / half_life_s))
        if credit <= 0.0:
            continue
        active.append(dict(ev, decayed_credit=credit, age_days=max(0.0, now - ts) / 86400.0))
        total_credit += credit
    anchor = int(launch_timestamp or 0) or first_ts
    reference = refiner_full_pay_reference(now, anchor)
    progress_share = refiner_progress_payable_share(total_credit, now, anchor)
    burn_floor = min(1.0, max(0.0, float(burn_floor or 0.0)))
    payable = progress_share * (1.0 - burn_floor)
    weights: dict[str, float] = {}
    for ev in active:
        hk = ev.get("hotkey", "")
        if hk and total_credit > 0.0:
            ev["share"] = ev["decayed_credit"] / total_credit
            weights[hk] = weights.get(hk, 0.0) + payable * ev["share"]
    paid = sum(weights.values())
    burned = max(0.0, 1.0 - paid)
    if burned > 1e-12:
        weights[BURN_KEY] = burned
    return {
        "scheme": SCHEME, "emission_share": share, "improvement_unit": REFINER_IMPROVEMENT_UNIT,
        "timestamp": now, "signature": signature, "confirmation_runs": FRONTIER_CONFIRMATION_RUNS,
        "half_life_days": PROGRESS_HALF_LIFE_DAYS, "launch_timestamp": int(launch_timestamp or 0),
        "burn_fraction_floor": burn_floor, "min_improvement_floor": MIN_REFINER_IMPROVEMENT,
        "min_improvement_threshold": the_bar, "baseline_score": baseline_score,
        "frontier_score": frontier_score if frontier_score > -math.inf else None,
        "leader_hotkey": events[-1].get("hotkey", "") if events else "",
        "new_event": new_event, "pending_confirmation": still_pending, "confirmation_outcomes": outcomes,
        "progress": total_credit, "full_pay_reference": reference, "progress_payable_share": progress_share,
        "ramp_anchor_timestamp": anchor, "payable_share": paid, "burned_share": burned,
        "events": active, "weights": weights,
    }


def blend_weights(optimizer: dict, pool: dict | None, share: float, burn_key: str = BURN_KEY) -> dict[str, float]:
    """``(1 - share) × optimizer + share × pool``, both burn-inclusive maps that
    sum to 1; the result sums to 1 with the burn key carrying both pools'
    unearned emission. ``share`` 0 (publish-only) returns the optimizer map as is."""
    share = float(share or 0.0)
    if share <= 0.0 or pool is None:
        return dict(optimizer)
    out: dict[str, float] = {}
    for hk, w in optimizer.items():
        out[hk] = out.get(hk, 0.0) + (1.0 - share) * float(w)
    for hk, w in pool.items():
        out[hk] = out.get(hk, 0.0) + share * float(w)
    burn = out.pop(burn_key, 0.0)
    out = {hk: w for hk, w in out.items() if w > 0.0 or hk in optimizer}
    if burn > 1e-12:
        out[burn_key] = burn
    return out
