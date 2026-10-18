"""Live FSM validator loop (SPEC §4.1) — the production `run()` rewrite, extracted
as an injectable function so the whole 24h round loop is unit-testable on CPU.

``run_fsm_validator(validator)`` is the ONLY validation loop — the ``validate``
CLI command calls it directly. It owns the per-round orchestration that the
lower layers were built for:

    at start-up:
      record the fee in force -> catch the credit ledger up to the finalized
      head (backfill) -> start the continuous payment watcher (payments/watch.py)
    per 24h round:
      pin the round fee  ->  sync credits from observed treasury transfers
      build RoundFSM + adapters (commit/reveal transport, source gate, cloud eval)
      drive_round(...)   ->  COMMIT_OPEN -> REVEAL_OPEN -> select<=8 -> evaluate -> publish
      build ScoreRecords for the SCORED submissions
      compute_weights    ->  save round  ->  set weights (if enabled)  ->  publish attestation
      carry DEFERRED commits forward to next round
      checkpoint the inter-round state (roundsm/state.py)
      honour a pending restart request (auto-update) at this safe point
      sleep to the 24h boundary

Restart safety: the round boundary — after the round is saved, published and
its state checkpointed, before the next round pins a fee — is the ONLY point
where stopping the process loses nothing. ``run_fsm_validator`` therefore
checks for a restart request there (a flag file named by ``SN125_RESTART_FLAG``
/ ``validator.restart_flag_path``, written by sn125/autoupdate.py) and returns
``"restart"`` instead of starting the next round; the next process resumes
from the checkpoint (round counter, deferred carryover, fairness window, last
weights) and the durable payment ledger.

Everything impure arrives through the ``validator`` object (its dendrite, cloud
orchestrator, metagraph sync, weight/attestation helpers, and the configured
``payment_registry`` / treasury / fee). A fake validator + ``MockChain`` therefore
drives the entire loop deterministically in milliseconds (see ``test_live.py``),
which is exactly the SPEC §4.4 "MockChain dry-run" property carried up to the
real loop. This module imports no bittensor/torch at load — the synapse classes
are read off the validator (``commit_synapse_cls`` / ``submission_synapse_cls``)
and ``ScoreRecord`` is imported lazily only when building results.

Concurrency: the <=8 selected submissions are evaluated through
``drive_round(evaluate_batch=...)``, which hands them to one call that fans them
across a thread pool — so the eval phase is wall-clock ~one 20h eval, not <=8
back-to-back (the sequential per-item ``evaluate`` path stays for tests).
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .. import settings
from ..config import next_round_start, round_hold_until
from .adapter import (
    classify_cloud_result,
    make_commit_collector,
    make_reveal_collector,
    make_source_gate,
    is_no_gpu_failure,
    no_box_ran,
)
from .audit import JsonlAuditLog, audit_emit, looks_like_crash, payload_record
from .driver import (DEFAULT_LAUNCH_RETRY_S, DEFAULT_LAUNCH_WINDOW_S,
                     DEFAULT_OUTAGE_BACKOFF_S, DEFAULT_REVEAL_RETRY_S,
                     EvalResult, drive_round)
from . import refiner_frontier
from .refiner_round import (build_refiner_block, is_refiner_payload, load_refiner_round_config,
                            refiner_job_for)
from .round_fsm import MAX_EVAL_PER_ROUND, RoundConfig, RoundFSM, SubStatus
from .state import default_state_path, load_loop_state, save_loop_state

log = logging.getLogger("sn125.fsm")

# Restart-at-safe-point protocol shared with sn125/autoupdate.py.
RESTART_FLAG_ENV = "SN125_RESTART_FLAG"
EXIT_RESTART = 75            # EX_TEMPFAIL: "restart me" (supervisor applies updates)
_BOUNDARY_POLL_S = 60.0      # how often the boundary sleep re-checks the flag

# Defaults for the round windows (seconds) when the validator does not override
# them. A 24h round = 3h commit + 1h reveal + ~20h evaluation, matching the
# fixed-budget regime (SPEC §7); the eval phase itself consumes the rest.
DEFAULT_COMMIT_WINDOW_S = 3 * 3600.0
DEFAULT_REVEAL_WINDOW_S = 1 * 3600.0
DEFAULT_ROUND_PERIOD_S = 24 * 3600.0
DEFAULT_FAIRNESS_WINDOW = 3   # retained for config compatibility; FIFO selection ignores it


def _audit_heartbeat_interval() -> float:
    return settings.env_float(settings.AUDIT_HEARTBEAT_S_ENV, 60.0, minimum=10.0)


def _audit_dir_for(validator) -> Path:
    configured = (getattr(validator, "audit_dir", "") or
                  os.environ.get("SN125_AUDIT_DIR", ""))
    if configured:
        return Path(configured)
    rounds_dir = Path(getattr(validator, "rounds_dir", "") or
                      (Path(__file__).resolve().parents[1] / "rounds"))
    return rounds_dir.parent / "audit"


def _make_round_audit(validator, round_id: str):
    audit_dir = _audit_dir_for(validator)
    vhk = (validator.wallet.hotkey.ss58_address
           if getattr(validator, "wallet", None) else "local")
    path = audit_dir / f"{round_id}.jsonl"
    return JsonlAuditLog(path, round_id=round_id, context={
        "validator_hotkey": vhk,
        "netuid": getattr(validator, "netuid", None),
        "network": getattr(validator, "network", ""),
        "mode": getattr(validator, "mode", ""),
        "backend": getattr(validator, "backend", ""),
        "cloud_resource": getattr(validator, "cloud_resource", ""),
    })


def _validator_fingerprint(validator) -> dict:
    root = Path(__file__).resolve().parents[2]
    git = {}
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=3)
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(root),
            capture_output=True, text=True, timeout=3)
        git = {
            "commit": rev.stdout.strip() if rev.returncode == 0 else "",
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except Exception as e:
        git = {"error": str(e)}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "repo_root": str(root),
        "git": git,
        "config": {
            "netuid": getattr(validator, "netuid", None),
            "network": getattr(validator, "network", ""),
            "mode": getattr(validator, "mode", ""),
            "backend": getattr(validator, "backend", ""),
            "cloud_resource": getattr(validator, "cloud_resource", ""),
            "timeout_s": getattr(validator, "submission_timeout", None),
            "round_fee_rao": getattr(validator, "round_fee_rao", None),
            "set_weights_enabled": getattr(validator, "set_weights_enabled", False),
            "rounds_dir": getattr(validator, "rounds_dir", ""),
            "audit_dir": getattr(validator, "audit_dir", ""),
        },
    }


def _start_validator_heartbeat(validator, audit, round_start: float, clock) -> threading.Event:
    stop = threading.Event()
    interval = _audit_heartbeat_interval()

    def loop() -> None:
        seq = 0
        while not stop.wait(interval):
            seq += 1
            orch = getattr(validator, "_cloud_orch", None)
            cloud_status = {}
            if orch is not None and hasattr(orch, "get_status"):
                try:
                    cloud_status = orch.get_status()
                except Exception as e:
                    cloud_status = {"error": str(e), "error_type": type(e).__name__}
            audit_emit(audit, "validator.heartbeat",
                       heartbeat_seq=seq,
                       elapsed_s=max(0.0, clock() - round_start),
                       cloud_status=cloud_status,
                       message=f"validator alive elapsed={max(0.0, clock() - round_start):.0f}s")

    t = threading.Thread(target=loop, daemon=True,
                         name=f"sn125-validator-audit-{audit.round_id[:16]}")
    t.start()
    return stop


def _checkpoint_inventory(curve_data: dict[str, dict]) -> list[dict]:
    out = []
    for hotkey, cd in (curve_data or {}).items():
        records = list(cd.get("checkpoints") or [])
        if not records:
            for task_id, curve in (cd.get("curves") or {}).items():
                for rec in curve.get("checkpoints", []) or []:
                    records.append({"task_id": task_id, **rec})
        for rec in records:
            item = dict(rec)
            item["hotkey"] = hotkey
            p = item.get("uri") or item.get("path") or item.get("file")
            if p and "byte_size" not in item and "bytes" not in item:
                try:
                    path = Path(str(p))
                    if path.exists() and path.is_file():
                        item["byte_size"] = path.stat().st_size
                except Exception:
                    pass
            out.append(item)
    out.sort(key=lambda r: (r.get("hotkey", ""), r.get("pct", 0), r.get("step", 0)))
    return out


def _active_axons(meta, my_hk) -> list:
    """Axons of registered miners (skip empty IPs and our own hotkey)."""
    n = int(meta.n) if hasattr(meta.n, "item") else len(meta.neurons)
    return [meta.neurons[uid].axon_info for uid in range(n)
            if meta.neurons[uid].axon_info.ip != "0.0.0.0"
            and meta.neurons[uid].axon_info.hotkey != my_hk]


def _round_config(validator, round_id: str, fee_rao: int,
                  commit_window_s: float, reveal_window_s: float) -> RoundConfig:
    """Published per-round metadata (dashboard §9.1). The rung scale/horizon come
    from the first configured task; values are display/audit only — the loop logic
    does not depend on them."""
    tasks = getattr(validator, "tasks", None) or []
    task = tasks[0] if tasks else None
    return RoundConfig(
        round_id=round_id,
        scale=getattr(task, "parameter_count", "") or "unknown",
        horizon_steps=int(getattr(task, "total_steps", 0) or 0),
        task_family=getattr(validator, "dataset_name", "fineweb-edu"),
        gpu_sku=getattr(validator, "cloud_resource", "") or "b200-small",
        fee_rao=fee_rao,
        commit_window_s=commit_window_s,
        reveal_window_s=reveal_window_s,
    )


def _env_or_attr_float(validator, attr: str, env: str, default: float) -> float:
    """Operator knob: ``validator.<attr>`` if set, else ``$<env>``, else default."""
    val = getattr(validator, attr, None)
    if val is None:
        raw = os.environ.get(env, "").strip()
        if raw:
            try:
                val = float(raw)
            except ValueError:
                log.warning("%s=%r is not a number; using %s", env, raw, default)
                val = None
    return float(default if val is None else val)


def _launch_window_s(validator) -> float:
    """How long after selection the round keeps trying to START evaluations
    (``fsm_launch_window_s`` / ``SN125_LAUNCH_WINDOW_S``, default 4h). Sporadic
    B200 inventory inside the window is picked up; work still un-started at the
    end carries to the next round with its credit."""
    return max(0.0, _env_or_attr_float(validator, "fsm_launch_window_s",
                                       "SN125_LAUNCH_WINDOW_S", DEFAULT_LAUNCH_WINDOW_S))


def _launch_retry_s(validator) -> float:
    """Inventory poll cadence inside the launch window (``fsm_launch_retry_s`` /
    ``SN125_LAUNCH_RETRY_S``, default 300s)."""
    return max(0.0, _env_or_attr_float(validator, "fsm_launch_retry_s",
                                       "SN125_LAUNCH_RETRY_S", DEFAULT_LAUNCH_RETRY_S))


def _reveal_retry_s(validator) -> float:
    """Reveal re-query cadence across the reveal window (``fsm_reveal_retry_s`` /
    ``SN125_REVEAL_RETRY_S``, default 300s; 0 = single query at window open)."""
    return max(0.0, _env_or_attr_float(validator, "fsm_reveal_retry_s",
                                       "SN125_REVEAL_RETRY_S", DEFAULT_REVEAL_RETRY_S))


_REFINER_CFG_FROM_VALIDATOR = object()


def _make_cloud_batch(validator, round_id: str, sink: dict[str, dict],
                      max_concurrent: int, audit=None,
                      baseline_bundle: dict | None = None,
                      refiner_cfg=_REFINER_CFG_FROM_VALIDATOR):
    """Build ``evaluate_batch(payloads) -> {commit_hash: EvalResult}`` that fans the
    <=8 selected submissions across a thread pool (each call to the orchestrator's
    ``evaluate_submission`` acquires/releases its own rental slot, SPEC §4.4). The
    raw cloud dict per commit is stashed in ``sink`` so the loop can rebuild full
    ``ScoreRecord``s (components/curves) for ``compute_weights`` afterwards.
    ``refiner_cfg`` is the refiner config in force for the round (None: refiner
    envelopes are not routed); it defaults to the validator's."""
    orch = validator._cloud_orch
    mode = getattr(validator, "mode", "prod")
    if refiner_cfg is _REFINER_CFG_FROM_VALIDATOR:
        refiner_cfg = getattr(validator, "_refiner_cfg", None)
    poll_s = max(1.0, _launch_retry_s(validator))
    stop = threading.Event()   # set on urgent restart: workers stop polling

    def _has_capacity() -> bool:
        probe = getattr(orch, "has_b200_capacity", None)
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception:
            return False

    def _wait_for_box(commit_hash: str, sub_uid: str, deadline: float | None) -> bool:
        """After a no-box failure: poll B200 inventory every ``poll_s`` until a slot
        appears (True) or the launch window closes / restart requested (False).
        This is what lets a submission pick up a GPU that shows up at hour three."""
        if deadline is None:
            return False
        while not stop.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            if _has_capacity():
                return True
            stop.wait(min(poll_s, remaining))
        return False

    def _attempt(commit_hash: str, payload: bytes, source: str, sub_uid: str,
                 attempt: int):
        started = time.time()
        rf = refiner_job_for(payload, refiner_cfg)
        if rf is not None:
            return _attempt_refiner(commit_hash, payload, rf[0], rf[1], sub_uid, attempt, started)
        audit_emit(audit, "cloud_eval.start", commit_hash=commit_hash,
                   sub_uid=sub_uid, mode=mode, attempt=attempt,
                   resource=getattr(validator, "cloud_resource", ""),
                   **payload_record(payload))
        try:
            try:
                res = orch.evaluate_submission(source, round_id, sub_uid,
                                               mode=mode, audit=audit,
                                               baseline_bundle=baseline_bundle)
            except TypeError as e:
                if ("unexpected keyword argument 'audit'" not in str(e)
                        and "unexpected keyword argument 'baseline_bundle'" not in str(e)):
                    raise
                try:
                    res = orch.evaluate_submission(source, round_id, sub_uid,
                                                   mode=mode, audit=audit)
                except TypeError as e2:
                    if "unexpected keyword argument 'audit'" not in str(e2):
                        raise
                    res = orch.evaluate_submission(source, round_id, sub_uid, mode=mode)
        except Exception as e:  # cloud layer already retried; a raise == our fault
            audit_emit(audit, "cloud_eval.exception", commit_hash=commit_hash,
                       sub_uid=sub_uid, mode=mode, elapsed_s=time.time() - started,
                       error=str(e), error_type=type(e).__name__,
                       crashed=looks_like_crash(str(e)))
            return commit_hash, None, EvalResult("infra_dq", reason=f"cloud raised: {e}")
        verdict = classify_cloud_result(res)
        err = str(res.get("error", ""))
        audit_emit(audit, "cloud_eval.result", commit_hash=commit_hash,
                   sub_uid=sub_uid, mode=mode, attempt=attempt,
                   elapsed_s=time.time() - started,
                   outcome=verdict.outcome, score=verdict.score,
                   reason=verdict.reason, failed=bool(res.get("failed")),
                   error=err, crashed=looks_like_crash(err),
                   raw_result=res)
        return commit_hash, res, verdict

    def _attempt_refiner(commit_hash: str, payload: bytes, source: str, job: dict,
                         sub_uid: str, attempt: int, started: float):
        """Refiner-task submission: same orchestrator, same box, a different
        evaluator on it (``refiner-eval``). The rolling-best optimizer baseline
        is not shipped; the job carries the pinned identity baseline instead."""
        audit_emit(audit, "cloud_eval.start", commit_hash=commit_hash, sub_uid=sub_uid,
                   mode=mode, attempt=attempt, task_id=job["task_id"],
                   resource=getattr(validator, "cloud_resource", ""), **payload_record(payload))
        try:
            res = orch.evaluate_submission(source, round_id, sub_uid, mode=mode, audit=audit,
                                           refiner_job=job)
        except Exception as e:
            audit_emit(audit, "cloud_eval.exception", commit_hash=commit_hash, sub_uid=sub_uid,
                       mode=mode, task_id=job["task_id"], elapsed_s=time.time() - started,
                       error=str(e), error_type=type(e).__name__, crashed=looks_like_crash(str(e)))
            return commit_hash, None, EvalResult("infra_dq", reason=f"cloud raised: {e}")
        res.setdefault("task_id", job["task_id"])
        verdict = classify_cloud_result(res)
        err = str(res.get("error", ""))
        audit_emit(audit, "cloud_eval.result", commit_hash=commit_hash, sub_uid=sub_uid,
                   mode=mode, attempt=attempt, task_id=job["task_id"],
                   elapsed_s=time.time() - started, outcome=verdict.outcome, score=verdict.score,
                   reason=verdict.reason, failed=bool(res.get("failed")), error=err,
                   crashed=looks_like_crash(err), raw_result=res)
        return commit_hash, res, verdict

    def _one(item: tuple[str, bytes], deadline: float | None = None):
        """Run one submission; inside the launch window a *no-box* failure (zero
        inventory, rental never provisioned) is not final: wait for a slot and
        launch again. Failures with a box (miner code ran) are returned as-is."""
        commit_hash, payload = item
        source = payload.decode("utf-8", errors="replace")
        sub_uid = commit_hash[:18]
        attempt = 0
        while True:
            attempt += 1
            commit_hash, raw, ev = _attempt(commit_hash, payload, source, sub_uid, attempt)
            if (ev.outcome == "flake" and raw is not None and no_box_ran(raw)
                    and deadline is not None and time.time() < deadline):
                audit_emit(audit, "cloud_eval.awaiting_box", commit_hash=commit_hash,
                           sub_uid=sub_uid, attempt=attempt, reason=ev.reason,
                           launch_deadline=deadline)
                log.info("    %s: no GPU on attempt %d (%s); polling inventory "
                         "until %s", sub_uid, attempt, ev.reason[:60],
                         time.strftime("%H:%M:%S", time.localtime(deadline)))
                if _wait_for_box(commit_hash, sub_uid, deadline):
                    continue
            return commit_hash, raw, ev

    def evaluate_batch(payloads: list[tuple[str, bytes]],
                       launch_deadline: float | None = None) -> dict[str, EvalResult]:
        out: dict[str, EvalResult] = {}
        if not payloads:
            return out
        workers = min(max_concurrent, len(payloads))
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            for commit_hash, raw, ev in ex.map(
                    lambda item: _one(item, launch_deadline), payloads):
                if raw is not None:
                    sink[commit_hash] = raw
                out[commit_hash] = ev
        except BaseException:
            # Urgent restart (SIGINT) while boxes are running: do NOT wait for
            # the in-flight evaluations (hours) — abandon them so the round can
            # be recorded as aborted and the state checkpoint written now.
            stop.set()
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
        return out

    return evaluate_batch


def _make_capacity_check(validator):
    """Build the driver's ``check_capacity() -> bool`` from the cloud orchestrator
    (B200-only). Returns None when no orchestrator is wired (local/MockChain
    dry-runs) so the capacity gate is a no-op there."""
    orch = getattr(validator, "_cloud_orch", None)
    if orch is None or not hasattr(orch, "has_b200_capacity"):
        return None
    return orch.has_b200_capacity


def _make_capacity_wait(validator, round_id: str):
    """Build the driver's ``await_capacity(emit)`` — the long B200 capacity delay:
    pause the round, poll inventory on a backoff, KEEP WAITING (never substitute),
    publish the public dashboard banner, and resume when B200 returns. None when no
    orchestrator is wired (the gate is a no-op for local dry-runs)."""
    orch = getattr(validator, "_cloud_orch", None)
    if orch is None or not hasattr(orch, "wait_for_b200_capacity"):
        return None

    def await_capacity(emit, max_wait_s: float | None = None):
        """``max_wait_s`` (the driver passes the remaining launch window) bounds
        the wait; the orchestrator raises ``CapacityWait`` when it expires and the
        driver then carries the un-started submissions to the next round."""
        orch.begin_capacity_wait(round_id)
        try:
            if max_wait_s is not None and max_wait_s > 0:
                orch.wait_for_b200_capacity(on_event=emit, max_wait_s=float(max_wait_s))
            else:
                orch.wait_for_b200_capacity(on_event=emit)
        finally:
            orch.clear_capacity_wait()

    return await_capacity


def _eval_provenance(raw: dict) -> dict:
    """Public-safe hardware provenance for the round artifact: which provider/
    box produced this score and the throughput-band evidence (pre/post probe,
    achieved rate). Every published score is auditable against the hardware
    that produced it. Only scalar, non-sensitive fields are copied — never
    box-local paths or raw fingerprint dumps."""
    out: dict = {}
    for k in ("provider", "resource", "achieved_steps_per_s", "box_seconds"):
        v = raw.get(k)
        if v is not None:
            out[k] = v
    probe = raw.get("box_probe") or {}
    if isinstance(probe, dict) and probe:
        out["box_probe"] = {k: probe[k] for k in
                            ("steps_per_s", "gpu", "floor", "ceiling", "band")
                            if probe.get(k) is not None}
    post = raw.get("box_probe_post") or {}
    if isinstance(post, dict) and post:
        out["box_probe_post"] = {k: post[k] for k in
                                 ("steps_per_s", "drift_vs_pre", "tolerance_pct")
                                 if post.get(k) is not None}
    return out


def _build_results(fsm: RoundFSM, sink: dict[str, dict]):
    """From the SCORED submissions build the (results, sources, curve_data) triple
    that ``compute_weights`` / ``_save_round`` / ``_publish_attestation`` consume."""
    from ..training import ScoreRecord  # lazy: keep this module torch-free at load

    results: dict[str, ScoreRecord] = {}
    sources: dict[str, str] = {}
    curve_data: dict[str, dict] = {}
    # One row per hotkey (the reward path is hotkey-keyed). A hotkey can have
    # more than one SCORED submission in a round (a carried-over deferral plus
    # a fresh commit), so keep its BEST-scoring run deterministically (ties →
    # lowest commit hash) instead of letting dict insertion order pick one and
    # silently mismatch score/source/code_hash.
    # Refiner-task submissions are settled in their own block (refiner_round):
    # their score is a suite-score delta, not something this frontier can rank.
    scored = [s for s in fsm.submissions.values()
              if s.status is SubStatus.SCORED and not is_refiner_payload(s.payload)]
    scored.sort(key=lambda s: (s.hotkey, -(s.score if s.score is not None else float("-inf")),
                               s.commit_hash))
    chosen: dict[str, object] = {}
    for sub in scored:
        if sub.hotkey in chosen:
            log.warning("  %s: dropping extra scored commit %s (kept %s)", sub.hotkey[:16],
                        sub.commit_hash[:12], chosen[sub.hotkey].commit_hash[:12])
            continue
        chosen[sub.hotkey] = sub
    for sub in chosen.values():
        raw = sink.get(sub.commit_hash, {})
        results[sub.hotkey] = ScoreRecord(
            sub.score, raw.get("components", {}) or {},
            raw.get("task_scores", {}) or {},
            raw.get("best_hparams", {}) or {})
        if sub.payload is not None:
            sources[sub.hotkey] = sub.payload.decode("utf-8", errors="replace")
        cd = raw.get("curve_data") or {}
        provenance = _eval_provenance(raw)
        if cd or provenance:
            cd = dict(cd)
            if provenance:
                cd["provenance"] = provenance
            curve_data[sub.hotkey] = cd
    return results, sources, curve_data


def _pending_confirmation_jobs(validator, audit=None) -> list[tuple[str, bytes, dict]]:
    """(confirm-key, payload, pending-record) for every frontier candidate
    awaiting its validator-funded confirmation run (see
    ``Validator.compute_weights``). Validators without the confirmation path
    (test doubles) yield nothing."""
    loader = getattr(validator, "_load_pending_confirmations", None)
    if loader is None or int(getattr(validator, "frontier_confirmation_runs", 0) or 0) <= 0:
        return []
    jobs = []
    for p in loader():
        code_hash = str(p.get("code_hash", ""))
        src = validator._pending_source(p)
        if not src:
            audit_emit(audit, "confirmation.source_missing", code_hash=code_hash,
                       hotkey=p.get("hotkey", ""), submission_round_id=p.get("source_round_id", ""))
            log.error("  confirmation: no source on disk for %s (%s)", code_hash[:16], p.get("hotkey", ""))
            continue
        jobs.append((validator._confirmation_key(code_hash), src.encode("utf-8"), p))
    return jobs


def _build_confirmations(jobs, verdicts: dict, sink: dict[str, dict]):
    """Translate confirmation-run verdicts into the ``confirmations`` argument
    of ``compute_weights`` ({code_hash: ScoreRecord | {outcome, reason}}) and the
    round-JSON ``confirmations`` section."""
    from ..training import ScoreRecord  # lazy: keep this module torch-free at load

    confirmations: dict = {}
    records: dict[str, dict] = {}
    for key, _payload, p in jobs:
        code_hash = str(p.get("code_hash", ""))
        ev = verdicts.get(key) or EvalResult("infra_dq", reason="confirmation returned no result")
        raw = sink.get(key) or {}
        rec = {"hotkey": p.get("hotkey", ""), "code_hash": code_hash,
               "submission_round_id": p.get("source_round_id", ""),
               "outcome": ev.outcome, "reason": ev.reason}
        if ev.outcome == "scored":
            sr = ScoreRecord(ev.score, raw.get("components", {}) or {},
                             raw.get("task_scores", {}) or {},
                             raw.get("best_hparams", {}) or {})
            confirmations[code_hash] = sr
            rec["score"] = asdict(sr)
            cd = raw.get("curve_data") or {}
            curves = (cd.get("curves") or {})
            if curves:
                rec["curves"] = curves
            prov = _eval_provenance(raw)
            if prov:
                rec["provenance"] = prov
        else:
            confirmations[code_hash] = {"outcome": ev.outcome, "reason": ev.reason}
        records[key] = rec
    return confirmations, records


def _build_refiner_confirmations(jobs, verdicts: dict, sink: dict[str, dict], provenance=None):
    """Translate refiner confirmation-run verdicts into the ``confirmations``
    argument of ``refiner_frontier.settle`` ({commit_hash: {outcome, reason,
    result}}) and the block's ``confirmations`` section (one record per run,
    with the suite score, gates and signature the replay needs)."""
    from .refiner_round import _RESULT_KEYS, refiner_signature
    confirmations: dict = {}
    records: dict[str, dict] = {}
    for key, _payload, p in jobs:
        ch = str(p.get("commit_hash", ""))
        ev = verdicts.get(key) or EvalResult("infra_dq", reason="confirmation returned no result")
        raw = sink.get(key) or {}
        rec = {"hotkey": p.get("hotkey", ""), "commit_hash": ch,
               "submission_round_id": p.get("source_round_id", ""),
               "outcome": ev.outcome, "reason": ev.reason}
        if ev.outcome == "scored":
            rec["score"] = ev.score
            s = raw.get("submission") or {}
            rec["bench"] = {"submission": (s.get("bench") or {}).get("score"),
                            "baseline": ((raw.get("baseline") or {}).get("bench") or {}).get("score"),
                            "submission_checkpoint_sha256": (s.get("bench") or {}).get("checkpoint_sha256")}
            for k in _RESULT_KEYS:
                if raw.get(k) is not None:
                    rec[k] = raw[k]
            rec["signature"] = refiner_signature(raw)
            if provenance is not None:
                prov = provenance(raw)
                if prov:
                    rec["provenance"] = prov
            confirmations[ch] = {"outcome": "scored", "reason": "", "result": rec}
        else:
            confirmations[ch] = {"outcome": ev.outcome, "reason": ev.reason, "result": None}
        records[key] = rec
    return confirmations, records


def run_fsm_validator(validator, *, max_rounds: int | None = None,
                      clock=time.time, sleep=time.sleep) -> None:
    """Drive the validator through the FSM round loop. Runs forever in production
    (``max_rounds=None``); ``max_rounds`` + injected ``clock``/``sleep`` make it a
    finite, deterministic dry-run for tests.

    Requires the validator to be FSM-configured: ``payment_registry`` (a
    ``PaymentRegistry`` over a real or mock ``ChainView``), ``treasury_coldkey``,
    ``round_fee_rao``, and the ``commit_synapse_cls`` / ``submission_synapse_cls``
    transport classes. Optional knobs: ``fsm_commit_window_s``,
    ``fsm_reveal_window_s``, ``fsm_round_period_s``, ``fsm_max_concurrent``,
    ``fsm_fairness_window``, ``fsm_launch_window_s`` (or ``$SN125_LAUNCH_WINDOW_S``;
    how long the round keeps trying to start evaluations on sporadic GPU
    inventory, default 4h), ``fsm_launch_retry_s`` (inventory poll cadence).
    """
    reg = validator.payment_registry
    if not hasattr(validator, "_refiner_cfg"):
        # Refiner task (Challenge 2): None unless SN125_REFINER_TASK is set.
        # Raises (the validator does not start) on a half-configured task.
        validator._refiner_cfg = load_refiner_round_config()
    fee_rao = int(validator.round_fee_rao)
    if fee_rao <= 0:
        raise ValueError("round_fee_rao must be positive for the FSM loop")
    commit_window_s = float(getattr(validator, "fsm_commit_window_s", DEFAULT_COMMIT_WINDOW_S))
    reveal_window_s = float(getattr(validator, "fsm_reveal_window_s", DEFAULT_REVEAL_WINDOW_S))
    round_period_s = float(getattr(validator, "fsm_round_period_s", DEFAULT_ROUND_PERIOD_S))
    base_max_concurrent = int(getattr(validator, "fsm_max_concurrent", MAX_EVAL_PER_ROUND))

    def _affordable_now() -> int:
        """Per-round concurrency: the configured cap, clamped to what the daily
        spend cap still admits today (spent + reserved), so extra selections
        defer instead of being refused at rental time."""
        orch = getattr(validator, "_cloud_orch", None)
        if orch is None or not hasattr(orch, "affordable_concurrency"):
            return base_max_concurrent
        try:
            affordable = int(orch.affordable_concurrency())
        except Exception as e:  # pricing lookup failure must not stop the loop
            log.warning("affordable_concurrency failed (%s); keeping %d", e, base_max_concurrent)
            return base_max_concurrent
        if affordable < base_max_concurrent:
            log.warning("daily spend cap admits %d concurrent evals (cap %d); "
                        "extra selections defer to the next round", affordable, base_max_concurrent)
        return max(1, min(base_max_concurrent, affordable))
    fairness_window = int(getattr(validator, "fsm_fairness_window", DEFAULT_FAIRNESS_WINDOW))
    commit_cls = validator.commit_synapse_cls
    submission_cls = validator.submission_synapse_cls

    vhk = (validator.wallet.hotkey.ss58_address[:8]
           if getattr(validator, "wallet", None) else "local")

    carryover: list = []
    recent_rounds: list[set[str]] = []   # last `fairness_window` rounds' evaluated hotkeys
    round_num = 0
    state_path = _state_path_for(validator)
    restored = load_loop_state(state_path) if state_path is not None else None
    if restored is not None:
        round_num = restored["round_num"]
        carryover = restored["carryover"]
        recent_rounds = restored["recent_rounds"][-fairness_window:] if fairness_window > 0 else []
        if restored["last_weights"] and not (getattr(validator, "_last_weights", None) or {}):
            try:
                validator._last_weights = dict(restored["last_weights"])
            except Exception:
                pass
        log.info("resumed validator state from %s: next round #%d, %d deferred "
                 "submission(s) carried over, %d weight(s) for the epoch refresh "
                 "(last round %s)", state_path, round_num, len(carryover),
                 len(restored["last_weights"]), restored.get("last_round_id"))
    restart_reason: str | None = None
    weight_refresh_stop = None
    if getattr(validator, "set_weights_enabled", False):
        weight_refresh_stop = _start_weight_refresh(validator)
    publisher = _start_artifact_publisher(validator)
    payment_watch_stop = _start_payment_watcher(validator, reg, fee_rao)
    # Ledger reconciliation before anyone commits again: credits a previous
    # process could not settle correctly are returned here, once, as ledger
    # events (see _reconcile_ledger_at_startup for the rules and the record).
    _reconcile_ledger_at_startup(validator, reg)
    rounds_this_process = 0     # max_rounds counts THIS process's rounds, not round_num
    # Round schedule (config.ROUND_GRID_ANCHOR_TIMESTAMP): rounds open on the
    # 24 h grid anchored on the launch instant when they can, and no round
    # opens so late before the launch that it would still be running at it.
    grid_anchor = int(getattr(validator, "round_grid_anchor", 0) or 0)
    refiner_start = int(getattr(validator, "refiner_start_timestamp", 0) or 0)
    while max_rounds is None or rounds_this_process < max_rounds:
        hold_until = round_hold_until(clock(), round_period_s, grid_anchor)
        if hold_until > clock():
            log.info("  holding the next round for the launch instant %s (in %.0fs); "
                     "payments, weight refresh and publishing keep running",
                     _utc(hold_until), hold_until - clock())
            while hold_until - clock() > 0:
                sleep(min(hold_until - clock(), _BOUNDARY_POLL_S))
                if _restart_requested(validator):
                    restart_reason = "restart"
                    break
            if restart_reason is not None:
                log.info("  restart requested while holding for the launch instant; "
                         "stopping before round #%d (state checkpointed to %s)",
                         round_num, state_path)
                break
        round_start = clock()
        round_id = f"round_{round_num:06d}_{int(round_start)}_{vhk}"
        # Challenge 2 is live from the round that opens at its start instant.
        rf_cfg = _refiner_cfg_for_round(validator, round_start)
        rf_not_yet = (rf_cfg is None and getattr(validator, "_refiner_cfg", None) is not None
                      and refiner_start > round_start)
        operator_interrupt: BaseException | None = None
        nothing_launched = False     # every selected commit carried for want of a GPU
        audit = _make_round_audit(validator, round_id)
        heartbeat_stop = _start_validator_heartbeat(validator, audit, round_start, clock)
        log.info("=" * 60)
        log.info("  FSM %s", round_id)
        max_concurrent = _affordable_now()
        chain_block = _chain_block(validator)      # the block the round opened at
        try:
            audit_emit(audit, "round.start",
                       round_num=round_num, round_start=round_start,
                       chain_block=chain_block,
                       round_grid_anchor=grid_anchor,
                       refiner_live=rf_cfg is not None,
                       refiner_start_timestamp=refiner_start,
                       commit_window_s=commit_window_s,
                       reveal_window_s=reveal_window_s,
                       round_period_s=round_period_s,
                       max_concurrent=max_concurrent,
                       fee_rao=fee_rao,
                       validator_fingerprint=_validator_fingerprint(validator))
            if rf_not_yet:
                log.info("  refiner task (Challenge 2) opens at %s; this round is Challenge 1 only",
                         _utc(refiner_start))
            meta = validator.sync_metagraph()
            my_hk = (validator.wallet.hotkey.ss58_address
                     if getattr(validator, "wallet", None) else None)
            axons = _active_axons(meta, my_hk)
            log.info("  active miners: %d", len(axons))
            audit_emit(audit, "metagraph.synced",
                       active_miner_count=len(axons),
                       active_miners=[
                           {
                               "hotkey": getattr(ai, "hotkey", ""),
                               "ip": getattr(ai, "ip", ""),
                               "port": getattr(ai, "port", None),
                           }
                           for ai in axons
                       ])

            # Fee pin must publish BEFORE the commit window opens (§8); then credit
            # observed finalized treasury transfers (BittensorChainView in prod).
            # The background payment watcher keeps the ledger current between
            # rounds; this sync just closes the gap to the commit window.
            reg.pin_fee(round_id, fee_rao)
            granted = _sync_payments(reg)
            log.info("  fee pinned %d rao; synced %d treasury transfer(s); %d funded coldkey(s)",
                     fee_rao, granted, len(_registry_balances(reg)))
            audit_emit(audit, "payments.synced", fee_rao=fee_rao,
                       granted_transfers=granted,
                       balances=_registry_balances(reg),
                       ledger_path=str(getattr(reg, "store_path", "") or ""),
                       treasury_coldkey=getattr(validator, "treasury_coldkey", ""))

            cfg = _round_config(validator, round_id, fee_rao,
                                commit_window_s, reveal_window_s)
            fsm = RoundFSM(cfg, reg, clock)

            cloud_sink: dict[str, dict] = {}
            baseline_bundle = None
            # Any cloud backend (single provider or a failover chain like
            # "lambda,targon") needs the rolling-best baseline bundle shipped
            # to the box — without it prod-eval would retrain the AdamW
            # baseline on every rental, doubling the eval cost.
            _backend = str(getattr(validator, "backend", "")).strip().lower()
            if _backend not in ("", "local") and getattr(validator, "mode", "prod") == "prod":
                audit_emit(audit, "baseline.prepare.start",
                           baseline=getattr(validator, "baseline", ""))
                baselines, _step_times, _baseline_hp = validator.compute_baselines(round_id)
                if hasattr(validator, "_remote_baseline_bundle"):
                    baseline_bundle = validator._remote_baseline_bundle(baselines)
                audit_emit(audit, "baseline.prepare.finish",
                           task_ids=sorted((baseline_bundle or {}).get("baseline_curves", {}).keys()),
                           baseline=getattr(validator, "baseline", ""))
            recent_hotkeys = tuple(hk for r in recent_rounds for hk in r)

            # Frontier confirmation runs (validator-funded; see config
            # FRONTIER_CONFIRMATION_RUNS). They rent from the same pool and run
            # concurrently with the round's evals, so each one takes a slot
            # away from this round's selection instead of blowing the round
            # past its period.
            conf_jobs = _pending_confirmation_jobs(validator, audit)
            # Refiner-frontier confirmations ride the same batch: their payload
            # is the envelope, which _make_cloud_batch routes to refiner-eval.
            rf_conf_jobs = refiner_frontier.pending_jobs(getattr(validator, "rounds_dir", ""), rf_cfg)
            conf_sink: dict[str, dict] = {}
            conf_verdicts: dict = {}
            conf_thread = None
            if conf_jobs or rf_conf_jobs:
                audit_emit(audit, "confirmation.launch",
                           jobs=[{"key": k, "hotkey": p.get("hotkey", ""),
                                  "code_hash": p.get("code_hash", ""),
                                  "submission_round_id": p.get("source_round_id", "")}
                                 for k, _b, p in conf_jobs] +
                                [{"key": k, "hotkey": p.get("hotkey", ""), "task": "refiner",
                                  "commit_hash": p.get("commit_hash", ""),
                                  "submission_round_id": p.get("source_round_id", "")}
                                 for k, _b, p in rf_conf_jobs])
                log.info("  launching %d frontier confirmation run(s)%s", len(conf_jobs),
                         f" + {len(rf_conf_jobs)} refiner" if rf_conf_jobs else "")
                _conf_batch = _make_cloud_batch(validator, round_id, conf_sink,
                                                max_concurrent, audit,
                                                baseline_bundle=baseline_bundle,
                                                refiner_cfg=rf_cfg)
                _conf_payloads = [(k, b) for k, b, _p in conf_jobs] + [(k, b) for k, b, _p in rf_conf_jobs]

                def _run_confirmations():
                    try:
                        conf_verdicts.update(_conf_batch(_conf_payloads))
                    except Exception as e:  # never let a confirmation kill the round
                        audit_emit(audit, "confirmation.exception", error=str(e),
                                   error_type=type(e).__name__)
                        log.error("  confirmation batch raised: %s", e)

                conf_thread = threading.Thread(target=_run_confirmations,
                                               name="sn125-confirmations", daemon=True)
                conf_thread.start()
            round_max_eval = max(1, max_concurrent - len(conf_jobs) - len(rf_conf_jobs))
            outcome = drive_round(
                fsm,
                collect_commits=make_commit_collector(
                    validator.dendrite, axons, round_id, commit_cls,
                    on_event=lambda m: log.info("    %s", m),
                    audit=audit),
                collect_reveals=make_reveal_collector(
                    validator.dendrite, axons, round_id, submission_cls,
                    on_event=lambda m: log.info("    %s", m),
                    audit=audit),
                gate=make_source_gate(
                    refiner_enabled=rf_cfg is not None,
                    refiner_disabled_reason=(f"refiner task opens at {_utc(refiner_start)}"
                                             if rf_not_yet else None)),
                evaluate_batch=_make_cloud_batch(validator, round_id, cloud_sink,
                                                 max_concurrent, audit,
                                                 baseline_bundle=baseline_bundle,
                                                 refiner_cfg=rf_cfg),
                recent_hotkeys=recent_hotkeys,
                carryover=carryover,
                sleep_until=lambda ts: _sleep_until(
                    ts, clock, sleep, abort=lambda: _restart_now_requested(validator)),
                clock=clock,
                outage_backoff_s=float(getattr(validator, "fsm_outage_backoff_s",
                                               DEFAULT_OUTAGE_BACKOFF_S)),
                launch_window_s=_launch_window_s(validator),
                launch_retry_s=_launch_retry_s(validator),
                reveal_retry_s=_reveal_retry_s(validator),
                max_eval=round_max_eval,
                check_capacity=_make_capacity_check(validator),
                await_capacity=_make_capacity_wait(validator, round_id),
                on_event=lambda m: log.info("    %s", m),
                audit=audit,
            )
            outcome.report["opened_at"] = round_start
            outcome.report["opened_at_chain_block"] = chain_block
            outcome.report["refiner_live"] = rf_cfg is not None
            outcome.report["audit_log_path"] = str(audit.path)
            outcome.report["audit_text_log_path"] = str(audit.text_path)
            outcome.report["audit_aggregate_log_path"] = str(audit.aggregate_path)
            outcome.report["audit_aggregate_text_log_path"] = str(audit.aggregate_text_path)
            audit_emit(audit, "round.outcome",
                       selected=outcome.selected,
                       deferred=outcome.deferred,
                       scores=outcome.scores,
                       carryover=[
                           {"hotkey": s.hotkey, "commit_hash": s.commit_hash,
                            "deferrals": s.deferrals}
                           for s in outcome.carryover
                       ])
            carryover = outcome.carryover
            recent_rounds.append(set(outcome.scores.keys()))
            del recent_rounds[:-fairness_window]
            unlaunched = list(getattr(outcome, "unlaunched", []) or [])
            nothing_launched = bool(outcome.selected) and \
                len(unlaunched) >= len(outcome.selected)
            log.info("  round %s: %d scored, %d deferred%s",
                     round_id, len(outcome.scores), len(outcome.deferred),
                     f", {len(unlaunched)} never got a GPU (carried)" if unlaunched else "")

            results, sources, curve_data = _build_results(fsm, cloud_sink)
            refiner_block, refiner_sources = build_refiner_block(
                fsm, cloud_sink, rf_cfg,
                scored_status=SubStatus.SCORED, provenance=_eval_provenance)
            validator._last_refiner_block = refiner_block
            # Passed to _save_round / _publish_attestation only when there is
            # one, so the record (and every existing caller) is unchanged otherwise.
            rf_kw = {"refiner_block": refiner_block} if refiner_block else {}
            # Published sources = optimizer sources + refiner envelopes. The
            # frontier (compute_weights) only ever sees the optimizer ones.
            published_sources = {**sources, **refiner_sources} if refiner_sources else sources
            if refiner_block:
                audit_emit(audit, "refiner.results", block=refiner_block)
            confirmations: dict = {}
            confirmation_records: dict = {}
            rf_confirmations: dict = {}
            if conf_thread is not None:
                conf_thread.join()
                confirmations, confirmation_records = _build_confirmations(
                    conf_jobs, conf_verdicts, conf_sink)
                rf_confirmations, rf_conf_records = _build_refiner_confirmations(
                    rf_conf_jobs, conf_verdicts, conf_sink, provenance=_eval_provenance)
                if refiner_block is not None and rf_conf_records:
                    refiner_block["confirmations"] = rf_conf_records
                audit_emit(audit, "confirmation.results",
                           results={k: {kk: vv for kk, vv in r.items() if kk != "curves"}
                                    for k, r in confirmation_records.items()},
                           refiner_results={k: {kk: vv for kk, vv in r.items() if kk != "result"}
                                            for k, r in rf_conf_records.items()})
            checkpoints = _checkpoint_inventory(curve_data)
            audit_emit(audit, "checkpoints.inventory",
                       count=len(checkpoints), checkpoints=checkpoints)
            # Burn bootstrap: ALWAYS publish weights, even with zero scored
            # submissions. compute_weights({}) -> {BURN_KEY: 1.0} routes 100% of
            # emission to the burn/owner UID, so a fresh subnet with no eligible
            # miners still burns its emission. Gating weight-setting on `if
            # results:` (the old behavior) left emission un-set on a no-submission
            # round, which both fails the configured-to-100% burn path and decays vtrust.
            # compute_weights promotes its result to validator._last_weights,
            # which the background epoch refresh re-sets on chain. Snapshot the
            # previous (already durable) map so a failed save/attestation below
            # can roll the promotion back — otherwise the refresh thread would
            # keep pushing weights that never passed the durability gate.
            prev_weights = dict(getattr(validator, "_last_weights", {}) or {})
            try:
                weights = validator.compute_weights(results, sources=sources, curve_data=curve_data,
                                                    confirmations=confirmations,
                                                    round_id=round_id)
            except TypeError:
                # Test doubles / older wrappers may still expose the historical
                # compute_weights(results) signature.
                weights = validator.compute_weights(results)
            if refiner_block is not None:
                # Refiner frontier (Challenge 2): its own bar, confirmation and
                # progress-scaled pool, blended into the weight vector at the
                # configured emission share. Replayed by sn125/verify.py from
                # the block this settles into.
                now_rf = int(time.time())
                burn_floor_fn = getattr(validator, "_clamped_burn_fraction_floor", None)
                frontier = refiner_frontier.settle(
                    rf_cfg, refiner_block.get("submissions") or {}, rf_confirmations,
                    rounds_dir=getattr(validator, "rounds_dir", ""), round_id=round_id, now=now_rf,
                    burn_floor=burn_floor_fn(now_rf) if callable(burn_floor_fn) else 0.0,
                    launch_timestamp=int(getattr(validator, "launch_timestamp", 0) or 0))
                refiner_block["frontier"] = frontier
                weights = refiner_frontier.blend_weights(weights, frontier["weights"], frontier["emission_share"],
                                                         getattr(validator, "BURN_KEY", refiner_frontier.BURN_KEY))
                # compute_weights promoted its own map; the blended one is what
                # goes on chain and what the epoch refresh must re-set.
                validator._last_weights = dict(weights)
                audit_emit(audit, "refiner.frontier", round_id=round_id,
                           frontier={k: v for k, v in frontier.items() if k != "events"})
                if frontier["new_event"]:
                    log.info("  refiner frontier: %s confirmed +%.4f suite points (event %s)",
                             frontier["new_event"]["hotkey"][:16], frontier["new_event"]["improvement"],
                             frontier["new_event"]["event_id"])
                if frontier["emission_share"] > 0:
                    log.info("  refiner pool (%.0f%% of emission): payable %.3f, burned %.3f, bar %.4f",
                             100 * frontier["emission_share"], frontier["payable_share"],
                             frontier["burned_share"], frontier["min_improvement_threshold"])
            audit_emit(audit, "weights.computed", weights=weights,
                       scored_hotkeys=sorted(results),
                       frontier_rewards=getattr(validator, "_last_frontier_rewards", {}))
            mode = "LIVE" if getattr(validator, "set_weights_enabled", False) else "read-only"
            if results:
                log.info("  weights (%s): %d miner(s)", mode, len(weights))
            else:
                log.warning("  round %s: no scored submissions -> burning 100%% "
                            "to owner UID (%s)", round_id, mode)
            round_saved = False
            try:
                validator._save_round(round_id, results, weights, published_sources, curve_data,
                                      pause_reasons=outcome.pause_reasons,
                                      round_report=outcome.report,
                                      confirmation_records=confirmation_records, **rf_kw)
                round_saved = True
                audit_emit(audit, "round.saved", round_id=round_id,
                           rounds_dir=getattr(validator, "rounds_dir", ""))
            except Exception as e:
                audit_emit(audit, "round.save_failed", round_id=round_id,
                           error=str(e), error_type=type(e).__name__)
                log.error("  _save_round failed: %s", e)

            attestation_ok = False
            try:
                published = validator._publish_attestation(
                    round_id, results, weights, published_sources, curve_data,
                    pause_reasons=outcome.pause_reasons,
                    round_report=outcome.report, **rf_kw)
                attestation_ok = published is not False
                audit_emit(audit, "attestation.publish_called", round_id=round_id,
                           published=attestation_ok)
            except Exception as e:
                audit_emit(audit, "attestation.publish_failed", round_id=round_id,
                           error=str(e), error_type=type(e).__name__)
                log.error("  _publish_attestation failed: %s", e)

            # Set the (burn-inclusive) weights on chain every round only after the
            # round is durable locally and remotely. A live emission write without
            # a saved/auditable round is harder to reconstruct than skipping a
            # weight write and retrying next round.
            if getattr(validator, "set_weights_enabled", False):
                if not round_saved or not attestation_ok:
                    audit_emit(audit, "weights.set_skipped",
                               reason="missing_durable_round_or_attestation",
                               round_saved=round_saved, attestation_ok=attestation_ok,
                               weights=weights)
                    log.error("  refusing live set_weights: round_saved=%s attestation_ok=%s",
                              round_saved, attestation_ok)
                    # Roll back the _last_weights promotion so the background
                    # epoch refresh keeps re-setting the last DURABLE round's
                    # weights instead of this round's unattested map.
                    try:
                        validator._last_weights = prev_weights
                    except Exception:
                        pass
                    audit_emit(audit, "weights.refresh_rollback",
                               round_id=round_id, restored_weights=prev_weights)
                else:
                    try:
                        meta = validator.sync_metagraph()
                    except Exception as e:
                        audit_emit(audit, "metagraph.resync_failed",
                                   error=str(e), error_type=type(e).__name__)
                        log.warning("  metagraph refresh failed, using stale: %s", e)
                    validator._set_weights_on_chain(meta, weights)
                    audit_emit(audit, "weights.set_on_chain", weights=weights)
        except (RestartNow, KeyboardInterrupt) as e:
            # Urgent update: the supervisor asked for an immediate restart
            # (RestartNow from a window wait, or SIGINT while an evaluation
            # was blocking). The round is abandoned here; its debited credits
            # are returned by _refund_abandoned_rounds at the next start-up.
            if isinstance(e, KeyboardInterrupt) and not _restart_now_requested(validator):
                audit_emit(audit, "round.aborted", reason="interrupted by operator",
                           elapsed_s=clock() - round_start)
                log.warning("  round %s ABANDONED (operator interrupt); credits debited "
                            "in it are returned when the validator starts again", round_id)
                operator_interrupt = e          # re-raised after the checkpoint below
            else:
                audit_emit(audit, "round.aborted",
                           reason="immediate restart requested (update)",
                           elapsed_s=clock() - round_start)
                log.warning("  round %s ABANDONED for an immediate restart; credits "
                            "debited in it are returned when the validator starts again",
                            round_id)
                restart_reason = "restart"
        except Exception as e:
            audit_emit(audit, "round.exception", error=str(e),
                       error_type=type(e).__name__, traceback=traceback.format_exc())
            log.error("  FSM round failed: %s", e)
            traceback.print_exc()
        finally:
            audit_emit(audit, "round.finish",
                       elapsed_s=clock() - round_start)
            heartbeat_stop.set()
            audit.close()
            if publisher is not None:
                # Round boundary: sync the freshly closed audit/round files now
                # instead of waiting out the cadence timer.
                publisher.publish_now()
            round_num += 1
            rounds_this_process += 1
            # Round boundary = the one lossless restart point. Checkpoint the
            # inter-round state first, then honour a pending restart request,
            # then sleep to the 24h boundary (re-checking the request as we go).
            if state_path is not None:
                try:
                    save_loop_state(state_path, round_num=round_num, carryover=carryover,
                                    recent_rounds=recent_rounds,
                                    last_weights=getattr(validator, "_last_weights", None),
                                    last_round_id=round_id)
                except Exception as e:
                    log.error("  could not checkpoint validator state to %s: %s",
                              state_path, e)
            if restart_reason is not None or operator_interrupt is not None:
                pass                                  # abandoned round: leave now
            elif _restart_requested(validator):
                restart_reason = "restart"
            elif nothing_launched:
                # No GPU for the whole launch window and nothing ran: the ~20h
                # evaluation budget was never spent, so waiting out the rest of
                # the period would only keep the carried commits idle. Open the
                # next round (and its launch window) now — that is how sporadic
                # capacity gets picked up: a fresh window every few hours until
                # a box is found or the carry limit refunds the credit.
                log.info("  no GPU during the launch window; opening the next round now "
                         "(%d carried commit(s) keep their credits and priority)",
                         len(carryover))
            else:
                # The commit/reveal windows + the ~20h eval already consume most
                # of the period; this just tops up to the boundary: the next
                # grid line (config.next_round_start) or, without a grid, the
                # period after this round's start.
                next_start = next_round_start(round_start, clock(), round_period_s, grid_anchor)
                remaining = next_start - clock()
                if remaining > 0:
                    log.info("  next round in %.0fs (%s)", remaining, _utc(next_start))
                while remaining > 0:
                    sleep(min(remaining, _BOUNDARY_POLL_S))
                    if _restart_requested(validator):
                        restart_reason = "restart"
                        break
                    remaining = next_start - clock()
        if operator_interrupt is not None:
            _stop_background(weight_refresh_stop, payment_watch_stop, publisher)
            raise operator_interrupt
        if restart_reason is not None:
            log.info("  restart requested; stopping after round #%d "
                     "(state checkpointed to %s)", round_num - 1, state_path)
            break
    _stop_background(weight_refresh_stop, payment_watch_stop, publisher)
    if restart_reason is not None:
        try:
            validator.restart_requested = True
        except Exception:
            pass
    return restart_reason


def _stop_background(weight_refresh_stop, payment_watch_stop, publisher) -> None:
    if weight_refresh_stop is not None:
        weight_refresh_stop.set()
    if payment_watch_stop is not None:
        payment_watch_stop.set()
    if publisher is not None:
        try:
            publisher.stop(flush=True)
        except Exception as e:
            log.warning("artifact publisher shutdown failed: %s", e)


def _state_path_for(validator) -> Path | None:
    """``validator.state_path`` (``""``/None disables) else
    ``<audit_dir>/validator_state.json``."""
    if hasattr(validator, "state_path"):
        configured = getattr(validator, "state_path")
        return Path(configured) if configured else None
    try:
        return default_state_path(_audit_dir_for(validator))
    except Exception as e:
        log.warning("no validator state path (%s); inter-round state is memory-only", e)
        return None


def _restart_flag_path(validator) -> Path | None:
    configured = (getattr(validator, "restart_flag_path", None)
                  or os.environ.get(RESTART_FLAG_ENV, ""))
    return Path(configured) if configured else None


def _refiner_cfg_for_round(validator, round_start: float):
    """The refiner config in force for a round opening at ``round_start``: the
    validator's, or None before the Challenge-2 start instant
    (``validator.refiner_start_timestamp``, config.REFINER_START_TIMESTAMP)."""
    cfg = getattr(validator, "_refiner_cfg", None)
    start = int(getattr(validator, "refiner_start_timestamp", 0) or 0)
    if cfg is None or float(round_start) >= start:
        return cfg
    return None


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(ts)))


def _chain_block(validator) -> int | None:
    """Current chain block for the round record's audit trail, None when the
    validator has no chain connection (dry run, tests)."""
    sub = getattr(validator, "subtensor", None)
    if sub is None:
        return None
    try:
        return int(sub.get_current_block())
    except Exception as e:
        log.debug("chain block unavailable: %s", e)
        return None


def _restart_requested(validator) -> bool:
    """True when the supervisor has asked for a restart at the next safe point."""
    if bool(getattr(validator, "restart_requested", False)):
        return True
    flag = _restart_flag_path(validator)
    try:
        return flag is not None and flag.exists()
    except OSError:
        return False


RESTART_NOW_MARKER = "now"


def _restart_now_requested(validator) -> bool:
    """True when the supervisor asked for an IMMEDIATE restart (urgent update):
    the flag file carries a ``now`` line. The round in progress is abandoned;
    its debited credits are returned at the next start-up
    (:func:`_refund_abandoned_rounds`)."""
    if bool(getattr(validator, "restart_now_requested", False)):
        return True
    flag = _restart_flag_path(validator)
    if flag is None:
        return False
    try:
        lines = flag.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return RESTART_NOW_MARKER in lines


class RestartNow(Exception):
    """Raised inside a round when an immediate restart was requested."""


# Audit events that settle a commit's credit for good inside a round: a
# miner-fault outcome burns the credit by rule, so an abandoned round must
# not hand it back.
_BURNED_EVENTS = ("gate.dq", "submission.unrevealed")
_SETTLED_EVENTS = ("round.published", "round.saved", "weights.computed")


def _audit_round_status(audit_dir: Path, round_id: str) -> tuple[bool | None, set[str]]:
    """(settled?, hotkeys whose credit was burned by a miner-fault outcome)
    from the round's audit JSONL. A missing audit file => (None, set()): we
    have no evidence either way, so the caller leaves the round alone."""
    path = audit_dir / f"{round_id}.jsonl"
    settled, burned = False, set()
    if not path.exists():
        return None, burned
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                ev = rec.get("event")
                if ev in _SETTLED_EVENTS:
                    settled = True
                elif ev in _BURNED_EVENTS and rec.get("hotkey"):
                    burned.add(str(rec["hotkey"]))
                elif (ev == "evaluation.finalized" and rec.get("hotkey")
                      and rec.get("status") in (SubStatus.DQ.value,
                                                SubStatus.EXPIRED.value)):
                    burned.add(str(rec["hotkey"]))       # miner-fault DQ / expired: consumed
    except OSError:
        pass
    return settled, burned


NO_GPU_RESTITUTION_TAG = "no-GPU restitution"
RECONCILIATION_FILE = "payments_reconciliation.jsonl"


def _software_version() -> str:
    try:
        from ..neuron import SN125_VERSION   # lazy: neuron imports this module
        return str(SN125_VERSION)
    except Exception:
        return "unknown"


def _reconcile_ledger_at_startup(validator, reg) -> dict:
    """Run every startup ledger correction and leave one durable record of it.

    Credit lifecycle (the rules the corrections enforce):
      paid          TAO to the treasury -> ``grant`` (remainder carried per coldkey)
      accepted      ``debit``: the credit is HELD by that commit
      evaluated     scored or miner-fault DQ: the credit is consumed
      infra fault   box ran, our side failed: ``refund``
      no GPU        commit carried to the next round with the credit and FIFO
                    priority, round after round; after 30 carries and 30 days
                    it expires and the credit is absorbed
      abandoned     the round never published: every debit not burned by miner
                    fault is refunded (``_refund_abandoned_rounds``)
      misrecorded   a no-GPU failure a past version burned as miner fault:
                    restitution refund (``_refund_no_gpu_dqs``)

    Each pass is idempotent (it re-reads the ledger it appends to) so restarts
    are safe. The outcome is appended to ``<audit_dir>/payments_reconciliation
    .jsonl`` so the correction is visible in the published logbook even when
    nothing was owed."""
    before = len(getattr(reg, "events", []) or [])
    counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    for name, fn in (("abandoned_rounds", _refund_abandoned_rounds),
                     ("no_gpu_restitution", _refund_no_gpu_dqs)):
        try:
            counts[name] = int(fn(validator, reg) or 0)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            log.error("  ledger reconciliation (%s) failed: %s", name, e)
    new_events = list(getattr(reg, "events", []) or [])[before:]
    record = {
        "ts": time.time(),
        "version": _software_version(),
        "refunds": counts,
        "errors": errors,
        "events": [
            {"kind": ev.kind, "round_id": ev.round_id, "coldkey": ev.coldkey,
             "hotkey": getattr(ev, "hotkey", None), "credits": ev.credits,
             "note": ev.note}
            for ev in new_events
        ],
    }
    total = sum(counts.values())
    if total or errors:
        log.warning("  ledger reconciliation: %d credit(s) returned %s%s",
                    total, counts, f"; errors {errors}" if errors else "")
    else:
        log.info("  ledger reconciliation: nothing owed")
    try:
        audit_dir = _audit_dir_for(validator)
        audit_dir.mkdir(parents=True, exist_ok=True)
        with (audit_dir / RECONCILIATION_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as e:
        log.warning("  could not write %s: %s", RECONCILIATION_FILE, e)
    return record


def _no_gpu_dqs_in_audit(audit_dir: Path, round_id: str) -> list[tuple[str, str, str]]:
    """(coldkey, hotkey, reason) for every commit the round's audit finalized as
    a miner-fault DQ whose recorded reason says no box was ever rented
    (adapter.is_no_gpu_failure). Missing audit file => []."""
    path = audit_dir / f"{round_id}.jsonl"
    out: list[tuple[str, str, str]] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                if (rec.get("event") == "evaluation.finalized"
                        and rec.get("status") == SubStatus.DQ.value
                        and rec.get("hotkey") and rec.get("coldkey")
                        and is_no_gpu_failure(str(rec.get("reason") or ""))):
                    out.append((str(rec["coldkey"]), str(rec["hotkey"]),
                                str(rec.get("reason") or "")))
    except OSError:
        pass
    return out


def _refund_no_gpu_dqs(validator, reg) -> int:
    """Restitution for credits burned by a no-GPU failure misread as miner fault.

    Before 0.2.2 a ``CapacityWait`` raised inside ``evaluate_submission`` (zero
    B200 inventory at rental time) came back as a failed result whose error
    string matched no marker, so ``classify_cloud_result`` fell through to the
    ambiguous ``dq`` branch: the round recorded a miner-fault DQ and the credit
    was burned, although no box was ever rented and the miner's code never ran
    (round_000000_1788861194, 2026-09-08: two credits). This re-reads every
    round's audit with the current rule — a failure with no box is never the
    miner's — and returns one credit per such commit, once. Idempotent: the
    refund event names the hotkey, and a commit already made whole (by this
    or by the round itself) is skipped. Published round files are not
    rewritten; the ledger event is the record of the correction.
    """
    events = getattr(reg, "events", None)
    refund = getattr(reg, "refund_credit", None)
    if events is None or refund is None:
        return 0
    try:
        events = list(events)
    except TypeError:
        return 0
    audit_dir = _audit_dir_for(validator)
    debited: dict[tuple[str, str, str], int] = {}     # (round, coldkey, hotkey) -> debits
    refunded_for: dict[tuple[str, str, str], int] = {}
    refunds_untagged: dict[tuple[str, str], int] = {}  # (round, coldkey) -> hotkey-less refunds
    rounds: set[str] = set()
    for ev in events:
        if not ev.round_id or ev.coldkey is None:
            continue
        if ev.kind == "debit" and ev.hotkey:
            debited[(ev.round_id, ev.coldkey, ev.hotkey)] =                 debited.get((ev.round_id, ev.coldkey, ev.hotkey), 0) + 1
            rounds.add(ev.round_id)
        elif ev.kind == "refund":
            if ev.hotkey:
                refunded_for[(ev.round_id, ev.coldkey, ev.hotkey)] =                     refunded_for.get((ev.round_id, ev.coldkey, ev.hotkey), 0) + 1
            else:
                refunds_untagged[(ev.round_id, ev.coldkey)] =                     refunds_untagged.get((ev.round_id, ev.coldkey), 0) + 1
    returned = 0
    for round_id in sorted(rounds):
        for coldkey, hotkey, reason in _no_gpu_dqs_in_audit(audit_dir, round_id):
            key = (round_id, coldkey, hotkey)
            owed = debited.get(key, 0) - refunded_for.get(key, 0)
            if owed <= 0:
                continue
            # A hotkey-less refund in the same round for this coldkey (older
            # infra-DQ refunds) may already cover it; consume those first.
            spare = refunds_untagged.get((round_id, coldkey), 0)
            if spare > 0:
                refunds_untagged[(round_id, coldkey)] = spare - 1
                continue
            try:
                refund(coldkey, round_id,
                       f"{NO_GPU_RESTITUTION_TAG} {hotkey}: recorded as miner-fault DQ "
                       f"but no GPU was rented ({reason[:120]}); credit returned",
                       hotkey=hotkey)
            except TypeError:     # registry without the hotkey keyword
                refund(coldkey, round_id,
                       f"{NO_GPU_RESTITUTION_TAG} {hotkey}: recorded as miner-fault DQ "
                       f"but no GPU was rented ({reason[:120]}); credit returned")
            except Exception as e:
                log.error("  could not return no-GPU credit to %s for %s: %s",
                          coldkey, round_id, e)
                continue
            refunded_for[key] = refunded_for.get(key, 0) + 1
            returned += 1
            log.warning("  %s: %s was DQ'd for want of a GPU (%s); credit returned to %s",
                        round_id, hotkey[:12], reason[:60], coldkey[:12])
    return returned


def _refund_abandoned_rounds(validator, reg) -> int:
    """Return the credits debited in rounds that never published.

    A round is abandoned when the process stopped mid-round (urgent update,
    crash, operator kill) or the round raised before ``round.published``.
    Every commit accepted in it was debited one credit, so without this the
    miner pays for an evaluation that never happened. For each such round:
    net debits minus refunds already made per coldkey, excluding hotkeys the
    audit shows were burned by miner fault (gate DQ, unrevealed, eval DQ).
    Evidence-based: a round whose audit file is missing (audit dir moved or
    wiped) is left alone rather than guessed at — every round opens its audit
    file before the first debit, so an abandoned round always has one.
    Idempotent: refunds are ledger events, so a second start-up finds the
    net at zero. Runs once, before the first round of a process.
    """
    events = getattr(reg, "events", None)
    refund = getattr(reg, "refund_credit", None)
    if events is None or refund is None:
        return 0
    try:
        events = list(events)
    except TypeError:
        return 0
    audit_dir = _audit_dir_for(validator)
    rounds_dir = Path(getattr(validator, "rounds_dir", "") or "")
    by_round: dict[str, list] = {}
    for ev in events:
        if ev.round_id and ev.kind in ("debit", "refund"):
            by_round.setdefault(ev.round_id, []).append(ev)
    refunded = 0
    for round_id, evs in sorted(by_round.items()):
        if rounds_dir and (rounds_dir / f"{round_id}.json").exists():
            continue                               # published: outcomes are final
        settled, burned = _audit_round_status(audit_dir, round_id)
        if settled is None or settled:
            continue
        net: dict[str, int] = {}
        hotkeys: dict[str, set[str]] = {}
        returned_here = 0
        for ev in evs:
            if ev.kind == "debit":
                net[ev.coldkey] = net.get(ev.coldkey, 0) + 1
                if ev.hotkey:
                    hotkeys.setdefault(ev.coldkey, set()).add(ev.hotkey)
            else:
                net[ev.coldkey] = net.get(ev.coldkey, 0) - 1
        for coldkey, owed in sorted(net.items()):
            if owed <= 0 or hotkeys.get(coldkey, set()) & burned:
                continue
            for _ in range(owed):
                try:
                    refund(coldkey, round_id,
                           f"round {round_id} abandoned before publication "
                           f"(restart or crash mid-round): credit returned")
                    returned_here += 1
                except Exception as e:
                    log.error("  could not refund %s for abandoned %s: %s",
                              coldkey, round_id, e)
                    break
        if returned_here:
            refunded += returned_here
            log.warning("  round %s never published; returned %d credit(s) debited in it",
                        round_id, returned_here)
    return refunded


def _sync_payments(reg) -> int:
    """Catch the registry up to the finalized head (all chunks), or one plain
    ``sync`` for registries/test doubles without ``sync_to_head``."""
    to_head = getattr(reg, "sync_to_head", None)
    return int(to_head() if to_head is not None else reg.sync())


def _registry_balances(reg) -> dict:
    getter = getattr(reg, "balances", None)
    try:
        return dict(getter()) if getter is not None else {}
    except Exception:
        return {}


def _payment_scan_interval(validator) -> float:
    """Seconds between background treasury scans: ``validator.payment_scan_s``,
    else ``SN125_PAYMENT_SCAN_S``, else the watcher default. <= 0 disables."""
    from ..payments.watch import DEFAULT_SCAN_INTERVAL_S, SCAN_INTERVAL_ENV
    explicit = getattr(validator, "payment_scan_s", None)
    if explicit is not None:
        return float(explicit)
    return settings.env_float(SCAN_INTERVAL_ENV, DEFAULT_SCAN_INTERVAL_S)


def _start_payment_watcher(validator, reg, fee_rao: int):
    """Bring the credit ledger current BEFORE the first round and keep it
    current for the validator's lifetime.

    1. Record ``fee_rao`` as the fee in force so deposits can be credited
       before the first round pin (and after a restart with a new fee).
    2. Blocking catch-up: scan every finalized block since the persisted
       cursor (or the configured backfill start) so deposits made while the
       validator was down — or before this code existed — are credited before
       the first commit window opens.
    3. Start the background watcher (payments/watch.py).
    Returns the watcher's stop event, or None when disabled.
    """
    set_fee = getattr(reg, "set_grant_fee", None)
    if set_fee is not None:
        try:
            if set_fee(fee_rao):
                log.info("  payment grant fee in force: %d rao", fee_rao)
        except Exception as e:
            log.error("  could not record the payment grant fee: %s", e)
    try:
        granted = _sync_payments(reg)
        balances = _registry_balances(reg)
        log.info("  payment ledger current: %d new treasury transfer(s) credited at "
                 "start-up, %d funded coldkey(s), %d credit(s) outstanding",
                 granted, len(balances), sum(balances.values()))
    except Exception as e:
        # The watcher below retries; the round loop must still start.
        log.error("  start-up payment catch-up failed (watcher will retry): %s", e)
    interval = _payment_scan_interval(validator)
    if interval <= 0:
        log.warning("  continuous payment scanning DISABLED (interval %.0f)", interval)
        return None
    from ..payments.watch import start_payment_watcher
    stop = start_payment_watcher(reg, interval_s=interval)
    log.info("  continuous payment watcher started (every %.0fs)", interval)
    return stop


def _start_artifact_publisher(validator):
    """Start the continuous artifact publisher (rounds/audit/cloud status ->
    R2 + private HF). Best-effort by design: publishing must never affect the
    round loop, so construction/start failures degrade to a warning."""
    if str(getattr(validator, "network", "") or "").lower() == "mock":
        # Dry-runs (`validate --mock`, tests) must never mirror their synthetic
        # rounds into the operator's real bucket: cloud.py loads .env at import,
        # so os.environ carries live R2 credentials even in a test process.
        log.info("continuous artifact publisher disabled for the mock network")
        return None
    try:
        from ..publisher import ContinuousPublisher
        publisher = ContinuousPublisher.from_validator(validator)
        if publisher is not None:
            publisher.start()
        return publisher
    except Exception as e:
        log.warning("continuous artifact publisher unavailable: %s", e)
        return None


def _start_weight_refresh(validator) -> threading.Event:
    """Epoch-cadence weight refresh (~360 blocks): between 24h rounds, re-set the
    validator's last computed weight map so vtrust does not decay. Runs on real
    wall time in a daemon thread; a set-weights failure is logged and retried on
    the next tick. Started only when live weight-setting is enabled."""
    from ..neuron import WEIGHT_REFRESH_SECONDS
    stop = threading.Event()
    interval = settings.env_float(settings.WEIGHT_REFRESH_S_ENV, float(WEIGHT_REFRESH_SECONDS))

    def loop() -> None:
        while not stop.wait(interval):
            weights = dict(getattr(validator, "_last_weights", {}) or {})
            if not weights:
                continue
            try:
                meta = validator.sync_metagraph()
                validator._set_weights_on_chain(meta, weights)
                log.info("  [weight-refresh] re-set %d weight(s)", len(weights))
            except Exception as e:
                log.warning("  [weight-refresh] failed (retry next tick): %s", e)

    t = threading.Thread(target=loop, daemon=True, name="sn125-weight-refresh")
    t.start()
    return stop


def _sleep_until(ts, clock, sleep, abort=None) -> None:
    """Block (via the injected ``sleep``) until the injected ``clock`` reaches the
    FSM's pause-extended window deadline. Tests pass a fake clock+sleep so a 24h
    window elapses instantly and deterministically.

    ``abort`` (optional, polled every ``_BOUNDARY_POLL_S`` of the wait) raises
    :class:`RestartNow` when it returns True, so an urgent update does not have
    to wait out a 3h commit window."""
    if ts is None:
        return
    remaining = ts - clock()
    while remaining > 0:
        if abort is not None and abort():
            raise RestartNow("immediate restart requested during a window wait")
        sleep(remaining if abort is None else min(remaining, _BOUNDARY_POLL_S))
        remaining = ts - clock()
