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
from .adapter import (
    classify_cloud_result,
    make_commit_collector,
    make_reveal_collector,
    make_source_gate,
)
from .audit import JsonlAuditLog, audit_emit, looks_like_crash, payload_record
from .driver import DEFAULT_OUTAGE_BACKOFF_S, EvalResult, drive_round
from .round_fsm import MAX_EVAL_PER_ROUND, RoundConfig, RoundFSM, SubStatus
from .state import default_state_path, load_loop_state, save_loop_state

log = logging.getLogger("sn125.fsm")

RESTART_FLAG_ENV = "SN125_RESTART_FLAG"
EXIT_RESTART = 75
_BOUNDARY_POLL_S = 60.0

DEFAULT_COMMIT_WINDOW_S = 3 * 3600.0
DEFAULT_REVEAL_WINDOW_S = 1 * 3600.0
DEFAULT_ROUND_PERIOD_S = 24 * 3600.0
DEFAULT_FAIRNESS_WINDOW = 3


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


def _make_cloud_batch(validator, round_id: str, sink: dict[str, dict],
                      max_concurrent: int, audit=None,
                      baseline_bundle: dict | None = None):
    """Build ``evaluate_batch(payloads) -> {commit_hash: EvalResult}`` that fans the
    <=8 selected submissions across a thread pool (each call to the orchestrator's
    ``evaluate_submission`` acquires/releases its own rental slot, SPEC §4.4). The
    raw cloud dict per commit is stashed in ``sink`` so the loop can rebuild full
    ``ScoreRecord``s (components/curves) for ``compute_weights`` afterwards."""
    orch = validator._cloud_orch
    mode = getattr(validator, "mode", "prod")

    def _one(item: tuple[str, bytes]):
        commit_hash, payload = item
        source = payload.decode("utf-8", errors="replace")
        sub_uid = commit_hash[:18]
        started = time.time()
        audit_emit(audit, "cloud_eval.start", commit_hash=commit_hash,
                   sub_uid=sub_uid, mode=mode, resource=getattr(validator, "cloud_resource", ""),
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
        except Exception as e:
            audit_emit(audit, "cloud_eval.exception", commit_hash=commit_hash,
                       sub_uid=sub_uid, mode=mode, elapsed_s=time.time() - started,
                       error=str(e), error_type=type(e).__name__,
                       crashed=looks_like_crash(str(e)))
            return commit_hash, None, EvalResult("infra_dq", reason=f"cloud raised: {e}")
        verdict = classify_cloud_result(res)
        err = str(res.get("error", ""))
        audit_emit(audit, "cloud_eval.result", commit_hash=commit_hash,
                   sub_uid=sub_uid, mode=mode, elapsed_s=time.time() - started,
                   outcome=verdict.outcome, score=verdict.score,
                   reason=verdict.reason, failed=bool(res.get("failed")),
                   error=err, crashed=looks_like_crash(err),
                   raw_result=res)
        return commit_hash, res, verdict

    def evaluate_batch(payloads: list[tuple[str, bytes]]) -> dict[str, EvalResult]:
        out: dict[str, EvalResult] = {}
        if not payloads:
            return out
        workers = min(max_concurrent, len(payloads))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for commit_hash, raw, ev in ex.map(_one, payloads):
                if raw is not None:
                    sink[commit_hash] = raw
                out[commit_hash] = ev
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

    def await_capacity(emit):
        orch.begin_capacity_wait(round_id)
        try:
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
    from ..training import ScoreRecord

    results: dict[str, ScoreRecord] = {}
    sources: dict[str, str] = {}
    curve_data: dict[str, dict] = {}
    scored = [s for s in fsm.submissions.values() if s.status is SubStatus.SCORED]
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
    from ..training import ScoreRecord

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
    ``fsm_fairness_window``.
    """
    reg = validator.payment_registry
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
        except Exception as e:
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
    recent_rounds: list[set[str]] = []
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
    rounds_this_process = 0
    while max_rounds is None or rounds_this_process < max_rounds:
        round_start = clock()
        round_id = f"round_{round_num:06d}_{int(round_start)}_{vhk}"
        audit = _make_round_audit(validator, round_id)
        heartbeat_stop = _start_validator_heartbeat(validator, audit, round_start, clock)
        log.info("=" * 60)
        log.info("  FSM %s", round_id)
        max_concurrent = _affordable_now()
        try:
            audit_emit(audit, "round.start",
                       round_num=round_num, round_start=round_start,
                       commit_window_s=commit_window_s,
                       reveal_window_s=reveal_window_s,
                       round_period_s=round_period_s,
                       max_concurrent=max_concurrent,
                       fee_rao=fee_rao,
                       validator_fingerprint=_validator_fingerprint(validator))
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

            conf_jobs = _pending_confirmation_jobs(validator, audit)
            conf_sink: dict[str, dict] = {}
            conf_verdicts: dict = {}
            conf_thread = None
            if conf_jobs:
                audit_emit(audit, "confirmation.launch",
                           jobs=[{"key": k, "hotkey": p.get("hotkey", ""),
                                  "code_hash": p.get("code_hash", ""),
                                  "submission_round_id": p.get("source_round_id", "")}
                                 for k, _b, p in conf_jobs])
                log.info("  launching %d frontier confirmation run(s)", len(conf_jobs))
                _conf_batch = _make_cloud_batch(validator, round_id, conf_sink,
                                                max_concurrent, audit,
                                                baseline_bundle=baseline_bundle)
                _conf_payloads = [(k, b) for k, b, _p in conf_jobs]

                def _run_confirmations():
                    try:
                        conf_verdicts.update(_conf_batch(_conf_payloads))
                    except Exception as e:
                        audit_emit(audit, "confirmation.exception", error=str(e),
                                   error_type=type(e).__name__)
                        log.error("  confirmation batch raised: %s", e)

                conf_thread = threading.Thread(target=_run_confirmations,
                                               name="sn125-confirmations", daemon=True)
                conf_thread.start()
            round_max_eval = max(1, max_concurrent - len(conf_jobs))
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
                gate=make_source_gate(),
                evaluate_batch=_make_cloud_batch(validator, round_id, cloud_sink,
                                                 max_concurrent, audit,
                                                 baseline_bundle=baseline_bundle),
                recent_hotkeys=recent_hotkeys,
                carryover=carryover,
                sleep_until=lambda ts: _sleep_until(ts, clock, sleep),
                clock=clock,
                outage_backoff_s=float(getattr(validator, "fsm_outage_backoff_s",
                                               DEFAULT_OUTAGE_BACKOFF_S)),
                max_eval=round_max_eval,
                check_capacity=_make_capacity_check(validator),
                await_capacity=_make_capacity_wait(validator, round_id),
                on_event=lambda m: log.info("    %s", m),
                audit=audit,
            )
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
            log.info("  round %s: %d scored, %d deferred",
                     round_id, len(outcome.scores), len(outcome.deferred))

            results, sources, curve_data = _build_results(fsm, cloud_sink)
            confirmations: dict = {}
            confirmation_records: dict = {}
            if conf_thread is not None:
                conf_thread.join()
                confirmations, confirmation_records = _build_confirmations(
                    conf_jobs, conf_verdicts, conf_sink)
                audit_emit(audit, "confirmation.results",
                           results={k: {kk: vv for kk, vv in r.items() if kk != "curves"}
                                    for k, r in confirmation_records.items()})
            checkpoints = _checkpoint_inventory(curve_data)
            audit_emit(audit, "checkpoints.inventory",
                       count=len(checkpoints), checkpoints=checkpoints)
            prev_weights = dict(getattr(validator, "_last_weights", {}) or {})
            try:
                weights = validator.compute_weights(results, sources=sources, curve_data=curve_data,
                                                    confirmations=confirmations,
                                                    round_id=round_id)
            except TypeError:
                weights = validator.compute_weights(results)
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
                validator._save_round(round_id, results, weights, sources, curve_data,
                                      pause_reasons=outcome.pause_reasons,
                                      round_report=outcome.report,
                                      confirmation_records=confirmation_records)
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
                    round_id, results, weights, sources, curve_data,
                    pause_reasons=outcome.pause_reasons,
                    round_report=outcome.report)
                attestation_ok = published is not False
                audit_emit(audit, "attestation.publish_called", round_id=round_id,
                           published=attestation_ok)
            except Exception as e:
                audit_emit(audit, "attestation.publish_failed", round_id=round_id,
                           error=str(e), error_type=type(e).__name__)
                log.error("  _publish_attestation failed: %s", e)

            if getattr(validator, "set_weights_enabled", False):
                if not round_saved or not attestation_ok:
                    audit_emit(audit, "weights.set_skipped",
                               reason="missing_durable_round_or_attestation",
                               round_saved=round_saved, attestation_ok=attestation_ok,
                               weights=weights)
                    log.error("  refusing live set_weights: round_saved=%s attestation_ok=%s",
                              round_saved, attestation_ok)
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
                publisher.publish_now()
            round_num += 1
            rounds_this_process += 1
            if state_path is not None:
                try:
                    save_loop_state(state_path, round_num=round_num, carryover=carryover,
                                    recent_rounds=recent_rounds,
                                    last_weights=getattr(validator, "_last_weights", None),
                                    last_round_id=round_id)
                except Exception as e:
                    log.error("  could not checkpoint validator state to %s: %s",
                              state_path, e)
            if _restart_requested(validator):
                restart_reason = "restart"
            else:
                remaining = round_period_s - (clock() - round_start)
                if remaining > 0:
                    log.info("  next round in %.0fs", remaining)
                while remaining > 0:
                    sleep(min(remaining, _BOUNDARY_POLL_S))
                    if _restart_requested(validator):
                        restart_reason = "restart"
                        break
                    remaining = round_period_s - (clock() - round_start)
        if restart_reason is not None:
            log.info("  restart requested; stopping at the round boundary after "
                     "round #%d (state checkpointed to %s)", round_num - 1, state_path)
            break
    if weight_refresh_stop is not None:
        weight_refresh_stop.set()
    if payment_watch_stop is not None:
        payment_watch_stop.set()
    if publisher is not None:
        try:
            publisher.stop(flush=True)
        except Exception as e:
            log.warning("artifact publisher shutdown failed: %s", e)
    if restart_reason is not None:
        try:
            validator.restart_requested = True
        except Exception:
            pass
    return restart_reason


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


def _restart_requested(validator) -> bool:
    """True when the supervisor has asked for a restart at the next safe point."""
    if bool(getattr(validator, "restart_requested", False)):
        return True
    flag = _restart_flag_path(validator)
    try:
        return flag is not None and flag.exists()
    except OSError:
        return False


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


def _sleep_until(ts, clock, sleep) -> None:
    """Block (via the injected ``sleep``) until the injected ``clock`` reaches the
    FSM's pause-extended window deadline. Tests pass a fake clock+sleep so a 24h
    window elapses instantly and deterministically."""
    if ts is None:
        return
    remaining = ts - clock()
    if remaining > 0:
        sleep(remaining)
