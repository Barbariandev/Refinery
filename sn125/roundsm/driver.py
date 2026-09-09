"""FSM round driver (SPEC §6, §4.5 wiring) — the deterministic orchestration that
walks a single ``RoundFSM`` through one 24h round and is what ``Validator.run()``
calls per round.

This is the seam between the *pure* state machine (``round_fsm.py`` — enforces
phase legality, payment debits, selection, pause/resume) and the *impure* world
(the chain, the cloud, the wall clock). The driver holds no chain/cloud code: all
side effects arrive through injected callbacks, and time advances through an
injected ``sleep_until``. That is exactly what makes a full 24h round replayable
in milliseconds under ``MockChain`` + a fake clock (SPEC §6 "make K-overflow
behavior testable"; SPEC §4.4 "MockChain dry-run").

Phase order (SPEC §6):

    open_commits
      -> re-admit prior round's DEFERRED carryover (credit already paid)
      -> collect commit hashes, canonicalize ties, debit 1 credit each at acceptance
    [sleep to commit deadline] close_commits
      -> collect reveals (preimage must SHA-256 to the commit)
    [sleep to reveal deadline] close_reveals
      -> gate every revealed payload; gate-fail => miner-fault DQ (fee burned)
      -> select <=8 (deferred FIFO, then first accepted); overflow => DEFERRED
      -> evaluate selected (<=8 concurrent); score / miner-DQ / infra-DQ-refund
    publish -> (report, carryover)

The driver never sets weights or signs attestations — that stays in the validator
loop around it (SPEC §6.4), so this module has no R2 / bittensor dependency.
"""
from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .audit import audit_emit, looks_like_crash, payload_record
from .round_fsm import MAX_EVAL_PER_ROUND, Phase, RoundFSM, Submission, SubStatus

DEFAULT_OUTAGE_BACKOFF_S = 600.0
DEFAULT_MAX_EVAL_ATTEMPTS = 4
DEFAULT_LAUNCH_WINDOW_S = 4 * 3600.0
DEFAULT_LAUNCH_RETRY_S = 300.0


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one submission's evaluation, as reported by ``evaluate``.

    outcome:
      - "scored":   ``score`` is the held-out improvement -> ``record_score``.
      - "dq":       miner fault (cap breach, replay divergence) -> ``disqualify``
                    (fee burned, no refund). ``reason`` required.
      - "infra_dq": our infra failed after the cloud layer's own relaunch retries
                    -> ``infra_dq`` (credit refunded). ``reason`` required.
      - "flake":    transient provisioning failure (the box never came up) -> the
                    SAME commitment relaunches at $0 (``record_provisioning_flake``);
                    3 consecutive flakes = a provider outage that PAUSES the round
                    (it never silently shrinks, §2.4). Distinct from "infra_dq",
                    which is terminal (the box ran but failed: timeout, cost-cap,
                    SIGSEGV, parse gap) and refunds rather than retries.
    """
    outcome: str
    score: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str = ""


@dataclass
class RoundOutcome:
    """What one driven round returns to the validator loop."""
    report: dict
    scores: dict[str, float]
    carryover: list[Submission] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    pause_reasons: list[str] = field(default_factory=list)
    unlaunched: list[str] = field(default_factory=list)


def _commit_acceptance_key(item: tuple[str, str]) -> tuple[str, str]:
    """Canonical tie-break for same-window commit fetches.

    The commit transport polls miners during a window; it does not provide a
    trustworthy per-miner arrival timestamp. Canonicalizing by hotkey then hash
    makes simultaneous fetches, duplicate hashes, and collector order differences
    replayable. The FSM records ``accepted_at`` / ``accept_index`` after this.
    """
    hotkey, commit_hash = item
    return (str(hotkey), str(commit_hash).lower())


def _credit_snapshot(fsm: RoundFSM) -> list[dict[str, Any]]:
    rows = []
    for s in fsm.submissions.values():
        bal = None
        try:
            bal = fsm.registry.balance(s.coldkey)
        except Exception:
            pass
        rows.append({
            "hotkey": s.hotkey,
            "coldkey": s.coldkey,
            "commit_hash": s.commit_hash,
            "status": s.status.value,
            "deferrals": s.deferrals,
            "selected_for_eval": s.selected_for_eval,
            "selection_rank": s.selection_rank,
            "credit_balance": bal,
        })
    return rows


def _real_sleep_until(ts: float) -> None:
    """Production ``sleep_until``: block until wall-clock ``ts`` (the FSM's
    pause-extended deadline). Tests pass a fake that advances the injected clock
    instead, so a 24h round runs instantly and deterministically."""
    remaining = ts - time.time()
    if remaining > 0:
        time.sleep(remaining)


def _apply_eval(fsm: RoundFSM, commit_hash: str, res: EvalResult,
                scores: dict[str, float]) -> None:
    """Apply one ``EvalResult`` to the FSM (assumes ``start_run`` already fired,
    i.e. the submission is RUNNING). Shared by the sequential and batch paths so
    both record scores / DQ / infra-DQ identically."""
    sub = fsm.submissions[commit_hash]
    if res.outcome == "scored":
        fsm.record_score(commit_hash, res.score)
        scores[sub.hotkey] = res.score
    elif res.outcome == "dq":
        fsm.disqualify(commit_hash, res.reason or "miner-fault DQ")
    elif res.outcome == "infra_dq":
        fsm.infra_dq(commit_hash, res.reason or "infra failure")
    else:
        raise ValueError(f"unknown EvalResult.outcome: {res.outcome!r}")


def _accepts_kw(fn: Callable, name: str) -> bool:
    """Does ``fn`` take keyword ``name`` (or **kwargs)? Lets the driver pass the
    launch deadline to hooks that understand it while older two-line lambdas in
    tests / dry-runs keep working unchanged."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _ensure_b200_capacity(fsm: RoundFSM, check_capacity, await_capacity,
                          emit: Callable[[str], None], audit=None, *,
                          deadline: float | None = None,
                          clock: Callable[[], float] = time.time) -> bool:
    """B200-only capacity-aware delay (operator directive 2026-06-24). Before
    renting any box, confirm the canonical B200 eval SKU has capacity. If the
    provider has ZERO inventory we DO NOT substitute another GPU class or shrink
    the round — we PAUSE it (§2.4: deadlines stretch, the round never silently
    shrinks) and wait, publishing the delay publicly the whole time, then resume.

    Returns True when capacity is (or became) available. With a ``deadline`` the
    wait is bounded by the round's launch window: if B200s are still dry when it
    closes the gate returns False and the caller carries every un-started
    submission to the next round (credit retained) instead of waiting forever.

    This is a pure CAPACITY wait, distinct from a host-802 *flake* (the existing
    consecutive-flake outage sweep in the eval loop): no submission is infra-DQ'd,
    no credit is refunded, and because no box is rented during the wait, the 20h
    eval budget clock (enforced on-box during training) counts only real training
    time, never the paused wait.

    ``check_capacity() -> bool`` is a cheap inventory probe (True = B200 available).
    ``await_capacity(emit) -> None`` blocks until capacity returns, KEEPS WAITING
    (never substitutes/fails), and publishes the public delay banner. Both are
    injected so tests drive the pause->resume transition on a CPU mock. When either
    is None the gate is a no-op (today's behaviour / local MockChain dry-runs)."""
    if check_capacity is None or await_capacity is None:
        audit_emit(audit, "capacity.check_skipped", reason="capacity hooks unavailable")
        return True
    if check_capacity():
        audit_emit(audit, "capacity.available")
        return True
    remaining: float | None = None
    if deadline is not None:
        remaining = deadline - clock()
        if remaining <= 0:
            audit_emit(audit, "capacity.window_closed", deadline=deadline)
            emit("B200 still dry and the launch window is closed — un-started "
                 "submissions carry to the next round (credits retained)")
            return False
    fsm.pause("awaiting B200 capacity (provider has 0 inventory) — §2.4 capacity delay")
    audit_emit(audit, "capacity.wait_started",
               reason=fsm.pause_reasons[-1] if fsm.pause_reasons else "",
               max_wait_s=remaining)
    emit("⏸ B200 capacity wait — round PAUSED (deadlines stretch; eval budget "
         "clock counts only training, never this wait)"
         + (f"; up to {remaining/3600:.1f}h" if remaining is not None else ""))
    restored = True
    try:
        if remaining is not None and _accepts_kw(await_capacity, "max_wait_s"):
            await_capacity(emit, max_wait_s=remaining)
        else:
            await_capacity(emit)
    except Exception as e:
        restored = False
        audit_emit(audit, "capacity.wait_expired", error=str(e),
                   error_type=type(e).__name__, max_wait_s=remaining)
        emit(f"B200 capacity did not return inside the launch window ({e})")
    finally:
        if fsm.phase is Phase.PAUSED:
            fsm.resume()
            audit_emit(audit, "capacity.wait_finished", restored=restored)
            emit("▶ B200 capacity restored — round RESUMED" if restored
                 else "▶ round RESUMED without capacity — carrying un-started work")
    if restored and not check_capacity():
        audit_emit(audit, "capacity.recheck_failed_after_wait")
        return False
    return restored


def drive_round(
    fsm: RoundFSM,
    *,
    collect_commits: Callable[[], list[tuple[str, str]]],
    collect_reveals: Callable[[list[str]], dict[str, bytes]],
    evaluate: Callable[[str, bytes], EvalResult] | None = None,
    evaluate_batch: Callable[[list[tuple[str, bytes]]], dict[str, EvalResult]] | None = None,
    gate: Callable[[bytes], GateResult] | None = None,
    recent_hotkeys: tuple[str, ...] = (),
    carryover: list[Submission] = (),
    sleep_until: Callable[[float], None] = _real_sleep_until,
    clock: Callable[[], float] = time.time,
    outage_backoff_s: float = DEFAULT_OUTAGE_BACKOFF_S,
    max_eval_attempts: int = DEFAULT_MAX_EVAL_ATTEMPTS,
    launch_window_s: float = DEFAULT_LAUNCH_WINDOW_S,
    launch_retry_s: float = DEFAULT_LAUNCH_RETRY_S,
    max_eval: int = MAX_EVAL_PER_ROUND,
    check_capacity: Callable[[], bool] | None = None,
    await_capacity: Callable[[Callable[[str], None]], None] | None = None,
    on_event: Callable[[str], None] | None = None,
    audit: Any = None,
) -> RoundOutcome:
    """Drive ``fsm`` through one full round. See module docstring for the phase
    order. All real-world effects are injected:

    - ``collect_commits()`` -> ``[(hotkey, commit_hash), ...]`` seen during the
      commit window. The batch is canonicalized by ``(hotkey, commit_hash)`` so
      same-window fetch conflicts cannot depend on network/collector ordering.
      Each accepted commit is debited 1 credit; a commit whose coldkey is out of
      credit (or a duplicate hash) is skipped, logged, and published as a commit
      rejection — one bad miner never stalls the round.
    - ``collect_reveals(open_hashes)`` -> ``{commit_hash: payload}`` revealed
      during the reveal window. A payload whose hash != commit is skipped
      (it stays COMMITTED and is consumed as UNREVEALED at close — E4/E5).
    - ``evaluate(commit_hash, payload)`` -> ``EvalResult`` (sequential path).
    - ``evaluate_batch(selected_payloads)`` -> ``{commit_hash: EvalResult}``
      (optional concurrent path, SPEC §4.4): when given, the <=8 selected
      submissions are handed to ONE call that may run them in parallel (the live
      wiring fans them across the Targon rental semaphore), so the round's eval
      phase is wall-clock ~one eval, not <=8 back-to-back. FSM transitions still
      apply deterministically in ``selected`` order; a missing entry or a raised
      batch is treated as our infra fault (refund). Exactly one of ``evaluate`` /
      ``evaluate_batch`` must be provided. Either may return ``EvalResult("flake")``
      for a submission whose box never came up — see the outage handling below.
    - ``gate(payload)`` -> ``GateResult`` (optional). Gate-fail => miner DQ
      *before* selection, so a bad submission never burns a selection slot.
    - ``recent_hotkeys`` is retained for API compatibility; selection is FIFO.
    - ``carryover`` re-admits last round's DEFERRED commits (already paid).
    - ``sleep_until`` advances time to each window deadline (real sleep in prod;
      fake clock-advance in tests).
    - ``clock`` reads the current time (to schedule the outage backoff sleep);
      ``outage_backoff_s`` is how long a detected provider outage pauses the round
      before resuming + retrying; ``max_eval_attempts`` bounds the relaunch budget.

    Outage handling (§2.4): a ``"flake"`` verdict relaunches the SAME commitment at
    $0 (``record_provisioning_flake``). 3 *consecutive* flakes = a provider outage
    and the FSM auto-PAUSES — the driver then waits ``outage_backoff_s`` and
    ``resume()``s, so the round's deadlines stretch by the paused time rather than
    silently shrinking to whoever happened to provision.

    Launch window: from selection the round has ``launch_window_s`` to START each
    evaluation. The capacity gate and every relaunch are bounded by that window
    (``evaluate_batch`` receives ``launch_deadline=`` when it accepts the keyword,
    so the live batch can poll inventory per submission and pick up sporadic
    GPUs). Anything still un-started when the window closes — or after
    ``max_eval_attempts`` flakes — is CARRIED to the next round as DEFERRED with
    its credit and FIFO priority (``fsm.defer_unlaunched``), round after round;
    a commit carried ``MAX_UNLAUNCHED_DEFERRALS`` rounds and older than
    ``UNLAUNCHED_MAX_AGE_S`` expires and its credit is absorbed. No credit is
    ever burned as miner fault by a failure to obtain a GPU, and the round
    always publishes.
    """
    emit = on_event or (lambda _m: None)
    if (evaluate is None) == (evaluate_batch is None):
        raise ValueError("provide exactly one of evaluate / evaluate_batch")

    fsm.open_commits()
    audit_emit(audit, "commit_window.opened",
               config=fsm.config.__dict__, deadline=fsm.deadline(),
               carryover_count=len(carryover))
    commit_rejections: list[dict[str, str]] = []
    for sub in carryover:
        try:
            fsm.readmit_deferred(sub)
            audit_emit(audit, "commit.carryover_readmitted",
                       hotkey=sub.hotkey, coldkey=sub.coldkey,
                       commit_hash=sub.commit_hash, deferrals=sub.deferrals,
                       accept_index=sub.accept_index, accepted_at=sub.accepted_at)
            emit(f"readmit carryover {sub.commit_hash[:12]} (deferrals={sub.deferrals})")
        except Exception as e:
            audit_emit(audit, "commit.carryover_rejected",
                       hotkey=sub.hotkey, coldkey=sub.coldkey,
                       commit_hash=sub.commit_hash, reason=str(e))
            emit(f"carryover {sub.commit_hash[:12]} skipped: {e}")
    collected_commits = sorted(collect_commits(), key=_commit_acceptance_key)
    audit_emit(audit, "commit.acceptance_order",
               commits=[{"hotkey": hk, "commit_hash": h}
                        for hk, h in collected_commits])
    for hotkey, commit_hash in collected_commits:
        try:
            sub = fsm.accept_commit(hotkey, commit_hash)
            audit_emit(audit, "commit.accepted",
                       hotkey=hotkey, coldkey=sub.coldkey,
                       commit_hash=commit_hash, accepted_at=sub.accepted_at,
                       accept_index=sub.accept_index, fee_rao=fsm.config.fee_rao)
        except Exception as e:
            commit_rejections.append({
                "hotkey": hotkey,
                "commit_hash": commit_hash,
                "reason": str(e),
            })
            audit_emit(audit, "commit.rejected", hotkey=hotkey,
                       commit_hash=commit_hash, reason=str(e))
            emit(f"commit {commit_hash[:12]} from {hotkey[:16]} rejected: {e}")
    sleep_until(fsm.deadline())
    fsm.close_commits()
    audit_emit(audit, "commit_window.closed",
               committed=sum(1 for s in fsm.submissions.values()
                             if s.status is SubStatus.COMMITTED),
               rejected=len(commit_rejections),
               credit_snapshot=_credit_snapshot(fsm))

    audit_emit(audit, "reveal_window.opened", deadline=fsm.deadline())
    open_hashes = [h for h, s in fsm.submissions.items()
                   if s.status is SubStatus.COMMITTED]
    for commit_hash, payload in collect_reveals(open_hashes).items():
        try:
            sub = fsm.reveal(commit_hash, payload)
            audit_emit(audit, "submission.revealed",
                       hotkey=sub.hotkey, coldkey=sub.coldkey,
                       commit_hash=commit_hash,
                       **payload_record(payload, include_source=True))
        except Exception as e:
            audit_emit(audit, "submission.reveal_rejected",
                       commit_hash=commit_hash, reason=str(e),
                       payload_sha256=hashlib.sha256(payload).hexdigest(),
                       payload_bytes=len(payload))
            emit(f"reveal {commit_hash[:12]} rejected: {e}")
    sleep_until(fsm.deadline())
    fsm.close_reveals()
    unrevealed = [s for s in fsm.submissions.values()
                  if s.status is SubStatus.UNREVEALED]
    for sub in unrevealed:
        audit_emit(audit, "submission.unrevealed",
                   hotkey=sub.hotkey, coldkey=sub.coldkey,
                   commit_hash=sub.commit_hash, credit_consumed=True)
    audit_emit(audit, "reveal_window.closed",
               revealed=sum(1 for s in fsm.submissions.values()
                            if s.status is SubStatus.REVEALED),
               unrevealed=len(unrevealed))

    if gate is not None:
        for commit_hash, sub in list(fsm.submissions.items()):
            if sub.status is not SubStatus.REVEALED or sub.payload is None:
                continue
            g = gate(sub.payload)
            if not g.ok:
                fsm.disqualify(commit_hash, g.reason or "gate failure")
                audit_emit(audit, "gate.dq",
                           hotkey=sub.hotkey, coldkey=sub.coldkey,
                           commit_hash=commit_hash, reason=g.reason,
                           **payload_record(sub.payload, include_source=False))
                emit(f"gate DQ {commit_hash[:12]}: {g.reason}")
            else:
                audit_emit(audit, "gate.passed",
                           hotkey=sub.hotkey, coldkey=sub.coldkey,
                           commit_hash=commit_hash,
                           **payload_record(sub.payload, include_source=False))

    selected, deferred = fsm.select_for_evaluation(
        recent_hotkeys=recent_hotkeys, max_eval=max_eval)
    audit_emit(audit, "selection.complete",
               max_eval=max_eval,
               selected=[
                   {
                       "hotkey": fsm.submissions[h].hotkey,
                       "commit_hash": h,
                       "selection_rank": fsm.submissions[h].selection_rank,
                       "deferrals": fsm.submissions[h].deferrals,
                   }
                   for h in selected
               ],
               deferred=[
                   {
                       "hotkey": fsm.submissions[h].hotkey,
                       "commit_hash": h,
                       "selection_rank": fsm.submissions[h].selection_rank,
                       "deferrals": fsm.submissions[h].deferrals,
                   }
                   for h in deferred
               ])
    emit(f"selected {len(selected)}/{len(selected) + len(deferred)} "
         f"(deferred {len(deferred)})")

    if evaluate_batch is None:
        _seq = evaluate

        def evaluate_batch(payloads):  # type: ignore[misc]
            out: dict[str, EvalResult] = {}
            for h, p in payloads:
                try:
                    out[h] = _seq(h, p)
                except Exception as e:
                    out[h] = EvalResult("infra_dq", reason=f"evaluate raised: {e}")
            return out

    scores: dict[str, float] = {}
    launch_opened = clock()
    launch_deadline = launch_opened + max(0.0, float(launch_window_s))
    batch_takes_deadline = _accepts_kw(evaluate_batch, "launch_deadline")
    capacity_ok = True
    if selected:
        audit_emit(audit, "launch_window.opened", opened_at=launch_opened,
                   deadline=launch_deadline, window_s=launch_window_s,
                   selected_count=len(selected))
        emit(f"launch window open for {launch_window_s/3600:.1f}h: "
             f"{len(selected)} submission(s) start as soon as a B200 is available")
        audit_emit(audit, "capacity.check", selected_count=len(selected))
        capacity_ok = _ensure_b200_capacity(fsm, check_capacity, await_capacity,
                                            emit, audit, deadline=launch_deadline,
                                            clock=clock)
    pending = list(selected)
    if capacity_ok:
        for commit_hash in selected:
            fsm.start_run(commit_hash)
            sub = fsm.submissions[commit_hash]
            audit_emit(audit, "evaluation.run_started",
                       hotkey=sub.hotkey, coldkey=sub.coldkey,
                       commit_hash=commit_hash, selection_rank=sub.selection_rank)
    attempt = 0
    while pending and capacity_ok and attempt < max_eval_attempts:
        attempt += 1
        if attempt > 1:
            if clock() >= launch_deadline:
                break
            audit_emit(audit, "capacity.recheck", attempt=attempt, pending=list(pending))
            capacity_ok = _ensure_b200_capacity(fsm, check_capacity, await_capacity,
                                                emit, audit, deadline=launch_deadline,
                                                clock=clock)
            if not capacity_ok:
                break
        audit_emit(audit, "evaluation.attempt_started",
                   attempt=attempt, pending=list(pending),
                   launch_deadline=launch_deadline)
        payloads = [(h, fsm.submissions[h].payload) for h in pending]
        try:
            if not payloads:
                results_map = {}
            elif batch_takes_deadline:
                results_map = evaluate_batch(payloads, launch_deadline=launch_deadline)
            else:
                results_map = evaluate_batch(payloads)
        except Exception as e:
            audit_emit(audit, "evaluation.batch_exception",
                       attempt=attempt, pending=list(pending),
                       error=str(e), error_type=type(e).__name__,
                       crashed=looks_like_crash(str(e)))
            emit(f"evaluate_batch raised: {e}")
            results_map = {h: EvalResult("infra_dq", reason=f"evaluate_batch raised: {e}")
                           for h in pending}
        retry: list[str] = []
        for commit_hash in pending:
            res = results_map.get(commit_hash) or EvalResult(
                "infra_dq", reason="evaluate_batch returned no result")
            if res.outcome == "flake":
                fsm.record_provisioning_flake(commit_hash)
                sub = fsm.submissions[commit_hash]
                audit_emit(audit, "evaluation.flake",
                           attempt=attempt, hotkey=sub.hotkey,
                           coldkey=sub.coldkey, commit_hash=commit_hash,
                           reason=res.reason)
                retry.append(commit_hash)
                if fsm.phase is Phase.PAUSED:
                    audit_emit(audit, "round.paused",
                               reason=fsm.pause_reasons[-1] if fsm.pause_reasons else "",
                               outage_backoff_s=outage_backoff_s)
                    emit(f"provider outage (consecutive flakes) — pausing round "
                         f"{outage_backoff_s:.0f}s")
                    sleep_until(min(clock() + outage_backoff_s,
                                    max(launch_deadline, clock())))
                    fsm.resume()
                    audit_emit(audit, "round.resumed",
                               reason="provider recovered after flake outage")
                    emit("provider recovered — resuming round")
            else:
                fsm.record_provision_ok()
                _apply_eval(fsm, commit_hash, res, scores)
                sub = fsm.submissions[commit_hash]
                audit_emit(audit, "evaluation.finalized",
                           attempt=attempt, hotkey=sub.hotkey,
                           coldkey=sub.coldkey, commit_hash=commit_hash,
                           outcome=res.outcome, status=sub.status.value,
                           score=res.score, reason=res.reason,
                           crashed=looks_like_crash(res.reason))
        pending = retry
        if pending and launch_retry_s > 0 and clock() < launch_deadline \
                and fsm.phase is not Phase.PAUSED:
            sleep_until(min(clock() + launch_retry_s, launch_deadline))
    unlaunched: list[str] = []
    for commit_hash in pending:
        if not capacity_ok:
            reason = "no B200 capacity inside the launch window"
        elif clock() >= launch_deadline:
            reason = (f"no GPU obtained inside the {launch_window_s/3600:.1f}h launch "
                      f"window ({attempt} attempt(s))")
        else:
            reason = f"relaunch budget exhausted ({attempt} attempt(s)) without a box"
        status = fsm.defer_unlaunched(commit_hash, reason)
        sub = fsm.submissions[commit_hash]
        unlaunched.append(commit_hash)
        if status is SubStatus.DEFERRED:
            emit(f"{commit_hash[:12]} never launched ({reason}) -> carried to next "
                 f"round, credit retained (carry #{sub.deferrals})")
            audit_emit(audit, "evaluation.deferred",
                       hotkey=sub.hotkey, coldkey=sub.coldkey,
                       commit_hash=commit_hash, reason=reason,
                       deferrals=sub.deferrals, credit_retained=True)
        else:
            emit(f"{commit_hash[:12]} never launched ({reason}); carried "
                 f"{sub.deferrals} rounds over 30+ days -> expired, credit absorbed")
            audit_emit(audit, "evaluation.finalized",
                       hotkey=sub.hotkey, coldkey=sub.coldkey,
                       commit_hash=commit_hash, outcome="expired",
                       status=sub.status.value, reason=reason,
                       deferrals=sub.deferrals, crashed=False)
    if selected:
        audit_emit(audit, "launch_window.closed", opened_at=launch_opened,
                   deadline=launch_deadline, closed_at=clock(),
                   launched=len(selected) - len(unlaunched),
                   unlaunched=list(unlaunched))

    report = fsm.publish()
    report["selection_rule"] = (
        "carryover deferrals desc, then validator-canonical commit acceptance "
        "order: hotkey asc, commit_hash asc; selection tiebreak: accepted_at, "
        "accept_index, commit_hash"
    )
    report["selected"] = list(selected)
    report["deferred"] = list(deferred) + [
        h for h in unlaunched if fsm.submissions[h].status is SubStatus.DEFERRED]
    report["unlaunched"] = list(unlaunched)
    report["launch_window"] = {"opened_at": launch_opened, "deadline": launch_deadline,
                               "window_s": launch_window_s}
    if commit_rejections:
        report["commit_rejections"] = commit_rejections
    audit_emit(audit, "payments.credit_snapshot",
               phase="final", credit_snapshot=_credit_snapshot(fsm))
    audit_emit(audit, "round.published", report=report)
    return RoundOutcome(
        report=report,
        scores=scores,
        carryover=fsm.carryover(),
        selected=selected,
        deferred=list(report["deferred"]),
        pause_reasons=list(fsm.pause_reasons),
        unlaunched=unlaunched,
    )
