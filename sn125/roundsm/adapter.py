"""Adapter layer (SPEC §4.1) — bridges the live dendrite transport + the Targon
cloud orchestrator into the injected callbacks ``drive_round`` consumes.

``driver.py`` is pure orchestration over injected callbacks and holds no chain /
cloud code; this module is the ONE place that knows the real synapses
(``CommitHash`` / ``GetSubmission``) and the cloud result-dict shape. Keeping the
seam here means:

  - ``driver.py`` stays bittensor-free and replayable under ``MockChain``;
  - this module is unit-testable with a fake dendrite + a fake orchestrator (no
    chain, no GPU) — exactly what the CPU tests do. To keep adapter.py itself
    import-light, the synapse *classes* are injected (the live wiring passes
    ``neuron.CommitHash`` / ``neuron.GetSubmission``; tests pass trivial fakes),
    so nothing here imports bittensor or torch at module load.

Live wiring (in the ``validate`` FSM entrypoint, ``roundsm/live.py``):

    from .neuron import CommitHash, GetSubmission
    commits  = make_commit_collector(self.dendrite, axons, round_id, CommitHash)
    reveals  = make_reveal_collector(self.dendrite, axons, round_id, GetSubmission)
    gate     = make_source_gate()
    evaluate = make_cloud_evaluator(self._cloud_orch, round_id, mode=self.mode)
    outcome  = drive_round(fsm, collect_commits=commits, collect_reveals=reveals,
                           evaluate=evaluate, gate=gate, carryover=carryover, ...)

The two-phase commit/reveal transport already exists at the synapse level:
``CommitHash`` carries the SHA-256 commit; ``GetSubmission`` carries the source
preimage. The FSM re-verifies ``sha256(payload) == commit_hash`` on reveal, so a
miner cannot reveal a different optimizer than it committed (E5).
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

from .audit import audit_emit, looks_like_crash, payload_record
from .driver import EvalResult, GateResult

MAX_SOURCE_BYTES = 1_048_576

FLAKE_ERROR_MARKERS: tuple[str, ...] = (
    "rental_provisioning_failed",
    "not_started",
    "box_throughput_floor",
    "box_throughput_ceiling",
    "box_throughput_drift",
    "capacity_wait",
    "capacity-aware delay",
    "b200 capacity",
)

INFRA_ERROR_MARKERS: tuple[str, ...] = (
    "timeout_exceeded",
    "[critical]",
    "no json result",
    "rc=139",
    "[pre-eval infra]",
    "network lockdown",
)

_NO_SCORE_SENTINEL = -1.0


def _is_hex64(s: str) -> bool:
    if len(s) != 64:
        return False
    try:
        int(s, 16)
    except ValueError:
        return False
    return True


def make_commit_collector(
    dendrite: Any,
    axons: list,
    round_id: str,
    commit_synapse_cls: Callable[..., Any],
    *,
    timeout: float = 10.0,
    on_event: Callable[[str], None] | None = None,
    audit: Any = None,
) -> Callable[[], list[tuple[str, str]]]:
    """Build ``collect_commits() -> [(hotkey, code_hash), ...]``.

    Queries each axon for its ``CommitHash`` (round-scoped). Only well-formed
    64-hex commits are returned lower-cased. A non-responding or malformed miner
    is skipped, never fatal. Duplicate commit hashes are intentionally returned:
    the driver canonicalizes the batch, debits the deterministic winner, and
    publishes duplicate-hash losers as commit rejections.
    """
    emit = on_event or (lambda _m: None)

    def collect_commits() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for ai in axons:
            hotkey = getattr(ai, "hotkey", "?")
            audit_emit(audit, "commit.query.start", hotkey=hotkey,
                       timeout_s=timeout)
            syn = commit_synapse_cls(round_id=round_id)
            try:
                resp = dendrite.query(ai, syn, timeout=timeout, deserialize=False)
            except Exception as e:
                audit_emit(audit, "commit.query.error", hotkey=hotkey,
                           error=str(e), error_type=type(e).__name__)
                emit(f"no commit from {hotkey[:16]}: {e}")
                continue
            h = (getattr(resp, "code_hash", "") or "").lower()
            if not _is_hex64(h):
                audit_emit(audit, "commit.malformed", hotkey=hotkey,
                           code_hash=h, code_hash_len=len(h))
                continue
            audit_emit(audit, "commit.fetched", hotkey=hotkey, commit_hash=h)
            out.append((hotkey, h))
        emit(f"collected {len(out)} commitment(s)")
        audit_emit(audit, "commit.collection.complete", count=len(out),
                   commits=[{"hotkey": hk, "commit_hash": h} for hk, h in out])
        return out

    return collect_commits


def make_reveal_collector(
    dendrite: Any,
    axons: list,
    round_id: str,
    submission_synapse_cls: Callable[..., Any],
    *,
    timeout: float = 30.0,
    max_source_bytes: int = MAX_SOURCE_BYTES,
    on_event: Callable[[str], None] | None = None,
    audit: Any = None,
) -> Callable[[list[str]], dict[str, bytes]]:
    """Build ``collect_reveals(open_hashes) -> {code_hash: source_bytes}``.

    Queries each axon for its ``GetSubmission`` source preimage, hashes it, and
    returns only payloads whose SHA-256 is among ``open_hashes`` (the still-
    COMMITTED commits the FSM is awaiting). Oversized or empty sources are
    dropped (those commits stay UNREVEALED -> consumed without refund at close).
    The FSM re-verifies the hash on ``reveal``, so this matching is belt-and-
    suspenders.
    """
    emit = on_event or (lambda _m: None)

    def collect_reveals(open_hashes: list[str]) -> dict[str, bytes]:
        wanted = {h.lower() for h in open_hashes}
        out: dict[str, bytes] = {}
        if not wanted:
            audit_emit(audit, "reveal.collection.skipped", reason="no open commits")
            return out
        for ai in axons:
            hotkey = getattr(ai, "hotkey", "?")
            audit_emit(audit, "reveal.query.start", hotkey=hotkey,
                       timeout_s=timeout, open_commit_count=len(wanted))
            syn = submission_synapse_cls(round_id=round_id)
            try:
                resp = dendrite.query(ai, syn, timeout=timeout, deserialize=False)
            except Exception as e:
                audit_emit(audit, "reveal.query.error", hotkey=hotkey,
                           error=str(e), error_type=type(e).__name__)
                emit(f"no reveal from {hotkey[:16]}: {e}")
                continue
            src = getattr(resp, "source_code", "") or ""
            if not src:
                audit_emit(audit, "reveal.empty", hotkey=hotkey)
                continue
            payload = src.encode()
            if len(payload) > max_source_bytes:
                audit_emit(audit, "reveal.oversized", hotkey=hotkey,
                           source_bytes=len(payload), max_source_bytes=max_source_bytes)
                emit(f"oversized reveal from {hotkey[:16]}: {len(payload)} bytes")
                continue
            h = hashlib.sha256(payload).hexdigest()
            if h in wanted and h not in out:
                out[h] = payload
                audit_emit(audit, "reveal.fetched", hotkey=hotkey,
                           commit_hash=h, matched_open_commit=True,
                           **payload_record(payload, include_source=False))
            else:
                audit_emit(audit, "reveal.ignored", hotkey=hotkey,
                           commit_hash=h, matched_open_commit=h in wanted,
                           duplicate_hash=h in out,
                           **payload_record(payload, include_source=False))
        emit(f"collected {len(out)}/{len(wanted)} reveal(s)")
        audit_emit(audit, "reveal.collection.complete", count=len(out),
                   open_commit_count=len(wanted), commit_hashes=sorted(out))
        return out

    return collect_reveals


def make_source_gate(
    *,
    max_bytes: int = MAX_SOURCE_BYTES,
    allow_native: bool = False,
    validate_fn: Callable[..., list[str]] | None = None,
) -> Callable[[bytes], GateResult]:
    """Build ``gate(payload) -> GateResult`` over the full pre-build gate
    (``gate.validate_single_source``: C2 binary/PTX/NUL screens + C8 size caps
    + the AST sandbox validator). Run BEFORE selection so a forbidden-import /
    oversized / un-parseable optimizer is DQ'd (fee burned) without ever
    occupying one of the <=8 evaluation slots (SPEC §6.2) — and so nothing that
    passes selection can then fail the identical gate ``prod_eval`` re-runs on
    the B200 box after the rental was already paid for. ``validate_fn`` is
    injectable for tests (violation-list contract, AST-only)."""
    if validate_fn is not None:
        def gate(payload: bytes) -> GateResult:
            try:
                source = payload.decode("utf-8")
            except UnicodeDecodeError:
                return GateResult(False, "source not valid UTF-8")
            violations = validate_fn(source, max_bytes=max_bytes,
                                     allow_native=allow_native)
            if violations:
                return GateResult(False, "; ".join(violations[:3]))
            return GateResult(True)

        return gate

    from ..gate import validate_single_source

    def gate(payload: bytes) -> GateResult:
        try:
            source = payload.decode("utf-8")
        except UnicodeDecodeError:
            return GateResult(False, "source not valid UTF-8")
        res = validate_single_source(source)
        if not res.ok:
            return GateResult(False, "; ".join(res.reasons[:3]))
        return GateResult(True)

    return gate


def no_box_ran(res: dict) -> bool:
    """True when the cloud layer says it never rented a box for this result
    (``no_box`` / ``capacity_wait`` stamped by cloud.evaluate_submission when the
    failure happened before any workload existed). The miner's code never
    executed, so the failure can only be ours (capacity, provider API, spend
    admission) — it must never burn a credit."""
    return bool(res.get("failed")) and bool(res.get("no_box") or res.get("capacity_wait"))


def is_no_gpu_failure(reason: str) -> bool:
    """Does a recorded failure reason describe 'no GPU was ever rented'? Used to
    re-read historical audit verdicts (live._refund_no_gpu_dqs) with the current
    classification; keep in step with FLAKE_ERROR_MARKERS' capacity entries."""
    r = (reason or "").lower()
    return any(m in r for m in ("capacity_wait", "capacity-aware delay", "b200 capacity",
                                "rental_provisioning_failed"))


def classify_cloud_result(res: dict, *, ambiguous: str = "dq") -> EvalResult:
    """Map a ``cloud.evaluate_submission`` result dict to an ``EvalResult``.

    - not failed + a real score (> the -1.0 sentinel) -> ``scored``.
    - failed + a "box never came up" marker (provisioning / not-started) ->
      ``flake`` (transient: same commitment relaunches; 3 consecutive = outage).
    - failed + a terminal infra marker (timeout / cost-cap / SIGSEGV) ->
      ``infra_dq`` (credit refunded; won't recover on a relaunch).
    - failed otherwise (e.g. "bench failed (rc=1)": the miner's optimizer crashed
      or produced no valid score in the sandbox) -> ``dq`` (fee burned).
    - not failed but score is still the -1.0 sentinel -> a harness parse gap (our
      fault) -> ``infra_dq``.
    ``ambiguous`` is the verdict for a failed result with no recognizable marker;
    it defaults to ``"dq"`` (no-refund-fraud-surface; only refund when confident).
    """
    failed = bool(res.get("failed"))
    score = res.get("score")
    err = str(res.get("error", "")).lower()

    if not failed:
        if isinstance(score, (int, float)) and float(score) > _NO_SCORE_SENTINEL:
            return EvalResult("scored", score=float(score))
        return EvalResult("infra_dq", reason="no parseable score (harness gap)")

    if any(m in err for m in FLAKE_ERROR_MARKERS) or no_box_ran(res):
        return EvalResult("flake", reason=res.get("error", "provisioning flake"))

    if any(m in err for m in INFRA_ERROR_MARKERS):
        return EvalResult("infra_dq", reason=res.get("error", "infra failure"))

    if ambiguous == "infra_dq":
        return EvalResult("infra_dq", reason=res.get("error", "unclassified failure"))
    return EvalResult("dq", reason=res.get("error", "miner-fault DQ"))


def make_cloud_evaluator(
    cloud_orch: Any,
    round_id: str,
    *,
    mode: str = "prod",
    baseline_bundle: dict | None = None,
    on_result: Callable[[str, dict], None] | None = None,
    ambiguous: str = "dq",
    audit: Any = None,
) -> Callable[[str, bytes], EvalResult]:
    """Build ``evaluate(commit_hash, payload) -> EvalResult`` over a
    ``TargonOrchestrator`` (or any object with a compatible
    ``evaluate_submission(source, round_id, sub_uid, mode=...)``). The cloud
    layer runs its own provisioning retries / box teardown; a raise here is
    treated as an infra fault (refund). ``commit_hash[:18]`` is the per-submission
    sub_uid (the workload name is hashed from round_id|sub_uid downstream).
    ``on_result`` receives the raw cloud dict (e.g. to ferry ``curve_data`` /
    ``components`` to the dashboard) — the driver itself only needs the verdict.
    """
    def evaluate(commit_hash: str, payload: bytes) -> EvalResult:
        source = payload.decode("utf-8", errors="replace")
        sub_uid = commit_hash[:18]
        started = time.time()
        audit_emit(audit, "cloud_eval.start", commit_hash=commit_hash,
                   sub_uid=sub_uid, mode=mode, **payload_record(payload))
        try:
            try:
                res = cloud_orch.evaluate_submission(
                    source, round_id, sub_uid, mode=mode, audit=audit,
                    baseline_bundle=baseline_bundle)
            except TypeError as e:
                if ("unexpected keyword argument 'audit'" not in str(e)
                        and "unexpected keyword argument 'baseline_bundle'" not in str(e)):
                    raise
                try:
                    res = cloud_orch.evaluate_submission(
                        source, round_id, sub_uid, mode=mode, audit=audit)
                except TypeError as e2:
                    if "unexpected keyword argument 'audit'" not in str(e2):
                        raise
                    res = cloud_orch.evaluate_submission(
                        source, round_id, sub_uid, mode=mode)
        except Exception as e:
            audit_emit(audit, "cloud_eval.exception", commit_hash=commit_hash,
                       sub_uid=sub_uid, mode=mode, elapsed_s=time.time() - started,
                       error=str(e), error_type=type(e).__name__,
                       crashed=looks_like_crash(str(e)))
            return EvalResult("infra_dq", reason=f"cloud raised: {e}")
        if on_result is not None:
            try:
                on_result(commit_hash, res)
            except Exception:
                pass
        verdict = classify_cloud_result(res, ambiguous=ambiguous)
        err = str(res.get("error", ""))
        audit_emit(audit, "cloud_eval.result", commit_hash=commit_hash,
                   sub_uid=sub_uid, mode=mode, elapsed_s=time.time() - started,
                   outcome=verdict.outcome, score=verdict.score,
                   reason=verdict.reason, failed=bool(res.get("failed")),
                   error=err, crashed=looks_like_crash(err), raw_result=res)
        return verdict

    return evaluate
