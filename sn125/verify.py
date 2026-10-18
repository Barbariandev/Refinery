"""Public verification of Refinery (SN125) round records.

Everything the validator pays on is written into the per-round JSON it
publishes (R2 ``rounds/<id>.json`` and the HF mirror; the same files live in
``sn125/rounds/`` on the validator). This module lets ANYONE recompute the
settlement from those files alone, with no chain access and no validator
keys:

  hashes       every published source hashes to its recorded ``code_hash``;
               every commit in the FSM report has a revealed preimage or an
               honest "unrevealed" status; task/data manifest pins are stable.
  baseline     the rolling-best baseline each round scored against is the
               best confirmed anchor available BEFORE that round (genesis +
               confirmed frontier events only).
  bar          the adaptive frontier bar recorded for the round equals the
               bar recomputed from the prior confirmed events.
  settlement   the weights (miner shares + burn) and the frontier outcome
               (new event / pending / rejected) recomputed by the SAME
               ``Validator.compute_weights`` code, fed only the prior public
               rounds and this round's recorded scores, at this round's
               recorded timestamp, equal what was published.
  attestation  when the file is a signed R2 bundle, the sr25519 signature
               over the canonical payload verifies against the validator
               hotkey (and the hotkey matches the pinned one).
  checkpoints  every quartile/final checkpoint carries a sha256 (and a public
               uri once uploaded); with ``--fetch-base`` the blobs are
               downloaded and re-hashed; with ``--rescore`` the FINAL
               checkpoint's held-out loss is recomputed on this machine from
               the pinned shards and compared to the recorded loss.

Trust boundary, stated plainly: what CANNOT be recomputed by a third party
is the training run itself (the harness runs non-deterministic kernels, so
the run is not bit-replayable; the checkpoint trajectory and the hardware
provenance recorded per score are the audit artifact). Everything from the
checkpoint onward — the held-out loss, the bar, the frontier events, the
weights and the burn — is a pure function of public data and this code.

Usage:
    python -m sn125.verify [--rounds-dir DIR] [--round ROUND_ID ...]
                           [--fetch-base https://.../artifacts] [--rescore]
                           [--atol 0.01] [--json report.json]
Exit code 0 iff every check that could run passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import AUTHORIZED_VALIDATOR_HOTKEY, NETUID, validator_hotkey_configured


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class RoundReport:
    round_id: str
    timestamp: int
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", skipped: bool = False) -> None:
        self.checks.append(Check(name, ok, detail, skipped))

    @property
    def passed(self) -> bool:
        return all(c.ok or c.skipped for c in self.checks)

    def as_dict(self) -> dict:
        return {"round_id": self.round_id, "timestamp": self.timestamp,
                "passed": self.passed,
                "checks": [c.__dict__ for c in self.checks]}


# ── loading ─────────────────────────────────────────────────────────────────

def load_rounds(rounds_dir: Path) -> list[tuple[Path, dict]]:
    out = []
    for fp in sorted(rounds_dir.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("round_id") and "submissions" in d:
            out.append((fp, d))
    out.sort(key=lambda t: (int(t[1].get("timestamp", 0) or 0), t[1].get("round_id", "")))
    return out


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── individual checks ───────────────────────────────────────────────────────

def check_hashes(rd: dict, rep: RoundReport) -> None:
    sources = rd.get("sources") or {}
    code_hashes = rd.get("code_hashes") or {}
    bad = []
    for hk, entry in (rd.get("submissions") or {}).items():
        ch = str(entry.get("code_hash") or code_hashes.get(hk) or "")
        if not ch:
            continue
        src = sources.get(ch)
        if isinstance(src, str) and _sha256_text(src) != ch:
            bad.append(f"{hk[:12]}: source does not hash to {ch[:12]}")
    for ch, src in sources.items():
        if isinstance(src, str) and _sha256_text(src) != ch:
            bad.append(f"sources[{ch[:12]}] mis-keyed")
    rep.add("hashes.sources", not bad, "; ".join(bad) or f"{len(sources)} source(s) hash-consistent")

    report = rd.get("fsm_report") or {}
    subs = report.get("submissions") or []
    problems = []
    seen = 0
    for s in subs:
        if not isinstance(s, dict):
            continue
        ch = str(s.get("commit_hash") or "")
        status = str(s.get("status") or "")
        if not ch:
            continue
        seen += 1
        if ch in sources:
            if _sha256_text(sources[ch]) != ch:
                problems.append(f"commit {ch[:12]} preimage mismatch")
        elif status.lower() in ("scored",):
            problems.append(f"scored commit {ch[:12]} has no published source")
    rep.add("hashes.commits", not problems,
            "; ".join(problems) or f"{seen} commit(s) consistent with revealed sources",
            skipped=seen == 0)


def check_task_pins(rd: dict, prior: list[dict], rep: RoundReport) -> None:
    from .rolling_baseline import task_signature
    tasks = rd.get("tasks") or []
    if not tasks:
        rep.add("task.pinned", False, "round carries no task definition")
        return
    sigs = {t.get("task_id"): task_signature(t) for t in tasks}
    drift = []
    for p in prior:
        for t in p.get("tasks") or []:
            tid = t.get("task_id")
            if tid in sigs and task_signature(t) != sigs[tid]:
                drift.append(f"{tid}: signature differs from {p.get('round_id')}")
    manifest = [t.get("data_manifest", "") for t in tasks]
    # A changed signature is a legitimate task switch (new data manifest, new
    # budget): it is reported, and the baseline check below enforces that no
    # curve from the old task can serve as this round's bar.
    rep.add("task.pinned", True,
            (f"task switch: {'; '.join(sorted(set(drift)))}" if drift else
             f"task signature(s) {sigs} stable") + f"; manifest {str(manifest[0])[:16]}")


_TASK_DEFAULTS = {"parameter_count": "", "eval_every": 0, "eval_sequences": 0,
                  "warmup_fraction": 0.0, "total_steps": 0, "batch_size": 0,
                  "sequence_length": 0}


def _task_from_dict(t: dict):
    """TaskSpec from a round's (possibly abbreviated) task dict. Only the
    signature fields matter for replay; presentation fields get defaults."""
    from .training import TaskSpec
    kw = dict(_TASK_DEFAULTS)
    kw.update({k: v for k, v in t.items() if k in TaskSpec.__dataclass_fields__})
    return TaskSpec(**kw)


def _validator_for(rd: dict, rounds_dir: Path):
    """A wallet-less Validator whose task panel is the round's own."""
    from .neuron import Validator
    tasks = [_task_from_dict(t) for t in rd.get("tasks") or []]
    fr = rd.get("frontier_rewards") or {}
    v = Validator(wallet=None, set_weights=False, tasks=tasks or None,
                  rounds_dir=str(rounds_dir), backend="local",
                  burn_fraction_floor=None,
                  frontier_confirmation_runs=int(fr.get("confirmation_runs", 1) or 0))
    if "launch_timestamp" in fr:
        # 0 means "anchor on the first confirmed event" — replay must keep
        # that even after the operator later sets config.LAUNCH_TIMESTAMP.
        v.launch_timestamp = int(fr.get("launch_timestamp") or 0)
    # The published floor is authoritative for the replay (it may have been
    # an explicit operator override at the time).
    if fr.get("burn_fraction_floor") is not None:
        v.burn_fraction_floor = float(fr["burn_fraction_floor"])
        v._burn_floor_explicit = True
    return v


def _prior_dir(prior: list[tuple[Path, dict]], tmp: Path) -> Path:
    """A rounds dir holding ONLY the rounds published before the one being
    replayed — the validator's view of history at settlement time."""
    d = tmp / "prior"
    d.mkdir(parents=True, exist_ok=True)
    for fp, _ in prior:
        shutil.copy(fp, d / fp.name)
    return d


def check_baseline(rd: dict, prior_dir: Path, rep: RoundReport) -> None:
    from .rolling_baseline import load_rolling_best
    baselines = rd.get("baselines") or {}
    if not baselines or rd.get("genesis"):
        rep.add("baseline.rolling_best", True, "genesis / no baseline to check", skipped=True)
        return
    problems, cold = [], []
    for t in rd.get("tasks") or []:
        tid = t.get("task_id")
        rec = baselines.get(tid) or {}
        pts = rec.get("eval_points") or []
        if not pts:
            continue
        used = float(pts[-1][1])
        rb = load_rolling_best(prior_dir, t, exclude_round_id=rd.get("round_id"))
        if rb is None:
            cold.append(f"{tid}: cold start, no prior curve for this task signature "
                        f"(used {used:.4f})")
            continue
        if abs(rb.final_eval_loss - used) > 1e-9:
            problems.append(f"{tid}: used {used:.4f}, best prior anchor is "
                            f"{rb.final_eval_loss:.4f} ({rb.source_round_id})")
    if problems:
        rep.add("baseline.rolling_best", False, "; ".join(problems))
    elif cold:
        rep.add("baseline.rolling_best", True, "; ".join(cold), skipped=True)
    else:
        rep.add("baseline.rolling_best", True, "baseline equals the best confirmed prior anchor")


def check_settlement(rd: dict, prior_dir: Path, rep: RoundReport, *, tol: float = 1e-9) -> None:
    from .training import ScoreRecord
    fr = rd.get("frontier_rewards") or {}
    if not fr:
        rep.add("settlement.replay", True, "no frontier_rewards block (pre-mechanism round)", skipped=True)
        return
    v = _validator_for(rd, prior_dir)
    now = int(rd.get("timestamp", 0) or 0)
    results, sources = {}, {}
    src_by_hash = rd.get("sources") or {}
    for hk, entry in (rd.get("submissions") or {}).items():
        sc = entry.get("score") or {}
        try:
            results[hk] = ScoreRecord(
                float(sc.get("final_score", -1.0)), sc.get("components") or {},
                sc.get("task_scores") or {}, sc.get("best_hparams") or {},
                list(sc.get("failed_tasks") or []))
        except Exception as e:
            rep.add("settlement.replay", False, f"{hk[:12]}: unreadable score record ({e})")
            return
        ch = str(entry.get("code_hash") or "")
        if ch and isinstance(src_by_hash.get(ch), str):
            sources[hk] = src_by_hash[ch]
    confirmations = {}
    for key, rec in (rd.get("confirmations") or {}).items():
        ch = str(rec.get("code_hash") or "")
        if not ch:
            continue
        if rec.get("outcome") == "scored" and isinstance(rec.get("score"), dict):
            sc = rec["score"]
            confirmations[ch] = ScoreRecord(
                float(sc.get("final_score", -1.0)), sc.get("components") or {},
                sc.get("task_scores") or {}, sc.get("best_hparams") or {},
                list(sc.get("failed_tasks") or []))
        else:
            confirmations[ch] = {"outcome": rec.get("outcome", "infra_dq"),
                                 "reason": rec.get("reason", "")}
    # The bar first: recomputed from prior events only.
    events = v._load_frontier_events()
    bar = v._min_improvement_threshold(events, now)
    rec_bar = fr.get("min_improvement_threshold")
    bar_ok = rec_bar is not None and abs(float(rec_bar) - bar) <= 1e-12
    rep.add("bar.adaptive", bar_ok,
            f"recomputed {bar:.6f} vs recorded {rec_bar}")
    weights = v.compute_weights(results, sources=sources, confirmations=confirmations,
                                round_id=str(rd.get("round_id", "")), now=now)
    rf = (rd.get("refiner_task") or {}).get("frontier")
    if isinstance(rf, dict) and float(rf.get("emission_share", 0.0) or 0.0) > 0.0:
        # Challenge 2 pays its share from its own pool (replayed by
        # check_refiner_frontier); the published vector is the blend.
        from .roundsm.refiner_frontier import blend_weights
        weights = blend_weights(weights, rf.get("weights") or {}, float(rf["emission_share"]), v.BURN_KEY)
    recorded = rd.get("weights") or {}
    diffs = []
    for hk in set(weights) | set(recorded):
        a, b = float(weights.get(hk, 0.0)), float(recorded.get(hk, 0.0))
        if abs(a - b) > tol:
            diffs.append(f"{hk[:12]}: recomputed {a:.6f} vs published {b:.6f}")
    rep.add("settlement.weights", not diffs,
            "; ".join(diffs) or f"{len(weights)} weight(s) reproduced "
            f"(burn {weights.get(v.BURN_KEY, 0.0):.4f})")
    got = v._last_frontier_rewards
    ev_a, ev_b = got.get("new_event"), fr.get("new_event")
    same_event = ((ev_a is None and ev_b is None) or
                  (isinstance(ev_a, dict) and isinstance(ev_b, dict)
                   and ev_a.get("code_hash") == ev_b.get("code_hash")
                   and abs(float(ev_a.get("improvement", 0)) - float(ev_b.get("improvement", 0))) < 1e-12))
    pend_a = sorted(p.get("code_hash", "") for p in got.get("pending_confirmation") or [])
    pend_b = sorted(p.get("code_hash", "") for p in fr.get("pending_confirmation") or [])
    rep.add("settlement.frontier", same_event and pend_a == pend_b,
            f"new_event={'yes' if ev_b else 'no'} pending={len(pend_b)} reproduced"
            if same_event and pend_a == pend_b else
            f"recomputed event={ev_a and ev_a.get('code_hash', '')[:12]} pending={pend_a} "
            f"vs published event={ev_b and ev_b.get('code_hash', '')[:12]} pending={pend_b}")
    if "progress_nats" in fr:
        ok = abs(float(fr["progress_nats"]) - float(got.get("progress_nats", -1))) < 1e-9 and \
            abs(float(fr.get("payable_share", 0)) - float(got.get("payable_share", -1))) < 1e-9
        rep.add("settlement.progress", ok,
                f"progress {got.get('progress_nats', 0):.5f} nats, reference "
                f"{got.get('full_pay_reference_nats', 0):.4f}, payable {got.get('payable_share', 0):.4f}")


def check_attestation(rd: dict, rep: RoundReport) -> None:
    sig = rd.get("signature")
    bh = rd.get("bundle_hash")
    if not sig or not bh:
        rep.add("attestation.signature", True, "unsigned local round file", skipped=True)
        return
    from .neuron import _signing_payload
    payload = _signing_payload(rd)
    if hashlib.sha256(payload).hexdigest() != bh:
        rep.add("attestation.signature", False, "bundle_hash does not match the canonical payload")
        return
    hk = str(rd.get("validator_hotkey") or "")
    try:
        from substrateinterface import Keypair
        ok = Keypair(ss58_address=hk).verify(payload, bytes.fromhex(sig))
    except Exception as e:  # noqa: BLE001
        rep.add("attestation.signature", False, f"could not verify: {e}")
        return
    pinned = validator_hotkey_configured() and hk == AUTHORIZED_VALIDATOR_HOTKEY
    netuid_ok = int(rd.get("netuid", NETUID) or NETUID) == NETUID
    rep.add("attestation.signature", bool(ok) and netuid_ok,
            f"sr25519 signature {'valid' if ok else 'INVALID'} for {hk[:12]}"
            + ("" if pinned else " (not the pinned validator hotkey)")
            + ("" if netuid_ok else f" netuid {rd.get('netuid')} != {NETUID}"))


def _checkpoint_records(rd: dict) -> list[tuple[str, dict]]:
    out = []
    for section in ("submissions", "confirmations"):
        for key, entry in (rd.get(section) or {}).items():
            recs = entry.get("checkpoints") or []
            if not recs:
                for c in (entry.get("curves") or {}).values():
                    recs = recs + list((c or {}).get("checkpoints") or [])
            for r in recs:
                if isinstance(r, dict):
                    out.append((f"{section}/{key}", r))
    # Refiner task (Challenge 2): the submission run's quartile checkpoints.
    for ch, entry in ((rd.get("refiner_task") or {}).get("submissions") or {}).items():
        for r in (entry or {}).get("checkpoints") or []:
            if isinstance(r, dict):
                out.append((f"refiner_task/{ch[:12]}", r))
    return out


def check_refiner_task(rd: dict, rep: RoundReport) -> None:
    """The ``refiner_task`` block (Challenge 2): every entry's envelope is
    published under its commit hash and hashes to it; the envelope's weight
    commitment and source hash are what the entry says; the score is
    suite(submission) - suite(baseline); the entry's signature is recomputed
    from its own task/suite/decontam/student blocks; the emission share is
    within [0, 1] and the frontier section agrees with it; the block's
    suite/decontamination pins match what every entry was scored on."""
    block = rd.get("refiner_task")
    if not block:
        rep.add("refiner.block", True, "no refiner task in this round", skipped=True)
        return
    from .refiner.submission import EnvelopeError, parse_envelope
    from .roundsm.refiner_round import SCHEME, refiner_signature
    problems = []
    if block.get("scheme") not in (SCHEME, "refiner-publish-only-v1"):
        problems.append(f"scheme {block.get('scheme')!r}")
    share = float(block.get("emission_share", 0.0) or 0.0)
    if not (0.0 <= share <= 1.0):
        problems.append(f"emission_share {block.get('emission_share')} outside [0, 1]")
    frontier = block.get("frontier")
    if block.get("scheme") == "refiner-publish-only-v1" and share != 0.0:
        problems.append(f"emission_share {share} on a publish-only record")
    if isinstance(frontier, dict) and abs(float(frontier.get("emission_share", 0.0) or 0.0) - share) > 1e-12:
        problems.append(f"frontier pays {frontier.get('emission_share')} but the block says {share}")
    if share > 0.0 and not isinstance(frontier, dict):
        problems.append(f"emission_share {share} without a frontier section")
    sources = rd.get("sources") or {}
    subs = block.get("submissions") or {}
    suite_sha = str((block.get("suite") or {}).get("suite_sha256") or "")
    decontam_sha = str(block.get("decontam_sha256") or "")
    for ch, e in subs.items():
        tag = ch[:12]
        if e.get("commit_hash") != ch:
            problems.append(f"{tag}: entry keyed under another commit")
        text = sources.get(ch)
        if not isinstance(text, str):
            problems.append(f"{tag}: envelope not published")
            continue
        if _sha256_text(text) != ch:
            problems.append(f"{tag}: envelope does not hash to its commit")
        try:
            env = parse_envelope(text)
        except EnvelopeError as ex:
            problems.append(f"{tag}: envelope invalid ({ex})")
            continue
        if _sha256_text(env["source"]) != e.get("source_sha256"):
            problems.append(f"{tag}: source_sha256 is not the envelope's source")
        want_w = ({"sha256": env["weights"]["sha256"], "bytes": env["weights"]["bytes"]} if env["weights"] else None)
        if (e.get("weights") or None) != want_w:
            problems.append(f"{tag}: weights commitment differs from the envelope")
        bench = e.get("bench") or {}
        if isinstance(bench.get("submission"), (int, float)) and isinstance(bench.get("baseline"), (int, float)) \
                and isinstance(e.get("score"), (int, float)):
            if abs(float(e["score"]) - (float(bench["submission"]) - float(bench["baseline"]))) > 1e-6:
                problems.append(f"{tag}: score != suite(submission) - suite(baseline)")
        got_suite = str((e.get("suite") or {}).get("suite_sha256") or "")
        if got_suite and suite_sha and got_suite != suite_sha:
            problems.append(f"{tag}: scored on suite {got_suite[:12]}, block pins {suite_sha[:12]}")
        got_dec = str((e.get("entry_stream") or {}).get("exclusions_sha256") or "")
        if got_dec and decontam_sha and got_dec != decontam_sha:
            problems.append(f"{tag}: decontamination list {got_dec[:12]} is not the pinned {decontam_sha[:12]}")
        if "signature" in e and refiner_signature(e) != (e.get("signature") or ""):
            problems.append(f"{tag}: signature does not recompute from the entry's task/suite/student blocks")
    sigs = sorted({str(e.get("signature") or "") for e in subs.values()} - {""})
    if len(sigs) > 1:
        problems.append(f"{len(sigs)} different task signatures in one round")
    rep.add("refiner.block", not problems,
            "; ".join(problems) or f"{len(subs)} refiner entr{'y' if len(subs) == 1 else 'ies'} consistent "
            f"({'publish-only' if share == 0 else f'pool {share:.0%} of emission'}, "
            f"signature {sigs[0] if sigs else (block.get('signature') or 'n/a')})")


def check_refiner_frontier(rd: dict, prior_dir: Path, rep: RoundReport, *, tol: float = 1e-9) -> None:
    """Replay the refiner frontier (roundsm/refiner_frontier.settle) from the
    prior public records, this round's refiner entries and its confirmation
    runs: the bar, the new event, the pending list, the pool's progress /
    payable share and its weight map must all reproduce."""
    block = rd.get("refiner_task")
    fr = (block or {}).get("frontier") if isinstance(block, dict) else None
    if not isinstance(fr, dict):
        rep.add("refiner.frontier", True, "no refiner frontier in this round", skipped=True)
        return
    from .roundsm import refiner_frontier as rff
    now = int(fr.get("timestamp", 0) or rd.get("timestamp", 0) or 0)
    if abs(now - int(rd.get("timestamp", 0) or 0)) > 600:
        rep.add("refiner.frontier", False, f"frontier settled at {now}, round recorded at {rd.get('timestamp')}")
        return
    cfg = {"signature": fr.get("signature", block.get("signature", "")),
           "emission_share": float(fr.get("emission_share", 0.0) or 0.0),
           "baseline_score": fr.get("baseline_score", block.get("baseline_score"))}
    confirmations = {}
    for rec in (block.get("confirmations") or {}).values():
        ch = str(rec.get("commit_hash") or "")
        if ch:
            confirmations[ch] = {"outcome": rec.get("outcome", "infra_dq"), "reason": rec.get("reason", ""),
                                 "result": rec if rec.get("outcome") == "scored" else None}
    got = rff.settle(cfg, block.get("submissions") or {}, confirmations, rounds_dir=prior_dir,
                     round_id=str(rd.get("round_id", "")), now=now,
                     burn_floor=float(fr.get("burn_fraction_floor", 0.0) or 0.0),
                     launch_timestamp=int(fr.get("launch_timestamp", 0) or 0))
    problems = []
    if abs(got["min_improvement_threshold"] - float(fr.get("min_improvement_threshold", -1))) > 1e-12:
        problems.append(f"bar recomputed {got['min_improvement_threshold']:.6f} vs recorded "
                        f"{fr.get('min_improvement_threshold')}")
    ev_a, ev_b = got.get("new_event"), fr.get("new_event")
    if (ev_a is None) != (ev_b is None) or (ev_a and (
            ev_a.get("commit_hash") != ev_b.get("commit_hash")
            or abs(float(ev_a.get("improvement", 0)) - float(ev_b.get("improvement", 0))) > 1e-12)):
        problems.append(f"new_event recomputed {ev_a and ev_a.get('commit_hash', '')[:12]} vs recorded "
                        f"{ev_b and ev_b.get('commit_hash', '')[:12]}")
    pend_a = sorted(p.get("commit_hash", "") for p in got.get("pending_confirmation") or [])
    pend_b = sorted(p.get("commit_hash", "") for p in fr.get("pending_confirmation") or [])
    if pend_a != pend_b:
        problems.append(f"pending recomputed {[p[:12] for p in pend_a]} vs recorded {[p[:12] for p in pend_b]}")
    for key in ("progress", "payable_share", "burned_share"):
        if abs(float(got.get(key, 0.0)) - float(fr.get(key, -1.0) or 0.0)) > tol:
            problems.append(f"{key} recomputed {got.get(key, 0.0):.6f} vs recorded {fr.get(key)}")
    rec_w = fr.get("weights") or {}
    for hk in set(got["weights"]) | set(rec_w):
        if abs(float(got["weights"].get(hk, 0.0)) - float(rec_w.get(hk, 0.0))) > tol:
            problems.append(f"pool weight {hk[:12]}: recomputed {got['weights'].get(hk, 0.0):.6f} "
                            f"vs recorded {rec_w.get(hk, 0.0)}")
    rep.add("refiner.frontier", not problems,
            "; ".join(problems) or
            f"bar {got['min_improvement_threshold']:.4f}, new_event={'yes' if ev_b else 'no'}, "
            f"pending={len(pend_b)}, progress {got['progress']:.4f}, payable {got['payable_share']:.4f} reproduced")


def check_checkpoints(rd: dict, rep: RoundReport, *, fetch_base: str = "",
                      workdir: Path | None = None) -> None:
    recs = _checkpoint_records(rd)
    if not recs:
        rep.add("checkpoints.recorded", True, "no checkpoints in this round", skipped=True)
        return
    missing = [k for k, r in recs if not r.get("sha256")]
    rep.add("checkpoints.recorded", not missing,
            f"{len(recs)} checkpoint record(s), {sum(1 for _, r in recs if r.get('uri'))} with a public uri"
            + (f"; {len(missing)} without sha256" if missing else ""))
    if not fetch_base:
        return
    import urllib.request
    bad, fetched = [], 0
    for key, r in recs:
        uri = str(r.get("uri") or "")
        if not uri.startswith("r2://"):
            continue
        _, _, rest = uri.partition("r2://")
        _bucket, _, obj = rest.partition("/")
        url = fetch_base.rstrip("/") + "/" + obj
        wd = workdir or Path(tempfile.mkdtemp(prefix="sn125_verify_"))
        wd.mkdir(parents=True, exist_ok=True)
        dest = wd / obj.replace("/", "_")
        try:
            with urllib.request.urlopen(url, timeout=600) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            got = _sha256_file(dest)
            fetched += 1
            if got != r.get("sha256"):
                bad.append(f"{key} q{int(round(float(r.get('pct', 0)) * 100))}: sha {got[:12]} != {str(r.get('sha256'))[:12]}")
            r["_local_path"] = str(dest)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{key}: fetch failed ({e})")
    rep.add("checkpoints.fetched", not bad,
            "; ".join(bad) or f"{fetched} blob(s) downloaded and sha256-verified",
            skipped=fetched == 0 and not bad)


def check_rescore(rd: dict, rep: RoundReport, *, atol: float, device: str = "cuda") -> None:
    """Recompute the FINAL checkpoint's held-out loss from the pinned shards."""
    from .engine.score_worker import export_eval_shard, score_checkpoint_isolated
    from .training import _load_real_data
    tasks = {t.get("task_id"): t for t in rd.get("tasks") or []}
    seed_manifest = rd.get("public_seed_manifest") or {}
    seeds = seed_manifest.get("main") or [42]
    done, problems = 0, []
    for key, r in _checkpoint_records(rd):
        if float(r.get("pct", 0) or 0) != 1.0 or not r.get("_local_path"):
            continue
        section, hk = key.split("/", 1)
        entry = (rd.get(section) or {}).get(hk) or {}
        task_scores = ((entry.get("score") or {}).get("task_scores") or {})
        tid = r.get("task_id") or next(iter(task_scores), None) or next(iter(tasks), None)
        t = tasks.get(tid)
        if not t:
            continue
        task = _task_from_dict(t)
        recorded = (task_scores.get(tid) or {}).get("sub_final_loss")
        try:
            with tempfile.TemporaryDirectory(prefix="sn125_rescore_") as td:
                eval_batches = max(1, task.eval_sequences // task.batch_size)
                split = "heldout" if task.dataset == "fineweb-edu-shards" else "train"
                data = _load_real_data(task.dataset, task.sequence_length, task.batch_size,
                                       eval_batches, int(seeds[0]) + 1_000_000, split=split,
                                       tokenizer_name=task.model_config)
                shard = str(Path(td) / "heldout.safetensors")
                export_eval_shard(list(data), shard, metadata={"task_id": tid})
                out = score_checkpoint_isolated(r["_local_path"], shard, f"hf:{task.model_config}",
                                                use_amp=True, device=device, timeout=3600)
            loss = float(out["eval_loss"])
            done += 1
            if recorded is None or abs(loss - float(recorded)) > atol:
                problems.append(f"{key}: rescored {loss:.5f} vs recorded {recorded} (atol {atol})")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{key}: rescore failed ({e})")
    rep.add("checkpoints.rescored", not problems,
            "; ".join(problems) or f"{done} final checkpoint(s) rescored within {atol} nats",
            skipped=done == 0 and not problems)


# ── driver ──────────────────────────────────────────────────────────────────

def verify_rounds(rounds_dir: Path, only: set[str] | None = None, *,
                  fetch_base: str = "", rescore: bool = False, atol: float = 0.01,
                  device: str = "cuda") -> list[RoundReport]:
    rounds = load_rounds(rounds_dir)
    reports: list[RoundReport] = []
    with tempfile.TemporaryDirectory(prefix="sn125_verify_") as td:
        tmp = Path(td)
        for i, (fp, rd) in enumerate(rounds):
            rid = str(rd.get("round_id", fp.stem))
            if only and rid not in only:
                continue
            rep = RoundReport(rid, int(rd.get("timestamp", 0) or 0))
            prior = rounds[:i]
            check_hashes(rd, rep)
            check_task_pins(rd, [d for _, d in prior], rep)
            prior_dir = _prior_dir(prior, tmp / f"r{i}")
            try:
                check_baseline(rd, prior_dir, rep)
            except Exception as e:  # noqa: BLE001
                rep.add("baseline.rolling_best", False, f"error: {e}")
            try:
                check_settlement(rd, prior_dir, rep)
            except Exception as e:  # noqa: BLE001
                rep.add("settlement.replay", False, f"error: {e}")
            check_attestation(rd, rep)
            check_refiner_task(rd, rep)
            try:
                check_refiner_frontier(rd, prior_dir, rep)
            except Exception as e:  # noqa: BLE001
                rep.add("refiner.frontier", False, f"error: {e}")
            check_checkpoints(rd, rep, fetch_base=fetch_base, workdir=tmp / f"blobs{i}")
            if rescore:
                check_rescore(rd, rep, atol=atol, device=device)
            reports.append(rep)
    return reports


def format_report(reports: list[RoundReport]) -> str:
    lines = []
    for rep in reports:
        lines.append(f"{'PASS' if rep.passed else 'FAIL'} {rep.round_id} "
                     f"({time.strftime('%Y-%m-%d %H:%M', time.gmtime(rep.timestamp))} UTC)")
        for c in rep.checks:
            mark = "skip" if c.skipped else ("ok  " if c.ok else "FAIL")
            lines.append(f"    [{mark}] {c.name}: {c.detail}")
    n_pass = sum(1 for r in reports if r.passed)
    lines.append(f"{n_pass}/{len(reports)} round(s) verified")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds-dir", default=str(Path(__file__).resolve().parent / "rounds"))
    ap.add_argument("--round", action="append", default=[], help="verify only these round ids")
    ap.add_argument("--fetch-base", default=os.environ.get("SN125_ARTIFACT_PUBLIC_BASE", ""),
                    help="public https base of the R2 artifact bucket (checkpoints/... keys)")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the final checkpoints' held-out loss (needs the "
                         "pinned shards staged locally and a GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--atol", type=float, default=0.01,
                    help="rescore tolerance in nats (cross-GPU kernel differences)")
    ap.add_argument("--json", default="", help="write the machine-readable report here")
    args = ap.parse_args(argv)
    reports = verify_rounds(Path(args.rounds_dir), set(args.round) or None,
                            fetch_base=args.fetch_base, rescore=args.rescore,
                            atol=args.atol, device=args.device)
    print(format_report(reports))
    if args.json:
        Path(args.json).write_text(json.dumps([r.as_dict() for r in reports], indent=2))
    return 0 if reports and all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
