"""Refiner task (Challenge 2) inside the round loop.

Everything the round loop needs to carry a second task, kept in one place so
the Challenge 1 path reads as before:

  * :func:`load_refiner_round_config` — resolve and pin the inputs every refiner
    eval ships to its box (suite, decontamination list, identity baseline).
    ``None`` when the task is off, which makes every hook below a no-op.
  * :func:`refiner_job_for` — turn a revealed payload into the job description
    ``cloud.evaluate_submission(refiner_job=...)`` takes (or ``None`` for an
    optimizer submission).
  * :func:`build_refiner_block` — the ``refiner_task`` section of the round
    record: scores and provenance of this round's refiner evaluations. The
    round loop adds the task's ``frontier`` (roundsm/refiner_frontier.py) and
    its confirmation runs to it.

Refiner results never enter ``compute_weights``: the task has its own score
unit (suite points, higher is better), its own frontier and its own pool —
``emission_share`` of miner emission (config.REFINER_EMISSION_SHARE, 0.5) —
blended into the round's weights by ``refiner_frontier.blend_weights``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from .. import settings
from ..refiner.refproc_limits import REFINER_TASK_ID
from ..refiner.submission import EnvelopeError, looks_like_envelope, parse_envelope

log = logging.getLogger(__name__)

SCHEME = "refiner-task-v2"   # v1 was publish-only (no frontier section)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


PINNED_DIR = Path(__file__).resolve().parent.parent / "refiner" / "pinned"       # suite + decontamination list
BASELINES_DIR = Path(__file__).resolve().parent.parent / "refiner" / "baselines"  # identity baselines per seed


def load_refiner_round_config(data_dir: str | os.PathLike | None = None, env: dict | None = None) -> dict | None:
    """Resolve the Refiner-task inputs once, at validator start.

    The task is on by default, paying ``emission_share`` of miner emission from
    its own frontier (refiner_frontier.py); its inputs resolve, in order,
    from the environment, the shard directory, and the copies shipped in the
    package (``sn125/refiner/pinned``, ``sn125/refiner/baselines``), so a stock
    validator needs no configuration. Two failure modes:

    * ``SN125_REFINER_TASK`` unset (default-on): anything that would make the
      task unscorable — missing inputs, a pinned baseline the box would reject
      — is logged and the task stays OFF for this process; Challenge 1 is
      never taken down by Challenge 2's configuration.
    * ``SN125_REFINER_TASK=1`` (explicit): the same conditions stop the
      validator, as does a non-zero emission share in either mode.
    """
    e = os.environ if env is None else env
    if not settings.refiner_task_enabled(env):
        return None
    share = settings.refiner_emission_share(env)
    if not (0.0 <= share <= 1.0):
        raise ValueError(f"{settings.REFINER_EMISSION_SHARE_ENV}={share}: the refiner pool's share of miner "
                         "emission must be within [0, 1] (0 = publish-only; docs/REFINER_TASK.md §9)")
    try:
        return _resolve_refiner_round_config(data_dir, e, share)
    except (OSError, ValueError) as exc:
        if settings.refiner_task_explicit(env):
            raise
        log.error("refiner task stays OFF for this validator: %s (set %s=1 to make this fatal, =0 to silence)",
                  exc, settings.REFINER_TASK_ENV)
        return None


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_refiner_round_config(data_dir, e: dict, share: float) -> dict:
    from ..bench.suite import default_suite_path, load_suite   # lazy: torch
    from ..fineweb import default_data_dir
    from ..refiner.entries import exclusions_path
    explicit = settings.refiner_task_explicit(e)
    base = Path(data_dir) if data_dir else default_data_dir()
    suite_path = Path(e.get(settings.REFINER_SUITE_ENV, "") or
                      _first_existing(default_suite_path(base), PINNED_DIR / default_suite_path(base).name))
    if not suite_path.exists():
        raise FileNotFoundError(f"refiner task enabled but the benchmark suite is missing: {suite_path}")
    suite = load_suite(suite_path)
    want_name = exclusions_path(base, "train", suite["suite_id"]).name
    decontam_path = Path(e.get(settings.REFINER_DECONTAM_ENV, "") or
                         _first_existing(exclusions_path(base, "train", suite["suite_id"]), PINNED_DIR / want_name))
    if not decontam_path.exists():
        raise FileNotFoundError(
            f"refiner task enabled but the decontamination list is missing: {decontam_path} "
            "(build it with scripts/refiner_decontam.py --write-exclusions)")
    if decontam_path.name != want_name:
        raise ValueError(f"decontamination list must be named {want_name} (the entry stream looks it up by name)")
    cfg = {
        "task_id": REFINER_TASK_ID, "emission_share": float(share),
        "suite_path": str(suite_path), "suite_id": suite["suite_id"], "suite_sha256": suite["suite_sha256"],
        "decontam_path": str(decontam_path), "decontam_sha256": _sha256_file(decontam_path),
        # Pinned identity baseline: its file, its suite score (the frontier
        # until an event confirms) and the task signature every result must
        # share to be ranked against it (refiner_signature of the baseline).
        "baseline_path": "", "baseline_sha256": "", "baseline_score": None, "signature": "",
        # The document index is derived from the shards (a box without it
        # rebuilds it); shipping ours saves the scan and pins the same file.
        "doc_index_path": "",
    }
    from ..refiner.entries import doc_index_path as _doc_index_path
    di = _doc_index_path(base, "train")
    if di.exists():
        cfg["doc_index_path"] = str(di)
    baseline = e.get(settings.REFINER_BASELINE_JSON_ENV, "").strip()
    operator_pinned = bool(baseline)
    if not baseline:
        # The packaged identity baseline for the seed the evals run at.
        seed = int(str(e.get(settings.REFINER_EVAL_SEED_ENV, "")).strip() or 42)
        packaged = BASELINES_DIR / f"{REFINER_TASK_ID}_{suite['suite_id']}_seed{seed}.json"
        if packaged.exists():
            baseline = str(packaged)
        elif not explicit:
            raise FileNotFoundError(f"no packaged identity baseline for seed {seed} ({packaged})")
    if baseline:
        bp = Path(baseline)
        try:
            payload = json.loads(bp.read_text())   # raises on a missing/corrupt file
            got = ((payload.get("suite") or {}).get("suite_sha256"),
                   (payload.get("entry_stream") or {}).get("exclusions_sha256"))
            if got != (cfg["suite_sha256"], cfg["decontam_sha256"]) or "bench" not in (payload.get("baseline") or {}):
                raise ValueError(f"{bp} is not an identity-baseline result for this suite and decontamination list")
            why = pinned_baseline_regime_mismatch(payload, base, env=e)
            if why:
                raise ValueError(
                    f"{bp} cannot serve as the pinned baseline for this validator's refiner evals ({why}): "
                    "the box would reject it after staging and every refiner eval would end as a harness fault")
        except (OSError, ValueError) as exc:
            if operator_pinned or not explicit:
                raise
            # Explicitly enabled with custom inputs the packaged baseline does
            # not fit: keep the old behaviour (per-eval baseline) and say so.
            log.warning("refiner task: packaged baseline not usable here (%s)", exc)
            baseline = ""
        else:
            cfg["baseline_path"], cfg["baseline_sha256"] = str(bp), _sha256_file(bp)
            cfg["baseline_score"] = float(payload["baseline"]["bench"]["score"])
            cfg["signature"] = refiner_signature(payload)
    if not baseline:
        if share > 0.0:
            # The pool pays against a frontier; without a pinned baseline there
            # is no frontier score and no stable task signature to key it on.
            raise ValueError(f"the refiner pool pays {share:.0%} of emission but no pinned identity baseline "
                             f"is usable here: set {settings.REFINER_BASELINE_JSON_ENV}, or "
                             f"{settings.REFINER_EMISSION_SHARE_ENV}=0 for publish-only")
        log.warning("refiner task: no %s set; every refiner eval will train its own identity baseline "
                    "(two training runs per submission)", settings.REFINER_BASELINE_JSON_ENV)
    log.info("refiner task ENABLED (%s): suite %s %s, decontam %s, baseline %s (suite %.4f), signature %s",
             f"pool {share:.0%} of emission" if share > 0 else "publish-only",
             cfg["suite_id"], cfg["suite_sha256"][:12], cfg["decontam_sha256"][:12],
             cfg["baseline_sha256"][:12] or "per-eval", cfg["baseline_score"] or float("nan"), cfg["signature"] or "n/a")
    return cfg


def pinned_baseline_regime_mismatch(payload: dict, data_dir: Path, env: dict | None = None) -> str:
    """Why a pinned identity baseline would be refused by ``refiner-eval
    --baseline-json`` on a box this validator rents ('' when it matches).

    Mirrors ``refiner_eval._baseline_mismatch`` from the validator's side: the
    eval seed, the training budget and step count, the pinned student
    optimizer, the data manifest and the execution regime (sandbox flavour of
    the provider chain, optproc, AMP, force_gc) all have to be the baseline's.
    A mismatch found here costs a log line; found on the box it costs a full
    20 h rental per submission and a round that publishes nothing."""
    e = os.environ if env is None else env
    from ..fineweb import MANIFEST_NAME, PROD_BUDGET_SECONDS, PROD_CONFIRM_STEPS, load_manifest
    from ..refiner.task import PINNED_STUDENT_SHA256, STUDENT_OPTIMIZER_ENV
    task = payload.get("task") or {}
    ex = payload.get("execution") or {}
    problems: list[str] = []

    seed = int(str(e.get(settings.REFINER_EVAL_SEED_ENV, "")).strip() or 42)
    if task.get("seed") != seed:
        problems.append(f"seed {task.get('seed')} vs eval seed {seed} (set {settings.REFINER_EVAL_SEED_ENV} "
                        "to the baseline's seed or pin a baseline made at this one)")
    budget = float(str(e.get("SN125_PROD_EVAL_BUDGET", "")).strip() or PROD_BUDGET_SECONDS)
    steps = int(str(e.get("SN125_PROD_EVAL_STEPS", "")).strip() or PROD_CONFIRM_STEPS)
    if float(task.get("compute_budget_seconds") or 0) != budget or int(task.get("total_steps") or 0) != steps:
        problems.append(f"budget {task.get('compute_budget_seconds')}s/{task.get('total_steps')} steps "
                        f"vs {budget:.0f}s/{steps} steps")

    override = str(e.get(STUDENT_OPTIMIZER_ENV, "")).strip()
    want_student = hashlib.sha256(Path(override).read_bytes()).hexdigest() if override else PINNED_STUDENT_SHA256
    if (payload.get("student_optimizer") or {}).get("sha256") != want_student:
        problems.append("student optimizer differs from the pinned one")

    mf = Path(data_dir) / MANIFEST_NAME
    if mf.exists():
        try:
            want_manifest = load_manifest(data_dir).get("manifest_hash")
        except Exception:  # noqa: BLE001 - a broken manifest is reported by stage-shards, not here
            want_manifest = None
        if want_manifest and task.get("data_manifest") != want_manifest:
            problems.append("data manifest differs from the local shards")

    from ..cloud import provider_isolation
    flavours = {"seccomp" if provider_isolation(p) == "container" else "namespaces"
                for p in settings.cloud_provider_chain(env=e)}
    if ex.get("sandbox_mode") not in flavours:
        problems.append(f"sandbox flavour {ex.get('sandbox_mode')!r} vs the provider chain's {sorted(flavours)}")
    elif len(flavours) > 1:
        log.warning("refiner task: the provider chain mixes sandbox flavours %s; the pinned baseline serves "
                    "only %r boxes (docs/REFINER_TASK.md §9)", sorted(flavours), ex.get("sandbox_mode"))
    # The box regime comes from the launch profile, not the validator's own env:
    # optproc is fixed on (LAUNCH_EVAL_ENV), AMP is on (no --no-amp in the CLI
    # the orchestrator builds), and SN125_FORCE_GC is not forwarded to boxes.
    want_exec = {"optproc_enabled": settings.LAUNCH_EVAL_ENV.get(settings.USE_OPTPROC_ENV) == "1",
                 "use_amp": True, "force_gc": False}
    for k, v in want_exec.items():
        if ex.get(k) != v:
            problems.append(f"execution.{k} {ex.get(k)!r} vs {v!r}")
    return "; ".join(problems)


def refiner_job_for(payload: bytes, cfg: dict | None) -> tuple[str, dict] | None:
    """``(refiner source, job)`` for a refiner envelope, ``None`` for anything
    else. Only called for payloads that passed the gate, so a malformed
    envelope here is treated as not-a-refiner and falls through."""
    if cfg is None or not looks_like_envelope(payload):
        return None
    try:
        env = parse_envelope(payload)
    except EnvelopeError:
        return None
    job = {k: cfg[k] for k in ("task_id", "suite_path", "suite_sha256", "decontam_path", "decontam_sha256",
                               "baseline_path")}
    job["doc_index_path"] = cfg.get("doc_index_path", "")
    job["weights"] = env["weights"]
    return env["source"], job


def is_refiner_payload(payload: bytes | None) -> bool:
    return payload is not None and looks_like_envelope(payload)


_RESULT_KEYS = ("components", "paired", "task_scores", "entry_stream", "student_optimizer", "suite", "execution",
                "task", "wall_seconds", "refiner_weights_archive", "gates")


def refiner_signature(result: dict) -> str:
    """Stable hash of everything that makes two refiner scores comparable, read
    from what the box actually ran (the result's own ``task``, ``suite``,
    ``entry_stream`` and ``student_optimizer`` blocks): the task id and student
    model, the training budget, batch and sequence length, the data manifest
    and seed, the benchmark suite, the decontamination list and the pinned
    student optimizer. The optimizer task's ``task_signature`` knows none of
    these, so a refiner entry carries its own. Two entries with different
    signatures were scored on different tasks and must never be ranked
    against each other; '' when the result lacks the blocks (a failed eval)."""
    task = result.get("task") or {}
    suite = result.get("suite") or {}
    stream = result.get("entry_stream") or {}
    student = result.get("student_optimizer") or {}
    if not (task and suite.get("suite_sha256") and student.get("sha256")):
        return ""
    fields = {
        "task_id": str(task.get("task_id", "")),
        "model_config": str(task.get("model_config", "")),
        "compute_budget_seconds": float(task.get("compute_budget_seconds", 0.0) or 0.0),
        "total_steps": int(task.get("total_steps", 0) or 0),
        "batch_size": int(task.get("batch_size", 0) or 0),
        "sequence_length": int(task.get("sequence_length", 0) or 0),
        "dataset": str(task.get("dataset", "")),
        "data_manifest": str(task.get("data_manifest", "")),
        "seed": int(task.get("seed", 0) or 0),
        "suite_sha256": str(suite.get("suite_sha256", "")),
        "decontam_sha256": str(stream.get("exclusions_sha256", "")),
        "max_entry_tokens": int(stream.get("max_entry_tokens", 0) or 0),
        "student_optimizer_sha256": str(student.get("sha256", "")),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def build_refiner_block(fsm, sink: dict, cfg: dict | None, *, scored_status, provenance=None):
    """``(block, sources)``: the round record's ``refiner_task`` section and the
    envelope texts to publish next to the optimizer sources (keyed
    ``"<hotkey>#refiner"`` so one hotkey can hold both). ``(None, {})`` when
    the task is off — the record is then exactly what it was before the task
    existed. With the task on the block is always present (``submissions`` may
    be empty): the frontier the round loop adds to it carries state — pending
    confirmations, decaying credit — that every round must publish."""
    if cfg is None:
        return None, {}
    subs, sources = {}, {}
    for sub in fsm.submissions.values():
        if sub.status is not scored_status or not is_refiner_payload(sub.payload):
            continue
        raw = sink.get(sub.commit_hash, {}) or {}
        text = sub.payload.decode("utf-8", errors="replace")
        try:
            env = parse_envelope(text)
        except EnvelopeError:
            continue
        entry = {"hotkey": sub.hotkey, "commit_hash": sub.commit_hash, "score": sub.score,
                 "weights": ({"sha256": env["weights"]["sha256"], "bytes": env["weights"]["bytes"]}
                             if env["weights"] else None),
                 "source_sha256": hashlib.sha256(env["source"].encode()).hexdigest()}
        for k in _RESULT_KEYS:
            if raw.get(k) is not None:
                entry[k] = raw[k]
        s = raw.get("submission") or {}
        b = raw.get("baseline") or {}
        entry["bench"] = {"submission": (s.get("bench") or {}).get("score"),
                          "baseline": (b.get("bench") or {}).get("score"),
                          "submission_checkpoint_sha256": (s.get("bench") or {}).get("checkpoint_sha256"),
                          "baseline_source": b.get("source")}
        entry["refiner_stats"] = (s.get("curve") or {}).get("refiner_stats")
        entry["signature"] = refiner_signature(raw)
        entry["checkpoints"] = [{k: c.get(k) for k in ("pct", "step", "sha256", "uri") if c.get(k) is not None}
                                for c in ((s.get("curve") or {}).get("checkpoints") or []) if isinstance(c, dict)]
        if provenance is not None:
            p = provenance(raw)
            if p:
                entry["provenance"] = p
        subs[sub.commit_hash] = entry
        sources[f"{sub.hotkey}#refiner"] = text
    sigs = sorted({e["signature"] for e in subs.values() if e.get("signature")})
    block = {
        "scheme": SCHEME, "task_id": cfg["task_id"], "emission_share": float(cfg.get("emission_share", 0.0) or 0.0),
        "suite": {"suite_id": cfg["suite_id"], "suite_sha256": cfg["suite_sha256"]},
        "decontam_sha256": cfg["decontam_sha256"], "baseline_sha256": cfg["baseline_sha256"],
        "baseline_score": cfg.get("baseline_score"),
        # The pinned baseline's signature keys the frontier; a reader ranks only within it.
        "signature": cfg.get("signature") or (sigs[0] if len(sigs) == 1 else ""),
        "signatures": sigs,
        "submissions": subs,
    }
    return block, sources
