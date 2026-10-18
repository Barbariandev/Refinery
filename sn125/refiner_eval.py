"""Refiner-task (Challenge 2) submission evaluator.

Mirror of prod_eval.py for refiner submissions. One command, fails closed:

  1. gate the refiner source (same C2/C8 gate as optimizer.py) and validate the
     safetensors weight file against the parameter cap WITHOUT loading it;
  2. train the student with the pinned student optimizer twice inside the
     isolated worker — identity baseline (plain next-token recipe through the
     plan protocol) and the submission (refiner in its own sandboxed process);
  3. score BOTH final checkpoints on the pre-tokenized benchmark suite with
     trusted code; score = suite score(submission) - suite score(baseline).

Held-out loss curves are kept in the result as a diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import settings
from .bench.score import paired_delta, score_checkpoint_on_suite
from .bench.suite import default_suite_path, load_suite
from .fineweb import PROD_CONFIRM_STEPS, SHARD_DATASET_NAME
from .gate import validate_single_source
from .references import extract_hparams
from .refiner.entries import DEFAULT_MAX_ENTRY_TOKENS, EntryStream, ensure_doc_index, exclusions_path
from .refiner.refproc import DEFAULT_PLAN_TIMEOUT_S, RefProcError, extract_refiner_capabilities, validate_refiner_weights
from .refiner.submission import RESULT_MARKER, compact_result, parse_result_line  # noqa: F401
from .refiner.task import REFINER_TASK_ID, load_student_optimizer, make_refiner_task, refiner_run_config
from .training import TrainingCurve, evaluate_submission_isolated

REFINER_ALLOW_INPROC_ENV = "SN125_REFINER_ALLOW_INPROC_OPT"  # dev-only: skip the optproc requirement


_MINER_FAULT_MARKERS = ("LessonError", "RefProcError", "RefProcTimeout", "Loss diverged")


def _fault_of(error: str) -> str:
    """Who a failed submission run is attributed to. Only protocol violations,
    refiner-process failures and a diverged loss are the miner's; anything else
    (I/O, CUDA, sandbox, timeouts of the worker itself) is the harness's until
    someone has looked — a harness fault must never settle as a miner score."""
    error = error or ""
    # The parent validated and staged the weight file; a worker that cannot
    # see it is a staging problem, not something the miner shipped.
    if "refiner weights not found" in error:
        return "harness"
    return "miner" if any(m in error for m in _MINER_FAULT_MARKERS) else "harness"


def _fail(error: str, fault: str = "harness", **extra) -> dict:
    out = {"failed": True, "score": -1.0, "error": error, "fault": fault, "task_id": REFINER_TASK_ID}
    out.update(extra)
    return out


def _attested_final_sha(curve_d: dict | None) -> str:
    for c in (curve_d or {}).get("checkpoints") or []:
        if float(c.get("pct", 0)) == 1.0:
            return str(c.get("sha256") or "")
    return ""


def _baseline_mismatch(payload: dict, meta: dict) -> str:
    """Why a previous refiner-eval result cannot stand in as this run's baseline
    ('' when it can). Everything that shapes the baseline run must be identical."""
    if not isinstance(payload.get("baseline"), dict) or not payload["baseline"].get("bench"):
        return "not a refiner-eval result (no baseline.bench)"
    if payload["baseline"]["bench"].get("suite_sha256") != meta["suite"]["suite_sha256"]:
        return "scored on a different suite"
    for block in ("task", "entry_stream"):
        if payload.get(block) != meta[block]:
            diff = sorted(k for k in set(payload.get(block) or {}) | set(meta[block])
                          if (payload.get(block) or {}).get(k) != meta[block].get(k))
            return f"different {block}: {', '.join(diff)}"
    if (payload.get("student_optimizer") or {}).get("sha256") != meta["student_optimizer"]["sha256"]:
        return "different student optimizer"
    if payload.get("execution") != meta["execution"]:
        return "different execution regime"
    return ""


def _curve_dict(curve: TrainingCurve) -> dict:
    return {
        "eval_points": curve.eval_points, "train_points": curve.train_points,
        "wall_seconds": curve.wall_seconds, "failed": curve.failed, "error": curve.error,
        "checkpoints": [{k: v for k, v in c.items() if k != "path"} for c in (curve.checkpoints or [])],
        "refiner_stats": getattr(curve, "refiner_stats", {}) or {},
        "final_heldout_loss": (curve.eval_points[-1][1] if curve.eval_points else None),
        "steps": (curve.eval_points[-1][0] if curve.eval_points else None),
    }


def _load_task(args):
    budget = float(args.budget) if args.budget > 0 else None
    task = make_refiner_task(args.total_steps, data_dir=args.data_dir,
                             **({"compute_budget_seconds": budget} if budget else {}))
    if task.dataset != SHARD_DATASET_NAME or not task.data_manifest:
        raise RuntimeError("refiner task must be manifest-pinned shard data")
    if budget:
        task = replace(task, eval_every=max(10, min(task.eval_every, int(max(budget, 1) / 10))))
    if getattr(args, "eval_every", 0) and args.eval_every > 0:
        task = replace(task, eval_every=int(args.eval_every))  # smoke runs: held-out eval is diagnostic only
    if getattr(args, "eval_sequences", 0) and args.eval_sequences > 0:
        # Production holds out 8192 sequences (~17M tokens): one pass is ~2 min on an
        # A100 and the quartile audits run four of them — negligible against 20h,
        # but it eats a 5-minute smoke whole. Held-out loss is diagnostic here.
        task = replace(task, eval_sequences=int(args.eval_sequences))
    if getattr(args, "expected_manifest", "") and task.data_manifest != args.expected_manifest:
        raise RuntimeError(f"manifest mismatch: box has {task.data_manifest}, expected {args.expected_manifest}")
    return task


def _run_one(label: str, task, student: dict, hp: dict, refiner_cfg: dict, args, ckpt_path: Path,
             log) -> TrainingCurve:
    timeout = args.timeout if args.timeout > 0 else task.compute_budget_seconds * 1.15 + 600
    log(f"[refiner-eval] {label}: training student ({task.compute_budget_seconds:.0f}s budget, "
        f"seed {args.seed}, optimizer {student['sha256'][:12]})")
    t0 = time.time()
    curve = evaluate_submission_isolated(
        student["source"], task, args.seed, hp["lr"], hp.get("weight_decay", 0.01),
        use_amp=not args.no_amp, dataset_name=task.dataset, timeout=timeout,
        compute_budget_seconds=task.compute_budget_seconds, checkpoint_out=str(ckpt_path),
        refiner_cfg=refiner_cfg)
    log(f"[refiner-eval] {label}: done in {time.time() - t0:.0f}s, failed={curve.failed} "
        f"{('error=' + curve.error[:200]) if curve.failed else ''}")
    return curve


def _prune_quartiles(ckpt_path: Path):
    """Budget-mode runs also emit q25/q50/q75 snapshots; drop them unless kept."""
    base, ext = os.path.splitext(str(ckpt_path))
    for q in ("_q25", "_q50", "_q75"):
        p = Path(base + q + ext)
        if p.exists():
            p.unlink()


def run_refiner_eval(args, log=print) -> dict:
    t_start = time.time()
    if not settings.use_optproc() and not settings.env_truthy(REFINER_ALLOW_INPROC_ENV, default=False):
        return _fail("refiner-eval requires SN125_USE_OPTPROC=1 (student optimizer in the CUDA-IPC "
                     "process); set SN125_REFINER_ALLOW_INPROC_OPT=1 only for dev runs")

    # 1. gate the submission ---------------------------------------------------
    source = Path(args.file).read_text()
    gate = validate_single_source(source, path="refiner.py")
    if not gate.ok:
        return _fail(gate.dq_reason(), fault="miner", gate_flags=gate.flags)
    weights_path = args.weights or ""
    try:
        weights_summary = validate_refiner_weights(weights_path)
    except RefProcError as e:
        return _fail(f"gate: {e}", fault="miner")
    try:
        capabilities = extract_refiner_capabilities(source, strict=True)
    except RefProcError as e:
        return _fail(f"gate: {e}", fault="miner")

    # 2. task, student, suite ---------------------------------------------------
    try:
        task = _load_task(args)
    except Exception as e:
        return _fail(f"task unavailable: {type(e).__name__}: {e}")
    student = load_student_optimizer()
    hp = extract_hparams(student["source"], {"use_pretrained": task.use_pretrained,
                                             "parameter_count": task.parameter_count,
                                             "sequence_length": task.sequence_length})
    suite_path = Path(args.suite) if args.suite else default_suite_path(args.data_dir)
    try:
        suite = load_suite(suite_path)
    except Exception as e:
        return _fail(f"benchmark suite unavailable at {suite_path}: {e} (build with `sn125 bench-build`)")
    want_suite = (getattr(args, "expected_suite_sha256", "") or "").strip()
    if want_suite and suite["suite_sha256"] != want_suite:
        return _fail(f"benchmark suite {suite['suite_sha256'][:16]} is not the pinned one {want_suite[:16]}")
    # Decontamination list for THIS suite (scripts/refiner_decontam.py --write-exclusions):
    # documents sharing long n-grams with suite items are withheld from the stream.
    ex_file = exclusions_path(args.data_dir, "train", suite["suite_id"])
    decontam = getattr(args, "decontam", "auto")
    if decontam == "off" or (decontam == "auto" and not ex_file.exists()):
        ex_name = ""
        if decontam == "auto":
            log(f"[refiner-eval] WARNING: no decontamination list at {ex_file}; the entry stream "
                f"includes documents that overlap the suite")
    elif not ex_file.exists():
        return _fail(f"decontamination list missing: {ex_file} (build it with "
                     f"`scripts/refiner_decontam.py --write-exclusions`)")
    else:
        ex_name = ex_file.name
    try:
        ensure_doc_index(args.data_dir, "train")
        stream_desc = EntryStream(args.data_dir, split="train", seed=args.seed,
                                  max_entry_tokens=args.max_entry_tokens, verify=False,
                                  exclusions=ex_name or None).describe()
    except Exception as e:
        return _fail(f"entry stream unavailable: {type(e).__name__}: {e}")
    ex_kw = {"exclusions": ex_name, "exclusions_sha256": stream_desc["exclusions_sha256"]}
    want_ex = (getattr(args, "expected_decontam_sha256", "") or "").strip()
    if want_ex and stream_desc["exclusions_sha256"] != want_ex:
        return _fail(f"decontamination list {stream_desc['exclusions_sha256'][:16] or '<none>'} is not the "
                     f"pinned one {want_ex[:16]}")

    meta = {
        "task_id": task.task_id,
        "task": {"task_id": task.task_id, "model_config": task.model_config,
                 "parameter_count": task.parameter_count, "total_steps": task.total_steps,
                 "batch_size": task.batch_size, "sequence_length": task.sequence_length,
                 "compute_budget_seconds": task.compute_budget_seconds, "dataset": task.dataset,
                 "data_manifest": task.data_manifest, "seed": args.seed},
        "entry_stream": stream_desc,
        "student_optimizer": {"sha256": student["sha256"], "origin": student["origin"],
                              "pinned": student["pinned"], "lr": hp["lr"], "weight_decay": hp.get("weight_decay", 0.01)},
        "suite": {"path": str(suite_path), "suite_id": suite["suite_id"], "suite_sha256": suite["suite_sha256"],
                  "tasks": {k: v["n"] for k, v in suite["tasks"].items()}},
        "submission": {"source_bytes": len(source.encode()), "weights": weights_summary,
                       "capabilities": capabilities, "gate_flags": gate.flags},
        # force_gc changes steps/s (and so how far a wall-clock run gets): part of the regime.
        "execution": {"optproc_enabled": settings.use_optproc(), "use_amp": not args.no_amp,
                      "sandbox_mode": settings.sandbox_mode(),
                      "force_gc": bool(os.environ.get("SN125_FORCE_GC"))},
    }

    # Checkpoints are written by the sandboxed worker, which sees a PRIVATE /tmp
    # and a read-only /root, /home, /etc, /var: the directory must live elsewhere
    # on the real filesystem (default: ./runs/... under the operator's cwd).
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = Path(os.environ.get("SN125_RUN_DIR_BASE") or Path.cwd() / "runs") / f"refiner_eval_{int(t_start)}"
    if str(ckpt_dir.resolve()).startswith("/tmp/"):
        return _fail(f"--checkpoint-dir {ckpt_dir} is under /tmp, which the sandbox worker replaces with a "
                     f"private tmpfs; choose a path on the real filesystem", **meta)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if not args.cpu else "cpu"
    model_spec = f"hf:{task.model_config}"

    def _bench(label: str, ckpt: Path, curve: TrainingCurve) -> dict:
        log(f"[refiner-eval] {label}: scoring {ckpt.name} on suite {suite['suite_id']}")
        res = score_checkpoint_on_suite(str(ckpt), suite, model_spec, seq_len=task.sequence_length,
                                        device=device, use_amp=not args.no_amp,
                                        rows_per_batch=args.bench_rows, log=log, per_item=True)
        # The scored file must be the one the training loop exported: the worker
        # reports the sha of the final checkpoint it wrote, and the directory is
        # writable by the sandbox uid until the worker is gone.
        want = _attested_final_sha(_curve_dict(curve))
        if not want or res.get("checkpoint_sha256") != want:
            raise RuntimeError(f"{label} checkpoint sha {str(res.get('checkpoint_sha256'))[:16]} does not match "
                               f"the training loop's final export {want[:16] or '<none>'}")
        return res

    # 3. baseline -------------------------------------------------------------------
    base_curve = None
    if args.baseline_json:
        payload = json.loads(Path(args.baseline_json).read_text())
        why = _baseline_mismatch(payload, meta)
        if why:
            return _fail(f"baseline JSON cannot be reused: {why}", **meta)
        base_bench = payload["baseline"]["bench"]
        base_curve_d = payload["baseline"].get("curve")
        base_result = {"bench": base_bench, "curve": base_curve_d, "source": str(args.baseline_json)}
    else:
        base_ckpt = ckpt_dir / "baseline.safetensors"
        base_cfg = refiner_run_config(mode="identity", seed=args.seed, data_dir=args.data_dir,
                                      max_entry_tokens=args.max_entry_tokens, **ex_kw)
        base_curve = _run_one("baseline(identity)", task, student, hp, base_cfg, args, base_ckpt, log)
        _prune_quartiles(base_ckpt) if not args.keep_checkpoints else None
        if base_curve.failed:
            return _fail(f"baseline failed: {base_curve.error}", baseline={"curve": _curve_dict(base_curve)}, **meta)
        try:
            base_bench = _bench("baseline(identity)", base_ckpt, base_curve)
        except Exception as e:
            return _fail(f"baseline bench scoring failed: {type(e).__name__}: {e}",
                         baseline={"curve": _curve_dict(base_curve)}, **meta)
        base_result = {"bench": base_bench, "curve": _curve_dict(base_curve), "source": "run"}
        if not args.keep_checkpoints:
            base_ckpt.unlink(missing_ok=True)

    if getattr(args, "baseline_only", False):
        # Operator run: produce the identity baseline once, to be pinned and
        # shipped to every scoring box (--baseline-json) so an eval is one
        # training run, not two.
        out = {"failed": False, "score": 0.0, "baseline_only": True, "baseline": base_result,
               "wall_seconds": time.time() - t_start}
        out.update({k: v for k, v in meta.items() if k != "submission"})
        return out

    # 4. submission -------------------------------------------------------------
    sub_ckpt = ckpt_dir / "submission.safetensors"
    sub_cfg = refiner_run_config(mode="refiner", source=source, weights_path=weights_path, seed=args.seed,
                                 data_dir=args.data_dir, max_entry_tokens=args.max_entry_tokens,
                                 plan_timeout=args.plan_timeout, **ex_kw)
    sub_curve = _run_one("submission", task, student, hp, sub_cfg, args, sub_ckpt, log)
    if not args.keep_checkpoints:
        _prune_quartiles(sub_ckpt)
    if sub_curve.failed:
        return _fail(sub_curve.error, fault=_fault_of(sub_curve.error), baseline=base_result,
                     submission_curve=_curve_dict(sub_curve), **meta)
    try:
        sub_bench = _bench("submission", sub_ckpt, sub_curve)
    except Exception as e:
        return _fail(f"submission bench scoring failed: {type(e).__name__}: {e}", baseline=base_result,
                     submission_curve=_curve_dict(sub_curve), **meta)
    if not args.keep_checkpoints:
        sub_ckpt.unlink(missing_ok=True)
        if not args.checkpoint_dir:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    # 5. score ------------------------------------------------------------------------
    score = float(sub_bench["score"]) - float(base_result["bench"]["score"])
    per_task = {}
    for name, t in sub_bench["tasks"].items():
        b = base_result["bench"]["tasks"].get(name, {})
        per_task[name] = {k: {"submission": t.get(k), "baseline": b.get(k),
                              "delta": (t.get(k) - b.get(k)) if isinstance(t.get(k), float) and isinstance(b.get(k), float) else None}
                          for k in ("acc", "acc_norm", "prob_correct_norm", "prob_correct", "bpb_gold", "ppl")
                          if k in t}
    sub_h = sub_curve.eval_points[-1][1] if sub_curve.eval_points else None
    base_h = base_result["curve"].get("final_heldout_loss") if base_result.get("curve") else None
    heldout_delta = (sub_h - base_h) if (sub_h is not None and base_h is not None) else None
    # Optional held-out non-regression gate (docs/REFINER_TASK.md §10, the
    # suite-design decision): a refiner that lifts the suite while making the
    # student WORSE at ordinary text by more than the allowance scores 0 — the
    # suite delta stays visible in components and task_scores, the verdict is
    # recorded in ``gates``. Off unless --heldout-max-regression is given.
    max_reg = getattr(args, "heldout_max_regression", None)
    gate = {"heldout_non_regression": {"enabled": max_reg is not None, "max_regression_nats": max_reg,
                                       "heldout_delta": heldout_delta, "passed": True}}
    suite_delta = score
    if max_reg is not None and heldout_delta is not None and heldout_delta > float(max_reg):
        gate["heldout_non_regression"]["passed"] = False
        score = 0.0
    result = {
        "failed": False,
        "score": score,
        "gates": gate,
        "components": {
            "bench_score_submission": sub_bench["score"],
            "bench_score_baseline": base_result["bench"]["score"],
            "mean_acc_norm_submission": sub_bench.get("mean_acc_norm"),
            "mean_acc_norm_baseline": base_result["bench"].get("mean_acc_norm"),
            "heldout_loss_submission": sub_h,
            "heldout_loss_baseline": base_h,
            "heldout_delta_diagnostic": heldout_delta,
            "suite_delta": suite_delta,     # the raw suite delta, whatever the gates did to ``score``
        },
        # Item-paired bootstrap of the suite delta (item-sampling noise only; None
        # when the reused baseline predates per-item scores).
        "paired": paired_delta(sub_bench, base_result["bench"]),
        "task_scores": {task.task_id: {"score": score, "suite_delta": suite_delta, "per_task": per_task}},
        "baseline": base_result,
        "submission": {**meta["submission"], "bench": sub_bench, "curve": _curve_dict(sub_curve)},
        "wall_seconds": time.time() - t_start,
    }
    result.update({k: v for k, v in meta.items() if k != "submission"})
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate one Refiner-task submission (refiner.py + weights)")
    ap.add_argument("file", help="refiner.py source")
    ap.add_argument("--weights", default="", help="refiner safetensors weights (optional, <=100M params)")
    ap.add_argument("--data-dir", default=None, help="production shard dir; default SN125_FINEWEB_DIR/repo data")
    ap.add_argument("--suite", default="", help="pre-tokenized suite JSON; default <data-dir>/bench/suite_easy_v0.json")
    ap.add_argument("--total-steps", type=int, default=PROD_CONFIRM_STEPS)
    ap.add_argument("--budget", type=float, default=0.0, help="override compute_budget_seconds (smoke runs)")
    ap.add_argument("--eval-every", type=int, default=0, help="override held-out eval interval (diagnostic; smoke runs)")
    ap.add_argument("--eval-sequences", type=int, default=0,
                    help="override held-out eval size (production 8192 ≈ 2 min/pass on A100; smoke: 256)")
    ap.add_argument("--timeout", type=float, default=0.0, help="worker timeout; default budget*1.15+600")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="bench scoring on CPU (tests)")
    ap.add_argument("--max-entry-tokens", type=int, default=DEFAULT_MAX_ENTRY_TOKENS)
    ap.add_argument("--plan-timeout", type=float, default=DEFAULT_PLAN_TIMEOUT_S)
    ap.add_argument("--bench-rows", type=int, default=8, help="rows per forward during bench scoring")
    ap.add_argument("--baseline-json", default="", help="reuse a previous result's baseline (same suite + data)")
    ap.add_argument("--heldout-max-regression", type=float, default=None,
                    help="non-regression gate: score 0 when the submission's final held-out loss exceeds the "
                         "baseline's by more than this many nats (default: gate off; the delta is always reported)")
    ap.add_argument("--expected-manifest", default="")
    ap.add_argument("--expected-suite-sha256", default="", help="fail (harness fault) unless the suite has this sha256")
    ap.add_argument("--expected-decontam-sha256", default="",
                    help="fail (harness fault) unless the decontamination list has this sha256")
    ap.add_argument("--baseline-only", action="store_true",
                    help="run and score only the identity baseline; the result is what --baseline-json reuses")
    ap.add_argument("--decontam", choices=("auto", "require", "off"), default="auto",
                    help="withhold documents that overlap the suite (DECONTAM_train_<suite>.npy next to the "
                         "shards): auto = use it when present, require = fail without it, off = never")
    ap.add_argument("--checkpoint-dir", default="", help="where to write checkpoints (default: temp dir)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args(argv)

    result = run_refiner_eval(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    # One machine-readable line (what the cloud layer parses), without the
    # per-item score lists: those are thousands of numbers and live in --out.
    print(RESULT_MARKER + json.dumps(compact_result(result), default=str))
    return 0 if not result.get("failed") else 1


if __name__ == "__main__":
    sys.exit(main())
