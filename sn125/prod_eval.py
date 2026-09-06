"""Production submission evaluator.

This is the on-box command the Targon/FSM scoring path should run for paid
production submissions. It deliberately does not use ``bench.py``: bench is a
developer comparison tool on Wikitext/legacy task panels, while this command
requires the hash-pinned FineWeb production task and fails closed when that task
is unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from . import settings
from .fineweb import PROD_CONFIRM_STEPS, SHARD_DATASET_NAME, production_tasks
from .gate import validate_single_source
from .references import ADAMW_SOURCE, extract_hparams
from .scoring import score_submission
from .training import TrainingCurve, evaluate_submission_isolated


def _checkpoint_required_result() -> dict:
    return {
        "failed": True,
        "score": -1.0,
        "error": (
            "prod-eval requires --checkpoint-out so reward-bearing scores are "
            "computed from a safetensors checkpoint in a fresh score-worker process"
        ),
        "components": {},
        "clean_scoring": {"required": True, "enabled": False},
    }


def _optproc_required_result() -> dict:
    return {
        "failed": True,
        "score": -1.0,
        "error": (
            "prod-eval requires SN125_USE_OPTPROC=1 so reward-bearing "
            "evaluations use the CUDA-IPC optimizer process"
        ),
        "components": {},
        "execution": {
            "optproc_required": True,
            "optproc_enabled": False,
            "optimizer_process": "missing",
        },
    }


def _clean_score_device() -> str:
    return settings.env_str(settings.CLEAN_SCORE_DEVICE_ENV, "cuda")


def _clean_score_timeout() -> float:
    return settings.env_float(settings.CLEAN_SCORE_TIMEOUT_ENV, 1800.0, minimum=1.0)


def _clean_score_atol() -> float:
    """Trainer-vs-clean consistency tolerance (tamper tripwire, NOT the score:
    apply_verified_loss substitutes the clean value either way).

    Eager trainer + eager scorer reproduce bit-identically -> tight 1e-4.
    Under SN125_LEAN_COMPILE the trainer evals through inductor kernels while
    the clean box stays eager (one pass; compile warmup would cost more than
    it saves), so the honest expectation is kernel-level drift ~1e-3. Widen
    the default to 3e-3 — still far below any exploitable margin (the
    frontier floor is 0.03). Explicit env always wins."""
    raw = settings.env_str(settings.CLEAN_SCORE_ATOL_ENV)
    if not raw:
        compile_on = os.environ.get(settings.LEAN_COMPILE_ENV, "").strip() not in ("", "0")
        return 3e-3 if compile_on else 1e-4
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1e-4


def _export_eval_shard_for_task(task, args, out_path: str) -> str:
    from .engine.score_worker import export_eval_shard
    from .training import _load_real_data

    eval_batches = max(1, task.eval_sequences // task.batch_size)
    eval_split = "heldout" if task.dataset == SHARD_DATASET_NAME else "train"
    eval_data = _load_real_data(
        task.dataset,
        task.sequence_length,
        task.batch_size,
        eval_batches,
        args.seed + 1_000_000,
        split=eval_split,
        tokenizer_name=task.model_config,
    )
    return export_eval_shard(
        list(eval_data),
        out_path,
        metadata={
            "task_id": task.task_id,
            "data_manifest": task.data_manifest,
            "seed": str(args.seed),
            "split": eval_split,
        },
    )


def _apply_clean_score(task, args, curve: TrainingCurve) -> dict:
    """Replace the trainer-reported final loss with fresh score-worker loss."""
    from .engine.score_worker import apply_verified_loss, score_checkpoint_isolated

    checkpoint_path = Path(args.checkpoint_out)
    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint_out missing after training: {checkpoint_path}")
    with tempfile.TemporaryDirectory(prefix="sn125_clean_score_") as td:
        eval_shard = str(Path(td) / "heldout_eval.safetensors")
        eval_shard_sha = _export_eval_shard_for_task(task, args, eval_shard)
        clean = score_checkpoint_isolated(
            str(checkpoint_path),
            eval_shard,
            f"hf:{task.model_config}",
            use_amp=not args.no_amp,
            device=_clean_score_device(),
            timeout=_clean_score_timeout(),
        )
    verified = apply_verified_loss(curve, clean, atol=_clean_score_atol())
    return {
        "required": True,
        "enabled": True,
        "source": "score_worker_subprocess",
        "checkpoint_sha256": clean.get("checkpoint_sha256", ""),
        "eval_shard_sha256": clean.get("eval_shard_sha256", eval_shard_sha),
        "eval_loss": verified.eval_loss,
        "trainer_reported": verified.trainer_reported,
        "consistent": verified.consistent,
        "atol": _clean_score_atol(),
        "result": clean,
    }


def _prod_optproc_enabled() -> bool:
    return settings.use_optproc()


def _execution_metadata() -> dict:
    return {
        "optproc_required": True,
        "optproc_enabled": True,
        "optimizer_process": "cuda_ipc_optproc",
        "optproc_step_timeout": os.environ.get("SN125_OPTPROC_STEP_TIMEOUT", ""),
    }


def _curve_dict(curve):
    return {
        "eval_points": curve.eval_points,
        "train_points": curve.train_points,
        "wall_seconds": curve.wall_seconds,
        "state_multiplier": getattr(curve, "state_multiplier", 1.0),
        "failed": curve.failed,
        "error": curve.error,
        "checkpoints": list(getattr(curve, "checkpoints", []) or []),
    }


def _load_production_task(args):
    tasks = production_tasks(total_steps=args.total_steps, data_dir=args.data_dir)
    if len(tasks) != 1:
        raise RuntimeError(f"production_tasks returned {len(tasks)} tasks; expected exactly 1")
    task = tasks[0]
    if task.dataset != SHARD_DATASET_NAME:
        raise RuntimeError(f"production task dataset must be {SHARD_DATASET_NAME}, got {task.dataset}")
    if not task.data_manifest:
        raise RuntimeError("production task is not manifest-pinned")
    expected_manifest = getattr(args, "expected_manifest", "")
    if expected_manifest and task.data_manifest != expected_manifest:
        raise RuntimeError(
            f"production manifest mismatch: box has {task.data_manifest}, "
            f"validator expected {expected_manifest}")
    if task.compute_budget_seconds <= 0:
        raise RuntimeError("production task must use a positive compute_budget_seconds")
    if args.budget > 0:
        task = replace(
            task,
            compute_budget_seconds=float(args.budget),
            eval_every=max(10, min(task.eval_every, int(max(args.budget, 1) / 10))),
        )
    return task


def _curve_from_dict(task_id: str, d: dict) -> TrainingCurve:
    state_multiplier = d.get("state_multiplier", 1.0)
    if state_multiplier is None:
        state_multiplier = 1.0
    return TrainingCurve(
        task_id=task_id,
        lr=float(d.get("lr", 0.0) or 0.0),
        wd=float(d.get("wd", d.get("weight_decay", 0.0)) or 0.0),
        eval_points=[tuple(p) for p in (d.get("eval_points") or [])],
        train_points=[tuple(p) for p in (d.get("train_points") or [])],
        wall_seconds=float(d.get("wall_seconds", 0.0) or 0.0),
        state_multiplier=float(state_multiplier),
        failed=bool(d.get("failed", False)),
        error=str(d.get("error", "") or ""),
    )


def _load_baseline_curve(path: str, task) -> TrainingCurve:
    payload = json.loads(Path(path).read_text())
    cd = payload.get("curve_data") or {}
    curves = (payload.get("baseline_curves") or payload.get("curves")
              or cd.get("curves") or cd.get("baseline_curves") or payload)
    if not isinstance(curves, dict):
        raise RuntimeError(f"baseline JSON has no curve mapping: {path}")
    d = curves.get(task.task_id)
    if d is None and len(curves) == 1:
        d = next(iter(curves.values()))
    if not isinstance(d, dict):
        raise RuntimeError(f"baseline JSON missing curve for task {task.task_id}")
    curve = _curve_from_dict(task.task_id, d)
    if curve.failed or not curve.eval_points:
        raise RuntimeError(f"baseline curve for {task.task_id} is failed/empty")
    return curve


def run_prod_eval(args) -> dict:
    if not _prod_optproc_enabled():
        return _optproc_required_result()
    if not getattr(args, "checkpoint_out", ""):
        return _checkpoint_required_result()

    execution = _execution_metadata()
    source = Path(args.file).read_text()
    gate = validate_single_source(source)
    if not gate.ok:
        return {
            "failed": True,
            "score": -1.0,
            "error": gate.dq_reason(),
            "components": {},
            "execution": execution,
        }

    task = _load_production_task(args)
    task_weights = {task.task_id: task.task_weight}
    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)

    sub_hp = extract_hparams(
        source,
        {
            "use_pretrained": task.use_pretrained,
            "parameter_count": task.parameter_count,
            "sequence_length": task.sequence_length,
        },
    )
    base_hp = extract_hparams(
        ADAMW_SOURCE,
        {
            "use_pretrained": task.use_pretrained,
            "parameter_count": task.parameter_count,
            "sequence_length": task.sequence_length,
        },
    )

    timeout = args.timeout if args.timeout > 0 else task.compute_budget_seconds * 1.15
    common = {
        "task": task,
        "seed": args.seed,
        "use_amp": not args.no_amp,
        "dataset_name": task.dataset,
        "timeout": timeout,
        "compute_budget_seconds": task.compute_budget_seconds,
        "checkpoint_out": args.checkpoint_out,
    }

    if args.baseline_json:
        base_curve = _load_baseline_curve(args.baseline_json, task)
    else:
        base_curve = evaluate_submission_isolated(
            ADAMW_SOURCE,
            lr=base_hp["lr"],
            wd=base_hp.get("weight_decay", 0.01),
            **common,
        )
    if base_curve.failed:
        return {
            "failed": True,
            "score": -1.0,
            "error": f"baseline failed: {base_curve.error}",
            "components": {},
            "execution": execution,
            "curve_data": {"baseline_curves": {task.task_id: _curve_dict(base_curve)}},
        }

    sub_curve = evaluate_submission_isolated(
        source,
        lr=sub_hp["lr"],
        wd=sub_hp.get("weight_decay", 0.01),
        **common,
    )
    if sub_curve.failed:
        return {
            "failed": True,
            "score": -1.0,
            "error": sub_curve.error,
            "components": {},
            "execution": execution,
            "curve_data": {
                "curves": {task.task_id: _curve_dict(sub_curve)},
                "baseline_curves": {task.task_id: _curve_dict(base_curve)},
                "hparams_per_task": {task.task_id: [sub_hp["lr"], sub_hp.get("weight_decay", 0.01)]},
            },
        }

    try:
        clean_scoring = _apply_clean_score(task, args, sub_curve)
    except Exception as e:
        return {
            "failed": True,
            "score": -1.0,
            "error": f"clean scoring failed: {e}",
            "components": {},
            "execution": execution,
            "clean_scoring": {"required": True, "enabled": True, "failed": True, "error": str(e)},
            "curve_data": {
                "curves": {task.task_id: _curve_dict(sub_curve)},
                "baseline_curves": {task.task_id: _curve_dict(base_curve)},
                "hparams_per_task": {task.task_id: [sub_hp["lr"], sub_hp.get("weight_decay", 0.01)]},
            },
        }
    if not clean_scoring.get("consistent", False):
        return {
            "failed": True,
            "score": -1.0,
            "error": (
                "clean scoring mismatch: trainer final loss "
                f"{clean_scoring.get('trainer_reported')} vs clean "
                f"{clean_scoring.get('eval_loss')}"
            ),
            "components": {},
            "execution": execution,
            "clean_scoring": clean_scoring,
            "curve_data": {
                "curves": {task.task_id: _curve_dict(sub_curve)},
                "baseline_curves": {task.task_id: _curve_dict(base_curve)},
                "hparams_per_task": {task.task_id: [sub_hp["lr"], sub_hp.get("weight_decay", 0.01)]},
            },
        }

    score = score_submission(
        {task.task_id: sub_curve},
        {task.task_id: base_curve},
        task_weights,
        len(source.encode()),
    )
    return {
        "failed": False,
        "score": score.final_score,
        "components": score.components,
        "task_scores": score.task_scores,
        "best_hparams": score.best_hparams,
        "execution": execution,
        "clean_scoring": clean_scoring,
        "curve_data": {
            "curves": {task.task_id: _curve_dict(sub_curve)},
            "baseline_curves": {task.task_id: _curve_dict(base_curve)},
            "hparams_per_task": {task.task_id: [sub_hp["lr"], sub_hp.get("weight_decay", 0.01)]},
        },
        "task": {
            "task_id": task.task_id,
            "model_config": task.model_config,
            "parameter_count": task.parameter_count,
            "total_steps": task.total_steps,
            "batch_size": task.batch_size,
            "sequence_length": task.sequence_length,
            "compute_budget_seconds": task.compute_budget_seconds,
            "dataset": task.dataset,
            "data_manifest": task.data_manifest,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one production FineWeb submission eval")
    ap.add_argument("file", help="optimizer.py source")
    ap.add_argument("--data-dir", default=None, help="production shard dir; default SN125_FINEWEB_DIR/repo data")
    ap.add_argument("--total-steps", type=int, default=PROD_CONFIRM_STEPS)
    ap.add_argument("--budget", type=float, default=0.0, help="override compute_budget_seconds for smoke runs")
    ap.add_argument("--timeout", type=float, default=0.0, help="subprocess timeout; default budget*1.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--baseline-json", default="",
                    help="JSON baseline bundle; when set, only the submission is trained")
    ap.add_argument("--expected-manifest", default="",
                    help="Require production shards to match this manifest hash")
    ap.add_argument("--checkpoint-out", default="",
                    help="Optional safetensors path for final plus q25/q50/q75 checkpoints")
    ap.add_argument("--worker-log", default="",
                    help="Optional JSONL sink for live sandbox worker stdout/stderr progress")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args(argv)

    if args.worker_log:
        os.environ["SN125_WORKER_LOG_PATH"] = args.worker_log

    try:
        result = run_prod_eval(args)
    except Exception as e:
        result = {"failed": True, "score": -1.0, "error": f"prod-eval setup failed: {e}", "components": {}}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    print("PROD_EVAL_RESULT " + json.dumps(result, default=float), flush=True)
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
