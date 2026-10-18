"""Refiner task (Challenge 2) definition: task spec + pinned student optimizer.

Same student, same budget, same shards as the optimizer task; what differs is
WHO decides what each step trains on. ``make_refiner_task`` is the single
source of truth for the production Refiner TaskSpec, the way
``fineweb.production_tasks`` is for the optimizer task.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

from ..fineweb import (FIXED_BUDGET_SEQ_LEN, MANIFEST_NAME, PROD_BATCH_SIZE, PROD_BUDGET_SECONDS,
                       PROD_MODEL_SIZE, PROD_MODEL_SIZE_ENV, default_data_dir, load_manifest,
                       make_fineweb_task)
from .entries import DEFAULT_MAX_ENTRY_TOKENS
from .refproc_limits import REFINER_TASK_ID  # noqa: F401  (defined torch-free; re-exported here)

STUDENT_OPTIMIZER_ENV = "SN125_STUDENT_OPTIMIZER_PATH"
STUDENT_DIR = Path(__file__).resolve().parent / "student"
PINNED_STUDENT_FILE = STUDENT_DIR / "kairos_b_96fe905a.py"
PINNED_STUDENT_SHA256 = "96fe905a538c32325bf550f0eed66e9ed5289489589752bbe0c8da81771ea383"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def load_student_optimizer() -> dict:
    """Return {"source", "sha256", "origin", "pinned": bool} for the trusted student.

    Resolution order: env override -> pinned file (hash-checked) -> ADAMW fallback.
    """
    override = os.environ.get(STUDENT_OPTIMIZER_ENV, "").strip()
    if override:
        src = Path(override).read_text()
        return {"source": src, "sha256": _sha256_text(src), "origin": f"env:{override}", "pinned": False}
    if PINNED_STUDENT_FILE.exists():
        src = PINNED_STUDENT_FILE.read_text()
        sha = _sha256_text(src)
        if sha != PINNED_STUDENT_SHA256:
            raise RuntimeError(f"pinned student optimizer hash mismatch: {sha} != {PINNED_STUDENT_SHA256}")
        return {"source": src, "sha256": sha, "origin": str(PINNED_STUDENT_FILE.relative_to(STUDENT_DIR.parent.parent.parent)),
                "pinned": True}
    from ..references import ADAMW_SOURCE
    return {"source": ADAMW_SOURCE, "sha256": _sha256_text(ADAMW_SOURCE),
            "origin": "sn125.references.ADAMW_SOURCE (pinned file missing)", "pinned": False}


def make_refiner_task(total_steps: int, *, data_dir: str | Path | None = None,
                      model_size: str | None = None, decay_fraction: float = 0.20,
                      compute_budget_seconds: float = PROD_BUDGET_SECONDS,
                      stop_at_total_steps: bool = True):
    """Production Refiner TaskSpec: the optimizer task's shape under a new task id."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    model_size = model_size or os.environ.get(PROD_MODEL_SIZE_ENV, PROD_MODEL_SIZE)
    if not (data_dir / MANIFEST_NAME).exists():
        raise FileNotFoundError(
            f"No production shards at {data_dir} (MANIFEST.json missing). "
            f"Build them first (scripts/build_prod_shards.py / build_shards()).")
    mf = load_manifest(data_dir)
    task = make_fineweb_task(model_size, total_steps=total_steps, manifest=mf["manifest_hash"],
                             decay_fraction=decay_fraction, seq_len=FIXED_BUDGET_SEQ_LEN,
                             batch_size=PROD_BATCH_SIZE,
                             compute_budget_seconds=compute_budget_seconds,
                             stop_at_total_steps=stop_at_total_steps)
    return replace(task, task_id=REFINER_TASK_ID)


def refiner_run_config(*, mode: str, source: str = "", weights_path: str = "", seed: int,
                       data_dir: str | Path | None = None, split: str = "train",
                       max_entry_tokens: int = DEFAULT_MAX_ENTRY_TOKENS,
                       plan_timeout: float | None = None, window_entries: int | None = None,
                       exclusions: str = "", exclusions_sha256: str = "") -> dict:
    """The dict handed to training.evaluate_submission(refiner_cfg=...).

    ``exclusions``: file name (next to the shards) of the decontamination list;
    ``exclusions_sha256`` pins it, and the worker refuses a different file."""
    cfg = {"mode": mode, "seed": int(seed), "split": split, "max_entry_tokens": int(max_entry_tokens),
           "data_dir": str(data_dir) if data_dir else ""}
    if exclusions:
        cfg["exclusions"] = str(exclusions)
        cfg["exclusions_sha256"] = str(exclusions_sha256)
    if mode == "refiner":
        cfg["source"] = source
        cfg["weights_path"] = weights_path or ""
        if plan_timeout is not None:
            cfg["plan_timeout"] = float(plan_timeout)
    elif mode == "identity":
        if window_entries is not None:
            cfg["window_entries"] = int(window_entries)
    else:
        raise ValueError(f"unknown refiner mode {mode!r}")
    return cfg
