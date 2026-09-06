"""
Rolling-best baseline loader (V24 / B-D scoring).

Operator goal (Apr 24): "the network should figure it out, and the first
submitter (or best of the first round) should be the baseline." Drops the
synthetic AdamW canonical baseline; instead, every miner is scored against
the best previously-observed curve for that task.

This module is the data layer only. Scoring integration lives in neuron.py
behind a `--baseline {adamw,rolling_best}` CLI flag. The active default is
`rolling_best`; `adamw` remains as a legacy diagnostic mode only.

Design points
-------------
- A "rolling best" for task T is the best (lowest final eval_loss) curve
  produced by any submission across all prior rounds, restricted to
  submissions whose `score.final_score > THRESHOLD` (so we don't pin the bar
  to a broken or sandboxed submission with score=-1).
- The first round (no priors) returns `None`; caller must use the cold-start
  fallback (raw eval-loss against conservative anchors). The best valid early
  submission then becomes the baseline for later rounds.
- Curves must match TaskSpec exactly (same model_config, total_steps,
  batch_size, sequence_length, compute_budget_seconds). We enforce a hash
  match to avoid pinning the bar to a curve from a different rung.
- Failed curves (`failed=True` or `final_loss <= 0` or `eval_points` empty)
  are excluded.
- Budget-transfer baseline = the same rolling-best curve sampled at
  50%/100% checkpoints (V23 in-curve extraction; no extra compute).

Round JSON schema this loader reads (see rounds/test_001.json for an example):
    {
      "round_id": str,
      "timestamp": int,
      "tasks": [TaskSpec dicts],
      "submissions": {
          hk_or_label: {
              "score": {"final_score": float, ...},
              "code_hash": str,
              "curves": {task_id: curve_dict},
              ...
          }
      }
    }

Curve dict:
    {
      "eval_points": [[step, eval_loss, wall_t], ...],
      "train_points": [[step, train_loss], ...],
      "wall_seconds": float,
      "state_multiplier": float,
      "failed": bool,
      "error": str
    }
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("sn125.rolling")


MIN_VALID_SCORE = -0.5


@dataclass
class RollingBestCurve:
    """A historical baseline curve plus provenance.

    Provenance fields (round_id, code_hash, score) are kept so the dashboard
    and the round JSON can show *which* prior submission set the bar.
    """
    task_id: str
    eval_points: list
    train_points: list
    wall_seconds: float
    state_multiplier: float
    final_eval_loss: float
    source_round_id: str
    source_code_hash: str
    source_score: float


def task_signature(task: dict) -> str:
    """Stable hash of the fields that define an apples-to-apples comparison.

    Two TaskSpecs are comparable iff they share model_config, total_steps,
    batch_size, sequence_length, use_pretrained, compute_budget_seconds,
    AND the data identity (dataset + data_manifest). The data fields are the
    S1 task-switch rule: a task on different data — or on a rebuilt shard
    manifest — is a different task, so no curve from the old data can ever
    be served as the bar (phase 6's carried-reference false winner).
    Defaults ("fineweb-edu", "") match round JSONs written before these
    fields existed. Differences in `eval_every` or `task_weight` are
    presentation-only and don't affect the curve shape, so they're excluded.

    Accepts either a TaskSpec dict (from rounds/*.json) or a TaskSpec
    dataclass (will be converted via __dict__).
    """
    if not isinstance(task, dict):
        task = {k: getattr(task, k) for k in (
            "task_id", "model_config", "total_steps", "batch_size",
            "sequence_length", "use_pretrained", "compute_budget_seconds",
            "dataset", "data_manifest")
            if hasattr(task, k)}
    budget = float(task.get("compute_budget_seconds", 0.0))
    fields = {
        "model_config": task.get("model_config", ""),
        "total_steps": 0 if budget > 0 else int(task.get("total_steps", 0)),
        "batch_size": int(task.get("batch_size", 0)),
        "sequence_length": int(task.get("sequence_length", 0)),
        "use_pretrained": bool(task.get("use_pretrained", False)),
        "compute_budget_seconds": budget,
        "dataset": str(task.get("dataset", "fineweb-edu")),
        "data_manifest": str(task.get("data_manifest", "")),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def _curve_is_usable(curve: dict) -> bool:
    """A curve must be non-failed, have eval_points, and end with a finite positive loss."""
    if not isinstance(curve, dict):
        return False
    if curve.get("failed"):
        return False
    eps = curve.get("eval_points") or []
    if not eps:
        return False
    last = eps[-1]
    if not (isinstance(last, (list, tuple)) and len(last) >= 2):
        return False
    final_loss = last[1]
    if not isinstance(final_loss, (int, float)):
        return False
    if final_loss <= 0 or final_loss != final_loss:
        return False
    return True


def load_rolling_best(
    rounds_dir: str | Path,
    task: dict,
    *,
    exclude_round_id: Optional[str] = None,
    min_score: float = MIN_VALID_SCORE,
) -> Optional[RollingBestCurve]:
    """Find the best historical curve for `task`.

    Args:
        rounds_dir: directory containing round_*.json files (typically
            sn125/rounds/).
        task: TaskSpec dict OR dataclass identifying which rung we want.
            Matched on `task_signature()` — same model + same budget.
        exclude_round_id: skip this round_id (e.g. the round currently
            being scored). Pass round_id to avoid scoring against itself.
        min_score: ignore submissions whose final_score is below this floor.
            Default -0.5 excludes failures and severe regressions.

    Returns:
        RollingBestCurve if a usable historical curve exists, else None.
        Caller falls back to cold_start_score() in the None case.
    """
    rounds_dir = Path(rounds_dir)
    if not rounds_dir.is_dir():
        log.debug(f"rolling_best: rounds dir {rounds_dir} does not exist")
        return None

    target_sig = task_signature(task)
    target_id = task["task_id"] if isinstance(task, dict) else task.task_id
    candidates: list[RollingBestCurve] = []
    anchors: list[RollingBestCurve] = []
    rounds: dict[str, dict] = {}
    event_refs: list[tuple[dict, dict]] = []

    def _record(rid: str, section: str, key: str, sub: dict) -> Optional[RollingBestCurve]:
        score_obj = sub.get("score") or {}
        final = score_obj.get("final_score")
        if not isinstance(final, (int, float)) or final < min_score:
            return None
        curves = sub.get("curves") or {}
        curve = curves.get(target_id)
        if not _curve_is_usable(curve):
            return None
        return RollingBestCurve(
            task_id=target_id,
            eval_points=curve["eval_points"],
            train_points=curve.get("train_points") or [],
            wall_seconds=float(curve.get("wall_seconds", 0.0)),
            state_multiplier=float(curve.get("state_multiplier", 1.0)),
            final_eval_loss=float(curve["eval_points"][-1][1]),
            source_round_id=rid,
            source_code_hash=str(sub.get("code_hash", ""))[:16],
            source_score=float(final),
        )

    for fp in sorted(rounds_dir.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception as e:
            log.debug(f"rolling_best: skip {fp.name} (parse fail: {e})")
            continue

        rid = d.get("round_id", fp.stem)
        if exclude_round_id and rid == exclude_round_id:
            continue

        round_tasks = {t.get("task_id"): t for t in d.get("tasks", [])}
        rt = round_tasks.get(target_id)
        if not rt or task_signature(rt) != target_sig:
            continue
        rounds[rid] = d

        is_genesis = bool(d.get("genesis"))
        for hk, sub in (d.get("submissions") or {}).items():
            rb = _record(rid, "submissions", hk, sub)
            if rb is None:
                continue
            candidates.append(rb)
            if is_genesis:
                anchors.append(rb)
        ev = (d.get("frontier_rewards") or {}).get("new_event")
        if isinstance(ev, dict) and ev.get("hotkey"):
            event_refs.append((ev, d))

    for ev, d in event_refs:
        ref = ev.get("baseline_curve_ref") or {}
        ref_rid = str(ref.get("round_id") or "") or str(d.get("round_id", ""))
        section = str(ref.get("section") or "submissions")
        key = str(ref.get("key") or ev.get("hotkey", ""))
        src_round = rounds.get(ref_rid)
        if src_round is None or (exclude_round_id and ref_rid == exclude_round_id):
            continue
        sub = (src_round.get(section) or {}).get(key)
        if not isinstance(sub, dict):
            continue
        ev_hash = str(ev.get("code_hash", "") or "")
        sub_hash = str(sub.get("code_hash", "") or "")
        if ev_hash and sub_hash and ev_hash != sub_hash:
            log.warning(f"rolling_best: event {ev.get('event_id', '')} curve ref "
                        f"{ref_rid}/{section}/{key} has code_hash {sub_hash[:12]} != "
                        f"event {ev_hash[:12]}; ignoring")
            continue
        rb = _record(ref_rid, section, key, sub)
        if rb is not None:
            anchors.append(rb)

    if not candidates and not anchors:
        log.info(f"rolling_best: no historical curves for {target_id} "
                 f"(sig={target_sig}); caller should cold-start")
        return None

    pool = anchors
    if not pool:
        log.warning(f"rolling_best: no genesis/frontier anchor for {target_id}; "
                    f"falling back to the best of {len(candidates)} scored curve(s)")
        pool = candidates

    best = min(pool, key=lambda c: c.final_eval_loss)
    log.info(f"rolling_best: {target_id} → loss={best.final_eval_loss:.4f} "
             f"from round={best.source_round_id} "
             f"hash={best.source_code_hash} score={best.source_score:+.4f} "
             f"(out of {len(pool)} anchor(s), {len(candidates)} scored curve(s))")
    return best



COLD_START_ANCHORS = {
    "R1_smol_135M": (11.3, 3.5, 2.0),
    "R2_smol_360M": (11.3, 3.2, 1.8),
    "R3_smol_1.7B": (11.3, 2.8, 1.5),
    "FW1_smol_135M": (11.3, 3.5, 2.0),
    "FW2_smol_360M": (11.3, 3.2, 1.8),
    "FW3_smol_1.7B": (11.3, 2.8, 1.5),
}
DEFAULT_ANCHORS = (11.3, 4.0, 2.0)


def cold_start_score(task_id: str, final_eval_loss: float) -> float:
    """Map raw eval-loss → [-1, +1] convergence score for round 1.

    Uses task-family anchors (init / ref / floor) and linear interpolation.
    This is intentionally simple — once rolling-best exists from round 2
    onwards, this function is only invoked for never-before-seen TaskSpecs.
    """
    init, ref, floor = COLD_START_ANCHORS.get(task_id, DEFAULT_ANCHORS)
    if final_eval_loss >= init:
        return -1.0
    if final_eval_loss <= floor:
        return 1.0
    if final_eval_loss >= ref:
        return -1.0 + (init - final_eval_loss) / (init - ref)
    return (ref - final_eval_loss) / (ref - floor)



def _self_test() -> None:
    """In-process self-test. Exits 0 on success, raises on failure."""
    import tempfile, shutil

    tmp = Path(tempfile.mkdtemp(prefix="rolling_baseline_test_"))
    try:
        task = {"task_id": "R1_smol_135M", "model_config": "HF/SmolLM2-135M",
                "total_steps": 500, "batch_size": 4, "sequence_length": 256,
                "use_pretrained": False, "compute_budget_seconds": 28800.0}
        rb = load_rolling_best(tmp, task)
        assert rb is None, f"empty dir should return None, got {rb}"

        round_a = {
            "round_id": "round_a",
            "tasks": [task],
            "submissions": {
                "miner_x": {
                    "score": {"final_score": 0.05},
                    "code_hash": "aabbccdd" + "0" * 56,
                    "curves": {
                        "R1_smol_135M": {
                            "eval_points": [[0, 11.3, 0.5], [500, 4.0, 100.0]],
                            "train_points": [[0, 11.3], [500, 4.1]],
                            "wall_seconds": 100.0, "state_multiplier": 1.5,
                            "failed": False, "error": ""}}}}}
        round_b = {
            "round_id": "round_b",
            "tasks": [task],
            "submissions": {
                "miner_y": {
                    "score": {"final_score": -0.10},
                    "code_hash": "ffeeddcc" + "0" * 56,
                    "curves": {
                        "R1_smol_135M": {
                            "eval_points": [[0, 11.3, 0.5], [500, 5.5, 100.0]],
                            "train_points": [[0, 11.3], [500, 5.6]],
                            "wall_seconds": 100.0, "state_multiplier": 1.5,
                            "failed": False, "error": ""}}}}}
        round_c = {
            "round_id": "round_c",
            "tasks": [{**task, "compute_budget_seconds": 3600.0}],
            "submissions": {
                "miner_z": {
                    "score": {"final_score": 0.30},
                    "code_hash": "deadbeef" + "0" * 56,
                    "curves": {
                        "R1_smol_135M": {
                            "eval_points": [[0, 11.3, 0.5], [100, 2.0, 50.0]],
                            "train_points": [[0, 11.3], [100, 2.0]],
                            "wall_seconds": 50.0, "state_multiplier": 1.5,
                            "failed": False, "error": ""}}}}}
        round_d = {
            "round_id": "round_d",
            "tasks": [task],
            "submissions": {
                "miner_w": {
                    "score": {"final_score": -1.0},
                    "code_hash": "f4f4f4f4" + "0" * 56,
                    "curves": {
                        "R1_smol_135M": {
                            "eval_points": [], "train_points": [],
                            "wall_seconds": 0.0, "state_multiplier": 1.5,
                            "failed": True, "error": "OOM"}}}}}

        for r in (round_a, round_b, round_c, round_d):
            (tmp / f"{r['round_id']}.json").write_text(json.dumps(r))

        rb = load_rolling_best(tmp, task)
        assert rb is not None, "should find a rolling best"
        assert rb.source_round_id == "round_a", \
            f"should pick round_a (loss=4.0), got {rb.source_round_id} (loss={rb.final_eval_loss})"
        assert abs(rb.final_eval_loss - 4.0) < 1e-6, f"unexpected final_loss {rb.final_eval_loss}"
        assert rb.source_code_hash == "aabbccdd" + "00000000"

        rb2 = load_rolling_best(tmp, task, exclude_round_id="round_a")
        assert rb2 is not None and rb2.source_round_id == "round_b", \
            f"with round_a excluded, should pick round_b, got {rb2}"

        rb3 = load_rolling_best(tmp, task, min_score=0.0)
        assert rb3.source_round_id == "round_a"

        s_init = cold_start_score("R1_smol_135M", 11.3)
        assert s_init == -1.0
        s_ref = cold_start_score("R1_smol_135M", 3.5)
        assert abs(s_ref) < 1e-6, f"ref should be 0, got {s_ref}"
        s_floor = cold_start_score("R1_smol_135M", 2.0)
        assert s_floor == 1.0
        s_mid = cold_start_score("R1_smol_135M", 5.0)
        assert -1.0 < s_mid < 0.0, f"mid-poor should be in (-1, 0), got {s_mid}"
        s_great = cold_start_score("R1_smol_135M", 2.5)
        assert 0.0 < s_great < 1.0, f"between ref and floor should be (0,1), got {s_great}"
        s_unknown = cold_start_score("FROBNICATE_99B", 4.0)
        assert s_unknown == 0.0, f"unknown task at default ref=4.0 should be 0, got {s_unknown}"

        print(f"[PASS] rolling_baseline self-test ({len(list(tmp.glob('*.json')))} fixtures)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _self_test()
