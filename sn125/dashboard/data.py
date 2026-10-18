"""Public dashboard data contract for Refinery / SN125 (formerly Forge).

The dashboard is intentionally static-first: production can publish the JSON
objects this module builds to an object store behind dashboard.*, while local
development serves them dynamically from ``serve.py``. The live production path
is not integration-tested here because it depends on the deployed validator,
artifact store, and chain indexer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import (
    BURN_UID, LAUNCH_BURN_FRACTION_FLOOR, LAUNCH_BURN_SCHEDULE,
    LAUNCH_FIRST_CYCLE_DAYS, REFINER_START_TIMESTAMP, ROUND_GRID_ANCHOR_TIMESTAMP,
)


SN125_VERSION_FALLBACK = "0.1.0"
BURN_KEY = "__BURN__"          # mirrors Validator.BURN_KEY
RAO_PER_TAO = 10**9

AUDIENCE_NOTES = {
    "investors": [
        "Is the subnet buying useful optimizer research rather than raw volume?",
        "Which discoveries improved the rolling baseline, and how much emission was earned or burned?",
        "How much paid evaluation demand, compute spend, and public research output is happening?",
    ],
    "miners": [
        "What exact task, budget, baseline, deadline, and GPU SKU must I beat?",
        "How many submissions can be evaluated, what is deferred, and what failure modes burn credits?",
        "Which optimizer families are winning, and where are gaps left to attack?",
    ],
    "observers": [
        "Can I reconstruct round selection, payment credits, scores, weights, and attestations?",
        "Can I verify code hashes, quartile checkpoint artifacts, clean scoring, and the audit trail?",
        "Are infra faults separated from miner faults and refunded consistently?",
    ],
}

PROJECT_INTENT = {
    "name": "Refinery SN125",
    "tagline": "A decentralized, incentivized market for better LLM training optimizers.",
    "mission": (
        "Miners submit optimizer update rules. Validators run a fixed-budget, "
        "commit-reveal evaluation against the rolling best optimizer and emit "
        "rewards only for useful held-out-loss improvement; unearned emission burns."
    ),
    "active_mechanism": "Rolling-baseline fixed-budget evaluation",
    "production_target": "b200-small, FineWeb-Edu, lean-360M decoder with SmolLM2 tokenizer, seq 2048, 20h wall-clock budget, flash attention (mostly-deterministic; 25/50/75/100% checkpoint audit)",
}

EMISSIONS = {
    "subnet_daily_tokens": 7200.0,
    "miner_emission_share": 0.41,
    "miner_daily_tokens": 7200.0 * 0.41,
    "burn_uid": BURN_UID,
    "burn_fraction_floor": LAUNCH_BURN_FRACTION_FLOOR,
    "burn_schedule": list(LAUNCH_BURN_SCHEDULE),
    "research_review_after_days": LAUNCH_FIRST_CYCLE_DAYS,
    "research_pause_scheduled": False,
    # Round schedule: rounds open on a 24 h grid anchored here (0 = no grid),
    # and Challenge 2 is live from the round that opens at its start instant.
    "round_grid_anchor": int(ROUND_GRID_ANCHOR_TIMESTAMP or 0),
    "round_period_s": 86400,
    "refiner_start_timestamp": int(REFINER_START_TIMESTAMP or 0),
}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _as_score(submission: dict[str, Any]) -> dict[str, Any]:
    score = submission.get("score", submission)
    if not isinstance(score, dict):
        return {}
    return score


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _point_loss(points: list[Any]) -> float | None:
    if not points:
        return None
    point = points[-1]
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return _num(point[1])
    return _num(point)


def _short_hash(value: str, n: int = 12) -> str:
    return value[:n] if value else ""


def _normalize_dash_checkpoints(cps: list[Any]) -> list[dict[str, Any]]:
    """Clean quartile audit-checkpoint records for dashboard display (pct/step/
    eval_loss/sha256/uri). Tolerant of partial records; never raises."""
    out = []
    for c in cps or []:
        if not isinstance(c, dict):
            continue
        uri = str(c.get("uri", ""))
        out.append({
            "pct": _num(c.get("pct"), 0.0),
            "step": int(_num(c.get("step"), 0)),
            "eval_loss": _num(c.get("eval_loss"), 0.0),
            "sha256": _short_hash(str(c.get("sha256", "")), 16),
            # Box-local artifact refs never belong in public dashboard JSON.
            "uri": "" if _is_local_artifact_ref(uri) else uri,
        })
    out.sort(key=lambda r: (r["pct"], r["step"]))
    return out


def _round_files(rounds_dir: Path) -> list[Path]:
    if not rounds_dir.exists():
        return []
    return sorted(rounds_dir.glob("*.json"), reverse=True)


def _refiner_frontier_summary(fr: Any) -> dict[str, Any] | None:
    """Public view of the refiner frontier settled in a round: bar, pool payout,
    the event (if one confirmed), what is awaiting confirmation."""
    if not isinstance(fr, dict):
        return None
    ev = fr.get("new_event") if isinstance(fr.get("new_event"), dict) else None
    return {
        "scheme": fr.get("scheme", ""),
        "emission_share": _num(fr.get("emission_share"), 0.0),
        "unit": fr.get("improvement_unit", "suite_points"),
        "bar": fr.get("min_improvement_threshold"),
        "floor": fr.get("min_improvement_floor"),
        "baseline_score": fr.get("baseline_score"),
        "frontier_score": fr.get("frontier_score"),
        "leader_hotkey": fr.get("leader_hotkey", ""),
        "progress": fr.get("progress"),
        "full_pay_reference": fr.get("full_pay_reference"),
        "payable_share": fr.get("payable_share"),
        "burned_share": fr.get("burned_share"),
        "new_event": ({"hotkey": ev.get("hotkey", ""), "commit_hash": _short_hash(str(ev.get("commit_hash", "")), 16),
                       "improvement": ev.get("improvement"), "suite_score": ev.get("suite_score"),
                       "event_id": ev.get("event_id", "")} if ev else None),
        "pending": [{"hotkey": p.get("hotkey", ""), "commit_hash": _short_hash(str(p.get("commit_hash", "")), 16),
                     "improvement": p.get("improvement"), "suite_score": p.get("suite_score"),
                     "submission_round_id": p.get("source_round_id", "")}
                    for p in (fr.get("pending_confirmation") or []) if isinstance(p, dict)],
        "events": [{"hotkey": e.get("hotkey", ""), "improvement": e.get("improvement"), "age_days": e.get("age_days"),
                    "share": e.get("share"), "suite_score": e.get("suite_score")}
                   for e in (fr.get("events") or []) if isinstance(e, dict)],
        "weights": {hk: w for hk, w in (fr.get("weights") or {}).items() if isinstance(w, (int, float))},
    }


def _refiner_round_summary(rd: dict[str, Any]) -> dict[str, Any] | None:
    """Public view of a round's ``refiner_task`` block (Challenge 2): per-entry
    suite scores and provenance, nothing box-local, plus the frontier the
    round settled. ``None`` when the round carried no refiner task at all —
    the summary is then exactly what it was before the task existed."""
    block = rd.get("refiner_task")
    if not isinstance(block, dict) or not (block.get("submissions") or block.get("frontier")):
        return None
    entries = []
    for ch, e in (block.get("submissions") or {}).items():
        if not isinstance(e, dict):
            continue
        bench = e.get("bench") or {}
        stats = e.get("refiner_stats") or {}
        task = e.get("task") or {}
        # task_scores = {task_id: {"score", "per_task": {bench task: {metric: {submission, baseline, delta}}}}}
        tasks = {}
        for rec in (e.get("task_scores") or {}).values():
            for name, metrics in ((rec.get("per_task") or {}).items() if isinstance(rec, dict) else []):
                pcn = (metrics or {}).get("prob_correct_norm") or (metrics or {}).get("acc") or {}
                if isinstance(pcn, dict):
                    tasks[name] = {"submission": pcn.get("submission"), "baseline": pcn.get("baseline"),
                                   "delta": pcn.get("delta")}
        comp = e.get("components") or {}
        entries.append({
            "hotkey": e.get("hotkey", ""),
            "commit_hash": _short_hash(str(ch), 16),
            "score": e.get("score"),
            "suite_submission": bench.get("submission"),
            "suite_baseline": bench.get("baseline"),
            "baseline_source": "pinned" if str(bench.get("baseline_source") or "").endswith(".json") else
                               ("trained on the box" if bench.get("baseline_source") == "run" else
                                str(bench.get("baseline_source") or "")),
            "heldout_submission": comp.get("heldout_loss_submission"),
            "heldout_baseline": comp.get("heldout_loss_baseline"),
            "weights": e.get("weights"),
            "source_sha256": _short_hash(str(e.get("source_sha256") or ""), 16),
            "signature": e.get("signature", ""),
            "seed": task.get("seed"),
            "budget_seconds": task.get("compute_budget_seconds"),
            "steps": stats.get("final_step"),
            "lessons": stats.get("lessons"),
            "entries_consumed": stats.get("entries_consumed"),
            "learn_fraction": stats.get("learn_fraction"),
            "plan_seconds": stats.get("plan_seconds"),
            "tasks": tasks,
            "paired": e.get("paired"),
            "suite_delta": comp.get("suite_delta", e.get("score")),
            "gates": e.get("gates"),
            "checkpoints": [{"pct": c.get("pct"), "step": c.get("step"), "sha256": c.get("sha256"),
                             "uri": c.get("uri")} for c in (e.get("checkpoints") or []) if isinstance(c, dict)],
            "weights_archive": e.get("refiner_weights_archive", ""),
            "provider": ((e.get("provenance") or {}).get("provider") if isinstance(e.get("provenance"), dict) else ""),
        })
    entries.sort(key=lambda x: _num(x.get("score"), -999.0), reverse=True)
    return {
        "scheme": block.get("scheme", ""),
        "task_id": block.get("task_id", ""),
        "emission_share": _num(block.get("emission_share"), 0.0),
        "suite": block.get("suite") or {},
        "decontam_sha256": block.get("decontam_sha256", ""),
        "baseline_sha256": block.get("baseline_sha256", ""),
        "signature": block.get("signature", ""),
        "entries": entries,
        "best": entries[0] if entries else None,
        "frontier": _refiner_frontier_summary(block.get("frontier")),
    }


def _refiner_overview(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Top-level Challenge-2 section: is the task on, what it pays, the latest
    frontier state (bar, leader, payable share, pending confirmations), every
    published refiner result newest first, and the best per hotkey."""
    rounds = [(r["round_id"], r["timestamp"], r["refiner"]) for r in summaries if r.get("refiner")]
    results, best_by_hotkey = [], {}
    for rid, ts, blk in rounds:
        for e in blk["entries"]:
            rec = {"round_id": rid, "timestamp": ts, **e}
            results.append(rec)
            cur = best_by_hotkey.get(e["hotkey"])
            if cur is None or _num(e.get("score"), -999.0) > _num(cur.get("score"), -999.0):
                best_by_hotkey[e["hotkey"]] = rec
    latest = rounds[0][2] if rounds else None
    share = _num(latest["emission_share"], 0.0) if latest else 0.0
    frontier = latest.get("frontier") if latest else None
    events = [{"round_id": rid, "timestamp": ts, **blk["frontier"]["new_event"]}
              for rid, ts, blk in rounds if blk.get("frontier") and blk["frontier"].get("new_event")]
    start_ts = int(REFINER_START_TIMESTAMP or 0)
    opens_at = (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(start_ts))
                if start_ts and not rounds else "")
    return {
        "enabled": bool(rounds),
        "status": ((f"opens {opens_at}" if opens_at else "not yet live") if not rounds else
                   "publish-only" if share <= 0 else f"pays {share:.0%} of emission"),
        "start_timestamp": start_ts,
        "round_grid_anchor": int(ROUND_GRID_ANCHOR_TIMESTAMP or 0),
        "task_id": latest["task_id"] if latest else "RF1_smol_360M",
        "emission_share": share,
        "suite": latest["suite"] if latest else {},
        "signature": latest["signature"] if latest else "",
        "frontier": frontier,
        "events": events,
        "rounds_with_results": sum(1 for _, _, blk in rounds if blk["entries"]),
        "total_results": len(results),
        "results": results[:200],
        "leaderboard": sorted(best_by_hotkey.values(), key=lambda x: _num(x.get("score"), -999.0), reverse=True),
        "docs": ["docs/REFINER_TASK.md", "docs/REFINER_RESEARCH.md"],
    }


def _round_summary(path: Path, rd: dict[str, Any]) -> dict[str, Any]:
    submissions = rd.get("submissions", {}) or {}
    fsm_report = rd.get("fsm_report", {}) or {}
    selection = rd.get("selection", {}) or {}
    fsm_submissions = fsm_report.get("submissions", []) or []
    results = rd.get("results", {}) or {}
    if not results:
        results = {hk: _as_score(sub) for hk, sub in submissions.items()}

    best_hotkey = ""
    best_score = None
    scored = 0
    dq = 0
    total_weight = 0.0
    burned_weight = 0.0
    for hotkey, raw in results.items():
        score = _as_score(raw)
        final = score.get("final_score")
        if final is None:
            dq += 1
            continue
        scored += 1
        final_f = _num(final, -999.0)
        if best_score is None or final_f > best_score:
            best_score = final_f
            best_hotkey = hotkey
        sub = submissions.get(hotkey, {})
        total_weight += _num(sub.get("weight", 0.0))

    # Burn can be explicit on real round files; otherwise infer conservatively.
    economics = rd.get("economics", {}) or {}
    if "burned_weight" in economics:
        burned_weight = _num(economics.get("burned_weight"))
    elif total_weight <= 1.0:
        burned_weight = max(0.0, 1.0 - total_weight)

    base_losses = {}
    for task_id, curve in (rd.get("baselines", {}) or {}).items():
        loss = _point_loss(curve.get("eval_points", []) or [])
        if loss is not None:
            base_losses[task_id] = loss

    # Quartile audit checkpoints (25/50/75/100%) — the audit-replay replacement.
    # Count across all submissions; expose the best submission's list for detail.
    audit_checkpoint_count = 0
    for sub in submissions.values():
        cps = sub.get("checkpoints", []) or []
        if not cps:
            for curve in (sub.get("curves", {}) or {}).values():
                if isinstance(curve, dict):
                    cps = cps + list(curve.get("checkpoints", []) or [])
        audit_checkpoint_count += len(cps)
    best_sub = submissions.get(best_hotkey, {}) or {}
    best_checkpoints = _normalize_dash_checkpoints(best_sub.get("checkpoints", []) or [])
    if not best_checkpoints:
        # Rich round files keep quartile checkpoints under per-task curves.
        for curve in (best_sub.get("curves", {}) or {}).values():
            if isinstance(curve, dict) and curve.get("checkpoints"):
                best_checkpoints = _normalize_dash_checkpoints(curve["checkpoints"])
                break

    # Best-vs-baseline held-out detail from the winning submission's task_scores.
    # This is what lets the dashboard chart actual loss frontier movement rather
    # than only the abstract final_score.
    best_score_rec = _as_score(best_sub)
    best_loss = base_loss = None
    best_steps = base_steps = 0
    for ts_rec in (best_score_rec.get("task_scores", {}) or {}).values():
        if not isinstance(ts_rec, dict):
            continue
        if ts_rec.get("sub_final_loss") is not None:
            best_loss = _num(ts_rec.get("sub_final_loss"))
        if ts_rec.get("base_final_loss") is not None:
            base_loss = _num(ts_rec.get("base_final_loss"))
        best_steps = int(_num(ts_rec.get("sub_steps"), 0))
        base_steps = int(_num(ts_rec.get("base_steps"), 0))
        break  # production is single-task
    if base_loss is None and base_losses:
        base_loss = next(iter(base_losses.values()))
    if best_loss is None and best_checkpoints:
        best_loss = best_checkpoints[-1]["eval_loss"] or None
    improvement = None
    if best_loss is not None and base_loss:
        improvement = max(0.0, (base_loss - best_loss) / base_loss)

    # Optimizer-family mix among scored submissions (for the round texture).
    families: dict[str, int] = {}
    for sub in submissions.values():
        score = _as_score(sub)
        if "final_score" not in score:
            continue
        fam = _optimizer_family(sub, score)
        families[fam] = families.get(fam, 0) + 1

    # Hardware provenance: which provider fleet scored each submission, plus
    # the winning submission's throughput-band evidence (pre/post box probe,
    # achieved rate). Written by the validator (roundsm/live.py) — public-safe
    # scalars only.
    providers: dict[str, int] = {}
    for sub in submissions.values():
        prov = ((sub.get("provenance") or {}).get("provider")
                if isinstance(sub, dict) else "")
        if prov:
            providers[prov] = providers.get(prov, 0) + 1
    best_provenance = (best_sub.get("provenance") or {}) if isinstance(best_sub, dict) else {}

    # Frontier state written by the validator's reward path: whether this
    # round created a (confirmed) frontier event, which candidates are still
    # awaiting their validator-funded confirmation run, and the bar (nats).
    fr = rd.get("frontier_rewards") or {}
    new_event = fr.get("new_event") if isinstance(fr.get("new_event"), dict) else None
    frontier = {
        "improvement_unit": fr.get("improvement_unit", "nats"),
        "min_improvement_threshold": _num(fr.get("min_improvement_threshold"), 0.0),
        "confirmation_runs": int(_num(fr.get("confirmation_runs"), 0)),
        "leader_hotkey": fr.get("leader_hotkey", ""),
        # Progress-scaled payout inputs (scheme frontier-progress-scaled-v3):
        # recent verified progress, the full-pay reference and the share of
        # emission that progress bought this round.
        "scheme": fr.get("scheme", ""),
        "progress_nats": _num(fr.get("progress_nats"), 0.0),
        "full_pay_reference_nats": _num(fr.get("full_pay_reference_nats"), 0.0),
        "progress_payable_share": _num(fr.get("progress_payable_share"), 0.0),
        "new_event": ({"hotkey": new_event.get("hotkey", ""),
                       "code_hash": str(new_event.get("code_hash", ""))[:16],
                       "improvement": _num(new_event.get("improvement"), 0.0),
                       "task_final_losses": new_event.get("task_final_losses", {}) or {}}
                      if new_event else None),
        "pending_confirmation": [
            {"hotkey": p.get("hotkey", ""), "code_hash": str(p.get("code_hash", ""))[:16],
             "improvement": _num(p.get("improvement"), 0.0),
             "submission_round_id": p.get("source_round_id", ""),
             "attempts": int(_num(p.get("attempts"), 0))}
            for p in (fr.get("pending_confirmation") or []) if isinstance(p, dict)],
        "confirmation_outcomes": [
            {"hotkey": o.get("hotkey", ""), "outcome": o.get("outcome", ""),
             "reason": o.get("reason", "")}
            for o in (fr.get("confirmation_outcomes") or []) if isinstance(o, dict)],
    }

    tasks = rd.get("tasks", []) or []
    task = tasks[0] if tasks else {}
    # Who was paid this round (frontier-event hotkeys, which need not have a
    # submission in this round) — the emission split the dashboard shows.
    paid_weights = {
        str(hk): _num(w) for hk, w in (rd.get("weights") or {}).items()
        if hk != BURN_KEY and _num(w) > 0
    }
    return {
        "round_id": rd.get("round_id", path.stem),
        "timestamp": int(_num(rd.get("timestamp"), 0)),
        "phase": rd.get("phase") or ("published" if scored or dq else "unknown"),
        "miners": len(submissions) or len(results) or len(fsm_submissions),
        "accepted": len(fsm_submissions),
        "selected": len(selection.get("selected", []) or []),
        "commit_rejections": len(selection.get("commit_rejections", []) or []),
        "scored": scored,
        "dq": dq,
        "deferred": len(rd.get("deferred", []) or selection.get("deferred", []) or []),
        "best_score": best_score if best_score is not None else 0.0,
        "best_hotkey": best_hotkey,
        "best_code_hash": submissions.get(best_hotkey, {}).get("code_hash", ""),
        "burned_weight": burned_weight,
        "paid_weight": max(0.0, 1.0 - burned_weight),
        "paid_weights": paid_weights,
        "base_losses": base_losses,
        "best_loss": best_loss,
        "base_loss": base_loss,
        "best_steps": best_steps,
        "base_steps": base_steps,
        "improvement": improvement,
        "frontier": frontier,
        "families": families,
        "providers": providers,
        "best_provenance": best_provenance,
        "audit_checkpoint_count": audit_checkpoint_count,
        "best_checkpoints": best_checkpoints,
        "pause_reasons": list(rd.get("pause_reasons", []) or []),
        "refiner": _refiner_round_summary(rd),
        "task": {
            "family": task.get("task_id", task.get("model_config", "FineWeb-Edu")),
            "model": task.get("model_config", "SmolLM2-family"),
            "steps": int(_num(task.get("total_steps"), 0)),
            "batch_size": int(_num(task.get("batch_size"), 0)),
            "sequence_length": int(_num(task.get("sequence_length"), 0)),
        },
    }


def _leaderboard(rounds: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    by_hotkey: dict[str, dict[str, Any]] = {}
    paid_total: dict[str, float] = {}
    for _path, rd in rounds:
        best_hotkey = ""
        best_score = None
        for hotkey, sub in (rd.get("submissions", {}) or {}).items():
            score = _as_score(sub)
            if "final_score" not in score:
                continue
            final = _num(score.get("final_score"), -999.0)
            if best_score is None or final > best_score:
                best_score = final
                best_hotkey = hotkey
        for hotkey, sub in (rd.get("submissions", {}) or {}).items():
            score = _as_score(sub)
            if "final_score" not in score:
                continue
            row = by_hotkey.setdefault(hotkey, {
                "hotkey": hotkey,
                "rounds": 0,
                "wins": 0,
                "best_score": -999.0,
                "ema_score": 0.0,
                "code_hash": "",
                "families": {},
                "components": {"convergence": 0.0, "generalization": 0.0, "cross_scale": 0.0},
                "weight": 0.0,
                "earned_tokens": 0.0,
            })
            final = _num(score.get("final_score"), -999.0)
            row["rounds"] += 1
            if hotkey == best_hotkey:
                row["wins"] += 1
            row["ema_score"] = 0.65 * row["ema_score"] + 0.35 * final if row["rounds"] > 1 else final
            comps = score.get("components", {}) or {}
            for key in row["components"]:
                row["components"][key] += _num(comps.get(key), 0.0)
            if final > row["best_score"]:
                row["best_score"] = final
                row["code_hash"] = sub.get("code_hash", "")
            family = _optimizer_family(sub, score)
            row["families"][family] = row["families"].get(family, 0) + 1
        # Emission is paid to confirmed frontier-event hotkeys via the round's
        # weight map, whether or not they submitted again this round; that is
        # what "earned" must count (the per-submission weight is 0 for a leader
        # who does not resubmit, and there is no submission at all in the
        # confirmation round).
        weight_map = rd.get("weights") if isinstance(rd.get("weights"), dict) else None
        if not weight_map:   # legacy round files carry weights per submission only
            weight_map = {hk: (sub or {}).get("weight", 0.0)
                          for hk, sub in (rd.get("submissions", {}) or {}).items()}
        for hotkey, weight in weight_map.items():
            w = _num(weight)
            if hotkey != BURN_KEY and w > 0:
                paid_total[str(hotkey)] = paid_total.get(str(hotkey), 0.0) + w
    for hotkey, w in paid_total.items():
        if hotkey in by_hotkey:   # rounds arrive newest-first; apply after the scan
            by_hotkey[hotkey]["weight"] = w
            by_hotkey[hotkey]["earned_tokens"] = w * EMISSIONS["miner_daily_tokens"]

    rows = []
    for row in by_hotkey.values():
        rounds_seen = max(1, int(row["rounds"]))
        family = max(row["families"], key=row["families"].get) if row["families"] else "unknown"
        rows.append({
            "hotkey": row["hotkey"],
            "rank": 0,
            "score": row["ema_score"],
            "best_score": row["best_score"],
            "rounds": row["rounds"],
            "wins": row["wins"],
            "weight": row["weight"],
            "earned_tokens": row["earned_tokens"],
            "code_hash": row["code_hash"],
            "code_hash_short": _short_hash(row["code_hash"]),
            "family": family,
            "conv": row["components"]["convergence"] / rounds_seen,
            "gen": row["components"]["generalization"] / rounds_seen,
            "cross": row["components"]["cross_scale"] / rounds_seen,
        })
    rows.sort(key=lambda r: (r["score"], r["best_score"]), reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def _optimizer_family(submission: dict[str, Any], score: dict[str, Any]) -> str:
    hparams = submission.get("hparams") or submission.get("hparams_per_task") or score.get("best_hparams") or {}
    code_hash = submission.get("code_hash", "")
    text = json.dumps(hparams, sort_keys=True).lower() + " " + code_hash.lower()
    if "triton" in text:
        return "fused/native"
    if "lion" in text:
        return "lion-like"
    if "muon" in text or "orth" in text:
        return "orthogonalized"
    if "adam" in text or hparams:
        return "adam-family"
    return "custom"


_LOCAL_ARTIFACT_KEYS = {
    "path",
    "checkpoint_out",
    "quartile_checkpoints",
    "prod_eval_result",
    "data_dir",
}


def _is_local_artifact_ref(value: str) -> bool:
    s = str(value or "")
    return (
        s.startswith("/workspace/")
        or s.startswith("/tmp/")
        or s.startswith("file:")
        or (s.startswith("targon:") and ":/workspace/" in s)
    )


def _scrub_public_artifact_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _LOCAL_ARTIFACT_KEYS:
                continue
            if k == "uri" and isinstance(v, str) and _is_local_artifact_ref(v):
                out[k] = ""
            else:
                out[k] = _scrub_public_artifact_paths(v)
        return out
    if isinstance(obj, list):
        return [
            _scrub_public_artifact_paths(v)
            for v in obj
            if not (isinstance(v, str) and _is_local_artifact_ref(v))
        ]
    if isinstance(obj, str) and _is_local_artifact_ref(obj):
        return ""
    return obj


def _latest_round_public(latest: dict[str, Any] | None) -> dict[str, Any] | None:
    if latest is None:
        return None
    public = _scrub_public_artifact_paths({k: v for k, v in latest.items() if k != "sources"})
    submissions = public.get("submissions", {}) or {}
    for sub in submissions.values():
        if isinstance(sub, dict) and "source" in sub:
            sub.pop("source", None)
    return public


def _verification(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    latest = rounds[0] if rounds else {}
    return {
        "commit_reveal": "all commits close before reveal; revealed source must hash to commitment",
        "payment_gate": "one prepaid evaluation credit debited at commit acceptance",
        "clean_scoring": "score is computed from checkpoints in a process that never imported miner code",
        "audit": "mostly-deterministic regime: 25/50/75/100% quartile checkpoint artifacts (held-out loss + state sha256) are published per round so a held-out re-score is independently checkable",
        "hardware": "B200-only across a trusted provider failover chain; every score publishes its provider and pre/post throughput probes inside the pinned band, so hardware luck cannot move the frontier",
        "latest_round": latest.get("round_id", ""),
        "latest_signature": bool(latest.get("signature")),
        "artifact_paths": [
            "dashboard.json",
            "rounds/{round_id}.json",
            "cloud-status.json",
            "optimizer-summaries.json",
        ],
    }


def _public_payments(ledger_path: Path | None, treasury_coldkey: str,
                     recent_events: int = 60) -> dict[str, Any]:
    """Credit ledger view for miners: who holds how many prepaid evaluation
    credits, what was deposited/spent/refunded, and the latest ledger events.
    The ledger is public by design (registry.events_json: 'publishable
    ledger'); nothing here is secret — treasury transfers are on chain."""
    from ..ops import build_credits
    base = {"available": False, "treasury_coldkey": treasury_coldkey, "accounts": [],
            "recent_events": [], "totals": {}}
    if ledger_path is None:
        return {**base, "reason": "no ledger configured"}
    try:
        credits = build_credits(Path(ledger_path), recent_events=recent_events)
    except Exception as exc:  # the dashboard must never fail on the ledger
        return {**base, "reason": f"{type(exc).__name__}: {exc}"}
    if not credits.get("available"):
        return {**base, "reason": credits.get("reason") or credits.get("error", "")}
    accounts = []
    for ck, r in (credits.get("coldkeys") or {}).items():
        accounts.append({
            "coldkey": ck, "credits": int(r.get("credits") or 0),
            "carry_rao": int(r.get("carry_rao") or 0),
            "carry_tao": _num(r.get("carry_tao")),
            "deposits": int(r.get("deposits") or 0),
            "deposited_tao": _num(r.get("deposited_tao")),
            "granted": int(r.get("granted") or 0), "debited": int(r.get("debited") or 0),
            "refunded": int(r.get("refunded") or 0),
            "hotkeys": list(r.get("hotkeys") or []),
            "last_round_id": r.get("last_round_id"),
        })
    accounts.sort(key=lambda a: (-a["credits"], -a["deposited_tao"], a["coldkey"]))
    events = []
    for ev in credits.get("recent_events") or []:
        if not isinstance(ev, dict):
            continue
        events.append({
            "seq": ev.get("seq"), "kind": ev.get("kind"), "round_id": ev.get("round_id"),
            "coldkey": ev.get("coldkey"), "hotkey": ev.get("hotkey"),
            "credits": int(ev.get("credits") or 0),
            "tao": _num(ev.get("rao")) / RAO_PER_TAO if ev.get("rao") else 0.0,
            "carry_tao": _num(ev.get("carry_rao")) / RAO_PER_TAO if ev.get("carry_rao") else 0.0,
            "tx_id": ev.get("tx_id"), "note": ev.get("note", ""),
        })
    fee_rao = credits.get("fee_rao")
    return {
        "available": True,
        "treasury_coldkey": treasury_coldkey,
        "fee_rao": fee_rao,
        "fee_tao": _num(fee_rao) / RAO_PER_TAO if fee_rao else None,
        "updated_at": credits.get("ledger_updated_at"),
        "last_block": credits.get("last_block"),
        "pinned_rounds": credits.get("pinned_rounds") or {},
        "totals": credits.get("totals") or {},
        "events_total": credits.get("events_total", len(events)),
        "accounts": accounts,
        "recent_events": events[::-1],          # newest first
    }


def _public_queue(state_path: Path | None) -> dict[str, Any]:
    """Deferred (paid, waiting) commits from the loop-state checkpoint."""
    from ..ops import _queue_from_state
    try:
        q = _queue_from_state(Path(state_path) if state_path else None)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "deferred": []}
    return {
        "available": bool(q.get("available")),
        "reason": q.get("reason") or q.get("error"),
        "round_num": q.get("round_num"),
        "last_round_id": q.get("last_round_id"),
        "state_saved_at": q.get("state_saved_at"),
        "deferred": [
            {k: s.get(k) for k in ("position", "hotkey", "coldkey", "commit_hash",
                                   "deferrals", "accepted_at")}
            for s in (q.get("deferred_queue") or [])
        ],
    }


def _public_live_round(audit_dir: Path | None) -> dict[str, Any] | None:
    """The round in progress (from its audit trail), or None between rounds."""
    from ..ops import _live_round
    try:
        live = _live_round(Path(audit_dir) if audit_dir else None)
    except Exception:
        return None
    if not isinstance(live, dict) or not live.get("round_id"):
        return None
    subs = []
    for s in live.get("submissions") or []:
        subs.append({k: s.get(k) for k in (
            "hotkey", "coldkey", "commit_hash", "status", "last_event", "last_event_ts",
            "reason", "score", "outcome", "deferrals", "selection_rank", "credit_balance",
            "timeline")})
    return {
        "round_id": live.get("round_id"), "phase": live.get("phase"),
        "started_at": live.get("started_at"), "round_period_s": live.get("round_period_s"),
        "last_event_at": live.get("last_event_at"),
        "events": live.get("events"), "submissions": subs,
    }


def _submission_loss(sub: dict[str, Any]) -> tuple[float | None, float | None]:
    """(held-out loss, baseline loss) for one scored submission, from its
    task_scores or, failing that, the last eval point of its curve."""
    score = _as_score(sub)
    sub_loss = base_loss = None
    for ts_rec in (score.get("task_scores", {}) or {}).values():
        if not isinstance(ts_rec, dict):
            continue
        if ts_rec.get("sub_final_loss") is not None:
            sub_loss = _num(ts_rec.get("sub_final_loss"))
        if ts_rec.get("base_final_loss") is not None:
            base_loss = _num(ts_rec.get("base_final_loss"))
        break
    if sub_loss is None:
        for curve in (sub.get("curves", {}) or {}).values():
            if isinstance(curve, dict):
                sub_loss = _point_loss(curve.get("eval_points", []) or [])
                if sub_loss is not None:
                    break
    if sub_loss is None:
        for key in ("sub_final_loss", "final_loss", "loss"):
            if score.get(key) is not None:
                sub_loss = _num(score.get(key))
                break
    return sub_loss, base_loss


def _is_miner_key(key: Any, known: dict[str, Any]) -> bool:
    """Weight-map keys are hotkeys plus the burn sentinel; accept known hotkeys
    and anything shaped like an ss58 address, never the sentinel."""
    k = str(key)
    return k != BURN_KEY and (k in known or (len(k) >= 40 and k[0] == "5"))


def _miners_index(round_pairs: list[tuple[Path, dict[str, Any]]],
                  summaries: list[dict[str, Any]], leaderboard: list[dict[str, Any]],
                  payments: dict[str, Any], queue: dict[str, Any],
                  live: dict[str, Any] | None, latest: dict[str, Any] | None,
                  miner_daily_tokens: float) -> list[dict[str, Any]]:
    """Hotkey-first view: one record per hotkey that has ever committed, with
    its per-round history, money (via its coldkey), payout and what it is
    doing right now. This is the dashboard's primary layer; every other panel
    is an aggregate of it."""
    by_hk: dict[str, dict[str, Any]] = {}
    lb = {r["hotkey"]: r for r in leaderboard}
    ck_of: dict[str, str] = {}
    accounts = {a["coldkey"]: a for a in (payments.get("accounts") or [])}
    for a in accounts.values():
        for hk in a.get("hotkeys") or []:
            ck_of[str(hk)] = a["coldkey"]

    def rec(hk: str) -> dict[str, Any]:
        return by_hk.setdefault(hk, {
            "hotkey": hk, "coldkey": None, "family": None, "code_hash_short": None,
            "credits": None, "carry_tao": 0.0, "deposited_tao": 0.0, "spent": 0, "refunded": 0,
            "earned_tokens": 0.0, "weight_now": 0.0,
            "best_loss": None, "best_round_id": None, "last_loss": None, "last_round_id": None,
            "last_seen": 0, "rounds_scored": 0, "rounds_dq": 0, "rounds_deferred": 0, "rounds_expired": 0,
            "frontier_events": 0, "pending_confirmation": False, "is_leader": False,
            "status": {"state": "idle", "detail": "", "since": None},
            "history": [],
        })

    # Per-round history, oldest first so "last" fields end on the newest round.
    frontier_by_round = {s["round_id"]: s for s in summaries}
    for path, rd in reversed(round_pairs):
        rid = str(rd.get("round_id", path.stem))
        ts = _num(rd.get("timestamp"))
        summ = frontier_by_round.get(rid) or {}
        subs = rd.get("submissions", {}) or {}
        weights = rd.get("weights") if isinstance(rd.get("weights"), dict) else {}
        report = rd.get("fsm_report", {}) or {}
        seen: set[str] = set()
        for s in report.get("submissions") or []:
            if not isinstance(s, dict) or not s.get("hotkey"):
                continue
            hk = str(s["hotkey"])
            seen.add(hk)
            r = rec(hk)
            status = str(s.get("status") or "")
            sub = subs.get(hk) or {}
            loss, base = _submission_loss(sub) if sub else (None, None)
            if base is None:
                base = summ.get("base_loss")
            entry = {"round_id": rid, "timestamp": ts, "status": status, "loss": loss,
                     "base_loss": base, "score": s.get("score"),
                     "weight": _num(weights.get(hk, (sub or {}).get("weight", 0.0))),
                     "deferrals": int(_num(s.get("deferrals"))),
                     "commit_hash": s.get("commit_hash")}
            r["history"].append(entry)
            r["last_seen"] = max(r["last_seen"], ts)
            r["last_round_id"] = rid
            if status == "scored":
                r["rounds_scored"] += 1
                if loss is not None:
                    r["last_loss"] = loss
                    if r["best_loss"] is None or loss < r["best_loss"]:
                        r["best_loss"], r["best_round_id"] = loss, rid
            elif status == "deferred":
                r["rounds_deferred"] += 1
            elif status == "expired":
                r["rounds_expired"] = r.get("rounds_expired", 0) + 1
            elif status:
                r["rounds_dq"] += 1
            if sub:
                fam = _optimizer_family(sub, _as_score(sub))
                r["family"] = fam
                if sub.get("code_hash"):
                    r["code_hash_short"] = str(sub["code_hash"])[:12]
        # Legacy round files: scored submissions without an FSM report.
        for hk, sub in subs.items():
            if hk in seen or not isinstance(sub, dict):
                continue
            r = rec(str(hk))
            loss, base = _submission_loss(sub)
            scored = _as_score(sub).get("final_score") is not None
            r["history"].append({"round_id": rid, "timestamp": ts,
                                 "status": "scored" if scored else "dq", "loss": loss,
                                 "base_loss": base or summ.get("base_loss"),
                                 "score": _as_score(sub).get("final_score"),
                                 "weight": _num(weights.get(hk, sub.get("weight", 0.0))),
                                 "deferrals": 0, "commit_hash": sub.get("code_hash")})
            r["last_seen"] = max(r["last_seen"], ts)
            r["last_round_id"] = rid
            if scored:
                r["rounds_scored"] += 1
                if loss is not None:
                    r["last_loss"] = loss
                    if r["best_loss"] is None or loss < r["best_loss"]:
                        r["best_loss"], r["best_round_id"] = loss, rid
            else:
                r["rounds_dq"] += 1
            r["family"] = _optimizer_family(sub, _as_score(sub))
            if sub.get("code_hash"):
                r["code_hash_short"] = str(sub["code_hash"])[:12]
        for hk, w in weights.items():
            if _is_miner_key(hk, by_hk) and _num(w) > 0:
                rec(str(hk))["earned_tokens"] += _num(w) * miner_daily_tokens

    # Payout now + frontier events from the newest published round.
    fr = (latest or {}).get("frontier_rewards") or {}
    newest_weights = (latest or {}).get("weights") or {}
    for hk, w in newest_weights.items():
        if _is_miner_key(hk, by_hk) and _num(w) > 0:
            rec(str(hk))["weight_now"] = _num(w)
    for ev in fr.get("events") or []:
        if isinstance(ev, dict) and ev.get("hotkey"):
            rec(str(ev["hotkey"]))["frontier_events"] += 1
    for p in fr.get("pending_confirmation") or []:
        if isinstance(p, dict) and p.get("hotkey"):
            rec(str(p["hotkey"]))["pending_confirmation"] = True
    if fr.get("leader_hotkey"):
        rec(str(fr["leader_hotkey"]))["is_leader"] = True

    # What each hotkey is doing right now: live round beats queue beats record.
    for q in queue.get("deferred") or []:
        if q.get("hotkey"):
            r = rec(str(q["hotkey"]))
            r["status"] = {"state": "queued", "detail": f"deferred {q.get('deferrals') or 0}×, position {q.get('position')}",
                           "since": q.get("accepted_at")}
            if q.get("coldkey"):
                ck_of.setdefault(str(q["hotkey"]), str(q["coldkey"]))
    for s in (live or {}).get("submissions") or []:
        if not s.get("hotkey"):
            continue
        r = rec(str(s["hotkey"]))
        r["status"] = {"state": str(s.get("status") or "committed"),
                       "detail": s.get("reason") or "", "since": s.get("last_event_ts"),
                       "round_id": (live or {}).get("round_id"),
                       "commit_hash": s.get("commit_hash"),
                       "credit_balance": s.get("credit_balance"),
                       "timeline": list(s.get("timeline") or [])}
        if s.get("coldkey"):
            ck_of.setdefault(str(s["hotkey"]), str(s["coldkey"]))
    for hk, r in by_hk.items():
        if r["status"]["state"] == "idle":
            last = r["history"][-1] if r["history"] else None
            if r["pending_confirmation"]:
                r["status"] = {"state": "pending", "detail": "candidate awaiting confirmation rerun", "since": r["last_seen"]}
            elif r["is_leader"]:
                r["status"] = {"state": "leader", "detail": "confirmed frontier leader", "since": r["last_seen"]}
            elif last:
                r["status"] = {"state": last["status"] or "idle", "detail": "last round " + last["round_id"][:13],
                               "since": last["timestamp"]}
        ck = ck_of.get(hk)
        r["coldkey"] = ck
        acct = accounts.get(ck) if ck else None
        if acct:
            r["credits"] = acct.get("credits")
            r["carry_tao"] = acct.get("carry_tao", 0.0)
            r["deposited_tao"] = acct.get("deposited_tao", 0.0)
            r["spent"] = acct.get("debited", 0)
            r["refunded"] = acct.get("refunded", 0)
            r["coldkey_hotkeys"] = [h for h in acct.get("hotkeys") or [] if h != hk]
        row = lb.get(hk)
        if row:
            r["family"] = r["family"] or row.get("family")
            r["code_hash_short"] = r["code_hash_short"] or row.get("code_hash_short")
            r["best_score"] = row.get("best_score")
            r["rank"] = row.get("rank")

    def sort_key(r: dict[str, Any]):
        return (r["best_loss"] if r["best_loss"] is not None else float("inf"), -r["last_seen"], r["hotkey"])
    return sorted(by_hk.values(), key=sort_key)


def build_dashboard_data(
    rounds_dir: Path,
    cloud_status_path: Path | None = None,
    *,
    version: str = SN125_VERSION_FALLBACK,
    validator_hotkey: str = "local-dev",
    live_sources: dict[str, Any] | None = None,
    ledger_path: Path | None = None,
    state_path: Path | None = None,
    audit_dir: Path | None = None,
    treasury_coldkey: str | None = None,
) -> dict[str, Any]:
    """Build the public dashboard JSON from round artifacts.

    ``ledger_path`` / ``state_path`` / ``audit_dir`` (optional) add the
    operations sections miners watch between rounds: ``payments`` (credit
    ledger), ``queue`` (deferred commits) and ``live_round`` (the round in
    progress). Absent inputs degrade to ``available: False`` / ``None``."""
    if treasury_coldkey is None:
        from .. import config as _config
        treasury_coldkey = getattr(_config, "TREASURY_COLDKEY", "") or ""
    round_pairs: list[tuple[Path, dict[str, Any]]] = []
    for path in _round_files(rounds_dir)[:240]:
        rd = _read_json(path, {})
        if isinstance(rd, dict):
            round_pairs.append((path, rd))
    round_pairs.sort(
        key=lambda pair: (
            _num(pair[1].get("timestamp"), 0.0),
            str(pair[1].get("round_id", pair[0].stem)),
        ),
        reverse=True,
    )

    summaries = [_round_summary(path, rd) for path, rd in round_pairs]
    leaderboard = _leaderboard(round_pairs)
    latest = _latest_round_public(round_pairs[0][1]) if round_pairs else None
    cloud = _read_json(cloud_status_path, {}) if cloud_status_path else {}
    # B200-only capacity-aware delay: surfaced at top-level so the dashboard can
    # render a prominent banner without digging into the raw cloud blob. None
    # when running normally; a dict while the round is PAUSED awaiting B200.
    capacity_delay = cloud.get("capacity_delay") if isinstance(cloud, dict) else None

    # Quartile audit checkpoints (25/50/75/100%) — the audit-replay replacement.
    # Top-level summary so the dashboard can show "audit artifacts captured" without
    # digging into per-submission curves; detail = latest round's best submission.
    latest_for_audit = summaries[0] if summaries else None
    audit = {
        "total_checkpoints": sum(r.get("audit_checkpoint_count", 0) for r in summaries),
        "latest_round_id": latest_for_audit["round_id"] if latest_for_audit else "",
        "latest_checkpoints": latest_for_audit["best_checkpoints"] if latest_for_audit else [],
        "quartiles_expected": [0.25, 0.5, 0.75, 1.0],
    }

    total_submissions = sum(r["miners"] for r in summaries)
    total_scored = sum(r["scored"] for r in summaries)
    total_dq = sum(r["dq"] for r in summaries)
    avg_burn = (
        sum(r["burned_weight"] for r in summaries) / len(summaries)
        if summaries else 0.0
    )
    latest_summary = summaries[0] if summaries else None
    latest_burn = latest_summary["burned_weight"] if latest_summary else 0.0
    miner_daily_tokens = EMISSIONS["miner_daily_tokens"]
    latest_paid_tokens = miner_daily_tokens * max(0.0, 1.0 - latest_burn)
    latest_burned_tokens = miner_daily_tokens * latest_burn
    payments = _public_payments(ledger_path, treasury_coldkey)
    queue = _public_queue(state_path)
    live_round = _public_live_round(audit_dir)
    total_paid_tokens = sum(r["paid_weight"] * miner_daily_tokens for r in summaries)
    total_burned_tokens = sum(r["burned_weight"] * miner_daily_tokens for r in summaries)

    data = {
        "schema": "refinery.dashboard.v1",
        "generated": int(time.time()),
        "version": version,
        "validator_hotkey": validator_hotkey,
        "project": PROJECT_INTENT,
        "audiences": AUDIENCE_NOTES,
        "network": {
            "subnet": 125,
            "status": "live-artifacts" if live_sources and live_sources.get("used_live_rounds") else ("pre-launch" if not summaries else "local-artifacts"),
            "dashboard_host": "dashboard.<project-domain>",
            "live_data_path": "/dashboard.json",
        },
        "live_sources": live_sources or {
            "enabled": False,
            "ok": True,
            "sources": [],
            "used_live_rounds": False,
            "used_live_cloud": False,
        },
        "emissions": {
            **EMISSIONS,
            "latest_paid_tokens": latest_paid_tokens,
            "latest_burned_tokens": latest_burned_tokens,
            "total_paid_tokens": total_paid_tokens,
            "total_burned_tokens": total_burned_tokens,
        },
        "metrics": {
            "rounds_published": len(summaries),
            "total_submissions": total_submissions,
            "total_scored": total_scored,
            "total_dq": total_dq,
            "active_miners": len({r["best_hotkey"] for r in summaries if r["best_hotkey"]}),
            "avg_burn_weight": avg_burn,
            "miner_daily_tokens": miner_daily_tokens,
            "latest_paid_tokens": latest_paid_tokens,
            "latest_burned_tokens": latest_burned_tokens,
            "total_paid_tokens": total_paid_tokens,
            "total_burned_tokens": total_burned_tokens,
            "latest_best_score": latest_summary["best_score"] if latest_summary else 0.0,
            "latest_round_id": latest_summary["round_id"] if latest_summary else "",
            "daily_spend": _num(cloud.get("daily_spend"), 0.0),
            "daily_limit": _num(cloud.get("daily_limit"), 0.0),
            "active_rentals": int(_num(cloud.get("active_rentals"), 0)),
            "max_concurrent": int(_num(cloud.get("max_concurrent"), 8)),
            "cost_per_sub": _num(cloud.get("cost_per_sub"), 0.0),
            "resource_sku": cloud.get("resource_sku", "b200-small"),
            # Ordered provider failover chain (rent from the first with B200
            # capacity; pause only when all are dry).
            "providers": list(cloud.get("providers") or []),
        },
        "capacity_delay": capacity_delay,
        "audit": audit,
        "payments": payments,
        "queue": queue,
        "live_round": live_round,
        "miners": _miners_index(round_pairs, summaries, leaderboard, payments, queue,
                                live_round, latest, miner_daily_tokens),
        "current_round": latest_summary,
        "leaderboard": leaderboard,
        "refiner": _refiner_overview(summaries),
        "history": summaries,
        "cloud": cloud,
        "verification": _verification(summaries),
        "research": {
            "claims": [
                "Refinery is a decentralized incentive market for optimizer update rules, aimed at discoveries likely to generalize toward large LLM training runs.",
                "Optimizer rankings are horizon-dependent; final fully-decayed held-out loss is the launch metric.",
                "The rolling best baseline prevents paying miners for merely exploiting stale AdamW tuning.",
                "Fixed wall-clock budget makes optimizer speed part of quality.",
                "Clean scoring and replay artifacts make validator decisions independently inspectable.",
            ],
            "references": [
                "README.md",
                "docs/INCENTIVE_MECHANISM.md",
                "docs/VERIFICATION.md",
                "docs/REFINER_TASK.md",
            ],
        },
        "latest": latest,
    }
    return data


def build_dashboard_json(
    rounds_dir: Path,
    cloud_status_path: Path | None = None,
    *,
    version: str = SN125_VERSION_FALLBACK,
    validator_hotkey: str = "local-dev",
    live_sources: dict[str, Any] | None = None,
    **operations: Any,
) -> str:
    return json.dumps(
        build_dashboard_data(
            rounds_dir,
            cloud_status_path,
            version=version,
            validator_hotkey=validator_hotkey,
            live_sources=live_sources,
            **operations,
        ),
        default=str,
    )
