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
    LAUNCH_FIRST_CYCLE_DAYS,
)


SN125_VERSION_FALLBACK = "0.1.0"

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
            "uri": "" if _is_local_artifact_ref(uri) else uri,
        })
    out.sort(key=lambda r: (r["pct"], r["step"]))
    return out


def _round_files(rounds_dir: Path) -> list[Path]:
    if not rounds_dir.exists():
        return []
    return sorted(rounds_dir.glob("*.json"), reverse=True)


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
        for curve in (best_sub.get("curves", {}) or {}).values():
            if isinstance(curve, dict) and curve.get("checkpoints"):
                best_checkpoints = _normalize_dash_checkpoints(curve["checkpoints"])
                break

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
        break
    if base_loss is None and base_losses:
        base_loss = next(iter(base_losses.values()))
    if best_loss is None and best_checkpoints:
        best_loss = best_checkpoints[-1]["eval_loss"] or None
    improvement = None
    if best_loss is not None and base_loss:
        improvement = max(0.0, (base_loss - best_loss) / base_loss)

    families: dict[str, int] = {}
    for sub in submissions.values():
        score = _as_score(sub)
        if "final_score" not in score:
            continue
        fam = _optimizer_family(sub, score)
        families[fam] = families.get(fam, 0) + 1

    providers: dict[str, int] = {}
    for sub in submissions.values():
        prov = ((sub.get("provenance") or {}).get("provider")
                if isinstance(sub, dict) else "")
        if prov:
            providers[prov] = providers.get(prov, 0) + 1
    best_provenance = (best_sub.get("provenance") or {}) if isinstance(best_sub, dict) else {}

    fr = rd.get("frontier_rewards") or {}
    new_event = fr.get("new_event") if isinstance(fr.get("new_event"), dict) else None
    frontier = {
        "improvement_unit": fr.get("improvement_unit", "nats"),
        "min_improvement_threshold": _num(fr.get("min_improvement_threshold"), 0.0),
        "confirmation_runs": int(_num(fr.get("confirmation_runs"), 0)),
        "leader_hotkey": fr.get("leader_hotkey", ""),
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
            weight = _num(sub.get("weight", 0.0))
            row["rounds"] += 1
            if hotkey == best_hotkey:
                row["wins"] += 1
            row["ema_score"] = 0.65 * row["ema_score"] + 0.35 * final if row["rounds"] > 1 else final
            row["weight"] += weight
            row["earned_tokens"] += weight * EMISSIONS["miner_daily_tokens"]
            comps = score.get("components", {}) or {}
            for key in row["components"]:
                row["components"][key] += _num(comps.get(key), 0.0)
            if final > row["best_score"]:
                row["best_score"] = final
                row["code_hash"] = sub.get("code_hash", "")
            family = _optimizer_family(sub, score)
            row["families"][family] = row["families"].get(family, 0) + 1

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


def build_dashboard_data(
    rounds_dir: Path,
    cloud_status_path: Path | None = None,
    *,
    version: str = SN125_VERSION_FALLBACK,
    validator_hotkey: str = "local-dev",
    live_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public dashboard JSON from round artifacts."""
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
    capacity_delay = cloud.get("capacity_delay") if isinstance(cloud, dict) else None

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
            "providers": list(cloud.get("providers") or []),
        },
        "capacity_delay": capacity_delay,
        "audit": audit,
        "current_round": latest_summary,
        "leaderboard": leaderboard,
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
                "docs/DESIGN.md",
                "docs/SIMPLE_MECHANISM.md",
                "docs/paper/Refinery_Subnet.md",
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
) -> str:
    return json.dumps(
        build_dashboard_data(
            rounds_dir,
            cloud_status_path,
            version=version,
            validator_hotkey=validator_hotkey,
            live_sources=live_sources,
        ),
        default=str,
    )
