"""Paper-1 provenance: build the Refinery progression record from round JSONs.

The first Refinery publication documents the *progression over time* — what the
miner network tried, what beat the baseline, when the frontier moved, and what
was burned versus paid. All of that already lives in the per-round JSONs the
validator writes (and the publisher mirrors to R2/HF). This module distills
those files into two figure-ready artifacts, with zero operator steps:

- a machine-readable ``progression.json`` (per-round series + frontier
  timeline + per-code-hash provenance), the data behind every paper figure;
- a human-readable Markdown summary for the repo/appendix.

Deliberately tolerant: round files from crashed/partial rounds, foreign JSON
in the rounds dir, and schema drift all degrade to skipped fields — the tool
must always produce a report from whatever survived.

Usage:
    python -m sn125.progression [--rounds-dir DIR] [--out progression.json]
                                [--markdown PROGRESSION.md]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def default_rounds_dir() -> Path:
    return Path(__file__).resolve().parent / "rounds"


def load_rounds(rounds_dir: str | Path | None = None) -> list[dict]:
    """All parseable round JSONs, sorted by (timestamp, round_id)."""
    rdir = Path(rounds_dir) if rounds_dir else default_rounds_dir()
    if not rdir.is_dir():
        return []
    rounds = []
    for fp in sorted(rdir.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("round_id") and "submissions" in d:
            rounds.append(d)
    rounds.sort(key=lambda d: (int(d.get("timestamp", 0) or 0), str(d.get("round_id", ""))))
    return rounds


def _best_submission(rd: dict) -> tuple[str, dict] | None:
    """(hotkey, submission entry) with the highest final_score, or None."""
    best = None
    for hk, entry in (rd.get("submissions") or {}).items():
        try:
            score = float((entry.get("score") or {}).get("final_score", -1.0))
        except (TypeError, ValueError):
            continue
        if best is None or score > best[0]:
            best = (score, hk, entry)
    if best is None:
        return None
    return best[1], best[2]


def _final_losses(entry_or_curves: dict) -> dict[str, float]:
    """task_id -> submission final held-out loss, from a ScoreRecord's task_scores."""
    out = {}
    for tid, d in ((entry_or_curves.get("score") or {}).get("task_scores") or {}).items():
        loss = (d or {}).get("sub_final_loss")
        if isinstance(loss, (int, float)):
            out[tid] = float(loss)
    return out


def _baseline_losses(rd: dict) -> dict[str, float]:
    """task_id -> baseline final held-out loss (last eval point of each baseline curve)."""
    out = {}
    for tid, curve in (rd.get("baselines") or {}).items():
        pts = (curve or {}).get("eval_points") or []
        if pts and isinstance(pts[-1], (list, tuple)) and len(pts[-1]) >= 2:
            try:
                out[tid] = float(pts[-1][1])
            except (TypeError, ValueError):
                pass
    return out


def round_summary(rd: dict) -> dict:
    """One figure-ready row per round. Missing data degrades to None/empty."""
    submissions = rd.get("submissions") or {}
    n_failed = 0
    for entry in submissions.values():
        score = (entry.get("score") or {})
        try:
            if float(score.get("final_score", 0.0)) <= -0.99 or score.get("failed_tasks"):
                n_failed += 1
        except (TypeError, ValueError):
            n_failed += 1

    econ = rd.get("economics") or {}
    fr = rd.get("frontier_rewards") or {}
    row = {
        "round_id": rd.get("round_id", ""),
        "timestamp": int(rd.get("timestamp", 0) or 0),
        "n_submissions": len(submissions),
        "n_failed": n_failed,
        "baseline_final_loss": _baseline_losses(rd),
        "burned_weight": econ.get("burned_weight"),
        "paid_weight": econ.get("paid_weight"),
        "leader_hotkey": fr.get("leader_hotkey", ""),
        "min_improvement_threshold": fr.get("min_improvement_threshold"),
        "new_frontier_event": None,
        "best": None,
    }
    best = _best_submission(rd)
    if best is not None:
        hk, entry = best
        score = entry.get("score") or {}
        row["best"] = {
            "hotkey": hk,
            "code_hash": entry.get("code_hash", ""),
            "final_score": score.get("final_score"),
            "final_loss": _final_losses(entry),
            "weight": entry.get("weight"),
        }
    ev = fr.get("new_event")
    if isinstance(ev, dict):
        row["new_frontier_event"] = {
            k: ev.get(k) for k in
            ("event_id", "hotkey", "code_hash", "improvement",
             "final_score", "timestamp", "task_final_losses")
        }
    return row


def build_progression(rounds: list[dict]) -> dict:
    """The full paper-1 record: per-round series, frontier timeline, provenance."""
    series = [round_summary(rd) for rd in rounds]

    frontier = [r["new_frontier_event"] for r in series if r["new_frontier_event"]]

    provenance: dict[str, dict] = {}
    for rd, row in zip(rounds, series):
        for hk, entry in (rd.get("submissions") or {}).items():
            ch = entry.get("code_hash", "")
            if not ch:
                continue
            rec = provenance.setdefault(ch, {
                "code_hash": ch, "hotkeys": [], "first_round": row["round_id"],
                "last_round": row["round_id"], "rounds_seen": 0,
                "best_final_score": None,
            })
            if hk not in rec["hotkeys"]:
                rec["hotkeys"].append(hk)
            rec["last_round"] = row["round_id"]
            rec["rounds_seen"] += 1
            try:
                fs = float((entry.get("score") or {}).get("final_score"))
                if rec["best_final_score"] is None or fs > rec["best_final_score"]:
                    rec["best_final_score"] = fs
            except (TypeError, ValueError):
                pass

    trajectory: dict[str, list] = {}
    running_best: dict[str, float] = {}
    for row in series:
        for tid, base in (row["baseline_final_loss"] or {}).items():
            best_loss = ((row["best"] or {}).get("final_loss") or {}).get(tid)
            if isinstance(best_loss, (int, float)):
                prev = running_best.get(tid)
                running_best[tid] = best_loss if prev is None else min(prev, best_loss)
            trajectory.setdefault(tid, []).append({
                "round_id": row["round_id"], "timestamp": row["timestamp"],
                "baseline_loss": base, "best_loss": best_loss,
                "running_best_loss": running_best.get(tid),
            })

    return {
        "generated_at": int(time.time()),
        "n_rounds": len(series),
        "rounds": series,
        "frontier_events": frontier,
        "provenance_by_code_hash": provenance,
        "loss_trajectory": trajectory,
    }


def to_markdown(progression: dict) -> str:
    """Human-readable summary (repo/appendix)."""
    lines = ["# Refinery progression record", ""]
    lines.append(f"Rounds: **{progression['n_rounds']}** · "
                 f"Frontier events: **{len(progression['frontier_events'])}** · "
                 f"Distinct submissions: **{len(progression['provenance_by_code_hash'])}**")
    lines += ["", "## Rounds", "",
              "| round | subs | failed | best score | best loss | baseline loss | burned |",
              "|---|---|---|---|---|---|---|"]
    for r in progression["rounds"]:
        best = r["best"] or {}
        best_loss = next(iter((best.get("final_loss") or {}).values()), None)
        base_loss = next(iter((r["baseline_final_loss"] or {}).values()), None)

        def _f(x, spec=".4f"):
            return format(x, spec) if isinstance(x, (int, float)) else "—"
        lines.append(
            f"| {r['round_id']} | {r['n_submissions']} | {r['n_failed']} "
            f"| {_f(best.get('final_score'))} | {_f(best_loss)} "
            f"| {_f(base_loss)} | {_f(r['burned_weight'], '.2f')} |")
    if progression["frontier_events"]:
        lines += ["", "## Frontier events", ""]
        for ev in progression["frontier_events"]:
            imp = ev.get("improvement")
            imp_s = format(imp, ".4%") if isinstance(imp, (int, float)) else "?"
            lines.append(f"- `{ev.get('event_id', '')}` — hotkey `{ev.get('hotkey', '')[:16]}…`, "
                         f"code `{(ev.get('code_hash') or '')[:12]}…`, improvement {imp_s}")
    lines += ["", "## Authorship", "",
              "Optimizers are credited to their submitting hotkeys (above). Authors can "
              "prove control of a hotkey offline with `python -m sn125.prove_authorship` "
              "(sign a claim binding the hotkey to the submission's code hash; anyone "
              "can verify it against this record)."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Refinery paper-1 progression record")
    ap.add_argument("--rounds-dir", default="", help="round JSON dir (default: sn125/rounds)")
    ap.add_argument("--out", default="", help="write progression JSON here")
    ap.add_argument("--markdown", default="", help="write Markdown summary here")
    args = ap.parse_args(argv)

    rounds = load_rounds(args.rounds_dir or None)
    prog = build_progression(rounds)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(prog, indent=2, sort_keys=True, default=str))
        print(f"wrote {args.out} ({prog['n_rounds']} rounds)")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(to_markdown(prog))
        print(f"wrote {args.markdown}")
    if not args.out and not args.markdown:
        print(json.dumps(prog, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
