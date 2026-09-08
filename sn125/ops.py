"""Miner-operations feed: credits, evaluation statuses, and the logbook index.

The raw archive the continuous publisher mirrors (round JSONs, audit JSONL,
the payment ledger, the loop-state checkpoint) is complete but not convenient:
answering "did my deposit land", "where is my submission", "why was it DQ'd"
means joining four files. This module derives ONE small, current view of that
state so an operator can answer any miner question from the private R2 mirror
alone, without shell access to the validator box:

    ops/ops.json          envelope: everything below plus validator identity
    ops/credits.json      per-coldkey credits, carried rao, grants/debits/refunds
    ops/evaluations.json  deferred queue, live round progress, recent outcomes
    ops/logbook.json      index of every log file in the archive + recent tail

It is rebuilt by the publisher on every cycle (60 s default) from the durable
files only, so it never touches validator objects and never holds a lock the
round loop needs. Every section degrades independently: a missing or torn
file yields ``{"available": false, "error": ...}`` for that section instead
of an empty feed. Payload bytes (miner source) are never copied into the feed
— the round JSONs and audit logs already carry them in the private archive.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .payments.store import LEDGER_VERSION, atomic_write_json

log = logging.getLogger("sn125.ops")

OPS_SCHEMA = "refinery.ops-snapshot.v1"
OPS_DIRNAME = "ops"
RAO_PER_TAO = 10**9

DEFAULT_RECENT_ROUNDS = 20
DEFAULT_RECENT_EVENTS = 500
DEFAULT_LOGBOOK_TAIL = 200
MAX_AUDIT_PARSE_BYTES = 64 * 1024 * 1024

_LIVE_STATUS_BY_EVENT = {
    "commit.carryover_readmitted": "readmitted",
    "commit.accepted": "committed",
    "commit.rejected": "rejected",
    "commit.carryover_rejected": "rejected",
    "submission.revealed": "revealed",
    "submission.reveal_rejected": "reveal_rejected",
    "submission.unrevealed": "unrevealed",
    "gate.dq": "gate_dq",
    "gate.passed": "gate_passed",
    "evaluation.run_started": "running",
    "evaluation.flake": "relaunching",
    "evaluation.finalized": None,
}
_LIVE_TERMINAL_EVENTS = ("round.finish", "round.exception", "audit.closed")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(section: str, exc: Exception, **extra: Any) -> dict[str, Any]:
    log.warning("ops feed: %s unavailable: %s: %s", section, type(exc).__name__, exc)
    out = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    out.update(extra)
    return out


def _tao(rao: int | None) -> float | None:
    return None if rao is None else round(int(rao) / RAO_PER_TAO, 9)




def build_credits(ledger_path: Path | None, *, recent_events: int = DEFAULT_RECENT_EVENTS
                  ) -> dict[str, Any]:
    """Per-coldkey view of the durable payment ledger (payments/store.py).

    Balances and carried remainders are copied from the ledger's own totals;
    the grants/debits/refunds breakdown is replayed from its events so an
    operator can see *why* a balance is what it is.
    """
    if ledger_path is None or not Path(ledger_path).exists():
        return {"available": False, "source": str(ledger_path or ""),
                "reason": "no ledger written yet", "coldkeys": {}, "totals": {}}
    ledger_path = Path(ledger_path)
    try:
        data = _read_json(ledger_path)
        if not isinstance(data, dict) or data.get("version") != LEDGER_VERSION:
            raise ValueError(f"unsupported ledger version {data.get('version')!r}")
        events = list(data.get("events") or [])
        balances = {str(k): int(v) for k, v in (data.get("balances") or {}).items()}
        carry = {str(k): int(v) for k, v in (data.get("carry_rao") or {}).items()}
    except Exception as exc:
        return _fail("credits", exc, source=str(ledger_path), coldkeys={}, totals={})

    coldkeys: dict[str, dict[str, Any]] = {}

    def row(ck: str) -> dict[str, Any]:
        return coldkeys.setdefault(ck, {
            "credits": 0, "carry_rao": 0, "carry_tao": 0.0,
            "deposits": 0, "deposited_rao": 0, "deposited_tao": 0.0,
            "granted": 0, "debited": 0, "refunded": 0,
            "last_seq": None, "last_round_id": None, "hotkeys": [],
        })

    for ev in events:
        if not isinstance(ev, dict):
            continue
        ck = ev.get("coldkey")
        if not ck:
            continue
        r = row(str(ck))
        kind = ev.get("kind")
        credits = int(ev.get("credits") or 0)
        if kind == "grant":
            r["deposits"] += 1
            r["deposited_rao"] += int(ev.get("rao") or 0)
            r["granted"] += credits
        elif kind == "debit":
            r["debited"] += -credits
            hk = ev.get("hotkey")
            if hk and hk not in r["hotkeys"]:
                r["hotkeys"].append(str(hk))
        elif kind == "refund":
            r["refunded"] += credits
        r["last_seq"] = ev.get("seq")
        if ev.get("round_id"):
            r["last_round_id"] = ev.get("round_id")

    for ck, bal in balances.items():
        row(ck)["credits"] = bal
    for ck, rao in carry.items():
        r = row(ck)
        r["carry_rao"] = rao
        r["carry_tao"] = _tao(rao)
    for r in coldkeys.values():
        r["deposited_tao"] = _tao(r["deposited_rao"])

    fee_rao = data.get("fee_rao")
    totals = {
        "coldkeys": len(coldkeys),
        "coldkeys_with_credit": sum(1 for r in coldkeys.values() if r["credits"] > 0),
        "credits_outstanding": sum(r["credits"] for r in coldkeys.values()),
        "carry_rao": sum(r["carry_rao"] for r in coldkeys.values()),
        "deposits": sum(r["deposits"] for r in coldkeys.values()),
        "deposited_rao": sum(r["deposited_rao"] for r in coldkeys.values()),
        "granted": sum(r["granted"] for r in coldkeys.values()),
        "debited": sum(r["debited"] for r in coldkeys.values()),
        "refunded": sum(r["refunded"] for r in coldkeys.values()),
    }
    totals["deposited_tao"] = _tao(totals["deposited_rao"])
    return {
        "available": True,
        "source": str(ledger_path),
        "ledger_updated_at": data.get("updated_at"),
        "fee_rao": fee_rao,
        "fee_tao": _tao(fee_rao) if fee_rao is not None else None,
        "last_block": data.get("last_block"),
        "scan_cursor": data.get("scan_cursor"),
        "pinned_rounds": dict(data.get("pinned_rounds") or {}),
        "coldkeys": dict(sorted(coldkeys.items())),
        "totals": totals,
        "events_total": len(events),
        "recent_events": events[-int(recent_events):] if recent_events else [],
    }




def _queue_from_state(state_path: Path | None) -> dict[str, Any]:
    """Deferred queue + counters from the loop-state checkpoint (roundsm/state.py).

    Payloads are deliberately dropped: the feed describes queue position, not
    the code waiting in it.
    """
    if state_path is None or not Path(state_path).exists():
        return {"available": False, "source": str(state_path or ""),
                "reason": "no state checkpoint written yet",
                "round_num": None, "last_round_id": None, "deferred_queue": []}
    state_path = Path(state_path)
    try:
        data = _read_json(state_path)
        if not isinstance(data, dict):
            raise ValueError("state checkpoint is not a JSON object")
        queue = []
        for pos, s in enumerate(data.get("carryover") or [], start=1):
            payload = s.get("payload_b64") or ""
            queue.append({
                "position": pos,
                "hotkey": s.get("hotkey"),
                "coldkey": s.get("coldkey"),
                "commit_hash": s.get("commit_hash"),
                "deferrals": int(s.get("deferrals") or 0),
                "accepted_at": s.get("accepted_at"),
                "accept_index": s.get("accept_index"),
                "payload_bytes": (len(payload) * 3) // 4 if payload else 0,
                "forensics": list(s.get("forensics") or []),
                "status": "deferred",
            })
        return {
            "available": True,
            "source": str(state_path),
            "state_saved_at": data.get("saved_at"),
            "round_num": data.get("round_num"),
            "last_round_id": data.get("last_round_id"),
            "deferred_queue": queue,
            "fairness_window": [list(r) for r in (data.get("recent_rounds") or [])],
            "last_weights": dict(data.get("last_weights") or {}),
        }
    except Exception as exc:
        return _fail("queue", exc, source=str(state_path),
                     round_num=None, last_round_id=None, deferred_queue=[])


def _iter_jsonl(path: Path, *, max_bytes: int = MAX_AUDIT_PARSE_BYTES):
    """Parsed records of a JSONL file; a torn last line (live append) is skipped."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            fh.readline()
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if isinstance(rec, dict):
                yield rec


def _latest_round_audit(audit_dir: Path) -> Path | None:
    candidates = [p for p in audit_dir.glob("*.jsonl")
                  if p.is_file() and p.name != "all_rounds.jsonl"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def _live_round(audit_dir: Path | None) -> dict[str, Any] | None:
    """Progress of the round currently being run, from its audit JSONL.

    Returns ``None`` when the newest round audit has already finished (the
    round JSON is then the record) or when there is no audit yet.
    """
    if audit_dir is None or not Path(audit_dir).is_dir():
        return None
    path = _latest_round_audit(Path(audit_dir))
    if path is None:
        return None
    subs: dict[str, dict[str, Any]] = {}
    round_id = None
    phase = None
    started = None
    last_ts = None
    finished = False
    n = 0
    credit_rows: list[dict[str, Any]] = []
    for rec in _iter_jsonl(path):
        n += 1
        event = str(rec.get("event") or "")
        round_id = rec.get("round_id") or round_id
        last_ts = rec.get("ts") or last_ts
        if event == "round.start":
            started = rec.get("ts")
        if event in _LIVE_TERMINAL_EVENTS:
            finished = True
        if event.endswith((".opened", ".closed", ".complete")) or event.startswith(
                ("round.", "capacity.", "evaluation.attempt", "evaluation.run")):
            phase = event
        if event == "payments.credit_snapshot":
            credit_rows = list(rec.get("credit_snapshot") or [])
        h = rec.get("commit_hash")
        if not h or event not in _LIVE_STATUS_BY_EVENT:
            continue
        status = _LIVE_STATUS_BY_EVENT[event] or rec.get("status") or "finalized"
        row = subs.setdefault(str(h), {"commit_hash": h, "hotkey": None, "coldkey": None})
        row.update({
            "hotkey": rec.get("hotkey") or row.get("hotkey"),
            "coldkey": rec.get("coldkey") or row.get("coldkey"),
            "status": status,
            "last_event": event,
            "last_event_ts": rec.get("ts"),
        })
        for key in ("reason", "score", "outcome", "deferrals", "selection_rank",
                    "accept_index", "accepted_at", "crashed"):
            if key in rec:
                row[key] = rec[key]
    if finished:
        return None
    balances = {r.get("commit_hash"): r.get("credit_balance") for r in credit_rows}
    for h, row in subs.items():
        if h in balances:
            row["credit_balance"] = balances[h]
    return {
        "round_id": round_id,
        "audit_file": path.name,
        "started_at": started,
        "last_event_at": last_ts,
        "phase": phase,
        "events": n,
        "submissions": sorted(subs.values(),
                              key=lambda r: (str(r.get("hotkey") or ""), str(r["commit_hash"]))),
    }


_FRONTIER_KEYS = ("leader_hotkey", "payable_share", "burned_share", "progress_nats",
                  "progress_payable_share", "pending_confirmation")


def _frontier_summary(fr: Any) -> dict[str, Any]:
    """The payout-relevant part of a round's frontier record; the full event
    history stays in the round JSON."""
    if not isinstance(fr, dict):
        return {}
    out = {k: fr[k] for k in _FRONTIER_KEYS if k in fr}
    outcomes = fr.get("confirmation_outcomes")
    if isinstance(outcomes, list):
        out["confirmation_outcomes"] = [
            {k: o.get(k) for k in ("hotkey", "outcome", "improvement", "submission_round_id")}
            for o in outcomes if isinstance(o, dict)]
    return out


def _recent_round_records(rounds_dir: Path | None, *, limit: int) -> list[dict[str, Any]]:
    """Miner-facing summary of the newest ``limit`` round JSONs (no sources)."""
    if rounds_dir is None or not Path(rounds_dir).is_dir():
        return []
    files = [p for p in Path(rounds_dir).glob("*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            rec = _read_json(path)
            if not isinstance(rec, dict) or "round_id" not in rec:
                continue
        except Exception as exc:
            out.append({"round_id": path.stem, "file": path.name,
                        "available": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        report = rec.get("fsm_report") or {}
        weights = rec.get("weights") or {}
        scored = rec.get("submissions") or {}
        rows = []
        for s in report.get("submissions") or []:
            if not isinstance(s, dict):
                continue
            hk = s.get("hotkey")
            entry = scored.get(hk) or {} if isinstance(scored, dict) else {}
            score = entry.get("score") if isinstance(entry, dict) else None
            rows.append({
                "hotkey": hk,
                "commit_hash": s.get("commit_hash"),
                "status": s.get("status"),
                "score": s.get("score"),
                "weight": weights.get(hk, 0.0) if isinstance(weights, dict) else None,
                "deferrals": s.get("deferrals"),
                "selected_for_eval": s.get("selected_for_eval"),
                "selection_rank": s.get("selection_rank"),
                "relaunched": s.get("relaunched"),
                "forensics": list(s.get("forensics") or []),
                "score_record": score if isinstance(score, dict) else None,
            })
        if not rows and isinstance(scored, dict):
            for hk, entry in scored.items():
                rows.append({
                    "hotkey": hk, "commit_hash": (entry or {}).get("code_hash"),
                    "status": "scored", "score": None,
                    "weight": weights.get(hk, 0.0) if isinstance(weights, dict) else None,
                    "score_record": (entry or {}).get("score"),
                })
        out.append({
            "round_id": rec.get("round_id"),
            "file": path.name,
            "timestamp": rec.get("timestamp"),
            "fee_rao": (report.get("config") or {}).get("fee_rao"),
            "pauses": list(report.get("pauses") or rec.get("pause_reasons") or []),
            "selected": list(report.get("selected") or []),
            "deferred": list(report.get("deferred") or rec.get("deferred") or []),
            "commit_rejections": list(report.get("commit_rejections") or []),
            "burned_weight": (rec.get("economics") or {}).get("burned_weight"),
            "frontier": _frontier_summary(rec.get("frontier_rewards")),
            "submissions": rows,
        })
    return out


def build_evaluations(*, rounds_dir: Path | None, audit_dir: Path | None,
                      state_path: Path | None,
                      recent_rounds: int = DEFAULT_RECENT_ROUNDS) -> dict[str, Any]:
    """Where every miner's work is: queued, in the live round, or decided."""
    queue = _queue_from_state(state_path)
    try:
        live = _live_round(audit_dir)
    except Exception as exc:
        live = _fail("live_round", exc)
    try:
        rounds = _recent_round_records(rounds_dir, limit=recent_rounds)
    except Exception as exc:
        rounds = [_fail("rounds", exc)]

    by_hotkey: dict[str, dict[str, Any]] = {}
    for rnd in reversed(rounds):
        for s in rnd.get("submissions") or []:
            hk = s.get("hotkey")
            if hk:
                by_hotkey[str(hk)] = {
                    "state": "decided", "round_id": rnd.get("round_id"),
                    "commit_hash": s.get("commit_hash"), "status": s.get("status"),
                    "score": s.get("score"), "weight": s.get("weight"),
                }
        for rej in rnd.get("commit_rejections") or []:
            hk = rej.get("hotkey") if isinstance(rej, dict) else None
            if hk:
                by_hotkey[str(hk)] = {
                    "state": "decided", "round_id": rnd.get("round_id"),
                    "commit_hash": rej.get("commit_hash"), "status": "rejected",
                    "reason": rej.get("reason"),
                }
    if isinstance(live, dict) and live.get("submissions"):
        for s in live["submissions"]:
            hk = s.get("hotkey")
            if hk:
                by_hotkey[str(hk)] = {
                    "state": "live", "round_id": live.get("round_id"),
                    "commit_hash": s.get("commit_hash"), "status": s.get("status"),
                    "last_event": s.get("last_event"), "reason": s.get("reason"),
                    "score": s.get("score"),
                }
    for q in queue.get("deferred_queue") or []:
        hk = q.get("hotkey")
        if hk:
            by_hotkey[str(hk)] = {
                "state": "queued", "position": q["position"],
                "commit_hash": q.get("commit_hash"), "status": "deferred",
                "deferrals": q.get("deferrals"),
            }

    return {
        "round_num": queue.get("round_num"),
        "last_round_id": queue.get("last_round_id"),
        "queue": queue,
        "live_round": live,
        "rounds": rounds,
        "by_hotkey": dict(sorted(by_hotkey.items())),
    }




def _tail_lines(path: Path, n: int, *, chunk: int = 64 * 1024) -> list[str]:
    """Last ``n`` complete lines of a text file without reading all of it."""
    if n <= 0:
        return []
    size = path.stat().st_size
    if size == 0:
        return []
    buf = b""
    with path.open("rb") as fh:
        pos = size
        while pos > 0 and buf.count(b"\n") <= n:
            step = min(chunk, pos)
            pos -= step
            fh.seek(pos)
            buf = fh.read(step) + buf
    lines = buf.decode("utf-8", errors="replace").splitlines()
    if size > len(buf) and lines:
        lines = lines[1:]
    return lines[-n:]


def build_logbook(audit_dir: Path | None, *, tail: int = DEFAULT_LOGBOOK_TAIL,
                  publish_prefix: str = "audit/") -> dict[str, Any]:
    """Index of every log in the audit dir (what the R2 mirror holds) + a
    recent tail of the human-readable all-rounds log."""
    if audit_dir is None or not Path(audit_dir).is_dir():
        return {"available": False, "source": str(audit_dir or ""), "files": [],
                "recent": []}
    audit_dir = Path(audit_dir)
    files = []
    try:
        for path in sorted(audit_dir.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            if not (name.endswith((".jsonl", ".log")) or ".log." in name):
                continue
            st = path.stat()
            files.append({
                "name": name,
                "key": f"{publish_prefix}{name}",
                "bytes": st.st_size,
                "modified": int(st.st_mtime),
                "kind": ("process-log" if name.startswith(("validator.log", "autoupdate.log"))
                         else "audit-jsonl" if name.endswith(".jsonl")
                         else "audit-text"),
            })
    except Exception as exc:
        return _fail("logbook", exc, source=str(audit_dir), files=[], recent=[])
    recent: list[str] = []
    recent_source = None
    for candidate in ("all_rounds.log", "validator.log"):
        p = audit_dir / candidate
        if p.is_file():
            try:
                recent = _tail_lines(p, tail)
                recent_source = candidate
                break
            except Exception as exc:
                log.warning("ops feed: could not tail %s: %s", p, exc)
    return {
        "available": True,
        "source": str(audit_dir),
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "recent_source": recent_source,
        "recent": recent,
    }




def build_ops_snapshot(*, rounds_dir: str | Path | None, audit_dir: str | Path | None,
                       ledger_path: str | Path | None = None,
                       state_path: str | Path | None = None,
                       validator_hotkey: str = "", version: str = "",
                       extra: dict[str, Any] | None = None,
                       recent_rounds: int = DEFAULT_RECENT_ROUNDS,
                       recent_events: int = DEFAULT_RECENT_EVENTS,
                       logbook_tail: int = DEFAULT_LOGBOOK_TAIL) -> dict[str, Any]:
    """Assemble the feed. Never raises: each section fails closed on its own."""
    rounds_dir = Path(rounds_dir) if rounds_dir else None
    audit_dir = Path(audit_dir) if audit_dir else None
    if ledger_path is None and audit_dir is not None:
        ledger_path = audit_dir / "payments" / "ledger.json"
    if state_path is None and audit_dir is not None:
        state_path = audit_dir / "validator_state.json"
    ledger_path = Path(ledger_path) if ledger_path else None
    state_path = Path(state_path) if state_path else None

    try:
        credits = build_credits(ledger_path, recent_events=recent_events)
    except Exception as exc:
        credits = _fail("credits", exc)
    try:
        evaluations = build_evaluations(rounds_dir=rounds_dir, audit_dir=audit_dir,
                                        state_path=state_path, recent_rounds=recent_rounds)
    except Exception as exc:
        evaluations = _fail("evaluations", exc)
    try:
        logbook = build_logbook(audit_dir, tail=logbook_tail)
    except Exception as exc:
        logbook = _fail("logbook", exc)

    now = int(time.time())
    validator = {
        "hotkey": validator_hotkey,
        "version": version,
        "pid": os.getpid(),
        "rounds_dir": str(rounds_dir or ""),
        "audit_dir": str(audit_dir or ""),
    }
    validator.update(extra or {})
    return {
        "schema": OPS_SCHEMA,
        "generated": now,
        "validator": validator,
        "credits": credits,
        "evaluations": evaluations,
        "logbook": logbook,
    }


def write_ops_snapshot(directory: str | Path, snapshot: dict[str, Any]) -> None:
    """Atomically write the envelope and its three sections as separate objects."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    common = {"schema": OPS_SCHEMA, "generated": snapshot.get("generated"),
              "validator": snapshot.get("validator")}
    for name, value in (
        ("credits.json", {**common, "credits": snapshot.get("credits")}),
        ("evaluations.json", {**common, "evaluations": snapshot.get("evaluations")}),
        ("logbook.json", {**common, "logbook": snapshot.get("logbook")}),
        ("ops.json", snapshot),
    ):
        atomic_write_json(directory / name, value)
