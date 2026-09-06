"""Structured JSONL audit logging for live validator rounds.

The normal Python logger is for operators reading text. This file is for durable
round forensics and future dashboard ingestion: one JSON object per event, one
file per round, with stable event names and enough identifiers to join commits,
reveals, cloud runs, scores, weights, and public attestations.
"""
from __future__ import annotations

import hashlib
import ast
import json
import re
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA = "refinery.round_audit.v1"
MAX_AUDIT_STRING = 100_000
MAX_AUDIT_ITEMS = 200
REDACT_KEYS = {
    "authorization",
    "api_key",
    "access_key",
    "secret_key",
    "targon_api_key",
    "token",
    "secret",
    "password",
    "private_key",
    "bearer",
}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sn4_[A-Za-z0-9_]{12,}"),
)


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def payload_record(payload: bytes, *, include_source: bool = False) -> dict[str, Any]:
    """Return stable source identifiers, optionally with the decoded code."""
    source = payload.decode("utf-8", errors="replace")
    out: dict[str, Any] = {
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "source_lines": source.count("\n") + (1 if source else 0),
        "source_meta": source_metadata(source),
    }
    if include_source:
        out["source_code"] = source
    return out


def source_metadata(source: str) -> dict[str, Any]:
    """Cheap dashboard-friendly source features, no execution."""
    meta: dict[str, Any] = {
        "loc": len([ln for ln in source.splitlines() if ln.strip()]),
        "imports": [],
        "defs": [],
        "classes": [],
        "uses_triton": "triton" in source.lower(),
        "uses_native_cuda": any(x in source.lower() for x in (".cu", "cuda", "cupy")),
        "uses_torch_compile": "torch.compile" in source,
        "uses_foreach": "foreach" in source.lower(),
        "uses_fused": "fused" in source.lower(),
        "has_build_optimizer": "def build_optimizer" in source,
        "optimizer_family": "custom",
        "hparams": {},
        "parse_error": "",
    }
    text_l = source.lower()
    if "lion" in text_l:
        meta["optimizer_family"] = "lion-like"
    elif "muon" in text_l or "orthogonal" in text_l or "newtonschulz" in text_l:
        meta["optimizer_family"] = "orthogonalized"
    elif "adamw" in text_l or "adam" in text_l:
        meta["optimizer_family"] = "adam-family"
    elif "sgd" in text_l:
        meta["optimizer_family"] = "sgd-family"
    for key, pat in {
        "lr": r"\blr\s*=\s*([0-9.eE+-]+)",
        "weight_decay": r"\bweight_decay\s*=\s*([0-9.eE+-]+)",
        "betas": r"\bbetas\s*=\s*(\([^)]+\))",
        "eps": r"\beps\s*=\s*([0-9.eE+-]+)",
    }.items():
        m = re.search(pat, source)
        if m:
            meta["hparams"][key] = m.group(1)
    try:
        tree = ast.parse(source)
        imports = []
        defs = []
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        meta["imports"] = sorted(set(x for x in imports if x))[:64]
        meta["defs"] = sorted(set(defs))[:64]
        meta["classes"] = sorted(set(classes))[:64]
    except SyntaxError as e:
        meta["parse_error"] = str(e)
    return meta


def looks_like_crash(reason: str) -> bool:
    text = str(reason or "").lower()
    return any(marker in text for marker in (
        "rc=",
        "sigsegv",
        "segmentation fault",
        "traceback",
        "exception",
        "crash",
        "cuda error",
        "device-side assert",
        "illegal memory access",
        "killed",
    ))


def sanitize_for_audit(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Bound recursive audit payloads and redact obvious credential fields."""
    lk = key.lower()
    if lk in REDACT_KEYS or lk.endswith("_token") or lk.endswith("_secret"):
        return "[REDACTED]"
    if depth > 6:
        return repr(value)[:MAX_AUDIT_STRING]
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= MAX_AUDIT_ITEMS:
                out["_truncated_items"] = len(value) - MAX_AUDIT_ITEMS
                break
            out[str(k)] = sanitize_for_audit(v, key=str(k), depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [sanitize_for_audit(v, depth=depth + 1) for v in seq[:MAX_AUDIT_ITEMS]]
        if len(seq) > MAX_AUDIT_ITEMS:
            out.append({"_truncated_items": len(seq) - MAX_AUDIT_ITEMS})
        return out
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[:MAX_AUDIT_STRING]
    if isinstance(value, str):
        text = value
        for pat in SECRET_PATTERNS:
            text = pat.sub("[REDACTED]", text)
        if len(text) > MAX_AUDIT_STRING:
            return text[:MAX_AUDIT_STRING] + f"...[truncated {len(text) - MAX_AUDIT_STRING} chars]"
        return text
    return value


class JsonlAuditLog:
    def __init__(self, path: str | Path, *, round_id: str,
                 context: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.text_path = self.path.with_suffix(".log")
        self.aggregate_path = self.path.parent / "all_rounds.jsonl"
        self.aggregate_text_path = self.path.parent / "all_rounds.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.round_id = round_id
        self.context = dict(context or {})
        self._lock = threading.Lock()
        self._seq = 0
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        self._text_fh = self.text_path.open("a", encoding="utf-8", buffering=1)
        self._agg_fh = self.aggregate_path.open("a", encoding="utf-8", buffering=1)
        self._agg_text_fh = self.aggregate_text_path.open("a", encoding="utf-8", buffering=1)
        self.emit("audit.opened", path=str(self.path))

    def emit(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._seq += 1
            rec = {
                "schema": SCHEMA,
                "seq": self._seq,
                "ts": utc_ts(),
                "ts_unix": time.time(),
                "event": event,
                "round_id": self.round_id,
            }
            rec.update(sanitize_for_audit(self.context))
            rec.update(sanitize_for_audit(fields))
            line_json = json.dumps(rec, sort_keys=True, default=str) + "\n"
            self._fh.write(line_json)
            self._agg_fh.write(line_json)
            msg = fields.get("message") or fields.get("line") or fields.get("reason") or ""
            ident = fields.get("commit_hash") or fields.get("wrk_uid") or fields.get("hotkey") or ""
            ident_s = f" {str(ident)[:18]}" if ident else ""
            msg_s = f" {str(msg).replace(chr(10), ' ')[:4000]}" if msg else ""
            line_text = f"{rec['ts']} #{self._seq:06d} {event}{ident_s}{msg_s}\n"
            self._text_fh.write(line_text)
            self._agg_text_fh.write(line_text)

    def close(self) -> None:
        try:
            self.emit("audit.closed", path=str(self.path))
        finally:
            self._fh.close()
            self._text_fh.close()
            self._agg_fh.close()
            self._agg_text_fh.close()


def audit_emit(audit: Any, event: str, **fields: Any) -> None:
    """Best-effort emit helper used by hot protocol paths."""
    if audit is None:
        return
    try:
        audit.emit(event, **fields)
    except Exception:
        pass
