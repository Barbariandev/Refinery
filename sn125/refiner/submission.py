"""Refiner-task submission envelope: what a miner commits to and reveals.

The commit/reveal transport carries one UTF-8 string of at most 1 MB and the
commit hash is its SHA-256. A refiner submission is two things (source and an
optional weight file of up to 450 MB), so the string a refiner miner serves is
a small canonical JSON ENVELOPE:

    {"refinery_submission": 1, "task": "RF1_smol_360M",
     "source": "<refiner.py>",
     "weights": {"sha256": "<64 hex>", "bytes": N, "url": "https://..."} | null}

Committing to the envelope commits to the source and to the weight file's
hash and size. The URL is seen by nobody before the reveal; the scoring box
downloads the file before its network is cut and refuses it unless size and
SHA-256 match, so what is hosted there cannot change after the commit. Nothing
else in the round machinery needs to know: the envelope is the payload, the
payload hashes to the commit, and the published ``sources`` map keeps hashing
to its keys.

This module is import-light (no torch at load) because the round loop and the
miner both use it.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from .refproc_limits import REFINER_TASK_ID

ENVELOPE_KEY = "refinery_submission"
ENVELOPE_VERSION = 1
MAX_WEIGHTS_URL_CHARS = 2048
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# What a weights URL may contain: it is interpolated into a shell command on
# the scoring box (quoted, but keep the alphabet boring anyway).
_URL_SAFE = re.compile(r"^[A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-]+$")


class EnvelopeError(ValueError):
    """The payload claims to be a refiner envelope but is not a valid one."""


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_ref(path: str | Path, url: str) -> dict:
    """The envelope's ``weights`` entry for a local file hosted at ``url``."""
    p = Path(path)
    ref = {"sha256": file_sha256(p), "bytes": int(p.stat().st_size), "url": str(url)}
    _check_weights_ref(ref)
    return ref


def build_envelope(source: str, weights: dict | None = None) -> str:
    """Canonical envelope text (sorted keys, no whitespace): the commit preimage."""
    if weights is not None:
        _check_weights_ref(weights)
        weights = {k: weights[k] for k in ("sha256", "bytes", "url")}
    env = {ENVELOPE_KEY: ENVELOPE_VERSION, "task": REFINER_TASK_ID, "source": source, "weights": weights}
    return json.dumps(env, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _check_weights_ref(ref) -> None:
    from .refproc_limits import REFINER_MAX_WEIGHT_BYTES
    if not isinstance(ref, dict):
        raise EnvelopeError("weights must be an object or null")
    sha, n, url = ref.get("sha256"), ref.get("bytes"), ref.get("url")
    if not isinstance(sha, str) or not _HEX64.match(sha):
        raise EnvelopeError("weights.sha256 must be 64 lowercase hex characters")
    if not isinstance(n, int) or isinstance(n, bool) or not 0 < n <= REFINER_MAX_WEIGHT_BYTES:
        raise EnvelopeError(f"weights.bytes must be in 1..{REFINER_MAX_WEIGHT_BYTES}")
    if not isinstance(url, str) or len(url) > MAX_WEIGHTS_URL_CHARS or not _URL_SAFE.match(url):
        raise EnvelopeError("weights.url must be a plain URL of at most 2048 characters")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise EnvelopeError("weights.url must be https://")
    if parts.username or parts.password:
        raise EnvelopeError("weights.url must not carry credentials")


def looks_like_envelope(payload: bytes | str) -> bool:
    """Cheap test used to route a payload: a refiner envelope is a JSON object
    whose text starts with the envelope key (an optimizer is Python source)."""
    head = payload[:64]
    if isinstance(head, bytes):
        head = head.decode("utf-8", errors="replace")
    return head.lstrip().startswith('{"' + ENVELOPE_KEY + '"')


def parse_envelope(payload: bytes | str) -> dict | None:
    """``None`` for anything that is not an envelope (an optimizer source);
    the validated envelope dict otherwise. Raises :class:`EnvelopeError` for a
    payload that starts like an envelope but is malformed."""
    if not looks_like_envelope(payload):
        return None
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        env = json.loads(text)
    except (UnicodeDecodeError, ValueError) as e:
        raise EnvelopeError(f"envelope is not valid JSON ({type(e).__name__})")
    if not isinstance(env, dict) or set(env) != {ENVELOPE_KEY, "task", "source", "weights"}:
        raise EnvelopeError("envelope must have exactly the keys refinery_submission, task, source, weights")
    if env[ENVELOPE_KEY] != ENVELOPE_VERSION:
        raise EnvelopeError(f"unsupported envelope version {env[ENVELOPE_KEY]!r}")
    if env["task"] != REFINER_TASK_ID:
        raise EnvelopeError(f"unknown task {env['task']!r}")
    if not isinstance(env["source"], str) or not env["source"].strip():
        raise EnvelopeError("envelope source must be a non-empty string")
    if env["weights"] is not None:
        _check_weights_ref(env["weights"])
    # One preimage per submission: the canonical form is the only accepted one,
    # so the same source + weights cannot be committed under two hashes.
    if build_envelope(env["source"], env["weights"]) != text:
        raise EnvelopeError("envelope is not in canonical form (use sn125.refiner.submission.build_envelope)")
    return env


def gate_envelope(env: dict) -> tuple[bool, str]:
    """The pre-selection gate for a refiner submission: the same source checks
    ``refiner-eval`` runs on the box (single-source gate + literal CAPABILITIES),
    so a submission that cannot run never takes an evaluation slot. The weight
    file itself is checked on the box, against the committed hash and size."""
    from ..gate import validate_single_source
    from .refproc_limits import extract_refiner_capabilities
    res = validate_single_source(env["source"], path="refiner.py")
    if not res.ok:
        return False, "; ".join(res.reasons[:3])
    try:
        extract_refiner_capabilities(env["source"], strict=True)
    except Exception as e:
        return False, f"CAPABILITIES: {e}"
    return True, ""


# ── refiner-eval result line (printed on the box, parsed by the cloud layer) ──

RESULT_MARKER = "REFINER_EVAL_RESULT "
_PER_ITEM_KEYS = ("items_pcn", "items_bpb")


def compact_result(result: dict) -> dict:
    """The result without per-item bench lists (the full one is the --out file)."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in _PER_ITEM_KEYS}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    return strip(result)


def parse_result_line(stdout: str, rc: int) -> dict:
    """Find the last ``REFINER_EVAL_RESULT`` line in a run's output. A run that
    printed none did not finish as refiner-eval: that is the harness's fault,
    never the miner's."""
    for line in reversed((stdout or "").splitlines()):
        s = line.strip()
        if not s.startswith(RESULT_MARKER):
            continue
        try:
            result = json.loads(s[len(RESULT_MARKER):])
        except json.JSONDecodeError:
            break
        if rc != 0 and not result.get("failed"):
            result["failed"] = True
            result.setdefault("error", f"refiner-eval failed (rc={rc})")
            result.setdefault("fault", "harness")
        result.setdefault("score", -1.0)
        result.setdefault("components", {})
        return result
    return {"failed": True, "score": -1.0, "components": {}, "fault": "harness", "task_id": REFINER_TASK_ID,
            "error": f"no json result from refiner-eval (rc={rc}): {(stdout or '')[-500:]}"}
