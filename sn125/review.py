"""Operator review verdicts for Challenge 2 submissions (docs/REFINER_TASK.md §3).

A refiner must make its decisions from the text it is shown: a submission must
not carry data keyed to specific documents of the training corpus — by
document ID, content hash, fingerprint or equivalent. Corpus-level statistics
(token and n-gram frequencies, domain priors, learned scorers) remain legal.
The rule is enforced by operator review, in the open: the operator records a
verdict per commit hash in a small JSON file, the round loop refuses to launch
a rejected submission — the initial run and any pending confirmation run
alike — and the rejection reason is published in the round record.

The file (``SN125_REFINER_REVIEW_FILE``, default
``sn125/rounds/refiner_review.json``) maps the commit hash — the SHA-256 of
the revealed envelope text — to a verdict:

    {"<64-hex commit hash>": {"verdict": "reject", "reason": "...",
                              "reviewer": "...", "date": "YYYY-MM-DD"}}

Only ``"verdict": "reject"`` acts; any other value (``"allow"`` is the
convention for reviewed-and-passed) is bookkeeping and leaves the submission
untouched, as do unknown hashes, a missing file and an unreadable file (the
review step must never take Challenge 2 down on its own). The file is re-read
on every decision, so a verdict added while the validator runs takes effect at
the submission's next launch — including a confirmation run — without a
restart.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

REVIEW_FILE_ENV = "SN125_REFINER_REVIEW_FILE"
_DEFAULT_REVIEW_FILE = Path(__file__).resolve().parent / "rounds" / "refiner_review.json"


def review_file(env: dict | None = None) -> Path:
    e = os.environ if env is None else env
    return Path(e.get(REVIEW_FILE_ENV, "") or _DEFAULT_REVIEW_FILE)


def load_verdicts(env: dict | None = None) -> dict[str, dict]:
    """``{commit_hash: verdict dict}``; empty on a missing or unreadable file."""
    path = review_file(env)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.error("refiner review file %s is unreadable (%s); treating it as empty", path, exc)
        return {}
    if not isinstance(raw, dict):
        log.error("refiner review file %s must hold a JSON object; treating it as empty", path)
        return {}
    return {str(k).strip().lower(): v for k, v in raw.items() if isinstance(v, dict)}


def review_rejection(payload: bytes | str, env: dict | None = None) -> str:
    """Non-empty rejection reason iff the operator rejected this exact payload
    (looked up by its commit hash, the SHA-256 of the payload text)."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    verdict = load_verdicts(env).get(hashlib.sha256(data).hexdigest())
    if verdict and str(verdict.get("verdict", "")).strip().lower() == "reject":
        return str(verdict.get("reason", "") or "").strip() or "rejected by operator review"
    return ""
