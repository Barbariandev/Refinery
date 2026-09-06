"""Authorship proof for SN125 optimizer submissions.

The Refinery publication pipeline credits optimizers to the hotkeys that submitted
them (round JSONs / attestations record ``code_hash = sha256(source)`` per
hotkey). This tool lets a miner PROVE, offline and unforgeably, that they
control the hotkey behind a submission — e.g. to claim authorship on the Refinery
optimizer paper, or to link a pseudonymous submission to a real identity later.

How it works:

- ``sign``: the miner signs a canonical JSON payload binding their hotkey SS58
  to the submission's code_hash (plus optional round_id / free-text statement)
  with the hotkey's sr25519 key — the same key that signed the original
  commit/reveal traffic, so the chain of custody is airtight.
- ``verify``: anyone checks the signature against the claimed SS58 address with
  no wallet and no chain access, then (optionally) re-hashes a provided source
  file to confirm it matches the claimed code_hash. Cross-checking the
  (hotkey, code_hash) pair against the published round JSONs / attestation
  bundles completes the proof that this hotkey submitted this optimizer.

Usage:

    python -m sn125.prove_authorship sign --wallet-name W --wallet-hotkey H \
        --source optimizer.py [--round-id R] [--statement "I am ..."] \
        [--out claim.json]

    python -m sn125.prove_authorship verify claim.json [--source optimizer.py]

The claim JSON is self-contained and safe to publish: it holds only public
data (SS58 address, hashes, signature). It never contains key material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

CLAIM_VERSION = 1

_UNSIGNED_FIELDS = frozenset({"signature"})


def source_code_hash(source: str) -> str:
    """The submission identifier used across rounds/attestations: sha256(source)."""
    return hashlib.sha256(source.encode()).hexdigest()


def claim_payload(claim: dict) -> bytes:
    """Canonical signing bytes for a claim (everything except the signature)."""
    body = {k: v for k, v in claim.items() if k not in _UNSIGNED_FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def build_claim(hotkey_ss58: str, code_hash: str, *, round_id: str = "",
                statement: str = "", timestamp: int | None = None) -> dict:
    """Assemble the unsigned claim body."""
    if not hotkey_ss58:
        raise ValueError("hotkey_ss58 is required")
    if not code_hash or len(code_hash) != 64:
        raise ValueError(f"code_hash must be a sha256 hex digest, got {code_hash!r}")
    return {
        "version": CLAIM_VERSION,
        "kind": "sn125_authorship_claim",
        "hotkey": hotkey_ss58,
        "code_hash": code_hash.lower(),
        "round_id": round_id,
        "statement": statement,
        "timestamp": int(time.time()) if timestamp is None else int(timestamp),
    }


def sign_claim(claim: dict, keypair) -> dict:
    """Sign the claim with an sr25519 keypair (bittensor wallet hotkey).
    Returns a new dict with the hex signature attached."""
    signed = dict(claim)
    signed["signature"] = keypair.sign(claim_payload(claim)).hex()
    return signed


def verify_claim(claim: dict, *, source: str | None = None,
                 keypair_cls=None) -> dict:
    """Verify a claim offline. Returns a report dict; ``ok`` is the verdict.

    Checks, in order (each independent, all reported):
      1. structural: required fields present, version known
      2. signature: sr25519 verify against the claimed SS58 hotkey
      3. source match (only when ``source`` given): sha256(source) == code_hash
    """
    report: dict = {"ok": False, "checks": {}}

    required = ("version", "kind", "hotkey", "code_hash", "signature")
    missing = [k for k in required if not claim.get(k)]
    structural = not missing and claim.get("kind") == "sn125_authorship_claim"
    report["checks"]["structure"] = (
        "ok" if structural else f"missing/invalid fields: {missing or ['kind']}")

    sig_ok = False
    if structural:
        try:
            if keypair_cls is None:
                import bittensor as bt
                keypair_cls = bt.Keypair
            kp = keypair_cls(ss58_address=claim["hotkey"])
            sig_ok = bool(kp.verify(claim_payload(claim),
                                    bytes.fromhex(claim["signature"])))
            report["checks"]["signature"] = "ok" if sig_ok else "signature invalid"
        except Exception as e:
            report["checks"]["signature"] = f"verification error: {e}"

    source_ok = True
    if source is not None:
        actual = source_code_hash(source)
        source_ok = actual == str(claim.get("code_hash", "")).lower()
        report["checks"]["source_hash"] = (
            "ok" if source_ok
            else f"source hashes to {actual}, claim says {claim.get('code_hash')}")

    report["ok"] = structural and sig_ok and source_ok
    report["hotkey"] = claim.get("hotkey", "")
    report["code_hash"] = claim.get("code_hash", "")
    return report




def _cmd_sign(args) -> int:
    if bool(args.source) == bool(args.code_hash):
        print("ERROR: provide exactly one of --source or --code-hash", file=sys.stderr)
        return 2
    code_hash = (source_code_hash(Path(args.source).read_text())
                 if args.source else args.code_hash)

    import bittensor as bt
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    keypair = wallet.hotkey

    claim = build_claim(keypair.ss58_address, code_hash,
                        round_id=args.round_id, statement=args.statement)
    signed = sign_claim(claim, keypair)
    out = json.dumps(signed, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(out + "\n")
        print(f"claim written to {args.out}")
    print(out)
    return 0


def _cmd_verify(args) -> int:
    claim = json.loads(Path(args.claim).read_text())
    source = Path(args.source).read_text() if args.source else None
    report = verify_claim(claim, source=source)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["ok"]:
        print(f"\nVALID: hotkey {report['hotkey']} authored code_hash "
              f"{report['code_hash'][:16]}…")
        print("Cross-check this (hotkey, code_hash) pair against the published "
              "round JSONs / attestation bundle to complete the proof.")
        return 0
    print("\nINVALID claim", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sn125.prove_authorship", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_sign = sub.add_parser("sign", help="Sign an authorship claim with your hotkey")
    p_sign.add_argument("--wallet-name", default="default")
    p_sign.add_argument("--wallet-hotkey", default="default")
    p_sign.add_argument("--source", default="", help="optimizer.py to hash and claim")
    p_sign.add_argument("--code-hash", default="",
                        help="claim an already-known sha256 code hash instead of --source")
    p_sign.add_argument("--round-id", default="", help="optional round id context")
    p_sign.add_argument("--statement", default="",
                        help="optional free-text statement (e.g. name/contact for paper credit)")
    p_sign.add_argument("--out", default="", help="write the signed claim JSON here")

    p_ver = sub.add_parser("verify", help="Verify a signed authorship claim (no wallet needed)")
    p_ver.add_argument("claim", help="path to the claim JSON")
    p_ver.add_argument("--source", default="",
                       help="optional optimizer.py to check against the claimed code_hash")

    args = ap.parse_args(argv)
    return _cmd_sign(args) if args.command == "sign" else _cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
