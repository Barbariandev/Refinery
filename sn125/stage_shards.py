"""Stage the hash-pinned FineWeb production shards on an eval worker.

The production evaluator must not silently train on legacy streaming data.  This
module gives cloud workers one narrow path:

1. accept an already-present shard directory if MANIFEST.json verifies;
2. otherwise download a prebuilt shard snapshot from a configured Hugging Face
   repo and verify it;
3. optionally rebuild from raw FineWeb only when explicitly requested.

Raw rebuilding is intentionally opt-in. It is deterministic, but it is far too
slow to do on every paid evaluation worker.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from .fineweb import MANIFEST_NAME, build_fineweb_shards, default_data_dir, load_manifest, verify_split


def _split_names(mf: dict) -> list[str]:
    return [s for s, entries in (mf.get("splits") or {}).items() if entries]


def _verify_dir(data_dir: Path, *, expected_manifest: str = "", verify_hashes: bool = True) -> dict:
    mf = load_manifest(data_dir)
    actual = str(mf.get("manifest_hash", ""))
    if expected_manifest and actual != expected_manifest:
        raise ValueError(
            f"manifest mismatch at {data_dir}: have {actual}, expected {expected_manifest}"
        )
    if verify_hashes:
        for split in _split_names(mf):
            verify_split(data_dir, mf, split)
    return {
        "ok": True,
        "data_dir": str(data_dir),
        "manifest_hash": actual,
        "seq_len": mf.get("seq_len"),
        "totals": mf.get("totals", {}),
        "verified_hashes": bool(verify_hashes),
    }


def _existing_ok(data_dir: Path, *, expected_manifest: str, verify_hashes: bool) -> dict | None:
    if not (data_dir / MANIFEST_NAME).exists():
        return None
    try:
        out = _verify_dir(data_dir, expected_manifest=expected_manifest, verify_hashes=verify_hashes)
        out["source"] = "existing"
        return out
    except Exception as e:
        return {
            "ok": False,
            "source": "existing",
            "data_dir": str(data_dir),
            "error": str(e),
            "error_type": type(e).__name__,
        }


# The production snapshot is 976 files; every file is ~2 Hub "resolver" requests
# and the quota is 5000 per 5 minutes per account, shared by every box staging
# at the same time. A third concurrent box therefore gets HTTP 429 — which
# huggingface_hub reports from its HEAD call as a bare LocalEntryNotFoundError
# ("cannot find the requested files in the local cache"), hiding the cause.
# snapshot_download resumes what is already on disk, so waiting out the quota
# window and calling again finishes the job instead of failing the box
# (12 evaluations lost this way in rounds 20, 26, 27, 28).
DOWNLOAD_ATTEMPTS = int(os.environ.get("SN125_FINEWEB_DOWNLOAD_ATTEMPTS", "6"))
DOWNLOAD_BACKOFF_S = (60, 120, 240, 300, 300)


def _describe(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__ or exc.__context__
    seen = 0
    while cause is not None and seen < 3:
        parts.append(f"caused by {type(cause).__name__}: {cause}")
        cause = cause.__cause__ or cause.__context__
        seen += 1
    return " | ".join(p.strip() for p in parts)


def _download_with_backoff(fn, **kwargs):
    """Call ``fn(**kwargs)`` (snapshot_download) up to DOWNLOAD_ATTEMPTS times,
    sleeping DOWNLOAD_BACKOFF_S between attempts; re-raises the last error with
    its cause chain in the message so a 429 is visible in the audit."""
    last: BaseException | None = None
    attempts = max(1, DOWNLOAD_ATTEMPTS)
    for i in range(attempts):
        try:
            return fn(**kwargs)
        except Exception as e:  # noqa: BLE001 — any Hub/network failure is retried
            last = e
            if i + 1 >= attempts:
                break
            wait = DOWNLOAD_BACKOFF_S[min(i, len(DOWNLOAD_BACKOFF_S) - 1)]
            print(f"stage-shards: download attempt {i + 1}/{attempts} failed ({_describe(e)}); "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"shard download failed after {attempts} attempts: {_describe(last)}") from last


def _download_from_hf(
    data_dir: Path,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    subdir: str,
    expected_manifest: str,
    verify_hashes: bool,
) -> dict:
    from huggingface_hub import snapshot_download

    tmp = data_dir.with_name(f".{data_dir.name}.stage-{os.getpid()}-{int(time.time())}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    subdir = subdir.strip("/")
    if subdir:
        patterns = [f"{subdir}/{MANIFEST_NAME}", f"{subdir}/*.npy"]
        snapshot = Path(_download_with_backoff(
            snapshot_download,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision or None,
            allow_patterns=patterns,
        ))
        src = snapshot / subdir
        shutil.copytree(src, tmp)
    else:
        _download_with_backoff(
            snapshot_download,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision or None,
            allow_patterns=[MANIFEST_NAME, "*.npy"],
            local_dir=str(tmp),
        )

    out = _verify_dir(tmp, expected_manifest=expected_manifest, verify_hashes=verify_hashes)
    if data_dir.exists():
        bad = data_dir.with_name(f"{data_dir.name}.replaced-{int(time.time())}")
        data_dir.rename(bad)
        out["replaced_existing"] = str(bad)
    tmp.rename(data_dir)
    out["data_dir"] = str(data_dir)
    out["source"] = "huggingface"
    out["hf_repo"] = repo_id
    out["hf_repo_type"] = repo_type
    out["hf_revision"] = revision
    out["hf_subdir"] = subdir
    return out


def make_world_readable(data_dir: Path) -> None:
    """The sandboxed eval worker runs as nobody (uid 65534) and must read the
    shards + MANIFEST.json: force 0755 dirs / 0644 files under ``data_dir``
    and make every ancestor traversable, whatever umask the staging process
    ran under. Best-effort (non-root callers may not own the ancestors)."""
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o755)
            except OSError:
                pass
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o644)
            except OSError:
                pass
    try:
        os.chmod(data_dir, 0o755)
    except OSError:
        pass
    # Ancestors: only inside the eval box's /workspace tree. Never widen a
    # home directory or anything else on a dev host (a 0700 /root must stay
    # 0700 — it holds .env and keys).
    p = data_dir.parent
    workspace = Path("/workspace")
    while p != p.parent and (p == workspace or workspace in p.parents):
        try:
            mode = p.stat().st_mode & 0o777
            if mode & 0o005 != 0o005:
                os.chmod(p, mode | 0o055)
        except OSError:
            break
        p = p.parent


def stage_prod_shards(
    data_dir: str | Path | None = None,
    *,
    expected_manifest: str = "",
    hf_repo: str = "",
    hf_repo_type: str = "dataset",
    hf_revision: str = "",
    hf_subdir: str = "",
    verify_hashes: bool = True,
    allow_build: bool = False,
) -> dict:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir = data_dir.expanduser().resolve()

    existing = _existing_ok(data_dir, expected_manifest=expected_manifest, verify_hashes=verify_hashes)
    if existing and existing.get("ok"):
        make_world_readable(data_dir)
        return existing

    hf_repo = hf_repo or os.environ.get("SN125_FINEWEB_HF_REPO", "").strip()
    hf_repo_type = hf_repo_type or os.environ.get("SN125_FINEWEB_HF_REPO_TYPE", "dataset").strip() or "dataset"
    hf_revision = hf_revision or os.environ.get("SN125_FINEWEB_HF_REVISION", "").strip()
    hf_subdir = hf_subdir or os.environ.get("SN125_FINEWEB_HF_SUBDIR", "").strip()
    if hf_repo:
        out = _download_from_hf(
            data_dir,
            repo_id=hf_repo,
            repo_type=hf_repo_type,
            revision=hf_revision,
            subdir=hf_subdir,
            expected_manifest=expected_manifest,
            verify_hashes=verify_hashes,
        )
        if existing and not existing.get("ok"):
            out["previous_error"] = existing
        make_world_readable(data_dir)
        return out

    if allow_build:
        mf = build_fineweb_shards(out_dir=data_dir)
        out = _verify_dir(data_dir, expected_manifest=expected_manifest, verify_hashes=verify_hashes)
        out["source"] = "rebuilt_from_raw_fineweb"
        out["manifest_hash"] = mf["manifest_hash"]
        if existing and not existing.get("ok"):
            out["previous_error"] = existing
        make_world_readable(data_dir)
        return out

    msg = (
        f"No verified production shards at {data_dir}. Set SN125_FINEWEB_HF_REPO "
        "to a Hugging Face dataset/model repo containing MANIFEST.json + *.npy, "
        "or pass --allow-build to rebuild from raw FineWeb."
    )
    if existing and not existing.get("ok"):
        msg += f" Existing shard dir failed verification: {existing.get('error')}"
    raise FileNotFoundError(msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="", help="target shard dir; default SN125_FINEWEB_DIR/repo data")
    ap.add_argument("--expected-manifest", default=os.environ.get("SN125_FINEWEB_MANIFEST_HASH", ""))
    ap.add_argument("--hf-repo", default=os.environ.get("SN125_FINEWEB_HF_REPO", ""))
    ap.add_argument("--hf-repo-type", default=os.environ.get("SN125_FINEWEB_HF_REPO_TYPE", "dataset"))
    ap.add_argument("--hf-revision", default=os.environ.get("SN125_FINEWEB_HF_REVISION", ""))
    ap.add_argument("--hf-subdir", default=os.environ.get("SN125_FINEWEB_HF_SUBDIR", ""))
    ap.add_argument("--no-verify-hashes", action="store_true",
                    help="only validate MANIFEST self-hash; skip per-shard sha256 checks")
    ap.add_argument("--allow-build", action="store_true",
                    default=os.environ.get("SN125_FINEWEB_BUILD_IF_MISSING", "") == "1",
                    help="rebuild deterministically from raw FineWeb if no HF snapshot is configured")
    args = ap.parse_args(argv)

    try:
        result = stage_prod_shards(
            args.data_dir or None,
            expected_manifest=args.expected_manifest,
            hf_repo=args.hf_repo,
            hf_repo_type=args.hf_repo_type,
            hf_revision=args.hf_revision,
            hf_subdir=args.hf_subdir,
            verify_hashes=not args.no_verify_hashes,
            allow_build=args.allow_build,
        )
    except Exception as e:
        result = {"ok": False, "error": str(e), "error_type": type(e).__name__}
        print("STAGE_SHARDS_RESULT " + json.dumps(result, default=str), flush=True)
        return 1
    print("STAGE_SHARDS_RESULT " + json.dumps(result, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
