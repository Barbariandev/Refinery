"""Continuous artifact publishing (launch item: no manual copy steps).

Continuously mirrors validator-local artifacts — round JSONs, JSONL audit logs,
the rotating process log, the durable payment ledger, the inter-round state
checkpoint, cloud status, and anything else configured — to two durable sinks:
a Cloudflare R2 bucket and a private Hugging Face dataset repo. On every cycle
it also rebuilds two derived feeds: the public dashboard contract
(dashboard/snapshot.py) and the private miner-operations view (ops.py:
credits per coldkey, deferred queue, live-round progress, recent outcomes,
logbook index). The dashboard consumes the R2 snapshot with
dashboard/r2_source.py or private HF repositories with
dashboard/live_sources.py.

Key layout under the prefix (default ``artifacts/``):

    rounds/<round>.json           full round records
    audit/<round>.jsonl|.log      per-round audit trail
    audit/all_rounds.jsonl|.log   the complete logbook across rounds
    audit/validator.log[.N]       rotating process log (what journalctl shows)
    audit/autoupdate.log[.N]      supervisor log when running --auto-update
    payments/ledger.json          durable credit ledger (source of truth for money)
    state/validator_state.json    deferred queue, round counter, last weights
    ops/{ops,credits,evaluations,logbook}.json   miner-operations feed
    dashboard/{snapshot,dashboard,cloud-status}.json
    cloud/cloud_status.json, publisher/publisher_state.json

Design constraints:

- Publishing must NEVER affect the validator loop. Every sink call is wrapped;
  a dead sink degrades to a log line and the file stays dirty for retry.
- Change detection is (size, mtime_ns) per file per sink. A failed upload keeps
  the old fingerprint so the next cycle retries automatically.
- Files may be mid-append when scanned (audit JSONL during a live round). A
  torn tail is acceptable: the next cycle re-uploads the grown file. Losing at
  most one cycle of tail beats losing 20h of audit trail to a dead box.
- Everything in watched roots gets published. This is not a sanitization
  boundary: keep the raw archive private and review extra paths before enabling.

Configuration (all env):

    R2_ENDPOINT / R2_ACCESS_KEY / R2_SECRET_KEY / R2_BUCKET   (same as attestation)
    SN125_PUBLISH_R2_PREFIX      key prefix, default "artifacts/"
    SN125_PUBLISH_HF_REPO        e.g. "125datasets/sn125-artifacts" (enables HF sink)
    SN125_PUBLISH_HF_REPO_TYPE   default "dataset"
    HF_WRITE_TOKEN / HF_TOKEN / HUGGINGFACE_HUB_TOKEN         (first non-empty)
    SN125_PUBLISH_INTERVAL_S     sync cadence, default 60
    SN125_PUBLISH_MAX_BYTES      per-file cap, default 256 MiB
    SN125_PUBLISH_EXTRA_PATHS    comma-separated extra files/dirs to watch
    SN125_PUBLISH_ENABLED        force on/off; default: on iff any sink configured
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("sn125.publisher")

DEFAULT_INTERVAL_S = 60.0
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_DIR_PATTERNS = ("*.json", "*.jsonl", "*.log", "*.md")
AUDIT_PATTERNS = ("*.jsonl", "*.log", "*.log.[0-9]*")

_CONTENT_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".log": "text/plain",
    ".md": "text/markdown",
}


def _content_type(path: Path) -> str:
    if ".log." in path.name:
        return "text/plain"
    return _CONTENT_TYPES.get(path.suffix, "application/octet-stream")


def _bool_env(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _hf_token(env: dict[str, str]) -> str:
    for key in ("HF_WRITE_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = env.get(key, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class WatchRoot:
    """A logical artifact source: a single file or a directory of artifacts.

    ``name`` becomes the key prefix in every sink, e.g. root "audit" with file
    round_x.jsonl publishes as "audit/round_x.jsonl".
    """
    name: str
    path: Path
    patterns: tuple[str, ...] = DEFAULT_DIR_PATTERNS


class R2Sink:
    """Mirrors files to an S3-compatible (R2) bucket under a key prefix."""

    name = "r2"

    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str, prefix: str = "artifacts/") -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self._client = None

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "R2Sink | None":
        from . import settings
        if not settings.r2_configured(env):
            return None
        cfg = settings.r2_config(env)
        return cls(cfg["endpoint"], cfg["access_key"], cfg["secret_key"], cfg["bucket"],
                   prefix=env.get("SN125_PUBLISH_R2_PREFIX", "artifacts/"))

    def _s3(self):
        if self._client is None:
            import boto3
            from botocore.config import Config
            self._client = boto3.client(
                "s3", endpoint_url=self.endpoint, region_name="auto",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}))
        return self._client

    def upload(self, files: list[tuple[str, Path]]) -> dict[str, str]:
        """Upload (key, path) pairs. Returns {key: error} for failures only."""
        failures: dict[str, str] = {}
        for key, path in files:
            try:
                body = path.read_bytes()
                self._s3().put_object(
                    Bucket=self.bucket,
                    Key=f"{self.prefix}{key}",
                    Body=body,
                    ContentType=_content_type(path),
                    CacheControl="no-cache",
                )
            except Exception as exc:
                failures[key] = type(exc).__name__
        return failures


class HFSink:
    """Mirrors files into a private Hugging Face repo, one commit per cycle."""

    name = "hf"

    def __init__(self, repo_id: str, token: str, repo_type: str = "dataset") -> None:
        self.repo_id = repo_id
        self.token = token
        self.repo_type = repo_type
        self._repo_ready = False

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "HFSink | None":
        repo_id = env.get("SN125_PUBLISH_HF_REPO", "").strip()
        if not repo_id:
            return None
        token = _hf_token(env)
        if not token:
            log.warning("SN125_PUBLISH_HF_REPO set but no HF token found; HF sink disabled")
            return None
        return cls(repo_id, token,
                   repo_type=env.get("SN125_PUBLISH_HF_REPO_TYPE", "dataset").strip() or "dataset")

    def _ensure_repo(self, api) -> None:
        if self._repo_ready:
            return
        api.create_repo(repo_id=self.repo_id, repo_type=self.repo_type,
                        private=True, exist_ok=True)
        self._repo_ready = True

    def upload(self, files: list[tuple[str, Path]]) -> dict[str, str]:
        """Upload (key, path) pairs as ONE commit. All-or-nothing per cycle."""
        if not files:
            return {}
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
            api = HfApi(token=self.token)
            self._ensure_repo(api)
            ops = [CommitOperationAdd(path_in_repo=key, path_or_fileobj=str(path))
                   for key, path in files]
            api.create_commit(
                repo_id=self.repo_id, repo_type=self.repo_type, operations=ops,
                commit_message=f"sn125 artifact sync ({len(ops)} files)")
            return {}
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            return {key: err for key, _ in files}


class ContinuousPublisher:
    """Background thread that scans watch roots and syncs changed files to sinks.

    ``publish_once()`` is the synchronous unit (used by tests and the final
    flush); ``start()``/``stop()`` run it on a cadence; ``publish_now()`` wakes
    the thread immediately (called at round boundaries for freshness).
    """

    def __init__(self, roots: list[WatchRoot], sinks: list[Any], *,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 state_path: Path | None = None, prepare=None) -> None:
        self.roots = list(roots)
        self.sinks = list(sinks)
        self.interval_s = max(5.0, float(interval_s))
        self.max_bytes = int(max_bytes)
        self.state_path = Path(state_path) if state_path else None
        self.prepare = prepare
        self._published: dict[tuple[str, str], tuple[int, int]] = {}
        self._skipped_large: set[str] = set()
        self._cycles = 0
        self._last_report: dict[str, Any] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None


    def scan(self) -> dict[str, Path]:
        """Map of publish key -> existing local file across all roots."""
        out: dict[str, Path] = {}
        for root in self.roots:
            try:
                if root.path.is_file():
                    out[f"{root.name}/{root.path.name}"] = root.path
                    continue
                if not root.path.is_dir():
                    continue
                for pattern in root.patterns:
                    for path in root.path.glob(pattern):
                        if path.is_file():
                            out[f"{root.name}/{path.name}"] = path
            except Exception as exc:
                log.warning("publisher scan failed for %s: %s", root.path, exc)
        return out

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int] | None:
        try:
            st = path.stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None


    def publish_once(self) -> dict[str, Any]:
        """One sync cycle: upload every file whose fingerprint changed, per sink.

        Never raises. Returns a report dict (also persisted to state_path).
        """
        started = time.time()
        report: dict[str, Any] = {"ts": int(started), "sinks": {}, "files_tracked": 0}
        try:
            if self.prepare is not None:
                try:
                    self.prepare()
                    report["dashboard_snapshot"] = {"ok": True}
                except Exception as exc:
                    report["dashboard_snapshot"] = {"ok": False, "error": type(exc).__name__}
                    log.warning("dashboard snapshot build failed: %s", type(exc).__name__)
            files = self.scan()
            report["files_tracked"] = len(files)
            fingerprints: dict[str, tuple[int, int]] = {}
            for key, path in files.items():
                fp = self._fingerprint(path)
                if fp is None:
                    continue
                if fp[0] > self.max_bytes:
                    if key not in self._skipped_large:
                        self._skipped_large.add(key)
                        log.warning("publisher skipping %s (%d bytes > cap %d)",
                                    key, fp[0], self.max_bytes)
                    continue
                fingerprints[key] = fp

            for sink in self.sinks:
                pending = [(key, files[key]) for key, fp in sorted(fingerprints.items())
                           if self._published.get((sink.name, key)) != fp]
                sink_report = {"pending": len(pending), "uploaded": 0,
                               "failed": 0, "last_error": ""}
                if pending:
                    try:
                        failures = sink.upload(pending) or {}
                    except Exception as exc:
                        failures = {key: f"{type(exc).__name__}: {exc}" for key, _ in pending}
                    for key, _path in pending:
                        if key in failures:
                            sink_report["failed"] += 1
                            sink_report["last_error"] = failures[key]
                        else:
                            self._published[(sink.name, key)] = fingerprints[key]
                            sink_report["uploaded"] += 1
                    if failures:
                        log.warning("publisher sink %s: %d/%d uploads failed (%s)",
                                    sink.name, sink_report["failed"], len(pending),
                                    sink_report["last_error"])
                report["sinks"][sink.name] = sink_report
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("publisher cycle failed: %s", exc)

        self._cycles += 1
        report["cycle"] = self._cycles
        report["elapsed_s"] = round(time.time() - started, 3)
        self._last_report = report
        self._write_state(report)
        return report

    def _write_state(self, report: dict[str, Any]) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        except Exception:
            pass


    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                self.publish_once()
                self._wake.wait(self.interval_s)
                self._wake.clear()

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="sn125-artifact-publisher")
        self._thread.start()
        log.info("continuous artifact publisher started: %d root(s), %d sink(s), every %.0fs",
                 len(self.roots), len(self.sinks), self.interval_s)

    def publish_now(self) -> None:
        """Wake the background thread for an immediate cycle (round boundaries)."""
        self._wake.set()

    def stop(self, *, flush: bool = True) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        if flush:
            self.publish_once()


    @classmethod
    def from_validator(cls, validator, env: dict[str, str] | None = None
                       ) -> "ContinuousPublisher | None":
        """Build the production publisher for a validator, or None when disabled.

        Watches: rounds dir, audit dir (incl. process logs), cloud_status.json,
        the payment ledger, the loop-state checkpoint, the derived dashboard
        and ops feeds, and any SN125_PUBLISH_EXTRA_PATHS. Enabled iff a sink
        is configured (or forced via SN125_PUBLISH_ENABLED).
        """
        env = dict(env if env is not None else os.environ)
        forced = _bool_env(env.get("SN125_PUBLISH_ENABLED"))
        if forced is False:
            return None

        sinks: list[Any] = []
        r2 = R2Sink.from_env(env)
        r2_keys = ("R2_ENDPOINT", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET")
        if r2 is None and any(env.get(key, "").strip() for key in r2_keys):
            log.warning("R2 publishing incomplete; missing: %s",
                        ", ".join(key for key in r2_keys if not env.get(key, "").strip()))
        if r2 is not None:
            sinks.append(r2)
        hf = HFSink.from_env(env)
        if hf is not None:
            sinks.append(hf)
        if not sinks:
            if forced:
                log.warning("SN125_PUBLISH_ENABLED set but no sink configured "
                            "(need R2_* and/or SN125_PUBLISH_HF_REPO)")
            else:
                log.info("continuous artifact publishing disabled: no sink configured")
            return None

        pkg_dir = Path(__file__).resolve().parent
        rounds_dir = Path(getattr(validator, "rounds_dir", "") or (pkg_dir / "rounds"))
        audit_dir = Path(getattr(validator, "audit_dir", "") or (rounds_dir.parent / "audit"))
        registry = getattr(validator, "payment_registry", None)
        ledger_path = Path(getattr(registry, "store_path", None)
                           or audit_dir / "payments" / "ledger.json")
        state_path = Path(getattr(validator, "state_path", None)
                          or audit_dir / "validator_state.json")
        roots = [
            WatchRoot("rounds", rounds_dir, ("*.json",)),
            WatchRoot("audit", audit_dir, AUDIT_PATTERNS),
            WatchRoot("cloud", pkg_dir / "cloud_status.json"),
            WatchRoot("payments", ledger_path),
            WatchRoot("state", state_path),
        ]
        for i, raw in enumerate(x.strip() for x in
                                env.get("SN125_PUBLISH_EXTRA_PATHS", "").split(",")):
            if raw:
                roots.append(WatchRoot(f"extra{i}", Path(raw).expanduser()))

        try:
            interval_s = float(env.get("SN125_PUBLISH_INTERVAL_S", DEFAULT_INTERVAL_S))
        except ValueError:
            interval_s = DEFAULT_INTERVAL_S
        try:
            max_bytes = int(env.get("SN125_PUBLISH_MAX_BYTES", DEFAULT_MAX_BYTES))
        except ValueError:
            max_bytes = DEFAULT_MAX_BYTES

        from .dashboard.snapshot import build_snapshot, write_snapshot
        from .neuron import SN125_VERSION
        from .ops import OPS_DIRNAME, build_ops_snapshot, write_ops_snapshot
        snapshot_dir = audit_dir / "dashboard"
        ops_dir = audit_dir / OPS_DIRNAME
        cloud_path = pkg_dir / "cloud_status.json"
        wallet = getattr(validator, "wallet", None)
        hotkey = wallet.hotkey.ss58_address if wallet is not None else "unconfigured"
        ops_extra = {
            "netuid": getattr(validator, "netuid", None),
            "network": getattr(validator, "network", ""),
            "started_at": int(time.time()),
        }

        def prepare():
            errors = []
            try:
                write_snapshot(snapshot_dir, build_snapshot(
                    rounds_dir, cloud_path, validator_hotkey=hotkey, version=SN125_VERSION,
                    ledger_path=ledger_path, state_path=state_path, audit_dir=audit_dir))
            except Exception as exc:
                errors.append(exc)
            try:
                write_ops_snapshot(ops_dir, build_ops_snapshot(
                    rounds_dir=rounds_dir, audit_dir=audit_dir, ledger_path=ledger_path,
                    state_path=state_path, validator_hotkey=hotkey,
                    version=SN125_VERSION, extra=ops_extra))
            except Exception as exc:
                errors.append(exc)
            if errors:
                raise errors[0]

        roots.append(WatchRoot("dashboard", snapshot_dir, ("*.json",)))
        roots.append(WatchRoot(OPS_DIRNAME, ops_dir, ("*.json",)))
        publisher = cls(roots, sinks, interval_s=interval_s, max_bytes=max_bytes,
                        state_path=audit_dir / "publisher_state.json", prepare=prepare)
        publisher.roots.append(WatchRoot("publisher", publisher.state_path))
        return publisher
