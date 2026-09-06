"""
RunPod client for the SN125 validator — the PRIMARY scoring provider
(Secure Cloud tier) since 2026-09-04.

RunPod rents GPU *containers* (pods), not VMs. A pod (probed live 2026-09-04)
runs as root with the default Docker capability set — no CAP_SYS_ADMIN, no
CAP_NET_ADMIN — under Docker's default seccomp profile (every ``unshare``
flavour is EPERM) and AppArmor. So the VM-host controls (``unshare``
namespace sandbox, ``iptables`` egress lockdown) cannot engage; instead the
orchestrator runs the SECCOMP sandbox flavour on RunPod boxes
(``sn125/seccomp_sandbox.py``: priv drop to nobody + NO_NEW_PRIVS + a
stacked syscall filter that denies every non-UNIX socket, io_uring, ptrace
and mount syscalls, positively probed and attested in the worker's
nonce-authenticated result) and a ``sandbox_guard`` stage in place of the
iptables stage. See ``cloud.CONTAINER_PROVIDERS``.

Trust: SECURE cloud pods run in RunPod-operated data centers and are
treated like any other first-party fleet. COMMUNITY pods run on third-party
hosts (the Lium problem) and are refused for scoring
(``cloud.ensure_scoring_providers``).

``RunpodClient`` speaks the SAME duck-typed client interface that
``cloud.TargonOrchestrator`` drives (``ensure_ssh_key``, ``check_availability``,
``create_workload``, ``deploy_workload``, ``wait_running``,
``get_workload_state``, ``delete_workload``, ``list_workloads``) plus the
optional ``ssh_base_cmd`` / ``cost_per_hour`` hooks, so the orchestrator —
spend cap, project upload, box probe, shard staging, detached runs, audit
trail, teardown — is reused unchanged:

    orch = TargonOrchestrator(client=RunpodClient(), resource="b200-small", ...)

API: the REST API **v2** (``https://api.runpod.io/v2``, Bearer auth). v1
(``rest.runpod.io/v1``) is deprecated and retires on 2026-11-15; the GraphQL
API is not used. No SDK dependency — stdlib urllib only, like the other
provider clients.

SSH: RunPod injects the account's registered public keys (``PUT
/v2/account/ssh-keys`` — a full REPLACE, so this client re-sends the existing
keys plus ours) into pods created with ``startSsh``; with ``22/tcp`` in
``ports`` the pod reports a *direct* endpoint (``host``/``port``/``username``)
that supports scp/rsync, which is what the orchestrator's tar-over-ssh
upload needs. The proxy endpoint (``ssh.runpod.io``) is shell-only and is
not used.

Resource naming: canonical Targon-style SKUs map to (gpu type id, count)
via ``SKU_GPU_MAP`` (``b200-small`` → 1× ``NVIDIA B200``); a raw RunPod GPU
type id (``NVIDIA ...``) passes through as a 1-GPU pod. ``is_b200_sku`` keeps
working for both.

Auto-termination: RunPod has no platform-side kill timer. The backstops are
the orchestrator's teardown + startup orphan sweep (deletes all ``sn125-*``
pods) and the periodic staleness sweep fed by ``createdAt``.

Credentials: ``RUNPOD_API_KEY`` env var, falling back to a chmod-600 file at
``~/.sn125/runpod_api_key``. NEVER hardcode the key in this file.

Knobs: ``SN125_RUNPOD_IMAGE`` (container image; default a CUDA 12.8 / torch
2.8 RunPod PyTorch image so ``setup_b200.sh`` does not reinstall torch),
``SN125_RUNPOD_CLOUD`` (``SECURE`` default; ``COMMUNITY`` is cheaper and
untrusted), ``SN125_RUNPOD_DATACENTERS`` (comma-separated preference),
``SN125_RUNPOD_DISK_GB`` (container disk, default 200 — the shards need
~35 GB plus checkpoints), ``SN125_RUNPOD_MIN_CUDA`` (default ``12.8``),
``SN125_RUNPOD_STARTUP_TIMEOUT``.
"""
import json
import logging
import os
import stat
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .cloud import SKU_COSTS, SSH_CONNECT_TIMEOUT, SSH_KEY_DIR, SSH_KEY_NAME
from .env import load_dotenv

log = logging.getLogger("sn125.cloud_runpod")

RUNPOD_BASE = "https://api.runpod.io/v2"
USER_AGENT = "sn125-validator/1.0 (+https://github.com/Barbariandev)"
RUNPOD_SSH_KEY_PATH = SSH_KEY_DIR / "runpod_id_ed25519"
RUNPOD_STARTUP_TIMEOUT = int(os.environ.get("SN125_RUNPOD_STARTUP_TIMEOUT", "900"))
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_DISK_GB = 200
DEFAULT_MIN_CUDA = "12.8"

SKU_GPU_MAP = {
    "b200-small": ("NVIDIA B200", 1),
    "b200-medium": ("NVIDIA B200", 2),
    "b200-large": ("NVIDIA B200", 4),
    "b200-xlarge": ("NVIDIA B200", 8),
    "h100-small": ("NVIDIA H100 80GB HBM3", 1),
    "h100-xlarge": ("NVIDIA H100 80GB HBM3", 8),
    "h200-small": ("NVIDIA H200", 1),
    "a100-small": ("NVIDIA A100-SXM4-80GB", 1),
}

_STATUS_MAP = {
    "PROVISIONING": "PROVISIONING",
    "STARTING": "PROVISIONING",
    "RUNNING": "RUNNING",
    "EXITED": "FAILED",
    "ERROR": "FAILED",
    "TERMINATED": "DELETED",
}
_TERMINAL_STATUSES = frozenset({"FAILED", "DELETED"})
_AVAILABLE_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})


def _load_runpod_api_key() -> str:
    """Resolve the RunPod API key: env var → ~/.sn125/runpod_api_key (must be
    chmod 600). Returns "" when nothing is configured; the client constructor
    fails loudly then."""
    load_dotenv()
    env_val = os.environ.get("RUNPOD_API_KEY", "").strip()
    if env_val:
        return env_val
    key_file = Path.home() / ".sn125" / "runpod_api_key"
    if key_file.exists():
        st = key_file.stat()
        if st.st_mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"{key_file} has insecure permissions (mode={oct(st.st_mode & 0o777)}). "
                f"Run: chmod 600 {key_file}")
        return key_file.read_text().strip()
    return ""


def resolve_resource(resource: str) -> tuple[str, int]:
    """Translate a resource string into (RunPod GPU type id, gpu count).

    Accepts the canonical Targon-style SKUs (via SKU_GPU_MAP) and raw RunPod
    GPU type ids (``NVIDIA ...``) as a single-GPU pod.
    """
    if resource in SKU_GPU_MAP:
        return SKU_GPU_MAP[resource]
    return resource.strip(), 1


def _cloud_tier() -> str:
    tier = os.environ.get("SN125_RUNPOD_CLOUD", "SECURE").strip().upper() or "SECURE"
    if tier not in ("SECURE", "COMMUNITY"):
        raise ValueError(f"SN125_RUNPOD_CLOUD must be SECURE or COMMUNITY, got {tier!r}")
    return tier


def _preferred_datacenters() -> list[str]:
    raw = os.environ.get("SN125_RUNPOD_DATACENTERS", "")
    return [r.strip() for r in raw.split(",") if r.strip()]


class RunpodClient:
    """Thin HTTP wrapper around the RunPod REST API v2. No SDK dependency."""

    provider_name = "runpod"
    provider_isolation = "container"

    def __init__(self, api_key: str = "", base_url: str = RUNPOD_BASE,
                 key_path: str = "", image: str = "", cloud: str = ""):
        self.api_key = (api_key or _load_runpod_api_key()).strip()
        if not self.api_key:
            raise RuntimeError(
                "No RunPod API key found. Set RUNPOD_API_KEY env var or write "
                "the key to ~/.sn125/runpod_api_key (chmod 600).")
        self.base = base_url.rstrip("/")
        self.key_path = Path(key_path) if key_path else RUNPOD_SSH_KEY_PATH
        self.image = image or os.environ.get("SN125_RUNPOD_IMAGE", "").strip() or DEFAULT_IMAGE
        self.cloud = (cloud or _cloud_tier()).upper()
        self._ssh_dest: dict[str, tuple[str, int]] = {}
        self._gpu_cache: dict[str, tuple[dict, float]] = {}
        self._gpu_ttl_s = 60.0

    def _req(self, method: str, path: str, body: dict = None,
             timeout: int = 30, query: dict = None):
        """Make an HTTP request, return the parsed JSON body ({} for empty)."""
        url = f"{self.base}{path}"
        if query:
            url += "?" + urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}",
                   "User-Agent": USER_AGENT}
        req = Request(url, data=data, headers=headers, method=method)
        for attempt in range(4):
            try:
                with urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    if not raw or resp.status in (202, 204):
                        return {}
                    return json.loads(raw)
            except HTTPError as e:
                if (e.code == 429 or e.code >= 500) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                body_text = e.read().decode()[:500] if hasattr(e, "read") else str(e)
                raise RuntimeError(f"RunPod {method} {path}: {e.code} {body_text}") from e
            except URLError as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"RunPod {method} {path}: {e}") from e
        raise RuntimeError(f"RunPod {method} {path}: retries exhausted")

    def get_gpu_type(self, gpu_type_id: str, count: int = 1) -> dict:
        """GET /catalog/gpus/{id} with POD availability for this cloud tier,
        cached ~1 min. Returns {} for an unknown type."""
        key = f"{gpu_type_id}|{count}|{self.cloud}"
        hit = self._gpu_cache.get(key)
        if hit and time.time() - hit[1] < self._gpu_ttl_s:
            return hit[0]
        try:
            data = self._req("GET", f"/catalog/gpus/{quote(gpu_type_id, safe='')}",
                             query={"include": "AVAILABILITY", "product": "POD",
                                    "count": int(count), "cloud": self.cloud})
        except RuntimeError as e:
            if " 404 " in str(e):
                data = {}
            else:
                raise
        if not isinstance(data, dict):
            data = {}
        self._gpu_cache[key] = (data, time.time())
        return data

    def check_availability(self, resource_name: str) -> int:
        """Number of data centers (or 1) that can serve this resource right
        now on the configured cloud tier; 0 when the catalog says NONE."""
        gpu_id, count = resolve_resource(resource_name)
        entry = self.get_gpu_type(gpu_id, count)
        if not entry:
            return 0
        tier_ok = entry.get("secure" if self.cloud == "SECURE" else "community", True)
        if tier_ok is False:
            return 0
        level = str(entry.get("availability") or "NONE").upper()
        if level not in _AVAILABLE_LEVELS:
            return 0
        dcs = [dc for dc in (entry.get("dataCenters") or [])
               if str(dc.get("availability", "NONE")).upper() in _AVAILABLE_LEVELS]
        pref = _preferred_datacenters()
        if pref and dcs:
            dcs = [dc for dc in dcs if dc.get("id") in pref]
            return len(dcs)
        return max(1, len(dcs))

    def cost_per_hour(self, resource_name: str) -> float:
        """Live list $/h for the whole pod (per-GPU catalog price × count on
        the configured tier). Falls back to the static SKU table so spend
        accounting never reads $0 during an API blip."""
        gpu_id, count = resolve_resource(resource_name)
        try:
            entry = self.get_gpu_type(gpu_id, count)
            price = (entry.get("price") or {}).get("secure" if self.cloud == "SECURE"
                                                    else "community")
            if price:
                return float(price) * int(count)
        except Exception as e:
            log.warning(f"live price lookup failed for {resource_name!r}: {e}")
        return SKU_COSTS.get(resource_name, 7.0)

    @staticmethod
    def _key_material(pubkey: str) -> str:
        """type + base64 blob of an OpenSSH public key (comment stripped)."""
        return " ".join((pubkey or "").split()[:2])

    def list_ssh_keys(self) -> list[str]:
        resp = self._req("GET", "/account/ssh-keys")
        keys = resp.get("keys") if isinstance(resp, dict) else None
        return [k for k in (keys or []) if isinstance(k, str)]

    def ensure_ssh_key(self) -> str:
        """Generate the local keypair if needed and make sure it is among the
        account's registered keys (PUT replaces the whole set, so the existing
        keys are re-sent alongside ours). Returns the key material."""
        SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.key_path.exists():
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(self.key_path),
                 "-N", "", "-C", SSH_KEY_NAME],
                check=True, capture_output=True)
            log.info(f"Generated SSH keypair at {self.key_path}")
        pubkey = self.key_path.with_suffix(".pub").read_text().strip()
        material = self._key_material(pubkey)
        existing = self.list_ssh_keys()
        if any(self._key_material(k) == material for k in existing):
            log.info("SSH key already registered with RunPod")
            return material
        self._req("PUT", "/account/ssh-keys", {"keys": existing + [pubkey]})
        log.info(f"Registered SSH key with RunPod ({len(existing) + 1} key(s) total)")
        return material

    def create_workload(self, name: str, resource_name: str, ssh_key_uid: str,
                        image: str = "", envs: list[dict] = None,
                        args: list[str] = None) -> dict:
        """Create one pod for this resource. Returns {"uid": pod_id, ...}.

        ``envs`` ([{"name", "value"}, ...], Targon-shaped) become the pod's
        environment; ``args`` are ignored (the eval command travels over SSH).
        ``ssh_key_uid`` is informational — RunPod injects the account keys via
        ``startSsh``.
        """
        gpu_id, count = resolve_resource(resource_name)
        env = {}
        for item in envs or []:
            if isinstance(item, dict) and item.get("name"):
                env[str(item["name"])] = str(item.get("value", ""))
        body = {
            "name": name,
            "image": image or self.image,
            "cloud": self.cloud,
            "gpu": {"id": gpu_id, "count": int(count),
                    "minCudaVersion": os.environ.get("SN125_RUNPOD_MIN_CUDA", DEFAULT_MIN_CUDA)},
            "disk": int(os.environ.get("SN125_RUNPOD_DISK_GB", str(DEFAULT_DISK_GB))),
            "ports": ["22/tcp"],
            "startSsh": True,
            "env": env,
        }
        dcs = _preferred_datacenters()
        if dcs:
            body["dataCenterIds"] = dcs
        resp = self._req("POST", "/pods", body, timeout=90)
        pod_id = str((resp or {}).get("id", "") or "")
        if not pod_id:
            raise RuntimeError(f"RunPod create returned no pod id: {resp}")
        log.info(f"Created RunPod pod {pod_id} ('{name}', {count}x {gpu_id}, {self.cloud})")
        return {"uid": pod_id, "name": name, "gpu_type_id": gpu_id, "cloud": self.cloud}

    def deploy_workload(self, wrk_uid: str) -> dict:
        """No-op: a RunPod create both registers and starts the pod."""
        return {}

    def get_pod(self, pod_id: str) -> dict:
        pod = self._req("GET", f"/pods/{quote(str(pod_id), safe='')}")
        return pod if isinstance(pod, dict) else {}

    def list_pods(self) -> list[dict]:
        resp = self._req("GET", "/pods")
        if isinstance(resp, list):
            return [p for p in resp if isinstance(p, dict)]
        if isinstance(resp, dict):
            items = resp.get("pods") or resp.get("items") or resp.get("data") or []
            return [p for p in items if isinstance(p, dict)]
        return []

    def _cache_ssh_dest(self, pod_id: str, pod: dict) -> None:
        direct = ((pod.get("ssh") or {}).get("direct")) if isinstance(pod.get("ssh"), dict) else None
        if isinstance(direct, dict) and direct.get("host") and direct.get("port"):
            user = direct.get("username") or "root"
            try:
                port = int(direct["port"])
            except (TypeError, ValueError):
                return
            self._ssh_dest[str(pod_id)] = (f"{user}@{direct['host']}", port)

    def get_workload_state(self, wrk_uid: str) -> dict:
        """Current pod state in the Targon-shaped {"status": ...} form."""
        try:
            pod = self.get_pod(wrk_uid)
        except RuntimeError as e:
            if " 404 " in str(e):
                return {"status": "DELETED"}
            raise
        self._cache_ssh_dest(wrk_uid, pod)
        raw = str(pod.get("status", "")).upper()
        return {"status": _STATUS_MAP.get(raw, raw or "UNKNOWN"),
                "name": pod.get("name") or "",
                "ssh_ready": str(wrk_uid) in self._ssh_dest}

    def wait_running(self, wrk_uid: str, timeout: int = RUNPOD_STARTUP_TIMEOUT,
                     poll_interval: int = 10) -> bool:
        """Poll until the pod is RUNNING with a direct SSH endpoint, then verify
        root SSH answers and /workspace exists (the orchestrator's contract).
        False on timeout or a terminal state."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get_workload_state(wrk_uid)
            status = state.get("status", "UNKNOWN")
            if status in _TERMINAL_STATUSES:
                log.error(f"Pod {wrk_uid} entered terminal state: {status}")
                return False
            if status == "RUNNING" and state.get("ssh_ready"):
                return self._verify_ssh(wrk_uid, deadline)
            time.sleep(poll_interval)
        log.error(f"Pod {wrk_uid} did not become reachable within {timeout}s")
        return False

    def _verify_ssh(self, wrk_uid: str, deadline: float) -> bool:
        while time.time() < deadline:
            r = subprocess.run(self.ssh_base_cmd(wrk_uid) + ["mkdir -p /workspace && echo rootok"],
                               capture_output=True, timeout=120)
            if r.returncode == 0 and b"rootok" in r.stdout:
                log.info(f"Pod {wrk_uid} reachable over direct SSH")
                return True
            time.sleep(10)
        log.error(f"Pod {wrk_uid}: sshd never answered before the deadline")
        return False

    def delete_workload(self, wrk_uid: str) -> None:
        try:
            self._req("DELETE", f"/pods/{quote(str(wrk_uid), safe='')}")
        except Exception as e:
            if " 404 " not in str(e):
                log.warning(f"Failed to terminate pod {wrk_uid}: {e}")
        self._ssh_dest.pop(str(wrk_uid), None)

    def list_workloads(self, status: str = None) -> list[dict]:
        """Pods mapped into the Targon workload shape the orchestrator's
        orphan/staleness sweeps expect: name, uid, created_at, state.status."""
        out = []
        for pod in self.list_pods():
            st = _STATUS_MAP.get(str(pod.get("status", "")).upper(), "UNKNOWN")
            if status and st != status.upper():
                continue
            out.append({
                "name": pod.get("name") or "",
                "uid": str(pod.get("id", "")),
                "created_at": pod.get("createdAt") or "",
                "state": {"status": st},
            })
        return out

    def ssh_base_cmd(self, wrk_uid: str) -> list[str]:
        """Base SSH command for a pod — the direct ``22/tcp`` mapping with the
        locally generated key that ensure_ssh_key registered."""
        dest = self._ssh_dest.get(str(wrk_uid))
        if dest is None:
            self._cache_ssh_dest(wrk_uid, self.get_pod(wrk_uid))
            dest = self._ssh_dest.get(str(wrk_uid))
        if dest is None:
            raise RuntimeError(f"no direct SSH endpoint for RunPod pod {wrk_uid}")
        user_host, port = dest
        return ["ssh", "-i", str(self.key_path),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                "-p", str(port), user_host]
