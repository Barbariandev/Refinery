"""
Lium cloud client for the SN125 validator — the Targon-alternative provider.

Lium (lium.io) is the GPU rental marketplace on Bittensor Subnet 51. This
module provides ``LiumClient``, which speaks the SAME duck-typed client
interface that ``cloud.TargonOrchestrator`` drives (``ensure_ssh_key``,
``check_availability``, ``create_workload``, ``deploy_workload``,
``wait_running``, ``get_workload_state``, ``delete_workload``,
``list_workloads``), so the whole orchestrator — spend cap, egress lockdown,
detached runs, audit trail, teardown — is reused unchanged:

    orch = TargonOrchestrator(client=LiumClient(), resource="b200-small", ...)

Two extra hooks the orchestrator consults when present:

* ``ssh_base_cmd(uid)`` — Lium pods are reached directly (``root@ip -p port``
  from the pod's ``ssh_connect_cmd``), not through a shared SSH proxy.
* ``cost_per_hour(resource)`` — live per-executor pricing (price_per_gpu ×
  gpu_count) instead of the static Targon SKU table.

Resource naming: the orchestrator keeps using the canonical Targon-style SKU
strings (``b200-small`` = 1×B200 ...), which this client translates to a Lium
machine name + GPU count (see ``SKU_MACHINE_MAP``). That keeps ``is_b200_sku``
and every audit/status field working identically across providers. A raw Lium
machine name ("NVIDIA B200" or "NVIDIA B200:4") is also accepted.

Credentials: LIUM_API_KEY env var, falling back to a chmod-600 file at
~/.sn125/lium_api_key, then to ~/.lium/config.ini ([api] api_key — written by
``lium init``). NEVER hardcode the key in this file.

API: https://lium.io/api (OpenAPI: https://lium.io/api/openapi.json), auth via
the ``X-API-Key`` header. No SDK dependency — stdlib urllib only, mirroring
TargonClient.
"""
import configparser
import json
import logging
import math
import os
import shlex
import stat
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .cloud import SKU_COSTS, SSH_CONNECT_TIMEOUT, SSH_KEY_DIR, SSH_KEY_NAME
from .env import load_dotenv

log = logging.getLogger("sn125.cloud_lium")

LIUM_BASE = "https://lium.io/api"
LIUM_SSH_KEY_PATH = SSH_KEY_DIR / "lium_id_ed25519"

LIUM_STARTUP_TIMEOUT = int(os.environ.get("SN125_LIUM_STARTUP_TIMEOUT", "1500"))

SKU_MACHINE_MAP = {
    "b200-small": ("NVIDIA B200", 1),
    "b200-medium": ("NVIDIA B200", 2),
    "b200-large": ("NVIDIA B200", 4),
    "b200-xlarge": ("NVIDIA B200", 8),
    "h200-small": ("NVIDIA H200", 1),
    "h100-small": ("NVIDIA H100 80GB HBM3", 1),
}

_TERMINAL_STATUSES = frozenset({
    "FAILED", "CREATION_FAILED", "BROKEN", "DELETING", "DELETED", "STOPPED",
})

DEFAULT_IMAGE_PREFIX = os.environ.get("SN125_LIUM_IMAGE", "daturaai/pytorch")


def _load_lium_api_key() -> str:
    """Resolve the Lium API key: env var → ~/.sn125/lium_api_key (must be
    chmod 600) → ~/.lium/config.ini (written by ``lium init``). Returns ""
    when nothing is configured; the client constructor fails loudly then."""
    load_dotenv()
    env_val = os.environ.get("LIUM_API_KEY", "").strip()
    if env_val:
        return env_val

    key_file = Path.home() / ".sn125" / "lium_api_key"
    if key_file.exists():
        st = key_file.stat()
        if st.st_mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"{key_file} has insecure permissions (mode={oct(st.st_mode & 0o777)}). "
                f"Run: chmod 600 {key_file}")
        val = key_file.read_text().strip()
        if val:
            return val

    ini = Path.home() / ".lium" / "config.ini"
    if ini.exists():
        cp = configparser.ConfigParser()
        try:
            cp.read(ini)
            val = cp.get("api", "api_key", fallback="").strip()
            if val:
                return val
        except configparser.Error:
            pass
    return ""


def resolve_resource(resource: str) -> tuple[str, int]:
    """Translate a resource string into (Lium machine name, gpu_count).

    Accepts the canonical Targon-style SKUs (via SKU_MACHINE_MAP) and raw Lium
    machine names, optionally suffixed ``:N`` for a GPU count.
    """
    if resource in SKU_MACHINE_MAP:
        return SKU_MACHINE_MAP[resource]
    name, _, cnt = resource.rpartition(":")
    if name and cnt.strip().isdigit():
        return name.strip(), max(1, int(cnt))
    return resource.strip(), 1


def termination_hours_for(timeout_s: int, margin_s: int = 600) -> int:
    """Platform-side auto-termination for a rental serving one eval of
    ``timeout_s``: the eval budget + orchestrator margin + 1h slack, clamped
    to Lium's accepted [1, 720] range. This is the backstop that kills the
    pod even if the validator process dies mid-round."""
    hours = math.ceil((timeout_s + margin_s) / 3600) + 1
    return max(1, min(720, hours))


class LiumClient:
    """Thin HTTP wrapper around the Lium platform API. No SDK dependency."""

    def __init__(self, api_key: str = "", base_url: str = LIUM_BASE,
                 key_path: str = "", termination_hours: int = 25):
        self.api_key = (api_key or _load_lium_api_key()).strip()
        if not self.api_key:
            raise RuntimeError(
                "No Lium API key found. Set LIUM_API_KEY env var, write the key "
                "to ~/.sn125/lium_api_key (chmod 600), or run `lium init`.")
        self.base = base_url.rstrip("/")
        self.key_path = Path(key_path) if key_path else LIUM_SSH_KEY_PATH
        self.termination_hours = termination_hours
        self._pubkey = ""
        self._template_id = ""
        self._ssh_dest: dict[str, tuple[str, int]] = {}
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._price_ttl_s = 600.0

    def _req(self, method: str, path: str, body: dict = None,
             params: dict = None, timeout: int = 30):
        """Make an HTTP request, return parsed JSON (or {} for empty bodies)."""
        url = f"{self.base}{path}"
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
        req = Request(url, data=data, headers=headers, method=method)
        for attempt in range(4):
            try:
                with urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    if not raw or resp.status in (202, 204):
                        return {}
                    return json.loads(raw)
            except HTTPError as e:
                if e.code >= 500 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                body_text = e.read().decode()[:500] if hasattr(e, "read") else str(e)
                raise RuntimeError(f"Lium {method} {path}: {e.code} {body_text}") from e
            except URLError as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Lium {method} {path}: {e}") from e
        return {}

    def list_executors(self, machine_name: str = "") -> list[dict]:
        """List rentable executors, optionally filtered to one machine name."""
        params = {"machine_names": machine_name} if machine_name else None
        resp = self._req("GET", "/executors", params=params)
        return resp if isinstance(resp, list) else []

    @staticmethod
    def _rentable(executors: list[dict], gpu_count: int) -> list[dict]:
        """Executors that can serve a ``gpu_count``-GPU rental right now,
        cheapest first (ties broken by reliability)."""
        out = []
        for e in executors:
            avail = e.get("available_gpu_count")
            if avail is None:
                avail = e.get("gpu_count") or 0
            min_gpus = e.get("min_gpu_count_for_rental") or 1
            if avail >= gpu_count and min_gpus <= gpu_count:
                out.append(e)
        out.sort(key=lambda e: (float(e.get("price_per_gpu") or 1e9),
                                -float(e.get("reliability_score") or 0.0)))
        return out

    def check_availability(self, resource_name: str) -> int:
        """Return the number of executors that could serve one rental of this
        resource (the orchestrator only needs >= 1)."""
        machine, gpu_count = resolve_resource(resource_name)
        return len(self._rentable(self.list_executors(machine), gpu_count))

    def cost_per_hour(self, resource_name: str) -> float:
        """Live $/h for one rental of this resource: the cheapest rentable
        executor's price_per_gpu × gpu_count, cached ~10 min (heartbeats and
        status writes ask constantly). Falls back to the static Targon SKU
        table (same GPU class, historically identical rates) so spend
        accounting never reads $0 during an API blip."""
        cached = self._price_cache.get(resource_name)
        if cached and time.time() - cached[1] < self._price_ttl_s:
            return cached[0]
        machine, gpu_count = resolve_resource(resource_name)
        price = None
        try:
            candidates = self._rentable(self.list_executors(machine), gpu_count)
            prices = [float(e["price_per_gpu"]) for e in candidates
                      if e.get("price_per_gpu")]
            if prices:
                price = min(prices) * gpu_count
        except Exception as e:
            log.warning(f"live price lookup failed for {resource_name!r}: {e}")
        if price is None:
            price = SKU_COSTS.get(resource_name, 5.3 * gpu_count)
        self._price_cache[resource_name] = (price, time.time())
        return price

    def list_ssh_keys(self) -> list[dict]:
        resp = self._req("GET", "/ssh-keys")
        return resp if isinstance(resp, list) else resp.get("items", [])

    def register_ssh_key(self, name: str, pubkey: str) -> str:
        resp = self._req("POST", "/ssh-keys", {"name": name, "public_key": pubkey})
        return str(resp.get("id", ""))

    @staticmethod
    def _key_material(pubkey: str) -> str:
        """type + base64 blob of an OpenSSH public key (comment stripped),
        for equality checks against registered keys."""
        return " ".join(pubkey.split()[:2])

    def ensure_ssh_key(self) -> str:
        """Generate the local keypair if needed, register it with Lium, return
        the key's id. Also caches the pubkey text used in every rent request."""
        SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.key_path.exists():
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(self.key_path),
                 "-N", "", "-C", SSH_KEY_NAME],
                check=True, capture_output=True)
            log.info(f"Generated SSH keypair at {self.key_path}")
        self._pubkey = self.key_path.with_suffix(".pub").read_text().strip()
        material = self._key_material(self._pubkey)
        for key in self.list_ssh_keys():
            if self._key_material(key.get("public_key", "")) == material:
                log.info(f"SSH key already registered with Lium: {key.get('id')}")
                return str(key.get("id", ""))
        kid = self.register_ssh_key(SSH_KEY_NAME, self._pubkey)
        log.info(f"Registered SSH key '{SSH_KEY_NAME}' with Lium: {kid}")
        return kid

    def _resolve_template_id(self) -> str:
        """The docker template a rent request needs. Operators pin one via
        SN125_LIUM_TEMPLATE_ID; otherwise pick the newest official template
        whose image matches DEFAULT_IMAGE_PREFIX (daturaai/pytorch keeps the
        pod alive with SSH — the harness installs its own torch on top)."""
        pinned = os.environ.get("SN125_LIUM_TEMPLATE_ID", "").strip()
        if pinned:
            return pinned
        if self._template_id:
            return self._template_id
        templates = self._req("GET", "/templates")
        if not isinstance(templates, list):
            templates = []
        matches = [t for t in templates
                   if str(t.get("docker_image", "")).startswith(DEFAULT_IMAGE_PREFIX)]
        if not matches:
            raise RuntimeError(
                f"no Lium template with image prefix {DEFAULT_IMAGE_PREFIX!r}; "
                "set SN125_LIUM_TEMPLATE_ID to a template uuid from GET /templates")
        matches.sort(key=lambda t: str(t.get("docker_image_tag", "")), reverse=True)
        self._template_id = str(matches[0]["id"])
        log.info(f"Using Lium template {self._template_id} "
                 f"({matches[0].get('docker_image')}:{matches[0].get('docker_image_tag')})")
        return self._template_id

    def create_workload(self, name: str, resource_name: str, ssh_key_uid: str,
                        image: str = "", envs: list[dict] = None,
                        args: list[str] = None) -> dict:
        """Rent an executor for this resource. Returns {"uid": pod_id, ...}.

        ``image``/``envs``/``args`` are part of the Targon-shaped interface;
        Lium pods boot from a template instead (image → template resolution)
        and eval env vars travel with the detached run command, so envs/args
        are intentionally ignored here.
        """
        machine, gpu_count = resolve_resource(resource_name)
        candidates = self._rentable(self.list_executors(machine), gpu_count)
        if not candidates:
            raise RuntimeError(
                f"no rentable Lium executor for {resource_name!r} "
                f"({machine} x{gpu_count})")
        if not self._pubkey:
            self.ensure_ssh_key()
        body = {
            "pod_name": name,
            "template_id": self._resolve_template_id(),
            "gpu_count": gpu_count,
            "user_public_key": [self._pubkey],
            "termination_hours": self.termination_hours,
        }
        last_err: Exception | None = None
        for executor in candidates[:3]:
            ex_id = executor["id"]
            try:
                resp = self._req("POST", f"/executors/{quote(str(ex_id))}/rent",
                                 body, timeout=60)
            except Exception as e:
                last_err = e
                log.warning(f"rent on executor {ex_id} failed: {e}")
                continue
            pod_id = self._extract_pod_id(resp) or self._find_pod_id_by_name(name)
            if pod_id:
                log.info(f"Rented Lium executor {ex_id} → pod {pod_id} ('{name}', "
                         f"{machine} x{gpu_count}, ${executor.get('price_per_gpu')}/gpu/h)")
                return {"uid": pod_id, "name": name, "executor_id": str(ex_id)}
            last_err = RuntimeError(f"rent accepted on {ex_id} but pod '{name}' "
                                    f"not visible in GET /pods")
        raise RuntimeError(f"Lium rent failed for {resource_name!r}: {last_err}")

    @staticmethod
    def _extract_pod_id(resp) -> str:
        """Pull a pod uuid out of whatever shape the rent endpoint returns."""
        if not isinstance(resp, dict):
            return ""
        for k in ("id", "pod_id", "uid"):
            if resp.get(k):
                return str(resp[k])
        pod = resp.get("pod")
        if isinstance(pod, dict) and pod.get("id"):
            return str(pod["id"])
        return ""

    def _find_pod_id_by_name(self, name: str, wait_s: int = 90) -> str:
        """Resolve a freshly rented pod's id by its (unique) pod_name."""
        deadline = time.time() + wait_s
        while True:
            try:
                for pod in self.list_pods():
                    if pod.get("pod_name") == name:
                        return str(pod.get("id", ""))
            except Exception as e:
                log.warning(f"pod-by-name lookup failed: {e}")
            if time.time() >= deadline:
                return ""
            time.sleep(5)

    def list_pods(self) -> list[dict]:
        resp = self._req("GET", "/pods")
        return resp if isinstance(resp, list) else []

    def get_pod(self, pod_id: str) -> dict:
        pod = self._req("GET", f"/pods/{quote(str(pod_id))}")
        return pod if isinstance(pod, dict) else {}

    def deploy_workload(self, wrk_uid: str) -> dict:
        """No-op: a Lium rent both registers and deploys the pod."""
        return {}

    def get_workload_state(self, wrk_uid: str) -> dict:
        """Current pod state in the Targon-shaped {"status": ...} form."""
        try:
            pod = self.get_pod(wrk_uid)
        except RuntimeError as e:
            if " 404 " in str(e):
                return {"status": "DELETED"}
            raise
        self._cache_ssh_dest(wrk_uid, pod)
        return {"status": str(pod.get("status", "UNKNOWN")).upper(),
                "pod_name": pod.get("pod_name", ""),
                "updated_at": pod.get("updated_at", "")}

    def wait_running(self, wrk_uid: str, timeout: int = LIUM_STARTUP_TIMEOUT,
                     poll_interval: int = 10) -> bool:
        """Poll until pod status == RUNNING. Returns False on timeout or a
        terminal status (image-build failure, broken host, ...)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.get_workload_state(wrk_uid).get("status", "UNKNOWN")
            if status == "RUNNING":
                return True
            if status in _TERMINAL_STATUSES:
                log.error(f"Pod {wrk_uid} entered terminal state: {status}")
                return False
            time.sleep(poll_interval)
        log.error(f"Pod {wrk_uid} did not reach RUNNING within {timeout}s")
        return False

    def delete_workload(self, wrk_uid: str) -> None:
        try:
            self._req("DELETE", f"/pods/{quote(str(wrk_uid))}")
        except Exception as e:
            log.warning(f"Failed to delete pod {wrk_uid}: {e}")

    def list_workloads(self, status: str = None) -> list[dict]:
        """Pods mapped into the Targon workload shape the orchestrator's
        orphan/staleness sweeps expect: name, uid, created_at, state.status."""
        out = []
        for pod in self.list_pods():
            st = str(pod.get("status", "UNKNOWN")).upper()
            if status and st != status.upper():
                continue
            out.append({
                "name": pod.get("pod_name", ""),
                "uid": str(pod.get("id", "")),
                "created_at": pod.get("created_at", ""),
                "state": {"status": st},
            })
        return out

    def _cache_ssh_dest(self, pod_id: str, pod: dict) -> None:
        dest = self._parse_ssh_connect_cmd(pod.get("ssh_connect_cmd", ""))
        if dest:
            self._ssh_dest[str(pod_id)] = dest

    @staticmethod
    def _parse_ssh_connect_cmd(cmd: str) -> tuple[str, int] | None:
        """Parse a pod's ssh_connect_cmd ("ssh root@1.2.3.4 -p 40001", flag
        order varies) into ("user@host", port)."""
        try:
            tokens = shlex.split(cmd or "")
        except ValueError:
            tokens = (cmd or "").split()
        user_host = ""
        port = 22
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "-p" and i + 1 < len(tokens):
                try:
                    port = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if "@" in tok and not tok.startswith("-"):
                user_host = tok
            i += 1
        return (user_host, port) if user_host else None

    def ssh_base_cmd(self, wrk_uid: str) -> list[str]:
        """Base SSH command for a pod — direct to the pod's mapped port, with
        the Lium-registered private key."""
        dest = self._ssh_dest.get(str(wrk_uid))
        if dest is None:
            self._cache_ssh_dest(wrk_uid, self.get_pod(wrk_uid))
            dest = self._ssh_dest.get(str(wrk_uid))
        if dest is None:
            raise RuntimeError(f"no SSH connect info for Lium pod {wrk_uid}")
        user_host, port = dest
        return ["ssh", "-i", str(self.key_path),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                "-p", str(port), user_host]
