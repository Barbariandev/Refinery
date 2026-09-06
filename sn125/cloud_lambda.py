"""
Lambda Cloud client for the SN125 validator — VM-based backup provider.

Lambda (lambda.ai, formerly Lambda Labs) rents whole GPU VMs, not containers.
That is exactly why it is in the provider set: the sandbox needs unshare
namespaces and the egress lockdown needs iptables/CAP_NET_ADMIN, and on a VM
with root those can never silently regress under us the way they did on
Targon's container fleet.

``LambdaClient`` speaks the SAME duck-typed client interface that
``cloud.TargonOrchestrator`` drives (``ensure_ssh_key``, ``check_availability``,
``create_workload``, ``deploy_workload``, ``wait_running``,
``get_workload_state``, ``delete_workload``, ``list_workloads``), so the whole
orchestrator — spend cap, egress lockdown, detached runs, audit trail,
teardown — is reused unchanged:

    orch = TargonOrchestrator(client=LambdaClient(), resource="b200-small", ...)

Plus the two optional orchestrator hooks:

* ``ssh_base_cmd(uid)`` — instances are reached directly at ``root@ip:22``.
  Lambda VMs boot with SSH for ``ubuntu`` only and remote commands in cloud.py
  assume root + /workspace (the shape Targon/Lium containers provide), so
  ``wait_running`` bootstraps root's authorized_keys and /workspace via sudo
  before reporting ready.
* ``cost_per_hour(resource)`` — live pricing from /instance-types
  (price_cents_per_hour is for the whole instance).

Resource naming: the canonical Targon-style SKUs translate to Lambda instance
types (``b200-small`` → ``gpu_1x_b200_sxm6``, see SKU_INSTANCE_MAP); raw Lambda
names (``gpu_*``) pass through. ``is_b200_sku`` keeps working for both.

Auto-termination: Lambda has NO platform-side kill timer (Lium's
termination_hours). The backstops are the orchestrator's teardown +
startup orphan sweep (deletes ALL sn125-* instances), and this client stamps
a local launch time into ``list_workloads`` so the periodic staleness sweep
works too (Lambda's API reports no created_at).

Credentials: LAMBDA_API_KEY env var, falling back to a chmod-600 file at
~/.sn125/lambda_api_key. NEVER hardcode the key in this file.

API: https://cloud.lambdalabs.com/api/v1 (Bearer auth, responses enveloped in
{"data": ...}). No SDK dependency — stdlib urllib only, mirroring the other
provider clients.
"""
import json
import logging
import os
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .cloud import SKU_COSTS, SSH_CONNECT_TIMEOUT, SSH_KEY_DIR, SSH_KEY_NAME
from .env import load_dotenv

log = logging.getLogger("sn125.cloud_lambda")

LAMBDA_BASE = "https://cloud.lambdalabs.com/api/v1"
LAMBDA_SSH_KEY_PATH = SSH_KEY_DIR / "lambda_id_ed25519"

LAMBDA_STARTUP_TIMEOUT = int(os.environ.get("SN125_LAMBDA_STARTUP_TIMEOUT", "1500"))

SKU_INSTANCE_MAP = {
    "b200-small": "gpu_1x_b200_sxm6",
    "b200-medium": "gpu_2x_b200_sxm6",
    "b200-large": "gpu_4x_b200_sxm6",
    "b200-xlarge": "gpu_8x_b200_sxm6",
    "h100-small": "gpu_1x_h100_sxm5",
    "h100-xlarge": "gpu_8x_h100_sxm5",
}

_STATUS_MAP = {
    "active": "RUNNING",
    "booting": "PROVISIONING",
    "unhealthy": "FAILED",
    "terminating": "DELETED",
    "terminated": "DELETED",
}
_TERMINAL_STATUSES = frozenset({"FAILED", "DELETED"})

_ROOT_BOOTSTRAP = (
    "sudo install -d -m 700 -o root -g root /root/.ssh && "
    "sudo install -m 600 -o root -g root ~/.ssh/authorized_keys "
    "/root/.ssh/authorized_keys && "
    "sudo mkdir -p /workspace && echo bootstrapped"
)


def _load_lambda_api_key() -> str:
    """Resolve the Lambda API key: env var → ~/.sn125/lambda_api_key (must be
    chmod 600). Returns "" when nothing is configured; the client constructor
    fails loudly then."""
    load_dotenv()
    env_val = os.environ.get("LAMBDA_API_KEY", "").strip()
    if env_val:
        return env_val
    key_file = Path.home() / ".sn125" / "lambda_api_key"
    if key_file.exists():
        st = key_file.stat()
        if st.st_mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"{key_file} has insecure permissions (mode={oct(st.st_mode & 0o777)}). "
                f"Run: chmod 600 {key_file}")
        return key_file.read_text().strip()
    return ""


def resolve_resource(resource: str) -> str:
    """Translate a resource string into a Lambda instance type name.

    Accepts the canonical Targon-style SKUs (via SKU_INSTANCE_MAP) and raw
    Lambda instance type names (``gpu_...``) unchanged.
    """
    if resource in SKU_INSTANCE_MAP:
        return SKU_INSTANCE_MAP[resource]
    return resource.strip()


def _preferred_regions() -> list[str]:
    """Operator region preference (SN125_LAMBDA_REGIONS, comma-separated, in
    order). Empty means any region with capacity."""
    raw = os.environ.get("SN125_LAMBDA_REGIONS", "")
    return [r.strip() for r in raw.split(",") if r.strip()]


class LambdaClient:
    """Thin HTTP wrapper around the Lambda Cloud API. No SDK dependency."""

    def __init__(self, api_key: str = "", base_url: str = LAMBDA_BASE,
                 key_path: str = ""):
        self.api_key = (api_key or _load_lambda_api_key()).strip()
        if not self.api_key:
            raise RuntimeError(
                "No Lambda API key found. Set LAMBDA_API_KEY env var or write "
                "the key to ~/.sn125/lambda_api_key (chmod 600).")
        self.base = base_url.rstrip("/")
        self.key_path = Path(key_path) if key_path else LAMBDA_SSH_KEY_PATH
        self._key_name = ""
        self._ssh_ip: dict[str, str] = {}
        self._launched_at: dict[str, float] = {}
        self._types_cache: tuple[dict, float] | None = None
        self._types_ttl_s = 60.0

    def _req(self, method: str, path: str, body: dict = None,
             timeout: int = 30):
        """Make an HTTP request, return the JSON "data" payload (Lambda
        envelopes every response) or {} for empty bodies."""
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        req = Request(url, data=data, headers=headers, method=method)
        for attempt in range(4):
            try:
                with urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                    if not raw or resp.status in (202, 204):
                        return {}
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and "data" in parsed:
                        return parsed["data"]
                    return parsed
            except HTTPError as e:
                if e.code >= 500 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                body_text = e.read().decode()[:500] if hasattr(e, "read") else str(e)
                raise RuntimeError(f"Lambda {method} {path}: {e.code} {body_text}") from e
            except URLError as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Lambda {method} {path}: {e}") from e
        return {}

    def list_instance_types(self) -> dict:
        """GET /instance-types: {type_name: {instance_type, regions_with_capacity_available}},
        cached ~1 min (availability and pricing are polled constantly)."""
        if self._types_cache and time.time() - self._types_cache[1] < self._types_ttl_s:
            return self._types_cache[0]
        data = self._req("GET", "/instance-types")
        if not isinstance(data, dict):
            data = {}
        self._types_cache = (data, time.time())
        return data

    def _regions_with_capacity(self, resource_name: str) -> list[str]:
        """Region names that can serve this resource right now, honoring the
        operator's SN125_LAMBDA_REGIONS preference order when set."""
        itype = resolve_resource(resource_name)
        entry = self.list_instance_types().get(itype) or {}
        regions = [r.get("name", "") for r in
                   entry.get("regions_with_capacity_available") or []]
        regions = [r for r in regions if r]
        pref = _preferred_regions()
        if pref:
            regions = [r for r in pref if r in regions]
        return regions

    def check_availability(self, resource_name: str) -> int:
        """Number of regions that could serve one rental of this resource
        (the orchestrator only needs >= 1)."""
        return len(self._regions_with_capacity(resource_name))

    def cost_per_hour(self, resource_name: str) -> float:
        """Live $/h for one instance of this resource (price_cents_per_hour is
        for the whole instance, all GPUs included). Falls back to the static
        SKU table so spend accounting never reads $0 during an API blip."""
        itype = resolve_resource(resource_name)
        try:
            entry = self.list_instance_types().get(itype) or {}
            cents = (entry.get("instance_type") or {}).get("price_cents_per_hour")
            if cents:
                return float(cents) / 100.0
        except Exception as e:
            log.warning(f"live price lookup failed for {resource_name!r}: {e}")
        return SKU_COSTS.get(resource_name, 7.0)

    def list_ssh_keys(self) -> list[dict]:
        resp = self._req("GET", "/ssh-keys")
        return resp if isinstance(resp, list) else []

    @staticmethod
    def _key_material(pubkey: str) -> str:
        """type + base64 blob of an OpenSSH public key (comment stripped),
        for equality checks against registered keys."""
        return " ".join((pubkey or "").split()[:2])

    def ensure_ssh_key(self) -> str:
        """Generate the local keypair if needed and register it with Lambda.
        Returns the key NAME (Lambda launches reference keys by name)."""
        SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.key_path.exists():
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(self.key_path),
                 "-N", "", "-C", SSH_KEY_NAME],
                check=True, capture_output=True)
            log.info(f"Generated SSH keypair at {self.key_path}")
        pubkey = self.key_path.with_suffix(".pub").read_text().strip()
        material = self._key_material(pubkey)
        taken_names = set()
        for key in self.list_ssh_keys():
            taken_names.add(key.get("name", ""))
            if self._key_material(key.get("public_key", "")) == material:
                self._key_name = key.get("name", "")
                log.info(f"SSH key already registered with Lambda: '{self._key_name}'")
                return self._key_name
        name = SSH_KEY_NAME
        if name in taken_names:
            name = f"{SSH_KEY_NAME}-{material[-8:].replace('/', '_').replace('+', '-')}"
        self._req("POST", "/ssh-keys", {"name": name, "public_key": pubkey})
        self._key_name = name
        log.info(f"Registered SSH key '{name}' with Lambda")
        return name

    def create_workload(self, name: str, resource_name: str, ssh_key_uid: str,
                        image: str = "", envs: list[dict] = None,
                        args: list[str] = None) -> dict:
        """Launch one instance of this resource. Returns {"uid": instance_id, ...}.

        ``image``/``envs``/``args`` are part of the Targon-shaped interface;
        Lambda VMs boot a fixed Ubuntu image and eval env vars travel with the
        detached run command, so they are intentionally ignored here.
        ``ssh_key_uid`` is the key NAME from ensure_ssh_key().
        """
        regions = self._regions_with_capacity(resource_name)
        if not regions:
            raise RuntimeError(
                f"no Lambda capacity for {resource_name!r} "
                f"({resolve_resource(resource_name)})")
        key_name = ssh_key_uid or self._key_name
        if not key_name:
            key_name = self.ensure_ssh_key()
        last_err: Exception | None = None
        for region in regions[:3]:
            body = {
                "region_name": region,
                "instance_type_name": resolve_resource(resource_name),
                "ssh_key_names": [key_name],
                "quantity": 1,
                "name": name,
            }
            try:
                resp = self._req("POST", "/instance-operations/launch", body,
                                 timeout=60)
            except Exception as e:
                last_err = e
                log.warning(f"launch in {region} failed: {e}")
                continue
            ids = resp.get("instance_ids") if isinstance(resp, dict) else None
            if ids:
                iid = str(ids[0])
                self._launched_at[iid] = time.time()
                log.info(f"Launched Lambda instance {iid} ('{name}', "
                         f"{resolve_resource(resource_name)} in {region})")
                return {"uid": iid, "name": name, "region": region}
            last_err = RuntimeError(f"launch in {region} returned no instance_ids: {resp}")
        raise RuntimeError(f"Lambda launch failed for {resource_name!r}: {last_err}")

    def list_instances(self) -> list[dict]:
        resp = self._req("GET", "/instances")
        return resp if isinstance(resp, list) else []

    def get_instance(self, instance_id: str) -> dict:
        inst = self._req("GET", f"/instances/{quote(str(instance_id))}")
        return inst if isinstance(inst, dict) else {}

    def deploy_workload(self, wrk_uid: str) -> dict:
        """No-op: a Lambda launch both registers and boots the instance."""
        return {}

    def get_workload_state(self, wrk_uid: str) -> dict:
        """Current instance state in the Targon-shaped {"status": ...} form."""
        try:
            inst = self.get_instance(wrk_uid)
        except RuntimeError as e:
            if " 404 " in str(e):
                return {"status": "DELETED"}
            raise
        ip = inst.get("ip", "")
        if ip:
            self._ssh_ip[str(wrk_uid)] = ip
        raw = str(inst.get("status", "")).lower()
        return {"status": _STATUS_MAP.get(raw, raw.upper() or "UNKNOWN"),
                "name": inst.get("name") or "",
                "ip": ip}

    def wait_running(self, wrk_uid: str, timeout: int = LAMBDA_STARTUP_TIMEOUT,
                     poll_interval: int = 15) -> bool:
        """Poll until the instance is active, then bootstrap root SSH +
        /workspace (see _ROOT_BOOTSTRAP) so the box serves the same root@
        contract as Targon/Lium containers. False on timeout or terminal."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.get_workload_state(wrk_uid).get("status", "UNKNOWN")
            if status == "RUNNING":
                return self._bootstrap_root(wrk_uid, deadline)
            if status in _TERMINAL_STATUSES:
                log.error(f"Instance {wrk_uid} entered terminal state: {status}")
                return False
            time.sleep(poll_interval)
        log.error(f"Instance {wrk_uid} did not reach active within {timeout}s")
        return False

    def _bootstrap_root(self, wrk_uid: str, deadline: float) -> bool:
        """SSH in as ubuntu (retrying while sshd comes up) and run the root
        bootstrap; then verify root@ answers. Only after this is the
        orchestrator's ssh_base_cmd (root@ip) usable."""
        ip = self._ssh_ip.get(str(wrk_uid), "")
        if not ip:
            log.error(f"Instance {wrk_uid} active but has no ip")
            return False
        while time.time() < deadline:
            r = subprocess.run(self._ssh_cmd_for(f"ubuntu@{ip}") + [_ROOT_BOOTSTRAP],
                               capture_output=True, timeout=180)
            if r.returncode == 0 and b"bootstrapped" in r.stdout:
                v = subprocess.run(self._ssh_cmd_for(f"root@{ip}") + ["echo rootok"],
                                   capture_output=True, timeout=120)
                if v.returncode == 0 and b"rootok" in v.stdout:
                    log.info(f"Instance {wrk_uid} root-bootstrapped at {ip}")
                    return True
                log.warning(f"root SSH verify failed on {wrk_uid}: "
                            f"{v.stderr.decode(errors='replace')[-200:]}")
            time.sleep(10)
        log.error(f"Instance {wrk_uid} bootstrap did not complete in time")
        return False

    def delete_workload(self, wrk_uid: str) -> None:
        try:
            self._req("POST", "/instance-operations/terminate",
                      {"instance_ids": [str(wrk_uid)]})
        except Exception as e:
            log.warning(f"Failed to terminate instance {wrk_uid}: {e}")

    def list_workloads(self, status: str = None) -> list[dict]:
        """Instances mapped into the Targon workload shape the orchestrator's
        orphan/staleness sweeps expect: name, uid, created_at, state.status.
        created_at is stamped from local launch time (Lambda reports none);
        instances launched by a previous process carry "" and are handled by
        the startup orphan sweep instead."""
        out = []
        for inst in self.list_instances():
            st = _STATUS_MAP.get(str(inst.get("status", "")).lower(), "UNKNOWN")
            if status and st != status.upper():
                continue
            iid = str(inst.get("id", ""))
            launched = self._launched_at.get(iid)
            created = (datetime.fromtimestamp(launched, tz=timezone.utc).isoformat()
                       if launched else "")
            out.append({
                "name": inst.get("name") or "",
                "uid": iid,
                "created_at": created,
                "state": {"status": st},
            })
        return out

    def _ssh_cmd_for(self, user_host: str) -> list[str]:
        return ["ssh", "-i", str(self.key_path),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                user_host]

    def ssh_base_cmd(self, wrk_uid: str) -> list[str]:
        """Base SSH command for an instance — root@ip:22 with the
        Lambda-registered private key (root access set up by wait_running)."""
        ip = self._ssh_ip.get(str(wrk_uid))
        if not ip:
            ip = self.get_instance(wrk_uid).get("ip", "")
            if ip:
                self._ssh_ip[str(wrk_uid)] = ip
        if not ip:
            raise RuntimeError(f"no SSH connect info for Lambda instance {wrk_uid}")
        return self._ssh_cmd_for(f"root@{ip}")
