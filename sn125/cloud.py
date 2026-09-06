"""
Targon cloud orchestrator for SN125 validator.

Credentials: read from TARGON_API_KEY env var, falling back to a chmod-600 file
at ~/.sn125/targon_api_key. NEVER hardcode the key in this file.

Provides TargonClient (HTTP wrapper) and TargonOrchestrator (submission lifecycle).

The orchestrator is provider-agnostic behind its ``client``: any object with
the TargonClient workload surface (ensure_ssh_key, check_availability,
create_workload, deploy_workload, wait_running, get_workload_state,
delete_workload, list_workloads) can drive it, optionally overriding SSH
addressing (``ssh_base_cmd``) and pricing (``cost_per_hour``). See
``cloud_lium.LiumClient`` for the Lium (Bittensor SN51) provider.
"""
import hashlib, json, logging, os, shlex, stat, subprocess, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from . import settings
from .env import load_dotenv
from .roundsm.audit import audit_emit, looks_like_crash

log = logging.getLogger("sn125.cloud")


def _load_targon_api_key() -> str:
    """Resolve Targon API key from env var or ~/.sn125/targon_api_key.

    Precedence:
    1. TARGON_API_KEY env var (preferred for production / supervised runs)
    2. ~/.sn125/targon_api_key (must be chmod 600; fails loudly if world/group readable)

    Returns "" if neither is found, with a loud WARNING so a misconfigured validator
    doesn't silently no-op. Callers that actually hit the API will get a 401 and
    surface a clear error.
    """
    load_dotenv()
    env_val = os.environ.get("TARGON_API_KEY", "").strip()
    if env_val:
        return env_val

    key_file = Path.home() / ".sn125" / "targon_api_key"
    if key_file.exists():
        st = key_file.stat()
        if st.st_mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"{key_file} has insecure permissions (mode={oct(st.st_mode & 0o777)}). "
                f"Run: chmod 600 {key_file}"
            )
        val = key_file.read_text().strip()
        if val:
            return val

    log.warning(
        "No Targon API key found. Set TARGON_API_KEY env var OR write the key to "
        "~/.sn125/targon_api_key (chmod 600). Cloud-backed evaluations will fail with 401."
    )
    return ""


TARGON_API_KEY = _load_targon_api_key()
TARGON_BASE = "https://api.targon.com"
SSH_HOST = "ssh.deployments.targon.com"
SSH_KEY_DIR = Path.home() / ".sn125"
SSH_KEY_PATH = SSH_KEY_DIR / "targon_id_ed25519"
SSH_KEY_NAME = "sn125-validator"

MAX_CONCURRENT = 8
DAILY_SPEND_LIMIT = 1000.0
RENTAL_STARTUP_TIMEOUT = 300
SSH_CONNECT_TIMEOUT = 120
SUBMISSION_MARGIN = 600

EVAL_SKU = "b200-small"
DEFAULT_TARGON_IMAGE = "python:3.11-slim-bookworm"


def is_b200_sku(sku: str) -> bool:
    """True iff `sku` is a Blackwell B200 resource (the only class allowed to
    score). Matches Targon SKU names ("b200-small"), Lium machine names
    ("NVIDIA B200"), Lambda instance types ("gpu_1x_b200_sxm6") and EC2
    instance types ("p6-b200.48xlarge"); B300/H200/... never match."""
    return isinstance(sku, str) and "b200" in sku.lower()


CLOUD_PROVIDERS = ("targon", "lium", "lambda", "aws", "runpod")


def make_cloud_client(backend: str, timeout_s: int = 86400):
    """Build the provider client for ``backend`` (one of CLOUD_PROVIDERS).

    Returns None for "targon" — the orchestrator constructs its own
    TargonClient default. Imports are lazy: provider modules import constants
    from this module."""
    if backend == "lium":
        from .cloud_lium import LiumClient, termination_hours_for
        return LiumClient(termination_hours=termination_hours_for(int(timeout_s)))
    if backend == "lambda":
        from .cloud_lambda import LambdaClient
        return LambdaClient()
    if backend == "aws":
        from .cloud_aws import AwsClient
        return AwsClient()
    if backend == "runpod":
        from .cloud_runpod import RunpodClient
        return RunpodClient()
    if backend != "targon":
        raise ValueError(f"unknown cloud backend {backend!r} "
                         f"(expected one of {CLOUD_PROVIDERS})")
    return None


SCORING_PROVIDERS = ("runpod", "targon", "lambda", "aws")

CONTAINER_PROVIDERS = ("runpod",)


def ensure_scoring_providers(providers: list[str] | tuple[str, ...]) -> list[str]:
    """Validate a provider failover chain for the SCORING path.

    Raises ValueError on an empty chain, an unknown provider, or any provider
    outside SCORING_PROVIDERS (i.e. lium). Returns the normalized list."""
    chain = [str(p).strip().lower() for p in providers if str(p).strip()]
    if not chain:
        raise ValueError("provider chain is empty — set SN125_CLOUD_PROVIDERS "
                         f"(e.g. {','.join(settings.DEFAULT_CLOUD_PROVIDER_CHAIN)})")
    unknown = [p for p in chain if p not in CLOUD_PROVIDERS]
    if unknown:
        raise ValueError(f"unknown cloud provider(s) {unknown} "
                         f"(expected among {CLOUD_PROVIDERS})")
    untrusted = [p for p in chain if p not in SCORING_PROVIDERS]
    if untrusted:
        raise ValueError(
            f"provider(s) {untrusted} are not allowed to score submissions: "
            f"scoring requires a trusted host (one of {SCORING_PROVIDERS}). "
            f"Lium hosts are untrusted third parties who could read revealed "
            f"miner source or forge results.")
    if "runpod" in chain:
        from .cloud_runpod import _cloud_tier
        tier = _cloud_tier()
        if tier != "SECURE":
            raise ValueError(
                f"runpod may score only on the SECURE cloud tier (provider-"
                f"operated data centers); SN125_RUNPOD_CLOUD={tier!r} is "
                f"third-party hosted and is refused for scoring.")
    return chain


def provider_isolation(name: str) -> str:
    """``container`` for providers that rent containers, else ``vm``."""
    return "container" if str(name).strip().lower() in CONTAINER_PROVIDERS else "vm"


def make_provider_client(backend: str, timeout_s: int = 86400):
    """Like make_cloud_client but always returns a concrete client (builds the
    TargonClient instead of the None sentinel) and stamps ``provider_name``."""
    client = make_cloud_client(backend, timeout_s)
    if client is None:
        client = TargonClient()
    try:
        client.provider_name = backend
    except Exception:
        pass
    return client


def make_failover_client(providers: list[str], timeout_s: int = 86400):
    """Build the client for an ordered provider chain: a single concrete client
    for a chain of one, a FailoverClient otherwise."""
    if len(providers) == 1:
        return make_provider_client(providers[0], timeout_s)
    return FailoverClient([(p, make_provider_client(p, timeout_s))
                           for p in providers])


class FailoverClient:
    """Ordered multi-provider client behind the single-client orchestrator seam.

    Presents the duck-typed workload surface (ensure_ssh_key,
    check_availability, create_workload, deploy_workload, wait_running,
    get_workload_state, delete_workload, list_workloads, ssh_base_cmd,
    cost_per_hour) over an ORDERED list of per-provider clients:

    * ``create_workload`` walks the chain and rents from the first provider
      with inventory whose create succeeds — this is the failover. Every
      later per-workload call (deploy/state/delete/ssh) routes to the owning
      provider via a uid map.
    * ``check_availability`` aggregates across providers, so the orchestrator's
      B200 capacity gate (has_b200_capacity / wait_for_b200_capacity) pauses
      the round only when EVERY provider is dry.
    * Throughput-band rejections strike the owning provider
      (``note_box_rejected``); a provider with STRIKE_LIMIT consecutive
      strikes is deprioritized for STRIKE_COOLDOWN_S so a systematically
      out-of-band fleet fails over instead of relaunching forever. A box that
      passes the gate (``note_box_ok``) clears its provider's strikes. Struck
      providers remain a last resort — the chain never refuses to rent while
      any provider has capacity.
    * ``cost_per_hour`` is the MAX across providers: the daily-cap admission
      check runs before the serving provider is known, and the cap must never
      undercount.
    """

    STRIKE_LIMIT = 2
    STRIKE_COOLDOWN_S = 3600.0

    def __init__(self, providers: list[tuple[str, object]]):
        if not providers:
            raise ValueError("FailoverClient requires at least one provider")
        self.providers = list(providers)
        self._by_name = dict(self.providers)
        self.provider_name = ",".join(name for name, _ in self.providers)
        self._owner: dict[str, str] = {}
        self._ssh_keys: dict[str, str] = {}
        self._strikes: dict[str, int] = {}
        self._struck_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def _is_struck(self, name: str, now: float) -> bool:
        with self._lock:
            if self._strikes.get(name, 0) < self.STRIKE_LIMIT:
                return False
            if now - self._struck_at.get(name, 0.0) >= self.STRIKE_COOLDOWN_S:
                self._strikes[name] = 0
                return False
            return True

    def _ordered(self) -> list[tuple[str, object]]:
        now = time.time()
        healthy = [(n, c) for n, c in self.providers if not self._is_struck(n, now)]
        struck = [(n, c) for n, c in self.providers if self._is_struck(n, now)]
        return healthy + struck

    def provider_of(self, wrk_uid: str) -> str:
        with self._lock:
            return self._owner.get(wrk_uid, "")

    def isolation_of(self, wrk_uid: str) -> str:
        """``vm`` / ``container`` for the provider that owns ``wrk_uid``."""
        client = self._client_for(wrk_uid)
        if client is None:
            return "vm"
        return str(getattr(client, "provider_isolation", "")
                   or provider_isolation(self.provider_of(wrk_uid)))

    def note_box_rejected(self, wrk_uid: str) -> None:
        """A box from this provider failed the throughput gate — strike it."""
        name = self.provider_of(wrk_uid)
        if not name:
            return
        with self._lock:
            self._strikes[name] = self._strikes.get(name, 0) + 1
            self._struck_at[name] = time.time()
            n = self._strikes[name]
        log.warning(f"provider {name!r} box gate rejection #{n}"
                    + (f" — deprioritized for {self.STRIKE_COOLDOWN_S:.0f}s"
                       if n >= self.STRIKE_LIMIT else ""))

    def note_box_ok(self, wrk_uid: str) -> None:
        name = self.provider_of(wrk_uid)
        if not name:
            return
        with self._lock:
            self._strikes[name] = 0

    def _client_for(self, wrk_uid: str):
        name = self.provider_of(wrk_uid)
        if name and name in self._by_name:
            return self._by_name[name]
        return None

    def ensure_ssh_key(self) -> str:
        """Register the validator SSH key with EVERY provider (each needs it
        before its first rental); per-provider uids are kept internally and
        substituted in create_workload."""
        for name, client in self.providers:
            self._ssh_keys[name] = client.ensure_ssh_key()
        return "failover:" + ",".join(self._ssh_keys)

    def check_availability(self, resource_name: str) -> int:
        total = 0
        for name, client in self.providers:
            try:
                total += int(client.check_availability(resource_name) or 0)
            except Exception as e:
                log.warning(f"availability probe failed on {name!r}: {e}")
        return total

    def create_workload(self, name: str, resource_name: str, ssh_key_uid: str,
                        image: str = "", envs: list[dict] = None,
                        args: list[str] = None) -> dict:
        """Rent from the first provider in (strike-adjusted) chain order with
        inventory whose create succeeds. Records the uid->provider owner."""
        errors: list[str] = []
        for prov, client in self._ordered():
            try:
                if int(client.check_availability(resource_name) or 0) < 1:
                    errors.append(f"{prov}: no capacity")
                    continue
            except Exception as e:
                errors.append(f"{prov}: availability check failed ({e})")
                continue
            key_uid = self._ssh_keys.get(prov, ssh_key_uid)
            try:
                resp = client.create_workload(name, resource_name, key_uid,
                                              image=image, envs=envs, args=args)
            except Exception as e:
                errors.append(f"{prov}: create failed ({e})")
                log.warning(f"create_workload failed on {prov!r}, trying next "
                            f"provider: {e}")
                continue
            uid = (resp or {}).get("uid", "")
            if not uid:
                errors.append(f"{prov}: no uid in response")
                continue
            with self._lock:
                self._owner[uid] = prov
            resp = dict(resp)
            resp["provider"] = prov
            log.info(f"[{uid}] rented on provider {prov!r} ({resource_name})")
            return resp
        raise RuntimeError(
            "rental_provisioning_failed: no provider could serve "
            f"{resource_name!r} ({'; '.join(errors) or 'empty chain'})")

    def deploy_workload(self, wrk_uid: str) -> dict:
        client = self._client_for(wrk_uid)
        if client is None:
            raise RuntimeError(f"unknown workload {wrk_uid!r} (no owning provider)")
        return client.deploy_workload(wrk_uid)

    def wait_running(self, wrk_uid: str, *args, **kwargs) -> bool:
        client = self._client_for(wrk_uid)
        if client is None:
            return False
        return client.wait_running(wrk_uid, *args, **kwargs)

    def get_workload_state(self, wrk_uid: str) -> dict:
        client = self._client_for(wrk_uid)
        if client is not None:
            return client.get_workload_state(wrk_uid)
        for _name, c in self.providers:
            try:
                state = c.get_workload_state(wrk_uid)
                if state:
                    return state
            except Exception:
                continue
        return {}

    def delete_workload(self, wrk_uid: str) -> None:
        client = self._client_for(wrk_uid)
        if client is not None:
            client.delete_workload(wrk_uid)
            return
        for name, c in self.providers:
            try:
                c.delete_workload(wrk_uid)
            except Exception as e:
                log.warning(f"delete_workload({wrk_uid}) failed on {name!r}: {e}")

    def list_workloads(self, status: str = None) -> list[dict]:
        """Aggregate across providers, learning uid->provider ownership from the
        listings (so orphan cleanup after a restart routes deletes correctly)."""
        out: list[dict] = []
        for name, client in self.providers:
            try:
                items = client.list_workloads(status) or []
            except Exception as e:
                log.warning(f"list_workloads failed on {name!r}: {e}")
                continue
            for w in items:
                uid = (w or {}).get("uid", "")
                if uid:
                    with self._lock:
                        self._owner.setdefault(uid, name)
                out.append(w)
        return out

    def ssh_base_cmd(self, wrk_uid: str) -> list[str]:
        """Route SSH addressing to the owning provider. Clients without their
        own ssh_base_cmd (Targon) get the shared-proxy form the orchestrator
        would otherwise build itself."""
        client = self._client_for(wrk_uid)
        if client is None:
            raise RuntimeError(f"unknown workload {wrk_uid!r} (no owning provider)")
        base_fn = getattr(type(client), "ssh_base_cmd", None)
        if base_fn is not None:
            return client.ssh_base_cmd(wrk_uid)
        return ["ssh", "-i", str(SSH_KEY_PATH),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                f"{wrk_uid}@{SSH_HOST}"]

    def cost_per_hour(self, resource: str) -> float:
        """Pessimistic $/h: the max across providers, because the daily-cap
        admission check runs before the serving provider is known."""
        costs: list[float] = []
        for name, client in self.providers:
            cost_fn = getattr(type(client), "cost_per_hour", None)
            if cost_fn is not None:
                try:
                    costs.append(float(client.cost_per_hour(resource)))
                    continue
                except Exception as e:
                    log.warning(f"cost_per_hour({resource!r}) failed on {name!r}: {e}")
            costs.append(SKU_COSTS.get(resource, 20.0))
        return max(costs) if costs else 20.0


class CapacityWait(Exception):
    """Raised when the canonical B200 eval SKU has ZERO inventory.

    This is a provider-capacity outage, NOT a host-802 flake and NOT a cue to
    substitute another GPU class. The round driver catches it and PAUSES the round
    (capacity-aware public delay): the round's deadlines stretch, the 20h eval
    budget clock counts only real training (never the paused wait), and the pause is
    surfaced prominently on the dashboard. Distinct from a provisioning flake (a host
    that 802s after we DID get capacity), which the existing retry-sweep handles.
    """

    def __init__(self, sku: str, waited_s: float = 0.0):
        self.sku = sku
        self.waited_s = waited_s
        super().__init__(
            f"no {sku} B200 capacity after waiting {waited_s:.0f}s — "
            f"capacity-aware delay (B200-only; no substitution)")


SKU_COSTS = {
    "b200-small": 5.3, "b200-medium": 10.6, "b200-large": 21.2, "b200-xlarge": 42.4,
    "b300-small": 7.0, "b300-medium": 11.6, "b300-large": 22.2, "b300-xlarge": 43.4,
    "h200-small": 3.29, "h200-medium": 6.58, "h200-large": 13.16, "h200-xlarge": 26.32,
    "h100-small": 2.5, "h100-medium": 5.0, "h100-large": 10.0, "h100-xlarge": 20.0,
    "rtx4090-small": 0.45, "rtx4090-medium": 0.9, "rtx4090-large": 1.8,
    "rtx6000b-small": 1.5,
    "cpu-small": 0.09, "cpu-medium": 0.18, "cpu-large": 0.36, "cpu-xlarge": 0.72,
}


@dataclass
class RentalRecord:
    wrk_uid: str
    name: str
    resource: str
    created_at: float
    cost_per_hour: float
    provider: str = ""
    deleted: bool = False
    cost_accrued: float = 0.0
    expected_cost: float = 0.0


class TargonClient:
    """Thin HTTP wrapper around Targon rentals API. No SDK dependency."""

    provider_name = "targon"

    def __init__(self, api_key: str = TARGON_API_KEY, base_url: str = TARGON_BASE):
        if not api_key:
            raise RuntimeError("TARGON_API_KEY env var must be set for cloud features")
        self.api_key = api_key
        self.base = base_url.rstrip("/")

    def _req(self, method: str, path: str, body: dict = None, auth: bool = True,
             timeout: int = 30) -> dict:
        """Make an HTTP request, return parsed JSON (or {} for 202/204)."""
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = Request(url, data=data, headers=headers, method=method)
        for attempt in range(4):
            try:
                with urlopen(req, timeout=timeout) as resp:
                    if resp.status in (202, 204):
                        return {}
                    return json.loads(resp.read())
            except HTTPError as e:
                if e.code >= 500 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                body_text = e.read().decode()[:500] if hasattr(e, 'read') else str(e)
                raise RuntimeError(f"Targon {method} {path}: {e.code} {body_text}") from e
            except URLError as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Targon {method} {path}: {e}") from e
        return {}

    def list_inventory(self, gpu_only: bool = True) -> list[dict]:
        """Get available rental resources. No auth required."""
        qs = "?type=rental" + ("&gpu=true" if gpu_only else "")
        return self._req("GET", f"/tha/v2/inventory{qs}", auth=False)

    def check_availability(self, resource_name: str) -> int:
        """Return available count for a specific SKU."""
        for item in self.list_inventory(gpu_only=False):
            if item.get("name") == resource_name:
                return item.get("available", 0)
        return 0

    def list_ssh_keys(self) -> list[dict]:
        resp = self._req("GET", "/tha/v2/ssh-keys")
        return resp.get("items", []) if isinstance(resp, dict) else resp

    def register_ssh_key(self, name: str, pubkey: str) -> str:
        """Register an SSH key, return its shk-... UID."""
        resp = self._req("POST", "/tha/v2/ssh-keys", {"name": name, "ssh_key": pubkey})
        return resp.get("uid", "")

    def ensure_ssh_key(self) -> str:
        """Generate keypair if needed, register with Targon, return shk- UID."""
        SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
        if not SSH_KEY_PATH.exists():
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(SSH_KEY_PATH), "-N", "", "-C", SSH_KEY_NAME],
                check=True, capture_output=True,
            )
            log.info(f"Generated SSH keypair at {SSH_KEY_PATH}")
        pubkey = SSH_KEY_PATH.with_suffix(".pub").read_text().strip()
        for key in self.list_ssh_keys():
            if key.get("name") == SSH_KEY_NAME:
                log.info(f"SSH key '{SSH_KEY_NAME}' already registered: {key['uid']}")
                return key["uid"]
        uid = self.register_ssh_key(SSH_KEY_NAME, pubkey)
        log.info(f"Registered SSH key '{SSH_KEY_NAME}': {uid}")
        return uid

    def create_workload(self, name: str, resource_name: str, ssh_key_uid: str,
                        image: str = "",
                        envs: list[dict] = None, args: list[str] = None) -> dict:
        """Register a new workload. Returns full workload dict with uid."""
        image = image or os.environ.get("SN125_TARGON_IMAGE", DEFAULT_TARGON_IMAGE)
        body = {
            "name": name,
            "image": image,
            "resource_name": resource_name,
            "type": "RENTAL",
            "ssh_keys": [ssh_key_uid],
            "args": args or ["sleep", "infinity"],
        }
        if envs:
            body["envs"] = envs
        return self._req("POST", "/tha/v2/workloads", body)

    def deploy_workload(self, wrk_uid: str) -> dict:
        """Deploy a registered workload. Returns 202."""
        return self._req("POST", f"/tha/v2/workloads/{wrk_uid}/deploy")

    def get_workload_state(self, wrk_uid: str) -> dict:
        """Get current workload state."""
        return self._req("GET", f"/tha/v2/workloads/{wrk_uid}/state")

    def delete_workload(self, wrk_uid: str) -> None:
        """Delete/teardown a workload. Returns 204."""
        try:
            self._req("DELETE", f"/tha/v2/workloads/{wrk_uid}")
        except Exception as e:
            log.warning(f"Failed to delete workload {wrk_uid}: {e}")

    def list_workloads(self, status: str = None) -> list[dict]:
        """List workloads, optionally filtered by status."""
        qs = f"?status={status}" if status else ""
        resp = self._req("GET", f"/tha/v2/workloads{qs}")
        if isinstance(resp, list):
            return resp
        return resp.get("items", resp.get("workloads", resp.get("data", [])))

    def wait_running(self, wrk_uid: str, timeout: int = RENTAL_STARTUP_TIMEOUT,
                     poll_interval: int = 7) -> bool:
        """Poll until workload status == RUNNING. Returns True if ready, False if timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get_workload_state(wrk_uid)
            status = state.get("status", "UNKNOWN").upper()
            if status == "RUNNING":
                return True
            if status in ("FAILED", "DELETED", "ERROR"):
                log.error(f"Workload {wrk_uid} entered terminal state: {status}")
                return False
            time.sleep(poll_interval)
        log.error(f"Workload {wrk_uid} did not reach RUNNING within {timeout}s")
        return False


class TargonOrchestrator:
    """Manages per-submission lifecycle on Targon cloud rentals."""

    def __init__(self, client: TargonClient = None, resource: str = "b200-small",
                 timeout: int = 86400, project_root: str = "",
                 expected_manifest: str = ""):
        self.client = client or TargonClient()
        self.resource = resource
        self.timeout = timeout
        self.project_root = project_root or str(Path(__file__).resolve().parent.parent)
        self.expected_manifest = expected_manifest
        self._semaphore = threading.Semaphore(MAX_CONCURRENT)
        self._ssh_key_uid: str = ""
        self._daily_spend: float = 0.0
        self._daily_spend_date: str = ""
        self._reserved_spend: float = 0.0
        self._spend_lock = threading.Lock()
        self._rentals: list[RentalRecord] = []
        self._capacity_wait: Optional[dict] = None
        self._capacity_lock = threading.Lock()

    @staticmethod
    def _heartbeat_interval() -> float:
        return settings.env_float(settings.TARGON_HEARTBEAT_S_ENV, 60.0, minimum=10.0)

    @staticmethod
    def _remote_heartbeat_every() -> int:
        return settings.env_int(settings.TARGON_REMOTE_HEARTBEAT_EVERY_ENV, 5, minimum=1)

    @staticmethod
    def _workload_status(state: dict) -> str:
        return str((state or {}).get("status") or
                   ((state or {}).get("state") or {}).get("status") or
                   "UNKNOWN").upper()

    @staticmethod
    def _cap_text(text: str, n: int = 4000) -> str:
        text = str(text or "")
        return text if len(text) <= n else text[:n] + f"...[truncated {len(text) - n} chars]"

    def _remote_health_snapshot(self, wrk_uid: str) -> dict:
        """Static, bounded remote probes. No miner-controlled command text."""
        cmd = (
            "set +e; "
            "echo '--- nvidia-smi ---'; "
            "nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw "
            "--format=csv,noheader 2>&1 | head -8; "
            "echo '--- disk ---'; df -h /workspace 2>&1 | tail -2; "
            "echo '--- memory ---'; free -h 2>&1 | head -3; "
            "echo '--- processes ---'; ps -eo pid,etimes,pcpu,pmem,comm --sort=-pcpu 2>/dev/null | head -8"
        )
        try:
            r = self._ssh_run(wrk_uid, cmd, timeout=25)
            return {
                "ok": r.returncode == 0,
                "rc": r.returncode,
                "output": self._cap_text(r.stdout.decode(errors="replace"), 8000),
                "stderr": self._cap_text(r.stderr.decode(errors="replace"), 2000),
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "error_type": type(e).__name__}

    def _remote_fingerprint(self, wrk_uid: str) -> dict:
        """One-shot remote hardware/software fingerprint for reproducibility."""
        cmd = (
            "set +e; "
            "echo '--- system ---'; uname -a; "
            "echo '--- python ---'; python --version 2>&1; "
            "echo '--- torch ---'; python - <<'PY'\n"
            "import json\n"
            "try:\n"
            " import torch\n"
            " print(json.dumps({'torch': torch.__version__, 'cuda': getattr(torch.version, 'cuda', None), "
            "'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(), "
            "'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))\n"
            "except Exception as e:\n"
            " print(json.dumps({'torch_error': str(e)}))\n"
            "PY\n"
            "echo '--- nvidia ---'; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1; "
            "echo '--- cuda ---'; (nvcc --version 2>/dev/null | tail -1) || true; "
            "echo '--- mounts ---'; df -h /workspace /tmp 2>&1; "
            "echo '--- image ---'; cat /etc/os-release 2>/dev/null | head -8"
        )
        try:
            r = self._ssh_run(wrk_uid, cmd, timeout=45)
            return {
                "ok": r.returncode == 0,
                "rc": r.returncode,
                "output": self._cap_text(r.stdout.decode(errors="replace"), 12000),
                "stderr": self._cap_text(r.stderr.decode(errors="replace"), 2000),
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "error_type": type(e).__name__}

    @staticmethod
    def _stage_done(started: float) -> float:
        return max(0.0, time.time() - started)

    def _start_workload_heartbeat(self, wrk_uid: str, record: RentalRecord,
                                  *, audit=None, round_id: str = "",
                                  sub_uid: str = "") -> threading.Event:
        """Emit regular Targon/API and occasional remote health checkups."""
        stop = threading.Event()
        interval = self._heartbeat_interval()
        remote_every = self._remote_heartbeat_every()

        def snapshot(seq: int, *, immediate: bool = False) -> None:
            state = {}
            status = "UNKNOWN"
            state_error = ""
            try:
                state = self.client.get_workload_state(wrk_uid)
                status = self._workload_status(state)
            except Exception as e:
                state_error = str(e)
            elapsed = max(0.0, time.time() - record.created_at)
            cost_so_far = (elapsed / 3600.0) * record.cost_per_hour
            audit_emit(
                audit,
                "targon.heartbeat",
                wrk_uid=wrk_uid,
                workload_name=record.name,
                round_id=round_id,
                sub_uid=sub_uid,
                resource=record.resource,
                status=status,
                state=state,
                state_error=state_error,
                elapsed_s=elapsed,
                cost_so_far=cost_so_far,
                daily_spend=self.daily_spend,
                active_rentals=sum(1 for r in self._rentals if not r.deleted),
                heartbeat_seq=seq,
                immediate=immediate,
                message=f"{status} elapsed={elapsed:.0f}s cost=${cost_so_far:.2f}",
            )
            if status == "RUNNING" and seq and seq % remote_every == 0:
                audit_emit(
                    audit,
                    "targon.remote_health",
                    wrk_uid=wrk_uid,
                    round_id=round_id,
                    sub_uid=sub_uid,
                    diagnostics=self._remote_health_snapshot(wrk_uid),
                )

        def loop() -> None:
            seq = 0
            snapshot(seq, immediate=True)
            while not stop.wait(interval):
                seq += 1
                snapshot(seq)

        t = threading.Thread(target=loop, daemon=True,
                             name=f"sn125-targon-heartbeat-{wrk_uid[:8]}")
        t.start()
        return stop

    def _track_cost(self, cost: float):
        """Thread-safe daily spend tracking."""
        today = time.strftime("%Y-%m-%d")
        with self._spend_lock:
            if self._daily_spend_date != today:
                self._daily_spend = 0.0
                self._daily_spend_date = today
            self._daily_spend += cost

    def _reserve_spend(self, expected: float) -> bool:
        """Admission check for a new rental against the daily cap.

        `daily_spend` only accrues at teardown, so completed spend alone is blind
        to in-flight boxes: N concurrent rentals could all pass a naive
        `daily_spend >= LIMIT` check while the counter still reads $0. Count the
        projected cost of every in-flight rental (the reservation) alongside
        completed spend; refuse any rental whose projection would push the total
        past the cap. Reservations are released at teardown when the actual cost
        lands.
        """
        today = time.strftime("%Y-%m-%d")
        with self._spend_lock:
            if self._daily_spend_date != today:
                self._daily_spend = 0.0
                self._daily_spend_date = today
            if self._daily_spend + self._reserved_spend + expected > DAILY_SPEND_LIMIT:
                return False
            self._reserved_spend += expected
            return True

    def _release_reserved_spend(self, expected: float) -> None:
        with self._spend_lock:
            self._reserved_spend = max(0.0, self._reserved_spend - expected)

    def affordable_concurrency(self, resource: str = "") -> int:
        """How many evals the daily cap can hold in flight at once.

        Each rental reserves ``(timeout + margin)/3600 × $/h`` against
        DAILY_SPEND_LIMIT before it starts (``_reserve_spend``), so with the
        24h default timeout on a $6.79/h RunPod B200 the reservation is ~$164
        and the $1000 cap admits 6, not 8. Counts what is already spent or
        reserved today; the round loop re-reads it every round so selections
        beyond the headroom are deferred instead of refused at rental time."""
        per_eval = ((self.timeout + SUBMISSION_MARGIN) / 3600.0) * \
            self._cost_per_hour(resource or self.resource)
        if per_eval <= 0:
            return MAX_CONCURRENT
        headroom = DAILY_SPEND_LIMIT - float(self.committed_spend)
        return max(0, min(MAX_CONCURRENT, int(headroom // per_eval)))

    @property
    def daily_spend(self) -> float:
        today = time.strftime("%Y-%m-%d")
        with self._spend_lock:
            return self._daily_spend if self._daily_spend_date == today else 0.0

    @property
    def committed_spend(self) -> float:
        """Completed spend today plus projected cost of in-flight rentals."""
        today = time.strftime("%Y-%m-%d")
        with self._spend_lock:
            done = self._daily_spend if self._daily_spend_date == today else 0.0
            return done + self._reserved_spend

    def initialize(self):
        """One-time setup: SSH key registration + orphan cleanup + periodic cleanup thread."""
        self._ssh_key_uid = self.client.ensure_ssh_key()
        self.cleanup_orphans()
        self.write_status_file()
        t = threading.Thread(target=self._periodic_cleanup, daemon=True)
        t.start()
        log.info(f"Orchestrator initialized. SSH key: {self._ssh_key_uid}, resource: {self.resource}")

    def _periodic_cleanup(self):
        """Every 30 min, check for stale rentals (older than timeout+600s) and delete them."""
        while True:
            time.sleep(1800)
            try:
                workloads = self.client.list_workloads()
                for w in workloads:
                    name = w.get("name", "")
                    if not name.startswith("sn125-"):
                        continue
                    created = w.get("created_at", "")
                    if created:
                        try:
                            from datetime import datetime, timezone
                            ct = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            age_s = (datetime.now(timezone.utc) - ct).total_seconds()
                            if age_s > self.timeout + SUBMISSION_MARGIN:
                                uid = w.get("uid", "")
                                log.warning(f"Periodic cleanup: stale rental {name} ({uid}, "
                                           f"age={age_s:.0f}s > {self.timeout + SUBMISSION_MARGIN}s)")
                                self.client.delete_workload(uid)
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                log.error(f"Periodic cleanup error: {e}")

    def cleanup_orphans(self):
        """DELETE any sn125-* rentals left by a crashed orchestrator (any status)."""
        try:
            workloads = self.client.list_workloads()
            for w in workloads:
                name = w.get("name", "")
                if name.startswith("sn125-"):
                    uid = w.get("uid", "")
                    status = w.get("state", {}).get("status", "?")
                    log.warning(f"Cleaning up orphaned rental: {name} ({uid}, {status})")
                    self.client.delete_workload(uid)
        except Exception as e:
            log.error(f"Orphan cleanup failed: {e}")

    def _check_inventory(self) -> str:
        """Return the canonical B200 eval SKU iff it has capacity; else raise CapacityWait.

        B200-ONLY (operator directive 2026-06-24): the production scoring path NEVER
        substitutes another GPU class (H200/H100) or another SKU — that would break
        the determinism class and the fixed-budget (20h) fairness. The old
        cross-SKU/cross-architecture fallback ladder has been removed.

        When the SKU is dry we do NOT silently fall back; we raise ``CapacityWait`` so
        the round driver pauses (capacity-aware public delay) and waits for B200 via
        ``wait_for_b200_capacity`` rather than scoring on the wrong hardware. A short
        in-call poll absorbs momentary 0-blips so we don't pause for a few seconds.
        """
        if not is_b200_sku(self.resource):
            raise RuntimeError(
                f"resource {self.resource!r} is not a B200 SKU — production eval is "
                f"B200-only; refusing to provision a non-B200 box for scoring")
        avail = self.client.check_availability(self.resource)
        if avail >= 1:
            return self.resource
        log.warning(f"{self.resource} unavailable (0 slots). Brief re-poll before capacity delay...")
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(15)
            if self.client.check_availability(self.resource) >= 1:
                return self.resource
        raise CapacityWait(self.resource, waited_s=60.0)

    def has_b200_capacity(self) -> bool:
        """Cheap, non-raising inventory probe for the round driver's capacity gate:
        True iff the canonical B200 eval SKU currently has >=1 slot. Never falls back
        to another GPU class (B200-only). A transient API error reads as 'no capacity'
        so the driver pauses + retries rather than charging ahead on stale info."""
        if not is_b200_sku(self.resource):
            return False
        try:
            return self.client.check_availability(self.resource) >= 1
        except Exception as e:
            log.warning(f"capacity probe failed ({e}) — treating as no capacity")
            return False

    def wait_for_b200_capacity(self, backoff_s: float = 300.0,
                               max_wait_s: float = 0.0,
                               alert_after_s: float = 21600.0,
                               on_event=None) -> str:
        """Block until the canonical B200 SKU has capacity, then return it.

        This is the long capacity-aware delay (operator directive 2026-06-24): poll
        inventory on a fixed backoff, KEEP WAITING (never substitute, never fail) and
        surface the delay publicly via ``cloud_status.json`` the whole time. If the
        wait exceeds ``alert_after_s`` (default 6h) emit a louder operator alert but
        keep waiting. ``max_wait_s`` is for tests only (0 = wait forever).

        Returns the SKU once available; the caller then ``resume()``s the round.
        """
        emit = on_event or (lambda _m: None)
        start = time.time()
        self.begin_capacity_wait()
        alerted = False
        try:
            while True:
                if self.client.check_availability(self.resource) >= 1:
                    waited = time.time() - start
                    emit(f"B200 capacity restored after {waited:.0f}s — resuming round")
                    return self.resource
                waited = time.time() - start
                if not alerted and waited >= alert_after_s:
                    alerted = True
                    log.error(
                        f"[CAPACITY] B200 ({self.resource}) still dry after "
                        f"{waited/3600:.1f}h — KEEP WAITING (never substitute). "
                        f"Operator: check Targon B200 availability.")
                    emit(f"⚠️ B200 capacity wait exceeded {alert_after_s/3600:.0f}h "
                         f"({waited/3600:.1f}h) — still waiting")
                self.update_capacity_wait(waited_s=waited)
                if max_wait_s and waited >= max_wait_s:
                    raise CapacityWait(self.resource, waited_s=waited)
                time.sleep(backoff_s)
        finally:
            self.clear_capacity_wait()

    def begin_capacity_wait(self, round_id: str = "") -> None:
        """Mark the orchestrator as DELAYED awaiting B200 capacity + publish it."""
        with self._capacity_lock:
            if self._capacity_wait is None:
                self._capacity_wait = {
                    "sku": self.resource,
                    "since": time.time(),
                    "waited_s": 0.0,
                    "round_id": round_id,
                    "reason": "awaiting_b200_capacity",
                }
        self.write_status_file()

    def update_capacity_wait(self, waited_s: float) -> None:
        with self._capacity_lock:
            if self._capacity_wait is not None:
                self._capacity_wait["waited_s"] = waited_s
        self.write_status_file()

    def clear_capacity_wait(self) -> None:
        """Clear the delay state (capacity restored / round resumed) + publish it."""
        with self._capacity_lock:
            self._capacity_wait = None
        self.write_status_file()

    def _ssh_cmd(self, wrk_uid: str) -> list[str]:
        """Base SSH command for a workload.

        Providers address boxes differently: Targon multiplexes through a shared
        SSH proxy (user = workload uid), while Lium pods are reached directly at
        an ip:port from the pod record. A client that defines ``ssh_base_cmd``
        owns the addressing; otherwise the Targon proxy form is used.
        """
        base_fn = getattr(type(self.client), "ssh_base_cmd", None)
        if base_fn is not None:
            return self.client.ssh_base_cmd(wrk_uid)
        return ["ssh", "-i", str(SSH_KEY_PATH), "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", f"{wrk_uid}@{SSH_HOST}"]

    def _cost_per_hour(self, resource: str, default: float = 20.0) -> float:
        """$/h estimate for one rental of ``resource``. A client that defines
        ``cost_per_hour`` (Lium: live marketplace pricing; FailoverClient: max
        across providers) wins; otherwise the static Targon SKU table. The
        pessimistic default keeps unknown SKUs counting against the spend cap
        rather than reading as free."""
        cost_fn = getattr(type(self.client), "cost_per_hour", None)
        if cost_fn is not None:
            try:
                return float(self.client.cost_per_hour(resource))
            except Exception as e:
                log.warning(f"client cost_per_hour({resource!r}) failed: {e}")
        return SKU_COSTS.get(resource, default)

    def _provider_name(self, wrk_uid: str = "") -> str:
        """Provider that owns ``wrk_uid`` (FailoverClient routing map), or the
        client's static provider name for single-provider setups. Published in
        the eval result / round artifacts so every score is auditable against
        the hardware that produced it."""
        fn = getattr(self.client, "provider_of", None)
        if fn is not None and wrk_uid:
            try:
                name = fn(wrk_uid)
                if name:
                    return name
            except Exception:
                pass
        return str(getattr(self.client, "provider_name", "") or "targon")

    def _provider_isolation(self, wrk_uid: str = "") -> str:
        """``vm`` (namespace sandbox + iptables lockdown) or ``container``
        (seccomp sandbox; no box-level netfilter) for the box ``wrk_uid``."""
        fn = getattr(self.client, "isolation_of", None)
        if fn is not None and wrk_uid:
            try:
                iso = fn(wrk_uid)
                if iso:
                    return str(iso)
            except Exception:
                pass
        explicit = getattr(self.client, "provider_isolation", "")
        if explicit:
            return str(explicit)
        return provider_isolation(self._provider_name(wrk_uid))

    def _note_box_rejected(self, wrk_uid: str) -> None:
        fn = getattr(self.client, "note_box_rejected", None)
        if fn is not None:
            try:
                fn(wrk_uid)
            except Exception:
                pass

    def _note_box_ok(self, wrk_uid: str) -> None:
        fn = getattr(self.client, "note_box_ok", None)
        if fn is not None:
            try:
                fn(wrk_uid)
            except Exception:
                pass

    @staticmethod
    def _strip_banner(data: bytes) -> bytes:
        """Strip Targon SSH proxy banner ('Connecting to container...\\r\\n') from output."""
        idx = data.find(b"\r\n")
        if idx >= 0 and data[:idx].startswith(b"Connecting to"):
            return data[idx + 2:]
        idx = data.find(b"\n")
        if idx >= 0 and data[:idx].startswith(b"Connecting to"):
            return data[idx + 1:]
        return data

    def _ssh_run(self, wrk_uid: str, cmd: str, input_data: bytes = None,
                 timeout: int = 120) -> subprocess.CompletedProcess:
        """Run command via SSH, stripping banner from stdout."""
        r = subprocess.run(self._ssh_cmd(wrk_uid) + [cmd],
                           input=input_data, capture_output=True, timeout=timeout)
        r.stdout = self._strip_banner(r.stdout)
        return r

    def _wait_ssh(self, wrk_uid: str, retries: int = 8, delay: int = 5):
        """Wait for SSH to accept connections after rental reaches RUNNING."""
        for attempt in range(retries):
            r = subprocess.run(self._ssh_cmd(wrk_uid) + ["echo ok"],
                               capture_output=True, timeout=30)
            if r.returncode == 0:
                return
            time.sleep(delay)
        raise RuntimeError(f"SSH refused after {retries * delay}s")

    def _upload_project(self, wrk_uid: str):
        """Upload sn125/ tree to /workspace/sn125 via tar-over-ssh."""
        ssh_part = " ".join(shlex.quote(t) for t in self._ssh_cmd(wrk_uid))
        sn125_dir = f"{self.project_root}/sn125"
        r = subprocess.run(
            f"tar czf - -C {sn125_dir} --exclude='__pycache__' --exclude='*.pyc' "
            f"--exclude='rounds' --exclude='search_results*' --exclude='archive' "
            f"--exclude='prod_calibration*' . | "
            f"{ssh_part} 'mkdir -p /workspace/sn125 && tar xzf - -C /workspace/sn125'",
            shell=True, capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace")[-1000:].strip()
            out = r.stdout.decode(errors="replace")[-1000:].strip()
            raise RuntimeError(f"Upload failed (rc={r.returncode}): stderr={err!r} stdout={out!r}")

    def _provision(self, wrk_uid: str):
        """Run setup script on rental."""
        setup_script = self._get_setup_script()
        r = self._ssh_run(wrk_uid, "bash -s", input_data=setup_script.encode(), timeout=600)
        if r.returncode != 0:
            err = r.stderr.decode()[-600:].strip()
            out = r.stdout.decode()[-600:].strip()
            raise RuntimeError(f"Setup failed (rc={r.returncode}): stderr={err!r} stdout={out!r}")

    def _pre_download_models(self, wrk_uid: str, mode: str = "dev") -> dict:
        """Pre-download HF assets while the box still has egress — the
        net_lockdown stage cuts all new outbound connections before the eval.

        Production needs only the SmolLM2 tokenizer (the lean model is built
        inline and the shards are pre-tokenized); the legacy dev path pulls
        the HF model weights + wikitext. Pulling the 360M/1.7B weights on
        every prod rental cost ~9 min per eval (RunPod smoke 2026-09-04)."""
        if mode == "prod":
            cmds = ["python -c \"from transformers import AutoTokenizer; "
                    "AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-360M')\""]
        else:
            models = ["HuggingFaceTB/SmolLM2-135M"]
            cmds = [f"python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
                    f"AutoModelForCausalLM.from_pretrained('{m}'); AutoTokenizer.from_pretrained('{m}')\"" for m in models]
            cmds.append("python -c \"from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)\"")
        script = " && ".join(cmds)
        log.info(f"[{wrk_uid}] Pre-downloading models ({mode})...")
        r = self._ssh_run(wrk_uid, script, timeout=300)
        result = {
            "rc": r.returncode,
            "stdout_tail": self._cap_text(r.stdout.decode(errors="replace")[-2000:], 2000),
            "stderr_tail": self._cap_text(r.stderr.decode(errors="replace")[-2000:], 2000),
        }
        if r.returncode != 0:
            log.warning(f"[{wrk_uid}] Pre-download partial failure: {r.stderr.decode()[:200]}")
        return result

    POLL_INTERVAL_S = 30
    POLL_FAIL_LIMIT = 30

    def _run_sn125_cmd(self, wrk_uid: str, cli_args: str, timeout: int,
                       env: dict = None, audit=None, round_id: str = "",
                       sub_uid: str = "") -> tuple[str, int]:
        """Run an sn125 CLI command on a provisioned rental, detached from SSH.

        Long evals (up to 20h) MUST NOT depend on one long-lived SSH stream:
        Targon's shared SSH proxy silently closed three concurrent ~100-min
        streams at the same instant (replication-floor run, 2026-07-03),
        which the old streaming implementation mis-read as a clean rc=0
        finish with no result line. Instead the command is launched under
        nohup with output redirected to a box-local log, and short-lived SSH
        connections poll the log tail + an exit-code sentinel file until the
        run finishes or the deadline passes. Stdout/stderr are merged into
        the single log for the same pipe-deadlock reason as before (OVN-3,
        2026-04-26).
        """
        run_id = hashlib.sha256(
            f"{cli_args}|{time.time()}".encode()).hexdigest()[:12]
        log_f = f"/workspace/logs/cmd_{run_id}.log"
        rc_f = f"/workspace/logs/cmd_{run_id}.rc"
        sh_f = f"/workspace/logs/cmd_{run_id}.sh"
        exports = "".join(
            f"export {k}={shlex.quote(str(v))}\n" for k, v in (env or {}).items()
            if v is not None and str(v) != "")
        launcher = (
            "mkdir -p /workspace/logs\n"
            "( umask 077; cat > " + sh_f + " <<'SN125_RUNNER_EOS'\n"
            "cd /workspace\n"
            f"{exports}"
            "export PYTHONPATH=/workspace\n"
            f"python -u -m sn125 {cli_args} 2>&1\n"
            "SN125_RUNNER_EOS\n"
            ")\n"
            "umask 022\n"
            f"nohup bash -c 'bash {sh_f} > {log_f} 2>&1; echo $? > {rc_f}' "
            ">/dev/null 2>&1 &\n"
            "echo SN125_LAUNCHED\n")
        audit_emit(audit, "remote.command.start", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, cli_args=cli_args,
                   timeout_s=timeout, run_id=run_id)
        r = self._ssh_run(wrk_uid, "bash -s", input_data=launcher.encode(),
                          timeout=60)
        if r.returncode != 0 or b"SN125_LAUNCHED" not in r.stdout:
            raise RuntimeError(
                f"detached launch failed (rc={r.returncode}): "
                f"{r.stderr.decode(errors='replace')[-500:]}")

        lines: list[str] = []
        pending = b""
        offset = 0
        fails = 0
        rc: int | None = None
        deadline = time.time() + timeout + SUBMISSION_MARGIN

        def _drain(chunk: bytes):
            nonlocal pending
            pending += chunk
            *full, pending = pending.split(b"\n")
            for raw in full:
                line = raw.decode(errors="replace").rstrip()
                lines.append(line)
                if line.strip():
                    log.info(f"[remote] {line}")
                    audit_emit(audit, "remote.log", wrk_uid=wrk_uid,
                               round_id=round_id, sub_uid=sub_uid,
                               line=self._cap_text(line, 4000),
                               message=self._cap_text(line, 4000))

        while True:
            if time.time() > deadline:
                try:
                    self._ssh_run(wrk_uid, f"pkill -f {shlex.quote(sh_f)}",
                                  timeout=30)
                except Exception:
                    pass
                audit_emit(audit, "remote.command.timeout", wrk_uid=wrk_uid,
                           round_id=round_id, sub_uid=sub_uid,
                           timeout_s=timeout, line_count=len(lines))
                raise subprocess.TimeoutExpired(f"sn125 {cli_args}", timeout)
            try:
                poll = self._ssh_run(
                    wrk_uid,
                    f"tail -c +{offset + 1} {shlex.quote(log_f)} 2>/dev/null; "
                    f"printf 'SN125_RC:'; cat {shlex.quote(rc_f)} 2>/dev/null; true",
                    timeout=120)
                if poll.returncode != 0:
                    raise RuntimeError(f"poll rc={poll.returncode}")
                body, _, rc_part = poll.stdout.rpartition(b"SN125_RC:")
                offset += len(body)
                _drain(body)
                fails = 0
                rc_text = rc_part.strip().decode(errors="replace")
                if rc_text:
                    rc = int(rc_text)
                    break
            except Exception as e:
                fails += 1
                log.warning(f"[{wrk_uid}] poll {fails}/{self.POLL_FAIL_LIMIT} failed: {e}")
                if fails >= self.POLL_FAIL_LIMIT:
                    audit_emit(audit, "remote.command.lost", wrk_uid=wrk_uid,
                               round_id=round_id, sub_uid=sub_uid,
                               poll_failures=fails, line_count=len(lines))
                    raise RuntimeError(
                        f"box unreachable for {fails} consecutive polls "
                        f"during detached run {run_id}") from e
            time.sleep(self.POLL_INTERVAL_S)

        if pending.strip():
            _drain(b"\n")
        audit_emit(audit, "remote.command.finish", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid,
                   rc=rc, line_count=len(lines),
                   crashed=(rc not in (0, None)))
        return "\n".join(lines), rc

    def _upload_checkpoints(self, wrk_uid: str, round_id: str, sub_uid: str,
                            result: dict, audit=None) -> None:
        """Publish the submission's q25/q50/q75/final safetensors to R2 (§6.3).

        The blobs live on the (still-alive, hostile) box; rather than pull GBs back
        through the validator or hand the box long-lived credentials, the validator
        mints a short-TTL, write-only PRESIGNED PUT URL per checkpoint and the box
        ``curl``s the blob straight up. Each record is then stamped with its public
        ``uri`` and the box-local ``path`` is dropped (it must never reach the signed,
        published bundle). Best-effort: any failure is logged, never raised — a
        publish hiccup must not fail the eval or block teardown.
        """
        curves = ((result or {}).get("curve_data") or {}).get("curves") or {}
        records = [(tid, rec) for tid, cd in curves.items()
                   for rec in (cd.get("checkpoints") or []) if rec.get("path")]
        if not records:
            return
        bucket = settings.r2_config().get("bucket")
        if not settings.r2_configured():
            for _tid, rec in records:
                rec.pop("path", None)
            log.info("  R2 not configured — skipping checkpoint upload")
            return
        try:
            s3 = settings.make_r2_client()
        except Exception as e:
            for _tid, rec in records:
                rec.pop("path", None)
            log.warning(f"[{wrk_uid}] checkpoint upload skipped (boto3 init failed): {e}")
            return
        uploaded = 0
        for tid, rec in records:
            path = rec.pop("path", None)
            if not path:
                continue
            try:
                pct_tag = int(round(float(rec.get("pct", 0.0)) * 100))
            except (TypeError, ValueError):
                pct_tag = 0
            key = f"checkpoints/{round_id}/{sub_uid}/{tid}/q{pct_tag:02d}.safetensors"
            try:
                url = s3.generate_presigned_url(
                    "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=900)
            except Exception as e:
                audit_emit(audit, "targon.checkpoint.presign_failed", wrk_uid=wrk_uid,
                           round_id=round_id, sub_uid=sub_uid, key=key, error=str(e))
                continue
            want_sha = str(rec.get("sha256", "") or "")
            sha_check = (f'[ "$(sha256sum {shlex.quote(path)} | cut -d\' \' -f1)" = '
                         f'{shlex.quote(want_sha)} ] && ') if want_sha else ""
            cmd = (f"[ ! -L {shlex.quote(path)} ] && [ -f {shlex.quote(path)} ] && "
                   f"{sha_check}"
                   f"curl -fsS --connect-timeout 30 --max-time 1800 -T "
                   f"{shlex.quote(path)} {shlex.quote(url)} >/dev/null "
                   f"&& echo SN125_CKPT_OK || echo SN125_CKPT_FAIL")
            try:
                r = self._ssh_run(wrk_uid, cmd, timeout=1900)
                out = r.stdout or b""
                ok = b"SN125_CKPT_OK" in out and b"SN125_CKPT_FAIL" not in out
            except Exception as e:
                audit_emit(audit, "targon.checkpoint.upload_failed", wrk_uid=wrk_uid,
                           round_id=round_id, sub_uid=sub_uid, key=key, error=str(e))
                continue
            if ok:
                rec["uri"] = f"r2://{bucket}/{key}"
                uploaded += 1
            else:
                audit_emit(audit, "targon.checkpoint.upload_failed", wrk_uid=wrk_uid,
                           round_id=round_id, sub_uid=sub_uid, key=key,
                           stderr=(r.stderr or b"")[-500:].decode("utf-8", "replace"))
        audit_emit(audit, "targon.checkpoints.uploaded", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, uploaded=uploaded, total=len(records))
        log.info(f"[{wrk_uid}] Uploaded {uploaded}/{len(records)} checkpoints to R2")

    def _parse_prod_eval_result(self, stdout: str, rc: int) -> dict:
        """Parse the strict prod-eval JSON line."""
        for line in reversed(stdout.splitlines()):
            s = line.strip()
            if not s.startswith("PROD_EVAL_RESULT "):
                continue
            try:
                result = json.loads(s[len("PROD_EVAL_RESULT "):])
            except json.JSONDecodeError:
                break
            if rc != 0:
                result["failed"] = True
                result.setdefault("error", f"prod-eval failed (rc={rc})")
            result.setdefault("score", -1.0)
            result.setdefault("components", {})
            return result
        return {
            "failed": True,
            "score": -1.0,
            "components": {},
            "error": f"prod-eval failed (rc={rc}): {stdout[-500:]}",
        }

    def _prod_shard_env(self) -> dict:
        """Remote shard-staging env. Do not forward a local SN125_FINEWEB_DIR.

        Secret hygiene: the worker box runs MINER code, so only a READ-scoped HF
        token may ever reach it (needed to download the shard dataset). The
        preferred source is SN125_FINEWEB_HF_READ_TOKEN; HF_TOKEN /
        HUGGINGFACE_HUB_TOKEN are forwarded as-is (operator must scope them
        read-only). HF_WRITE_TOKEN is NEVER forwarded — uploads from the box go
        through short-TTL presigned PUT URLs instead (_upload_checkpoints)."""
        load_dotenv()
        env = {"HF_HUB_DISABLE_XET": "1"}
        remote_dir = os.environ.get("SN125_TARGON_FINEWEB_DIR", "").strip()
        if remote_dir:
            env["SN125_FINEWEB_DIR"] = remote_dir
        if self.expected_manifest:
            env["SN125_FINEWEB_MANIFEST_HASH"] = self.expected_manifest
        for k in (
            "SN125_FINEWEB_HF_REPO",
            "SN125_FINEWEB_HF_REPO_TYPE",
            "SN125_FINEWEB_HF_REVISION",
            "SN125_FINEWEB_HF_SUBDIR",
            "SN125_FINEWEB_BUILD_IF_MISSING",
            "HF_HOME",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        ):
            v = os.environ.get(k, "").strip()
            if v:
                env[k] = v
        read_token = os.environ.get("SN125_FINEWEB_HF_READ_TOKEN", "").strip()
        if read_token:
            env["HF_TOKEN"] = read_token
        if "HF_TOKEN" not in env and os.environ.get("HF_WRITE_TOKEN", "").strip():
            log.warning("HF_WRITE_TOKEN is set but will NOT be forwarded to the "
                        "worker box; set SN125_FINEWEB_HF_READ_TOKEN (read-only "
                        "scope) if the shard repo is private.")
        return env

    def _stage_prod_shards(self, wrk_uid: str, *, audit=None,
                           round_id: str = "", sub_uid: str = "") -> dict:
        """Verify/download the production shards on a B200 worker."""
        cli = "stage-shards"
        if self.expected_manifest:
            cli += f" --expected-manifest {shlex.quote(self.expected_manifest)}"
        timeout = int(os.environ.get("SN125_FINEWEB_STAGE_TIMEOUT", "3600"))
        stdout, rc = self._run_sn125_cmd(
            wrk_uid, cli, timeout, env=self._prod_shard_env(),
            audit=audit, round_id=round_id, sub_uid=sub_uid)
        parsed = {
            "ok": rc == 0,
            "rc": rc,
            "stdout_tail": self._cap_text(stdout[-4000:], 4000),
        }
        for line in reversed(stdout.splitlines()):
            s = line.strip()
            if s.startswith("STAGE_SHARDS_RESULT "):
                try:
                    parsed.update(json.loads(s[len("STAGE_SHARDS_RESULT "):]))
                except json.JSONDecodeError:
                    pass
                break
        if rc != 0:
            raise RuntimeError(parsed.get("error") or f"stage-shards failed rc={rc}")
        return parsed

    @staticmethod
    def _net_lockdown_enabled() -> bool:
        """Egress lockdown is ON by default for prod. Operators can disable it for
        a supervised bring-up run with ``SN125_NET_LOCKDOWN=0``, but production
        evals MUST keep it enabled: fail-closed means miner code never runs on a
        box that can still reach the network."""
        return settings.net_lockdown_enabled()

    @staticmethod
    def _net_probe_hosts() -> list[tuple[str, str]]:
        """Public host:port pairs the post-lockdown probe tries to reach. Uses raw
        IPs so the probe tests routing, not DNS (a blocked box may fail DNS too,
        but a reachable box with broken DNS must still be caught)."""
        raw = os.environ.get("SN125_NET_LOCKDOWN_PROBE",
                             "1.1.1.1:443,8.8.8.8:443").strip()
        hosts: list[tuple[str, str]] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            h, _, p = tok.partition(":")
            if h:
                hosts.append((h, p or "443"))
        return hosts or [("1.1.1.1", "443")]

    def _lockdown_network(self, wrk_uid: str, *, round_id: str = "",
                          sub_uid: str = "", audit=None) -> dict:
        """Cut miner-initiated egress on the box, then PROVE egress is dead.

        C1/C2 boundary (DESIGN §5.1), scoped to what still matters once box escape
        is out of scope: the box holds only this run's code + training shards and is
        destroyed after the eval, so the one control that counts while miner code
        runs is that it cannot reach the network — no exfiltration of the held-out
        set (materialized on-box for the clean score), no external warm-start / code
        pull that would defeat the source-size cap (C8). Egress is dropped with
        loopback + ESTABLISHED,RELATED preserved so the Targon SSH control/polling
        channel (inbound-initiated) keeps working while every NEW outbound
        connection is refused.

        FAIL-CLOSED: raises if the firewall could not be applied OR a positive probe
        can still open a TCP connection to a public host. The caller MUST let that
        propagate so the miner optimizer never executes on a networked box.
        """
        allow_cidrs = [c.strip() for c in
                       os.environ.get("SN125_NET_LOCKDOWN_ALLOW_CIDRS", "").split(",")
                       if c.strip()]
        allow_rules = "".join(
            f"iptables -A OUTPUT -d {shlex.quote(c)} -j ACCEPT || "
            "{ echo NET_LOCKDOWN_FAIL allowlist; exit 0; }\n"
            for c in allow_cidrs)
        probe_hosts = self._net_probe_hosts()
        probe_pairs = " ".join(f"{h}/{p}" for h, p in probe_hosts)
        script = (
            "set +e\n"
            "command -v iptables >/dev/null 2>&1 || { echo NET_LOCKDOWN_FAIL no_iptables; exit 0; }\n"
            "iptables -A OUTPUT -o lo -j ACCEPT || { echo NET_LOCKDOWN_FAIL lo; exit 0; }\n"
            "iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT || "
            "{ echo NET_LOCKDOWN_FAIL established; exit 0; }\n"
            f"{allow_rules}"
            "iptables -P OUTPUT DROP || { echo NET_LOCKDOWN_FAIL policy; exit 0; }\n"
            "echo NET_LOCKDOWN_APPLIED\n"
            "blocked=1\n"
            f"for hp in {probe_pairs}; do\n"
            "  h=${hp%/*}; p=${hp#*/}\n"
            "  if timeout 8 bash -c \"exec 3<>/dev/tcp/$h/$p\" 2>/dev/null; then\n"
            "    echo NET_PROBE_REACHED $h:$p; blocked=0\n"
            "  fi\n"
            "done\n"
            "echo NET_PROBE_RESULT blocked=$blocked\n"
        )
        try:
            r = self._ssh_run(wrk_uid, "bash -s", input_data=script.encode(),
                              timeout=90)
            out = (r.stdout or b"").decode(errors="replace")
            rc = r.returncode
        except Exception as e:
            audit_emit(audit, "targon.net_lockdown.error", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid,
                       error=str(e), error_type=type(e).__name__)
            raise RuntimeError(
                f"network lockdown could not be applied ({e}); fail-closed — "
                "refusing to run miner code with reachable egress") from e
        applied = "NET_LOCKDOWN_APPLIED" in out
        egress_blocked = "NET_PROBE_RESULT blocked=1" in out
        info = {"applied": applied, "egress_blocked": egress_blocked, "rc": rc,
                "probe_hosts": [f"{h}:{p}" for h, p in probe_hosts],
                "output": self._cap_text(out, 2000)}
        if not (applied and egress_blocked):
            audit_emit(audit, "targon.net_lockdown.failed", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid, **info)
            raise RuntimeError(
                "network lockdown fail-closed: refusing to run miner code with "
                f"reachable egress (applied={applied} egress_blocked={egress_blocked}); "
                f"out={self._cap_text(out, 400)!r}")
        audit_emit(audit, "targon.net_lockdown.applied", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, **info)
        return info

    def _sandbox_guard(self, wrk_uid: str, *, round_id: str = "",
                       sub_uid: str = "", audit=None) -> dict:
        """Container-host replacement for the iptables lockdown: PROVE on this
        box that the seccomp sandbox flavour engages — the filter installs
        as an unprivileged process, AF_INET/AF_INET6 sockets are refused with
        EPERM, AF_UNIX still works (the CUDA-IPC optimizer process needs it)
        and a forked child inherits the denial. Runs the same code path the
        worker will run (``python -m sn125.seccomp_sandbox``), as root here
        and as nobody in the worker.

        FAIL-CLOSED: raises unless the self-test prints ``"ok": true``. The
        worker's own attestation (``_enforce_sandbox_attestation``) is the
        second, per-run proof."""
        cmd = ("cd /workspace && PYTHONPATH=/workspace timeout 120 "
               "python -m sn125.seccomp_sandbox 2>&1; echo SN125_GUARD_RC=$?")
        try:
            r = self._ssh_run(wrk_uid, cmd, timeout=180)
            out = (r.stdout or b"").decode(errors="replace")
        except Exception as e:
            audit_emit(audit, "targon.sandbox_guard.error", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid,
                       error=str(e), error_type=type(e).__name__)
            raise RuntimeError(
                f"network lockdown (seccomp guard) could not run ({e}); fail-closed — "
                "refusing to run miner code on a container without a proven filter") from e
        parsed: dict = {}
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
        ok = bool(parsed.get("ok")) and bool(parsed.get("net_denied")) \
            and bool(parsed.get("child_inherits"))
        info = {"mode": "process-seccomp", "applied": ok, "egress_blocked": ok,
                "guard": parsed, "output": self._cap_text(out, 2000)}
        if not ok:
            audit_emit(audit, "targon.sandbox_guard.failed", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid, **info)
            raise RuntimeError(
                "network lockdown fail-closed (seccomp guard): the syscall filter did "
                f"not prove egress denial on this box (guard={parsed!r}); "
                f"out={self._cap_text(out, 400)!r}")
        audit_emit(audit, "targon.sandbox_guard.applied", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, **info)
        return info

    def _restore_network(self, wrk_uid: str, *, round_id: str = "",
                         sub_uid: str = "", audit=None) -> None:
        """Re-open egress (best-effort) so post-eval checkpoint upload can PUT to
        R2. Runs only after the miner run has fully finished; never raises.
        No-op on container hosts (nothing was applied at box level)."""
        if self._provider_isolation(wrk_uid) == "container":
            audit_emit(audit, "targon.net_restore.done", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid, ok=True,
                       mode="process-seccomp")
            return
        script = ("set +e\n"
                  "iptables -P OUTPUT ACCEPT 2>/dev/null\n"
                  "iptables -F OUTPUT 2>/dev/null\n"
                  "echo NET_RESTORE_DONE\n")
        try:
            r = self._ssh_run(wrk_uid, "bash -s", input_data=script.encode(),
                              timeout=30)
            ok = b"NET_RESTORE_DONE" in (r.stdout or b"")
        except Exception as e:
            audit_emit(audit, "targon.net_restore.error", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid, error=str(e))
            return
        audit_emit(audit, "targon.net_restore.done", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, ok=ok)

    def _provision_and_run(self, wrk_uid: str, source: str, round_id: str,
                           sub_uid: str, timeout: int, mode: str = "prod",
                           audit=None, baseline_bundle: dict | None = None) -> dict:
        """SSH into rental, provision, run eval, fetch results."""
        stage_timings: dict[str, float] = {}

        def stage(name: str, fn, *, result_field: str = "", pre_eval: bool = True):
            started = time.time()
            audit_emit(audit, "targon.stage.start", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid, stage=name)
            try:
                result_value = fn()
            except Exception as e:
                elapsed = self._stage_done(started)
                stage_timings[f"{name}_s"] = elapsed
                audit_emit(audit, "targon.stage.error", wrk_uid=wrk_uid,
                           round_id=round_id, sub_uid=sub_uid, stage=name,
                           elapsed_s=elapsed, error=str(e),
                           error_type=type(e).__name__,
                           crashed=looks_like_crash(str(e)))
                if pre_eval and not isinstance(e, subprocess.TimeoutExpired):
                    raise RuntimeError(
                        f"[pre-eval infra] stage {name} failed: {e}") from e
                raise
            elapsed = self._stage_done(started)
            stage_timings[f"{name}_s"] = elapsed
            fields = {}
            if result_field:
                fields[result_field] = result_value
            audit_emit(audit, "targon.stage.finish", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid, stage=name,
                       elapsed_s=elapsed, **fields)
            return result_value

        stage("ssh_wait", lambda: self._wait_ssh(wrk_uid))
        audit_emit(audit, "targon.ssh.ready", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid)
        fingerprint = stage("fingerprint",
                            lambda: self._remote_fingerprint(wrk_uid),
                            result_field="fingerprint")
        audit_emit(audit, "targon.fingerprint", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid,
                   fingerprint=fingerprint)
        log.info(f"[{wrk_uid}] Uploading project via tar...")
        stage("upload", lambda: self._upload_project(wrk_uid))
        log.info(f"[{wrk_uid}] Running setup...")
        stage("setup", lambda: self._provision(wrk_uid))
        post_setup_fingerprint = stage("fingerprint_post_setup",
                                       lambda: self._remote_fingerprint(wrk_uid),
                                       result_field="fingerprint")
        audit_emit(audit, "targon.fingerprint_post_setup", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid,
                   fingerprint=post_setup_fingerprint)
        box_probe = None
        if mode == "prod" and settings.env_truthy(settings.BOX_PROBE_ENV, default=True):
            box_probe = stage("box_probe",
                              lambda: self._throughput_gate(
                                  wrk_uid, audit=audit, round_id=round_id,
                                  sub_uid=sub_uid),
                              result_field="box_probe")
        predownload = stage("predownload",
                            lambda: self._pre_download_models(wrk_uid, mode),
                            result_field="predownload")
        data_stage = None
        if mode == "prod":
            data_stage = stage("data_stage",
                               lambda: self._stage_prod_shards(
                                   wrk_uid, audit=audit, round_id=round_id, sub_uid=sub_uid),
                               result_field="data_stage")

        source_bytes = source.encode()
        stage("submission_upload", lambda: self._ssh_run(
            wrk_uid, "cat > /tmp/sub.py", input_data=source_bytes, timeout=30))
        if mode == "prod" and baseline_bundle:
            baseline_bytes = json.dumps(baseline_bundle, default=float).encode()
            stage("baseline_upload", lambda: self._ssh_run(
                wrk_uid, "cat > /tmp/baseline.json", input_data=baseline_bytes, timeout=30))
            audit_emit(audit, "targon.baseline_uploaded", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid,
                       baseline_bytes=len(baseline_bytes),
                       task_ids=sorted((baseline_bundle.get("baseline_curves") or {}).keys()))
        audit_emit(audit, "targon.submission_uploaded", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid,
                   source_bytes=len(source_bytes),
                   source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                   predownload=predownload, data_stage=data_stage)
        log.info(f"[{wrk_uid}] Running eval (mode={mode}, timeout={timeout}s)...")
        eval_env = None
        worker_log = ""
        if mode == "prod":
            _log_id = hashlib.sha256(f"{round_id}|{sub_uid}".encode()).hexdigest()[:18]
            worker_log = f"/workspace/logs/prod_worker_{_log_id}.jsonl"
            checkpoint_out = f"/workspace/sn125/rounds/ckpt_{_log_id}/prod_checkpoint_{_log_id}.safetensors"
            cli = (
                "prod-eval /tmp/sub.py "
                f"--worker-log {shlex.quote(worker_log)} "
                f"--checkpoint-out {shlex.quote(checkpoint_out)} "
                "--out /workspace/sn125/rounds/prod_eval_result.json"
            )
            prod_eval_steps = os.environ.get("SN125_PROD_EVAL_STEPS", "").strip()
            prod_eval_budget = os.environ.get("SN125_PROD_EVAL_BUDGET", "").strip()
            prod_eval_timeout = os.environ.get("SN125_PROD_EVAL_TIMEOUT", "").strip()
            if prod_eval_steps:
                cli += f" --total-steps {shlex.quote(prod_eval_steps)}"
            if prod_eval_budget:
                cli += f" --budget {shlex.quote(prod_eval_budget)}"
            if prod_eval_timeout:
                cli += f" --timeout {shlex.quote(prod_eval_timeout)}"
            if self.expected_manifest:
                cli += f" --expected-manifest {shlex.quote(self.expected_manifest)}"
            if baseline_bundle:
                cli += " --baseline-json /tmp/baseline.json"
            eval_env = self._prod_shard_env()
            eval_env.update(settings.launch_eval_env())
            eval_env.setdefault("SN125_WORKER_LOG_PATH", worker_log)
            if self._net_lockdown_enabled():
                eval_env.setdefault("HF_HUB_OFFLINE", "1")
                eval_env.setdefault("TRANSFORMERS_OFFLINE", "1")
            audit_emit(audit, "targon.prod_eval.env",
                       wrk_uid=wrk_uid, round_id=round_id, sub_uid=sub_uid,
                       optproc=True,
                       optproc_step_timeout=eval_env.get("SN125_OPTPROC_STEP_TIMEOUT", ""),
                       worker_log=worker_log,
                       checkpoint_out=checkpoint_out,
                       prod_eval_steps=prod_eval_steps,
                       prod_eval_budget=prod_eval_budget,
                       prod_eval_timeout=prod_eval_timeout)
        else:
            raise ValueError(f"unsupported eval mode {mode!r} (production prod-eval only)")
        isolation = self._provider_isolation(wrk_uid)
        if mode == "prod":
            eval_env[settings.SANDBOX_MODE_ENV] = (
                "seccomp" if isolation == "container" else "namespaces")
        lockdown_info = None
        if mode == "prod" and self._net_lockdown_enabled():
            if isolation == "container":
                lockdown_info = stage(
                    "sandbox_guard",
                    lambda: self._sandbox_guard(
                        wrk_uid, round_id=round_id, sub_uid=sub_uid, audit=audit),
                    result_field="net_lockdown")
            else:
                lockdown_info = stage(
                    "net_lockdown",
                    lambda: self._lockdown_network(
                        wrk_uid, round_id=round_id, sub_uid=sub_uid, audit=audit),
                    result_field="net_lockdown")
        stdout, rc = stage("eval", lambda: self._run_sn125_cmd(
            wrk_uid, cli, timeout, env=eval_env,
            audit=audit, round_id=round_id, sub_uid=sub_uid), pre_eval=False)
        if rc != 0:
            log.error(f"[{wrk_uid}] Eval failed (rc={rc})")
        result = self._parse_prod_eval_result(stdout, rc)
        result["stage_timings"] = dict(stage_timings)
        result["isolation"] = isolation
        result["sandbox_mode"] = eval_env.get(settings.SANDBOX_MODE_ENV, "")
        if lockdown_info is not None:
            result["net_lockdown"] = {k: v for k, v in lockdown_info.items()
                                      if k != "output"}
        result["hardware_fingerprint"] = fingerprint
        result["hardware_fingerprint_post_setup"] = post_setup_fingerprint
        if box_probe is not None:
            result["box_probe"] = box_probe
        achieved = self._achieved_steps_per_s(result)
        if achieved is not None:
            result["achieved_steps_per_s"] = achieved
        if worker_log:
            result["worker_log"] = worker_log
        if (mode == "prod" and box_probe is not None
                and not result.get("failed")
                and settings.env_truthy(settings.BOX_PROBE_DRIFT_ENV, default=True)):
            result["box_probe_post"] = stage(
                "box_probe_post",
                lambda: self._drift_gate(wrk_uid, box_probe, audit=audit,
                                         round_id=round_id, sub_uid=sub_uid),
                result_field="box_probe_post", pre_eval=False)
        audit_emit(audit, "targon.eval.parsed", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, rc=rc,
                   failed=bool(result.get("failed")),
                   score=result.get("score"), error=result.get("error", ""),
                   worker_log=worker_log,
                   crashed=rc != 0 or looks_like_crash(result.get("error", "")))
        cd = result.get("curve_data") or {}
        if cd:
            audit_emit(audit, "targon.curves.fetched", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid,
                       elapsed_s=0.0,
                       curve_keys=sorted((cd.get("curves") or {}).keys()),
                       baseline_keys=sorted((cd.get("baseline_curves") or {}).keys()))
        return result

    def _run_box_probe(self, wrk_uid: str, *, audit=None, round_id: str = "",
                       sub_uid: str = "") -> dict:
        """Run the standardized box probe on the rental and parse its result.

        The probe must run under the production compile regime — the same
        knobs launch_eval_env() gives the real eval — or the band/floor would
        be calibrated against a different kernel set. Raises (with the flake
        marker supplied by the caller's context) if the probe crashes or emits
        no parseable result: that is a box fault, never the miner's."""
        from .boxprobe import parse_result
        probe_env = {k: v for k, v in settings.launch_eval_env().items()
                     if k in ("SN125_LEAN_COMPILE", "SN125_LEAN_LOSS_CKPT",
                              "SN125_LEAN_LOSS_CHUNKS")}
        stdout, rc = self._run_sn125_cmd(
            wrk_uid, "box-probe", timeout=1800, env=probe_env,
            audit=audit, round_id=round_id, sub_uid=sub_uid)
        probe = parse_result(stdout)
        if rc != 0 or not probe or "steps_per_s" not in probe:
            raise RuntimeError(f"probe failed on this box "
                               f"(rc={rc}, parsed={bool(probe)})")
        return probe

    def _throughput_gate(self, wrk_uid: str, audit=None, round_id: str = "",
                         sub_uid: str = "") -> dict:
        """Pre-eval hardware gate: run the standardized box probe
        (sn125/boxprobe.py) and enforce the operator-pinned throughput BAND.

        The eval is a fixed wall-clock budget, so box speed converts directly
        into steps and therefore loss. The band is two-sided on purpose:

        * too SLOW (< floor): the box starves every LR schedule miners planned
          against total_steps;
        * too FAST (> ceiling): a lucky box inflates the score, and because
          the frontier is a ratchet, one out-of-band fast box would set a
          rolling-best that honest miners on in-band hardware can never beat
          (and pay its miner legacy emissions indefinitely). The ceiling also
          closes the multi-provider lottery: with mixed fleets, miners would
          otherwise time submissions to land on a systematically faster fleet.

        Band = SN125_BOX_PROBE_REF_STS * (1 ± SN125_BOX_PROBE_BAND_PCT/100).
        The legacy SN125_BOX_PROBE_MIN_STS floor is still honored (the tighter
        floor wins). With neither pinned the gate is measure-only telemetry.

        Always returns the measurement dict (recorded in the audit trail and
        the eval result). An out-of-band box raises with a
        ``box_throughput_floor`` / ``box_throughput_ceiling`` flake marker so
        the round driver relaunches the SAME submission on a fresh box (never
        a miner DQ: miner code has not run yet), and the owning provider is
        struck (FailoverClient) so a systematically out-of-band fleet fails
        over instead of relaunching forever.
        """
        try:
            probe = self._run_box_probe(wrk_uid, audit=audit,
                                        round_id=round_id, sub_uid=sub_uid)
        except RuntimeError as e:
            self._note_box_rejected(wrk_uid)
            raise RuntimeError(f"box_throughput_floor: {e} — rejecting box") from e
        floor = settings.env_float(settings.BOX_PROBE_MIN_STS_ENV, 0.0, minimum=0.0)
        ref = settings.box_probe_ref_sts()
        band_pct = settings.box_probe_band_pct()
        ceiling = 0.0
        if ref > 0.0:
            floor = max(floor, ref * (1.0 - band_pct / 100.0))
            ceiling = ref * (1.0 + band_pct / 100.0)
        probe["floor"] = floor
        probe["ceiling"] = ceiling
        probe["band"] = ({"ref_sts": ref, "band_pct": band_pct,
                          "min_sts": floor, "max_sts": ceiling}
                         if ref > 0.0 else None)
        audit_emit(audit, "targon.box_probe", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, probe=probe,
                   provider=self._provider_name(wrk_uid),
                   enforced=floor > 0.0 or ceiling > 0.0)
        sts = float(probe["steps_per_s"])
        log.info(f"[{wrk_uid}] Box probe: {sts:.3f} st/s on "
                 f"{probe.get('gpu', '?')} "
                 f"(band: {f'{floor:.3f}..{ceiling:.3f}' if ceiling else (floor or 'measure-only')})")
        if floor > 0.0 and sts < floor:
            self._note_box_rejected(wrk_uid)
            raise RuntimeError(
                f"box_throughput_floor: {sts:.3f} st/s < pinned floor {floor:.3f} "
                f"({probe.get('gpu', '?')}) — rejecting box")
        if ceiling > 0.0 and sts > ceiling:
            self._note_box_rejected(wrk_uid)
            raise RuntimeError(
                f"box_throughput_ceiling: {sts:.3f} st/s > pinned ceiling "
                f"{ceiling:.3f} ({probe.get('gpu', '?')}) — a faster-than-band box "
                f"would inflate the fixed-budget score; rejecting box")
        self._note_box_ok(wrk_uid)
        return probe

    def _drift_gate(self, wrk_uid: str, pre_probe: dict, *, audit=None,
                    round_id: str = "", sub_uid: str = "") -> dict:
        """Post-eval drift check: re-run the SAME probe on the SAME box and
        compare against the pre-eval reading.

        The pre-eval probe is a point sample — a box can pass it and then
        deliver less (thermal throttling, noisy neighbor) or more (a host
        that sandbagged the probe to sneak a fast box past the ceiling)
        compute during the 20h run. Pre-vs-post on the same box + workload is
        apples-to-apples, unlike probe-vs-harness rates which differ
        systematically. A deviation beyond the band tolerance (either
        direction) quarantines the finished run: the raise carries the
        ``box_throughput_drift`` flake marker, so the same submission
        relaunches on a fresh box and the anomalous score is never published.
        """
        band_pct = settings.box_probe_band_pct()
        try:
            post = self._run_box_probe(wrk_uid, audit=audit,
                                       round_id=round_id, sub_uid=sub_uid)
        except RuntimeError as e:
            self._note_box_rejected(wrk_uid)
            raise RuntimeError(
                f"box_throughput_drift: post-eval probe failed ({e}) — "
                f"quarantining run") from e
        pre_sts = float(pre_probe.get("steps_per_s") or 0.0)
        post_sts = float(post["steps_per_s"])
        drift = (post_sts - pre_sts) / pre_sts if pre_sts > 0 else 0.0
        post["drift_vs_pre"] = drift
        post["tolerance_pct"] = band_pct
        audit_emit(audit, "targon.box_probe_post", wrk_uid=wrk_uid,
                   round_id=round_id, sub_uid=sub_uid, probe=post,
                   pre_steps_per_s=pre_sts, drift_vs_pre=drift,
                   tolerance_pct=band_pct,
                   provider=self._provider_name(wrk_uid))
        log.info(f"[{wrk_uid}] Post-eval probe: {post_sts:.3f} st/s "
                 f"(pre {pre_sts:.3f}, drift {drift * 100:+.1f}%, "
                 f"tol ±{band_pct:.1f}%)")
        if pre_sts > 0 and abs(drift) * 100.0 > band_pct:
            self._note_box_rejected(wrk_uid)
            raise RuntimeError(
                f"box_throughput_drift: post-eval probe {post_sts:.3f} st/s "
                f"deviates {drift * 100:+.1f}% from pre-eval {pre_sts:.3f} "
                f"(tol ±{band_pct:.1f}%) — box speed changed during the run; "
                f"quarantining run, relaunching on a fresh box")
        return post

    @staticmethod
    def _achieved_steps_per_s(result: dict) -> float | None:
        """Telemetry: the submission's realized end-to-end training rate,
        derived from its own curve (final step / wall seconds). NOT enforced —
        a slower rate is legitimately the miner's own optimizer cost under the
        fixed wall-clock budget — but published for auditability alongside the
        probe readings."""
        curves = ((result or {}).get("curve_data") or {}).get("curves") or {}
        for cd in curves.values():
            pts = cd.get("eval_points") or []
            try:
                ws = float(cd.get("wall_seconds") or 0.0)
                if pts and ws > 0:
                    return float(pts[-1][0]) / ws
            except (TypeError, ValueError, IndexError):
                continue
        return None

    def _get_setup_script(self) -> str:
        """Return the per-rental provisioning script."""
        script_path = Path(__file__).parent / "scripts" / "setup_b200.sh"
        if script_path.exists():
            return script_path.read_text()
        return _DEFAULT_SETUP_SCRIPT

    def evaluate_submission(self, source: str, round_id: str, sub_uid: str,
                            mode: str = "prod", audit=None,
                            baseline_bundle: dict | None = None) -> dict:
        """Full per-submission lifecycle: inventory check, create, deploy, SSH, eval, teardown."""
        import os as _os
        _ks = "/root/.sn125/cost_kill_switch"
        if _os.path.exists(_ks):
            audit_emit(audit, "targon.eval.refused", round_id=round_id,
                       sub_uid=sub_uid, reason="cost_kill_switch")
            return {"failed": True, "error": f"[CRITICAL] External cost watchdog kill-switch set ({_ks}). Refusing new rental. Inspect file to see trip reason; remove file only after operator approval and account top-up.", "score": -1.0}
        expected_cost = ((self.timeout + SUBMISSION_MARGIN) / 3600) * self._cost_per_hour(self.resource)
        if not self._reserve_spend(expected_cost):
            audit_emit(audit, "targon.eval.refused", round_id=round_id,
                       sub_uid=sub_uid, reason="daily_spend_limit",
                       daily_spend=self.daily_spend,
                       committed_spend=self.committed_spend, limit=DAILY_SPEND_LIMIT)
            return {"failed": True, "error": f"[CRITICAL] Daily spend ${self.daily_spend:.0f} + in-flight reservations = ${self.committed_spend:.0f} >= ${DAILY_SPEND_LIMIT}. Refusing new rental.", "score": -1.0}

        wrk_uid = None
        resource_used = self.resource
        rental_start = None
        heartbeat_stop = None
        result: dict | None = None
        wl_name = ""
        cost_per_hour = self._cost_per_hour(resource_used)

        self._semaphore.acquire()
        try:
            inventory_started = time.time()
            audit_emit(audit, "targon.inventory.check_start",
                       round_id=round_id, sub_uid=sub_uid, resource=self.resource)
            resource_used = self._check_inventory()
            cost_per_hour = self._cost_per_hour(resource_used)
            audit_emit(audit, "targon.inventory.available",
                       round_id=round_id, sub_uid=sub_uid,
                       resource=resource_used, cost_per_hour=cost_per_hour,
                       elapsed_s=self._stage_done(inventory_started))

            for attempt in range(2):
                import hashlib as _hl
                _digest = _hl.sha256(f"{round_id}|{sub_uid}".encode()).hexdigest()[:18]
                wl_name = f"sn125-{_digest}"
                envs = [
                    {"name": "SN125_ROUND_ID", "value": str(round_id)},
                    {"name": "SN125_SUB_UID", "value": str(sub_uid)},
                ]
                attempt_started = time.time()
                try:
                    audit_emit(audit, "targon.workload.create_start",
                               round_id=round_id, sub_uid=sub_uid,
                               attempt=attempt + 1, workload_name=wl_name,
                               resource=resource_used,
                               image=os.environ.get("SN125_TARGON_IMAGE", DEFAULT_TARGON_IMAGE))
                    resp = self.client.create_workload(wl_name, resource_used, self._ssh_key_uid, envs=envs)
                    wrk_uid = resp.get("uid", "")
                    if not wrk_uid:
                        raise RuntimeError(f"No UID in workload response: {resp}")
                    log.info(f"[{wrk_uid}] Created workload '{wl_name}' on {resource_used}")
                    audit_emit(audit, "targon.workload.created",
                               round_id=round_id, sub_uid=sub_uid,
                               attempt=attempt + 1, wrk_uid=wrk_uid,
                               workload_name=wl_name, resource=resource_used,
                               response=resp)

                    audit_emit(audit, "targon.workload.deploy_start",
                               wrk_uid=wrk_uid, round_id=round_id,
                               sub_uid=sub_uid, attempt=attempt + 1)
                    self.client.deploy_workload(wrk_uid)
                    audit_emit(audit, "targon.workload.deploy_requested",
                               wrk_uid=wrk_uid, round_id=round_id,
                               sub_uid=sub_uid, attempt=attempt + 1)
                    rental_start = time.time()
                    record = RentalRecord(wrk_uid, wl_name, resource_used,
                                          rental_start, cost_per_hour,
                                          provider=self._provider_name(wrk_uid))
                    self._rentals.append(record)
                    self.write_status_file()
                    heartbeat_stop = self._start_workload_heartbeat(
                        wrk_uid, record, audit=audit, round_id=round_id, sub_uid=sub_uid)

                    if self.client.wait_running(wrk_uid):
                        audit_emit(audit, "targon.workload.running",
                                   wrk_uid=wrk_uid, round_id=round_id,
                                   sub_uid=sub_uid, attempt=attempt + 1,
                                   elapsed_s=self._stage_done(attempt_started))
                        break
                    log.warning(f"[{wrk_uid}] Never reached RUNNING (attempt {attempt+1})")
                    audit_emit(audit, "targon.workload.not_running",
                               wrk_uid=wrk_uid, round_id=round_id,
                               sub_uid=sub_uid, attempt=attempt + 1,
                               elapsed_s=self._stage_done(attempt_started))
                    if heartbeat_stop is not None:
                        heartbeat_stop.set()
                        heartbeat_stop = None
                    self.client.delete_workload(wrk_uid)
                    wrk_uid = None
                except Exception as e:
                    log.error(f"Workload create/deploy failed (attempt {attempt+1}): {e}")
                    audit_emit(audit, "targon.workload.create_deploy_error",
                               wrk_uid=wrk_uid or "", round_id=round_id,
                               sub_uid=sub_uid, attempt=attempt + 1,
                               elapsed_s=self._stage_done(attempt_started),
                               error=str(e), error_type=type(e).__name__)
                    if wrk_uid:
                        if heartbeat_stop is not None:
                            heartbeat_stop.set()
                            heartbeat_stop = None
                        self.client.delete_workload(wrk_uid)
                        wrk_uid = None

            if not wrk_uid:
                audit_emit(audit, "targon.eval.provisioning_failed",
                           round_id=round_id, sub_uid=sub_uid)
                result = {"failed": True, "error": "rental_provisioning_failed", "score": -1.0}
                return result

            result = self._provision_and_run(
                wrk_uid, source, round_id, sub_uid, self.timeout, mode=mode,
                audit=audit, baseline_bundle=baseline_bundle)
            if result.get("failed"):
                diag = self._capture_failure_diag(wrk_uid)
                result["remote_diag"] = diag
                audit_emit(audit, "targon.remote_diag", wrk_uid=wrk_uid,
                           round_id=round_id, sub_uid=sub_uid,
                           diagnostics=diag,
                           crashed=looks_like_crash(result.get("error", "")) or
                           looks_like_crash(diag))
            audit_emit(audit, "targon.eval.result", wrk_uid=wrk_uid,
                       round_id=round_id, sub_uid=sub_uid,
                       failed=bool(result.get("failed")),
                       score=result.get("score"), error=result.get("error", ""),
                       crashed=looks_like_crash(result.get("error", "")),
                       result=result)
            return result

        except subprocess.TimeoutExpired:
            log.error(f"[{wrk_uid}] Submission exceeded timeout+margin ({self.timeout}+{SUBMISSION_MARGIN}s)")
            diag = self._capture_failure_diag(wrk_uid) if wrk_uid else ""
            audit_emit(audit, "targon.eval.timeout", wrk_uid=wrk_uid or "",
                       round_id=round_id, sub_uid=sub_uid,
                       timeout_s=self.timeout, margin_s=SUBMISSION_MARGIN,
                       diagnostics=diag)
            result = {"failed": True, "error": "timeout_exceeded", "score": -1.0,
                      "remote_diag": diag}
            return result
        except Exception as e:
            log.error(f"[{wrk_uid or 'no-wrk'}] Eval failed: {e}")
            diag = self._capture_failure_diag(wrk_uid) if wrk_uid else ""
            audit_emit(audit, "targon.eval.exception", wrk_uid=wrk_uid or "",
                       round_id=round_id, sub_uid=sub_uid,
                       error=str(e), error_type=type(e).__name__,
                       diagnostics=diag, crashed=looks_like_crash(str(e)))
            result = {"failed": True, "error": str(e), "score": -1.0,
                      "remote_diag": diag}
            return result
        finally:
            self._release_reserved_spend(expected_cost)
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if wrk_uid:
                elapsed = time.time() - (rental_start or time.time())
                cost = (elapsed / 3600) * cost_per_hour
                self._track_cost(cost)
                if isinstance(result, dict):
                    result.update({
                        "wrk_uid": wrk_uid,
                        "workload_name": wl_name,
                        "resource": resource_used,
                        "provider": self._provider_name(wrk_uid),
                        "box_seconds": elapsed,
                        "est_cost": cost,
                        "cost_per_hour": cost_per_hour,
                        "daily_spend_after": self.daily_spend,
                    })
                    if mode == "prod" and not result.get("failed"):
                        if self._net_lockdown_enabled():
                            self._restore_network(wrk_uid, round_id=round_id,
                                                  sub_uid=sub_uid, audit=audit)
                        try:
                            self._upload_checkpoints(wrk_uid, round_id, sub_uid, result, audit)
                        except Exception as _e:
                            log.warning(f"[{wrk_uid}] checkpoint upload error (non-fatal): {_e}")
                    audit_emit(audit, "targon.result.metadata",
                               wrk_uid=wrk_uid, round_id=round_id,
                               sub_uid=sub_uid, result=result)
                log.info(f"[{wrk_uid}] Deleting workload. Elapsed: {elapsed:.0f}s, cost: ${cost:.2f}")
                audit_emit(audit, "targon.workload.delete_start",
                           wrk_uid=wrk_uid, round_id=round_id, sub_uid=sub_uid,
                           elapsed_s=elapsed, est_cost=cost,
                           daily_spend=self.daily_spend)
                self.client.delete_workload(wrk_uid)
                audit_emit(audit, "targon.workload.delete_requested",
                           wrk_uid=wrk_uid, round_id=round_id, sub_uid=sub_uid,
                           elapsed_s=elapsed, est_cost=cost,
                           daily_spend=self.daily_spend)
                for rec in self._rentals:
                    if rec.wrk_uid == wrk_uid:
                        rec.deleted = True
                        rec.cost_accrued = cost
                self.write_status_file()
            self._semaphore.release()

    def calibrate_remote(self, model: str = "Qwen/Qwen3-0.6B", seq: int = 2048,
                         batch: int = 16, steps: int = 200, warmup: int = 100,
                         target_hours: float = 20.0, timeout: int = 1800,
                         spend_ledger: str = "", empty_cache_every: int = 0,
                         profile: bool = False, flash: bool = False,
                         nondet: bool = False, compile_model: bool = False,
                         chunked_ce: int = 0, fp8: bool = False) -> dict:
        """§3 throughput probe: provision a b200-small, run the AdamW reference
        for ``steps`` steps at (model, seq, batch), parse the CALIBRATION_RESULT
        line, tear down. CHEAP (~minutes ≈ a few $). Box deleted in finally;
        cost tracked + optionally appended to the spend ledger JSON."""
        import os as _os
        if _os.path.exists("/root/.sn125/cost_kill_switch"):
            return {"failed": True, "error": "[CRITICAL] External cost watchdog kill-switch set."}
        if self.daily_spend >= DAILY_SPEND_LIMIT:
            return {"failed": True, "error": f"[CRITICAL] Daily spend ${self.daily_spend:.0f} >= ${DAILY_SPEND_LIMIT}."}

        wrk_uid = None
        resource_used = self.resource
        rental_start = None
        cost_per_hour = 20.0
        result = {"failed": True, "error": "not_started"}
        self._semaphore.acquire()
        try:
            resource_used = self._check_inventory()
            cost_per_hour = self._cost_per_hour(resource_used)
            for attempt in range(2):
                wl_name = f"sn125-calib-{int(time.time())}"[:63].lower()
                try:
                    resp = self.client.create_workload(wl_name, resource_used, self._ssh_key_uid)
                    wrk_uid = resp.get("uid", "")
                    if not wrk_uid:
                        raise RuntimeError(f"No UID: {resp}")
                    log.info(f"[{wrk_uid}] Created '{wl_name}' on {resource_used}")
                    self.client.deploy_workload(wrk_uid)
                    rental_start = time.time()
                    self._rentals.append(RentalRecord(wrk_uid, wl_name, resource_used, rental_start, cost_per_hour))
                    if self.client.wait_running(wrk_uid):
                        break
                    log.warning(f"[{wrk_uid}] Never RUNNING (attempt {attempt+1})")
                    self.client.delete_workload(wrk_uid)
                    wrk_uid = None
                except Exception as e:
                    log.error(f"Create/deploy failed (attempt {attempt+1}): {e}")
                    if wrk_uid:
                        self.client.delete_workload(wrk_uid)
                        wrk_uid = None
            if not wrk_uid:
                return {"failed": True, "error": "rental_provisioning_failed"}

            self._wait_ssh(wrk_uid)
            self._upload_project(wrk_uid)
            self._provision(wrk_uid)
            if not (model.startswith("lean-") or model.startswith("forge-")):
                self._ssh_run(wrk_uid, f"cd /workspace && PYTHONPATH=/workspace python -c "
                              f"\"from transformers import AutoConfig; AutoConfig.from_pretrained('{model}', trust_remote_code=True)\"",
                              timeout=180)
            r = self._ssh_run(wrk_uid, "python -c \"import torch; print(torch.cuda.get_device_name(0))\"", timeout=30)
            log.info(f"[{wrk_uid}] GPU: {r.stdout.decode().strip()}")

            cli = (f"calibrate --model {model} --seq {seq} --batch {batch} "
                   f"--steps {steps} --warmup {warmup} --target-hours {target_hours} "
                   f"--empty-cache-every {empty_cache_every}")
            if compile_model:
                cli += " --compile"
            if chunked_ce:
                cli += f" --chunked-ce {int(chunked_ce)}"
            if fp8:
                cli += " --fp8"
            log.info(f"[{wrk_uid}] Running: sn125 {cli}")
            run_env = {"SN125_PROGRESS": "1",
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            if profile:
                run_env["SN125_PROFILE"] = "1"
            if flash:
                run_env["SN125_DIAG_FLASH"] = "1"
            if nondet:
                run_env["SN125_DIAG_NONDET"] = "1"
            stdout, rc = self._run_sn125_cmd(wrk_uid, cli, timeout, env=run_env)
            result = self._parse_calibration_result(stdout, rc)
            if result.get("failed") and wrk_uid:
                result["remote_diag"] = self._capture_failure_diag(wrk_uid)
                log.warning(f"[{wrk_uid}] calibrate FAILED — remote_diag:\n{result['remote_diag']}")
            return result
        except subprocess.TimeoutExpired:
            return {"failed": True, "error": "timeout_exceeded"}
        except Exception as e:
            log.error(f"[{wrk_uid or 'no-wrk'}] Calibration failed: {e}")
            return {"failed": True, "error": str(e)}
        finally:
            if wrk_uid:
                elapsed = time.time() - (rental_start or time.time())
                cost = (elapsed / 3600) * cost_per_hour
                self._track_cost(cost)
                log.info(f"[{wrk_uid}] Deleting. Elapsed: {elapsed:.0f}s, cost: ${cost:.2f}, daily: ${self.daily_spend:.2f}")
                self.client.delete_workload(wrk_uid)
                for rec in self._rentals:
                    if rec.wrk_uid == wrk_uid:
                        rec.deleted = True
                        rec.cost_accrued = cost
                self.write_status_file()
                result["box_seconds"] = elapsed
                result["est_cost"] = cost
                result["resource"] = resource_used
                if spend_ledger:
                    self._append_spend_ledger(spend_ledger, result)
            self._semaphore.release()

    def _capture_failure_diag(self, wrk_uid: str) -> str:
        """Best-effort remote post-mortem on a failed run, gathered BEFORE the box
        is deleted. dmesg surfaces an OOM-killer SIGKILL (the kind of abrupt death
        that leaves no Python traceback in the streamed log); nvidia-smi/df/free
        distinguish CUDA-OOM vs host-OOM vs disk-full. Never raises."""
        probes = [
            ("nvidia-smi", "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 || true"),
            ("dmesg_oom", "(dmesg 2>/dev/null | grep -iE 'oom|kill|out of memory' | tail -15) || echo 'dmesg unavailable'"),
            ("disk", "df -h /workspace / 2>&1 | tail -3 || true"),
            ("mem", "free -g 2>&1 | head -2 || true"),
        ]
        out = []
        for label, cmd in probes:
            try:
                r = self._ssh_run(wrk_uid, cmd, timeout=30)
                out.append(f"--- {label} ---\n{self._strip_banner(r.stdout).decode(errors='replace').strip()}")
            except Exception as e:
                out.append(f"--- {label} --- (probe failed: {e})")
        return "\n".join(out)

    def calibrate_matrix_remote(self, arms: list[dict], model: str = "Qwen/Qwen3-0.6B",
                                seq: int = 2048, batch: int = 8, target_hours: float = 20.0,
                                per_arm_timeout: int = 900, spend_ledger: str = "") -> dict:
        """§3 loop-efficiency PROFILING matrix: provision ONE b200-small and run a
        list of calibrate ``arms`` back-to-back on the SAME box, so the fwd/bwd/opt/
        clone breakdown and the determinism tax (det-MATH vs efficient) are measured
        without box-to-box hardware variance and with a single provisioning cost.

        Each arm is a dict: ``{label, empty_cache_every, profile, flash, nondet,
        steps, warmup}`` (all but ``label`` optional). Returns
        ``{arms: [{label, calibration, profile, rc}], box_seconds, est_cost}``.
        Box deleted in finally; total cost tracked + optionally ledgered as ONE row.
        """
        import os as _os
        if _os.path.exists("/root/.sn125/cost_kill_switch"):
            return {"failed": True, "error": "[CRITICAL] External cost watchdog kill-switch set."}
        if self.daily_spend >= DAILY_SPEND_LIMIT:
            return {"failed": True, "error": f"[CRITICAL] Daily spend ${self.daily_spend:.0f} >= ${DAILY_SPEND_LIMIT}."}

        wrk_uid = None
        resource_used = self.resource
        rental_start = None
        cost_per_hour = 20.0
        out = {"failed": True, "error": "not_started", "arms": []}
        self._semaphore.acquire()
        try:
            resource_used = self._check_inventory()
            cost_per_hour = self._cost_per_hour(resource_used)
            for attempt in range(2):
                wl_name = f"sn125-calmtx-{int(time.time())}"[:63].lower()
                try:
                    resp = self.client.create_workload(wl_name, resource_used, self._ssh_key_uid)
                    wrk_uid = resp.get("uid", "")
                    if not wrk_uid:
                        raise RuntimeError(f"No UID: {resp}")
                    log.info(f"[{wrk_uid}] Created '{wl_name}' on {resource_used}")
                    self.client.deploy_workload(wrk_uid)
                    rental_start = time.time()
                    self._rentals.append(RentalRecord(wrk_uid, wl_name, resource_used, rental_start, cost_per_hour))
                    if self.client.wait_running(wrk_uid):
                        break
                    log.warning(f"[{wrk_uid}] Never RUNNING (attempt {attempt+1})")
                    self.client.delete_workload(wrk_uid)
                    wrk_uid = None
                except Exception as e:
                    log.error(f"Create/deploy failed (attempt {attempt+1}): {e}")
                    if wrk_uid:
                        self.client.delete_workload(wrk_uid)
                        wrk_uid = None
            if not wrk_uid:
                out["error"] = "rental_provisioning_failed"
                return out

            self._wait_ssh(wrk_uid)
            self._upload_project(wrk_uid)
            self._provision(wrk_uid)
            if not (model.startswith("lean-") or model.startswith("forge-")):
                self._ssh_run(wrk_uid, f"cd /workspace && PYTHONPATH=/workspace python -c "
                              f"\"from transformers import AutoConfig; AutoConfig.from_pretrained('{model}', trust_remote_code=True)\"",
                              timeout=180)
            r = self._ssh_run(wrk_uid, "python -c \"import torch; print(torch.cuda.get_device_name(0))\"", timeout=30)
            gpu_name = self._strip_banner(r.stdout).decode(errors="replace").strip()
            log.info(f"[{wrk_uid}] GPU: {gpu_name}")

            out["gpu_name"] = gpu_name
            out["failed"] = False
            out["error"] = ""
            for arm in arms:
                label = arm.get("label", "arm")
                steps = int(arm.get("steps", 80))
                warmup = int(arm.get("warmup", 20))
                ece = int(arm.get("empty_cache_every", 0))
                cli = (f"calibrate --model {model} --seq {seq} --batch {batch} "
                       f"--steps {steps} --warmup {warmup} --target-hours {target_hours} "
                       f"--empty-cache-every {ece}")
                run_env = {"SN125_PROGRESS": "1", "SN125_PROGRESS_INTERVAL": "20",
                           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                if arm.get("profile"):
                    run_env["SN125_PROFILE"] = "1"
                if arm.get("flash"):
                    run_env["SN125_DIAG_FLASH"] = "1"
                if arm.get("nondet"):
                    run_env["SN125_DIAG_NONDET"] = "1"
                log.info(f"[{wrk_uid}] arm '{label}': sn125 {cli}  env={sorted(run_env)}")
                stdout, rc = self._run_sn125_cmd(wrk_uid, cli, per_arm_timeout, env=run_env)
                cal = self._parse_calibration_result(stdout, rc)
                prof = self._parse_profile_breakdown(stdout)
                out["arms"].append({"label": label, "rc": rc, "calibration": cal,
                                    "profile": prof, "config": {"steps": steps, "warmup": warmup,
                                    "empty_cache_every": ece, "flash": bool(arm.get("flash")),
                                    "nondet": bool(arm.get("nondet")), "profile": bool(arm.get("profile"))}})
                log.info(f"[{wrk_uid}] arm '{label}' rc={rc} "
                         f"iter_s={cal.get('median_step_time_s')} prof={prof}")
            return out
        except subprocess.TimeoutExpired:
            out["error"] = "timeout_exceeded"
            return out
        except Exception as e:
            log.error(f"[{wrk_uid or 'no-wrk'}] Matrix failed: {e}")
            out["error"] = str(e)
            return out
        finally:
            if wrk_uid:
                elapsed = time.time() - (rental_start or time.time())
                cost = (elapsed / 3600) * cost_per_hour
                self._track_cost(cost)
                log.info(f"[{wrk_uid}] Deleting. Elapsed: {elapsed:.0f}s, cost: ${cost:.2f}, daily: ${self.daily_spend:.2f}")
                self.client.delete_workload(wrk_uid)
                for rec in self._rentals:
                    if rec.wrk_uid == wrk_uid:
                        rec.deleted = True
                        rec.cost_accrued = cost
                self.write_status_file()
                out["box_seconds"] = elapsed
                out["est_cost"] = cost
                out["resource"] = resource_used
                if spend_ledger:
                    self._append_spend_ledger(spend_ledger, {
                        "model_config": model, "resource": resource_used,
                        "box_seconds": elapsed, "est_cost": cost,
                        "computed_N": None, "median_step_time_s": None,
                        "chinchilla_ratio": None, "mfu": None})
            self._semaphore.release()

    @staticmethod
    def _parse_profile_breakdown(stdout: str) -> dict:
        """Extract the single 'PROFILE_BREAKDOWN {json}' line (the per-phase
        forward/backward/grad_clone/empty_cache/opt_step split). Empty dict if the
        arm was not profiled (SN125_PROFILE unset)."""
        marker = "PROFILE_BREAKDOWN "
        for line in reversed(stdout.splitlines()):
            idx = line.find(marker)
            if idx != -1:
                try:
                    return json.loads(line[idx + len(marker):])
                except json.JSONDecodeError:
                    break
        return {}

    @staticmethod
    def _parse_calibration_result(stdout: str, rc: int) -> dict:
        """Extract the single 'CALIBRATION_RESULT {json}' line from probe stdout."""
        marker = "CALIBRATION_RESULT "
        for line in reversed(stdout.splitlines()):
            idx = line.find(marker)
            if idx != -1:
                try:
                    d = json.loads(line[idx + len(marker):])
                    d["failed"] = False
                    return d
                except json.JSONDecodeError:
                    break
        return {"failed": True, "error": f"no CALIBRATION_RESULT (rc={rc}): {stdout[-400:]}"}

    @staticmethod
    def _append_spend_ledger(path: str, entry: dict) -> None:
        """Append one calibration spend record to the JSON-array ledger, carrying
        a running cumulative total (matches the existing targon_spend.json shape)."""
        try:
            ledger = []
            if os.path.exists(path):
                with open(path) as f:
                    ledger = json.load(f)
            prior = ledger[-1].get("cumulative", 0.0) if ledger else 0.0
            rec = {"kind": "calibration_probe",
                   "model": entry.get("model_config", entry.get("model", "?")),
                   "resource": entry.get("resource", "b200-small"),
                   "box_seconds": entry.get("box_seconds", 0.0),
                   "est_cost": entry.get("est_cost", 0.0),
                   "cumulative": prior + entry.get("est_cost", 0.0),
                   "computed_N": entry.get("computed_N"),
                   "median_step_time_s": entry.get("median_step_time_s"),
                   "chinchilla_ratio": entry.get("chinchilla_ratio"),
                   "mfu": entry.get("mfu")}
            ledger.append(rec)
            with open(path, "w") as f:
                json.dump(ledger, f, indent=2)
        except Exception as e:
            log.error(f"spend ledger append failed: {e}")

    def get_status(self) -> dict:
        """Return current orchestrator status for dashboard."""
        active = [r for r in self._rentals if not r.deleted]
        with self._capacity_lock:
            cw = dict(self._capacity_wait) if self._capacity_wait is not None else None
        capacity_delay = None
        if cw is not None:
            waited = max(cw.get("waited_s", 0.0), time.time() - cw.get("since", time.time()))
            capacity_delay = {
                "delayed": True,
                "sku": cw.get("sku", self.resource),
                "since": cw.get("since"),
                "waited_s": waited,
                "waited_min": round(waited / 60.0, 1),
                "round_id": cw.get("round_id", ""),
                "alert": waited >= 21600.0,
                "reason": cw.get("reason", "awaiting_b200_capacity"),
            }
        return {
            "active_rentals": len(active),
            "max_concurrent": MAX_CONCURRENT,
            "daily_spend": self.daily_spend,
            "committed_spend": self.committed_spend,
            "daily_limit": DAILY_SPEND_LIMIT,
            "rentals": [
                {
                    "uid": r.wrk_uid, "name": r.name, "resource": r.resource,
                    "provider": r.provider,
                    "elapsed_s": time.time() - r.created_at,
                    "cost_so_far": ((time.time() - r.created_at) / 3600) * r.cost_per_hour,
                }
                for r in active
            ],
            "providers": [name for name, _c in getattr(self.client, "providers", [])]
                         or [str(getattr(self.client, "provider_name", "") or "targon")],
            "total_rentals_today": sum(1 for r in self._rentals if r.deleted),
            "updated": time.time(),
            "resource_sku": self.resource,
            "timeout_h": self.timeout / 3600,
            "cost_per_sub": (self.timeout / 3600) * self._cost_per_hour(self.resource, default=5.0),
            "capacity_delay": capacity_delay,
        }

    def write_status_file(self):
        """Write status to JSON file for dashboard consumption."""
        try:
            p = Path(__file__).resolve().parent / "cloud_status.json"
            p.write_text(json.dumps(self.get_status()))
        except Exception:
            pass


_DEFAULT_SETUP_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# iptables is required for the C1/C2 egress lockdown around miner execution
# (TargonOrchestrator._lockdown_network); missing iptables fails lockdown closed.
apt-get update -qq && apt-get install -y -qq rsync openssh-sftp-server build-essential iptables
# Retry pip installs: ~1/3 of identical boxes hit a transient empty pip index
# ("from versions: none") on a single-shot install. Retry with --retries before
# giving up so we don't burn a whole box on a momentary network/index blip.
pip_retry() {
    for i in 1 2 3; do
        pip install --no-cache-dir --retries 5 --timeout 60 "$@" && return 0
        echo "pip install attempt $i failed for: $* -- retrying in 15s"; sleep 15
    done
    echo "ERROR: pip install failed after 3 attempts for: $*"; return 1
}
# B200 = Blackwell sm_100: default cu124 wheels only ship sm_50..sm_90 kernels
# → "no kernel image available" at first CUDA op. cu128 wheels + torch>=2.7 carry
# sm_100 (calib probe, 2026-06-13). Keep in sync with scripts/setup_b200.sh.
# Exact pin: every scoring fleet (RunPod / Lambda / Targon) must run the same
# torch build as the genesis baseline (2.8.0+cu128). A ">=" range let the
# RunPod image's 2.8.0.dev nightly satisfy the constraint (seen 2026-09-04).
pip_retry --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0' 'torchvision==0.23.0'
python -c "import torch" || { echo "ERROR: torch not importable after install"; exit 1; }
pip_retry \
    'transformers>=4.57,<4.60' \
    'datasets>=4.8,<5' \
    'accelerate>=1.0,<2' \
    'safetensors>=0.4,<1' \
    'tiktoken>=0.8,<1' \
    'triton>=3.2,<4' \
    'bittensor>=10.2,<11' \
    'boto3>=1.34,<2'
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "PyTorch $(python -c 'import torch;print(torch.__version__)'), GPUs: $NGPU"
[ "$NGPU" -ge 1 ] || { echo "ERROR: No GPUs found"; exit 1; }
cuda_smoke() {
python - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable after torch install")

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch = torch.cuda.get_arch_list()
need = f"sm_{cap[0]}{cap[1]}"
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": name,
    "capability": cap,
    "arch_list": arch,
}))
if name and ("B200" in name or "B100" in name or cap[0] >= 10):
    if need not in arch:
        raise SystemExit(
            f"ERROR: installed torch wheel lacks {need}; B200 requires cu128/sm_100 support")

x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert y.isfinite().all().item()
print("CUDA kernel smoke OK")
PY
}

for i in 1 2 3 4 5 6 7 8; do
    if cuda_smoke; then
        echo "CUDA smoke passed on attempt $i"
        break
    fi
    if [ "$i" -eq 8 ]; then
        echo "ERROR: CUDA smoke failed after $i attempts"
        exit 1
    fi
    echo "CUDA smoke attempt $i failed; waiting for GPU runtime readiness"
    sleep 20
done
echo "Setup complete."
"""
