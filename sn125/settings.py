"""Runtime settings sourced from the environment — one home for the ``SN125_*``
knobs and the infra credentials (R2, Hugging Face).

Companion to :mod:`sn125.config` (which holds public launch-economics constants).
This module holds the *runtime* switches. Two properties are load-bearing and are
why everything here is a **live accessor** (a function that reads ``os.environ``
on every call) rather than a module-level constant captured at import:

1. Some flags are toggled at runtime — the test suite and ``python -m sn125``
   set ``SN125_PROFILE`` / ``SN125_PROGRESS`` / ``SN125_DIAG_*`` after import.
2. Most eval flags are the *transport* to the disposable Targon worker: the
   orchestrator injects them into the remote process env (see ``_ENV_ALLOWLIST``
   in ``training.py`` and ``eval_env`` in ``cloud.py``), and the worker reads its
   own ``os.environ``. The flag *name* is therefore a cross-process contract and
   must stay a stable string — hence the ``*_ENV`` name constants below.

Secrets (``TARGON_API_KEY``, ``R2_*``, ``HF_*``, ``ANTHROPIC_API_KEY``) stay in
the environment by design; this module only centralizes *reading* them.

Boolean parsing has two intentional flavors, preserved from the call sites:

* :func:`env_truthy` — "on unless explicitly off" (``""``/``0``/``false``/``no``/
  ``off`` are false). Used by the fail-closed prod switches so that an empty or
  garbage value never silently disables a guard.
* :func:`env_present` — Python-truthy on any non-empty value (even ``"0"``).
  Used by the dev/diagnostic flags where mere presence means "on".
"""
from __future__ import annotations

import os

# ── env-var NAME registry (the cross-process contract; keep stable) ──────────
# Fail-closed production guards (env_truthy semantics).
REQUIRE_SANDBOX_ENV = "SN125_REQUIRE_SANDBOX"
NET_LOCKDOWN_ENV = "SN125_NET_LOCKDOWN"
NET_LOCKDOWN_PROBE_ENV = "SN125_NET_LOCKDOWN_PROBE"
NET_LOCKDOWN_ALLOW_CIDRS_ENV = "SN125_NET_LOCKDOWN_ALLOW_CIDRS"
USE_OPTPROC_ENV = "SN125_USE_OPTPROC"
# Refiner task (Challenge 2) in live rounds — see docs/REFINER_TASK.md §9.
REFINER_TASK_ENV = "SN125_REFINER_TASK"                      # accept + evaluate refiner submissions
REFINER_EMISSION_SHARE_ENV = "SN125_REFINER_EMISSION_SHARE"  # refiner pool share of emission (default 0.5; 0 = publish-only)
REFINER_BASELINE_JSON_ENV = "SN125_REFINER_BASELINE_JSON"    # pinned identity-baseline result to reuse
REFINER_SUITE_ENV = "SN125_REFINER_SUITE"                    # suite JSON shipped to the box (default: <data>/bench/…)
REFINER_DECONTAM_ENV = "SN125_REFINER_DECONTAM"              # exclusion list shipped to the box (default: next to shards)
# Calibration knobs for operator drivers (scripts/refiner_calibration.py); a live
# validator never sets them. Both are recorded in the result by refiner-eval.
REFINER_EVAL_SEED_ENV = "SN125_REFINER_EVAL_SEED"            # --seed for the box's refiner-eval (default 42)
REFINER_BASELINE_ONLY_ENV = "SN125_REFINER_BASELINE_ONLY"    # 1 = train/score only the identity baseline
# Optional held-out non-regression gate (nats; unset = off): a refiner whose student
# ends worse than the identity baseline on held-out FineWeb-Edu by more than this
# scores 0 for the round. The suite delta is still published (components.suite_delta).
REFINER_HELDOUT_MAX_REGRESSION_ENV = "SN125_REFINER_HELDOUT_MAX_REGRESSION"
# OS sandbox flavour for miner code: "namespaces" (VM hosts — unshare +
# bind mounts + priv drop; the historical default) or "seccomp" (container
# hosts such as RunPod pods — priv drop + syscall filter, see
# sn125/seccomp_sandbox.py). Set per provider by the cloud orchestrator.
SANDBOX_MODE_ENV = "SN125_SANDBOX_MODE"
SANDBOX_MODES = ("namespaces", "seccomp")
OPTPROC_STEP_TIMEOUT_ENV = "SN125_OPTPROC_STEP_TIMEOUT"
OPTPROC_NO_SNAPSHOT_ENV = "SN125_OPTPROC_NO_SNAPSHOT"
OPTPROC_UNSYNCED_ENV = "SN125_OPTPROC_UNSYNCED"

# Clean-score (fresh-process rescoring) knobs.
CLEAN_SCORE_DEVICE_ENV = "SN125_CLEAN_SCORE_DEVICE"
CLEAN_SCORE_TIMEOUT_ENV = "SN125_CLEAN_SCORE_TIMEOUT"
CLEAN_SCORE_ATOL_ENV = "SN125_CLEAN_SCORE_ATOL"

# Harness throughput / determinism / diagnostics (env_present semantics unless noted).
LEAN_COMPILE_ENV = "SN125_LEAN_COMPILE"
PROGRESS_ENV = "SN125_PROGRESS"
PROFILE_ENV = "SN125_PROFILE"

# Validator-loop cadence knobs (numeric).
AUDIT_HEARTBEAT_S_ENV = "SN125_AUDIT_HEARTBEAT_S"
WEIGHT_REFRESH_S_ENV = "SN125_WEIGHT_REFRESH_S"
TARGON_HEARTBEAT_S_ENV = "SN125_TARGON_HEARTBEAT_S"
TARGON_REMOTE_HEARTBEAT_EVERY_ENV = "SN125_TARGON_REMOTE_HEARTBEAT_EVERY"

# Cloud provider failover chain + the per-box compute-standardization gates.
# The scoring path rents from the FIRST provider in the chain with B200
# inventory and fails over down the chain; the round PAUSES only when every
# provider is dry. Lium (a marketplace of untrusted third-party hosts) is
# refused for scoring — see cloud.ensure_scoring_providers().
CLOUD_PROVIDERS_ENV = "SN125_CLOUD_PROVIDERS"
CLOUD_PROVIDER_ENV = "SN125_CLOUD_PROVIDER"   # legacy single-provider knob
BOX_PROBE_ENV = "SN125_BOX_PROBE"
BOX_PROBE_MIN_STS_ENV = "SN125_BOX_PROBE_MIN_STS"
# Two-sided throughput band (multi-provider fairness): a box outside
# ref*(1 ± band/100) is rejected as a provisioning flake, both too SLOW (the
# miner's schedule is starved) and too FAST (a lucky box would inflate the
# score and poison the frontier ratchet). Pin the reference with the box-probe
# procedure in boxprobe.py; unset = band gate off (legacy MIN_STS floor only).
BOX_PROBE_REF_STS_ENV = "SN125_BOX_PROBE_REF_STS"
BOX_PROBE_BAND_PCT_ENV = "SN125_BOX_PROBE_BAND_PCT"
# Post-eval drift probe: re-run the box probe after a successful eval and
# quarantine the run when the box's own speed moved more than the band vs the
# pre-eval probe (mid-run throttling / noisy neighbor / probe sandbagging).
BOX_PROBE_DRIFT_ENV = "SN125_BOX_PROBE_DRIFT"

# Infra credentials (secrets; read-only here).
R2_ENDPOINT_ENV = "R2_ENDPOINT"
R2_ACCESS_KEY_ENV = "R2_ACCESS_KEY"
R2_SECRET_KEY_ENV = "R2_SECRET_KEY"
R2_BUCKET_ENV = "R2_BUCKET"
HF_HOME_ENV = "HF_HOME"

_OFF_VALUES = frozenset({"", "0", "false", "no", "off"})
_DEFAULT_HF_HOME = "~/.cache/huggingface"

DEFAULT_CLOUD_PROVIDER_CHAIN = ("runpod",)
DEFAULT_BOX_PROBE_BAND_PCT = 5.0


# ── LAUNCH PROFILE ────────────────────────────────────────────────────────────
# The canonical production configuration the validator injects into every worker
# at launch. Declared here as data (not scattered string literals in cloud.py) so
# "what are we launching with" is auditable in one place.
# These are the CROSS-BOX EVAL knobs only. The other halves of the launch config
# live where they are computed and must not be duplicated here:
#   * schedule horizon / token budget  → fineweb.PROD_CONFIRM_STEPS (188k)
#   * burn schedule / frontier thresholds / public seeds → config.py
#   * egress lockdown default          → net_lockdown_enabled() (ON unless off)
LAUNCH_EVAL_ENV: dict[str, str] = {
    USE_OPTPROC_ENV: "1",              # route miner steps through CUDA-IPC isolation
    REQUIRE_SANDBOX_ENV: "1",          # fail-closed OS sandbox (namespaces or seccomp + priv drop)
    "SN125_LEAN_COMPILE": "1",         # inductor-compiled forward (~2x B200 throughput)
    "SN125_LEAN_LOSS_CKPT": "0",       # CE-tile recompute off (measured 2026-07-03)
    OPTPROC_STEP_TIMEOUT_ENV: "30",    # per-step wall cap seconds (kill runaway miner step)
    PROGRESS_ENV: "1",                 # emit worker progress heartbeats
    "SN125_FORWARD_WORKER_PROGRESS": "1",
}

# Launch knobs an operator MAY override from host env for a supervised bring-up
# (e.g. SN125_REQUIRE_SANDBOX=0 on a box without unshare, or SN125_LEAN_COMPILE=0
# to fall back to the eager forward). Everything else in LAUNCH_EVAL_ENV is fixed.
LAUNCH_OPERATOR_OVERRIDABLE = frozenset({
    "SN125_LEAN_COMPILE", "SN125_LEAN_LOSS_CKPT", "SN125_LEAN_LOSS_CHUNKS",
    REQUIRE_SANDBOX_ENV,
})


def launch_eval_env(host_env: dict | None = None) -> dict[str, str]:
    """Return a fresh dict of the launch worker-env parameters.

    Fixed launch values from :data:`LAUNCH_EVAL_ENV`, with the
    :data:`LAUNCH_OPERATOR_OVERRIDABLE` knobs replaced by any non-empty host-env
    value. The orchestrator merges this over the shard-staging env before
    spawning a production eval."""
    src = os.environ if host_env is None else host_env
    env = dict(LAUNCH_EVAL_ENV)
    for key in LAUNCH_OPERATOR_OVERRIDABLE:
        val = (src.get(key) or "").strip()
        if val:
            env[key] = val
    return env


# ── parse helpers ────────────────────────────────────────────────────────────

def env_truthy(name: str, default: bool = False, env: dict | None = None) -> bool:
    """"On unless explicitly off": true for any value not in ``_OFF_VALUES``.

    ``default`` is returned only when the variable is unset. This is the
    fail-closed flavor — a garbage value reads as ON, never silently OFF.
    """
    src = os.environ if env is None else env
    if name not in src:
        return default
    return src.get(name, "").strip().lower() not in _OFF_VALUES


def env_present(name: str, env: dict | None = None) -> bool:
    """Python-truthy on any non-empty value (matches ``bool(os.environ.get(x))``).

    Note: ``"0"`` reads as present/ON here — this is the dev/diagnostic flavor
    where operators toggle a flag by merely setting it.
    """
    src = os.environ if env is None else env
    return bool(src.get(name))


def env_str(name: str, default: str = "", env: dict | None = None) -> str:
    src = os.environ if env is None else env
    v = src.get(name, "")
    v = v.strip() if v is not None else ""
    return v or default


def env_int(name: str, default: int, *, minimum: int | None = None,
            env: dict | None = None) -> int:
    src = os.environ if env is None else env
    raw = src.get(name, "")
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, val) if minimum is not None else val


def env_float(name: str, default: float, *, minimum: float | None = None,
              env: dict | None = None) -> float:
    src = os.environ if env is None else env
    raw = src.get(name, "")
    try:
        val = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, val) if minimum is not None else val


# ── fail-closed production guards ─────────────────────────────────────────────

def sandbox_required(env: dict | None = None) -> bool:
    """Whether the OS sandbox (unshare namespaces + priv drop) is mandatory.

    When true, the harness refuses to run miner code outside the sandbox and
    rejects results that don't attest ``sandboxed=True``/``uid=65534``. The prod
    eval env (``cloud.py``) sets this to ``1``; unset defaults to OFF so local
    dev/CI still runs without namespaces."""
    return env_truthy(REQUIRE_SANDBOX_ENV, default=False, env=env)


def net_lockdown_enabled(env: dict | None = None) -> bool:
    """Egress lockdown, ON by default for prod. Disable only for a supervised
    bring-up with ``SN125_NET_LOCKDOWN=0``."""
    return env_truthy(NET_LOCKDOWN_ENV, default=True, env=env)


def refiner_task_enabled(env: dict | None = None) -> bool:
    """Whether the validator accepts and evaluates Refiner-task (Challenge 2)
    submissions in its rounds. ON by default, paying the refiner pool's share
    of emission (``refiner_emission_share``); ``SN125_REFINER_TASK=0`` turns it off,
    and with it off a refiner envelope is rejected at the gate and nothing else
    in a round changes. With the variable unset the inputs (suite,
    decontamination list, pinned baseline) must resolve from the package or
    the task silently stays off; set to ``1`` the validator refuses to start on
    a half-configured task instead (roundsm.refiner_round)."""
    return env_truthy(REFINER_TASK_ENV, default=True, env=env)


def refiner_task_explicit(env: dict | None = None) -> bool:
    """True when the operator set ``SN125_REFINER_TASK`` at all (either way)."""
    src = os.environ if env is None else env
    return REFINER_TASK_ENV in src


def refiner_emission_share(env: dict | None = None) -> float:
    """Share of miner emission routed to the Refiner task's pool (default
    ``config.REFINER_EMISSION_SHARE``, 0.5): the refiner frontier pays that
    share of the weight vector by its own progress-scaled rule, the optimizer
    frontier the rest. 0 makes the task publish-only (scored and published,
    nothing paid); anything outside [0, 1] makes the validator refuse to start."""
    from .config import REFINER_EMISSION_SHARE
    return env_float(REFINER_EMISSION_SHARE_ENV, REFINER_EMISSION_SHARE, env=env)


def use_optproc(env: dict | None = None) -> bool:
    """Route miner optimizer steps through the CUDA-IPC isolated process."""
    return env_truthy(USE_OPTPROC_ENV, default=False, env=env)


def sandbox_mode(env: dict | None = None) -> str:
    """Sandbox flavour (``namespaces`` default, ``seccomp`` for container
    hosts). An unknown value fails closed to ``namespaces`` on a box that has
    them and to a refused eval otherwise — never to "no sandbox"."""
    raw = env_str(SANDBOX_MODE_ENV, "namespaces", env=env).strip().lower()
    return raw if raw in SANDBOX_MODES else "namespaces"


# ── cloud provider chain + box-probe band ─────────────────────────────────────

def cloud_provider_chain(env: dict | None = None) -> list[str]:
    """Ordered provider failover chain for B200 rentals.

    Precedence: SN125_CLOUD_PROVIDERS (comma-separated, in failover order) →
    legacy SN125_CLOUD_PROVIDER (single provider, kept as an explicit operator
    override) → ``DEFAULT_CLOUD_PROVIDER_CHAIN`` (``runpod``). Names are lower-cased and
    de-duplicated preserving order; membership/lium checks live in
    ``cloud.ensure_scoring_providers`` (this module stays import-light)."""
    raw = env_str(CLOUD_PROVIDERS_ENV, env=env)
    if not raw:
        legacy = env_str(CLOUD_PROVIDER_ENV, env=env)
        if legacy:
            return [legacy.strip().lower()]
        return list(DEFAULT_CLOUD_PROVIDER_CHAIN)
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if tok and tok not in out:
            out.append(tok)
    return out or list(DEFAULT_CLOUD_PROVIDER_CHAIN)


def box_probe_band_pct(env: dict | None = None) -> float:
    """Half-width of the throughput band (and the drift tolerance), in percent."""
    return env_float(BOX_PROBE_BAND_PCT_ENV, DEFAULT_BOX_PROBE_BAND_PCT,
                     minimum=0.0, env=env)


def box_probe_ref_sts(env: dict | None = None) -> float:
    """Operator-pinned reference probe throughput (steps/s). 0 = band gate off.

    MUST be pinned with the standardized box probe (boxprobe.py pinning
    procedure), never with in-run harness throughput — the probe reads
    systematically faster than a real eval on the same box."""
    return env_float(BOX_PROBE_REF_STS_ENV, 0.0, minimum=0.0, env=env)


# ── infra: R2 + Hugging Face ──────────────────────────────────────────────────

def hf_home(env: dict | None = None) -> str:
    """Resolved Hugging Face home dir (``HF_HOME`` or the user default)."""
    src = os.environ if env is None else env
    return src.get(HF_HOME_ENV) or os.path.expanduser(_DEFAULT_HF_HOME)


def r2_config(env: dict | None = None) -> dict[str, str]:
    """Read the four R2 credentials into a dict (values stripped)."""
    src = os.environ if env is None else env
    return {
        "endpoint": (src.get(R2_ENDPOINT_ENV) or "").strip(),
        "access_key": (src.get(R2_ACCESS_KEY_ENV) or "").strip(),
        "secret_key": (src.get(R2_SECRET_KEY_ENV) or "").strip(),
        "bucket": (src.get(R2_BUCKET_ENV) or "").strip(),
    }


def r2_configured(env: dict | None = None) -> bool:
    """True only when all four R2 credentials are present."""
    cfg = r2_config(env)
    return all([cfg["endpoint"], cfg["access_key"], cfg["secret_key"], cfg["bucket"]])


def make_r2_client(env: dict | None = None):
    """Construct a boto3 S3 client for R2 from the environment credentials.

    Returns ``None`` if R2 is not fully configured. Raises only if boto3 import
    or client construction fails (callers decide whether that is fatal)."""
    cfg = r2_config(env)
    if not all([cfg["endpoint"], cfg["access_key"], cfg["secret_key"], cfg["bucket"]]):
        return None
    import boto3
    from botocore.config import Config
    # SigV4 + region "auto" explicitly: with boto3's defaults a PRESIGNED URL for
    # a custom endpoint is signed with SigV2, which R2 rejects with 401 — the
    # box-side checkpoint uploads (cloud._upload_checkpoints) and any presigned
    # download depend on this. Plain API calls were already SigV4.
    return boto3.client(
        "s3", endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        region_name="auto", config=Config(signature_version="s3v4"))
