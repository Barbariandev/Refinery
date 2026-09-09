"""
SN125 Optimizer Discovery Subnet — Neuron (Miner + Validator)
Single file: protocol, miner, validator, round management, attestation.

The production evaluation itself runs remotely (roundsm/live.py drives the FSM
round loop; cloud.py rents the B200 worker; prod_eval.py scores on the box).
This module owns the chain-facing pieces: synapses, the miner axon, baselines,
the frontier reward path, round persistence, and attestation.
"""
import hashlib, json, logging, math, os, time, traceback
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, Optional

import bittensor as bt

SN125_VERSION = "0.2.3"

from . import settings
from .references import ADAMW_SOURCE, extract_hparams
from .sandbox import SandboxViolation, validate_source
from .training import ScoreRecord, TaskSpec, TrainingCurve
from .config import (
    AUTHORIZED_VALIDATOR_HOTKEY,
    BURN_UID,
    FRONTIER_THRESHOLD_DECAY_HALF_LIFE_DAYS,
    FRONTIER_THRESHOLD_FRACTION,
    FRONTIER_THRESHOLD_WINDOW,
    FRONTIER_CONFIRMATION_MAX_ATTEMPTS,
    FRONTIER_CONFIRMATION_RUNS,
    FRONTIER_IMPROVEMENT_UNIT,
    LAUNCH_BURN_FRACTION_FLOOR,
    LAUNCH_TIMESTAMP,
    MIN_FRONTIER_IMPROVEMENT,
    NETUID,
    PROGRESS_HALF_LIFE_DAYS,
    full_pay_reference_nats,
    launch_burn_fraction_for_day,
    launch_day,
    progress_payable_share,
    public_seed_manifest,
    validator_hotkey_configured,
)

log = logging.getLogger("sn125")

WEIGHT_REFRESH_SECONDS = 360 * 12

_SIGNING_EXCLUDE = frozenset(("sources", "signature", "bundle_hash"))


def _signing_payload(bundle: dict) -> bytes:
    """JSON-encode bundle excluding non-signed fields, for signature verification."""
    core = {k: v for k, v in bundle.items() if k not in _SIGNING_EXCLUDE}
    return json.dumps(core, sort_keys=True, default=str).encode()


_CHECKPOINT_KEYS = ("pct", "step", "eval_loss", "wall_t", "sha256", "uri")


def _normalize_checkpoints(cps) -> list:
    """Coerce emitted checkpoint records to a clean, JSON/sign-stable list of dicts.
    Tolerant of partial records; preserves unknown keys; never raises (audit data
    must not crash a scoring round). Sorted by pct then step for deterministic output."""
    out = []
    if not cps:
        return out
    try:
        items = list(cps)
    except TypeError:
        return out
    for c in items:
        if not isinstance(c, dict):
            continue
        rec = {}
        for k, v in c.items():
            if k == "path":
                continue
            if k in ("pct", "eval_loss", "wall_t"):
                try: rec[k] = float(v)
                except (TypeError, ValueError): rec[k] = v
            elif k == "step":
                try: rec[k] = int(v)
                except (TypeError, ValueError): rec[k] = v
            else:
                rec[k] = v
        out.append(rec)
    out.sort(key=lambda r: (r.get("pct", 0.0) if isinstance(r.get("pct"), (int, float)) else 0.0,
                            r.get("step", 0) if isinstance(r.get("step"), int) else 0))
    return out


class GetSubmission(bt.Synapse):
    """Validator requests a miner's optimizer source for a round."""
    required_hash_fields: ClassVar[tuple[str, ...]] = ("round_id",)
    round_id: str = ""
    source_code: str = ""
    code_hash: str = ""
    name: str = ""
    description: str = ""

class CommitHash(bt.Synapse):
    """Miner pre-commits optimizer hash before seeing the reveal seed."""
    required_hash_fields: ClassVar[tuple[str, ...]] = ("round_id",)
    round_id: str = ""
    code_hash: str = ""

class Ping(bt.Synapse):
    """Health check."""
    status: str = ""


def _require_authorized_validator(synapse: bt.Synapse) -> None:
    """Fail closed unless this request is signed by the configured validator.

    Bittensor 10.2.0's default Axon verifier only verifies the signature when
    ``synapse.dendrite.signature`` is truthy. Commit/reveal carries optimizer
    source, so SN125 requires the signature field to be present before delegating
    to the SDK verifier.
    """
    if not validator_hotkey_configured():
        raise PermissionError("SN125 authorized validator hotkey is not configured")
    dendrite = getattr(synapse, "dendrite", None)
    if dendrite is None:
        raise PermissionError("missing dendrite terminal")
    hotkey = getattr(dendrite, "hotkey", "") or ""
    if hotkey != AUTHORIZED_VALIDATOR_HOTKEY:
        raise PermissionError("unauthorized validator hotkey")
    if not getattr(dendrite, "signature", None):
        raise PermissionError("missing validator dendrite signature")
    if getattr(dendrite, "nonce", None) is None:
        raise PermissionError("missing validator dendrite nonce")
    if not getattr(dendrite, "uuid", None):
        raise PermissionError("missing validator dendrite uuid")


def _set_weights_from_map(subtensor, wallet, netuid: int, meta: bt.Metagraph,
                          weights: dict[str, float], burn_uid: int = BURN_UID):
    """Shared: resolve hotkeys→UIDs, set weights on chain. The `Validator.BURN_KEY`
    sentinel routes its (unearned) weight to `burn_uid` (the subnet-owner UID 0 by
    convention), destroying that emission rather than paying a non-improver."""
    n = int(meta.n) if hasattr(meta.n, 'item') else len(meta.neurons)
    hk_to_uid = {meta.neurons[uid].axon_info.hotkey: uid for uid in range(n)}
    uid_vals: dict[int, float] = {}
    for hk, w in weights.items():
        if w <= 0:
            continue
        if hk == "__BURN__":
            uid = burn_uid
        elif hk in hk_to_uid:
            uid = hk_to_uid[hk]
        else:
            log.warning("  hotkey %s not in metagraph; routing its %.4f weight to burn UID %d",
                        hk[:16], w, burn_uid)
            uid = burn_uid
        uid_vals[uid] = uid_vals.get(uid, 0.0) + float(w)
    uids, vals = list(uid_vals), [uid_vals[u] for u in uid_vals]
    if not uids:
        log.warning("  No UIDs to set weights for")
        return
    try:
        subtensor.set_weights(wallet=wallet, netuid=netuid,
                              uids=uids, weights=vals, wait_for_inclusion=True)
        log.info(f"  Weights set on chain for {len(uids)} miners")
    except Exception as e:
        log.error(f"  Failed to set weights: {e}")


class Miner:
    """Serves an optimizer submission via Axon."""

    def __init__(self, wallet: bt.Wallet, port: int = 8091,
                 optimizer_path: Optional[str] = None):
        self.wallet = wallet
        self.source_code = self._load_source(optimizer_path)
        self.code_hash = hashlib.sha256(self.source_code.encode()).hexdigest()
        self.name = "miner-submission"
        self.description = ""
        violations = validate_source(self.source_code)
        if violations:
            raise SandboxViolation(f"Optimizer source invalid: {violations}")
        self.axon = bt.Axon(wallet=wallet, port=port)
        self.axon.attach(forward_fn=self.handle_get_submission,
                         verify_fn=self.verify_get_submission)
        self.axon.attach(forward_fn=self.handle_commit_hash,
                         verify_fn=self.verify_commit_hash)
        self.axon.attach(forward_fn=self.handle_ping)

    def _load_source(self, path: Optional[str]) -> str:
        if path and os.path.exists(path):
            return Path(path).read_text()
        if path:
            log.warning(f"Optimizer file {path} not found, using default AdamW")
        return ADAMW_SOURCE

    async def verify_get_submission(self, synapse: GetSubmission) -> None:
        _require_authorized_validator(synapse)
        await self.axon.default_verify(synapse)

    async def verify_commit_hash(self, synapse: CommitHash) -> None:
        _require_authorized_validator(synapse)
        await self.axon.default_verify(synapse)

    def handle_get_submission(self, synapse: GetSubmission) -> GetSubmission:
        _require_authorized_validator(synapse)
        synapse.source_code = self.source_code
        synapse.code_hash = self.code_hash
        synapse.name = self.name
        synapse.description = self.description
        return synapse

    def handle_commit_hash(self, synapse: CommitHash) -> CommitHash:
        _require_authorized_validator(synapse)
        synapse.code_hash = self.code_hash
        synapse.round_id = synapse.round_id
        return synapse

    def handle_ping(self, synapse: Ping) -> Ping:
        synapse.status = "alive"
        return synapse

    def serve(self):
        self.axon.start()
        log.info(f"Miner serving on port {self.axon.port}, hash={self.code_hash[:16]}...")

    def stop(self):
        self.axon.stop()

    def run(self):
        """Serve and block until interrupted."""
        self.serve()
        log.info("Miner running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            self.stop()



class Validator:
    """Fetches miner submissions, evaluates them, computes weights."""

    def __init__(self, wallet: Optional[bt.Wallet] = None, netuid: int = NETUID,
                 network: str = "finney", tasks: list[TaskSpec] = None,
                 dataset_name: str = "fineweb-edu", num_trials: int = 1,
                 set_weights: bool = False, submission_timeout: float = 86400.0,
                 mode: str = "prod", backend: str = "local",
                 cloud_resource: str = "b200-small",
                 baseline: str = "rolling_best", rounds_dir: str = "",
                 audit_dir: str = "",
                 burn_uid: int | None = None, min_improvement: float | None = None,
                 burn_fraction_floor: float | None = None,
                 frontier_confirmation_runs: int | None = None):
        self.wallet = wallet
        if int(netuid) != NETUID:
            raise ValueError(f"Refinery runs on netuid {NETUID} only (got {netuid})")
        self.netuid = NETUID
        self.network = network
        self.tasks = tasks or []
        self.dataset_name = dataset_name
        self.num_trials = num_trials
        self.set_weights_enabled = set_weights
        self.submission_timeout = submission_timeout
        self.mode = mode
        self.backend = backend
        if wallet is not None:
            hk = wallet.hotkey.ss58_address
            if not validator_hotkey_configured():
                raise ValueError("SN125 AUTHORIZED_VALIDATOR_HOTKEY must be set in sn125/config.py")
            if hk != AUTHORIZED_VALIDATOR_HOTKEY:
                raise ValueError("validator wallet hotkey does not match sn125/config.py")
        self._cloud_orch = None
        providers = [p.strip().lower() for p in str(backend or "").split(",")
                     if p.strip()]
        if providers and providers != ["local"]:
            from .cloud import (TargonOrchestrator, ensure_scoring_providers,
                                make_failover_client)
            providers = ensure_scoring_providers(providers)
            expected_manifest = next(
                (getattr(t, "data_manifest", "") for t in self.tasks
                 if getattr(t, "data_manifest", "")),
                "",
            )
            cloud_client = make_failover_client(providers, int(submission_timeout))
            self._cloud_orch = TargonOrchestrator(
                client=cloud_client,
                resource=cloud_resource, timeout=int(submission_timeout),
                expected_manifest=expected_manifest)
            self._cloud_orch.initialize()
        self.dendrite = bt.Dendrite(wallet=wallet) if wallet else None
        self.subtensor = bt.Subtensor(network=network) if (wallet and set_weights) else None
        self.baseline_cache: dict[str, dict] = {}
        if baseline != "rolling_best":
            raise ValueError(f"baseline must be 'rolling_best', got {baseline!r}")
        self.baseline = baseline
        self.rounds_dir = rounds_dir or str(Path(__file__).parent / "rounds")
        self.audit_dir = audit_dir or str(Path(self.rounds_dir).parent / "audit")
        self.burn_uid = BURN_UID if burn_uid is None else int(burn_uid)
        self.min_improvement = (
            MIN_FRONTIER_IMPROVEMENT if min_improvement is None else float(min_improvement)
        )
        self.frontier_confirmation_runs = (
            FRONTIER_CONFIRMATION_RUNS if frontier_confirmation_runs is None
            else max(0, int(frontier_confirmation_runs)))
        self.frontier_confirmation_max_attempts = FRONTIER_CONFIRMATION_MAX_ATTEMPTS
        self._last_weights: dict[str, float] = {}
        self.burn_fraction_floor = (
            LAUNCH_BURN_FRACTION_FLOOR if burn_fraction_floor is None
            else float(burn_fraction_floor)
        )
        self.progress_half_life_days = PROGRESS_HALF_LIFE_DAYS
        self.launch_timestamp = int(LAUNCH_TIMESTAMP or 0)
        self._burn_floor_explicit = burn_fraction_floor is not None
        self._last_frontier_rewards: dict = {}

    @staticmethod
    def _hparams_for_task(source: str, task: TaskSpec) -> tuple[float, float]:
        """Extract (lr, wd) from optimizer source for a given task."""
        cfg = {"use_pretrained": task.use_pretrained,
               "parameter_count": task.parameter_count,
               "sequence_length": task.sequence_length}
        hp = extract_hparams(source, task_config=cfg)
        return hp["lr"], hp.get("weight_decay", 0.01)

    def _cache_baselines(self, round_id: str, baselines, step_times, baseline_hp):
        """Store baseline results and evict old entries (keep last 2)."""
        self.baseline_cache[round_id] = {"baselines": baselines, "step_times": step_times,
                                          "baseline_hp": baseline_hp}
        while len(self.baseline_cache) > 2:
            oldest = next(iter(self.baseline_cache))
            del self.baseline_cache[oldest]

    def _remote_baseline_bundle(self, baselines: dict) -> dict:
        """Serialize one baseline curve per production task for remote prod-eval.

        Rolling-best mode makes this a zero-training local operation: prior winner
        curves or cold-start anchors are shipped to the B200, and the B200 spends
        its budget only on the submitted optimizer.
        """
        def _curve_dict(c: TrainingCurve) -> dict:
            return {
                "eval_points": c.eval_points,
                "train_points": c.train_points,
                "wall_seconds": c.wall_seconds,
                "state_multiplier": getattr(c, "state_multiplier", 1.0),
                "failed": c.failed,
                "error": c.error,
                "lr": c.lr,
                "wd": c.wd,
            }

        curves = {}
        for task in self.tasks:
            cs = baselines.get(task.task_id) or []
            if cs:
                curves[task.task_id] = _curve_dict(cs[0])
        return {
            "baseline_curves": curves,
            "tasks": [asdict(t) for t in self.tasks],
            "baseline_mode": self.baseline,
        }

    def sync_metagraph(self) -> bt.Metagraph:
        sub = self.subtensor or bt.Subtensor(network=self.network)
        return sub.metagraph(netuid=self.netuid)

    def compute_baselines(self, round_id: str) -> tuple[dict, dict, dict]:
        """Rolling-best baselines (zero training cost).

        Returns (baselines, step_times, baseline_hp):
          baselines[task_id]   = [TrainingCurve]  (best historical curve, or a
                                 cold-start anchor curve on round 1)
          step_times[task_id]  = {"per_step": float, "total": float, ...}
          baseline_hp[task_id] = (lr, wd) from AdamW's declared HPARAMS
                                 (presentation only)."""
        if round_id in self.baseline_cache:
            cached = self.baseline_cache[round_id]
            return cached["baselines"], cached["step_times"], cached["baseline_hp"]
        return self._compute_rolling_best_baselines(round_id)

    def _compute_rolling_best_baselines(self, round_id: str) -> tuple[dict, dict, dict]:
        """B-D: Build baselines from prior-round submission curves (zero training cost).

        For each task in self.tasks, look up the best historical curve via
        `rolling_baseline.load_rolling_best`. If found, package as a 1-curve list
        (matches the K-trials shape that downstream scoring expects). If not
        found (round-1 cold start), synthesize a curve at the task's cold-start
        anchor "ref" loss so `convergence = (base_final - sub_final) / base_final`
        remains well-defined: a miner that beats the anchor scores positive.

        BT and held-out transfer audit are SKIPPED in rolling-best mode for now —
        BT requires a curve at 2× nominal wall, which we don't store, and the
        held-out task pool may have no historical curve. They can be re-enabled
        once we run rolling-best long enough to accumulate curves.

        Returns the same `(baselines, step_times, baseline_hp)` shape as the
        AdamW path so callers don't need to special-case this mode.
        """
        from .rolling_baseline import load_rolling_best, COLD_START_ANCHORS, DEFAULT_ANCHORS
        baselines: dict[str, list[TrainingCurve]] = {}
        step_times: dict[str, dict] = {}
        baseline_hp: dict[str, tuple[float, float]] = {}

        for task in self.tasks:
            best_lr, best_wd = self._hparams_for_task(ADAMW_SOURCE, task)
            baseline_hp[task.task_id] = (best_lr, best_wd)

            rb = load_rolling_best(self.rounds_dir, task, exclude_round_id=round_id)
            if rb is not None:
                curve = TrainingCurve(
                    task_id=task.task_id, lr=best_lr, wd=best_wd,
                    eval_points=[tuple(p) for p in rb.eval_points],
                    train_points=[tuple(p) for p in rb.train_points],
                    wall_seconds=rb.wall_seconds,
                    state_multiplier=rb.state_multiplier,
                    failed=False, error="")
                baselines[task.task_id] = [curve]
                actual_steps = max(curve.eval_points[-1][0], 1) if curve.eval_points else max(task.total_steps, 1)
                step_times[task.task_id] = {
                    "per_step": rb.wall_seconds / actual_steps if rb.wall_seconds > 0 else 0.0,
                    "total": rb.wall_seconds,
                    "rolling_best_provenance": {
                        "round_id": rb.source_round_id,
                        "code_hash": rb.source_code_hash,
                        "score": rb.source_score,
                        "final_eval_loss": rb.final_eval_loss}}
                log.info(f"  {task.task_id}: rolling-best from round={rb.source_round_id} "
                         f"hash={rb.source_code_hash} loss={rb.final_eval_loss:.4f}")
            else:
                init_loss, ref_loss, _floor_loss = COLD_START_ANCHORS.get(
                    task.task_id, DEFAULT_ANCHORS)
                _budget = float(getattr(task, "compute_budget_seconds", 0.0) or 0.0)
                synth_wall = _budget if _budget > 0 else 1.0
                synth_steps = max(task.total_steps, 1)
                synth_curve = TrainingCurve(
                    task_id=task.task_id, lr=best_lr, wd=best_wd,
                    eval_points=[(0, float(init_loss), 0.0),
                                 (synth_steps, float(ref_loss), synth_wall)],
                    train_points=[(0, float(init_loss)),
                                  (synth_steps, float(ref_loss))],
                    wall_seconds=synth_wall,
                    state_multiplier=0.0,
                    failed=False, error="")
                baselines[task.task_id] = [synth_curve]
                step_times[task.task_id] = {
                    "per_step": synth_wall / synth_steps,
                    "total": synth_wall,
                    "rolling_best_provenance": {
                        "round_id": "cold_start",
                        "code_hash": "",
                        "score": 0.0,
                        "final_eval_loss": float(ref_loss),
                        "anchor": [float(init_loss), float(ref_loss), float(_floor_loss)]}}
                log.info(f"  {task.task_id}: cold-start anchor "
                         f"(init={init_loss}, ref={ref_loss}, budget={synth_wall:.0f}s)")

        self._cache_baselines(round_id, baselines, step_times, baseline_hp)
        return baselines, step_times, baseline_hp

    BURN_KEY = "__BURN__"

    @staticmethod
    def _task_losses(sr: ScoreRecord, key: str = "sub_final_loss") -> dict[str, float]:
        """Per-task finite final held-out losses recorded on a score record
        (`sub_final_loss` = the submission's run, `base_final_loss` = the
        rolling-best baseline it was scored against)."""
        out: dict[str, float] = {}
        for tid, d in (sr.task_scores or {}).items():
            if not isinstance(d, dict):
                continue
            v = d.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            v = float(v)
            if math.isfinite(v) and v > 0.0:
                out[tid] = v
        return out

    @staticmethod
    def _improvement_nats(base_losses: dict[str, float],
                          sub_losses: dict[str, float]) -> float | None:
        """Frontier improvement in nats: min over the common tasks of
        (baseline loss - submission loss). None when no task is comparable."""
        deltas = [float(base_losses[t]) - float(sub_losses[t])
                  for t in sub_losses if t in base_losses]
        return min(deltas) if deltas else None

    @classmethod
    def _improvement_and_significance(cls, sr: ScoreRecord) -> tuple[float, bool]:
        """Extract a submission's improvement magnitude g_i and whether it is
        significant over the seed-noise floor.

        g_i is the held-out final-loss improvement over the active baseline in
        NATS (`L_base - L_sub`, worst task), read from the recorded losses so the
        bar is in the unit the replication noise is measured in. A record that
        carries no losses (dev/test doubles) falls back to the relative
        `cross_scale` component — a STRICTER reading of the same constant, never
        a looser one."""
        base = cls._task_losses(sr, "base_final_loss")
        sub = cls._task_losses(sr, "sub_final_loss")
        g = cls._improvement_nats(base, sub)
        if g is None:
            comp = sr.components or {}
            g = comp.get("cross_scale", comp.get("convergence", sr.final_score))
            try:
                g = float(g)
            except (TypeError, ValueError):
                g = sr.final_score
        comp = sr.components or {}
        sp = comp.get("significance_pareto") or {}
        if sp.get("evaluable"):
            significant = bool(sp.get("sig_wins")) and not sp.get("sig_losses")
        else:
            significant = g > 0.0
        return g, significant

    def _frontier_task_signatures(self) -> dict[str, str]:
        from .rolling_baseline import task_signature
        return {t.task_id: task_signature(t) for t in self.tasks}

    @staticmethod
    def _frontier_event_id(event: dict) -> str:
        material = json.dumps({
            "hotkey": event.get("hotkey", ""),
            "code_hash": event.get("code_hash", ""),
            "timestamp": int(event.get("timestamp", 0)),
            "improvement": round(float(event.get("improvement", 0.0) or 0.0), 12),
            "task_signatures": event.get("task_signatures", {}),
        }, sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def _load_frontier_events(self) -> list[dict]:
        """Load prior frontier-setting events for this exact task panel."""
        rdir = Path(self.rounds_dir)
        if not rdir.is_dir():
            return []
        sigs = self._frontier_task_signatures()
        events, seen = [], set()
        for fp in sorted(rdir.glob("*.json")):
            try:
                d = json.loads(fp.read_text())
            except Exception:
                continue
            fr = d.get("frontier_rewards") or {}
            ev = fr.get("new_event")
            if not isinstance(ev, dict):
                continue
            if ev.get("task_signatures") != sigs:
                continue
            try:
                imp = float(ev.get("improvement", 0.0))
            except (TypeError, ValueError):
                continue
            if imp <= 0.0:
                continue
            ev = dict(ev)
            ev["improvement"] = imp
            ev.setdefault("timestamp", int(d.get("timestamp", 0) or 0))
            ev.setdefault("event_id", self._frontier_event_id(ev))
            if ev["event_id"] in seen:
                continue
            seen.add(ev["event_id"])
            events.append(ev)
        events.sort(key=lambda e: (int(e.get("timestamp", 0) or 0), e.get("event_id", "")))
        return events

    @staticmethod
    def _source_hash_for_hotkey(hotkey: str, sources: dict[str, str] | None) -> str:
        if not sources or hotkey not in sources:
            return ""
        return hashlib.sha256(sources[hotkey].encode()).hexdigest()

    @staticmethod
    def _frontier_loss_snapshot(sr: ScoreRecord) -> dict:
        out = {}
        for tid, d in (sr.task_scores or {}).items():
            if isinstance(d, dict) and "sub_final_loss" in d:
                out[tid] = d["sub_final_loss"]
        return out

    def _load_pending_confirmations(self) -> list[dict]:
        """Bar-clearing candidates awaiting their validator-funded confirmation
        run. The validator rewrites the full pending list into every round JSON
        (`frontier_rewards.pending_confirmation`), so the newest round that
        carries the key is authoritative; entries are filtered to this exact
        task panel."""
        rdir = Path(self.rounds_dir)
        if not rdir.is_dir():
            return []
        sigs = self._frontier_task_signatures()
        newest: tuple[int, str] | None = None
        pending: list[dict] = []
        for fp in sorted(rdir.glob("*.json")):
            try:
                d = json.loads(fp.read_text())
            except Exception:
                continue
            fr = d.get("frontier_rewards")
            if not isinstance(fr, dict) or "pending_confirmation" not in fr:
                continue
            key = (int(d.get("timestamp", 0) or 0), fp.name)
            if newest is not None and key <= newest:
                continue
            newest = key
            pending = [dict(p) for p in (fr.get("pending_confirmation") or [])
                       if isinstance(p, dict) and p.get("code_hash")
                       and (p.get("task_signatures") or sigs) == sigs]
        return pending

    def _pending_source(self, pending: dict) -> str:
        """Recover the revealed source of a pending candidate from the round
        JSON that scored it (`sources` is keyed by code hash)."""
        rdir = Path(self.rounds_dir)
        code_hash = str(pending.get("code_hash", ""))
        if not code_hash or not rdir.is_dir():
            return ""
        names = []
        rid = str(pending.get("source_round_id", "") or "")
        if rid:
            names.append(rdir / f"{rid}.json")
        names.extend(sorted(rdir.glob("*.json"), reverse=True))
        for fp in names:
            try:
                d = json.loads(fp.read_text())
            except Exception:
                continue
            src = (d.get("sources") or {}).get(code_hash)
            if isinstance(src, str) and src:
                return src
        return ""

    def _min_improvement_threshold(self, events: list[dict], now: float) -> float:
        """Adaptive frontier bar: a new event must improve by more than

            0.5 × mean(improvement of the last 3 frontier events)
                × 0.5 ** (days_since_last_event / 7)

        floored at `self.min_improvement`. Early rounds see large real
        improvements, so the bar sits well above the run-to-run noise floor and
        a copycat resubmission of the leader cannot win on noise. Once progress
        stalls, the bar decays (halving per stagnant week) so a genuine small
        advance is not locked out forever. With no prior events (round 1) only
        the absolute floor applies."""
        recent = [float(ev.get("improvement", 0.0) or 0.0)
                  for ev in events[-FRONTIER_THRESHOLD_WINDOW:]]
        recent = [g for g in recent if g > 0.0]
        if not recent:
            return self.min_improvement
        bar = FRONTIER_THRESHOLD_FRACTION * (sum(recent) / len(recent))
        last_ts = max(int(ev.get("timestamp", 0) or 0) for ev in events)
        stagnant_days = max(0.0, (now - last_ts) / 86400.0)
        bar *= 0.5 ** (stagnant_days / FRONTIER_THRESHOLD_DECAY_HALF_LIFE_DAYS)
        return max(self.min_improvement, bar)

    def _clamped_burn_fraction_floor(self, now: float | None = None) -> float:
        """Burn floor in force: the launch schedule's phase for today when
        LAUNCH_TIMESTAMP is set (and no explicit --burn-fraction-floor
        override was given), else the configured constant."""
        floor = None
        if not getattr(self, "_burn_floor_explicit", False):
            day = launch_day(time.time() if now is None else now,
                             getattr(self, "launch_timestamp", 0) or None)
            if day is not None:
                floor = launch_burn_fraction_for_day(day)
        if floor is None:
            try:
                floor = float(self.burn_fraction_floor)
            except (TypeError, ValueError):
                floor = 0.0
        if not math.isfinite(floor):
            floor = 0.0
        return min(1.0, max(0.0, floor))

    def compute_weights(self, results: dict[str, ScoreRecord],
                        temperature: float = 0.5,
                        sources: dict[str, str] | None = None,
                        curve_data: dict[str, dict] | None = None,
                        confirmations: dict[str, object] | None = None,
                        round_id: str = "", now: float | None = None) -> dict[str, float]:
        """Frontier reward path (progress-scaled payout, config.py).

        Only confirmed frontier events earn. Each carries a decaying credit

            credit_i(t) = improvement_i * 0.5 ** (age_days / PROGRESS_HALF_LIFE_DAYS)

        and the round pays

            payable = min(1, sum_i credit_i / full_pay_reference(t)) * (1 - burn_floor)
            weight_i = payable * credit_i / sum_i credit_i
            burn     = 1 - payable

        so emission tracks recent verified progress (stagnation burns more
        every half-life) and is split across every contributor pro rata to
        the size and recency of their improvement — no leader bonus, no
        winner-take-all. A non-winning current submission receives zero even
        if it beats the old baseline but loses to another same-round submission.

        Confirmation (``frontier_confirmation_runs`` > 0): a submission that
        clears the adaptive bar this round is recorded as PENDING, not paid.
        The round loop re-runs its source next round and passes the outcome in
        ``confirmations`` ({code_hash: ScoreRecord | {"outcome", "reason"}}).
        The frontier event is created only when the confirmation also clears
        the bar, with the WORSE of the two runs as the frontier loss.

        `temperature` is accepted for back-compat but unused. ``now`` (unix
        seconds) defaults to the wall clock; sn125/verify.py passes a round's
        recorded timestamp to replay its settlement from public data.
        """
        now = int(time.time() if now is None else now)
        weights = {hk: 0.0 for hk in results}
        events = self._load_frontier_events()
        confirmations = confirmations or {}
        conf_runs = int(self.frontier_confirmation_runs)
        pending = self._load_pending_confirmations() if conf_runs > 0 else []
        min_improvement = self._min_improvement_threshold(events, now)
        sigs = self._frontier_task_signatures()

        outcomes: list[dict] = []
        confirmed: list[tuple] = []
        still_pending: list[dict] = []
        for p in pending:
            code_hash = str(p.get("code_hash", ""))
            rec = {"hotkey": p.get("hotkey", ""), "code_hash": code_hash,
                   "submission_round_id": p.get("source_round_id", "")}
            c = confirmations.get(code_hash)
            if c is None:
                still_pending.append(dict(p))
                outcomes.append({**rec, "outcome": "pending",
                                 "reason": "no confirmation result this round"})
                continue
            if isinstance(c, ScoreRecord) and c.final_score > -0.99:
                base = self._task_losses(c, "base_final_loss")
                conf_losses = self._task_losses(c, "sub_final_loss")
                run1 = {t: float(v) for t, v in (p.get("task_final_losses") or {}).items()
                        if isinstance(v, (int, float)) and math.isfinite(float(v))}
                worst = {t: max(run1.get(t, conf_losses[t]), conf_losses[t])
                         for t in conf_losses}
                g_run1 = self._improvement_nats(base, run1)
                g_conf = self._improvement_nats(base, conf_losses)
                g = self._improvement_nats(base, worst)
                if (g is not None and g_run1 is not None and g_conf is not None
                        and g > min_improvement and g_run1 > min_improvement
                        and g_conf > min_improvement):
                    done = int(p.get("confirmations_done", 0) or 0) + 1
                    p2 = dict(p)
                    p2["task_final_losses"] = worst
                    p2["confirmations_done"] = done
                    p2.setdefault("confirmation_losses", []).append(conf_losses)
                    if done >= conf_runs:
                        confirmed.append((float(g), p2, c, worst, base, run1, conf_losses))
                    else:
                        still_pending.append(p2)
                        outcomes.append({**rec, "outcome": "confirmed_partial",
                                         "improvement": float(g),
                                         "confirmations_done": done,
                                         "confirmations_required": conf_runs})
                else:
                    outcomes.append({**rec, "outcome": "rejected",
                                     "reason": "confirmation run did not clear the bar",
                                     "improvement_submission": g_run1,
                                     "improvement_confirmation": g_conf,
                                     "bar": min_improvement,
                                     "baseline_losses": base,
                                     "confirmation_losses": conf_losses})
                continue
            if isinstance(c, ScoreRecord):
                verdict, reason = "dq", "confirmation run failed"
            elif isinstance(c, dict):
                verdict = str(c.get("outcome", "infra_dq"))
                reason = str(c.get("reason", ""))
            else:
                verdict, reason = "infra_dq", str(c)
            if verdict == "dq":
                outcomes.append({**rec, "outcome": "rejected",
                                 "reason": f"confirmation run failed: {reason}"})
                continue
            attempts = int(p.get("attempts", 0) or 0) + 1
            if attempts >= int(self.frontier_confirmation_max_attempts):
                outcomes.append({**rec, "outcome": "rejected",
                                 "reason": (f"confirmation could not be run after "
                                            f"{attempts} attempts: {reason}")})
            else:
                p2 = dict(p)
                p2["attempts"] = attempts
                still_pending.append(p2)
                outcomes.append({**rec, "outcome": "retry", "attempts": attempts,
                                 "reason": reason})

        new_event = None
        if confirmed:
            confirmed.sort(key=lambda x: (-x[0], x[1].get("code_hash", "")))
            g, p, c, worst, base, run1, conf_losses = confirmed[0]
            primary = next(iter(worst), "")
            if primary and run1.get(primary, float("inf")) >= conf_losses.get(primary, float("inf")):
                curve_ref = {"round_id": p.get("source_round_id", ""),
                             "section": "submissions", "key": p.get("hotkey", "")}
            else:
                curve_ref = {"round_id": round_id, "section": "confirmations",
                             "key": self._confirmation_key(p.get("code_hash", ""))}
            new_event = {
                "event_id": "",
                "hotkey": p.get("hotkey", ""),
                "code_hash": p.get("code_hash", ""),
                "timestamp": now,
                "improvement": float(g),
                "improvement_unit": FRONTIER_IMPROVEMENT_UNIT,
                "final_score": float(c.final_score),
                "task_signatures": sigs,
                "task_final_losses": worst,
                "baseline_losses": base,
                "confirmation": {
                    "submission_round_id": p.get("source_round_id", ""),
                    "confirmation_round_id": round_id,
                    "runs": {"submission": run1,
                             "confirmations": list(p.get("confirmation_losses") or [conf_losses])},
                },
                "baseline_curve_ref": curve_ref,
            }
            new_event["event_id"] = self._frontier_event_id(new_event)
            events.append(new_event)
            outcomes.append({"hotkey": new_event["hotkey"], "code_hash": new_event["code_hash"],
                             "submission_round_id": p.get("source_round_id", ""),
                             "outcome": "confirmed", "improvement": float(g),
                             "event_id": new_event["event_id"]})
            for g2, p2, *_ in confirmed[1:]:
                outcomes.append({"hotkey": p2.get("hotkey", ""), "code_hash": p2.get("code_hash", ""),
                                 "submission_round_id": p2.get("source_round_id", ""),
                                 "outcome": "superseded", "improvement": float(g2),
                                 "reason": "a better candidate confirmed in the same round"})

        frontier_losses = new_event["task_final_losses"] if new_event else None
        candidates: list[tuple[float, str, ScoreRecord]] = []
        for hk, sr in results.items():
            if sr.final_score <= -0.99:
                continue
            g, significant = self._improvement_and_significance(sr)
            if frontier_losses:
                g2 = self._improvement_nats(frontier_losses, self._task_losses(sr, "sub_final_loss"))
                if g2 is not None:
                    g = g2
            if significant and g > min_improvement:
                candidates.append((max(0.0, g), hk, sr))

        new_pending = None
        if candidates:
            g, hk, sr = max(candidates, key=lambda x: (x[0], x[2].final_score))
            code_hash = self._source_hash_for_hotkey(hk, sources)
            sub_losses = self._frontier_loss_snapshot(sr)
            if conf_runs <= 0:
                ev = {
                    "event_id": "",
                    "hotkey": hk,
                    "code_hash": code_hash,
                    "timestamp": now,
                    "improvement": float(g),
                    "improvement_unit": FRONTIER_IMPROVEMENT_UNIT,
                    "final_score": float(sr.final_score),
                    "task_signatures": sigs,
                    "task_final_losses": sub_losses,
                    "baseline_curve_ref": {"round_id": round_id, "section": "submissions",
                                           "key": hk},
                }
                ev["event_id"] = self._frontier_event_id(ev)
                events.append(ev)
                new_event = ev
            elif code_hash and any(p.get("code_hash") == code_hash for p in still_pending):
                outcomes.append({"hotkey": hk, "code_hash": code_hash, "outcome": "duplicate",
                                 "reason": "same source already awaiting confirmation"})
            else:
                new_pending = {
                    "hotkey": hk,
                    "code_hash": code_hash,
                    "source_round_id": round_id,
                    "timestamp": now,
                    "improvement": float(g),
                    "improvement_unit": FRONTIER_IMPROVEMENT_UNIT,
                    "final_score": float(sr.final_score),
                    "task_signatures": sigs,
                    "task_final_losses": sub_losses,
                    "baseline_losses": self._task_losses(sr, "base_final_loss"),
                    "attempts": 0,
                    "confirmations_done": 0,
                    "confirmation_losses": [],
                }
                still_pending.append(new_pending)
                outcomes.append({"hotkey": hk, "code_hash": code_hash,
                                 "submission_round_id": round_id,
                                 "outcome": "pending_new", "improvement": float(g),
                                 "reason": "cleared the bar; awaiting confirmation run"})

        audit_common = {
            "scheme": "frontier-progress-scaled-v3",
            "improvement_unit": FRONTIER_IMPROVEMENT_UNIT,
            "confirmation_runs": conf_runs,
            "half_life_days": self.progress_half_life_days,
            "launch_timestamp": self.launch_timestamp,
            "burn_fraction_floor": self._clamped_burn_fraction_floor(now),
            "burn_uid": self.burn_uid,
            "min_improvement_threshold": min_improvement,
            "new_event": new_event,
            "pending_confirmation": still_pending,
            "confirmation_outcomes": outcomes,
        }

        if not events:
            weights[self.BURN_KEY] = 1.0
            self._last_frontier_rewards = {
                **audit_common,
                "burned_share": 1.0,
                "payable_share": 0.0,
                "leader_hotkey": "",
                "events": [],
            }
            self._last_weights = dict(weights)
            return weights

        leader_hk = events[-1].get("hotkey", "")
        half_life_s = max(1.0, self.progress_half_life_days * 86400.0)
        active_events = []
        total_credit = 0.0
        first_ts = None
        for ev in events:
            try:
                imp = float(ev.get("improvement", 0.0))
                ts = int(ev.get("timestamp", now) or now)
            except (TypeError, ValueError):
                continue
            first_ts = ts if first_ts is None else min(first_ts, ts)
            age_s = max(0.0, now - ts)
            credit = imp * (0.5 ** (age_s / half_life_s))
            if credit <= 0.0:
                continue
            item = dict(ev)
            item["decayed_credit"] = credit
            item["age_days"] = age_s / 86400.0
            active_events.append(item)
            total_credit += credit

        anchor = self.launch_timestamp or first_ts
        full_pay_ref = full_pay_reference_nats(now, anchor)
        progress_share = progress_payable_share(total_credit, now, anchor)
        burn_floor = self._clamped_burn_fraction_floor(now)
        payable = progress_share * (1.0 - burn_floor)
        for ev in active_events:
            hk = ev.get("hotkey", "")
            if not hk or total_credit <= 0.0:
                continue
            ev["share"] = ev["decayed_credit"] / total_credit
            weights[hk] = weights.get(hk, 0.0) + payable * ev["share"]
        paid = sum(w for hk, w in weights.items() if hk != self.BURN_KEY)
        burned = max(0.0, 1.0 - paid)
        if burned > 1e-12:
            weights[self.BURN_KEY] = burned
        else:
            weights.pop(self.BURN_KEY, None)

        self._last_frontier_rewards = {
            **audit_common,
            "progress_nats": total_credit,
            "full_pay_reference_nats": full_pay_ref,
            "progress_payable_share": progress_share,
            "ramp_anchor_timestamp": anchor,
            "burned_share": burned,
            "payable_share": paid,
            "leader_hotkey": leader_hk,
            "events": active_events,
        }
        self._last_weights = dict(weights)
        return weights

    @staticmethod
    def _confirmation_key(code_hash: str) -> str:
        """Round-JSON key (and cloud sub_uid stem) for a confirmation run."""
        return f"confirm-{str(code_hash)[:16]}"

    def _baselines_and_hashes(self, sources, curve_data):
        """Shared extraction used by _save_round and _build_attestation."""
        baselines = {}
        if curve_data:
            first = next(iter(curve_data.values()), {})
            baselines = first.get("baseline_curves", {})
        code_hashes = {hk: hashlib.sha256(s.encode()).hexdigest()
                       for hk, s in (sources or {}).items()}
        sources_by_hash = ({hashlib.sha256(s.encode()).hexdigest(): s
                            for s in sources.values()} if sources else {})
        return baselines, code_hashes, sources_by_hash

    def _task_dicts(self):
        return [{"task_id": t.task_id, "model_config": t.model_config,
                 "total_steps": t.total_steps, "batch_size": t.batch_size,
                 "sequence_length": t.sequence_length, "use_pretrained": t.use_pretrained,
                 "task_weight": t.task_weight, "eval_every": t.eval_every,
                 "compute_budget_seconds": t.compute_budget_seconds,
                 "dataset": getattr(t, "dataset", "fineweb-edu"),
                 "data_manifest": getattr(t, "data_manifest", "")} for t in self.tasks]

    def _submission_curves(self, hk, curve_data):
        if curve_data and hk in curve_data:
            cd = curve_data[hk]
            out = {"curves": cd.get("curves", {}),
                   "hparams_per_task": cd.get("hparams_per_task", {})}
            if cd.get("provenance"):
                out["provenance"] = cd["provenance"]
            cps = cd.get("checkpoints")
            if not cps:
                agg = []
                for tid, c in (cd.get("curves") or {}).items():
                    for rec in (c.get("checkpoints") or []):
                        agg.append({"task_id": tid, **rec})
                cps = agg
            cps = _normalize_checkpoints(cps)
            if cps:
                out["checkpoints"] = cps
            return out
        return {}

    def _save_round(self, round_id: str, results: dict[str, ScoreRecord],
                     weights: dict[str, float], sources: dict[str, str] = None,
                     curve_data: dict[str, dict] = None,
                     pause_reasons: list = None,
                     round_report: dict = None,
                     confirmation_records: dict = None):
        """Persist round results as JSON for dashboard consumption.

        ``confirmation_records`` ({confirm-key: {hotkey, code_hash, score,
        curves, outcome}}) are the validator-funded frontier confirmation
        runs of this round; they are stored under their own top-level
        ``confirmations`` section (never under ``submissions``, which is the
        paid-miner ledger) so the rolling baseline can follow a frontier
        event's ``baseline_curve_ref`` to the worse of the two runs."""
        baselines, code_hashes, sources_by_hash = self._baselines_and_hashes(sources, curve_data)
        submissions_out = {}
        for hk, sr in results.items():
            entry = {"score": asdict(sr), "weight": weights.get(hk, 0.0)}
            if hk in code_hashes:
                entry["code_hash"] = code_hashes[hk]
            entry.update(self._submission_curves(hk, curve_data))
            submissions_out[hk] = entry

        out = {
            "round_id": round_id, "timestamp": int(time.time()),
            "public_seed_manifest": public_seed_manifest(self.num_trials),
            "tasks": self._task_dicts(), "baselines": baselines,
            "submissions": submissions_out, "weights": weights,
            "economics": {
                "burned_weight": float(weights.get(self.BURN_KEY, 0.0) or 0.0),
                "paid_weight": max(0.0, 1.0 - float(weights.get(self.BURN_KEY, 0.0) or 0.0)),
                "burn_uid": self.burn_uid,
                "burn_fraction_floor": self._clamped_burn_fraction_floor(),
            },
            "code_hashes": code_hashes,
            "frontier_rewards": getattr(self, "_last_frontier_rewards", {}),
        }
        if confirmation_records:
            out["confirmations"] = confirmation_records
        if pause_reasons:
            out["pause_reasons"] = list(pause_reasons)
        if round_report:
            out["fsm_report"] = round_report
            out["selection"] = {
                "rule": round_report.get("selection_rule", ""),
                "selected": list(round_report.get("selected", []) or []),
                "deferred": list(round_report.get("deferred", []) or []),
                "commit_rejections": list(round_report.get("commit_rejections", []) or []),
            }
            out["deferred"] = list(round_report.get("deferred", []) or [])
        if sources_by_hash:
            out["sources"] = sources_by_hash
        out["_writer_pid"] = os.getpid()
        outdir = Path(self.rounds_dir); outdir.mkdir(parents=True, exist_ok=True)
        dest = outdir / f"{round_id}.json"
        if dest.exists():
            try:
                existing = json.loads(dest.read_text())
                epid = existing.get("_writer_pid")
                if epid and epid != os.getpid():
                    try:
                        os.kill(epid, 0)
                        logging.warning("Round file %s owned by live PID %s, refusing to overwrite", dest, epid)
                        return
                    except OSError:
                        pass
            except Exception:
                pass
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", dir=str(outdir), suffix=".tmp", delete=False)
        try:
            tmp.write(json.dumps(out, indent=2, default=str))
            tmp.close()
            os.replace(tmp.name, str(dest))
        except Exception:
            try: os.unlink(tmp.name)
            except OSError: pass
            raise

    def _set_weights_on_chain(self, meta: bt.Metagraph, weights: dict[str, float]):
        """Set weights on chain. Only called when --set-weights is enabled."""
        if not (self.subtensor and self.wallet):
            return
        _set_weights_from_map(self.subtensor, self.wallet, self.netuid, meta, weights,
                              burn_uid=self.burn_uid)

    def _build_attestation(self, round_id: str, results: dict[str, ScoreRecord],
                            weights: dict[str, float], sources: dict[str, str] = None,
                            curve_data: dict[str, dict] = None,
                            pause_reasons: list = None,
                            round_report: dict = None) -> dict:
        """Build signed attestation bundle for R2 publishing."""
        baselines, code_hashes, sources_by_hash = self._baselines_and_hashes(sources, curve_data)
        submissions = {}
        for hk, sr in results.items():
            entry = {"final_score": sr.final_score, "components": sr.components,
                     "task_scores": sr.task_scores, "best_hparams": sr.best_hparams,
                     "failed_tasks": sr.failed_tasks,
                     "weight": weights.get(hk, 0.0)}
            if hk in code_hashes:
                entry["code_hash"] = code_hashes[hk]
            entry.update(self._submission_curves(hk, curve_data))
            submissions[hk] = entry

        bundle = {
            "round_id": round_id, "timestamp": int(time.time()),
            "version": SN125_VERSION, "netuid": self.netuid,
            "validator_hotkey": self.wallet.hotkey.ss58_address if self.wallet else None,
            "public_seed_manifest": public_seed_manifest(self.num_trials),
            "tasks": self._task_dicts(), "baselines": baselines,
            "submissions": submissions, "weights": weights,
            "economics": {
                "burned_weight": float(weights.get(self.BURN_KEY, 0.0) or 0.0),
                "paid_weight": max(0.0, 1.0 - float(weights.get(self.BURN_KEY, 0.0) or 0.0)),
                "burn_uid": self.burn_uid,
                "burn_fraction_floor": self._clamped_burn_fraction_floor(),
            },
            "code_hashes": code_hashes,
            "frontier_rewards": getattr(self, "_last_frontier_rewards", {}),
        }
        if pause_reasons:
            bundle["pause_reasons"] = list(pause_reasons)
        if round_report:
            bundle["fsm_report"] = round_report
            bundle["selection"] = {
                "rule": round_report.get("selection_rule", ""),
                "selected": list(round_report.get("selected", []) or []),
                "deferred": list(round_report.get("deferred", []) or []),
                "commit_rejections": list(round_report.get("commit_rejections", []) or []),
            }
            bundle["deferred"] = list(round_report.get("deferred", []) or [])
        if sources_by_hash:
            bundle["sources"] = sources_by_hash
        payload = _signing_payload(bundle)
        bundle["bundle_hash"] = hashlib.sha256(payload).hexdigest()
        if self.wallet:
            bundle["signature"] = self.wallet.hotkey.sign(payload).hex()
        return bundle

    def _publish_attestation(self, round_id: str, results: dict[str, ScoreRecord],
                              weights: dict[str, float], sources: dict[str, str] = None,
                              curve_data: dict[str, dict] = None,
                              pause_reasons: list = None,
                              round_report: dict = None):
        """Publish round attestation to R2. Returns True only after durable publish.
        Publishes both full round file (with sources) and lightweight latest.json (without)."""
        bucket = settings.r2_config()["bucket"]
        if not settings.r2_configured():
            log.info("  R2 not configured (set R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET) — skipping attestation publish")
            return False
        try:
            bundle = self._build_attestation(round_id, results, weights, sources, curve_data,
                                             pause_reasons=pause_reasons,
                                             round_report=round_report)
            s3 = settings.make_r2_client()
            body = json.dumps(bundle, indent=2, default=str)
            s3.put_object(Bucket=bucket, Key=f"rounds/{round_id}.json",
                          Body=body.encode(), ContentType="application/json")
            log.info(f"  Full round published to R2: rounds/{round_id}.json ({len(body)} bytes)")
            latest = {k: v for k, v in bundle.items() if k != "sources"}
            latest_body = json.dumps(latest, indent=2, default=str)
            s3.put_object(Bucket=bucket, Key="latest.json",
                          Body=latest_body.encode(), ContentType="application/json")
            log.info(f"  latest.json updated ({len(latest_body)} bytes)")
            self._publish_dashboard_json(s3, bucket, round_id, bundle)
            return True
        except Exception as e:
            log.warning(f"  R2 publish failed: {e}")
            return False

    def _publish_dashboard_json(self, s3, bucket: str, round_id: str, latest_bundle: dict):
        """Publish the current web contract, not the obsolete leaderboard-only feed."""
        try:
            from .dashboard.snapshot import build_snapshot
            registry = getattr(self, "payment_registry", None)
            audit_dir = Path(getattr(self, "audit_dir", "") or Path(self.rounds_dir).parent / "audit")
            snapshot = build_snapshot(
                Path(self.rounds_dir), Path(__file__).resolve().parent / "cloud_status.json",
                validator_hotkey=latest_bundle.get("validator_hotkey", ""),
                version=SN125_VERSION,
                ledger_path=Path(getattr(registry, "store_path", None)
                                 or audit_dir / "payments" / "ledger.json"),
                state_path=Path(getattr(self, "state_path", None)
                                or audit_dir / "validator_state.json"),
                audit_dir=audit_dir,
            )
            for key, value in (("dashboard.json", snapshot["dashboard"]),
                               ("cloud-status.json", snapshot["cloud"]),
                               ("snapshot.json", snapshot)):
                s3.put_object(Bucket=bucket, Key=key,
                              Body=json.dumps(value, allow_nan=False, default=str).encode(),
                              ContentType="application/json", CacheControl="no-cache")
            log.info("  dashboard snapshot published for %s", round_id)
            return True
        except Exception as exc:
            log.warning("  dashboard snapshot publish failed (non-fatal): %s",
                        type(exc).__name__)
            return False


class AttestationValidator:
    """Lightweight validator that trusts a core validator's attestation from R2.
    Fetches the latest round attestation, verifies the signature, sets weights."""

    def __init__(self, wallet: bt.Wallet, netuid: int = NETUID, network: str = "finney",
                 r2_endpoint: str = None, r2_access_key: str = None,
                 r2_secret_key: str = None, r2_bucket: str = None,
                 trusted_hotkey: str = None, poll_interval: int = 300,
                 set_weights: bool = False):
        self.wallet = wallet
        if int(netuid) != NETUID:
            raise ValueError(f"Refinery runs on netuid {NETUID} only (got {netuid})")
        self.netuid = NETUID
        self.network = network
        self.subtensor = bt.Subtensor(network=network)
        self.r2_endpoint = r2_endpoint or os.environ.get("R2_ENDPOINT", "")
        self.r2_access_key = r2_access_key or os.environ.get("R2_ACCESS_KEY", "")
        self.r2_secret_key = r2_secret_key or os.environ.get("R2_SECRET_KEY", "")
        self.r2_bucket = r2_bucket or os.environ.get("R2_BUCKET", "")
        self.trusted_hotkey = (
            trusted_hotkey
            or os.environ.get("TRUSTED_VALIDATOR_HOTKEY", "")
            or (AUTHORIZED_VALIDATOR_HOTKEY if validator_hotkey_configured() else "")
        )
        if not self.trusted_hotkey:
            raise ValueError("trusted_hotkey is required — without it, any keypair can set arbitrary weights. "
                             "Set --trusted-hotkey or TRUSTED_VALIDATOR_HOTKEY env var.")
        self.poll_interval = poll_interval
        self.set_weights_enabled = set_weights
        self._state_file = Path(__file__).resolve().parent / "attest_state.json"
        self.last_round_id = self._load_last_round_id()

    def _load_last_round_id(self) -> Optional[str]:
        try:
            return json.loads(self._state_file.read_text()).get("last_round_id")
        except Exception:
            return None

    def _save_last_round_id(self, round_id: str):
        try:
            self._state_file.write_text(json.dumps({"last_round_id": round_id}))
        except Exception as e:
            log.warning(f"Failed to persist last_round_id: {e}")

    def _fetch_latest_attestation(self) -> Optional[dict]:
        """Fetch the most recent round attestation from R2 via latest.json (O(1)).
        Falls back to listing rounds/ prefix if latest.json doesn't exist."""
        import boto3
        s3 = boto3.client("s3", endpoint_url=self.r2_endpoint,
                          aws_access_key_id=self.r2_access_key,
                          aws_secret_access_key=self.r2_secret_key)
        try:
            obj = s3.get_object(Bucket=self.r2_bucket, Key="latest.json")
            return json.loads(obj["Body"].read().decode())
        except s3.exceptions.NoSuchKey:
            pass
        except Exception as e:
            log.debug(f"  latest.json fetch failed, falling back to listing: {e}")
        try:
            resp = s3.list_objects_v2(Bucket=self.r2_bucket, Prefix="rounds/",
                                      MaxKeys=1000)
            contents = resp.get("Contents", [])
            if not contents:
                return None
            newest = max(contents, key=lambda o: o["LastModified"])
            obj = s3.get_object(Bucket=self.r2_bucket, Key=newest["Key"])
            return json.loads(obj["Body"].read().decode())
        except Exception:
            return None

    def _verify_attestation(self, bundle: dict) -> bool:
        """Verify the attestation signature against the publisher's hotkey.
        Non-destructive: reads but does not modify the bundle dict."""
        sig_hex = bundle.get("signature")
        bundle_hash = bundle.get("bundle_hash")
        if not sig_hex or not bundle_hash:
            log.warning("Attestation missing signature or bundle_hash")
            return False
        validator_hk = bundle.get("validator_hotkey", "")
        if self.trusted_hotkey and validator_hk != self.trusted_hotkey:
            log.warning(f"Attestation from untrusted hotkey {validator_hk[:16]}")
            return False
        payload = _signing_payload(bundle)
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != bundle_hash:
            log.warning(f"Bundle hash mismatch: expected {bundle_hash[:16]}, got {actual_hash[:16]}")
            return False
        ts = bundle.get("timestamp", 0)
        age = time.time() - ts
        max_age = 3 * 24 * 3600
        if age > max_age:
            log.warning(f"Attestation too old: {age:.0f}s > {max_age}s max age")
            return False
        try:
            kp = bt.Keypair(ss58_address=validator_hk)
            sig_bytes = bytes.fromhex(sig_hex)
            if not kp.verify(payload, sig_bytes):
                log.warning("Attestation signature verification failed")
                return False
        except Exception as e:
            log.warning(f"Signature verification error: {e}")
            return False
        log.info(f"  Attestation verified from {validator_hk[:16]}")
        return True

    def _set_weights(self, weights: dict[str, float]):
        """Set weights on chain from attestation data."""
        meta = self.subtensor.metagraph(netuid=self.netuid)
        _set_weights_from_map(self.subtensor, self.wallet, self.netuid, meta, weights)

    def run(self):
        """Poll R2 for new attestations, verify, set weights.

        Weights are also RE-set every ~360 blocks (WEIGHT_REFRESH_SECONDS) from
        the last verified bundle, so vtrust does not decay across the 24h gap
        between rounds."""
        log.info(f"AttestationValidator starting (poll every {self.poll_interval}s)")
        if not all([self.r2_endpoint, self.r2_access_key, self.r2_secret_key, self.r2_bucket]):
            log.error("R2 config incomplete — set R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET")
            return
        last_weights: dict[str, float] = {}
        last_set_ts = 0.0
        while True:
            try:
                bundle = self._fetch_latest_attestation()
                if not bundle:
                    log.info("  No attestations found in R2")
                elif bundle.get("round_id") == self.last_round_id:
                    log.debug("  No new round since last check")
                else:
                    round_id = bundle.get("round_id", "unknown")
                    log.info(f"  New attestation: {round_id}")
                    if self._verify_attestation(bundle):
                        weights = bundle.get("weights", {})
                        log.info(f"  Weights from attestation: {len(weights)} miners")
                        for hk, w in sorted(weights.items(), key=lambda x: -x[1]):
                            log.info(f"    {hk[:16]}: {w:.4f}")
                        if self.set_weights_enabled:
                            self._set_weights(weights)
                            last_weights, last_set_ts = dict(weights), time.time()
                        else:
                            log.info("  --set-weights not set; verified attestation but did NOT write to chain.")
                        self.last_round_id = round_id
                        self._save_last_round_id(round_id)
                    else:
                        log.warning(f"  Attestation verification failed for {round_id}")
                if (self.set_weights_enabled and last_weights
                        and time.time() - last_set_ts >= WEIGHT_REFRESH_SECONDS):
                    log.info("  Re-setting last verified weights (epoch refresh)")
                    self._set_weights(last_weights)
                    last_set_ts = time.time()
            except Exception as e:
                log.error(f"  Attestation poll error: {e}")
                traceback.print_exc()
            time.sleep(self.poll_interval)



def load_leaderboard(rounds_dir: str = None, ema_alpha: float = 0.3) -> list[dict]:
    """Aggregate round results into a ranked leaderboard."""
    rdir = Path(rounds_dir) if rounds_dir else Path(__file__).resolve().parent / "rounds"
    if not rdir.exists():
        return []

    rounds = []
    for f in sorted(rdir.glob("*.json")):
        try:
            rounds.append(json.loads(f.read_text()))
        except Exception:
            continue
    if not rounds:
        return []
    rounds.sort(key=lambda r: r.get("timestamp", 0))

    entries: dict[str, dict] = {}
    for rd in rounds:
        hashes = rd.get("code_hashes", {})
        results = rd.get("results", {})

        subs = rd.get("submissions", {})
        if not results and subs:
            results = {hk: s.get("score", s) for hk, s in subs.items()}

        if hashes:
            present_keys = set()
            for hk in results:
                present_keys.add(hashes.get(hk, hk))
            for key, e in entries.items():
                if key not in present_keys:
                    e["ema_score"] *= (1 - ema_alpha)

        for hk, res in results.items():
            key = hashes.get(hk, hk)
            score = res.get("final_score", 0.0)
            comps = res.get("components", {})

            if key not in entries:
                entries[key] = {
                    "code_hash": key, "hotkeys": set(), "ema_score": score,
                    "best_score": score, "rounds_seen": 0, "last_round": "",
                    "component_sums": {}, "component_counts": 0,
                }
            e = entries[key]
            e["hotkeys"].add(hk)
            e["ema_score"] = ema_alpha * score + (1 - ema_alpha) * e["ema_score"]
            e["best_score"] = max(e["best_score"], score)
            e["rounds_seen"] += 1
            e["last_round"] = rd.get("round_id", "")
            for k, v in comps.items():
                if isinstance(v, (int, float)):
                    e["component_sums"][k] = e["component_sums"].get(k, 0.0) + v
            e["component_counts"] += 1

    board = []
    for e in entries.values():
        n = max(e["component_counts"], 1)
        board.append({
            "code_hash": e["code_hash"],
            "hotkeys": sorted(e["hotkeys"]),
            "ema_score": e["ema_score"],
            "best_score": e["best_score"],
            "rounds_seen": e["rounds_seen"],
            "last_round": e["last_round"],
            "components_avg": {k: v / n for k, v in e["component_sums"].items()},
        })
    board.sort(key=lambda x: x["ema_score"], reverse=True)
    return board


def print_leaderboard(rounds_dir: str = None):
    """Print the current leaderboard to stdout."""
    board = load_leaderboard(rounds_dir)
    if not board:
        print("No round results found.")
        return
    print(f"\n{'='*72}")
    print(f"  SN125 OPTIMIZER LEADERBOARD  ({len(board)} entries)")
    print(f"{'='*72}")
    print(f"{'#':>3s} {'Hash':>10s} {'EMA':>8s} {'Best':>8s} {'Rounds':>6s} {'Hotkeys'}")
    print("-" * 72)
    for i, e in enumerate(board, 1):
        h = e["code_hash"][:10]
        hks = ", ".join(hk[:12] for hk in e["hotkeys"][:3])
        print(f"{i:3d} {h:>10s} {e['ema_score']:+8.4f} {e['best_score']:+8.4f} "
              f"{e['rounds_seen']:6d} {hks}")
    print()
    for i, e in enumerate(board[:5], 1):
        ca = e["components_avg"]
        if ca:
            parts = " ".join(f"{k}={v:+.3f}" for k, v in sorted(ca.items())
                             if k not in ("raw_combined",))
            print(f"  #{i} components: {parts}")
    print()


def export_optimizer(rank: int = 1, output: str = None, rounds_dir: str = None):
    """Export the Nth-ranked optimizer source code from round history.
    Returns the source code string, or None if not found."""
    rdir = Path(rounds_dir) if rounds_dir else Path(__file__).resolve().parent / "rounds"
    board = load_leaderboard(str(rdir) if rounds_dir else None)
    if not board or rank < 1 or rank > len(board):
        print(f"No optimizer at rank {rank} (leaderboard has {len(board)} entries)")
        return None

    target_hash = board[rank - 1]["code_hash"]
    for f in sorted(rdir.glob("*.json"), reverse=True):
        try:
            rd = json.loads(f.read_text())
        except Exception:
            continue
        sources = rd.get("sources", {})
        if target_hash in sources:
            src = sources[target_hash]
            if output:
                Path(output).write_text(src)
                print(f"Exported rank #{rank} optimizer ({target_hash[:10]}) → {output}")
            else:
                print(src)
            return src

    print(f"Source code for {target_hash[:10]} not found in round data "
          f"(older rounds may not have persisted sources)")
    return None
