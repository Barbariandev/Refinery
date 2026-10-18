"""CLI-driven local end-to-end proof of the single blessed FSM validator
entrypoint (P2.2).

The unit suite (``test_live.py``) already drives ``run_fsm_validator`` over a
``MockChain`` with a *fully fake* validator. This module is the heavier proof the
operator asked for: a **real invoked process** (``python -m sn125 validate
--mock``) that runs the WHOLE pipeline against a local mock chain using the
**real** :class:`~sn125.neuron.Validator` methods — ``compute_weights`` (configured burn
floor), ``_save_round``, ``_build_attestation`` (real ed25519 signing →
tamper-evident), and the real dashboard read path — faking ONLY the three
external surfaces a chain would provide: the metagraph, the dendrite transport,
and the GPU cloud orchestrator.

It NEVER touches a real chain: ``_set_weights_on_chain`` is redirected to a mock
ledger, so no ``set_weights`` call leaves this box. Eval is a canned/mock result
(no Targon spend). The flow proven, in order:

    sync metagraph → pin fee → credit (observed mock transfers) → commit/reveal
    transport → source gate → concurrent eval → score vs rolling-best →
    compute_weights (configured burn floor; 100% when no eligible miner) →
    set-weights AGAINST THE MOCK CHAIN → signed attestation → dashboard JSON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bittensor as bt

from ..payments.registry import MockChain, PaymentRegistry, RAO_PER_TAO
from .live import run_fsm_validator

log = logging.getLogger("sn125.dryrun")

# A trivially valid AdamW optimizer (passes the source gate) and a banned one
# (imports os, no build_optimizer → gate-DQ).
_GOOD = ("import torch\n"
         "def build_optimizer(params, lr=1e-3):\n"
         "    return torch.optim.AdamW(params, lr=lr)\n")
_BAD = ("import os\n"
        "def build_optimizer(params):\n"
        "    return None\n")


def _good(tag: str) -> str:
    return _GOOD + f"# {tag}\n"


# ── fakes for the three external surfaces (metagraph / dendrite / cloud) ──────
class _Axon:
    def __init__(self, hotkey, ip="1.2.3.4"):
        self.hotkey = hotkey
        self.ip = ip


class _Meta:
    def __init__(self, hotkeys):
        self.neurons = [SimpleNamespace(axon_info=_Axon(hk)) for hk in hotkeys]
        self.n = len(self.neurons)


class _Commit:
    def __init__(self, round_id):
        self.round_id = round_id


class _Submission:
    def __init__(self, round_id):
        self.round_id = round_id


class _Resp:
    def __init__(self, code_hash="", source_code=""):
        self.code_hash = code_hash
        self.source_code = source_code


class _Dendrite:
    """Returns each miner's commit hash / source preimage by hotkey + synapse."""
    def __init__(self, sources):
        self.sources = sources

    def query(self, ai, syn, timeout=None, deserialize=False):
        src = self.sources.get(ai.hotkey, "")
        if isinstance(syn, _Commit):
            return _Resp(code_hash=hashlib.sha256(src.encode()).hexdigest())
        return _Resp(source_code=src)


class _Cloud:
    """Mock GPU orchestrator: evaluate_submission returns a canned result keyed by
    the (unique) source string. No Targon, no GPU — keeps the dry-run $0."""
    def __init__(self, results_by_src):
        self.results_by_src = results_by_src
        self.calls = []

    def evaluate_submission(self, source, round_id, sub_uid, mode="prod"):
        self.calls.append(source)
        return self.results_by_src[source]


def _make_wallet():
    """Real ed25519 keypair wrapped as a minimal wallet so the attestation bundle
    is genuinely SIGNED (and therefore verifiable / tamper-evident), exactly as in
    production — without creating on-disk wallet files."""
    kp = bt.Keypair.create_from_uri("//Sn125DryRunValidator")
    return SimpleNamespace(hotkey=kp)


class DryRunValidator:
    """Wraps the REAL :class:`Validator` and overrides only the three external
    touch-points. Everything reward-bearing (compute_weights / _save_round /
    _build_attestation / _build_results) is the real production code."""

    def __init__(self, meta, dendrite, cloud, reg, *, set_weights=True,
                 burn_fraction_floor=None, fee=2 * RAO_PER_TAO):
        from ..neuron import Validator
        from ..training import TaskSpec  # a lightweight synthetic task for serialization
        try:
            task = TaskSpec(task_id="dryrun-360M", model_config="SmolLM2-360M",
                            parameter_count="360M", total_steps=1000, batch_size=8,
                            sequence_length=2048, eval_every=100, eval_sequences=64,
                            warmup_fraction=0.05, compute_budget_seconds=72000,
                            task_weight=1.0, use_pretrained=True)
            tasks = [task]
        except Exception:
            tasks = None  # fall back to defaults if TaskSpec signature differs

        # Construct with wallet=None/set_weights=False so the real __init__ never
        # builds a real Subtensor/Dendrite against a chain; we wire them by hand.
        self._real = Validator(wallet=None, netuid=125, network="mock",
                               tasks=tasks, set_weights=False, backend="local",
                               audit_dir=tempfile.mkdtemp(prefix="sn125_audit_dryrun_"),
                               burn_fraction_floor=burn_fraction_floor)
        self._real.wallet = _make_wallet()  # real keypair → real attestation signing
        # Inject the three fakes.
        self._real.dendrite = dendrite
        self._real._cloud_orch = cloud
        self._meta = meta
        # FSM wiring (same fields the production __main__ path sets).
        self._real.payment_registry = reg
        self._real.treasury_coldkey = "treasury"
        self._real.round_fee_rao = fee
        self._real.commit_synapse_cls = _Commit
        self._real.submission_synapse_cls = _Submission
        self._real.set_weights_enabled = set_weights
        self._real.fsm_commit_window_s = 10
        self._real.fsm_reveal_window_s = 10
        self._real.fsm_round_period_s = 100
        # Synthetic clock (t = 1000 s): no real-world round grid or Challenge-2
        # start instant, or every dry-run round would sit before the launch.
        self._real.round_grid_anchor = 0
        self._real.refiner_start_timestamp = 0
        # records for the proof summary
        self.weight_set_calls = []
        self.mock_chain_weights = {}

    # The real Validator method we DON'T want hitting a chain: redirect to a mock
    # ledger so weights (incl. configured floor→100% burn) are demonstrably emitted, but
    # NO set_weights call ever leaves this box.
    def sync_metagraph(self):
        return self._meta

    def _set_weights_on_chain(self, meta, weights):
        self.weight_set_calls.append(dict(weights))
        self.mock_chain_weights = dict(weights)
        log.info("  [MOCK-CHAIN] set_weights(%d entries): %s", len(weights),
                 {k[:10]: round(v, 4) for k, v in weights.items()})

    def _publish_attestation(self, *args, **kwargs):
        # Dry-run has no remote R2 durability target. The proof summary builds
        # and verifies the signed attestation locally, so this mock publication
        # is allowed to unblock the mock-chain weight-set path only.
        return True

    # Everything else proxies to the real Validator (compute_weights, _save_round,
    # _build_results helpers it calls internally, _publish_attestation, attrs).
    def __getattr__(self, name):
        return getattr(self._real, name)


def _build_scenario():
    """A single round exercising all four economic outcomes plus a genuine earner:
    one SCORED earner (g=1.0 → receives the configured payable pool), one infra-DQ
    (credit refunded), one miner-fault DQ (burned), one gate-DQ (banned source)."""
    chain = MockChain(finality_depth=2)
    reg = PaymentRegistry(chain)
    fee_credits = 1
    miners = [
        # hotkey, coldkey, source, fund?, cloud result (None ⇒ gate-DQ pre-eval)
        # Production-shaped score: the reward path reads the recorded final
        # losses (nats) — a 0.10-nat win over the genesis-like baseline.
        ("hk-earn", "ck-earn", _good("earn"), True,
         {"failed": False, "score": 0.5,
          "components": {"convergence": 0.036, "cross_scale": 0.036},
          "task_scores": {"dryrun-360M": {"sub_final_loss": 2.6764,
                                          "base_final_loss": 2.7764,
                                          "convergence": 0.036}}}),
        # A box ran and our side failed it (wall-clock budget breach on the
        # harness side): terminal infra-DQ, credit refunded. (A *no-box* failure
        # such as rental_provisioning_failed is not terminal any more: it is
        # retried inside the launch window and then carried to the next round.)
        ("hk-infra", "ck-infra", _good("infra"), True,
         {"failed": True, "error": "timeout_exceeded: eval budget"}),
        ("hk-fault", "ck-fault", _good("fault"), True,
         {"failed": True, "error": "bench failed (rc=1)"}),
        ("hk-gate", "ck-gate", _BAD, True, None),
    ]
    sources, results_by_src = {}, {}
    for hk, ck, src, fund, res in miners:
        chain.register_hotkey(hk, ck)
        sources[hk] = src
        if fund:
            chain.send_to_treasury(ck, (2 * RAO_PER_TAO) * fee_credits)
        if res is not None:
            results_by_src[src] = res
    chain.advance_blocks(3)  # finalize funding transfers
    meta = _Meta([hk for hk, *_ in miners])
    return chain, reg, meta, _Dendrite(sources), _Cloud(results_by_src)


def run_dry_run(*, rounds: int = 2, verbose: bool = True) -> dict:
    """Drive the real FSM entrypoint end-to-end against the mock chain and return a
    proof summary. Cleans up any round JSON it writes to ``sn125/rounds`` so the
    real dashboard read path is exercised without leaving dry-run artifacts.

    Two rounds are the minimum faithful run: round 1 scores the earner, which
    clears the bar and becomes PENDING (100% burn); round 2 runs the
    validator-funded confirmation, creates the frontier event and pays the
    configured payable pool (``FRONTIER_CONFIRMATION_RUNS``)."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    chain, reg, meta, dendrite, cloud = _build_scenario()
    v = DryRunValidator(meta, dendrite, cloud, reg, set_weights=True)

    from .. import neuron as _neuron_mod
    rounds_dir = Path(_neuron_mod.__file__).resolve().parent / "rounds"
    before = set(rounds_dir.glob("*.json")) if rounds_dir.exists() else set()

    # Deterministic injected clock so the 24h windows elapse instantly.
    t = [1000.0]
    clock = lambda: t[0]
    sleep = lambda dt: t.__setitem__(0, t[0] + max(0.0, dt))

    new_files: list[Path] = []
    try:
        run_fsm_validator(v, max_rounds=rounds, clock=clock, sleep=sleep)
        after = set(rounds_dir.glob("*.json")) if rounds_dir.exists() else set()
        new_files = sorted(after - before)

        # Real signed attestation over the scored round (tamper-evidence proof).
        results, sources, curve_data = _last_round_results(v, cloud)
        weights = v.mock_chain_weights
        att = None
        if results:
            att = v._real._build_attestation("dryrun_attest", results, weights,
                                             sources, curve_data)

        # Real dashboard read path over the just-written round file(s).
        from ..dashboard.data import build_dashboard_json
        dash = json.loads(build_dashboard_json(rounds_dir,
                                               validator_hotkey="dryrun-local"))
        dryrun_ids = {p.stem for p in new_files}
        history = dash.get("history", []) or []
        dryrun_entries = sorted((h for h in history if h.get("round_id") in dryrun_ids),
                                key=lambda h: str(h.get("round_id", "")))
        dryrun_entry = dryrun_entries[-1] if dryrun_entries else None
        frontier = getattr(v._real, "_last_frontier_rewards", {}) or {}

        summary = {
            "rounds": rounds,
            "cloud_evals": list(cloud.calls),
            "weights_on_mock_chain": v.mock_chain_weights,
            "weights_per_round": list(v.weight_set_calls),
            "frontier_event": frontier.get("new_event"),
            "pending_confirmation": frontier.get("pending_confirmation", []),
            "confirmation_outcomes": frontier.get("confirmation_outcomes", []),
            "burn_weight": v.mock_chain_weights.get("__BURN__", 0.0),
            "scored_miners": [k for k in v.mock_chain_weights if k != "__BURN__"],
            "balances": {ck: reg.balance(ck) for ck in
                         ("ck-earn", "ck-infra", "ck-fault", "ck-gate")},
            "round_files_written": [p.name for p in new_files],
            "attestation_bundle_hash": att.get("bundle_hash") if att else None,
            "attestation_signed": bool(att and att.get("signature")) if att else False,
            "dashboard_history_rounds": len(history),
            "dashboard_leaderboard": len(dash.get("leaderboard", []) or []),
            "dashboard_dryrun_round_present": dryrun_entry is not None,
            "dashboard_dryrun_burned_weight": (dryrun_entry or {}).get("burned_weight", 0.0),
        }
        if att:
            summary["attestation_verifies"] = _verify_attestation(att)
        return summary
    finally:
        # Clean up dry-run round artifacts (keep sn125/rounds pristine).
        for p in new_files:
            try:
                p.unlink()
            except OSError:
                pass


def _last_round_results(v, cloud):
    """Rebuild the (results, sources, curve_data) of the just-driven round from the
    real ScoreRecord path, so the attestation bundle is over real records."""
    from ..training import ScoreRecord
    results, sources, curve_data = {}, {}, {}
    # The scored earner is the only source the cloud returned a non-failed result for.
    for src, res in cloud.results_by_src.items():
        if not res.get("failed"):
            hk = "hk-earn"
            results[hk] = ScoreRecord(res.get("score", 0.0),
                                      res.get("components", {}) or {}, {}, {})
            sources[hk] = src
    return results, sources, curve_data


def _verify_attestation(att: dict) -> bool:
    """Re-verify the ed25519 signature over the signing payload — proves the bundle
    is tamper-evident (any field flip breaks the signature)."""
    from ..neuron import _signing_payload
    sig = att.get("signature")
    hk = att.get("validator_hotkey")
    if not (sig and hk):
        return False
    try:
        kp = bt.Keypair(ss58_address=hk)
        return bool(kp.verify(_signing_payload(att), bytes.fromhex(sig)))
    except Exception:
        return False
