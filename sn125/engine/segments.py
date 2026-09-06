"""Segmented execution + random re-verification (DESIGN.md §6.3).

A run executes as N chained segment-jobs.  Each job loads checkpoint k,
runs m harness steps, writes checkpoint k+1.  A segment checkpoint is ONE
safetensors file (validated byte-level by ``validate_safetensors_file``
before any parser touches it — same gate as §6.2 scoring artifacts) holding
EVERYTHING the next segment needs to be a pure function of the file:

    model/<key>           full model state_dict (params + buffers)
    opt/<NNNNN>           every tensor in the miner optimizer's state_dict
    rng/harness/torch     torch CPU RNG          (uint8 state tensor)
    rng/harness/cuda/<i>  torch CUDA RNG per device (when on CUDA)
    rng/miner/*           the miner PROCESS's RNG (§6.1 split optimizers)
    __metadata__          opt-state structure (JSON; tensors by reference),
                          python/numpy RNG (JSON), step bounds, train losses

Because execution is checkpoint→steps→checkpoint, **re-verification is
indistinguishable from normal execution**: ``verify_segment`` re-runs a
randomly chosen segment from checkpoint k with a fresh model + fresh miner
process and asserts the produced checkpoint matches the claimed checkpoint
k+1 bit-for-bit.  Any influence on the weights not reproducible from
(checkpoint, harness, miner step()) — hidden state outside state_dict(),
wall-clock/file/env dependence, engineered nondeterminism — surfaces as a
tensor mismatch ⇒ DQ.

Contract consequence (enforced here): the miner's ``state_dict()`` may
contain only tensors and JSON-able scalars (str/int/float/bool/None) in
dicts/lists/tuples.  Anything else cannot be checkpointed safely (pickle is
banned) and raises SegmentError — a gate, not an accident.

The step math is an exact mirror of the inner loop of
``training.train_and_eval`` (clip → miner step → external WSD scale →
validate → ``p.add_``), device-agnostic so a toy CPU model proves the loop.
Wall-clock / memory caps are launcher-side per Q9_RESULTS.md and are
deliberately absent here — bit-exactness must not depend on timing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..training import wsd_schedule_scale
from .score_worker import (
    MAX_CHECKPOINT_BYTES, CheckpointRejected, validate_safetensors_file,
)

FORMAT_VERSION = "sn125-segment-v1"


def content_digest(path: str) -> str:
    """Canonical sha256 of (metadata, tensors) — THE bit-exact comparator.

    safetensors does not guarantee header key order (HashMap), so identical
    content can produce different file bytes; a plain file hash would flag
    honest runs.  This digest is order-independent: sorted tensor names,
    each hashed as (name, dtype, shape, raw little-endian bytes), plus the
    canonical-JSON metadata.  The file is byte-validated before parsing.
    """
    info = validate_safetensors_file(path)
    from safetensors.torch import load_file
    tensors = load_file(path)
    h = hashlib.sha256()
    h.update(json.dumps(info["metadata"], sort_keys=True).encode())
    for name in sorted(tensors):
        t = tensors[name].contiguous().flatten()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(json.dumps(list(tensors[name].shape)).encode())
        h.update(bytes(t.view(torch.uint8).numpy()) if t.numel() else b"")
    return h.hexdigest()


class SegmentError(RuntimeError):
    """A segment could not be executed/checkpointed faithfully (DQ-able)."""


# ── miner opt-state encoding: tensors out-of-band, structure as JSON ─────────

def _encode_state(obj, sink: dict[str, torch.Tensor], counter: list[int]):
    if torch.is_tensor(obj):
        name = f"opt/{counter[0]:05d}"
        counter[0] += 1
        sink[name] = obj.detach().cpu().contiguous().clone()
        return {"__t__": name}
    if isinstance(obj, dict):
        pairs = []
        for k, v in obj.items():
            if not (isinstance(k, (str, int, float, bool)) or k is None):
                raise SegmentError(f"opt state dict key {k!r}: keys must be scalars")
            pairs.append([_encode_state(k, sink, counter), _encode_state(v, sink, counter)])
        return {"__d__": pairs}
    if isinstance(obj, tuple):
        return {"__u__": [_encode_state(v, sink, counter) for v in obj]}
    if isinstance(obj, list):
        return {"__l__": [_encode_state(v, sink, counter) for v in obj]}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise SegmentError(
        f"opt state contains {type(obj).__name__}: only tensors and JSON-able "
        "scalars are checkpointable (pickle is banned)")


def _decode_state(node, tensors: dict[str, torch.Tensor], device, used: set[str]):
    if isinstance(node, dict):
        if "__t__" in node:
            name = node["__t__"]
            if name not in tensors:
                raise CheckpointRejected(f"opt structure references missing tensor {name!r}")
            used.add(name)
            return tensors[name].to(device)
        if "__d__" in node:
            return {_decode_state(k, tensors, device, used): _decode_state(v, tensors, device, used)
                    for k, v in node["__d__"]}
        if "__u__" in node:
            return tuple(_decode_state(v, tensors, device, used) for v in node["__u__"])
        if "__l__" in node:
            return [_decode_state(v, tensors, device, used) for v in node["__l__"]]
        raise CheckpointRejected(f"malformed opt structure node: {list(node)!r}")
    return node



def _capture_rng(opt, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors = {"rng/harness/torch": torch.get_rng_state()}
    if device.type == "cuda":
        for i, s in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"rng/harness/cuda/{i}"] = s
    if hasattr(opt, "get_rng_state"):
        miner = opt.get_rng_state()
        tensors["rng/miner/torch"] = miner["torch"]
        for i, s in enumerate(miner.get("cuda", [])):
            tensors[f"rng/miner/cuda/{i}"] = s
    state = random.getstate()
    meta = {"py_random": json.dumps([state[0], list(state[1]), state[2]])}
    try:
        import numpy as np
        ns = np.random.get_state()
        meta["np_random"] = json.dumps([ns[0], [int(x) for x in ns[1]], int(ns[2]),
                                        int(ns[3]), float(ns[4])])
    except ImportError:
        meta["np_random"] = ""
    return tensors, meta


def _restore_rng(opt, tensors: dict[str, torch.Tensor], meta: dict[str, str],
                 device: torch.device):
    torch.set_rng_state(tensors["rng/harness/torch"].to(torch.uint8).cpu())
    if device.type == "cuda":
        cuda_states = []
        i = 0
        while f"rng/harness/cuda/{i}" in tensors:
            cuda_states.append(tensors[f"rng/harness/cuda/{i}"].to(torch.uint8).cpu())
            i += 1
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    if hasattr(opt, "set_rng_state") and "rng/miner/torch" in tensors:
        miner: dict = {"torch": tensors["rng/miner/torch"].to(torch.uint8).cpu()}
        cuda_states = []
        i = 0
        while f"rng/miner/cuda/{i}" in tensors:
            cuda_states.append(tensors[f"rng/miner/cuda/{i}"].to(torch.uint8).cpu())
            i += 1
        if cuda_states:
            miner["cuda"] = cuda_states
        opt.set_rng_state(miner)
    s = json.loads(meta["py_random"])
    random.setstate((s[0], tuple(s[1]), s[2]))
    if meta.get("np_random"):
        import numpy as np
        ns = json.loads(meta["np_random"])
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))



def save_segment_checkpoint(path: str, model: nn.Module, opt, step: int,
                            train_losses: list[float] | None = None,
                            start_step: int | None = None) -> str:
    """Write the full segment state as one safetensors file; returns sha256.

    Returns ``content_digest(path)`` — identical state always yields an
    identical digest (the file bytes themselves may differ because
    safetensors header order is unspecified).
    """
    tensors: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        tensors[f"model/{k}"] = v.detach().cpu().contiguous().clone()

    counter = [0]
    structure = _encode_state(opt.state_dict(), tensors, counter)

    device = next(model.parameters()).device
    rng_tensors, rng_meta = _capture_rng(opt, device)
    tensors.update(rng_tensors)

    meta = {
        "format": FORMAT_VERSION,
        "step": str(step),
        "start_step": str(step if start_step is None else start_step),
        "opt_structure": json.dumps(structure),
        "train_losses": json.dumps(train_losses or []),
        **rng_meta,
    }
    from safetensors.torch import save_file
    save_file({k: tensors[k] for k in sorted(tensors)}, path, metadata=meta)
    return content_digest(path)


def load_segment_checkpoint(path: str, model: nn.Module, opt,
                            device: str = "cpu",
                            max_bytes: int = MAX_CHECKPOINT_BYTES) -> int:
    """Validate (raw bytes first), then restore model + opt + ALL RNG.

    Rejects: any model/* inventory differing from the freshly built model,
    any opt/* tensor not referenced by the structure (a smuggling channel),
    any tensor name outside the three namespaces.  Returns the step the
    checkpoint represents (the next step to execute).
    """
    info = validate_safetensors_file(path, max_bytes=max_bytes)
    meta = info["metadata"]
    if meta.get("format") != FORMAT_VERSION:
        raise CheckpointRejected(f"not a segment checkpoint (format={meta.get('format')!r})")

    expected_model = {f"model/{k}" for k in model.state_dict()}
    got_model = {n for n in info["tensors"] if n.startswith("model/")}
    if got_model != expected_model:
        raise CheckpointRejected(
            f"model tensor inventory mismatch: missing={sorted(expected_model - got_model)} "
            f"extra={sorted(got_model - expected_model)}")
    for n in info["tensors"]:
        if not (n.startswith("model/") or n.startswith("opt/") or n.startswith("rng/")):
            raise CheckpointRejected(f"tensor {n!r} outside checkpoint namespaces")

    from safetensors.torch import load_file
    tensors = load_file(path)

    model.load_state_dict(
        {k[len("model/"):]: v for k, v in tensors.items() if k.startswith("model/")},
        strict=True)

    try:
        structure = json.loads(meta["opt_structure"])
    except (KeyError, ValueError) as e:
        raise CheckpointRejected(f"bad opt_structure metadata: {e}") from e
    used: set[str] = set()
    opt_state = _decode_state(structure, tensors, device, used)
    unreferenced = {n for n in tensors if n.startswith("opt/")} - used
    if unreferenced:
        raise CheckpointRejected(f"unreferenced opt tensors {sorted(unreferenced)} (smuggling)")
    opt.load_state_dict(opt_state)

    _restore_rng(opt, tensors, meta, torch.device(device))
    try:
        return int(meta["step"])
    except (KeyError, ValueError) as e:
        raise CheckpointRejected(f"bad step metadata: {e}") from e



@dataclass
class SegmentSpec:
    """Harness hyperparameters — identical for every segment of a run.

    `total_steps` is the WSD schedule denominator (in budget mode the per-box,
    probe-derived, signature-pinned ``schedule_total_steps``). `executed_steps` is
    how many steps the live run actually performed before the wall-clock budget
    stopped it — the replay LOOP BOUND. They differ only in budget mode where a
    fast box may run past `total_steps` (those tail steps apply WSD scale 0 to the
    weights but still advance optimizer state, so replay must reproduce them).
    `executed_steps == 0` ⇒ use `total_steps` (legacy step-mode, unchanged)."""
    lr: float
    weight_decay: float
    total_steps: int
    warmup_steps: int
    max_grad_norm: float = 1.0
    decay_fraction: float = 0.0
    use_amp: bool = False
    executed_steps: int = 0

    @property
    def loop_steps(self) -> int:
        """The number of steps to (re-)execute; the schedule still keys on total_steps."""
        return self.executed_steps if self.executed_steps > 0 else self.total_steps


@dataclass
class SegmentResult:
    start_step: int
    end_step: int
    checkpoint_path: str
    checkpoint_digest: str
    train_losses: list[float] = field(default_factory=list)


def run_segment(model: nn.Module, opt, train_data: list[torch.Tensor],
                spec: SegmentSpec, n_steps: int, ckpt_out: str,
                ckpt_in: str | None = None, device: str = "cpu") -> SegmentResult:
    """Execute one segment: load checkpoint, run n harness steps, checkpoint.

    With ``ckpt_in=None`` this is segment 0 from the freshly initialized
    (model, opt) — same code path either way, so verification of any segment
    is indistinguishable from normal execution.
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False

    start_step = 0
    if ckpt_in is not None:
        start_step = load_segment_checkpoint(ckpt_in, model, opt, device=device)
    end_step = min(start_step + n_steps, spec.loop_steps)
    if not train_data:
        raise SegmentError("no training data")

    dev = torch.device(device)
    losses: list[float] = []
    model.train()
    for step in range(start_step, end_step):
        batch = train_data[step % len(train_data)].to(dev, non_blocking=True)
        amp = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if spec.use_amp and dev.type == "cuda" else nullcontext())
        with amp:
            out = model(batch, labels=batch)
        loss_val = out.loss.float().item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise SegmentError(f"loss diverged (NaN/Inf) at step {step}")
        losses.append(loss_val)
        out.loss.backward()
        del batch, out

        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.max_grad_norm)
        gradients, param_values = {}, {}
        for name, p in model.named_parameters():
            gradients[name] = (p.grad.detach().clone() if p.grad is not None
                               else torch.zeros_like(p))
            param_values[name] = p.detach().clone()
        model.zero_grad()

        updates = opt.step(gradients, param_values, step)
        scale = wsd_schedule_scale(step, spec.total_steps, spec.warmup_steps,
                                   spec.decay_fraction)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in updates:
                    u = updates[name]
                    if type(u) is not torch.Tensor:
                        u = torch.as_tensor(u)
                    if u.device != p.device:
                        raise SegmentError(f"device mismatch for {name} at step {step}")
                    if u.shape != p.shape:
                        raise SegmentError(
                            f"shape mismatch for {name}: {tuple(u.shape)} != "
                            f"{tuple(p.shape)} at step {step}")
                    if not torch.isfinite(u).all():
                        raise SegmentError(f"NaN/Inf in update for {name} at step {step}")
                    p.add_(u * scale)

    sha = save_segment_checkpoint(ckpt_out, model, opt, end_step,
                                  train_losses=losses, start_step=start_step)
    return SegmentResult(start_step, end_step, ckpt_out, sha, losses)


def run_segmented(model_factory, opt_factory, train_data: list[torch.Tensor],
                  spec: SegmentSpec, segment_len: int, ckpt_dir: str,
                  device: str = "cpu") -> list[SegmentResult]:
    """The §6.3 main path: the whole run as chained fresh-process-shaped
    segment jobs.  Every segment gets a FRESH model + miner optimizer and
    knows the previous state only through the checkpoint file — exactly the
    conditions a re-verifier sees, so hidden state cannot help even the
    honest-looking run.  Returns one SegmentResult per segment, including
    the step-0 initialization checkpoint (segment_000 = init, no steps).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    results: list[SegmentResult] = []

    model, opt = model_factory(), opt_factory()
    try:
        init_path = os.path.join(ckpt_dir, "segment_000.safetensors")
        sha = save_segment_checkpoint(init_path, model, opt, 0)
        results.append(SegmentResult(0, 0, init_path, sha))
    finally:
        if hasattr(opt, "close"):
            opt.close()

    k, step = 0, 0
    while step < spec.loop_steps:
        k += 1
        model, opt = model_factory(), opt_factory()
        try:
            out_path = os.path.join(ckpt_dir, f"segment_{k:03d}.safetensors")
            res = run_segment(model, opt, train_data, spec, segment_len,
                              ckpt_out=out_path, ckpt_in=results[-1].checkpoint_path,
                              device=device)
        finally:
            if hasattr(opt, "close"):
                opt.close()
        results.append(res)
        step = res.end_step
    return results



@dataclass
class VerificationReport:
    match: bool
    start_step: int
    end_step: int
    claimed_digest: str
    recomputed_digest: str
    mismatches: list[str] = field(default_factory=list)


def compare_checkpoints(claimed_path: str, recomputed_path: str) -> list[str]:
    """Bit-exact comparison; returns the list of differing tensors/metadata.

    Tensor-level (not just file-hash) so a DQ report can name exactly what
    diverged.  Both files are byte-validated before parsing.
    """
    a = validate_safetensors_file(claimed_path)
    b = validate_safetensors_file(recomputed_path)
    mismatches: list[str] = []
    for key in sorted(set(a["metadata"]) | set(b["metadata"])):
        if a["metadata"].get(key) != b["metadata"].get(key):
            mismatches.append(f"metadata:{key}")
    from safetensors.torch import load_file
    ta, tb = load_file(claimed_path), load_file(recomputed_path)
    for name in sorted(set(ta) | set(tb)):
        if name not in ta or name not in tb:
            mismatches.append(f"tensor-missing:{name}")
        elif ta[name].dtype != tb[name].dtype or not torch.equal(ta[name], tb[name]):
            mismatches.append(f"tensor:{name}")
    return mismatches


def verify_segment(model_factory, opt_factory, train_data: list[torch.Tensor],
                   spec: SegmentSpec, ckpt_in: str, claimed_ckpt_out: str,
                   n_steps: int, device: str = "cpu") -> VerificationReport:
    """Re-execute one segment from checkpoint k with a fresh model + fresh
    miner optimizer (process); assert the result matches the claimed
    checkpoint k+1 bit-for-bit.  Any mismatch is evidence, not noise: the
    main run executes under identical conditions (run_segmented), so an
    honest segment reproduces exactly and a dishonest one cannot tell it is
    being audited.
    """
    model, opt = model_factory(), opt_factory()
    try:
        with tempfile.TemporaryDirectory(prefix="sn125_verify_") as td:
            recomputed = os.path.join(td, "recomputed.safetensors")
            res = run_segment(model, opt, train_data, spec, n_steps,
                              ckpt_out=recomputed, ckpt_in=ckpt_in, device=device)
            mismatches = compare_checkpoints(claimed_ckpt_out, recomputed)
            return VerificationReport(
                match=not mismatches,
                start_step=res.start_step,
                end_step=res.end_step,
                claimed_digest=content_digest(claimed_ckpt_out),
                recomputed_digest=res.checkpoint_digest,
                mismatches=mismatches,
            )
    finally:
        if hasattr(opt, "close"):
            opt.close()


def pick_audit_segments(results: list[SegmentResult], n_audits: int,
                        seed_material: bytes) -> list[int]:
    """Deterministically choose which segments to re-verify from public seed
    material (e.g. a block hash revealed AFTER the run completed) — the
    chooser must be unpredictable to the miner at run time but auditable
    after the fact.  Returns indices into ``results`` (skipping segment 0,
    which has no execution to verify)."""
    runnable = list(range(1, len(results)))
    rng = random.Random(hashlib.sha256(seed_material).digest())
    rng.shuffle(runnable)
    return sorted(runnable[:max(0, n_audits)])
