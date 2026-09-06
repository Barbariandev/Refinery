"""Scoring split (DESIGN.md §6.2) — the score is computed where miner code never ran.

    BOX A (hostile)                          BOX B (clean)
    trainer + miner process (§6.1)           this module, fresh process
    exports ONLY a safetensors checkpoint ─► validates file BEFORE parsing,
    NO eval shards on disk                   loads checkpoint into a freshly
                                             built model, evaluates on held-out
                                             shards, emits the metric

The only artifact that crosses the boundary is a **safetensors** file — never
pickle.  ``validate_safetensors_file`` checks the raw bytes (header length,
JSON header, dtype allowlist, exact offset tiling, expected tensor shapes)
before any parser library touches the file, so a malicious "checkpoint" is
rejected without executing anything.  Held-out eval shards travel the same
format (token tensors), mounted only on the clean box.

Eval math is an exact mirror of the harness eval loop in
``training.train_and_eval`` (mean over batches of ``out.loss.float().item()``,
batches moved to device one at a time) so the clean-box score is bit-identical
to what an honest trainer would have reported — any difference is a C3 lie.

Process isolation: ``score_checkpoint_isolated`` runs the scorer as a fresh
``python -m sn125.engine.score_worker`` subprocess.  The worker is *trusted*
code; the fresh process guarantees no in-memory state from the training box
(monkey-patched torch, poisoned caches) can influence the score.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import struct
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

MAX_CHECKPOINT_BYTES = 16 * 1024**3
MAX_SHARD_BYTES = 2 * 1024**3
_MAX_HEADER_BYTES = 100 * 1024**2

_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
_TORCH_DTYPE_NAME = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
}


class CheckpointRejected(ValueError):
    """Artifact failed structural validation — never parsed, never loaded."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_safetensors_file(
    path: str,
    max_bytes: int = MAX_CHECKPOINT_BYTES,
    expected_tensors: dict[str, tuple[tuple[int, ...], str]] | None = None,
) -> dict:
    """Validate a safetensors file structurally BEFORE any parser touches it.

    Checks, in order, on the raw bytes:
      1. size within [9, max_bytes]
      2. u64 header length sane; header is a JSON object (first byte ``{`` —
         this alone rejects every pickle, which starts ``\\x80``)
      3. every tensor entry: allowlisted dtype, non-negative int shape,
         byte length == prod(shape) * itemsize
      4. data offsets tile [0, data_size] EXACTLY — no gaps, no overlaps,
         no smuggled trailing bytes
      5. ``__metadata__`` values are strings
      6. if ``expected_tensors`` given (name -> (shape, dtype-name)): every
         tensor in the file must match it exactly; no extras allowed

    Returns ``{"tensors": {name: (shape, dtype)}, "metadata": {...},
    "file_bytes": int}``.  Raises CheckpointRejected on any failure.
    """
    if not os.path.isfile(path):
        raise CheckpointRejected(f"not a file: {path}")
    file_bytes = os.path.getsize(path)
    if file_bytes > max_bytes:
        raise CheckpointRejected(f"file is {file_bytes} bytes > cap {max_bytes}")
    if file_bytes < 9:
        raise CheckpointRejected(f"file too small to be safetensors ({file_bytes} bytes)")

    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len < 2 or header_len > min(_MAX_HEADER_BYTES, file_bytes - 8):
            raise CheckpointRejected(f"implausible header length {header_len}")
        raw = f.read(header_len)

    if not raw.lstrip(b" ").startswith(b"{"):
        raise CheckpointRejected("header is not a JSON object (pickle or garbage)")
    try:
        header = json.loads(raw)
    except Exception as e:
        raise CheckpointRejected(f"header is not valid JSON: {e}") from e
    if not isinstance(header, dict):
        raise CheckpointRejected("header JSON is not an object")

    metadata = header.pop("__metadata__", {})
    if not (isinstance(metadata, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())):
        raise CheckpointRejected("__metadata__ must be a str->str object")

    data_size = file_bytes - 8 - header_len
    spans: list[tuple[int, int]] = []
    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, entry in header.items():
        if not (isinstance(entry, dict)
                and set(entry) == {"dtype", "shape", "data_offsets"}):
            raise CheckpointRejected(f"malformed entry for {name!r}")
        dtype = entry["dtype"]
        if dtype not in _DTYPE_BYTES:
            raise CheckpointRejected(f"{name!r}: dtype {dtype!r} not allowlisted")
        shape = entry["shape"]
        if not (isinstance(shape, list)
                and all(isinstance(d, int) and d >= 0 for d in shape)):
            raise CheckpointRejected(f"{name!r}: bad shape {shape!r}")
        offs = entry["data_offsets"]
        if not (isinstance(offs, list) and len(offs) == 2
                and all(isinstance(o, int) for o in offs)
                and 0 <= offs[0] <= offs[1] <= data_size):
            raise CheckpointRejected(f"{name!r}: bad data_offsets {offs!r}")
        if offs[1] - offs[0] != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise CheckpointRejected(f"{name!r}: byte length does not match shape*dtype")
        spans.append((offs[0], offs[1]))
        tensors[name] = (tuple(shape), dtype)

    spans.sort()
    cursor = 0
    for b, e in spans:
        if b != cursor:
            raise CheckpointRejected(
                f"data region has a {'gap' if b > cursor else 'overlap'} at byte {cursor}")
        cursor = e
    if cursor != data_size:
        raise CheckpointRejected(
            f"{data_size - cursor} smuggled bytes after last tensor")

    if expected_tensors is not None:
        for name, (shape, dtype) in tensors.items():
            exp = expected_tensors.get(name)
            if exp is None:
                raise CheckpointRejected(f"unexpected tensor {name!r}")
            if (tuple(exp[0]), exp[1]) != (shape, dtype):
                raise CheckpointRejected(
                    f"{name!r}: got {shape}/{dtype}, expected {tuple(exp[0])}/{exp[1]}")

    return {"tensors": tensors, "metadata": metadata, "file_bytes": file_bytes}


def expected_tensors_of(model: nn.Module) -> dict[str, tuple[tuple[int, ...], str]]:
    """name -> (shape, safetensors dtype) for everything in model.state_dict()."""
    return {name: (tuple(t.shape), _TORCH_DTYPE_NAME[t.dtype])
            for name, t in model.state_dict().items()}



def export_checkpoint(model: nn.Module, path: str,
                      metadata: dict[str, str] | None = None) -> str:
    """Write the model's state as safetensors (tied weights handled).

    The trainer NEVER writes pickle.  Returns the file's sha256 so the
    trainer can commit to the artifact it shipped.
    """
    from safetensors.torch import save_model
    save_model(model, path, metadata=dict(metadata or {}))
    return _sha256(path)


def export_eval_shard(batches: list[torch.Tensor], path: str,
                      metadata: dict[str, str] | None = None) -> str:
    """Write held-out token batches as safetensors (mounted ONLY on Box B).

    Returns a CONTENT hash (sorted tensor names -> raw bytes), not a file hash:
    safetensors serializes the __metadata__ dict in nondeterministic key order,
    so byte-identical eval sets produce different file hashes across boxes —
    an auditor comparing shard hashes across validators needs content identity."""
    from safetensors.torch import save_file
    tensors = {}
    h = hashlib.sha256()
    for i, b in enumerate(batches):
        if b.dtype != torch.int64 or b.dim() != 2:
            raise ValueError(f"batch {i}: eval shards are 2-D int64 token tensors")
        t = b.detach().cpu().contiguous()
        tensors[f"batch_{i:05d}"] = t
        h.update(t.numpy().tobytes())
    md = {"n_batches": str(len(batches)), **(metadata or {})}
    save_file(tensors, path, metadata=md)
    return h.hexdigest()



def _validate_shard(path: str, max_bytes: int = MAX_SHARD_BYTES) -> dict:
    """Validate the raw bytes + tensor inventory of a held-out eval shard.
    Returns the parsed header info; raises before any tensor is materialized."""
    info = validate_safetensors_file(path, max_bytes=max_bytes)
    for name, (shape, dtype) in info["tensors"].items():
        if dtype != "I64" or len(shape) != 2:
            raise CheckpointRejected(f"shard tensor {name!r}: not a 2-D I64 batch")
    return info


def _shard_content_sha256(path: str) -> str:
    """Content hash of a shard: raw tensor bytes in sorted-key order. Matches
    export_eval_shard's hash regardless of safetensors' nondeterministic
    __metadata__ serialization order (file hashes differ box-to-box)."""
    from safetensors import safe_open
    h = hashlib.sha256()
    with safe_open(path, framework="pt") as f:
        for k in sorted(f.keys()):
            h.update(f.get_tensor(k).numpy().tobytes())
    return h.hexdigest()


def iter_eval_shard(path: str, max_bytes: int = MAX_SHARD_BYTES):
    """Stream held-out token batches one CPU tensor at a time (§5 OOM fix).

    Unlike ``load_eval_shard`` this never materializes the whole shard in host
    RAM: it validates the header eagerly, then lazily ``get_tensor``s each batch
    in sorted-key order. A ≥300M-token held-out split is iterated in O(1 batch)
    host memory, so the eval boundary no longer OOMs (the historical 1.7B
    failure, README §11.1). Key order is identical to ``load_eval_shard`` →
    bit-identical mean."""
    _validate_shard(path, max_bytes=max_bytes)
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for k in sorted(f.keys()):
            yield f.get_tensor(k)


def load_eval_shard(path: str, max_bytes: int = MAX_SHARD_BYTES) -> list[torch.Tensor]:
    """Validate then load held-out token batches (CPU tensors, ordered).

    Materializes the full shard in host RAM — fine for small shards/tests. The
    scoring path uses ``iter_eval_shard`` instead to bound peak host memory."""
    _validate_shard(path, max_bytes=max_bytes)
    from safetensors.torch import load_file
    tensors = load_file(path)
    return [tensors[k] for k in sorted(tensors)]


def compute_eval_loss(model: nn.Module, eval_data, use_amp: bool = False,
                      return_count: bool = False):
    """Exact mirror of the harness eval loop in train_and_eval.

    ``eval_data`` is any iterable of CPU token batches (a list, or the
    ``iter_eval_shard`` streaming generator — only one batch is held at a time).
    Batches visit the model's device one at a time; the metric is the plain mean
    of per-batch ``out.loss.float().item()``, accumulated left-to-right exactly
    as ``sum(losses)/len(losses)`` would (bit-identical to trainer curves —
    changing the accumulation order breaks the C3 consistency tripwire).
    With ``return_count`` returns ``(loss, n_batches)`` so streaming callers
    avoid a second pass to count.
    """
    device = next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for eb in eval_data:
            gpu_eb = eb.to(device, non_blocking=True)
            amp_ctx = (torch.amp.autocast(device.type, dtype=torch.bfloat16)
                       if use_amp and device.type == "cuda" else nullcontext())
            with amp_ctx:
                out = model(gpu_eb, labels=gpu_eb)
            total += out.loss.float().item()
            n += 1
            del gpu_eb, out
    loss = total / n if n else float("inf")
    return (loss, n) if return_count else loss


def score_checkpoint(checkpoint_path: str, eval_shard_path: str,
                     model: nn.Module, use_amp: bool = False,
                     max_checkpoint_bytes: int = MAX_CHECKPOINT_BYTES) -> dict:
    """Validate + load a checkpoint into a freshly built model and score it.

    ``model`` must be freshly constructed by TRUSTED code on this box — its
    state_dict defines the expected tensor inventory (validation rejects any
    checkpoint whose tensors are not an exact-shape subset of it; tied
    weights may legitimately be deduplicated by export, so strict load
    enforces completeness).
    """
    expected = expected_tensors_of(model)
    info = validate_safetensors_file(checkpoint_path, max_bytes=max_checkpoint_bytes,
                                     expected_tensors=expected)
    from safetensors.torch import load_model
    missing, unexpected = load_model(model, checkpoint_path, strict=True)
    if missing or unexpected:
        raise CheckpointRejected(f"state mismatch: missing={missing} unexpected={unexpected}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loss, n_batches = compute_eval_loss(
        model, iter_eval_shard(eval_shard_path), use_amp=use_amp, return_count=True)
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    cuda_peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    if os.environ.get("SN125_PROGRESS"):
        print(f"    [score] eval_loss={loss:.4f} n_batches={n_batches} "
              f"peak_rss={peak_rss_bytes/1e9:.2f}GB cuda_peak={cuda_peak_bytes/1e9:.2f}GB",
              flush=True)
    return {
        "eval_loss": loss,
        "n_eval_batches": n_batches,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "eval_shard_sha256": _shard_content_sha256(eval_shard_path),
        "checkpoint_metadata": info["metadata"],
        "use_amp": use_amp,
        "peak_host_rss_bytes": peak_rss_bytes,
        "cuda_max_mem_bytes": cuda_peak_bytes,
    }



@dataclass
class VerifiedScore:
    """Clean-box result + the C3 tripwire against the trainer-reported value."""
    eval_loss: float
    trainer_reported: float | None
    consistent: bool
    result: dict


def apply_verified_loss(curve, result: dict, atol: float = 0.0) -> VerifiedScore:
    """Substitute the clean-box loss for the curve's final eval point.

    The trainer's own number is never trusted for scoring — Box B's loss IS
    the score.  Mutates ``curve.eval_points[-1]`` in place so the curve feeds
    straight into ``scoring.score_submission``.  A mismatch beyond ``atol``
    (default exact) flags a C3 lie; the caller decides DQ policy.
    """
    clean = float(result["eval_loss"])
    reported = None
    consistent = True
    if curve.eval_points:
        step, reported, wall = curve.eval_points[-1]
        consistent = abs(reported - clean) <= atol
        curve.eval_points[-1] = (step, clean, wall)
    else:
        curve.eval_points.append((0, clean, 0.0))
    return VerifiedScore(clean, reported, consistent, result)



def _worker_env(cache_root: str | None = None) -> dict[str, str]:
    """Minimal env for the scorer subprocess (trusted code, hostile inputs)."""
    keep = ("PATH", "HOME", "TMPDIR", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
            "SN125_DIAG_FLASH", "SN125_DIAG_NONDET", "SN125_LEAN_LOSS_CHUNKS")
    prefixes = ("CUDA", "NVIDIA", "HF_", "TRANSFORMERS_", "TORCH", "PYTORCH")
    env = {k: v for k, v in os.environ.items()
           if k in keep or k.startswith(prefixes)}
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = repo_root
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cache_root = cache_root or tempfile.mkdtemp(prefix="sn125_score_cache_")
    for key, sub in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                     ("TRITON_CACHE_DIR", "triton_cache"),
                     ("TRITON_HOME", "triton_home"),
                     ("XDG_CACHE_HOME", ".cache"), ("TMPDIR", "tmp")):
        path = os.path.join(cache_root, sub)
        os.makedirs(path, exist_ok=True)
        env[key] = path
    if "PYTORCH_CUDA_ALLOC_CONF" not in env:
        if env.get("SN125_DIAG_FLASH"):
            env["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
        else:
            env["PYTORCH_CUDA_ALLOC_CONF"] = (
                "garbage_collection_threshold:0.8,max_split_size_mb:512")
    return env


def score_checkpoint_isolated(checkpoint_path: str, eval_shard_path: str,
                              model_spec: str, use_amp: bool = False,
                              device: str = "cpu", timeout: float = 1800.0,
                              max_checkpoint_bytes: int = MAX_CHECKPOINT_BYTES) -> dict:
    """Score in a fresh subprocess that has never run trainer or miner code.

    model_spec: ``hf:<config_name>`` (allowlist enforced by training._build_model)
    or ``toy:<json kwargs>`` (tests / CPU paths).  Raises CheckpointRejected
    when the worker rejected the artifact (rc=2), OptError otherwise.
    """
    with tempfile.TemporaryDirectory(prefix="sn125_score_") as td:
        out_path = os.path.join(td, "result.json")
        cmd = [sys.executable, "-m", "sn125.engine.score_worker",
               "--checkpoint", os.path.abspath(checkpoint_path),
               "--eval-shard", os.path.abspath(eval_shard_path),
               "--model", model_spec, "--device", device,
               "--max-checkpoint-bytes", str(max_checkpoint_bytes),
               "--out", out_path]
        if use_amp:
            cmd.append("--use-amp")
        proc = subprocess.run(cmd, env=_worker_env(os.path.join(td, "cache")), timeout=timeout,
                              capture_output=True, text=True)
        payload = None
        if os.path.isfile(out_path):
            with open(out_path) as f:
                payload = json.load(f)
        if proc.returncode == 2:
            raise CheckpointRejected((payload or {}).get("error", proc.stderr[-2000:]))
        if proc.returncode != 0 or payload is None or "eval_loss" not in payload:
            raise RuntimeError(
                f"score worker failed rc={proc.returncode}: {proc.stderr[-2000:]}")
        return payload



class ToyCausalLM(nn.Module):
    """Tiny causal LM exposing the HF call contract ``model(x, labels=x).loss``."""

    def __init__(self, vocab_size: int = 257, d_model: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        h = torch.tanh(self.ff(self.embed(input_ids)))
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1))
        return type("Out", (), {"loss": loss, "logits": logits})()


def build_model_from_spec(model_spec: str, device: str = "cpu",
                          dtype: torch.dtype | None = None) -> nn.Module:
    """``hf:<name>`` -> training._build_model (CUDA, allowlisted);
    ``toy:<json>`` -> ToyCausalLM(**json) on ``device``."""
    kind, _, arg = model_spec.partition(":")
    if kind == "hf":
        from ..training import _build_model
        return _build_model(arg, seed=0, dtype=dtype)
    if kind == "toy":
        kwargs = json.loads(arg) if arg else {}
        model = ToyCausalLM(**kwargs).to(device)
        if dtype is not None:
            model = model.to(dtype)
        return model
    raise ValueError(f"unknown model spec {model_spec!r}")


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="SN125 clean-box score worker (§6.2)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-shard", required=True)
    ap.add_argument("--model", required=True, help="hf:<config> | toy:<json>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--max-checkpoint-bytes", type=int, default=MAX_CHECKPOINT_BYTES)
    ap.add_argument("--out", required=True, help="ABS path for the result JSON")
    args = ap.parse_args(argv)

    if args.device.startswith("cuda"):
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        if os.environ.get("SN125_DIAG_NONDET"):
            torch.use_deterministic_algorithms(False)
        if not os.environ.get("SN125_DIAG_FLASH"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

    try:
        dtype = torch.bfloat16 if args.use_amp else None
        model = build_model_from_spec(args.model, device=args.device, dtype=dtype)
        result = score_checkpoint(args.checkpoint, args.eval_shard, model,
                                  use_amp=args.use_amp,
                                  max_checkpoint_bytes=args.max_checkpoint_bytes)
    except CheckpointRejected as e:
        with open(args.out, "w") as f:
            json.dump({"error": str(e), "rejected": True}, f)
        return 2
    with open(args.out, "w") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
