"""Isolated refiner process (the Refiner-task analogue of engine/optproc.py).

The miner's refiner — source plus a safetensors weight file — runs in its OWN
process for the whole run and is never imported by the harness. Per plan call
the harness sends the current window of entries (small CPU tensors), any
signals the refiner asked for, and the run context; the refiner answers with
lessons (spans + weights). Everything that crosses the pipe is validated on
the harness side (lessons.parse_plan) before it touches training.

Isolation properties:
  * the refiner process never imports harness code and holds no student model,
    no optimizer, no eval data — a different address space; on CUDA it has its
    own context and can reach only its own allocations.
  * the source is loaded through the same restricted-namespace loader the
    optimizer uses (sn125.sandbox), weights through safetensors (no pickle),
    and the weight file is validated on its raw bytes (clean-box checkpoint
    validator) and checked against the caps BEFORE loading, on both sides.
  * what comes back over the pipe is size-capped and unpickled with a type
    allowlist (numpy arrays + builtins) before parse_plan validates it.
  * a per-call deadline is enforced from the harness side; a breach kills the
    process and fails the run.
  * the refiner's GPU memory is fenced with a per-process allocator fraction
    set in trusted code before the miner source runs.

The CPU path (device="cpu") exercises the identical protocol so all logic is
testable without a GPU.
"""
from __future__ import annotations

import io
import os
import pickle
import traceback
import weakref

import numpy as np
import torch
import torch.multiprocessing as tmp

from .lessons import LessonError, PlanResult, parse_plan

from .refproc_limits import (DEFAULT_PLAN_TIMEOUT_S, MAX_WINDOW_ENTRIES, REFINER_ALLOWED_DTYPES,  # noqa: F401
                             REFINER_GPU_MEMORY_FRACTION, REFINER_MAX_PARAMS, REFINER_MAX_TENSORS,
                             REFINER_MAX_WEIGHT_BYTES, RefProcError, RefProcTimeout,
                             extract_refiner_capabilities)


def validate_refiner_weights(path: str | None, *, max_params: int = REFINER_MAX_PARAMS,
                             max_bytes: int = REFINER_MAX_WEIGHT_BYTES) -> dict:
    """Check the weight file against the caps WITHOUT loading tensor data.

    The raw bytes are validated first by the clean-box checkpoint validator
    (engine.score_worker.validate_safetensors_file: size cap, sane header
    length, JSON-object header — which alone rejects every pickle — dtype
    allowlist, exact offset tiling, no smuggled bytes) before any parser
    library touches the file; the refiner caps (tensor count, parameter
    count, narrower dtype allowlist) are then applied to the parsed header.

    Returns a summary dict (n_tensors, n_params, bytes, dtypes). ``None`` or an
    empty path means a weight-free refiner (allowed)."""
    if not path:
        return {"n_tensors": 0, "n_params": 0, "bytes": 0, "dtypes": []}
    if not os.path.isfile(path):
        raise RefProcError(f"refiner weights not found: {path}")
    from sn125.engine.score_worker import CheckpointRejected, validate_safetensors_file
    try:
        info = validate_safetensors_file(path, max_bytes=max_bytes)
    except CheckpointRejected as e:
        raise RefProcError(f"refiner weights rejected (must be a safetensors file): {e}")
    tensors = info["tensors"]
    if len(tensors) > REFINER_MAX_TENSORS:
        raise RefProcError(f"refiner weights have more than {REFINER_MAX_TENSORS} tensors")
    n_params = 0
    dtypes: set[str] = set()
    for key, (shape, dt) in tensors.items():
        if dt not in REFINER_ALLOWED_DTYPES:
            raise RefProcError(f"refiner weight {key!r} has disallowed dtype {dt}")
        dtypes.add(dt)
        n = 1
        for d in shape:
            n *= int(d)
        n_params += n
        if n_params > max_params:
            raise RefProcError(f"refiner weights exceed {max_params} parameters")
    return {"n_tensors": len(tensors), "n_params": n_params, "bytes": int(info["file_bytes"]),
            "dtypes": sorted(dtypes)}


# ── bounded, restricted receive from the untrusted process ──────────────────
# Lesson payloads come back over the control pipe (small CPU arrays, unlike
# optproc whose per-step data lives in pre-shared buffers). The parent therefore
# (a) caps the message size before reading it — a refiner cannot make the
# harness allocate an arbitrarily large pickle — and (b) unpickles with an
# allowlist of exactly the types _pack produces: numpy arrays and builtins.
# Anything else (any other global) is a protocol violation and fails the run.
MAX_PLAN_PAYLOAD_BYTES = 64 * 1024**2  # 4096 lessons × 2048 fp32 weights ≈ 32 MB, plus spans

_PICKLE_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _PICKLE_ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refiner payload references forbidden global {module}.{name}")


def _recv_restricted(conn, max_bytes: int = MAX_PLAN_PAYLOAD_BYTES):
    """Receive one message from the refiner process with a size cap and the
    restricted unpickler. Raises RefProcError on oversize or forbidden content;
    EOFError/OSError propagate for the caller's death handling."""
    try:
        raw = conn.recv_bytes(max_bytes)
    except OSError as e:
        # multiprocessing raises OSError("bad message length") for a message
        # longer than maxlength and marks the connection unreadable, so the
        # caller must kill the process.
        if "bad message length" in str(e):
            raise RefProcError(f"refiner payload exceeds {max_bytes} bytes") from None
        raise
    try:
        return _RestrictedUnpickler(io.BytesIO(raw)).load()
    except (pickle.UnpicklingError, AttributeError, ValueError, TypeError) as e:
        raise RefProcError(f"refiner payload rejected: {e}")


def _pack(obj):
    """Tensors -> numpy for the pipe. Plain pickle-by-value: no dependence on
    torch's shared-memory tensor reductions (/dev/shm is not guaranteed inside
    the sandbox), and no CUDA handles cross the boundary."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().contiguous().numpy()
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_pack(v) for v in obj)
    return obj


def _unpack(obj):
    """numpy -> CPU tensors on the receiving side (copies, so the arrays are owned)."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.array(obj, copy=True))
    if isinstance(obj, dict):
        return {k: _unpack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unpack(v) for v in obj)
    return obj


# ── refiner-process entry point (the ONLY code that touches miner source) ───

def _refproc_main(conn, source: str, weights_path: str | None, device: str,
                  sandboxed: bool, init_ctx: dict):
    try:
        if device.startswith("cuda"):
            dev = torch.device(device)
            torch.cuda.set_device(dev.index if dev.index is not None else 0)
            try:
                torch.cuda.set_per_process_memory_fraction(REFINER_GPU_MEMORY_FRACTION, dev.index or 0)
            except Exception:
                pass
        weights: dict[str, torch.Tensor] = {}
        if weights_path:
            # Re-validate the bytes that are actually being loaded (the parent
            # validated its copy; this process may see a different mount).
            validate_refiner_weights(weights_path)
            from safetensors.torch import load_file
            weights = load_file(weights_path, device="cpu")
        if sandboxed:
            from sn125.sandbox import load_optimizer_sandboxed
            cls = load_optimizer_sandboxed(source, class_name="Refiner")
        else:
            ns: dict = {}
            exec(compile(source, "<refiner_source>", "exec"), ns)
            cls = ns["Refiner"]
        refiner = cls(weights, dict(init_ctx))
        plan_fn = getattr(refiner, "plan", None)
        if not callable(plan_fn):
            raise RefProcError("Refiner must define plan(window, signals, ctx)")
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        finally:
            return

    conn.send(("ready", os.getpid()))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg[0]
        try:
            if op == "plan":
                _, window, signals, ctx = msg
                raw = plan_fn(_unpack(window), _unpack(signals), ctx)
                out = {
                    "lessons": [{"spans": [tuple(int(x) for x in sp) for sp in L.get("spans", [])],
                                 "weights": _pack(L.get("weights"))}
                                for L in (raw.get("lessons", []) if isinstance(raw, dict) else [])],
                    "retire": [int(x) for x in (raw.get("retire", []) if isinstance(raw, dict) else [])],
                    "rescore": [int(x) for x in (raw.get("rescore", []) if isinstance(raw, dict) else [])],
                } if isinstance(raw, dict) else raw
                conn.send(("ok", out))
            elif op == "shutdown":
                conn.send(("ok", None))
                return
            else:
                conn.send(("error", f"unknown op {op!r}"))
        except BaseException:
            conn.send(("error", traceback.format_exc()))


def _kill_proc(proc):
    if proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)


# ── harness-side handle ──────────────────────────────────────────────────────

class RefProcHandle:
    """Harness-side controller for one isolated refiner process."""

    def __init__(self, source: str, weights_path: str | None, *, device: str = "cpu",
                 sandboxed: bool = True, plan_timeout: float = DEFAULT_PLAN_TIMEOUT_S,
                 start_timeout: float = 120.0, init_ctx: dict | None = None):
        self.weights_summary = validate_refiner_weights(weights_path)
        self.capabilities = extract_refiner_capabilities(source)
        self.plan_timeout = float(plan_timeout)
        self.device = device
        ctx = tmp.get_context("spawn")
        self._conn, child = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(
            target=_refproc_main,
            args=(child, source, weights_path, device, sandboxed, dict(init_ctx or {})),
            daemon=True)
        self._proc.start()
        child.close()
        self._finalizer = weakref.finalize(self, _kill_proc, self._proc)
        if not self._conn.poll(start_timeout):
            _kill_proc(self._proc)
            raise RefProcTimeout(f"refiner process did not start within {start_timeout:.0f}s")
        try:
            kind, payload = _recv_restricted(self._conn)
        except (EOFError, OSError) as e:
            _kill_proc(self._proc)
            raise RefProcError(f"refiner process died during startup ({type(e).__name__}); "
                               f"exitcode={self._proc.exitcode}")
        except RefProcError:
            _kill_proc(self._proc)
            raise
        if kind != "ready":
            _kill_proc(self._proc)
            raise RefProcError(f"refiner failed to load: {payload}")
        self.pid = int(payload)
        self.calls = 0

    def plan(self, window: list[dict], signals: dict, ctx: dict) -> PlanResult:
        if not self._proc.is_alive():
            raise RefProcError("refiner process is dead")
        try:
            self._conn.send(("plan", _pack(window), _pack(signals), ctx))
        except (BrokenPipeError, OSError) as e:
            raise RefProcError(f"refiner pipe broken: {e}")
        if not self._conn.poll(self.plan_timeout):
            _kill_proc(self._proc)
            raise RefProcTimeout(f"refiner plan() exceeded {self.plan_timeout:.0f}s (call {self.calls})")
        try:
            kind, payload = _recv_restricted(self._conn)
        except (EOFError, OSError) as e:
            _kill_proc(self._proc)
            raise RefProcError(f"refiner process died during plan() ({type(e).__name__}); "
                               f"exitcode={self._proc.exitcode}")
        except RefProcError:
            _kill_proc(self._proc)  # oversize or forbidden payload: the pipe is no longer trustworthy
            raise
        self.calls += 1
        if kind != "ok":
            raise RefProcError(f"refiner plan() failed: {str(payload)[-4000:]}")
        window_ids = {int(e["wid"]) for e in window}
        try:
            return parse_plan(_unpack(payload), window_ids)
        except LessonError as e:
            raise RefProcError(f"refiner protocol violation: {e}")

    def close(self):
        try:
            if self._proc.is_alive():
                self._conn.send(("shutdown",))
                self._conn.poll(5.0)
        except Exception:
            pass
        _kill_proc(self._proc)
        try:
            self._conn.close()
        except Exception:
            pass
