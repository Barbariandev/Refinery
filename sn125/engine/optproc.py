"""Q9 — optimizer-process isolation (DESIGN.md §6.1).

The miner's update rule runs in its OWN process for the entire run and is
never imported by the harness.  Per step the two processes share exactly two
persistent buffers per parameter:

    grad_buf[name]   harness -> miner : the clipped gradient
                     miner -> harness : the UPDATE (written back in place)
    param_buf[name]  harness -> miner : detached snapshot of current params

Buffers are allocated once by the harness and handed to the miner process at
spawn time — torch.multiprocessing turns CPU tensors into shared memory and
CUDA tensors into CUDA-IPC handles, so per-step tensor traffic is zero-copy.
The only per-step cost is a control-pipe round trip plus device syncs.

Why the update comes back in a buffer instead of the miner writing params in
place (the literal §6.1 sketch): the harness applies the external WSD
envelope and validates the update (shape / finiteness) before touching the
model — ``p.add_(u * schedule_scale)`` stays in trusted code, and the
existing miner contract (``Optimizer(param_groups, config)`` /
``step(gradients, param_values, step) -> updates``) is preserved bit-for-bit.
Isolation is identical: the miner still only ever reads grads and writes an
update.

Isolation properties:
  * the miner process never imports harness code and holds no model, data,
    schedule, logger, or harness RNG — a different address space.
  * on CUDA its driver context can reach only the buffers we explicitly
    shared (plus its own allocations).
  * harness-side caps (per-step timeout on the control pipe, kill on breach)
    are enforced from a process the miner cannot touch.
  * TOCTOU defense: a hostile miner could keep a background thread writing
    the shared update buffer AFTER the harness validates it.  Therefore the
    harness SNAPSHOTS (clones) every update out of shared memory before any
    validation/apply (``snapshot_updates=True``, the default — only the
    benchmark turns it off to price it).

Known consequence of the split (documented, by design): harness-side
``torch.cuda.memory_allocated()`` accounting no longer sees miner state —
memory caps move to per-process accounting (NVML / cgroup) at the launcher.

The CPU path (device="cpu") exercises the identical protocol over
shared-memory tensors so all logic is testable without a GPU.
"""
from __future__ import annotations

import os
import traceback
import weakref

import torch
import torch.multiprocessing as tmp

ParamSpec = list[dict]


class OptProcError(RuntimeError):
    """Fatal error from / about the miner optimizer process."""


class OptProcTimeout(OptProcError):
    """Miner process failed to respond within the deadline (process killed)."""



def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_cpu(v) for v in obj)
    return obj


def _to_device(obj, device):
    if torch.is_tensor(obj) and obj.dtype not in (torch.uint8,):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_device(v, device) for v in obj)
    return obj



def _optproc_main(conn, source: str, param_groups_spec: ParamSpec, config: dict,
                  grad_bufs: dict, param_bufs: dict, device: str, sandboxed: bool):
    """Runs in the miner process. Loads the optimizer, then serves the
    per-step protocol until 'shutdown' or parent death (EOFError)."""
    try:
        if device.startswith("cuda"):
            dev = torch.device(device)
            torch.cuda.set_device(dev.index if dev.index is not None else 0)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
        if sandboxed:
            from sn125.sandbox import load_optimizer_sandboxed
            opt_cls = load_optimizer_sandboxed(source)
        else:
            ns: dict = {}
            exec(compile(source, "<miner_source>", "exec"), ns)
            opt_cls = ns["Optimizer"]
        opt = opt_cls(param_groups_spec, config)
        param_names = [name for pg in param_groups_spec
                       for name, _, _ in pg["params"]]
        param_name_set = set(param_names)
        grad_list = [grad_bufs[name] for name in param_names]
        param_list = [param_bufs[name] for name in param_names
                      if name in param_bufs]
        returns_full_update = bool(config.get("_returns_full_update", False))
        ordered_step = None
        if bool(config.get("_ordered_buffers", False)):
            ordered_step = getattr(opt, "step_ordered", None)
            if not callable(ordered_step):
                ordered_step = None
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
            if op == "step":
                step_idx = msg[1]
                param_values_available = True if len(msg) < 3 else bool(msg[2])
                names = []
                if ordered_step is not None:
                    pvals = param_list if param_values_available else []
                    updates = ordered_step(grad_list, pvals, step_idx)
                    if updates is None:
                        if not returns_full_update:
                            raise OptProcError(
                                "step_ordered returned None without returns_full_update=True")
                        names = list(param_names)
                    elif isinstance(updates, dict):
                        for name, u in updates.items():
                            buf = grad_bufs.get(name)
                            if buf is None:
                                raise OptProcError(f"update for unknown param {name!r}")
                            if u is not buf:
                                if not torch.is_tensor(u):
                                    u = torch.as_tensor(u, dtype=buf.dtype, device=buf.device)
                                if tuple(u.shape) != tuple(buf.shape):
                                    raise OptProcError(
                                        f"Shape mismatch for {name}: update {tuple(u.shape)} != param {tuple(buf.shape)}")
                                buf.copy_(u)
                            names.append(name)
                    else:
                        if len(updates) != len(param_names):
                            raise OptProcError(
                                f"step_ordered returned {len(updates)} updates for {len(param_names)} params")
                        for name, buf, u in zip(param_names, grad_list, updates):
                            if u is None:
                                if returns_full_update:
                                    raise OptProcError(f"missing ordered update for {name!r}")
                                continue
                            if u is not buf:
                                if not torch.is_tensor(u):
                                    u = torch.as_tensor(u, dtype=buf.dtype, device=buf.device)
                                if tuple(u.shape) != tuple(buf.shape):
                                    raise OptProcError(
                                        f"Shape mismatch for {name}: update {tuple(u.shape)} != param {tuple(buf.shape)}")
                                buf.copy_(u)
                            names.append(name)
                else:
                    pvals = dict(param_bufs) if param_values_available else {}
                    updates = opt.step(dict(grad_bufs), pvals, step_idx)
                    for name, u in (updates or {}).items():
                        buf = grad_bufs.get(name)
                        if buf is None:
                            raise OptProcError(f"update for unknown param {name!r}")
                        if u is not buf:
                            if not torch.is_tensor(u):
                                u = torch.as_tensor(u, dtype=buf.dtype, device=buf.device)
                            if tuple(u.shape) != tuple(buf.shape):
                                raise OptProcError(
                                    f"Shape mismatch for {name}: update {tuple(u.shape)} != param {tuple(buf.shape)}")
                            buf.copy_(u)
                        names.append(name)
                if returns_full_update:
                    seen = set(names)
                    if seen != param_name_set:
                        missing = sorted(param_name_set - seen)[:8]
                        extra = sorted(seen - param_name_set)[:8]
                        raise OptProcError(
                            f"returns_full_update=True but update set differs; missing={missing}, extra={extra}")
                if device.startswith("cuda"):
                    if not bool(config.get("_unsynced_ipc", False)):
                        torch.cuda.synchronize()
                conn.send(("ok", step_idx, None if returns_full_update else names))
            elif op == "get_state":
                conn.send(("state", _to_cpu(opt.state_dict())))
            elif op == "set_state":
                opt.load_state_dict(_to_device(msg[1], device))
                conn.send(("ok", -1, []))
            elif op == "get_rng":
                rng = {"torch": torch.get_rng_state()}
                if device.startswith("cuda"):
                    rng["cuda"] = torch.cuda.get_rng_state_all()
                conn.send(("rng", _to_cpu(rng)))
            elif op == "set_rng":
                rng = msg[1]
                torch.set_rng_state(rng["torch"])
                if device.startswith("cuda") and "cuda" in rng:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                conn.send(("ok", -1, []))
            elif op == "shutdown":
                conn.send(("ok", -1, []))
                return
            else:
                conn.send(("error", f"unknown op {op!r}"))
        except BaseException:
            conn.send(("error", traceback.format_exc()))


def _kill_proc(proc):
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)



class OptProcHandle:
    """Harness-side controller for one isolated miner optimizer process.

    ``param_groups_spec`` is exactly what train_and_eval already builds:
    ``[{"params": [(name, shape, dtype), ...], "lr": .., "weight_decay": ..}]``.
    """

    def __init__(self, source: str, param_groups_spec: ParamSpec, config: dict,
                 device: str = "cpu", *, sandboxed: bool = True,
                 step_timeout: float = 60.0, spawn_timeout: float = 300.0,
                 snapshot_updates: bool = True):
        self.device = torch.device(device)
        self.step_timeout = float(step_timeout)
        self.snapshot_updates = snapshot_updates
        self._param_groups_spec = param_groups_spec
        self._requires_param_values = bool(config.get("_requires_param_values", True))
        try:
            self._param_values_interval = int(config.get(
                "_param_values_interval", 1 if self._requires_param_values else 0))
        except (TypeError, ValueError):
            self._param_values_interval = 1 if self._requires_param_values else 0
        self._param_values_interval = max(0, self._param_values_interval)
        self._trusted_weight_decay = bool(config.get("_trusted_weight_decay", False))
        self._returns_full_update = bool(config.get("_returns_full_update", False))
        self._ordered_buffers = bool(config.get("_ordered_buffers", False))
        self._unsynced_ipc = bool(config.get("_unsynced_ipc", False))
        self._param_names = [name for pg in param_groups_spec
                             for name, _, _ in pg["params"]]
        self._name_to_idx = {name: i for i, name in enumerate(self._param_names)}
        self._wd_groups: list[tuple[float, float, list[int]]] = []
        offset = 0
        for pg in param_groups_spec:
            n = len(pg["params"])
            idxs = list(range(offset, offset + n))
            self._wd_groups.append((
                float(pg.get("lr", 0.0) or 0.0),
                float(pg.get("weight_decay", 0.0) or 0.0),
                idxs,
            ))
            offset += n
        self._grad_bufs: dict[str, torch.Tensor] = {}
        self._param_bufs: dict[str, torch.Tensor] = {}
        self._grad_buf_list: list[torch.Tensor] = []
        self._param_buf_list: list[torch.Tensor] = []
        for pg in param_groups_spec:
            for name, shape, dtype in pg["params"]:
                g = torch.zeros(shape, dtype=dtype, device=self.device)
                if self.device.type == "cpu":
                    g.share_memory_()
                self._grad_bufs[name] = g
                self._grad_buf_list.append(g)
                if self._param_values_interval:
                    p = torch.zeros(shape, dtype=dtype, device=self.device)
                    if self.device.type == "cpu":
                        p.share_memory_()
                    self._param_bufs[name] = p
                    self._param_buf_list.append(p)
        self._bound_model_id: int | None = None
        self._target_params: list[torch.Tensor] = []
        self._target_by_name: dict[str, torch.Tensor] = {}

        ctx = tmp.get_context("spawn")
        self._conn, child_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(
            target=_optproc_main,
            args=(child_conn, source, param_groups_spec, config,
                  self._grad_bufs, self._param_bufs, str(self.device), sandboxed),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._finalizer = weakref.finalize(self, _kill_proc, self._proc)
        msg = self._recv(spawn_timeout)
        if msg[0] != "ready":
            raise OptProcError(f"miner process failed to start: {msg!r}")
        self.pid = msg[1]

    def _recv(self, timeout: float):
        if not self._conn.poll(timeout):
            _kill_proc(self._proc)
            raise OptProcTimeout(f"miner process unresponsive after {timeout:.1f}s (killed)")
        try:
            msg = self._conn.recv()
        except EOFError:
            raise OptProcError("miner process died (pipe EOF)") from None
        if msg[0] == "error":
            raise OptProcError(f"miner process error:\n{msg[1]}")
        return msg

    def _request(self, msg, timeout: float):
        if not self._proc.is_alive():
            raise OptProcError("miner process is not alive")
        self._conn.send(msg)
        return self._recv(timeout)

    def _param_values_available(self, step_number: int) -> bool:
        return bool(self._param_values_interval and
                    step_number % self._param_values_interval == 0)

    def _apply_trusted_weight_decay(self, params: list[torch.Tensor],
                                    update_scale: float) -> None:
        if not self._trusted_weight_decay or update_scale == 0:
            return
        with torch.no_grad():
            for lr, wd, idxs in self._wd_groups:
                if lr == 0.0 or wd == 0.0:
                    continue
                group = [params[i] for i in idxs]
                if not group:
                    continue
                scale = 1.0 - lr * wd * float(update_scale)
                try:
                    torch._foreach_mul_(group, scale)
                except RuntimeError:
                    for p in group:
                        p.mul_(scale)

    def _bind_model_params(self, model) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        model_id = id(model)
        if model_id == self._bound_model_id:
            return self._target_params, self._target_by_name
        by_name = dict(model.named_parameters())
        targets = []
        for name, buf in zip(self._param_names, self._grad_buf_list):
            p = by_name.get(name)
            if p is None:
                raise OptProcError(f"shared buffer for unknown model param {name!r}")
            if p.device != buf.device:
                raise OptProcError(
                    f"Device mismatch for {name}: buffer on {buf.device}, param on {p.device}")
            if tuple(p.shape) != tuple(buf.shape):
                raise OptProcError(
                    f"Shape mismatch for {name}: buffer {tuple(buf.shape)} != param {tuple(p.shape)}")
            targets.append(p)
        self._bound_model_id = model_id
        self._target_params = targets
        self._target_by_name = by_name
        return targets, by_name

    def step(self, gradients: dict, param_values: dict, step_number: int) -> dict:
        """Fill shared buffers, run one isolated step, return the updates.

        With ``snapshot_updates`` (default) the returned tensors are CLONES of
        the shared buffers, immune to post-validation tampering by a hostile
        miner thread (TOCTOU). With it off, returned tensors alias shared
        memory and MUST be consumed before the next step.
        """
        for name, buf in self._grad_bufs.items():
            g = gradients.get(name)
            buf.copy_(g) if g is not None else buf.zero_()
        param_values_available = self._param_values_available(step_number)
        if param_values_available:
            for name, buf in self._param_bufs.items():
                p = param_values.get(name)
                buf.copy_(p) if p is not None else buf.zero_()
        if self.device.type == "cuda":
            if not self._unsynced_ipc:
                torch.cuda.synchronize()
        msg = self._request(("step", step_number, param_values_available),
                            self.step_timeout)
        names = self._param_names if msg[2] is None else msg[2]
        if self.snapshot_updates:
            return {n: self._grad_bufs[n].clone() for n in names}
        return {n: self._grad_bufs[n] for n in names}

    def step_model(self, model, step_number: int, update_scale: float = 1.0,
                   finite_check: bool = True,
                   clear_model_grads: bool = True) -> list[str]:
        """Fast trusted-parent path for train_and_eval.

        The miner still runs in the child process and only sees the persistent
        shared grad/param buffers.  The parent copies model grads/params directly
        into those buffers, asks the child for updates, validates the named update
        buffers, and applies them to the real model with a foreach add.  This
        removes the old intermediate cloned ``gradients``/``param_values`` dicts
        from the harness hot path without giving the child CUDA-IPC access to the
        live model parameters.
        """
        target_params, target_by_name = self._bind_model_params(model)
        for p, buf in zip(target_params, self._grad_buf_list):
            g = p.grad
            buf.copy_(g, non_blocking=True) if g is not None else buf.zero_()
        param_values_available = self._param_values_available(step_number)
        if param_values_available:
            for p, buf in zip(target_params, self._param_buf_list):
                buf.copy_(p.detach(), non_blocking=True)
        if clear_model_grads:
            model.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            if not self._unsynced_ipc:
                torch.cuda.synchronize()

        msg = self._request(("step", step_number, param_values_available),
                            self.step_timeout)
        names = self._param_names if msg[2] is None else msg[2]
        update_tensors = []
        apply_params = []
        for name in names:
            idx = self._name_to_idx.get(name)
            if idx is None:
                raise OptProcError(f"update for unknown param {name!r}")
            p = target_params[idx]
            u = self._grad_buf_list[idx]
            if self.snapshot_updates:
                u = u.clone()
            update_tensors.append(u)
            apply_params.append(p)

        if finite_check and update_tensors:
            flags = [torch.isfinite(u).all() for u in update_tensors]
            if not bool(torch.stack(flags).all().item()):
                for name, u in zip(names, update_tensors):
                    if not bool(torch.isfinite(u).all().item()):
                        raise OptProcError(f"NaN/Inf in update for {name}")

        self._apply_trusted_weight_decay(target_params, update_scale)
        if update_tensors:
            with torch.no_grad():
                try:
                    torch._foreach_add_(apply_params, update_tensors,
                                        alpha=float(update_scale))
                except RuntimeError:
                    for p, u in zip(apply_params, update_tensors):
                        p.add_(u * update_scale)
        return list(names)

    def state_dict(self) -> dict:
        return self._request(("get_state",), max(self.step_timeout, 300.0))[1]

    def load_state_dict(self, state: dict):
        self._request(("set_state", _to_cpu(state)), max(self.step_timeout, 300.0))

    def get_rng_state(self) -> dict:
        return self._request(("get_rng",), self.step_timeout)[1]

    def set_rng_state(self, rng: dict):
        self._request(("set_rng", rng), self.step_timeout)

    def close(self):
        try:
            if self._proc.is_alive():
                self._conn.send(("shutdown",))
                self._proc.join(timeout=5.0)
        except (BrokenPipeError, OSError):
            pass
        _kill_proc(self._proc)

    @property
    def is_alive(self) -> bool:
        return self._proc.is_alive()


def make_split_optimizer_cls(source: str, device: str, *, sandboxed: bool = True,
                             step_timeout: float = 60.0,
                             snapshot_updates: bool = True) -> type:
    """Adapter factory: returns a class with the in-process optimizer
    interface (``cls(param_groups, config)`` + ``step(...) -> updates``) that
    transparently runs ``source`` in an isolated process — a drop-in
    ``optimizer_cls`` for train_and_eval, which is exactly how the
    bit-exactness check runs the same source both ways."""

    class SplitOptimizer:
        def __init__(self, param_groups, config):
            self._sn125_is_split_optimizer = True
            self._handle = OptProcHandle(
                source, param_groups, config, device=device, sandboxed=sandboxed,
                step_timeout=step_timeout, snapshot_updates=snapshot_updates)

        def step(self, gradients, param_values, step_number):
            return self._handle.step(gradients, param_values, step_number)

        def step_model(self, model, step_number, update_scale=1.0,
                       finite_check=True, clear_model_grads=True):
            return self._handle.step_model(
                model, step_number, update_scale=update_scale,
                finite_check=finite_check,
                clear_model_grads=clear_model_grads)

        def state_dict(self):
            return self._handle.state_dict()

        def load_state_dict(self, s):
            self._handle.load_state_dict(s)

        def get_rng_state(self):
            return self._handle.get_rng_state()

        def set_rng_state(self, rng):
            self._handle.set_rng_state(rng)

        def close(self):
            self._handle.close()

    return SplitOptimizer
