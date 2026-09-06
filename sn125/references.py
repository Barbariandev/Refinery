"""SN125 — Reference optimizer source strings + metadata extraction."""
import ast


ADAMW_SOURCE = '''
HPARAMS = {"lr": 1e-3, "weight_decay": 0.01,
           "pretrained": {"lr": 3e-5}, "scratch": {"lr": 1e-3}}
import torch, math
class Optimizer:
    """AdamW with an internal WSD (warmup-stable-decay) LR schedule.

    The harness applies updates at scale 1.0 — the submission owns its own
    schedule. config carries total_steps / warmup_steps / decay_fraction."""
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
        self.config = config
        self.total_steps = int(config.get("total_steps", 0) or 0)
        self.warmup_steps = int(config.get("warmup_steps", 0) or 0)
        self.decay_fraction = float(config.get("decay_fraction", 0.0) or 0.0)
        self.state = {}
        for pg in param_groups:
            for name, shape, dtype in pg["params"]:
                self.state[name] = {
                    "m": torch.zeros(shape, dtype=dtype, device="cuda"),
                    "v": torch.zeros(shape, dtype=dtype, device="cuda"),
                }
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def _schedule(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.decay_fraction > 0.0 and self.total_steps > 0:
            decay_steps = max(1, int(self.total_steps * self.decay_fraction))
            decay_start = self.total_steps - decay_steps
            if step >= decay_start:
                progress = min(1.0, (step - decay_start) / decay_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def step(self, gradients, param_values, step_number):
        t = step_number + 1
        scale = self._schedule(step_number)
        updates = {}
        for pg in self.param_groups:
            lr, wd = pg["lr"] * scale, pg["weight_decay"]
            for name, shape, dtype in pg["params"]:
                g = gradients[name]
                s = self.state[name]
                s["m"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                s["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                m_hat = s["m"] / (1 - self.beta1 ** t)
                v_hat = s["v"] / (1 - self.beta2 ** t)
                update = -lr * (m_hat / (v_hat.sqrt() + self.eps))
                if wd > 0:
                    update.add_(param_values[name], alpha=-lr * wd)
                updates[name] = update
        return updates

    def state_dict(self):
        return {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in self.state.items()}
    def load_state_dict(self, s):
        for k, v in s.items():
            for kk, vv in v.items():
                self.state[k][kk].copy_(vv)
'''

SGDM_SOURCE = '''
HPARAMS = {"lr": 1e-2, "weight_decay": 0.0,
           "pretrained": {"lr": 1e-3}, "scratch": {"lr": 1e-2}}
import torch, math
class Optimizer:
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
        self.state = {}
        for pg in param_groups:
            for name, shape, dtype in pg["params"]:
                self.state[name] = {"v": torch.zeros(shape, dtype=dtype, device="cuda")}
        self.momentum = 0.9

    def step(self, gradients, param_values, step_number):
        updates = {}
        for pg in self.param_groups:
            lr = pg["lr"]
            for name, shape, dtype in pg["params"]:
                g = gradients[name]
                s = self.state[name]
                s["v"].mul_(self.momentum).add_(g)
                updates[name] = -lr * s["v"]
        return updates

    def state_dict(self):
        return {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in self.state.items()}
    def load_state_dict(self, s):
        for k, v in s.items():
            for kk, vv in v.items():
                self.state[k][kk].copy_(vv)
'''

ADAM_SOURCE = '''
HPARAMS = {"lr": 1e-3, "weight_decay": 0.0,
           "pretrained": {"lr": 3e-5}, "scratch": {"lr": 1e-3}}
import torch, math
class Optimizer:
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
        self.state = {}
        for pg in param_groups:
            for name, shape, dtype in pg["params"]:
                self.state[name] = {"m": torch.zeros(shape, dtype=dtype, device="cuda"),
                                    "v": torch.zeros(shape, dtype=dtype, device="cuda")}
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
    def step(self, gradients, param_values, step_number):
        t = step_number + 1
        updates = {}
        for pg in self.param_groups:
            lr = pg["lr"]
            for name, shape, dtype in pg["params"]:
                g = gradients[name]
                s = self.state[name]
                s["m"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                s["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                m_hat = s["m"] / (1 - self.beta1 ** t)
                v_hat = s["v"] / (1 - self.beta2 ** t)
                updates[name] = -lr * (m_hat / (v_hat.sqrt() + self.eps))
        return updates
    def state_dict(self): return {}
    def load_state_dict(self, s): pass
'''

SGD_SOURCE = '''
HPARAMS = {"lr": 1e-2, "weight_decay": 0.0,
           "pretrained": {"lr": 1e-3}, "scratch": {"lr": 1e-2}}
import torch, math
class Optimizer:
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
    def step(self, gradients, param_values, step_number):
        updates = {}
        for pg in self.param_groups:
            lr = pg["lr"]
            for name, shape, dtype in pg["params"]:
                updates[name] = -lr * gradients[name]
        return updates
    def state_dict(self): return {}
    def load_state_dict(self, s): pass
'''

LION_SOURCE = '''
HPARAMS = {"lr": 1e-4, "weight_decay": 0.01,
           "pretrained": {"lr": 1e-5}, "scratch": {"lr": 1e-4}}
import torch, math
class Optimizer:
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
        self.state = {}
        for pg in param_groups:
            for name, shape, dtype in pg["params"]:
                self.state[name] = {"m": torch.zeros(shape, dtype=dtype, device="cuda")}
        self.beta1, self.beta2 = 0.9, 0.99
    def step(self, gradients, param_values, step_number):
        updates = {}
        for pg in self.param_groups:
            lr, wd = pg["lr"], pg["weight_decay"]
            for name, shape, dtype in pg["params"]:
                g = gradients[name]
                m = self.state[name]["m"]
                update = torch.sign(m * self.beta1 + g * (1 - self.beta1))
                u = -lr * update
                if wd > 0:
                    u.add_(param_values[name], alpha=-lr * wd)
                updates[name] = u
                m.mul_(self.beta2).add_(g, alpha=1 - self.beta2)
        return updates
    def state_dict(self): return {}
    def load_state_dict(self, s): pass
'''

SCHEDULE_FREE_SOURCE = '''
HPARAMS = {"lr": 1e-3, "weight_decay": 0.01,
           "pretrained": {"lr": 3e-5}, "scratch": {"lr": 1e-3}}
import torch, math
class Optimizer:
    """Schedule-Free AdamW: interpolates between iterate and average,
    no LR schedule needed. Based on Defazio et al. 2024."""
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
        self.config = config
        self.state = {}
        self.initialized = False
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
    def step(self, gradients, param_values, step_number):
        t = step_number + 1
        if not self.initialized:
            for pg in self.param_groups:
                for name, shape, dtype in pg["params"]:
                    self.state[name] = {
                        "z": param_values[name].clone(),
                        "v": torch.zeros(shape, dtype=dtype, device="cuda"),
                    }
            self.initialized = True
        updates = {}
        for pg in self.param_groups:
            lr, wd = pg["lr"], pg["weight_decay"]
            for name, shape, dtype in pg["params"]:
                g = gradients[name]
                s = self.state[name]
                x = param_values[name]
                s["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                v_hat = s["v"] / (1 - self.beta2 ** t)
                denom = v_hat.sqrt() + self.eps
                s["z"].add_(g / denom, alpha=-lr)
                if wd > 0:
                    s["z"].add_(x, alpha=-lr * wd)
                updates[name] = (s["z"] - x) * (1 - self.beta1)
        return updates
    def state_dict(self):
        return {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in self.state.items()}
    def load_state_dict(self, s):
        for k, v in s.items():
            for kk, vv in v.items():
                self.state[k][kk].copy_(vv)
'''

PRODIGY_SOURCE = '''
HPARAMS = {"lr": 1.0, "weight_decay": 0.01,
           "pretrained": {"lr": 1.0, "weight_decay": 0.01},
           "scratch":    {"lr": 1.0, "weight_decay": 0.05}}
import torch, math
class Optimizer:
    """D-Adapted AdamW (Prodigy): auto-tunes LR via distance estimation.
    lr=1.0 is a confidence knob, not a step size. Based on Mishchenko & Defazio 2024."""
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
        self.state = {}
        self.d = 1e-6
        self.d_numer = 0.0
        self.d_max = 0.0
        for pg in param_groups:
            for name, shape, dtype in pg["params"]:
                self.state[name] = {
                    "m": torch.zeros(shape, dtype=dtype, device="cuda"),
                    "v": torch.zeros(shape, dtype=dtype, device="cuda"),
                    "s": torch.zeros(shape, dtype=dtype, device="cuda"),
                }
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
    def step(self, gradients, param_values, step_number):
        t = step_number + 1
        dlr = self.d * self.param_groups[0]["lr"]
        ip = 0.0
        for pg in self.param_groups:
            for name, _, _ in pg["params"]:
                ip += (gradients[name] * self.state[name]["s"]).sum().item()
        self.d_numer += dlr * ip
        s_sq = sum(self.state[n]["s"].pow(2).sum().item()
                   for pg in self.param_groups for n, _, _ in pg["params"])
        self.d_max = max(self.d_max, math.sqrt(s_sq))
        if self.d_max > 0:
            self.d = max(self.d, abs(self.d_numer) / self.d_max)
            dlr = self.d * self.param_groups[0]["lr"]
        updates = {}
        for pg in self.param_groups:
            wd = pg["weight_decay"]
            for name, shape, dtype in pg["params"]:
                g = gradients[name]
                s = self.state[name]
                s["s"].mul_(self.beta2 ** 0.5).add_(g, alpha=dlr)
                s["m"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                s["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                m_hat = s["m"] / (1 - self.beta1 ** t)
                v_hat = s["v"] / (1 - self.beta2 ** t)
                update = -dlr * m_hat / (v_hat.sqrt() + self.eps * self.d)
                if wd > 0:
                    update.add_(param_values[name], alpha=-dlr * wd)
                updates[name] = update
        return updates
    def state_dict(self):
        return {k: {sk: sv.clone() for sk, sv in v.items()} for k, v in self.state.items()}
    def load_state_dict(self, s):
        for k, v in s.items():
            if k in self.state:
                for sk, sv in v.items():
                    self.state[k][sk].copy_(sv)
'''

MUON_SOURCE = '''
HPARAMS = {"lr": 0.02, "weight_decay": 0.01,
           "pretrained": {"lr": 0.005}, "scratch": {"lr": 0.02}}
import torch, math
class Optimizer:
    """Muon: momentum + Newton-Schulz orthogonalization for 2D weights.
    1D params use AdamW. Based on Jordan et al. 2024 (Muon)."""
    def __init__(self, param_groups, config):
        self.pg = param_groups
        self.state = {}
        for pg in param_groups:
            for name, shape, dtype in pg["params"]:
                is2d = len(shape) == 2 and min(shape) >= 16
                s = {"m": torch.zeros(shape, dtype=torch.float32, device="cuda"),
                     "is2d": is2d}
                if not is2d:
                    s["v"] = torch.zeros(shape, dtype=torch.float32, device="cuda")
                self.state[name] = s
        self.b1, self.b2, self.eps = 0.95, 0.999, 1e-8
    def _ns_orth(self, M):
        a, b, c = 3.4445, -4.7750, 2.0315
        t = M.T if M.shape[0] > M.shape[1] else None
        X = (t if t is not None else M).float()
        X /= (X.norm() + 1e-7)
        for _ in range(3):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))
        return X.T if t is not None else X
    def step(self, gradients, param_values, step_number):
        t = step_number + 1
        bc1 = 1 - self.b1 ** t
        updates = {}
        for pg in self.pg:
            lr, wd = pg["lr"], pg["weight_decay"]
            for name, shape, dtype in pg["params"]:
                g = gradients[name].float()
                s = self.state[name]
                s["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
                if s["is2d"]:
                    update = (-lr * self._ns_orth(s["m"] / bc1)).to(dtype)
                else:
                    s["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                    m_h = s["m"] / bc1
                    v_h = s["v"] / (1 - self.b2 ** t)
                    update = (-lr * 0.1 * m_h / (v_h.sqrt() + self.eps)).to(dtype)
                if wd > 0:
                    update.add_(param_values[name], alpha=-lr * wd)
                updates[name] = update
        return updates
    def state_dict(self):
        return {k: {sk: sv.clone() for sk, sv in v.items() if sk != "is2d"}
                for k, v in self.state.items()}
    def load_state_dict(self, s):
        for k, v in s.items():
            if k in self.state:
                for sk, sv in v.items():
                    self.state[k][sk].copy_(sv)
'''

FUSED_ADAMW_SOURCE = '''
HPARAMS = {"lr": 1e-3, "weight_decay": 0.01,
           "pretrained": {"lr": 3e-5}, "scratch": {"lr": 1e-3}}
import torch, math, triton, triton.language as tl

@triton.jit
def _fused_adam_k(p_ptr, g_ptr, m_ptr, v_ptr, out_ptr,
                  lr, beta1, beta2, eps, wd, bc1, bc2,
                  N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    g = tl.load(g_ptr + off, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + off, mask=mask)
    v = tl.load(v_ptr + off, mask=mask)
    p = tl.load(p_ptr + off, mask=mask).to(tl.float32)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    u = -lr * (m / bc1 / (tl.sqrt(v / bc2) + eps)) - lr * wd * p
    tl.store(m_ptr + off, m, mask=mask)
    tl.store(v_ptr + off, v, mask=mask)
    tl.store(out_ptr + off, u, mask=mask)

class Optimizer:
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
        self.state = {}
        for pg in param_groups:
            for name, shape, dtype in pg["params"]:
                n = 1
                for s in shape:
                    n *= s
                self.state[name] = {
                    "m": torch.zeros(n, dtype=torch.float32, device="cuda"),
                    "v": torch.zeros(n, dtype=torch.float32, device="cuda"),
                    "n": n,
                }
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def step(self, gradients, param_values, step_number):
        t = step_number + 1
        bc1 = 1.0 - self.beta1 ** t
        bc2 = 1.0 - self.beta2 ** t
        updates = {}
        for pg in self.param_groups:
            lr, wd = pg["lr"], pg["weight_decay"]
            for name, shape, dtype in pg["params"]:
                s = self.state[name]
                n = s["n"]
                g = gradients[name].reshape(-1)
                p = param_values[name].reshape(-1)
                out = torch.empty(n, dtype=torch.float32, device="cuda")
                _fused_adam_k[((n + 1023) // 1024,)](
                    p, g, s["m"], s["v"], out,
                    lr, self.beta1, self.beta2, self.eps, wd, bc1, bc2,
                    n, BLOCK=1024)
                updates[name] = out.reshape(shape).to(dtype)
        return updates

    def state_dict(self):
        return {k: {kk: vv.clone() for kk, vv in v.items() if kk != "n"} for k, v in self.state.items()}
    def load_state_dict(self, s):
        for k, v in s.items():
            for kk, vv in v.items():
                self.state[k][kk].copy_(vv)
'''

REFERENCE_OPTIMIZERS = {"adamw": ADAMW_SOURCE, "adamw_fused": FUSED_ADAMW_SOURCE,
                        "sgdm": SGDM_SOURCE, "adam": ADAM_SOURCE, "sgd": SGD_SOURCE,
                        "lion": LION_SOURCE, "schedule_free": SCHEDULE_FREE_SOURCE,
                        "prodigy": PRODIGY_SOURCE, "muon": MUON_SOURCE}


def sweep_verdict(runs: list) -> dict:
    """Interior-LR-optimum verdict for a reference sweep (DESIGN §3 rule 3 / S1).

    `runs` are sweep records with at least {lr, final_loss, failed}. Returns
    the summary block the sweep worker writes: per-LR mean final loss across
    seeds, best_lr, interior_optimum, and boundary_winner ("low"/"high") when
    the mandatory extend-on-boundary rule fires. A reference whose verdict is
    not interior_optimum=True must never be served as the bar for its task.
    """
    by_lr: dict = {}
    for r in runs:
        if not r.get("failed"):
            by_lr.setdefault(r["lr"], []).append(r["final_loss"])
    grid = sorted(by_lr)
    means = {lr: sum(v) / len(v) for lr, v in by_lr.items()}
    best = min(means, key=means.get) if means else None
    interior = best is not None and bool(grid) and best not in (grid[0], grid[-1])
    return dict(per_lr_mean_final={f"{lr:g}": means[lr] for lr in grid},
                per_lr_n={f"{lr:g}": len(by_lr[lr]) for lr in grid},
                best_lr=best, interior_optimum=interior,
                boundary_winner=None if interior or best is None else
                ("low" if best == grid[0] else "high"))


def extract_hparams(source: str, task_config: dict = None) -> dict:
    """Extract HPARAMS dict from optimizer source via AST (no execution).
    Returns {"lr": float, "weight_decay": float}. Safe: uses ast.literal_eval.

    Supports per-task overrides in HPARAMS:
        HPARAMS = {"lr": 1e-3, "weight_decay": 0.01,
                   "pretrained": {"lr": 3e-5},
                   "scratch": {"lr": 1e-3}}
    If task_config has use_pretrained=True, "pretrained" overrides merge on top.
    If use_pretrained=False, "scratch" overrides merge on top.
    """
    try:
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name) and t.id == "HPARAMS":
                    hp = ast.literal_eval(node.value)
                    if isinstance(hp, dict) and "lr" in hp:
                        if task_config:
                            key = "pretrained" if task_config.get("use_pretrained") else "scratch"
                            overrides = hp.get(key, {})
                            if isinstance(overrides, dict):
                                result = {k: v for k, v in hp.items()
                                          if k not in ("pretrained", "scratch")}
                                result.update(overrides)
                                return result
                        return {k: v for k, v in hp.items()
                                if k not in ("pretrained", "scratch")}
    except Exception:
        pass
    return {"lr": 1e-3, "weight_decay": 0.01}


def extract_capabilities(source: str) -> dict:
    """Extract optional CAPABILITIES metadata via AST, without executing source.

    Supported forms:
        CAPABILITIES = {"requires_param_values": False,
                        "trusted_weight_decay": True}
        CAPABILITIES = {"param_values": "never"}
        CAPABILITIES = {"param_values_interval": 16}
        CAPABILITIES = {"returns_full_update": True,
                        "ordered_buffers": True}

    Defaults preserve the legacy contract: full ``param_values`` every step and
    optimizer-owned weight decay.
    """
    caps = {
        "requires_param_values": True,
        "param_values_interval": 1,
        "trusted_weight_decay": False,
        "returns_full_update": False,
        "ordered_buffers": False,
    }
    try:
        tree = ast.parse(source)
        raw = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name) and t.id == "CAPABILITIES":
                    raw = ast.literal_eval(node.value)
                    break
            if isinstance(node, ast.AnnAssign):
                t = node.target
                if isinstance(t, ast.Name) and t.id == "CAPABILITIES" and node.value is not None:
                    raw = ast.literal_eval(node.value)
                    break
        if not isinstance(raw, dict):
            return caps

        mode = raw.get("param_values", None)
        if isinstance(mode, str):
            m = mode.lower().strip()
            if m in ("never", "none", "off", "false"):
                caps["requires_param_values"] = False
                caps["param_values_interval"] = 0
            elif m in ("always", "every_step", "true"):
                caps["requires_param_values"] = True
                caps["param_values_interval"] = 1

        if "requires_param_values" in raw:
            req = bool(raw["requires_param_values"])
            caps["requires_param_values"] = req
            if not req:
                caps["param_values_interval"] = 0

        if "param_values_interval" in raw:
            interval = int(raw["param_values_interval"])
            caps["param_values_interval"] = max(0, interval)
            caps["requires_param_values"] = interval != 0

        if "trusted_weight_decay" in raw:
            caps["trusted_weight_decay"] = bool(raw["trusted_weight_decay"])

        if "returns_full_update" in raw:
            caps["returns_full_update"] = bool(raw["returns_full_update"])

        if "ordered_buffers" in raw:
            caps["ordered_buffers"] = bool(raw["ordered_buffers"])
    except Exception:
        return caps
    return caps
