# KAIROS-B -- bucketed Muon + fp32-master error feedback + truncation-hedged 1-sqrt cooldown.
# SN125 production task FW2_smol_360M (lean-llama-360m, seq 2048, batch 32, 188k steps / 72,000 s).
#
# METHOD (three levers, in order of value)
#
# 1. MUON ON THE HIDDEN MATRICES, RUN AS FOUR BATCHED bmm CHAINS.
#    lean_model.py yields exactly 290 tensors: 224 hidden matrices in four distinct
#    shapes -- (960,960)x64 [q,o], (320,960)x64 [k,v], (2560,960)x64 [gate,up],
#    (960,2560)x32 [down] = 314.57M params -- plus the tied embed_tokens.weight
#    (49152,960) and 65 RMSNorm gains. Each shape is stacked into one contiguous
#    (k,r,c) buffer, so the whole Newton-Schulz orthogonalization is 4 x 5 x 3 = 60
#    batched matmuls instead of ~3400 per-matrix launches. Tall buckets use the
#    algebraically identical transposed iteration (A = X^T X ; X <- aX + XB, B
#    symmetric), which cuBLAS handles as a transpose flag: no transpose copy, and the
#    FLOP count stays the min-dimension one (7.2 TFLOP/step total, not 10.1).
#    The tied embedding is excluded from orthogonalization by BOTH name and aspect
#    ratio (49152/960 = 51.2 > 16) -- flattening its singular values would destroy the
#    token-frequency scaling the unembedding logits depend on.
#
# 2. fp32 MASTER WEIGHTS WITH EXACT bf16 ERROR FEEDBACK.
#    training.py builds the model in pure bf16 and the harness applies
#    `param += update` in bf16 with round-to-nearest. Muon's update has a CONSTANT
#    per-element RMS of 0.2*lr*s(t); at |w| ~ 0.03 the bf16 half-ulp is ~6e-5, so any
#    update below s(t) ~ 0.15 is deterministically discarded and 87% of the model
#    silently freezes for the whole cooldown. We keep an fp32 master W, apply the
#    update and the decoupled weight decay there, and emit u = bf16(W) - p with p the
#    freshly read param_values. By Sterbenz's lemma that subtraction is exact whenever
#    p and bf16(W) are within a factor of two, so p + u lands exactly on bf16(W) for
#    the elements this is aimed at; where the master has moved further than that (an
#    element near a sign change) p + u is still within one bf16 ulp of bf16(W) and the
#    residue is picked up on the next step. Because p is re-read
#    EVERY step (param_values_interval = 1) the scheme is stateless and
#    self-correcting -- no shadow tensor, no drift, no resync bookkeeping, and no
#    exposure to the empty-param_values list the harness hands out when the interval
#    is > 1 (engine/optproc.py: `pvals = param_list if param_values_available else []`).
#
# 3. A SCHEDULE PLACED FOR THE CLOCK, NOT FOR total_steps.
#    There is NO clock inside the sandbox: `time` is not importable and
#    torch.Event / torch.Stream / torch.cuda.Event are stripped by the safe-torch
#    proxy. So the schedule cannot adapt -- it must be shaped to lose gracefully.
#    188,000 was calibrated as 72,000 s / reference-AdamW iteration time, which
#    budgets nothing for ~20 in-loop evals (256 forward batches each), quartile
#    checkpoint exports or compile warmup. A ~+2% optimizer realistically completes
#    ~180k steps, so the cooldown is keyed to H = 0.957 * total_steps and uses the
#    1-sqrt shape, which does most of its LR reduction EARLY in the window. Realised
#    LR at plausible cut points (T = 188,000):
#        cut 188.0k -> 0.020    cut 176k -> 0.051    cut 168k -> 0.118
#        cut 180.0k -> 0.020    cut 172k -> 0.084    cut 160k -> 0.189
#    The reference WSD (constant 80%, cosine over the last 20%) is still at 0.55 of
#    peak when cut at 168k. Overshoot costs only ~8k steps at a 2% floor.
#
# HYPERPARAMETERS
#    lr 2.0e-3 (matrices, Moonlight RMS-matched via 0.2*sqrt(max(r,c)) so the
#      per-element update RMS is exactly 0.2*lr = 4.0e-4 for every shape and the
#      number is directly comparable to an AdamW lr); Adam group at 0.30x = 6.0e-4.
#    weight_decay 0.04 on matrices only, decoupled, scaled by s(t), applied to the
#      fp32 master. Equilibrium weight RMS = sqrt(0.04*lr/(2*wd)) = 0.032, i.e. 1.6x
#      the 0.02 init std -- Muon's constant-RMS updates grow weight norms as sqrt(t)
#      and a collapsing effective LR late in the run is the documented failure mode.
#      Zero decay on the tied embedding (a ~10k-step relaxation time erases rare-token
#      rows) and on the norm gains.
#    Muon momentum 0.95 with Nesterov look-ahead, ramped to 0.98 across the cooldown
#      (free variance reduction exactly where the final weights are decided; NS is
#      scale-invariant so only the averaging window changes).
#    Newton-Schulz: 5 quintic iterations, coefficients (3.4445, -4.7750, 2.0315), in
#      bf16, on X pre-normalized by 1.01*||X||_F so sigma_max sits strictly inside the
#      contractive region. A per-matrix Frobenius cap at 1.3*sqrt(min(r,c)) follows;
#      it is inert in normal operation and costs one reduction, but the harness only
#      checks update finiteness every 128 steps and one unbounded step loses the round.
#    Adam path (tied embedding + 65 gains): fp32 m/v/master, betas (0.9, 0.99),
#      eps 1e-8. fp32 matters: the reference allocates m and v in the PARAMETER dtype,
#      and in bf16 `v.mul_(0.999)` is a bit-exact no-op (0.1% step vs a 0.195%
#      half-ulp), so the baseline's second moment is a monotone ratchet and its
#      nominal 1e-3 has been behaving like ~3e-4. eps 1e-8 rather than 1e-12 because
#      rare-token embedding rows see exactly-zero gradients for long stretches.
#    Warmup 1500 steps linear (not the config's 9400): gradients arrive pre-clipped
#      and Muon's update is scale-conditioned from step one; the ramp only exists to
#      cover the Adam bias-correction transient on the tied embedding.
#
# CAPABILITIES / INTERFACE NOTES
#    trusted_weight_decay is deliberately False: the harness applies it as a separate
#    p.mul_(1 - lr*wd), which at ~8e-5 relative is ~0.01 of a bf16 half-ulp -- a
#    complete no-op on this substrate. Decay belongs on the fp32 master.
#    step_ordered returns None with returns_full_update=True, which is legal in both
#    drivers and means updates are written in place into the shared grad buffers: no
#    dict, no name list over IPC, no allocation. Every one of the 290 buffers is
#    written by the two closing foreach calls. step() is a real wrapper over
#    step_ordered so `validate-file` and the non-ordered path behave identically.
#    Every op used (bmm, baddbmm, mul_/add_/div_, vector_norm, and only the thirteen
#    whitelisted torch._foreach_* entries) has a deterministic CUDA kernel, which is
#    required because the optproc child forces
#    torch.use_deterministic_algorithms(True, warn_only=False) before loading source;
#    CUBLAS_WORKSPACE_CONFIG=":4096:8" is already set by training.py at import.
#
# FIRST THINGS TO TUNE: _HORIZON_FRAC from a measured steps/s smoke run, then lr.

HPARAMS = {
    "lr": 2.0e-3,
    "weight_decay": 0.04,
    "pretrained": {"lr": 2.0e-4},
    "scratch": {"lr": 2.0e-3},
}

CAPABILITIES = {
    "requires_param_values": True,
    "param_values_interval": 1,
    "trusted_weight_decay": False,
    "returns_full_update": True,
    "ordered_buffers": True,
}

import torch
import math

# --- Newton-Schulz quintic (modded-nanogpt coefficients) ---------------------
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315
_NS_ITERS = 5
_NS_PRENORM = 1.01          # >1 so bf16 rounding cannot push sigma_max above 1
_NS_CAP = 1.30              # per-matrix ||.||_F cap, in units of sqrt(min(r,c))

# --- Muon --------------------------------------------------------------------
_MOM = 0.95
_MOM_END = 0.98
_RMS_TARGET = 0.2           # per-element update RMS = _RMS_TARGET * lr * s(t)
_MIN_DIM = 32
_MAX_ASPECT = 16.0

# --- Adam group (tied embedding + RMSNorm gains) -----------------------------
_ADAM_MULT = 0.30
_B1 = 0.9
_B2 = 0.99
_EPS = 1e-8

# --- Schedule ----------------------------------------------------------------
_WARMUP = 1500
_HORIZON_FRAC = 0.957       # anneal lands here, not at total_steps
_COOL_FRAC = 0.33           # cooldown width as a fraction of the horizon
_FLOOR = 0.02


class Optimizer:

    def __init__(self, param_groups, config):
        total = int(config.get("total_steps", 0) or 0)
        if total <= 0:
            total = 188000
        self.total_steps = total
        self.warmup = min(_WARMUP, max(4, total // 20))

        horizon = _HORIZON_FRAC * float(total)
        cool_start = horizon * (1.0 - _COOL_FRAC)
        if cool_start < float(self.warmup) + 1.0:
            cool_start = float(self.warmup) + 1.0
        if horizon <= cool_start:
            horizon = cool_start + 1.0
        self.cool_start = cool_start
        self.cool_end = horizon
        self.cool_span = horizon - cool_start

        base_lr = 2.0e-3
        base_wd = 0.04
        specs = []
        # The harness builds exactly one group (training.py: param_groups =
        # [{"params": [...], "lr": lr, "weight_decay": weight_decay}]). Read
        # lr/wd with an explicit None test rather than `or`, so a group that
        # deliberately asks for 0.0 is honoured instead of silently falling
        # back to the default.
        for pg in param_groups:
            _v = pg.get("lr", None)
            if _v is not None:
                base_lr = float(_v)
            _v = pg.get("weight_decay", None)
            if _v is not None:
                base_wd = float(_v)
            for name, shape, dtype in pg["params"]:
                specs.append((name, tuple(shape), dtype))
        self.base_lr = base_lr
        self.wd = base_wd
        self.names = [sp[0] for sp in specs]
        n = len(specs)
        self.n = n

        dev = config.get("_device", "") or "cuda"

        # --- classify: 2-D, not too thin, not too eccentric, not the embedding
        buckets = {}
        order = []
        adam_idx = []
        for i in range(n):
            name = specs[i][0]
            shape = specs[i][1]
            is_mat = False
            if len(shape) == 2:
                lo = min(shape[0], shape[1])
                hi = max(shape[0], shape[1])
                is_mat = (lo >= _MIN_DIM
                          and float(hi) <= _MAX_ASPECT * float(lo)
                          and "embed" not in name
                          and "head" not in name
                          and "tok" not in name)
            if is_mat:
                if shape in buckets:
                    buckets[shape].append(i)
                else:
                    buckets[shape] = [i]
                    order.append(shape)
            else:
                adam_idx.append(i)

        master = [None] * n
        self.bk = []
        for shape in order:
            idxs = buckets[shape]
            k = len(idxs)
            r = shape[0]
            c = shape[1]
            d = min(r, c)
            xb = torch.zeros((k, r, c), dtype=torch.bfloat16, device=dev)
            yb = torch.zeros((k, r, c), dtype=torch.bfloat16, device=dev)
            mb = torch.zeros((k, r, c), dtype=torch.bfloat16, device=dev)
            ab = torch.zeros((k, d, d), dtype=torch.bfloat16, device=dev)
            bb = torch.zeros((k, d, d), dtype=torch.bfloat16, device=dev)
            wb = torch.zeros((k, r, c), dtype=torch.float32, device=dev)
            for j in range(k):
                master[idxs[j]] = wb[j]
            self.bk.append({
                "idx": idxs,
                "x": xb, "y": yb, "m": mb, "a": ab, "b": bb, "w": wb,
                "xv": [xb[j] for j in range(k)],
                "tall": r > c,
                "scale": _RMS_TARGET * math.sqrt(float(max(r, c))),
                "inv_cap": 1.0 / (_NS_CAP * math.sqrt(float(d))),
            })

        self.adam_idx = adam_idx
        self.ag = []
        self.am = []
        self.av = []
        self.aw = []
        for i in adam_idx:
            shape = specs[i][1]
            self.ag.append(torch.zeros(shape, dtype=torch.float32, device=dev))
            self.am.append(torch.zeros(shape, dtype=torch.float32, device=dev))
            self.av.append(torch.zeros(shape, dtype=torch.float32, device=dev))
            w = torch.zeros(shape, dtype=torch.float32, device=dev)
            self.aw.append(w)
            master[i] = w
        self.master = master

        self.started = False
        self.step_count = 0

        # mixed-dtype _foreach_copy_ is the one API surprise that would be fatal
        self.fast_copy = True
        try:
            _a = torch.zeros(2, dtype=torch.bfloat16, device=dev)
            _b = torch.zeros(2, dtype=torch.float32, device=dev)
            torch._foreach_copy_([_a], [_b])
            torch._foreach_copy_([_b], [_a])
        except Exception:
            self.fast_copy = False

    # ------------------------------------------------------------------ sched
    def _sched(self, t):
        if t < self.warmup:
            return (t + 1.0) / float(self.warmup)
        if t >= self.cool_end:
            return _FLOOR
        if t <= self.cool_start:
            return 1.0
        x = (t - self.cool_start) / self.cool_span
        return _FLOOR + (1.0 - _FLOOR) * (1.0 - math.sqrt(x))

    def _mom(self, t):
        if t <= self.cool_start:
            return _MOM
        x = (t - self.cool_start) / self.cool_span
        if x > 1.0:
            x = 1.0
        return _MOM + (_MOM_END - _MOM) * x

    # ----------------------------------------------------------------- helpers
    def _copy_into(self, dst, src):
        if self.fast_copy:
            torch._foreach_copy_(dst, src)
        else:
            for i in range(len(dst)):
                dst[i].copy_(src[i])

    # -------------------------------------------------------------------- core
    def _core(self, grads, pvals, step_number):
        n = self.n
        have_p = len(pvals) == n
        if not have_p:
            # Cannot maintain the master exactly without the live weights.
            # Unreachable with param_values_interval == 1; a no-op step keeps
            # master and model in sync rather than silently desynchronising.
            torch._foreach_mul_(grads, 0.0)
            return
        if not self.started:
            self._copy_into(self.master, pvals)
            self.started = True

        self.step_count = step_number + 1
        s = self._sched(step_number)
        mu = self._mom(step_number)
        lr_m = self.base_lr * s
        lr_a = self.base_lr * _ADAM_MULT * s

        # ---- Muon: four batched Newton-Schulz chains ------------------------
        for bk in self.bk:
            idxs = bk["idx"]
            self._copy_into(bk["xv"], [grads[i] for i in idxs])
            X = bk["x"]
            M = bk["m"]
            M.mul_(mu).add_(X, alpha=1.0 - mu)          # EMA momentum (bf16)
            X.mul_(1.0 - mu).add_(M, alpha=mu)          # Nesterov look-ahead

            nrm = torch.linalg.vector_norm(X, dim=(1, 2), keepdim=True,
                                           dtype=torch.float32)
            X.div_(nrm.mul_(_NS_PRENORM).add_(1e-7).to(torch.bfloat16))

            Y = bk["y"]
            A = bk["a"]
            B = bk["b"]
            if bk["tall"]:
                for _ in range(_NS_ITERS):
                    torch.bmm(X.transpose(1, 2), X, out=A)
                    torch.baddbmm(A, A, A, beta=_NS_B, alpha=_NS_C, out=B)
                    torch.baddbmm(X, X, B, beta=_NS_A, alpha=1.0, out=Y)
                    X, Y = Y, X
            else:
                for _ in range(_NS_ITERS):
                    torch.bmm(X, X.transpose(1, 2), out=A)
                    torch.baddbmm(A, A, A, beta=_NS_B, alpha=_NS_C, out=B)
                    torch.baddbmm(X, B, X, beta=_NS_A, alpha=1.0, out=Y)
                    X, Y = Y, X

            nrm = torch.linalg.vector_norm(X, dim=(1, 2), keepdim=True,
                                           dtype=torch.float32)
            X.div_(nrm.mul_(bk["inv_cap"]).clamp_min_(1.0).to(torch.bfloat16))

            W = bk["w"]
            if self.wd > 0.0:
                W.mul_(1.0 - lr_m * self.wd)
            W.add_(X, alpha=-lr_m * bk["scale"])

        # ---- AdamW on the tied embedding and the RMSNorm gains --------------
        if self.adam_idx:
            self._copy_into(self.ag, [grads[i] for i in self.adam_idx])
            torch._foreach_mul_(self.am, _B1)
            torch._foreach_add_(self.am, self.ag, alpha=1.0 - _B1)
            torch._foreach_mul_(self.av, _B2)
            torch._foreach_addcmul_(self.av, self.ag, self.ag, value=1.0 - _B2)
            t1 = step_number + 1
            bc1 = 1.0 - _B1 ** t1
            bc2 = 1.0 - _B2 ** t1
            den = torch._foreach_sqrt(self.av)
            torch._foreach_mul_(den, 1.0 / math.sqrt(bc2))
            torch._foreach_add_(den, _EPS)
            torch._foreach_addcdiv_(self.aw, self.am, den, value=-lr_a / bc1)

        # ---- exact bf16 emission: u = bf16(master) - p ----------------------
        self._copy_into(grads, self.master)
        torch._foreach_sub_(grads, pvals)

    # --------------------------------------------------------------- interface
    def step_ordered(self, gradients, param_values, step_number):
        self._core(gradients, param_values, step_number)
        return None

    def step(self, gradients, param_values, step_number):
        names = self.names
        glist = [gradients[nm] for nm in names]
        plist = []
        if param_values:
            plist = [param_values[nm] for nm in names]
        self._core(glist, plist, step_number)
        out = {}
        for i in range(self.n):
            out[names[i]] = glist[i]
        return out

    def state_dict(self):
        return {
            "step": self.step_count,
            "started": 1 if self.started else 0,
            "mom": [bk["m"].clone() for bk in self.bk],
            "wmat": [bk["w"].clone() for bk in self.bk],
            "am": [x.clone() for x in self.am],
            "av": [x.clone() for x in self.av],
            "aw": [x.clone() for x in self.aw],
        }

    def load_state_dict(self, state):
        self.step_count = int(state.get("step", 0) or 0)
        self.started = bool(int(state.get("started", 1) or 0))
        mom = state.get("mom", [])
        for i in range(min(len(mom), len(self.bk))):
            self.bk[i]["m"].copy_(mom[i])
        wmat = state.get("wmat", [])
        for i in range(min(len(wmat), len(self.bk))):
            self.bk[i]["w"].copy_(wmat[i])
        for key in ("am", "av", "aw"):
            src = state.get(key, [])
            dst = self.am if key == "am" else (self.av if key == "av" else self.aw)
            for i in range(min(len(src), len(dst))):
                dst[i].copy_(src[i])
