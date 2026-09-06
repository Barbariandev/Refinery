"""
SN125 Miner Template — Starter Optimizer

This is a working AdamW implementation (with the reference WSD learning-rate
schedule built in) that roughly matches the rolling baseline. Modify the update
rule in step() to beat the current frontier. That's the game.

RULES:
  - Your file must define HPARAMS (dict) and class Optimizer at module level.
  - Optional: define CAPABILITIES (dict) to request less per-step data.
  - Allowed imports: torch, math, dataclasses, collections, functools,
    itertools, triton (incl. triton.language), typing, enum, abc. Nothing else.
  - `random` is NOT allowed: the stdlib RNG is not part of the torch/CUDA RNG
    state the audit replay restores, so it injects non-reproducible
    nondeterminism (a DQ). For stochastic updates use torch's RNG
    (torch.rand* / torch.Generator), which IS snapshotted.
  - No exec/eval/compile/importlib/subprocess/os/sys/socket/open.
  - No getattr/setattr/delattr/hasattr/type/print/super.
  - No accessing frame objects, code objects, or class internals via dunders.
  - Max source size: 1 MB total (gate.MAX_TOTAL_BYTES). Source bytes also add
    a tiny complexity tie-breaker penalty (~0.002 x bytes/1MB) — irrelevant
    unless you ship megabytes of constants, which the gate flags anyway.
  - Memory: optimizer state must fit the harness memory cap on the eval GPU
    alongside the model and activations; state heavier than the baseline
    optimizer's also costs a small relative score penalty (~0.01 per doubling
    vs baseline) and, far more importantly, per-step wall-clock inside your
    fixed budget.
  - The learning-rate schedule is YOURS. The harness applies your updates at
    scale 1.0 — no external warmup, no external decay. step() receives
    step_number every call and config carries total_steps / warmup_steps /
    decay_fraction, so you can reproduce the reference WSD shape (as this
    template does), use a different schedule, or none (schedule-free methods).

HPARAMS FORMAT:
  HPARAMS = {
      "lr": <default_lr>,
      "weight_decay": <default_wd>,
      "pretrained": {"lr": <lr_for_finetuning>},   # optional per-task overrides
      "scratch":    {"lr": <lr_for_pretraining>},   # optional per-task overrides
  }

CAPABILITIES FORMAT:
  CAPABILITIES = {
      "requires_param_values": True,   # default: current weights are provided every step
      "param_values_interval": 1,      # use 0/False or "param_values": "never" to omit them
      "trusted_weight_decay": False,   # True means harness applies decoupled weight decay
      "returns_full_update": False,    # True if every parameter gets an update every step
      "ordered_buffers": False,        # True to use optional step_ordered(list, list, step)
  }

  If you do not need current parameter tensors, set:
      CAPABILITIES = {"requires_param_values": False, "trusted_weight_decay": True}

  Then step(..., param_values, ...) receives an empty dict. Use trusted_weight_decay=True
  only if your optimizer does not also apply weight decay inside step().

  If you update every parameter, set returns_full_update=True so the validator
  does not have to send a large name list back over IPC every step. Advanced
  optimizers can also define:

      step_ordered(self, gradients, param_values, step_number) -> list[Tensor] | None

  where gradients/param_values are ordered exactly like param_groups. Returning
  None means you wrote updates in-place into the gradient buffers; only do this
  with returns_full_update=True.

OPTIMIZER INTERFACE:
  __init__(self, param_groups, config)
    param_groups: [{"params": [(name, shape, dtype), ...], "lr": float, "weight_decay": float}]
    config: {"total_steps": int, "warmup_steps": int, "decay_fraction": float,
             "max_grad_norm": float}

  step(self, gradients, param_values, step_number) -> dict[str, Tensor]
    gradients:    {param_name: gradient_tensor}     (already clipped to max_grad_norm)
    param_values: {param_name: current_param_tensor} or {} if omitted by CAPABILITIES
    step_number:  0-indexed
    returns:      {param_name: update_tensor}        (harness applies: param += update)

  state_dict(self) -> dict
  load_state_dict(self, state) -> None

SCORING (what matters):
  One production task: a lean Llama-style ~360M model (SmolLM2 tokenizer/data
  contract, built in-repo as lean-llama-360m) trained from scratch on pinned
  FineWeb-Edu shards (seq 2048, batch 32) under a FIXED wall-clock budget
  (20h on the validator's B200). Whatever loss your optimizer reaches on the
  held-out split when the clock runs out is your score — a slow step rate is
  its own penalty.

  The run also ends once `total_steps` (the pinned schedule horizon, 188k)
  have executed, so everyone who fits inside the budget trains for exactly
  the same number of steps; a slower optimizer is cut off by the clock with
  its schedule un-decayed.

  Your held-out loss is compared against the rolling best (the confirmed
  frontier). Emission goes only to frontier winners: beating the baseline by
  more than the adaptive improvement threshold (in NATS of held-out loss,
  floor 0.03) makes you a frontier CANDIDATE; the validator then re-runs your
  exact source once more at its own cost, and only if that confirmation run
  also clears the bar does it become a confirmed frontier EVENT. Events are
  what get paid: your event carries a credit equal to its improvement in
  nats that halves every 14 days; each round splits emission across every
  confirmed event pro rata to credit (no leader bonus, no winner-take-all),
  and the share of emission paid at all is min(1, total credit / reference)
  where the reference ramps from 0.03 nats at launch to 0.10 nats (one
  floor-level improvement a week keeps the subnet fully paid); the rest
  burns. The worse of your two runs is the loss the next challengers must
  beat. Everything else earns zero. Good luck.

LICENSE GRANT (submission terms):
  By submitting source code to SN125 you grant the subnet operator a
  perpetual, worldwide, non-exclusive, royalty-free license to use, evaluate,
  reproduce, publish, and redistribute the submitted source and derived
  results (scores, curves, checkpoints) under an open-source license
  (Apache-2.0 or MIT), WITH ATTRIBUTION TO YOUR SUBMITTING HOTKEY. You keep
  full ownership and may license your work to anyone else however you like.
  This is what makes the Refinery publication pipeline possible: frontier
  optimizers are published (paper + repo) credited to their hotkeys, and you
  can claim that credit under any identity — or stay pseudonymous — by
  signing an authorship proof with your hotkey:

      python -m sn125.prove_authorship sign --wallet-name W --wallet-hotkey H \\
          --source your_optimizer.py --statement "name/contact for credit"

  Anyone can verify the claim offline against the published round record. Do
  not submit code you lack the rights to license this way.
"""

HPARAMS = {
    "lr": 1e-3, "weight_decay": 0.01,
    "pretrained": {"lr": 3e-5},
    "scratch":    {"lr": 1e-3},
}

CAPABILITIES = {
    "requires_param_values": True,
    "param_values_interval": 1,
    "trusted_weight_decay": False,
    "returns_full_update": True,
    "ordered_buffers": False,
}

import torch, math

class Optimizer:
    def __init__(self, param_groups, config):
        self.param_groups = param_groups
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
        return {k: {sk: sv.clone() for sk, sv in v.items()}
                for k, v in self.state.items()}

    def load_state_dict(self, s):
        for k, v in s.items():
            if k in self.state:
                for sk, sv in v.items():
                    self.state[k][sk].copy_(sv)
