"""Per-box throughput probe — the hardware gate for multi-provider fairness.

The production eval is a fixed WALL-CLOCK budget run (SPEC §4.2), so box speed
converts directly into achievable steps and therefore loss. Miners own their
LR schedules and plan them against ``total_steps``; the platform's side of that
contract is that any box we score on can actually deliver ``total_steps``
inside the budget. GPU class alone does not guarantee this: B200s across
providers (and across hosts within one provider) differ in power/thermal
headroom, driver, and neighbor load — the calibration history in
sn125/fineweb.py shows a 3.37 → 2.75 st/s spread between a bare probe box and
the genesis box.

This probe measures the box, never the miner: a fixed synthetic-data workload
at the production shape (lean 360M, seq 2048, batch 32, bf16, the production
compile knobs) with a reference foreach-AdamW step that stages param values
each step (the requires_param_values cost every real submission pays). It runs
BEFORE shard staging and miner code, takes a few minutes, and reports
steady-state steps/sec. The orchestrator compares that against the operator-
pinned throughput BAND and rejects the box (provisioning flake → fresh box)
when out of band — too slow (starves miners' schedules) OR too fast (inflates
the fixed-budget score and poisons the frontier ratchet). After a successful
eval the probe runs AGAIN on the same box: a pre-vs-post deviation beyond the
band quarantines the run (mid-run throttling / noisy neighbor / probe
sandbagging) — see cloud._drift_gate.

The band must be calibrated with THIS probe (in-process, no optproc IPC), not
with in-run harness throughput — the probe reads systematically faster than a
real eval on the same box. Pinning procedure: run `python -m sn125 box-probe`
on a known-good box of EACH provider in the failover chain, confirm the
readings agree, set SN125_BOX_PROBE_REF_STS to that reading and (optionally)
SN125_BOX_PROBE_BAND_PCT (default 5%). The legacy SN125_BOX_PROBE_MIN_STS
floor is still honored; with nothing pinned the probe is measure-only.
"""
import json
import time

RESULT_MARKER = "SN125_BOX_PROBE_RESULT"

PROBE_MODEL_CONFIG = "lean-llama-360m"
PROBE_SEQ_LEN = 2048
PROBE_BATCH_SIZE = 32


def run_box_probe(config_name: str = PROBE_MODEL_CONFIG,
                  seq_len: int = PROBE_SEQ_LEN,
                  batch_size: int = PROBE_BATCH_SIZE,
                  warmup_steps: int = 40,
                  measure_steps: int = 120,
                  seed: int = 1234) -> dict:
    """Run the standardized throughput probe on this box's GPU.

    Returns {"steps_per_s", "ms_per_step", "warmup_s", "gpu", "torch", ...}.
    Honors the same env knobs as the production eval (SN125_LEAN_COMPILE,
    SN125_LEAN_LOSS_CKPT, SN125_FORCE_GC) because they change throughput;
    warmup absorbs inductor compile time.
    """
    import torch
    from .training import _build_model

    if not torch.cuda.is_available():
        raise RuntimeError("box probe requires a CUDA GPU")
    device = torch.device("cuda")
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    model = _build_model(config_name, seed, dtype=torch.bfloat16)
    vocab = int(model.config.vocab_size)
    tokens = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    m_list = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    v_list = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    beta1, beta2, eps, lr, wd = 0.9, 0.999, 1e-8, 1e-3, 0.01

    def step(t: int):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, labels=tokens).loss
        loss.backward()
        grads = [p.grad.float() for p in params]
        staged = [p.detach().float().clone() for p in params]
        torch._foreach_mul_(m_list, beta1)
        torch._foreach_add_(m_list, grads, alpha=1 - beta1)
        torch._foreach_mul_(v_list, beta2)
        torch._foreach_addcmul_(v_list, grads, grads, value=1 - beta2)
        m_hat = torch._foreach_div(m_list, 1 - beta1 ** t)
        v_hat = torch._foreach_div(v_list, 1 - beta2 ** t)
        denom = torch._foreach_sqrt(v_hat)
        torch._foreach_add_(denom, eps)
        updates = torch._foreach_div(m_hat, denom)
        torch._foreach_mul_(updates, -lr)
        torch._foreach_add_(updates, staged, alpha=-lr * wd)
        with torch.no_grad():
            for p, u in zip(params, updates):
                p.add_(u.to(p.dtype))
        model.zero_grad(set_to_none=True)

    t0 = time.time()
    for i in range(warmup_steps):
        step(i + 1)
    torch.cuda.synchronize()
    warmup_s = time.time() - t0

    t1 = time.time()
    for i in range(measure_steps):
        step(warmup_steps + i + 1)
    torch.cuda.synchronize()
    measure_s = time.time() - t1

    steps_per_s = measure_steps / measure_s
    return {
        "steps_per_s": round(steps_per_s, 4),
        "ms_per_step": round(1000.0 * measure_s / measure_steps, 2),
        "warmup_s": round(warmup_s, 1),
        "measure_s": round(measure_s, 1),
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "config": config_name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


def format_result(result: dict) -> str:
    """The single stdout line the orchestrator parses."""
    return f"{RESULT_MARKER} {json.dumps(result, sort_keys=True)}"


def parse_result(stdout: str) -> dict | None:
    """Extract the probe result dict from remote stdout (last marker wins)."""
    found = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(RESULT_MARKER):
            try:
                found = json.loads(line[len(RESULT_MARKER):].strip())
            except json.JSONDecodeError:
                continue
    return found if isinstance(found, dict) else None
