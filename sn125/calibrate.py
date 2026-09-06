"""SN125 §3 calibration throughput probe.

Empirically pins (model, total_steps N, batch_size, sequence_length) so the
reference AdamW does a complete, fully-decayed Chinchilla-optimal run in
``target_hours`` on one ``b200-small``. Runs the reference for a short burst of
steps, measures the median per-step wall time, then derives:

  * N            = floor(target_seconds / iteration_time)        (SPEC §3.3)
                   where iteration_time = FULL per-step wall (fwd+bwd+harness+opt),
                   NOT the opt.step()-only time (which is ~36× smaller and would
                   inflate N ~36×). opt.step() time is reported separately.
  * tokens       = N * batch * seq
  * chinchilla   = tokens / (20 * params)   (want ≈ 1.0)
  * mfu          = achieved_flops / peak_flops  (6N-per-token estimate)

The GPU run path (:func:`run_probe`) reuses the production
``_build_model``/``_generate_data``/``train_and_eval`` so the timing reflects the
real harness (WSD schedule, optimizer-process isolation, per-step empty_cache).
The pure helpers below are device-free and unit-tested on CPU.

Cost: ~``probe_steps`` reference steps ≈ minutes ≈ a few dollars on b200-small.
Prints a single ``CALIBRATION_RESULT {json}`` line the cloud launcher parses.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

B200_BF16_DENSE_PEAK_FLOPS = 1.1e15


def median_step_time(step_times: list[float], drop_warmup: int = 20) -> float:
    """Median per-step wall seconds, discarding the first ``drop_warmup`` steps
    (CUDA graph capture, autotune, allocator warmup, first-touch page faults)."""
    usable = step_times[drop_warmup:] if len(step_times) > drop_warmup else step_times
    if not usable:
        raise ValueError("no step times to summarize")
    return statistics.median(usable)


def compute_n(step_time: float, target_seconds: float) -> int:
    """SPEC §3.3: N = floor(target_seconds / step_time)."""
    if step_time <= 0:
        raise ValueError(f"step_time must be > 0, got {step_time}")
    return int(target_seconds // step_time)


def chinchilla_ratio(n_steps: int, batch_size: int, seq_len: int, n_params: int) -> tuple[int, float]:
    """Return (total_tokens, tokens/(20*params)). Chinchilla-optimal ⇒ ratio ≈ 1."""
    tokens = n_steps * batch_size * seq_len
    ratio = tokens / (20.0 * n_params) if n_params else 0.0
    return tokens, ratio


def estimate_mfu(step_time: float, n_params: int, batch_size: int, seq_len: int,
                 peak_flops: float = B200_BF16_DENSE_PEAK_FLOPS) -> float:
    """Model FLOPs utilization from the standard 6N-per-token training estimate."""
    tokens_per_step = batch_size * seq_len
    flops_per_step = 6.0 * n_params * tokens_per_step
    achieved = flops_per_step / step_time if step_time > 0 else 0.0
    return achieved / peak_flops if peak_flops else 0.0


def summarize(step_times: list[float], n_params: int, batch_size: int, seq_len: int,
              target_hours: float, drop_warmup: int = 20,
              peak_flops: float = B200_BF16_DENSE_PEAK_FLOPS,
              iter_times: list[float] | None = None) -> dict:
    """Pure: turn measured step times + config into the full calibration verdict.

    ``iter_times`` (FULL end-to-end per-iteration wall seconds) is the correct N
    driver — it includes forward/backward, the harness's per-step param cloning +
    empty_cache, and the optimizer step. ``step_times`` measures only ``opt.step()``
    and is reported as telemetry. When ``iter_times`` is absent we fall back to
    ``step_times`` (keeps the pure unit tests / legacy callers working), but the GPU
    probe ALWAYS passes iter_times — using opt-only time inflated N ~36× the first
    time and made a "20h" run actually ~21k steps, not 763k.
    """
    target_seconds = target_hours * 3600.0
    opt_st = median_step_time(step_times, drop_warmup)
    if iter_times:
        st = median_step_time(iter_times, drop_warmup)
        n_measured = len(iter_times)
    else:
        st = opt_st
        n_measured = len(step_times)
    n = compute_n(st, target_seconds)
    tokens, ratio = chinchilla_ratio(n, batch_size, seq_len, n_params)
    mfu = estimate_mfu(st, n_params, batch_size, seq_len, peak_flops)
    return {
        "median_step_time_s": round(st, 5),
        "iteration_time_s": round(st, 5),
        "optimizer_step_time_s": round(opt_st, 5),
        "measured_steps": n_measured,
        "dropped_warmup": min(drop_warmup, n_measured),
        "params": n_params,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": batch_size * seq_len,
        "target_hours": target_hours,
        "computed_N": n,
        "projected_tokens": tokens,
        "chinchilla_ratio": round(ratio, 4),
        "mfu": round(mfu, 4),
        "peak_flops": peak_flops,
    }


def run_probe(model_config: str, seq_len: int, batch_size: int, probe_steps: int,
              warmup_steps: int = 100, drop_warmup: int = 20, use_amp: bool = True,
              seed: int = 42, target_hours: float = 20.0,
              empty_cache_every: int = 0,
              gradient_checkpointing: bool | None = None,
              compile_model: bool = False,
              chunked_ce: int = 0,
              fp8: bool = False,
              peak_flops: float = B200_BF16_DENSE_PEAK_FLOPS) -> dict:
    """GPU path: build the candidate model, run the AdamW reference for
    ``probe_steps`` steps, and summarize. Requires CUDA (the production builders
    hardcode ``.cuda()``).

    DIAGNOSTIC throughput levers (do NOT change scoring defaults; probe-only):
    ``gradient_checkpointing`` — None mirrors production (``use_amp or seq>=1024``);
      set False to measure the recompute-drop speedup once flash leaves memory
      headroom, or True to force it on.
    ``compile_model`` — wrap the model in ``torch.compile`` (kernel fusion) before
      timing. Compile touches only the in-process forward/backward; the miner's
      optimizer still runs isolated over CUDA IPC, so the security boundary is
      unaffected. Build cost (graph capture) is absorbed by ``drop_warmup``."""
    import torch
    from contextlib import nullcontext
    from .training import _build_model, _generate_data, train_and_eval
    from .references import ADAMW_SOURCE
    from .sandbox import load_optimizer_sandboxed, rebaseline_torch_identities

    if not torch.cuda.is_available():
        raise RuntimeError("run_probe requires CUDA — the throughput numbers are GPU-specific")

    t_build = time.time()
    dtype = torch.bfloat16 if use_amp else None
    flash_on = bool(os.environ.get("SN125_DIAG_FLASH"))
    gc_ckpt = ((use_amp or seq_len >= 1024) and not flash_on) \
        if gradient_checkpointing is None else gradient_checkpointing
    model = _build_model(model_config, seed=seed, dtype=dtype,
                         gradient_checkpointing=gc_ckpt, fp8=fp8)
    n_params = sum(p.numel() for p in model.parameters())
    if compile_model:
        model = torch.compile(model)
        _warm = torch.randint(0, model.config.vocab_size, (batch_size, seq_len),
                              device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
            _wo = model(_warm, labels=_warm)
        _wo.loss.backward()
        model.zero_grad(set_to_none=True)
        del _warm, _wo
        torch.cuda.synchronize()
        rebaseline_torch_identities()
    n_batches = min(probe_steps + 1, 64)
    train_data = _generate_data(batch_size, seq_len, model.config.vocab_size, n_batches, seed)
    eval_data = _generate_data(batch_size, seq_len, model.config.vocab_size, 2, seed + 1)
    optimizer_cls = load_optimizer_sandboxed(ADAMW_SOURCE)
    build_s = time.time() - t_build

    curve = train_and_eval(
        model, train_data, eval_data, optimizer_cls,
        lr=3e-4, weight_decay=0.01, total_steps=probe_steps,
        eval_every=max(probe_steps, 1), warmup_steps=warmup_steps,
        max_step_time=120.0, use_amp=use_amp,
        empty_cache_every=empty_cache_every,
        max_total_time=probe_steps * 120.0 + 600.0,
        chunked_ce=chunked_ce,
    )
    if curve.failed:
        raise RuntimeError(f"probe reference run failed: {curve.error}")

    out = summarize(curve.step_times, n_params, batch_size, seq_len,
                    target_hours, drop_warmup, peak_flops,
                    iter_times=curve.iter_times)
    out.update({"model_config": model_config, "use_amp": use_amp,
                "gradient_checkpointing": gc_ckpt, "compiled": compile_model,
                "chunked_ce": chunked_ce, "fp8": fp8,
                "flash": flash_on,
                "build_seconds": round(build_s, 2),
                "gpu_name": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="SN125 §3 throughput calibration probe")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200, help="probe steps to time")
    ap.add_argument("--warmup", type=int, default=100, help="WSD warmup steps")
    ap.add_argument("--drop-warmup", type=int, default=20, help="leading steps dropped from median")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--target-hours", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--empty-cache-every", type=int, default=0,
                    help="per-step empty_cache() cadence (0=off/boundary-only)")
    gc_grp = ap.add_mutually_exclusive_group()
    gc_grp.add_argument("--no-gc", dest="gc", action="store_false", default=None,
                        help="DIAG: force gradient checkpointing OFF (measure recompute-drop)")
    gc_grp.add_argument("--gc", dest="gc", action="store_true", default=None,
                        help="DIAG: force gradient checkpointing ON")
    ap.add_argument("--compile", dest="compile_model", action="store_true",
                    help="DIAG: torch.compile the model before timing (kernel fusion)")
    ap.add_argument("--chunked-ce", dest="chunked_ce", type=int, default=0, metavar="N",
                    help="DIAG: tile cross-entropy into N chunks to free the fp32-logits "
                         "materialization (enables larger batch with no_gc); 0 = stock HF loss")
    ap.add_argument("--fp8", dest="fp8", action="store_true",
                    help="DIAG (Path C): swap Linear layers to torchao Float8 training "
                         "(master weights stay bf16; pairs with --compile to realize throughput)")
    ap.add_argument("--out", default="", help="also write the result JSON to this path")
    args = ap.parse_args(argv)

    result = run_probe(args.model, args.seq, args.batch, args.steps,
                       warmup_steps=args.warmup, drop_warmup=args.drop_warmup,
                       use_amp=not args.no_amp, seed=args.seed,
                       target_hours=args.target_hours,
                       empty_cache_every=args.empty_cache_every,
                       gradient_checkpointing=args.gc,
                       compile_model=args.compile_model,
                       chunked_ce=args.chunked_ce,
                       fp8=args.fp8)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("CALIBRATION_RESULT " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
