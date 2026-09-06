"""SN125 — Training harness: types, training loop, isolation, self-test."""
import ast, math, time, json, copy, os, signal, sys, tempfile, logging, itertools
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, asdict

_log = logging.getLogger("sn125.training")
import torch
import torch.nn as nn


def _progress_event(event: str, **fields) -> None:
    """Emit machine-readable progress on stdout when run telemetry is enabled.

    The sandbox parent treats these lines as observability only; scoring still
    accepts only the nonce-authenticated final JSON.
    """
    if not (os.environ.get("SN125_PROGRESS") or os.environ.get("SN125_PROGRESS_JSON")):
        return
    rec = {"event": event, "time": time.time()}
    rec.update(fields)
    try:
        print("SN125_PROGRESS_JSON " + json.dumps(rec, sort_keys=True, default=float), flush=True)
    except Exception:
        pass


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from . import settings
from .sandbox import (
    SandboxViolation, validate_source, load_optimizer_sandboxed,
    ALLOWED_IMPORTS, check_torch_tamper,
)
from .references import (
    ADAMW_SOURCE, SGDM_SOURCE, ADAM_SOURCE, SGD_SOURCE,
    LION_SOURCE, SCHEDULE_FREE_SOURCE, PRODIGY_SOURCE,
    MUON_SOURCE, FUSED_ADAMW_SOURCE, REFERENCE_OPTIMIZERS,
    extract_hparams, extract_capabilities,
)


@dataclass
class TaskSpec:
    task_id: str
    model_config: str
    parameter_count: str
    total_steps: int
    batch_size: int
    sequence_length: int
    eval_every: int
    eval_sequences: int
    warmup_fraction: float
    compute_budget_seconds: float = 0.0
    task_weight: float = 1.0
    use_pretrained: bool = False
    decay_fraction: float = 0.0
    dataset: str = "fineweb-edu"
    data_manifest: str = ""
    stop_at_total_steps: bool = False

@dataclass
class TrainingCurve:
    task_id: str
    lr: float
    wd: float
    eval_points: list[tuple[int, float, float]]
    train_points: list[tuple[int, float]]
    wall_seconds: float = 0.0
    state_multiplier: float = 0.0
    failed: bool = False
    error: str = ""
    step_times: list[float] = field(default_factory=list)
    iter_times: list[float] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)


@dataclass
class ScoreRecord:
    final_score: float
    components: dict
    task_scores: dict
    best_hparams: dict
    failed_tasks: list[str] = field(default_factory=list)


def _apply_fp8_training(model: nn.Module) -> nn.Module:
    """DIAG (Path C): swap eligible nn.Linear layers to torchao Float8 training
    (tensorwise dynamic scaling) IN PLACE.

    Isolation contract: the master weights stay ordinary bf16 ``nn.Parameter``s —
    FP8 quantization happens only on the transient GEMM inputs inside the forward
    pass. So ``named_parameters()`` (the optimizer-IPC contract: param metadata at
    build + detached bf16 grad/param clones per step) is byte-for-byte unchanged,
    and the optimizer process never sees an FP8 tensor. The swap is confined to
    this harness-owned build path; optproc.py (the CUDA-IPC boundary) is untouched.

    Throughput note: eager FP8 cast/scaling overhead largely cancels the GEMM win —
    FP8 only pays off under torch.compile (Dynamo fuses the cast into the matmul
    prologue), so the probe pairs ``--fp8`` with ``--compile``."""
    from torchao.float8 import convert_to_float8_training

    def _filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if "lm_head" in fqn:
            return False
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0

    convert_to_float8_training(model, module_filter_fn=_filter)
    return model


def _build_model(config_name: str, seed: int, dtype: torch.dtype = None,
                  use_pretrained: bool = False,
                  gradient_checkpointing: bool = False,
                  fp8: bool = False) -> nn.Module:
    """Build a transformer model for evaluation.
    use_pretrained=True: load pretrained weights (fine-tuning regime).
    use_pretrained=False: random init from config (from-scratch regime).
    gradient_checkpointing=True: trade compute for memory (needed for >2B models).
    fp8=True: DIAG — swap Linear layers to torchao Float8 training (master weights
    stay bf16; FP8 only inside the forward GEMM — see _apply_fp8_training)."""
    _ALLOWED_MODELS = {"HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M",
                        "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B", "gpt2"}
    _LOCAL_CONFIGS = {"forge-llama-1b"}
    from .lean_model import LEAN_CONFIGS, build_lean_model
    _LEAN = set(LEAN_CONFIGS)
    if config_name not in _ALLOWED_MODELS and config_name not in _LOCAL_CONFIGS and config_name not in _LEAN:
        raise ValueError(f"Model {config_name!r} not in allowed list: "
                         f"{_ALLOWED_MODELS | _LOCAL_CONFIGS | _LEAN}")
    if os.environ.get("SN125_FORCE_GC"):
        gradient_checkpointing = True
    if config_name in _LEAN:
        _lc = os.environ.get("SN125_LEAN_LOSS_CHUNKS")
        model = build_lean_model(config_name, seed, dtype=dtype,
                                 loss_chunks=int(_lc) if _lc else None)
        model.cuda()
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.train()
        return model
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(seed)
    kwargs = {}
    if dtype:
        kwargs["dtype"] = dtype
    if config_name in _LOCAL_CONFIGS:
        from transformers import LlamaConfig
        config = LlamaConfig(
            vocab_size=128256, hidden_size=2048, intermediate_size=8192,
            num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
            max_position_embeddings=8192, rope_theta=500000.0,
            tie_word_embeddings=True, attn_implementation="sdpa")
        model = AutoModelForCausalLM.from_config(config, **kwargs)
    elif use_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            config_name, trust_remote_code=True, **kwargs)
    else:
        config = AutoConfig.from_pretrained(config_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, **kwargs)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.cuda()
    if fp8:
        _apply_fp8_training(model)
    model.train()
    return model


def _generate_data(batch_size: int, seq_len: int, vocab_size: int,
                   num_batches: int, seed: int) -> list[torch.Tensor]:
    """Generate deterministic synthetic data batches."""
    rng = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=rng)
        batches.append(tokens.cuda())
    return batches


def _load_real_data(dataset_name: str, seq_len: int, batch_size: int,
                    num_batches: int, seed: int, split: str = "train",
                    tokenizer_name: str = "Qwen/Qwen3-0.6B",
                    ) -> list[torch.Tensor]:
    """Load real text data, tokenize, chunk into batches. Deterministic from seed.
    Supports: 'fineweb-edu-shards' (production: pre-tokenized hash-pinned shards,
    sn125.fineweb — `split` selects train vs heldout), 'fineweb-edu' (legacy
    streaming), 'wikitext', 'c4'.
    """
    if dataset_name == "fineweb-edu-shards":
        from .fineweb import shard_batch_sequence
        return shard_batch_sequence(None, split=split, batch_size=batch_size,
                                    num_batches=num_batches, seed=seed, seq_len=seq_len)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    import random

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    needed_tokens = num_batches * batch_size * seq_len + seq_len

    if "fineweb" in dataset_name.lower():
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        all_ids = []
        for row in ds:
            all_ids.extend(tok.encode(row["text"], add_special_tokens=False))
            if len(all_ids) >= needed_tokens:
                break
    elif "c4" in dataset_name.lower():
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        all_ids = []
        for row in ds:
            all_ids.extend(tok.encode(row["text"], add_special_tokens=False))
            if len(all_ids) >= needed_tokens:
                break
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 20]
        random.Random(seed).shuffle(texts)
        all_ids = tok.encode("\n".join(texts), add_special_tokens=False)

    if not all_ids:
        raise ValueError(f"Dataset yielded 0 tokens (dataset={dataset_name}, seed={seed})")
    if len(all_ids) < needed_tokens:
        all_ids = all_ids * (needed_tokens // len(all_ids) + 1)
    all_ids = all_ids[:num_batches * batch_size * seq_len]

    batches = []
    block = batch_size * seq_len
    for i in range(num_batches):
        chunk = all_ids[i * block:(i + 1) * block]
        batches.append(torch.tensor(chunk, dtype=torch.long).reshape(batch_size, seq_len).cuda())
    return batches


def compute_auc(curve: list[tuple[int, float]]) -> float:
    """Trapezoidal area under the loss curve."""
    if len(curve) < 2:
        return curve[0][1] if curve else float("inf")
    area = 0.0
    for i in range(1, len(curve)):
        dx = curve[i][0] - curve[i - 1][0]
        area += dx * (curve[i][1] + curve[i - 1][1]) / 2
    return area


def wsd_schedule_scale(step: int, total_steps: int, warmup_steps: int,
                       decay_fraction: float = 0.0) -> float:
    """Warmup-Stable-Decay LR multiplier — REFERENCE implementation.

    The harness NO LONGER applies any schedule to miner updates: the learning-rate
    schedule is owned by the optimizer submission itself. Every step() call
    receives the current `step_number`, and `config` carries `total_steps`,
    `warmup_steps` and `decay_fraction`, so a submission can reproduce this exact
    WSD shape — or use any other schedule (or none, e.g. schedule-free methods).
    This function is kept as the canonical WSD reference used by baselines,
    templates and tests.

      - warmup : linear 0 → 1 over the first `warmup_steps`
      - stable : 1.0 until the decay tail begins
      - decay  : cosine 1 → 0 over the final `decay_fraction` of total steps"""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if decay_fraction and decay_fraction > 0.0 and total_steps > 0:
        decay_steps = max(1, int(total_steps * decay_fraction))
        decay_start = total_steps - decay_steps
        if step >= decay_start:
            progress = min(1.0, (step - decay_start) / decay_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


def _chunked_ce_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                     n_chunks: int, ignore_index: int = -100) -> torch.Tensor:
    """Causal-LM cross-entropy computed in ``n_chunks`` row-tiles to avoid
    materializing the full ``(B*S, vocab)`` fp32 logit tensor that HF's
    ``ForCausalLMLoss`` builds via ``logits.float()`` (~vocab·B·S·4 bytes — the
    exact allocation that OOMs no-checkpoint large-batch runs at b48/b64). The
    result is numerically equal to the HF labels-loss: identical causal shift,
    identical fp32 cross-entropy, identical token-mean.

    With grad enabled each tile's fp32 softmax is wrapped in
    ``torch.utils.checkpoint`` so it is freed after forward and recomputed in
    backward — peak fp32 stays one tile in BOTH passes. Under ``no_grad`` (eval)
    no graph is kept, so tiles are summed directly. This is harness forward-only
    code; it never crosses the optimizer-isolation IPC boundary."""
    import torch.utils.checkpoint as _cp
    bsz, seq, vocab = logits.shape
    flat_logits = logits.reshape(-1, vocab)
    shifted = input_ids.new_full((bsz, seq), ignore_index)
    shifted[:, :-1] = input_ids[:, 1:]
    flat_labels = shifted.reshape(-1)
    n = flat_logits.size(0)
    n_chunks = max(1, min(n_chunks, n))
    chunk = (n + n_chunks - 1) // n_chunks

    def _chunk_sum(lc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            lc.float(), yc, ignore_index=ignore_index, reduction="sum")

    grad_on = torch.is_grad_enabled()
    total = flat_logits.new_zeros((), dtype=torch.float32)
    valid = (flat_labels != ignore_index).sum()
    for i in range(0, n, chunk):
        lc = flat_logits[i:i + chunk]
        yc = flat_labels[i:i + chunk]
        if grad_on and lc.requires_grad:
            s = _cp.checkpoint(_chunk_sum, lc, yc, use_reentrant=False)
        else:
            s = _chunk_sum(lc, yc)
        total = total + s
    return total / valid.clamp(min=1).to(total.dtype)


def _model_loss(model: nn.Module, batch: torch.Tensor, chunked_ce: int = 0) -> torch.Tensor:
    """Forward + LM loss. ``chunked_ce > 0`` tiles the cross-entropy (see
    ``_chunked_ce_loss``) to free the fp32-logits materialization; ``0`` =
    stock HF labels-loss path (production default — unchanged)."""
    if getattr(model, "uses_fused_lm_head_ce", False):
        return model(batch, labels=batch).loss
    if chunked_ce and chunked_ce > 0:
        return _chunked_ce_loss(model(batch).logits, batch, chunked_ce)
    return model(batch, labels=batch).loss


def train_and_eval(
    model: nn.Module,
    train_data: list[torch.Tensor],
    eval_data: list[torch.Tensor],
    optimizer_cls: type,
    lr: float,
    weight_decay: float,
    total_steps: int,
    eval_every: int,
    warmup_steps: int,
    max_step_time: float = 10.0,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
    max_total_time: float = 1200.0,
    extra_config: dict = None,
    compute_budget_seconds: float = 0.0,
    decay_fraction: float = 0.0,
    checkpoint_out: str = "",
    empty_cache_every: int = 0,
    chunked_ce: int = 0,
    stop_at_total_steps: bool = False,
) -> TrainingCurve:
    """Run one training job, return the eval loss curve.
    use_amp: enable bf16 autocast for forward/backward (needed for large models).
    compute_budget_seconds: if >0, stop training gracefully when wall time exceeds
      this budget (do final eval, return curve). Implements fixed-FLOPs scoring:
      slower-per-step optimizers get fewer steps within the same compute budget.
      Distinct from max_total_time which is a hard safety kill.

    GPU memory isolation: eval_data and train_data are kept on CPU.  Only one
    batch at a time is moved to GPU and is deleted before the miner's optimizer
    step() runs.  This prevents CUDA kernels submitted by the miner from scanning
    GPU memory to find the evaluation set.
    """
    _force_det = bool(os.environ.get("SN125_FORCE_DETERMINISM"))
    if _force_det and not os.environ.get("SN125_DIAG_NONDET"):
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
    if _force_det and not os.environ.get("SN125_DIAG_FLASH"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    _env_cce = int(os.environ.get("SN125_CHUNKED_CE", "0") or "0")
    if _env_cce > 0 and not chunked_ce:
        chunked_ce = _env_cce

    _env_ec = int(os.environ.get("SN125_EMPTY_CACHE_EVERY", "0") or "0")
    if _env_ec > 0 and not empty_cache_every:
        empty_cache_every = _env_ec

    def _cpu_guard(d):
        if d.__class__.__name__ == "ShardBatchSequence":
            return d
        return [t.cpu() if t.is_cuda else t for t in d]
    train_data = _cpu_guard(train_data)
    eval_data = _cpu_guard(eval_data)

    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    def _mem_allocated():
        return torch.cuda.memory_allocated() if is_cuda else 0

    def _cfg_int(key: str, env: str, default: int) -> int:
        raw = os.environ.get(env, None)
        if key in config:
            raw = config.get(key)
        if raw is None:
            return default
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return default

    param_groups = [{"params": [], "lr": lr, "weight_decay": weight_decay}]
    for name, p in model.named_parameters():
        param_groups[0]["params"].append((name, tuple(p.shape), p.dtype))
    param_names = [name for pg in param_groups for name, _, _ in pg["params"]]
    params_by_name = dict(model.named_parameters())
    ordered_params = [params_by_name[name] for name in param_names]
    param_name_to_idx = {name: i for i, name in enumerate(param_names)}
    ordered_wd_groups: list[tuple[float, float, list[int]]] = []
    _offset = 0
    for pg in param_groups:
        _n = len(pg["params"])
        ordered_wd_groups.append((
            float(pg.get("lr", 0.0) or 0.0),
            float(pg.get("weight_decay", 0.0) or 0.0),
            list(range(_offset, _offset + _n)),
        ))
        _offset += _n

    config = {"total_steps": total_steps, "warmup_steps": warmup_steps, "max_grad_norm": max_grad_norm,
              "decay_fraction": decay_fraction, "_device": str(device),
              "_warmup_applied_externally": False, "_schedule_applied_externally": False}
    if extra_config:
        config.update(extra_config)
    try:
        param_values_interval = int(config.get(
            "_param_values_interval",
            1 if bool(config.get("_requires_param_values", True)) else 0))
    except (TypeError, ValueError):
        param_values_interval = 1
    param_values_interval = max(0, param_values_interval)
    config["_param_values_interval"] = param_values_interval
    config["_requires_param_values"] = bool(param_values_interval)
    trusted_weight_decay = bool(config.get("_trusted_weight_decay", False))
    config["_returns_full_update"] = bool(config.get("_returns_full_update", False))
    config["_ordered_buffers"] = bool(config.get("_ordered_buffers", False))
    config["_unsynced_ipc"] = bool(config.get("_unsynced_ipc", False) or
                                   os.environ.get("SN125_OPTPROC_UNSYNCED"))

    train_loss_interval = _cfg_int(
        "_train_loss_interval", "SN125_TRAIN_LOSS_INTERVAL",
        128 if is_cuda else 1)
    update_finite_interval = 0 if os.environ.get("SN125_SKIP_UPDATE_FINITE") else _cfg_int(
        "_update_finite_interval", "SN125_UPDATE_FINITE_INTERVAL",
        128 if is_cuda else 1)
    memory_check_interval = _cfg_int(
        "_memory_check_interval", "SN125_MEMORY_CHECK_INTERVAL",
        128 if is_cuda else 1)

    def _param_values_available(step_number: int) -> bool:
        return bool(param_values_interval and step_number % param_values_interval == 0)

    def _interval_due(interval: int, step_number: int) -> bool:
        return bool(interval and step_number % interval == 0)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_bytes_f32 = sum(p.numel() for p in model.parameters()) * 4
    max_optimizer_mem = param_bytes_f32 * 200 + 500 * 1024 * 1024
    mem_before_opt = _mem_allocated()

    try:
        opt = optimizer_cls(param_groups, config)
    except Exception as e:
        return TrainingCurve("", lr, weight_decay, [], [], failed=True, error=str(e))
    _step_model = getattr(opt, "step_model", None)
    _split_optimizer = bool(getattr(opt, "_sn125_is_split_optimizer", False))
    tamper_check_interval = 0 if _split_optimizer else _cfg_int(
        "_tamper_check_interval", "SN125_TAMPER_CHECK_INTERVAL", 1)

    def _close_optimizer():
        close = getattr(opt, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _fail_curve(eval_points, train_points, wall_seconds, error):
        _progress_event("train.failed",
                        wall_seconds=float(wall_seconds),
                        eval_points=len(eval_points or []),
                        train_points=len(train_points or []),
                        error=str(error)[-1000:])
        _close_optimizer()
        return TrainingCurve("", lr, weight_decay, eval_points, train_points,
                             wall_seconds, failed=True, error=error)

    _tamper = check_torch_tamper(restore=True)
    if _tamper:
        return _fail_curve([], [], 0.0,
                           f"Optimizer tampered with torch global state in __init__: {_tamper}")

    opt_init_mem = _mem_allocated() - mem_before_opt
    if opt_init_mem > max_optimizer_mem:
        return _fail_curve([], [], 0.0,
                           f"Optimizer state too large: {opt_init_mem/1e9:.1f}GB > {max_optimizer_mem/1e9:.1f}GB limit")

    if compute_budget_seconds > 0 and max_total_time < compute_budget_seconds * 1.05:
        max_total_time = compute_budget_seconds * 1.1

    def _apply_trusted_weight_decay(scale: float) -> None:
        if not trusted_weight_decay or scale == 0:
            return
        with torch.no_grad():
            for lr_pg, wd_pg, idxs in ordered_wd_groups:
                if lr_pg == 0.0 or wd_pg == 0.0:
                    continue
                group = [ordered_params[i] for i in idxs]
                if not group:
                    continue
                decay = 1.0 - lr_pg * wd_pg * float(scale)
                try:
                    torch._foreach_mul_(group, decay)
                except RuntimeError:
                    for p in group:
                        p.mul_(decay)

    def _make_inprocess_ordered_step_model():
        if _step_model is not None:
            return None
        if not bool(config.get("_ordered_buffers", False)):
            return None
        ordered_step = getattr(opt, "step_ordered", None)
        if not callable(ordered_step):
            return None
        returns_full_update = bool(config.get("_returns_full_update", False))
        zero_grads: list[torch.Tensor | None] = [None] * len(ordered_params)

        def _zero_like_param(i: int, p: torch.Tensor) -> torch.Tensor:
            z = zero_grads[i]
            if z is None or z.shape != p.shape or z.device != p.device or z.dtype != p.dtype:
                z = torch.zeros_like(p)
                zero_grads[i] = z
            else:
                z.zero_()
            return z

        def _ordered_step_model(_model, step_number: int, update_scale: float = 1.0,
                                finite_check: bool = True,
                                clear_model_grads: bool = True) -> list[str]:
            grads = []
            for i, p in enumerate(ordered_params):
                grads.append(p.grad if p.grad is not None else _zero_like_param(i, p))
            if _param_values_available(step_number):
                param_values = [p.detach().clone() for p in ordered_params]
            else:
                param_values = []

            updates = ordered_step(grads, param_values, step_number)
            if updates is None:
                if not returns_full_update:
                    raise RuntimeError(
                        "step_ordered returned None without returns_full_update=True")
                update_tensors = grads
                apply_params = ordered_params
                names = param_names
            elif isinstance(updates, dict):
                update_tensors, apply_params, names = [], [], []
                for name, u in updates.items():
                    idx = param_name_to_idx.get(name)
                    if idx is None:
                        raise RuntimeError(f"update for unknown param {name!r}")
                    update_tensors.append(u)
                    apply_params.append(ordered_params[idx])
                    names.append(name)
            else:
                if len(updates) != len(ordered_params):
                    raise RuntimeError(
                        f"step_ordered returned {len(updates)} updates for {len(ordered_params)} params")
                update_tensors, apply_params, names = [], [], []
                for name, p, u in zip(param_names, ordered_params, updates):
                    if u is None:
                        if returns_full_update:
                            raise RuntimeError(f"missing ordered update for {name!r}")
                        continue
                    update_tensors.append(u)
                    apply_params.append(p)
                    names.append(name)

            if returns_full_update and len(names) != len(param_names):
                missing = sorted(set(param_names) - set(names))[:8]
                raise RuntimeError(
                    f"returns_full_update=True but update set is incomplete; missing={missing}")

            if finite_check and update_tensors:
                flags = [torch.isfinite(u).all() for u in update_tensors]
                if not bool(torch.stack(flags).all().item()):
                    for name, u in zip(names, update_tensors):
                        if not bool(torch.isfinite(u).all().item()):
                            raise RuntimeError(f"NaN/Inf in update for {name}")

            _apply_trusted_weight_decay(update_scale)
            if update_tensors:
                with torch.no_grad():
                    try:
                        torch._foreach_add_(apply_params, update_tensors,
                                            alpha=float(update_scale))
                    except RuntimeError:
                        for p, u in zip(apply_params, update_tensors):
                            p.add_(u * update_scale)
            if clear_model_grads:
                model.zero_grad(set_to_none=True)
            return list(names)

        return _ordered_step_model

    _ordered_step_model = _make_inprocess_ordered_step_model()
    if _ordered_step_model is not None:
        _step_model = _ordered_step_model

    eval_curve, train_curve = [], []
    step_times = []
    iter_times = []
    _prev_iter = None
    recent_train_losses = []
    last_loss_val = float("nan")
    _accum_persist = 0
    step = -1
    completed_steps = 0
    t0 = time.time()

    budget_mode = compute_budget_seconds > 0
    step_source = itertools.count() if budget_mode else range(total_steps)

    _quartiles = [0.25, 0.5, 0.75] if (budget_mode and checkpoint_out) else []
    _quartiles_done: set[float] = set()
    _ckpt_records: list[dict] = []

    def _heldout_loss() -> float:
        """Fresh held-out loss at the current model state (read-only; toggles eval/train)."""
        model.eval()
        with torch.no_grad():
            _ls = []
            for _eb in eval_data:
                _gpu = _eb.to(device, non_blocking=is_cuda)
                with torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else nullcontext():
                    _loss = _model_loss(model, _gpu, chunked_ce)
                _ls.append(_loss.float().item())
                del _gpu, _loss
        model.train()
        return sum(_ls) / len(_ls) if _ls else float("inf")

    def _emit_quartile(frac: float, at_step: int) -> None:
        hl = _heldout_loss()
        base, ext = os.path.splitext(checkpoint_out)
        qpath = f"{base}_q{int(round(frac * 100)):02d}{ext or '.safetensors'}"
        from .engine.score_worker import export_checkpoint
        sha = export_checkpoint(model, qpath,
                                metadata={"quartile": f"{frac:.2f}", "step": str(at_step),
                                          "heldout_loss": f"{hl:.6f}",
                                          "elapsed_s": f"{time.time() - t0:.1f}",
                                          "budget_s": f"{compute_budget_seconds:.1f}",
                                          "lr": str(lr), "wd": str(weight_decay)})
        _ckpt_records.append({"pct": frac, "step": at_step, "eval_loss": hl,
                              "wall_t": time.time() - t0, "sha256": sha, "path": qpath})
        _progress_event("train.quartile",
                        pct=float(frac),
                        step=int(at_step),
                        eval_loss=float(hl),
                        elapsed_s=float(time.time() - t0),
                        budget_s=float(compute_budget_seconds),
                        checkpoint_path=qpath,
                        sha256=sha)
        if _progress:
            print(f"    [quartile] {int(round(frac * 100))}% step={at_step} "
                  f"heldout_loss={hl:.4f} sha={sha[:12]} -> {qpath}", flush=True)

    def _check_quartiles(at_step: int) -> None:
        if not _quartiles or len(_quartiles_done) >= len(_quartiles):
            return
        _frac_elapsed = (time.time() - t0) / compute_budget_seconds
        if stop_at_total_steps and total_steps > 0:
            _frac_elapsed = max(_frac_elapsed, at_step / total_steps)
        for _q in _quartiles:
            if _q not in _quartiles_done and _frac_elapsed >= _q:
                _quartiles_done.add(_q)
                _emit_quartile(_q, at_step)

    def _budget_exhausted() -> bool:
        if not budget_mode or time.time() - t0 < compute_budget_seconds:
            return False
        _check_quartiles(completed_steps)
        _progress_event("train.budget_stop", step=int(completed_steps),
                        elapsed_s=float(time.time() - t0),
                        budget_s=float(compute_budget_seconds))
        return True

    _progress = bool(os.environ.get("SN125_PROGRESS"))
    _hb_interval = float(os.environ.get("SN125_PROGRESS_INTERVAL", "60"))
    _last_hb = t0
    _progress_event("train.start",
                    total_steps=int(total_steps),
                    eval_every=int(eval_every),
                    compute_budget_seconds=float(compute_budget_seconds),
                    budget_mode=bool(budget_mode),
                    lr=float(lr),
                    wd=float(weight_decay),
                    device=str(device),
                    param_count=int(sum(p.numel() for p in model.parameters())),
                    param_bytes=int(param_bytes))

    _profile = bool(os.environ.get("SN125_PROFILE"))
    _prof: dict[str, float] = {}
    _prof_steps = 0

    @contextmanager
    def _phase(name: str):
        if not _profile:
            yield
            return
        if is_cuda:
            torch.cuda.synchronize()
        _pt = time.time()
        try:
            yield
        finally:
            if is_cuda:
                torch.cuda.synchronize()
            _prof[name] = _prof.get(name, 0.0) + (time.time() - _pt)

    for step in step_source:
        if _budget_exhausted():
            break
        _now = time.time()
        if _prev_iter is not None:
            iter_times.append(_now - _prev_iter)
        _prev_iter = _now
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for eb in eval_data:
                    gpu_eb = eb.to(device, non_blocking=is_cuda)
                    with torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else nullcontext():
                        _el = _model_loss(model, gpu_eb, chunked_ce)
                    losses.append(_el.float().item())
                    del gpu_eb, _el
                eval_loss = sum(losses) / len(losses) if losses else float("inf")
            eval_curve.append((step, eval_loss, time.time() - t0))
            _elapsed = time.time() - t0
            _progress_event("train.eval",
                            step=int(step),
                            total_steps=int(total_steps),
                            eval_loss=float(eval_loss),
                            elapsed_s=float(_elapsed),
                            rate_steps_per_s=float(step / _elapsed)
                            if step > 0 and _elapsed > 0 else 0.0)
            if os.environ.get("SN125_PROGRESS"):
                _el = _elapsed
                _rate = step / _el if step > 0 and _el > 0 else 0.0
                print(f"    [progress] step {step}/{total_steps} eval_loss={eval_loss:.4f} "
                      f"elapsed={_el:.0f}s rate={_rate:.2f} steps/s", flush=True)
            if recent_train_losses:
                train_curve.append((step, sum(recent_train_losses) / len(recent_train_losses)))
                recent_train_losses.clear()
            elif step == 0:
                train_curve.append((step, eval_loss))
            model.train()

        _check_quartiles(step)

        if _budget_exhausted():
            break

        if not train_data:
            return _fail_curve(eval_curve, train_curve, time.time() - t0, "No training data")
        gpu_batch = train_data[step % len(train_data)].to(device, non_blocking=is_cuda)
        with _phase("forward"):
            with torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else nullcontext():
                loss = _model_loss(model, gpu_batch, chunked_ce)
            sample_train_loss = _interval_due(train_loss_interval, step)
            if sample_train_loss:
                loss_val = loss.detach().float().item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    return _fail_curve(eval_curve, train_curve, time.time() - t0,
                                       f"Loss diverged (NaN/Inf) at step {step}")
                last_loss_val = loss_val
                recent_train_losses.append(loss_val)
        with _phase("backward"):
            loss.backward()
        del gpu_batch, loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        warmup_scale = 1.0

        if _step_model is None:
            param_values_available = _param_values_available(step)
            with _phase("grad_clone"):
                gradients, param_values = {}, {}
                for name, p in model.named_parameters():
                    gradients[name] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    if param_values_available:
                        param_values[name] = p.detach().clone()
            model.zero_grad(set_to_none=True)
        else:
            gradients = param_values = None

        _ec_every = 0 if os.environ.get("SN125_DIAG_NO_EMPTY_CACHE") else empty_cache_every
        with _phase("empty_cache"):
            if is_cuda and _ec_every and (step % _ec_every == 0):
                torch.cuda.empty_cache()
        step_t0 = time.time()
        memory_check_due = _interval_due(memory_check_interval, step)
        mem_pre_step = _mem_allocated() if memory_check_due else 0
        if is_cuda and memory_check_due:
            torch.cuda.reset_peak_memory_stats()
        _alarm_count = [0]
        def _alarm_handler(signum, frame):
            _alarm_count[0] += 1
            if _alarm_count[0] >= 3:
                raise SystemExit("step timeout: miner caught TimeoutError 3x")
            signal.alarm(1)
            raise TimeoutError(f"step() exceeded {int(max_step_time)}s hard limit")
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(max_step_time) + 1)
        _step_error = None
        _updates_applied = False
        try:
            with _phase("opt_step"):
                if _step_model is not None:
                    _step_model(model, step, warmup_scale,
                                finite_check=_interval_due(update_finite_interval, step))
                    updates = {}
                    _updates_applied = True
                else:
                    updates = opt.step(gradients, param_values, step)
        except SystemExit:
            _step_error = f"step() killed at {step}: miner caught timeout 3x"
        except TimeoutError:
            _step_error = f"step() timed out at {step} (>{max_step_time}s)"
        except Exception as e:
            _step_error = f"step() failed at {step}: {e}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if _step_error:
            return _fail_curve(eval_curve, train_curve, time.time() - t0, _step_error)

        if _interval_due(tamper_check_interval, step):
            _tamper = check_torch_tamper(restore=True)
            if _tamper:
                return _fail_curve(eval_curve, train_curve, time.time() - t0,
                                   f"Optimizer tampered with torch global state at step {step}: {_tamper}")

        step_dt = time.time() - step_t0
        step_times.append(step_dt)
        _prof_steps += 1
        if _progress and (time.time() - _last_hb) >= _hb_interval:
            _last_hb = time.time()
            _el = _last_hb - t0
            _alloc_gb = torch.cuda.memory_allocated() / 1e9 if is_cuda else 0.0
            _reserved_gb = torch.cuda.memory_reserved() / 1e9 if is_cuda else 0.0
            _mem = (f"alloc={_alloc_gb:.2f}GB "
                    f"reserved={_reserved_gb:.2f}GB") if is_cuda else "cpu"
            _progress_event("train.heartbeat",
                            step=int(step),
                            total_steps=int(total_steps),
                            elapsed_s=float(_el),
                            rate_steps_per_s=float((step + 1) / _el if _el > 0 else 0.0),
                            train_loss=float(last_loss_val),
                            cuda_alloc_gb=float(_alloc_gb),
                            cuda_reserved_gb=float(_reserved_gb))
            print(f"    [hb] step {step}/{total_steps} elapsed={_el:.0f}s "
                  f"rate={(step + 1) / _el if _el > 0 else 0:.2f} st/s "
                  f"loss={last_loss_val:.4f} {_mem}", flush=True)
        if step_dt > max_step_time:
            return _fail_curve(eval_curve, train_curve, time.time() - t0,
                               f"Step {step} took {step_dt:.1f}s > {max_step_time}s")
        if time.time() - t0 > max_total_time:
            return _fail_curve(eval_curve, train_curve, time.time() - t0,
                               f"Total time {time.time()-t0:.0f}s > {max_total_time}s limit")

        if memory_check_due and is_cuda:
            torch.cuda.synchronize()
        if memory_check_due:
            mem_post_step = _mem_allocated()
            mem_peak_step = torch.cuda.max_memory_allocated() if is_cuda else 0
            step_persist = max(0, mem_post_step - mem_pre_step)
            _accum_persist += step_persist
            step_peak_growth = mem_peak_step - mem_pre_step
            total_opt_mem = opt_init_mem + max(_accum_persist, step_peak_growth)
            if total_opt_mem > max_optimizer_mem:
                return _fail_curve(eval_curve, train_curve, time.time() - t0,
                                   f"Optimizer memory peaked at {total_opt_mem/1e9:.1f}GB > {max_optimizer_mem/1e9:.1f}GB limit at step {step}")

        if not _updates_applied:
            target_params, update_tensors, update_names = [], [], []
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in updates:
                        u = updates[name]
                        if type(u) is not torch.Tensor:
                            u = torch.as_tensor(u)
                        if u.device != p.device:
                            return _fail_curve(eval_curve, train_curve, time.time() - t0,
                                               f"Device mismatch for {name}: update on {u.device}, param on {p.device} at step {step}")
                        if u.shape != p.shape:
                            return _fail_curve(eval_curve, train_curve, time.time() - t0,
                                               f"Shape mismatch for {name}: update {tuple(u.shape)} != param {tuple(p.shape)} at step {step}")
                        target_params.append(p)
                        update_tensors.append(u)
                        update_names.append(name)
                if update_tensors and _interval_due(update_finite_interval, step):
                    flags = [torch.isfinite(u).all() for u in update_tensors]
                    if not bool(torch.stack(flags).all().item()):
                        for name, u in zip(update_names, update_tensors):
                            if not bool(torch.isfinite(u).all().item()):
                                return _fail_curve(eval_curve, train_curve, time.time() - t0,
                                                   f"NaN/Inf in update for {name} at step {step}")
                _apply_trusted_weight_decay(warmup_scale)
                if update_tensors:
                    try:
                        torch._foreach_add_(target_params, update_tensors,
                                            alpha=float(warmup_scale))
                    except RuntimeError:
                        for p, u in zip(target_params, update_tensors):
                            p.add_(u * warmup_scale)

        completed_steps = step + 1
        if _budget_exhausted():
            break
        if budget_mode and stop_at_total_steps and total_steps > 0 and completed_steps >= total_steps:
            _check_quartiles(completed_steps)
            _progress_event("train.horizon_cap", step=int(completed_steps),
                            total_steps=int(total_steps),
                            elapsed_s=float(time.time() - t0),
                            budget_s=float(compute_budget_seconds))
            break

    final_step = completed_steps
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        losses = []
        for eb in eval_data:
            gpu_eb = eb.to(device, non_blocking=is_cuda)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else nullcontext():
                out = model(gpu_eb, labels=gpu_eb)
            losses.append(out.loss.float().item())
            del gpu_eb, out
        eval_loss = sum(losses) / len(losses) if losses else float("inf")
    eval_curve.append((final_step, eval_loss, time.time() - t0))
    _progress_event("train.final_eval",
                    step=int(final_step),
                    eval_loss=float(eval_loss),
                    elapsed_s=float(time.time() - t0))

    if _profile and _prof_steps > 0:
        per_step = {k: round(v / _prof_steps, 6) for k, v in _prof.items()}
        print("PROFILE_BREAKDOWN " + json.dumps({
            "steps_timed": _prof_steps,
            "per_step_seconds": per_step,
            "sum_profiled_per_step_s": round(sum(per_step.values()), 6),
        }), flush=True)

    if os.environ.get("SN125_PROGRESS"):
        import resource
        _rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        _cmax = torch.cuda.max_memory_allocated() if is_cuda else 0
        print(f"    [mem] final-eval peak_host_rss={_rss/1e9:.2f}GB "
              f"cuda_max={_cmax/1e9:.2f}GB (n_eval_batches={len(eval_data)})", flush=True)
    if recent_train_losses:
        train_curve.append((final_step, sum(recent_train_losses) / len(recent_train_losses)))
    elif train_curve:
        train_curve.append((final_step, train_curve[-1][1]))

    if checkpoint_out:
        from .engine.score_worker import export_checkpoint
        _final_sha = export_checkpoint(model, checkpoint_out,
                          metadata={"final_step": str(final_step),
                                    "lr": str(lr), "wd": str(weight_decay)})
        _ckpt_records.append({"pct": 1.0, "step": final_step,
                              "eval_loss": (eval_curve[-1][1] if eval_curve else float("nan")),
                              "wall_t": time.time() - t0, "sha256": _final_sha,
                              "path": checkpoint_out})
        _progress_event("train.checkpoint_final",
                        pct=1.0,
                        step=int(final_step),
                        eval_loss=float(eval_curve[-1][1] if eval_curve else float("nan")),
                        elapsed_s=float(time.time() - t0),
                        checkpoint_path=checkpoint_out,
                        sha256=_final_sha)

    _peak_opt_mem = opt_init_mem + _accum_persist
    _state_mult = _peak_opt_mem / max(param_bytes_f32, 1)
    _close_optimizer()

    return TrainingCurve("", lr, weight_decay, eval_curve, train_curve, time.time() - t0,
                         state_multiplier=max(0.0, _state_mult), step_times=step_times,
                         iter_times=iter_times, checkpoints=_ckpt_records)


def _should_gradient_checkpoint(use_amp: bool, sequence_length: int, flash_on: bool) -> bool:
    """Gradient-checkpointing policy for the production scoring build.

    GC is a MEMORY lever: AMP-scale models and long-sequence (>=1024) MATH attention
    materialize a per-layer (batch, heads, seq, seq) score tensor that OOMs an 80 GB
    card without it. But flash attention streams that score tensor (never materializes
    it), so when flash is on the memory wall is gone and GC is pure ~2× recompute tax
    with no benefit — the measured flash+no_gc spec (frac 0.924) needs GC OFF. So:
    checkpoint only when (AMP or long seq) AND flash is off."""
    return (use_amp or sequence_length >= 1024) and not flash_on


def _flash_regime_on() -> bool:
    """Whether train_and_eval will run with flash/mem-efficient SDP.

    Must mirror the kernel-regime block at the top of train_and_eval: flash is
    the PRODUCTION DEFAULT and is only off under SN125_FORCE_DETERMINISM
    (SN125_DIAG_FLASH re-enables it even then). The GC policy used to key off
    SN125_DIAG_FLASH alone — stale after the 2026-06-24 default flip — so every
    production run silently paid the flash+GC double cost (~0.9 vs ~0.59 s/iter
    on B200 at 360M/b32/seq2048)."""
    if os.environ.get("SN125_FORCE_DETERMINISM"):
        return bool(os.environ.get("SN125_DIAG_FLASH"))
    return True


def evaluate_submission(
    source: str,
    task: TaskSpec,
    seed: int,
    lr: float,
    wd: float,
    model_cache: nn.Module | None = None,
    data_cache: tuple | None = None,
    use_amp: bool = False,
    dataset_name: str | None = None,
    rebuild_model: bool = False,
    baseline_step_time: float | None = None,
    max_total_time: float = 0,
    build_dir: str = "",
    compute_budget_seconds: float = 0.0,
    checkpoint_out: str = "",
) -> tuple[TrainingCurve, nn.Module | None, tuple]:
    """Evaluate one optimizer on one task with given hyperparams. Single run, no grid.
    Returns (curve, model_cache, data_cache).

    checkpoint_out: forwarded to train_and_eval. In budget mode the loop writes the
    final weights here AND quartile snapshots `{base}_q25/_q50/_q75{ext}` (each tagged
    with held-out loss) — the audit trail the launch pipeline publishes. Empty = off."""
    if dataset_name is None:
        dataset_name = getattr(task, "dataset", "fineweb-edu")
    if getattr(task, "data_manifest", ""):
        try:
            from .fineweb import load_manifest
            actual = load_manifest()["manifest_hash"]
        except Exception as e:
            return (TrainingCurve(task.task_id, lr, wd, [], [], failed=True,
                                  error=f"Pinned data manifest unavailable: {e}"),
                    model_cache, data_cache)
        if actual != task.data_manifest:
            return (TrainingCurve(task.task_id, lr, wd, [], [], failed=True,
                                  error=f"Data manifest mismatch: box has {actual[:16]}, "
                                        f"task pins {task.data_manifest[:16]}"),
                    model_cache, data_cache)
    optimizer_cls = load_optimizer_sandboxed(source, build_dir=build_dir)

    max_step = max(baseline_step_time * 1.5, 2.0) if baseline_step_time is not None else 10.0
    max_step = min(max_step, 30.0)
    if max_total_time <= 0:
        if compute_budget_seconds > 0:
            max_total_time = compute_budget_seconds * 1.1
        else:
            max_total_time = max(600.0, task.total_steps * max_step * 2)

    dtype = torch.bfloat16 if use_amp else None
    gc_ckpt = _should_gradient_checkpoint(use_amp, task.sequence_length,
                                          _flash_regime_on())
    if model_cache is None and not rebuild_model:
        model_cache = _build_model(task.model_config, seed, dtype=dtype,
                                   use_pretrained=task.use_pretrained,
                                   gradient_checkpointing=gc_ckpt)

    if data_cache is None:
        if dataset_name == "fineweb-edu-shards":
            total_batches = task.total_steps + 10
        else:
            total_batches = min(task.total_steps + 10, 2000)
        eval_batches = max(1, task.eval_sequences // task.batch_size)
        eval_split = "heldout" if dataset_name == "fineweb-edu-shards" else "train"
        train_data = _load_real_data(dataset_name, task.sequence_length, task.batch_size,
                                     total_batches, seed, tokenizer_name=task.model_config)
        eval_data = _load_real_data(dataset_name, task.sequence_length, task.batch_size,
                                    eval_batches, seed + 1_000_000, split=eval_split,
                                    tokenizer_name=task.model_config)
        data_cache = (train_data, eval_data)
    else:
        train_data, eval_data = data_cache

    warmup_steps = int(task.total_steps * task.warmup_fraction)
    capabilities = extract_capabilities(source)
    extra_cfg = {"use_pretrained": task.use_pretrained,
                 "parameter_count": task.parameter_count,
                 "sequence_length": task.sequence_length,
                 "_requires_param_values": capabilities["requires_param_values"],
                 "_param_values_interval": capabilities["param_values_interval"],
                 "_trusted_weight_decay": capabilities["trusted_weight_decay"],
                 "_returns_full_update": capabilities["returns_full_update"],
                 "_ordered_buffers": capabilities["ordered_buffers"]}

    def _optimizer_cls_for_model(model: nn.Module) -> type:
        """Optionally route miner optimizer steps through the CUDA-IPC process.

        Gated for measurement/rollout by SN125_USE_OPTPROC. Native multi-file
        submissions still use the existing in-worker path until optproc grows a
        build_dir-aware loader.
        """
        if not os.environ.get("SN125_USE_OPTPROC") or build_dir:
            return optimizer_cls
        from .engine.optproc import make_split_optimizer_cls
        device_s = str(next(model.parameters()).device)
        try:
            step_timeout = float(os.environ.get("SN125_OPTPROC_STEP_TIMEOUT", ""))
        except ValueError:
            step_timeout = 0.0
        if step_timeout <= 0.0:
            step_timeout = max(1.0, float(max_step))
        snapshot_updates = os.environ.get("SN125_OPTPROC_NO_SNAPSHOT", "") == ""
        return make_split_optimizer_cls(
            source, device=device_s, sandboxed=True,
            step_timeout=step_timeout, snapshot_updates=snapshot_updates)

    if rebuild_model:
        if model_cache is not None:
            del model_cache; model_cache = None
        torch.cuda.empty_cache()
        gpu_model = _build_model(task.model_config, seed, dtype=dtype,
                                 use_pretrained=task.use_pretrained,
                                 gradient_checkpointing=gc_ckpt)
        run_optimizer_cls = _optimizer_cls_for_model(gpu_model)
        try:
            curve = train_and_eval(
                gpu_model, train_data, eval_data, run_optimizer_cls,
                lr, wd, task.total_steps, task.eval_every, warmup_steps,
                max_step_time=max_step, use_amp=use_amp, extra_config=extra_cfg,
                max_total_time=max_total_time,
                compute_budget_seconds=compute_budget_seconds,
                decay_fraction=getattr(task, "decay_fraction", 0.0),
                checkpoint_out=checkpoint_out,
                stop_at_total_steps=bool(getattr(task, "stop_at_total_steps", False)),
            )
            curve.task_id = task.task_id
        finally:
            del gpu_model
            torch.cuda.empty_cache()
        return curve, None, data_cache
    else:
        model_copy = copy.deepcopy(model_cache)
        run_optimizer_cls = _optimizer_cls_for_model(model_copy)
        try:
            curve = train_and_eval(
                model_copy, train_data, eval_data, run_optimizer_cls,
                lr, wd, task.total_steps, task.eval_every, warmup_steps,
                max_step_time=max_step, use_amp=use_amp, extra_config=extra_cfg,
                max_total_time=max_total_time,
                compute_budget_seconds=compute_budget_seconds,
                decay_fraction=getattr(task, "decay_fraction", 0.0),
                checkpoint_out=checkpoint_out,
                stop_at_total_steps=bool(getattr(task, "stop_at_total_steps", False)),
            )
            curve.task_id = task.task_id
        finally:
            del model_copy
            torch.cuda.empty_cache()
        return curve, model_cache, data_cache



_SANDBOX_SETUP_SCRIPT = '''
import os as _os, sys as _sys, subprocess as _sp, shlex as _shlex

def _mount(cmd):
    """Run mount command, abort on failure for critical mounts."""
    r = _sp.run(_shlex.split(cmd), shell=False, capture_output=True)
    return r.returncode == 0

# Mount namespace setup — principled OS-level isolation
_project_root = _os.environ.get("PYTHONPATH", "")
_hf_cache = _os.environ.get("_SN125_HF_CACHE", "/root/.cache/huggingface")

# Critical mounts — abort if any fail
if not _mount("mount --make-rprivate /"):
    print('{"failed":true,"error":"Sandbox: rprivate mount failed"}'); _sys.exit(1)
if not _mount("mount -t tmpfs -o size=2G tmpfs /tmp"):
    print('{"failed":true,"error":"Sandbox: tmpfs mount failed"}'); _sys.exit(1)

_os.makedirs("/tmp/sn125_project", exist_ok=True)
_os.makedirs("/tmp/hf_cache", exist_ok=True)
if _project_root:
    if not (_mount(f"mount --bind {_project_root} /tmp/sn125_project") and
            _mount("mount -o remount,ro,bind /tmp/sn125_project")):
        print('{"failed":true,"error":"Sandbox: project bind-mount failed"}'); _sys.exit(1)
if _os.path.isdir(_hf_cache):
    _mount(f"mount --bind {_hf_cache} /tmp/hf_cache")
    _mount("mount -o remount,ro,bind /tmp/hf_cache")
    _ds_cache = _os.path.join(_hf_cache, "datasets")
    if _os.path.isdir(_ds_cache):
        _os.makedirs("/tmp/ds_upper", exist_ok=True)
        _os.makedirs("/tmp/ds_work", exist_ok=True)
        _mount(f"mount -t overlay overlay -o lowerdir=/tmp/hf_cache/datasets,"
               f"upperdir=/tmp/ds_upper,workdir=/tmp/ds_work /tmp/hf_cache/datasets")
        _os.chmod("/tmp/ds_upper", 0o777)
        _os.chmod("/tmp/hf_cache/datasets", 0o777)

# Make sensitive host dirs read-only — critical for security
for _d in ("/root", "/home", "/etc", "/var"):
    if not (_mount(f"mount --bind {_d} {_d}") and _mount(f"mount -o remount,ro,bind {_d}")):
        print(f'{{"failed":true,"error":"Sandbox: ro mount {_d} failed"}}'); _sys.exit(1)

# Redirect paths to /tmp locations
_os.environ["PYTHONPATH"] = "/tmp/sn125_project"
_sys.path.insert(0, "/tmp/sn125_project")
_os.environ["HF_HOME"] = "/tmp/hf_cache"
_os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache/hub"
_os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache/datasets"
_os.environ["TRITON_CACHE_DIR"] = "/tmp/sn125_triton_cache"
_os.makedirs("/tmp/sn125_build", exist_ok=True)
_os.chmod("/tmp/sn125_build", 0o777)
_os.makedirs("/tmp/sn125_triton_cache", exist_ok=True)
_os.chmod("/tmp/sn125_triton_cache", 0o777)
_os.environ["_SN125_SANDBOX_MODE"] = "namespaces"
'''


_SECCOMP_SANDBOX_SETUP_SCRIPT = '''
import os as _os, sys as _sys
from sn125 import seccomp_sandbox as _sn125_seccomp
_run_dir = _os.environ.get("_SN125_RUN_DIR", "")
if not (_run_dir and _os.path.isdir(_run_dir)):
    print('{"failed":true,"error":"Sandbox(seccomp): private run dir missing"}'); _sys.exit(1)
for _k, _sub in (("TMPDIR", "tmp"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                 ("TRITON_CACHE_DIR", "triton_cache"), ("TRITON_HOME", "triton_home"),
                 ("XDG_CACHE_HOME", ".cache"), ("", "build")):
    _p = _os.path.join(_run_dir, _sub)
    _os.makedirs(_p, exist_ok=True)
    try:
        _os.chown(_p, 65534, 65534)  # still root here; the drop comes next
    except OSError:
        pass
    if _k:
        _os.environ[_k] = _p
_os.environ["_SN125_SANDBOX_MODE"] = "seccomp"
'''


_WORKER_SCRIPT = '''
import sys, json, os, resource, subprocess

# Phase 1: Read args (before priv drop — stdin might not be readable after)
def _fail(msg, tid="unknown", lr=0, wd=0):
    print(json.dumps({"task_id": tid, "lr": lr, "wd": wd,
        "eval_points": [], "train_points": [], "wall_seconds": 0.0,
        "failed": True, "error": msg}))
    sys.exit(0)

try:
    _input = sys.stdin.read()
    args = json.loads(_input)
except Exception as _e:
    _fail(f"Worker args parse failed: {type(_e).__name__}: {_e}")

_files = args.get("files", {})
# Sandbox flavour: "namespaces" (unshare + mounts, the VM-host path; the setup
# script created /tmp/sn125_project) or "seccomp" (container hosts; the
# prologue exported _SN125_SANDBOX_MODE and a private run dir). Either way
# the privilege drop below is mandatory and attested.
# The flavour comes ONLY from the trusted prologue's env export — never from
# a filesystem probe (on a container host /tmp is shared and a miner could
# pre-create a marker path to steer a later worker off the seccomp path).
_sandbox_mode = os.environ.get("_SN125_SANDBOX_MODE", "")
_in_sandbox = _sandbox_mode in ("namespaces", "seccomp")
_build = (os.path.join(os.environ["_SN125_RUN_DIR"], "build")
          if _sandbox_mode == "seccomp" else "/tmp/sn125_build")
os.makedirs(_build, exist_ok=True)
_has_native = any(f.endswith((".cu", ".c")) for f in _files)
_seccomp_att = {}

# Rebind _fail with task context + nonce now that args are parsed
def _fail(msg):
    print(json.dumps({"_nonce": args.get("_nonce", ""), "task_id": args.get("task", {}).get("task_id", "unknown"),
        "lr": args.get("lr", 0), "wd": args.get("wd", 0),
        "eval_points": [], "train_points": [], "wall_seconds": 0.0,
        "failed": True, "error": msg}))
    sys.exit(0)

# Phase 2: Drop privileges FIRST, BEFORE compilation.
# Compilation as root is dangerous: #include "/root/.ssh/id_rsa" leaks secrets via
# gcc error messages, __attribute__((constructor)) executes arbitrary code.
# /tmp/sn125_build has mode 0o777 (set by setup script), so nobody can compile there.
# NPROC starts at 256 to allow nvcc/gcc fork, lowered to 64 after compilation.
if _in_sandbox:
    try:
        os.setgroups([])
    except Exception as _e:
        _fail(f"Sandbox: setgroups failed: {_e}")
    try:
        os.setgid(65534)
    except Exception as _e:
        _fail(f"Sandbox: setgid failed: {_e}")
    try:
        os.setuid(65534)
    except Exception as _e:
        _fail(f"Sandbox: setuid failed: {_e}")
    # Verify we're actually nobody — defense against no-op setuid
    if os.getuid() != 65534 or os.getgid() != 65534:
        _fail(f"Sandbox: priv drop verification failed: uid={os.getuid()} gid={os.getgid()}")
    # PR_SET_NO_NEW_PRIVS: prevent regaining privileges via setuid binaries or
    # other escalation vectors. Once set, cannot be unset by the process or children.
    import ctypes as _ctypes
    try:
        _ctypes.CDLL("libc.so.6").prctl(38, 1, 0, 0, 0)  # 38 = PR_SET_NO_NEW_PRIVS
    except Exception:
        pass  # best-effort — priv drop + namespace isolation is the primary defense
    if _sandbox_mode == "seccomp":
        # Container hosts: no network namespace exists, so the syscall filter IS
        # the egress boundary. Install + positively probe it now (before any
        # compilation or miner import); a failure here is fatal, never degraded.
        # `engage` also sets NO_NEW_PRIVS (mandatory for an unprivileged filter).
        try:
            from sn125 import seccomp_sandbox as _sn125_seccomp
            _seccomp_att = _sn125_seccomp.engage()
        except Exception as _e:
            _fail(f"Sandbox(seccomp): could not engage syscall filter: {_e}")
        # Host-RAM backstop replacing the cgroup v2 RSS cap (unavailable in a
        # container): RLIMIT_DATA bounds anonymous memory (heap + private mmaps);
        # the mmap'd shards are file-backed and do not count.
        try: resource.setrlimit(resource.RLIMIT_DATA, (64 * 1024**3, 64 * 1024**3))
        except Exception: pass
    # RLIMIT_AS must cover the worker's WHOLE virtual address space, which on a GPU box
    # is dominated by THREE VA consumers, not just resident GPU tensors:
    #   (1) CUDA reserves device-sized VA on context init (UVA) — ~= GPU VRAM;
    #   (2) PyTorch's caching-allocator pool — cudaMalloc'd blocks each consume VA (<= VRAM);
    #   (3) the mmap'd FineWeb shards — fineweb.py:334 np.load(mmap_mode="r") maps EVERY
    #       split file; the production seq2048 set is ~33 GB of VA (pages lazy, but the
    #       whole mapping counts against RLIMIT_AS), plus host cuBLAS/cuDNN workspaces.
    # The old `GPU_VRAM + 32GB` ceiling was far too tight: on a 177 GB B200 (= 209 GB AS)
    # a 240 MiB cudaMalloc FAILED amid ~130 GiB FREE device memory (genesis smoke #3,
    # 2026-06-25) — a spurious "CUDA OutOfMemory" that was actually the process hitting its
    # virtual-address ceiling once CUDA's device VA reservation + the 33 GB shard mmap were
    # in. None of the allocator-config tweaks (split-cap / expandable / no-cap) can fix a VA
    # limit. Size AS as 2*VRAM (device reservation + pool) + 64 GB (shards + host). Physical
    # CPU memory stays hard-limited by cgroup (24 GB RSS), so this only loosens the VA
    # backstop, not the real memory defense.
    # Auto-detect from torch to support A100 (80GB), B200 (192GB), etc.
    try:
        import torch as _t
        if _t.cuda.is_available():
            _props = _t.cuda.get_device_properties(0)
            _gpu_bytes = getattr(_props, "total_memory", getattr(_props, "total_mem", 0))
        else:
            _gpu_bytes = 0
    except Exception:
        _gpu_bytes = 0
    _as_limit = max(96 * 1024**3, _gpu_bytes * 2 + 64 * 1024**3)  # 2*VRAM (CUDA VA + pool) + 64GB (shard mmap + host)
    # RLIMIT_CPU bounds runaway/CPU-spinning miner code, but it must scale with the
    # legitimate wall-clock budget: a fixed-budget eval (e.g. the 20h confirmation, or
    # any multi-minute production round) accumulates CPU-seconds ~= wall-seconds because
    # CUDA synchronizes busy-wait on the host thread. A hardcoded 300s killed every run
    # longer than 5 minutes (SIGXCPU) before it could emit output. The parent wall-clock
    # proc_timeout remains the primary limiter; this is a generous secondary backstop.
    _cpu_limit = int(args.get("cpu_seconds_limit", 300))
    _fsize_limit = int(args.get("file_size_limit", 100 * 1024**2))
    for _lim, _val in [(resource.RLIMIT_NPROC, 256), (resource.RLIMIT_FSIZE, _fsize_limit),
                        (resource.RLIMIT_AS, _as_limit), (resource.RLIMIT_CPU, _cpu_limit)]:
        try: resource.setrlimit(_lim, (_val, _val))
        except Exception: pass
    # The production shard set is ~1k mmap'd files (one fd each); a container's
    # default soft nofile of 1024 dies at the optproc import (RunPod smoke,
    # 2026-09-04). Raise the soft limit toward the hard limit, capped.
    try:
        _nf_soft, _nf_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        _nf_want = min(_nf_hard if _nf_hard != resource.RLIM_INFINITY else 1 << 20, 1 << 16)
        if _nf_soft < _nf_want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (_nf_want, _nf_hard))
    except Exception: pass
else:
    try:
        import torch as _t
        if _t.cuda.is_available():
            _props = _t.cuda.get_device_properties(0)
            _gpu_bytes = getattr(_props, "total_memory", getattr(_props, "total_mem", 0))
        else:
            _gpu_bytes = 0
    except Exception:
        _gpu_bytes = 0
    # Same VA accounting as the in-sandbox branch above: 2*VRAM (CUDA device reservation +
    # PyTorch pool) + 64GB (mmap'd shards + host workspaces). GPU_VRAM+32GB AS-OOM'd a
    # 240MiB alloc amid 130GiB free VRAM on B200 (genesis smoke #3).
    _as_limit = max(96 * 1024**3, _gpu_bytes * 2 + 64 * 1024**3)
    try: resource.setrlimit(resource.RLIMIT_AS, (_as_limit, _as_limit))
    except Exception: pass

# Phase 3: Write files + compile CUDA/C (now running as nobody — can't read secrets)
_compile_errors = []
for _fname, _src in _files.items():
    _base = os.path.basename(_fname)
    _path = os.path.join(_build, _base)
    with open(_path, "w") as _f:
        _f.write(_src)
    if _base.endswith(".cu"):
        _so = _path.replace(".cu", ".so")
        try:
            _r = subprocess.run(["nvcc", "--shared", "-Xcompiler", "-fPIC", "-O2",
                                 "-o", _so, _path], capture_output=True, timeout=60, cwd=_build)
            if _r.returncode != 0:
                # Truncate stderr to avoid leaking file contents from #include attacks
                _compile_errors.append(f"nvcc {_base}: exit {_r.returncode}")
        except subprocess.TimeoutExpired:
            _compile_errors.append(f"nvcc {_base}: timed out")
    elif _base.endswith(".c"):
        _so = _path.replace(".c", ".so")
        try:
            _r = subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", _so, _path],
                                capture_output=True, timeout=60, cwd=_build)
            if _r.returncode != 0:
                _compile_errors.append(f"gcc {_base}: exit {_r.returncode}")
        except subprocess.TimeoutExpired:
            _compile_errors.append(f"gcc {_base}: timed out")

if _compile_errors:
    _fail(f"Compilation failed: {_compile_errors}")

# Phase 4: Tighten NPROC after compilation (nvcc/gcc done, no more forking needed)
if _in_sandbox:
    try: resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception: pass

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Phase 5: Run training
_tid = "unknown"; _lr = _wd = 0.0
_nonce = args.get("_nonce", "")
try:
    from sn125.harness import evaluate_submission, TaskSpec
    import torch as _torch
    task = TaskSpec(**args["task"])
    _tid = task.task_id; _lr = args["lr"]; _wd = args["wd"]
    # Load pre-generated data on CPU — train_and_eval moves batches to GPU
    # one at a time and deletes them before miner code runs, preventing
    # CUDA memory scanning of eval data.
    _data_cache = None
    _dp = args.get("data_path", "")
    if _dp and os.path.exists(_dp):
        _saved = _torch.load(_dp, map_location="cpu", weights_only=True)
        _data_cache = (_saved["train"], _saved["eval"])
        del _saved
    curve, _, _ = evaluate_submission(
        args["source"], task, args["seed"], args["lr"], args["wd"],
        data_cache=_data_cache,
        use_amp=args.get("use_amp", False), dataset_name=args.get("dataset_name", "fineweb-edu"),
        rebuild_model=True, baseline_step_time=args.get("baseline_step_time"),
        max_total_time=args.get("max_total_time", 0),
        compute_budget_seconds=args.get("compute_budget_seconds", 0.0),
        checkpoint_out=args.get("checkpoint_out", ""),
        build_dir=_build if _has_native else "")
    sys.stdout.flush()  # flush any native code stdout before our authenticated output
    print(json.dumps({"_nonce": _nonce, "task_id": curve.task_id, "lr": curve.lr, "wd": curve.wd,
        "eval_points": curve.eval_points, "train_points": curve.train_points,
        "wall_seconds": curve.wall_seconds, "state_multiplier": curve.state_multiplier,
        "checkpoints": getattr(curve, "checkpoints", []),
        "step_times": getattr(curve, "step_times", []),
        "iter_times": getattr(curve, "iter_times", []),
        # Sandbox-engagement attestation (nonce-authenticated, so miner code cannot
        # forge it): the parent verifies these when SN125_REQUIRE_SANDBOX is on.
        "sandboxed": _in_sandbox, "worker_uid": os.getuid(),
        "sandbox_mode": _sandbox_mode or None,
        **{k: v for k, v in _seccomp_att.items() if k != "sandbox_mode"},
        "failed": curve.failed, "error": curve.error}))
except BaseException as _e:
    sys.stdout.flush()
    try:
        import traceback as _traceback
        _tb = "".join(_traceback.format_exception(type(_e), _e, _e.__traceback__))[-4000:]
    except Exception:
        _tb = ""
    try:
        import torch as _torch
        _cuda_mem = {
            "allocated": float(_torch.cuda.memory_allocated()),
            "reserved": float(_torch.cuda.memory_reserved()),
            "max_allocated": float(_torch.cuda.max_memory_allocated()),
        } if _torch.cuda.is_available() else {}
    except Exception:
        _cuda_mem = {}
    print(json.dumps({"_nonce": _nonce, "task_id": _tid, "lr": _lr, "wd": _wd,
        "eval_points": [], "train_points": [], "wall_seconds": 0.0,
        "failed": True,
        "error": f"Worker crash: {type(_e).__name__}: {_e}",
        "traceback": _tb,
        "cuda_memory": _cuda_mem}))
'''


def _has_sandbox() -> bool:
    """Check if full sandbox (unshare --mount --net --ipc --uts) is available."""
    import subprocess
    try:
        r = subprocess.run(
            ["unshare", "--mount", "--net", "--ipc", "--uts",
             "--pid", "--fork", "--mount-proc", "true"],
            capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _has_seccomp_sandbox() -> bool:
    """Check that the seccomp flavour can engage here: root (for the priv drop)
    on x86_64, and a throwaway subprocess installs + probes the filter."""
    import subprocess
    if os.geteuid() != 0:
        return False
    try:
        r = subprocess.run(
            [sys.executable, "-m", "sn125.seccomp_sandbox"],
            capture_output=True, timeout=120,
            env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))})
        return r.returncode == 0 and b'"ok": true' in r.stdout
    except Exception:
        return False


def _sandbox_mode() -> str:
    """Which OS sandbox flavour this box uses (settings.SANDBOX_MODE_ENV):
    ``namespaces`` (VM hosts: unshare + mounts + priv drop) or ``seccomp``
    (container hosts: priv drop + syscall filter). The cloud orchestrator sets
    it per provider; unset means namespaces (the historical behaviour)."""
    return settings.sandbox_mode()


_SANDBOX_AVAILABLE = None
_SECCOMP_SANDBOX_AVAILABLE = None


def _sandbox_required() -> bool:
    """FAIL-CLOSED sandbox toggle (mirrors the C1/C6 egress-lockdown pattern).

    When SN125_REQUIRE_SANDBOX is truthy, evaluate_submission_isolated refuses to
    run miner code unless the full unshare sandbox (namespaces + priv drop) both
    engages AND attests engagement in its authenticated output. Without it the
    sandbox silently degrades to a bare subprocess when `unshare` is unavailable —
    the exact silent-degradation class the egress lockdown closed. Production
    (cloud.py prod eval env) sets this to 1 by default."""
    return settings.sandbox_required()


def _enforce_sandbox_attestation(result: dict, task_id: str, lr: float, wd: float) -> dict:
    """FAIL-CLOSED sandbox attestation: a successful curve is only trusted when
    the worker's nonce-authenticated output confirms it actually ran inside the
    sandbox AND as the dropped-privilege user (nobody/65534). Catches the setup
    script being skipped/bypassed in ways the parent-side availability probe
    can't see. No-op unless SN125_REQUIRE_SANDBOX is on or the result failed."""
    if not _sandbox_required() or result.get("failed"):
        return result
    mode = _sandbox_mode()
    if mode == "seccomp":
        from .seccomp_sandbox import attestation_ok
        if result.get("sandboxed") is True and attestation_ok(result):
            return result
        why = (f"sandboxed={result.get('sandboxed')!r}, "
               f"sandbox_mode={result.get('sandbox_mode')!r}, "
               f"worker_uid={result.get('worker_uid')!r}, "
               f"seccomp_mode={result.get('seccomp_mode')!r}, "
               f"no_new_privs={result.get('no_new_privs')!r}, "
               f"net_denied={result.get('net_denied')!r}")
    else:
        if (result.get("sandboxed") is True and result.get("worker_uid") == 65534
                and result.get("sandbox_mode") in (None, "namespaces")):
            return result
        why = (f"sandboxed={result.get('sandboxed')!r}, "
               f"sandbox_mode={result.get('sandbox_mode')!r}, "
               f"worker_uid={result.get('worker_uid')!r}")
    return {
        "task_id": result.get("task_id", task_id),
        "lr": result.get("lr", lr), "wd": result.get("wd", wd),
        "eval_points": [], "train_points": [], "wall_seconds": 0.0,
        "failed": True,
        "error": ("SN125_REQUIRE_SANDBOX: worker did not attest sandbox "
                  f"engagement [{mode}] ({why}); result rejected"),
    }

_CGROUP_ROOT = "/sys/fs/cgroup"
_CGROUP_AVAILABLE = None


def _has_cgroup_v2() -> bool:
    """Check if we can create cgroup v2 child groups with memory+pids controllers."""
    try:
        with open(f"{_CGROUP_ROOT}/cgroup.controllers") as _f:
            ctrl = _f.read()
        if "memory" not in ctrl or "pids" not in ctrl:
            return False
        subtree = f"{_CGROUP_ROOT}/cgroup.subtree_control"
        with open(subtree, "w") as f:
            f.write("+memory +pids")
        probe = f"{_CGROUP_ROOT}/sn125_probe_{os.getpid()}"
        os.makedirs(probe, exist_ok=True)
        os.rmdir(probe)
        return True
    except Exception:
        return False


def _cgroup_create(name: str, memory_max_bytes: int, pids_max: int) -> str | None:
    """Create a cgroup v2 group with memory and pids limits. Returns path or None."""
    global _CGROUP_AVAILABLE
    if _CGROUP_AVAILABLE is None:
        _CGROUP_AVAILABLE = _has_cgroup_v2()
    if not _CGROUP_AVAILABLE:
        return None
    path = f"{_CGROUP_ROOT}/{name}"
    try:
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/memory.max", "w") as f:
            f.write(str(memory_max_bytes))
        try:
            with open(f"{path}/memory.swap.max", "w") as f:
                f.write("0")
        except (FileNotFoundError, PermissionError):
            pass
        with open(f"{path}/pids.max", "w") as f:
            f.write(str(pids_max))
        return path
    except Exception:
        return None


def _cgroup_add_pid(cgroup_path: str, pid: int) -> bool:
    """Add a process to a cgroup. Returns success."""
    try:
        with open(f"{cgroup_path}/cgroup.procs", "w") as f:
            f.write(str(pid))
        return True
    except Exception:
        return False


def _cgroup_cleanup(cgroup_path: str):
    """Remove a cgroup (must have no live processes)."""
    try:
        try:
            with open(f"{cgroup_path}/cgroup.procs") as f:
                pids = f.read().strip()
            for pid_str in pids.split():
                try: os.kill(int(pid_str), 9)
                except ProcessLookupError: pass
        except Exception:
            pass
        import time
        for _ in range(10):
            try:
                os.rmdir(cgroup_path)
                return
            except OSError:
                time.sleep(0.1)
        _log.warning(f"Failed to remove cgroup after 10 retries: {cgroup_path}")
    except Exception as e:
        _log.warning(f"cgroup cleanup error for {cgroup_path}: {e}")


def _make_safe_env() -> dict:
    """Build sanitized environment for sandbox subprocess.
    Blocks ALL secrets and attack vectors. Only passes CUDA/Python essentials."""
    _ENV_ALLOWLIST = frozenset([
        "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM",
        "CUDA_HOME", "CUDA_PATH",
        "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
        "TORCH_CUDA_ARCH_LIST",
        "SN125_DIAG_FLASH", "SN125_DIAG_NONDET", "SN125_CHUNKED_CE", "SN125_EMPTY_CACHE_EVERY",
        "SN125_FORCE_GC", "SN125_LEAN_LOSS_CHUNKS",
        "SN125_LEAN_COMPILE", "SN125_LEAN_LOSS_CKPT",
        "SN125_USE_OPTPROC", "SN125_OPTPROC_STEP_TIMEOUT",
        "SN125_PROGRESS", "SN125_PROGRESS_INTERVAL", "SN125_PROFILE", "SN125_FINEWEB_DIR",
        "SN125_FORCE_DETERMINISM", "SN125_SKIP_UPDATE_FINITE",
    ])
    safe_env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    safe_env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safe_env["HF_HUB_OFFLINE"] = "1"
    safe_env["TRANSFORMERS_OFFLINE"] = "1"
    safe_env["HF_DATASETS_OFFLINE"] = "1"
    safe_env["HF_HOME"] = settings.hf_home()
    safe_env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    safe_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    safe_env["OPENBLAS_NUM_THREADS"] = "1"
    safe_env["MKL_NUM_THREADS"] = "1"
    safe_env["OMP_NUM_THREADS"] = "1"
    safe_env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor"
    safe_env["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    safe_env["TRITON_HOME"] = "/tmp/triton_home"
    safe_env["XDG_CACHE_HOME"] = "/tmp/.cache"
    safe_env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    if os.environ.get("SN125_DIAG_FLASH"):
        safe_env["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    else:
        safe_env["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"
    safe_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    safe_env["NVIDIA_VISIBLE_DEVICES"] = safe_env["CUDA_VISIBLE_DEVICES"]
    safe_env["_SN125_HF_CACHE"] = settings.hf_home()
    return safe_env


def _kill_process_group(proc) -> None:
    """SIGKILL everything in the worker's session (start_new_session=True):
    the container flavour has no PID namespace, so this is what guarantees
    no miner-forked process outlives the eval. Best-effort, never raises."""
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        return
    if pgid == os.getpgid(0):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        pass


def _preflight_shard_manifest(task: "TaskSpec") -> str | None:
    """For a manifest-pinned shard task, verify the pinned shards are present and
    hash-matched on this box. Returns an error string (fail fast, no sandbox launch)
    or None if OK / not applicable. Mirrors the S1 guard in evaluate_submission so
    the §4.3 lazy-load path (worker builds the sequence from on-disk shards instead
    of a pickled .pt) fails the same way before spending a sandbox provision."""
    pin = getattr(task, "data_manifest", "")
    if not pin:
        return None
    try:
        from .fineweb import load_manifest
        actual = load_manifest()["manifest_hash"]
    except Exception as e:
        return f"Pinned data manifest unavailable: {e}"
    if actual != pin:
        return (f"Data manifest mismatch: box has {actual[:16]}, "
                f"task pins {pin[:16]}")
    return None


def evaluate_submission_isolated(
    source: str | dict, task: TaskSpec, seed: int, lr: float, wd: float,
    use_amp: bool = False, dataset_name: str | None = None,
    baseline_step_time: float | None = None,
    timeout: float = 0,
    net_isolate: bool = True,
    compute_budget_seconds: float = 0.0,
    checkpoint_out: str = "",
) -> TrainingCurve:
    """Run miner code in a principled OS-level sandbox.

    Sandbox layers:
      1. Namespaces (mount+net+ipc+uts+pid): no network, private /tmp, ro filesystem, private /proc
      2. Privilege drop: nobody user (uid 65534)
      3. Resource limits: NPROC=64, FSIZE=100M, AS=auto(2*GPU+64GB), CPU=300s
      4. cgroup v2: hard physical memory limit (24GB RSS), pids limit (256)
      5. Env sanitization: no secrets, HF offline
      6. Parent timeout + SIGKILL
      7. Data pre-generation: parent generates data (with network), saves to shared file.
         Subprocess loads from file — no network needed for data. Validator controls
         exactly what data the miner sees.

    source: Python source string (legacy) or dict of {filename: source_code}
      for multi-language submissions (optimizer.py + optional .cu/.c files).
    """
    import subprocess
    global _SANDBOX_AVAILABLE, _SECCOMP_SANDBOX_AVAILABLE
    sandbox_mode = _sandbox_mode()
    if sandbox_mode == "seccomp":
        if _SECCOMP_SANDBOX_AVAILABLE is None:
            _SECCOMP_SANDBOX_AVAILABLE = _has_seccomp_sandbox()
        sandbox_available = _SECCOMP_SANDBOX_AVAILABLE
    else:
        if _SANDBOX_AVAILABLE is None:
            _SANDBOX_AVAILABLE = _has_sandbox()
        sandbox_available = _SANDBOX_AVAILABLE
    if dataset_name is None:
        dataset_name = getattr(task, "dataset", "fineweb-edu")

    if isinstance(source, str):
        files = {"optimizer.py": source}
    else:
        files = dict(source)
        if "optimizer.py" not in files:
            return TrainingCurve(task.task_id, lr, wd, [], [], 0.0, True,
                                 "Submission must include optimizer.py")

    max_step = max(baseline_step_time * 1.5, 2.0) if baseline_step_time is not None else 10.0
    max_step = min(max_step, 30.0)
    max_total_time = max(600.0, task.total_steps * max_step * 2) if timeout <= 0 else timeout
    has_native = any(f.endswith((".cu", ".c")) for f in files)
    compile_overhead = 120 if has_native else 0
    load_overhead = max(30, min(180, max_total_time * 0.2))
    proc_timeout = max_total_time + load_overhead + compile_overhead

    use_sandbox = net_isolate and sandbox_available
    if _sandbox_required() and not use_sandbox:
        why = ("net_isolate=False" if not net_isolate else
               ("seccomp sandbox unavailable on this box (needs root + x86_64 + "
                "a kernel that accepts an unprivileged filter)"
                if sandbox_mode == "seccomp" else
                "unshare namespaces unavailable on this box"))
        return TrainingCurve(
            task.task_id, lr, wd, [], [], failed=True,
            error=f"SN125_REQUIRE_SANDBOX: OS sandbox required but not engaged "
                  f"({why}); fail-closed — refusing to run miner code unsandboxed")
    safe_env = _make_safe_env()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    use_seccomp = use_sandbox and sandbox_mode == "seccomp"
    run_dir = ""
    data_dir = os.path.join(settings.hf_home(), "sn125_data")
    os.makedirs(data_dir, exist_ok=True)
    _cfg_slug = task.model_config.replace("/", "_")
    _manifest_slug = getattr(task, "data_manifest", "")[:12]
    data_hash = f"{task.task_id}_{seed}_{task.total_steps}_{task.batch_size}_{task.sequence_length}_{dataset_name}_{_cfg_slug}_{task.eval_sequences}_{_manifest_slug}"
    data_path = os.path.join(data_dir, f"{data_hash}.pt")
    sandbox_data_path = f"/tmp/hf_cache/sn125_data/{data_hash}.pt"
    if dataset_name == "fineweb-edu-shards":
        _pf = _preflight_shard_manifest(task)
        if _pf is not None:
            return TrainingCurve(task.task_id, lr, wd, [], [], failed=True, error=_pf)
        data_path = ""
    else:
        try:
            if not os.path.exists(data_path):
                total_batches = min(task.total_steps + 10, 2000)
                eval_batches = max(1, task.eval_sequences // task.batch_size)
                train_data = _load_real_data(dataset_name, task.sequence_length, task.batch_size,
                                             total_batches, seed, tokenizer_name=task.model_config)
                eval_data = _load_real_data(dataset_name, task.sequence_length, task.batch_size,
                                            eval_batches, seed + 1_000_000, split="train",
                                            tokenizer_name=task.model_config)
                fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix=".pt.tmp")
                os.close(fd)
                try:
                    torch.save({"train": [t.cpu() for t in train_data],
                                 "eval": [t.cpu() for t in eval_data]}, tmp_path)
                    os.replace(tmp_path, data_path)
                    os.chmod(data_path, 0o644)
                except BaseException:
                    try: os.unlink(tmp_path)
                    except OSError: pass
                    raise
                del train_data, eval_data
        except Exception as e:
            return TrainingCurve(task.task_id, lr, wd, [], [], failed=True,
                                 error=f"Data pre-generation failed: {e}")

    if use_seccomp:
        run_dir = tempfile.mkdtemp(prefix="sn125_run_", dir=os.environ.get("SN125_RUN_DIR_BASE") or None)
        os.chown(run_dir, 65534, 65534)
        safe_env["_SN125_RUN_DIR"] = run_dir
    if use_sandbox and checkpoint_out:
        try:
            ck_dir = os.path.dirname(os.path.abspath(checkpoint_out)) or "."
            os.makedirs(ck_dir, exist_ok=True)
            if os.geteuid() == 0:
                os.chown(ck_dir, 65534, 65534)
                os.chmod(ck_dir, 0o700)
        except OSError as e:
            return TrainingCurve(task.task_id, lr, wd, [], [], failed=True,
                                 error=f"checkpoint dir not writable for the sandbox worker: {e}")

    py_flags = ["-P"] if sys.version_info >= (3, 11) else []
    if use_seccomp:
        worker_full = _SECCOMP_SANDBOX_SETUP_SCRIPT + _WORKER_SCRIPT
        cmd = [sys.executable, *py_flags, "-c", worker_full]
    elif use_sandbox:
        worker_full = _SANDBOX_SETUP_SCRIPT + _WORKER_SCRIPT
        cmd = ["unshare", "--mount", "--net", "--ipc", "--uts",
               "--pid", "--fork", "--mount-proc",
               sys.executable, *py_flags, "-c", worker_full]
    else:
        cmd = [sys.executable, "-c", _WORKER_SCRIPT]

    import secrets
    _nonce = secrets.token_hex(16)
    args_json = json.dumps({
        "source": files["optimizer.py"],
        "files": files,
        "task": asdict(task), "seed": seed, "lr": lr, "wd": wd,
        "use_amp": use_amp, "dataset_name": dataset_name,
        "baseline_step_time": baseline_step_time, "max_total_time": max_total_time,
        "compute_budget_seconds": compute_budget_seconds, "_nonce": _nonce,
        "checkpoint_out": checkpoint_out,
        "file_size_limit": int(4 * 1024**3) if checkpoint_out else int(100 * 1024**2),
        "cpu_seconds_limit": int(proc_timeout * 2 + 600),
        "data_path": "" if not data_path else (
            sandbox_data_path if (use_sandbox and not use_seccomp) else data_path),
    })

    worker_log_path = os.environ.get("SN125_WORKER_LOG_PATH", "").strip()
    forward_worker_progress = bool(os.environ.get("SN125_FORWARD_WORKER_PROGRESS") or
                                   os.environ.get("SN125_PROGRESS"))
    try:
        worker_log_max_bytes = int(os.environ.get("SN125_WORKER_LOG_MAX_BYTES", str(16 * 1024 * 1024)))
    except ValueError:
        worker_log_max_bytes = 16 * 1024 * 1024
    worker_log_written = 0
    worker_line_max = 8192

    def _interesting_worker_line(text: str) -> bool:
        s = text.lstrip()
        return (
            s.startswith("SN125_PROGRESS_JSON ")
            or s.startswith("PROFILE_BREAKDOWN ")
            or "[progress]" in s
            or "[quartile]" in s
            or "[hb]" in s
            or "[mem]" in s
        )

    def _record_worker_line(stream: str, raw: bytes) -> None:
        nonlocal worker_log_written
        if not raw:
            return
        text = raw.decode(errors="replace").replace("\x00", "")
        if len(text) > worker_line_max:
            text = text[:worker_line_max] + "...<truncated>"
        rec = {
            "ts": time.time(),
            "event": "worker.output",
            "stream": stream,
            "task_id": task.task_id,
            "pid": proc.pid if proc is not None else None,
            "line": text.rstrip("\r\n"),
        }
        if worker_log_path and worker_log_written < worker_log_max_bytes:
            try:
                payload = (json.dumps(rec, sort_keys=True, default=float) + "\n").encode()
                if worker_log_written + len(payload) <= worker_log_max_bytes:
                    parent = os.path.dirname(worker_log_path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(worker_log_path, "ab") as f:
                        f.write(payload)
                    worker_log_written += len(payload)
            except Exception:
                pass
        if forward_worker_progress and _interesting_worker_line(text):
            print(f"[worker:{stream}] {text.rstrip()}", flush=True)

    _cgroup_counter = getattr(evaluate_submission_isolated, "_ctr", 0) + 1
    evaluate_submission_isolated._ctr = _cgroup_counter
    cgroup_name = f"sn125_worker_{os.getpid()}_{_cgroup_counter}"
    cgroup_path = _cgroup_create(cgroup_name, memory_max_bytes=24 * 1024**3, pids_max=256) if use_sandbox else None

    proc = None
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=project_root, env=safe_env, start_new_session=True,
        )
        if cgroup_path:
            _cgroup_add_pid(cgroup_path, proc.pid)

        _MAX_STDOUT = 10 * 1024 * 1024
        proc.stdin.write(args_json.encode())
        proc.stdin.close()
        import select
        from collections import deque
        _stdout_chunks = []
        _stdout_len = 0
        _stdout_tail = deque(maxlen=4)
        _stderr_chunks = []
        _open_fds = {proc.stdout, proc.stderr}
        _deadline = time.time() + proc_timeout
        _stdout_line_buf = b""
        _stderr_line_buf = b""

        def _consume_worker_lines(fd, chunk: bytes) -> None:
            nonlocal _stdout_line_buf, _stderr_line_buf
            if fd is proc.stdout:
                _stdout_line_buf += chunk
                lines = _stdout_line_buf.split(b"\n")
                _stdout_line_buf = lines.pop()
                for line in lines:
                    _record_worker_line("stdout", line)
            else:
                _stderr_line_buf += chunk
                lines = _stderr_line_buf.split(b"\n")
                _stderr_line_buf = lines.pop()
                for line in lines:
                    _record_worker_line("stderr", line)

        def _flush_worker_lines() -> None:
            nonlocal _stdout_line_buf, _stderr_line_buf
            if _stdout_line_buf:
                _record_worker_line("stdout", _stdout_line_buf)
                _stdout_line_buf = b""
            if _stderr_line_buf:
                _record_worker_line("stderr", _stderr_line_buf)
                _stderr_line_buf = b""

        while True:
            remaining = _deadline - time.time()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, proc_timeout)
            if not _open_fds:
                proc.wait(timeout=max(remaining, 0.1))
                _flush_worker_lines()
                break
            ready, _, _ = select.select(list(_open_fds), [], [], min(remaining, 1.0))
            for fd in ready:
                chunk = fd.read1(65536) if hasattr(fd, 'read1') else fd.read(65536)
                if not chunk:
                    _open_fds.discard(fd)
                    continue
                if fd is proc.stdout:
                    _stdout_len += len(chunk)
                    if _stdout_len <= _MAX_STDOUT:
                        _stdout_chunks.append(chunk)
                    else:
                        _stdout_tail.append(chunk)
                else:
                    _stderr_chunks.append(chunk[-2048:])
                _consume_worker_lines(fd, chunk)
            if proc.poll() is not None:
                for fd, store, limit in [(proc.stdout, _stdout_chunks, _MAX_STDOUT),
                                          (proc.stderr, _stderr_chunks, 4096)]:
                    while True:
                        chunk = fd.read(65536)
                        if not chunk:
                            break
                        total = sum(len(c) for c in store)
                        if total < limit:
                            store.append(chunk)
                        elif fd is proc.stdout:
                            _stdout_tail.append(chunk)
                        _consume_worker_lines(fd, chunk)
                _flush_worker_lines()
                break
        stdout = b"".join(_stdout_chunks)
        if _stdout_tail:
            stdout += b"\n" + b"".join(_stdout_tail)
        stderr = b"".join(_stderr_chunks)
        result = None
        for line in reversed(stdout.decode(errors="replace").strip().split("\n")):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if parsed.get("_nonce") == _nonce:
                if not (isinstance(parsed.get("eval_points"), list) and
                        isinstance(parsed.get("train_points"), list)):
                    continue
                parsed.pop("_nonce", None)
                result = parsed
                break
        if result is None:
            result = {"task_id": task.task_id, "lr": lr, "wd": wd,
                      "eval_points": [], "train_points": [], "wall_seconds": 0.0,
                      "failed": True, "error": f"No authenticated output. stderr: {stderr.decode(errors='replace')[-500:]}"}
        proc.wait()
        _kill_process_group(proc)
        for fd in (proc.stdout, proc.stderr):
            try: fd.close()
            except Exception: pass
    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
            _kill_process_group(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.waitpid(proc.pid, os.WNOHANG)
                except ChildProcessError: pass
            for fd in (proc.stdin, proc.stdout, proc.stderr):
                try: fd.close()
                except Exception: pass
        result = {"task_id": task.task_id, "lr": lr, "wd": wd,
                  "eval_points": [], "train_points": [], "wall_seconds": proc_timeout,
                  "failed": True, "error": f"Process killed: exceeded {proc_timeout:.0f}s timeout"}
    except Exception as e:
        result = {"task_id": task.task_id, "lr": lr, "wd": wd,
                  "eval_points": [], "train_points": [], "wall_seconds": 0.0,
                  "failed": True, "error": f"Subprocess error: {e}"}
        if proc:
            try: proc.kill()
            except Exception: pass
            _kill_process_group(proc)
            for fd in (proc.stdin, proc.stdout, proc.stderr):
                try: fd.close()
                except Exception: pass
            try: proc.wait(timeout=5)
            except Exception:
                try: os.waitpid(proc.pid, os.WNOHANG)
                except ChildProcessError: pass
    finally:
        if cgroup_path:
            _cgroup_cleanup(cgroup_path)
        if run_dir:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)

    if result.get("failed"):
        _detail_parts = []
        if result.get("traceback"):
            _detail_parts.append("traceback_tail=" + str(result.get("traceback")))
        if result.get("cuda_memory"):
            _detail_parts.append("cuda_memory=" + json.dumps(result.get("cuda_memory"), sort_keys=True))
        if _detail_parts:
            result["error"] = str(result.get("error", "")) + "\n" + "\n".join(_detail_parts)

    result = _enforce_sandbox_attestation(result, task.task_id, lr, wd)

    return TrainingCurve(
        task_id=result.get("task_id", task.task_id),
        lr=result.get("lr", lr), wd=result.get("wd", wd),
        eval_points=result.get("eval_points", []),
        train_points=result.get("train_points", []),
        wall_seconds=result.get("wall_seconds", 0.0),
        state_multiplier=result.get("state_multiplier", 0.0),
        failed=result.get("failed", False),
        error=result.get("error", ""),
        step_times=result.get("step_times", []),
        iter_times=result.get("iter_times", []),
        checkpoints=result.get("checkpoints", []),
    )


def cleanup_data_cache(max_age_s: float = 3600):
    """Remove old pre-generated data files from sn125_data/. Call between rounds.
    Only deletes files older than max_age_s to avoid racing with concurrent evals."""
    data_dir = os.path.join(settings.hf_home(), "sn125_data")
    if os.path.isdir(data_dir):
        cutoff = time.time() - max_age_s
        for f in os.listdir(data_dir):
            fp = os.path.join(data_dir, f)
            try:
                if os.path.getmtime(fp) < cutoff:
                    os.unlink(fp)
            except OSError:
                pass


