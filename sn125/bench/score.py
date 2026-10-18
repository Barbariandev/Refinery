"""Likelihood scoring of a causal LM on a pre-tokenized suite.

Per multiple-choice item, with ``ll_i`` = sum log p(choice_i tokens | ctx):
  acc                 argmax_i ll_i == gold
  acc_norm            argmax_i ll_i / chars_i == gold          (lm-eval byte-norm)
  prob_correct_norm   softmax_i( ll_i * mean_chars / chars_i )[gold]
                      — the model's probability mass on the gold choice after
                      rescaling every choice to the item's mean length. Smooth
                      (no argmax), length-debiased, in (0, 1); PRIMARY metric.
  prob_correct        softmax_i(ll_i)[gold]                   (raw, length-biased)
  bpb_gold            -ll_gold / (bytes_gold * ln 2)           (lower is better)

Per completion item (lambada-style): acc = every target token is the argmax
(teacher-forced), plus loglik / bpb of the target.

The aggregate ``score`` is the mean of ``prob_correct_norm`` over MC tasks
(completion tasks are diagnostic). Everything is reported per task so the
aggregate can be re-weighted later without re-running.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

DEFAULT_ROWS_PER_BATCH = 8


def _hidden_logprobs(model, ids: torch.Tensor, use_amp: bool):
    """Return a callable ``lp(positions) -> (n, V) float32 log-probs`` for one batch of
    rows, using the lean fast path (backbone + head on gathered positions) when the
    model exposes it, else full logits."""
    dev = ids.device
    ac = (torch.amp.autocast(dev.type, dtype=torch.bfloat16)
          if (use_amp and dev.type == "cuda") else nullcontext())
    with torch.no_grad(), ac:
        backbone = getattr(model, "_backbone", None)
        head = getattr(model, "lm_head", None)
        if callable(backbone) and head is not None:
            h = backbone(ids)  # (B, S, H)

            def lp(b: int, pos: torch.Tensor) -> torch.Tensor:
                hs = h[b].index_select(0, pos)
                return F.log_softmax(head(hs).float(), dim=-1)
            return lp
        logits = model(ids).logits

        def lp2(b: int, pos: torch.Tensor) -> torch.Tensor:
            return F.log_softmax(logits[b].index_select(0, pos).float(), dim=-1)
        return lp2


def _rows_for_item(item: dict, eos_id: int, seq_len: int) -> list[tuple[list[int], int]]:
    """(row ids, n_context_tokens) per continuation; ctx left-truncated to fit."""
    rows = []
    conts = item["choices"] if "choices" in item else [item["target"]]
    for cont in conts:
        ctx = [eos_id] + list(item["ctx"])
        room = seq_len - len(cont)
        if room < 1:
            # Never silently truncate what is being scored: the suite is incompatible
            # with this seq_len (not a real risk at seq 2048; guards toy configs).
            raise ValueError(f"continuation of {len(cont)} tokens does not fit seq_len {seq_len}")
        if len(ctx) > room:
            ctx = ctx[-room:]
        rows.append((ctx + list(cont), len(ctx)))
    return rows


@torch.no_grad()
def score_suite(model, suite: dict, *, seq_len: int, device: str | torch.device = "cpu",
                use_amp: bool = False, rows_per_batch: int = DEFAULT_ROWS_PER_BATCH,
                tasks: list[str] | None = None, log=None, per_item: bool = False) -> dict:
    """``per_item=True`` also returns, per task, the per-item value of the primary
    metric (``items_pcn`` for MC, ``items_bpb`` for completion) so two checkpoints
    scored on the same suite can be compared item by item (``paired_delta``)."""
    model.eval()
    device = torch.device(device)
    eos_id = int(suite.get("eos_id", 0))
    results: dict = {"tasks": {}, "suite_sha256": suite.get("suite_sha256", ""),
                     "text_sha256": suite.get("text_sha256", ""), "tokenizer": suite.get("tokenizer", ""),
                     "suite_id": suite.get("suite_id", ""), "seq_len": int(seq_len)}
    t_all = time.time()
    for name, task in suite["tasks"].items():
        if tasks and name not in tasks:
            continue
        kind = task["kind"]
        items = task["items"]
        t0 = time.time()
        # Flatten all (item, continuation) rows, score in batches, then reduce per item.
        flat: list[tuple[int, int, list[int], int]] = []  # (item_idx, cont_idx, ids, n_ctx)
        for ii, it in enumerate(items):
            for ci, (ids, nctx) in enumerate(_rows_for_item(it, eos_id, seq_len)):
                flat.append((ii, ci, ids, nctx))
        ll = [[0.0] * (len(it["choices"]) if kind == "mc" else 1) for it in items]
        greedy_ok = [[True] * (len(it["choices"]) if kind == "mc" else 1) for it in items]
        n_tok = [[0] * (len(it["choices"]) if kind == "mc" else 1) for it in items]
        for s in range(0, len(flat), rows_per_batch):
            chunk = flat[s:s + rows_per_batch]
            L = max(len(r[2]) for r in chunk)
            ids = torch.full((len(chunk), L), eos_id, dtype=torch.long)
            for b, (_, _, row, _) in enumerate(chunk):
                ids[b, :len(row)] = torch.tensor(row, dtype=torch.long)
            ids = ids.to(device)
            lp = _hidden_logprobs(model, ids, use_amp)
            for b, (ii, ci, row, nctx) in enumerate(chunk):
                n = len(row)
                pos = torch.arange(nctx - 1, n - 1, device=device)  # positions predicting the continuation
                tgt = ids[b, nctx:n]
                logp = lp(b, pos)                                     # (n_cont, V)
                tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
                ll[ii][ci] = float(tok_lp.sum().item())
                n_tok[ii][ci] = int(tgt.numel())
                greedy_ok[ii][ci] = bool((logp.argmax(dim=-1) == tgt).all().item())
        # reduce
        tr: dict = {"kind": kind, "n": len(items)}
        if kind == "mc":
            acc = acc_norm = pcn = pc = bpb = 0.0
            item_pcn: list[float] = []
            for ii, it in enumerate(items):
                lls = torch.tensor(ll[ii], dtype=torch.float64)
                chars = torch.tensor(it["choice_chars"], dtype=torch.float64).clamp_min(1.0)
                bytes_ = torch.tensor(it["choice_bytes"], dtype=torch.float64).clamp_min(1.0)
                g = int(it["gold"])
                acc += float(int(lls.argmax()) == g)
                acc_norm += float(int((lls / chars).argmax()) == g)
                item_pcn.append(float(torch.softmax(lls * chars.mean() / chars, dim=0)[g]))
                pcn += item_pcn[-1]
                pc += float(torch.softmax(lls, dim=0)[g])
                bpb += float(-lls[g] / (bytes_[g] * math.log(2)))
            n = max(1, len(items))
            tr.update({"acc": acc / n, "acc_norm": acc_norm / n, "prob_correct_norm": pcn / n,
                       "prob_correct": pc / n, "bpb_gold": bpb / n})
            if per_item:
                tr["items_pcn"] = [round(x, 6) for x in item_pcn]
        else:
            acc = bpb = nll = 0.0
            ntok = 0
            item_bpb: list[float] = []
            for ii, it in enumerate(items):
                acc += float(greedy_ok[ii][0])
                item_bpb.append(float(-ll[ii][0] / (max(1, it["target_bytes"]) * math.log(2))))
                bpb += item_bpb[-1]
                nll += -ll[ii][0]
                ntok += n_tok[ii][0]
            n = max(1, len(items))
            tr.update({"acc": acc / n, "bpb_gold": bpb / n,
                       "ppl": math.exp(nll / max(1, ntok)), "n_target_tokens": ntok})
            if per_item:
                tr["items_bpb"] = [round(x, 6) for x in item_bpb]
        tr["seconds"] = time.time() - t0
        results["tasks"][name] = tr
        if log:
            log(f"[bench] {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in tr.items()
                                                 if isinstance(v, float) and k != "seconds")
                + f" ({tr['seconds']:.1f}s)")
    mc = [t["prob_correct_norm"] for t in results["tasks"].values() if t["kind"] == "mc"]
    results["score"] = (sum(mc) / len(mc)) if mc else 0.0
    results["mean_acc_norm"] = (sum(t["acc_norm"] for t in results["tasks"].values() if t["kind"] == "mc")
                                / len(mc)) if mc else 0.0
    results["seconds"] = time.time() - t_all
    return results


def paired_delta(sub: dict, base: dict, *, n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Item-paired comparison of two ``score_suite(per_item=True)`` results on the
    same suite: the suite-score delta (mean over MC tasks of the mean per-item
    ``prob_correct_norm`` difference) with a bootstrap standard error and 95 %
    interval, resampling items within each task.

    This quantifies ITEM-sampling noise only — how much the delta would move on
    another draw of benchmark items. It says nothing about run-to-run training
    noise (seed, hardware, step count), which needs repeated runs."""
    if sub.get("suite_sha256") != base.get("suite_sha256"):
        return None
    diffs = []
    for name, t in sub.get("tasks", {}).items():
        b = base.get("tasks", {}).get(name, {})
        if t.get("kind") != "mc":
            continue
        xs, ys = t.get("items_pcn"), b.get("items_pcn")
        if not xs or not ys or len(xs) != len(ys):
            return None
        diffs.append(torch.tensor(xs, dtype=torch.float64) - torch.tensor(ys, dtype=torch.float64))
    if not diffs:
        return None
    g = torch.Generator().manual_seed(int(seed))
    boots = torch.zeros(n_boot, dtype=torch.float64)
    for d in diffs:
        idx = torch.randint(0, d.numel(), (n_boot, d.numel()), generator=g)
        boots += d[idx].mean(dim=1)
    boots /= len(diffs)
    delta = float(sum(float(d.mean()) for d in diffs) / len(diffs))
    se = float(boots.std(unbiased=True))
    lo, hi = (float(x) for x in torch.quantile(boots, torch.tensor([0.025, 0.975], dtype=torch.float64)))
    return {"delta": delta, "se_items": se, "ci95_items": [lo, hi], "z_items": (delta / se) if se > 0 else 0.0,
            "n_boot": int(n_boot), "per_task": {name: float(d.mean()) for name, d in
                                                zip([n for n, t in sub["tasks"].items() if t.get("kind") == "mc"], diffs)}}


# ── reference base models ────────────────────────────────────────────────────
# Pretrained models scorable on the SmolLM2-tokenized suite. They are loaded ONLY
# from a local safetensors file — validated on the raw bytes first, never via a
# framework loader, never with remote code, never over the network — into the
# lean reimplementation of the same architecture, so reference and student
# numbers come from the identical forward pass.
# Value = a LEAN_CONFIGS name, or LeanConfig overrides on top of the SmolLM2-360M
# defaults (each entry checked against the repo's config.json; the weight-inventory
# check below catches any shape drift).
_SMOL_135M = {"hidden_size": 576, "intermediate_size": 1536, "num_hidden_layers": 30,
              "num_attention_heads": 9, "num_key_value_heads": 3}
BASE_MODELS: dict[str, str | dict] = {
    "HuggingFaceTB/SmolLM2-360M": "lean-llama-360m",
    "HuggingFaceTB/SmolLM2-135M": "lean-llama-135m",
    # SmolLM v1 (600B tokens): same shapes as v2, RoPE theta 1e4, 2k context.
    "HuggingFaceTB/SmolLM-360M": {"rope_theta": 10000.0, "max_position_embeddings": 2048},
    "HuggingFaceTB/SmolLM-135M": {**_SMOL_135M, "rope_theta": 10000.0, "max_position_embeddings": 2048},
    # SmolLM2-1.7B (11T tokens): MHA (32 kv heads), RoPE theta 1.3e5.
    "HuggingFaceTB/SmolLM2-1.7B": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 24,
                                   "num_attention_heads": 32, "num_key_value_heads": 32, "rope_theta": 130000.0},
}
MAX_BASE_MODEL_BYTES = 4 * 1024**3
_HF_PREFIX = "model."


def _lean_twin(name: str, dtype: torch.dtype | None):
    """Deterministically construct the lean twin of an allowlisted base model.
    Returns ``(model, config_label)``."""
    import copy
    from ..lean_model import LEAN_CONFIGS, LeanConfig, LeanLlama, maybe_compile_lean
    spec = BASE_MODELS[name]
    if isinstance(spec, str):
        cfg, label = copy.copy(LEAN_CONFIGS[spec]), spec
    else:
        cfg = LeanConfig(**spec)
        label = "lean-llama(" + ",".join(f"{k}={v}" for k, v in sorted(spec.items())) + ")"
    torch.manual_seed(0)
    model = LeanLlama(cfg)
    if dtype is not None:
        model = model.to(dtype)
    return maybe_compile_lean(model), label


def resolve_base_model_file(name: str) -> str:
    """Local HF-cache path of ``model.safetensors`` for an allowlisted base model.
    Never downloads: a missing file is an error, not a fetch."""
    if name not in BASE_MODELS:
        raise ValueError(f"base model {name!r} is not allowlisted; known: {sorted(BASE_MODELS)}")
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(name, "model.safetensors")
    if not isinstance(p, str):
        raise FileNotFoundError(f"{name}: model.safetensors is not in the local HF cache "
                                f"(the bench never downloads; stage it explicitly)")
    return p


def load_base_model(name: str, *, device: str = "cpu", dtype: torch.dtype | None = None,
                    path: str | None = None):
    """Build the lean twin of an allowlisted base model and load its HF safetensors
    (``model.`` prefix stripped, tied ``lm_head``) after raw-byte validation.
    Returns ``(model, provenance)``."""
    from safetensors import safe_open
    from ..engine.score_worker import CheckpointRejected, _sha256, validate_safetensors_file
    path = path or resolve_base_model_file(name)
    model, cfg_name = _lean_twin(name, dtype)
    want = {k: tuple(t.shape) for k, t in model.state_dict().items() if k != "lm_head.weight"}
    info = validate_safetensors_file(path, max_bytes=MAX_BASE_MODEL_BYTES)
    got = {k[len(_HF_PREFIX):]: v for k, v in info["tensors"].items() if k.startswith(_HF_PREFIX)}
    if len(got) != len(info["tensors"]) or set(got) != set(want):
        raise CheckpointRejected(f"{name}: tensor inventory does not match {cfg_name}")
    for k, (shape, dt) in got.items():
        if shape != want[k] or dt not in ("F32", "BF16", "F16"):
            raise CheckpointRejected(f"{name}: {k!r} is {shape}/{dt}, expected {want[k]}/float")
    target_dtype = model.embed_tokens.weight.dtype
    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in want:
            state[k] = f.get_tensor(_HF_PREFIX + k).to(target_dtype)
    state["lm_head.weight"] = state["embed_tokens.weight"]  # tied, as in the source config
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {"base_model": name, "lean_config": cfg_name, "weights_path": path,
                   "weights_sha256": _sha256(path), "weights_bytes": int(info["file_bytes"])}


def score_base_model_on_suite(name: str, suite: dict, *, seq_len: int,
                              device: str = "cuda" if torch.cuda.is_available() else "cpu",
                              use_amp: bool = True, rows_per_batch: int = DEFAULT_ROWS_PER_BATCH,
                              path: str | None = None, log=None) -> dict:
    """Reference numbers for the suite from a pretrained base model (see BASE_MODELS)."""
    dtype = torch.bfloat16 if (use_amp and device.startswith("cuda")) else None
    model, prov = load_base_model(name, device=device, dtype=dtype, path=path)
    res = score_suite(model, suite, seq_len=seq_len, device=device, use_amp=use_amp,
                      rows_per_batch=rows_per_batch, log=log)
    res.update(prov)
    res["model_spec"] = f"hf:{prov['lean_config']}"
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


# ── reference models outside the SmolLM2 tokenizer family ───────────────────
# Scored on a suite rebuilt with THEIR tokenizer from the same text items (the
# suites share ``text_sha256``); acc / acc_norm / prob_correct_norm / bpb are
# text-level, so they compare across tokenizers (token-level lambada ``ppl``
# does not). Loading is confined to: allowlisted name, native transformers
# architecture (remote code refused), local HF cache only, safetensors only,
# every file raw-byte validated before the framework parses it.
HF_REFERENCE_MODELS: dict[str, int] = {  # name -> max positions (caps --seq-len)
    "gpt2": 1024,
    "EleutherAI/pythia-160m": 2048,
    "EleutherAI/pythia-410m": 2048,
    "Qwen/Qwen2.5-0.5B": 32768,
    "Qwen/Qwen3-0.6B-Base": 32768,
}
MAX_HF_REFERENCE_BYTES = 8 * 1024**3


def resolve_hf_snapshot(name: str, revision: str | None = None) -> str:
    """Local HF-cache snapshot directory of an allowlisted reference model.
    Never downloads: a missing snapshot is an error, not a fetch."""
    if name not in HF_REFERENCE_MODELS:
        raise ValueError(f"reference model {name!r} is not allowlisted; known: {sorted(HF_REFERENCE_MODELS)}")
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(name, revision=revision, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - hub raises several "not cached" types
        raise FileNotFoundError(f"{name}@{revision or 'main'}: not in the local HF cache "
                                f"(the bench never downloads; stage it explicitly): {e}") from None


def load_hf_reference_model(name: str, *, revision: str | None = None, device: str = "cpu",
                            dtype: torch.dtype | None = None, snapshot: str | None = None):
    """Load an allowlisted reference model from its local snapshot, safetensors only,
    after raw-byte validation of every weight file. Returns ``(model, provenance)``."""
    from pathlib import Path
    from ..engine.score_worker import CheckpointRejected, _sha256, validate_safetensors_file
    snap = snapshot or resolve_hf_snapshot(name, revision)
    files = sorted(Path(snap).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{name}: no safetensors weights in {snap}")
    total = 0
    shas = {}
    for f in files:
        info = validate_safetensors_file(str(f), max_bytes=MAX_HF_REFERENCE_BYTES)
        total += int(info["file_bytes"])
        shas[f.name] = _sha256(str(f))
    if total > MAX_HF_REFERENCE_BYTES:
        raise CheckpointRejected(f"{name}: weights total {total} bytes > cap {MAX_HF_REFERENCE_BYTES}")
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=False, local_files_only=True)
    if getattr(cfg, "auto_map", None):
        raise CheckpointRejected(f"{name}: config requests remote code (auto_map); refused")
    model = AutoModelForCausalLM.from_pretrained(
        snap, config=cfg, use_safetensors=True, trust_remote_code=False, local_files_only=True,
        dtype=dtype or torch.float32)
    model.to(device).eval()
    return model, {"base_model": name, "revision": revision or "main", "architecture": type(model).__name__,
                   "weights_path": snap, "weights_sha256": shas, "weights_bytes": total,
                   "n_params": int(sum(p.numel() for p in model.parameters()))}


def score_hf_reference_on_suite(name: str, suite: dict, *, seq_len: int, revision: str | None = None,
                                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                                use_amp: bool = True, rows_per_batch: int = DEFAULT_ROWS_PER_BATCH,
                                snapshot: str | None = None, log=None) -> dict:
    """Reference numbers from a non-SmolLM2-tokenizer model (see HF_REFERENCE_MODELS).
    The suite must have been tokenized with the model's own tokenizer."""
    if suite.get("tokenizer") != name:
        raise ValueError(f"suite was tokenized with {suite.get('tokenizer')!r}, not {name!r}; "
                         f"rebuild it with `bench-build --tokenizer {name}`")
    if name in HF_REFERENCE_MODELS and seq_len > HF_REFERENCE_MODELS[name]:
        raise ValueError(f"{name} supports at most {HF_REFERENCE_MODELS[name]} positions; seq_len {seq_len} too long")
    dtype = torch.bfloat16 if (use_amp and device.startswith("cuda")) else None
    model, prov = load_hf_reference_model(name, revision=revision, device=device, dtype=dtype, snapshot=snapshot)
    res = score_suite(model, suite, seq_len=seq_len, device=device, use_amp=use_amp,
                      rows_per_batch=rows_per_batch, log=log)
    res.update(prov)
    res["model_spec"] = f"hf:{name}@{prov['revision']}"
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def score_checkpoint_on_suite(checkpoint_path: str, suite: dict, model_spec: str, *, seq_len: int,
                              device: str = "cuda" if torch.cuda.is_available() else "cpu",
                              use_amp: bool = True, rows_per_batch: int = DEFAULT_ROWS_PER_BATCH,
                              log=None, per_item: bool = False) -> dict:
    """Trusted path: build the model from spec, validate + load the safetensors
    checkpoint exactly as the clean-score worker does, then score."""
    from ..engine.score_worker import (_sha256, build_model_from_spec, expected_tensors_of,
                                       validate_safetensors_file)
    from safetensors.torch import load_model
    model = build_model_from_spec(model_spec, device=device,
                                  dtype=torch.bfloat16 if (use_amp and device.startswith("cuda")) else None)
    model = model.to(device)
    expected = expected_tensors_of(model)
    validate_safetensors_file(checkpoint_path, expected_tensors=expected)
    missing, unexpected = load_model(model, checkpoint_path, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint state mismatch: missing={missing} unexpected={unexpected}")
    res = score_suite(model, suite, seq_len=seq_len, device=device, use_amp=use_amp,
                      rows_per_batch=rows_per_batch, log=log, per_item=per_item)
    res["checkpoint_sha256"] = _sha256(checkpoint_path)
    res["model_spec"] = model_spec
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res
