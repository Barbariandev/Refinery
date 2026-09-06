"""Lean, dependency-free LLaMA-style decoder for the Refinery training substrate.

nanoGPT in spirit (one small auditable file, no `transformers`/AutoModel), but with
the modern components the production substrate uses — RoPE, RMSNorm, SwiGLU, GQA —
sized to match SmolLM2-360M so the substrate stays a faithful, representative LLM
training workload. Built fully INLINE (every hyperparameter here): as auditable as
source, no gated repo, no `trust_remote_code`, no expansion of the trust surface.

WHY THIS EXISTS (2026-06-26): the HF `transformers` path OOM'd the isolated scoring
worker — its `ForCausalLMLoss` materializes the full (B*S, vocab) logits AND an fp32
cast (~12.9GB at batch32/seq2048/vocab49152), and that churn physically fragments
device memory so a later small cudaMalloc fails amid free VRAM. This module's
`fused_ce` path tiles the final projection + cross-entropy over the sequence so the
full logits tensor is NEVER materialized — bounding peak memory to one tile and
removing the dominant fragmenting allocation. It also enables the banked V2 levers
(clean torch.compile / FP8) later.

DETERMINISM (operator 2026-06-26): seed everything cheap, leave the expensive kernel
nondeterminism. Init is deterministic under the caller's `torch.manual_seed(seed)`
(applied in fixed module order); dropout is 0; the forward calls no unseeded RNG.
Data order/batching stay CRN-seeded in fineweb.py. Attention uses
F.scaled_dot_product_attention so the harness's flash regime (SN125_DIAG_FLASH)
applies unchanged; that flash/cuDNN nondeterminism is the only residual, by design.

Output objects expose `.logits` and `.loss` so the existing harness
(`model(batch).logits`, `model(batch, labels=batch).loss`, eval, checkpoint export,
`named_parameters()` IPC contract) works without changes.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


@dataclass
class LeanConfig:
    vocab_size: int = 49152
    hidden_size: int = 960
    intermediate_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 15
    num_key_value_heads: int = 5
    max_position_embeddings: int = 8192
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    loss_chunks: int = 8
    loss_ckpt: bool = True


LEAN_CONFIGS = {
    "lean-llama-360m": LeanConfig(),
    "lean-llama-135m": LeanConfig(hidden_size=576, intermediate_size=1536,
                                  num_hidden_layers=30, num_attention_heads=9,
                                  num_key_value_heads=3),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def _rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: LeanConfig):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.rep = self.nh // self.nkv
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nkv, self.hd).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if self.rep > 1:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, S, self.nh * self.hd)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg: LeanConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: LeanConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


@dataclass
class _Out:
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class LeanLlama(nn.Module):
    def __init__(self, cfg: LeanConfig):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._grad_ckpt = False
        self.uses_fused_lm_head_ce = True
        _cos, _sin = _rope_cache(cfg.max_position_embeddings,
                                 cfg.hidden_size // cfg.num_attention_heads,
                                 cfg.rope_theta, "cpu", torch.float32)
        self.register_buffer("_rope_cos", _cos, persistent=False)
        self.register_buffer("_rope_sin", _sin, persistent=False)
        self.apply(self._init)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init(self, m):
        std = self.cfg.initializer_range
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def gradient_checkpointing_enable(self, **kwargs):
        self._grad_ckpt = True

    def gradient_checkpointing_disable(self):
        self._grad_ckpt = False

    def _get_rope(self, seq_len: int, head_dim: int, device, dtype):
        return (self._rope_cos[:seq_len].to(dtype),
                self._rope_sin[:seq_len].to(dtype))

    def _backbone(self, idx):
        B, S = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = self._get_rope(S, self.cfg.hidden_size // self.cfg.num_attention_heads,
                                  idx.device, x.dtype)
        for blk in self.layers:
            if self._grad_ckpt and self.training:
                x = _ckpt(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return self.norm(x)

    def forward(self, input_ids, labels=None):
        h = self._backbone(input_ids)
        if labels is None:
            return _Out(logits=self.lm_head(h))
        B, S, H = h.shape
        hs = h.reshape(-1, H)
        ys2 = labels.new_full((B, S), -100)
        ys2[:, :-1] = labels[:, 1:]
        ys = ys2.reshape(-1)
        n = hs.size(0)
        chunks = max(1, int(getattr(self.cfg, "loss_chunks", 1) or 1))
        step = (n + chunks - 1) // chunks
        total = hs.new_zeros((), dtype=torch.float32)
        count = B * max(0, S - 1)
        w = self.lm_head.weight

        def _chunk_loss(hc, yc):
            return F.cross_entropy(F.linear(hc, w).float(), yc,
                                   ignore_index=-100, reduction="sum")

        use_ckpt = self.training and chunks > 1 and getattr(self.cfg, "loss_ckpt", True)
        for i in range(0, n, step):
            hc = hs[i:i + step]
            yc = ys[i:i + step]
            if use_ckpt:
                s = _ckpt(_chunk_loss, hc, yc, use_reentrant=False)
            else:
                s = _chunk_loss(hc, yc)
            total = total + s
        return _Out(loss=total / max(1, count))


def maybe_compile_lean(model: nn.Module) -> nn.Module:
    """Apply torch.compile per SN125_LEAN_COMPILE (default off; "1"/mode string on).

    Uses the IN-PLACE nn.Module.compile() so the module keeps its identity:
    named_parameters() stays un-prefixed (no "_orig_mod."), which the optproc
    CUDA-IPC contract, checkpoint export, and param_groups naming all rely on.
    Values: "0"/"" = eager; "1" = default mode; anything else is passed through
    as the inductor mode (e.g. "max-autotune"). Failure falls back to eager —
    a slow run beats a failed paid eval.
    """
    mode = os.environ.get("SN125_LEAN_COMPILE", "").strip()
    if not mode or mode == "0":
        return model
    try:
        kwargs = {} if mode == "1" else {"mode": mode}
        model.compile(**kwargs)
    except Exception as e:  # pragma: no cover - depends on backend availability
        import logging
        logging.getLogger(__name__).warning(
            "SN125_LEAN_COMPILE=%s failed (%s); continuing eager", mode, e)
    return model


def build_lean_model(config_name: str, seed: int, dtype: torch.dtype | None = None,
                     loss_chunks: int | None = None) -> nn.Module:
    """Construct a lean model by name under a deterministic seed (mirrors _build_model)."""
    import copy
    cfg = copy.copy(LEAN_CONFIGS[config_name])
    if loss_chunks is not None:
        cfg.loss_chunks = loss_chunks
    _ckpt_env = os.environ.get("SN125_LEAN_LOSS_CKPT", "").strip()
    if _ckpt_env:
        cfg.loss_ckpt = _ckpt_env != "0"
    torch.manual_seed(seed)
    model = LeanLlama(cfg)
    if dtype is not None:
        model = model.to(dtype)
    return maybe_compile_lean(model)
