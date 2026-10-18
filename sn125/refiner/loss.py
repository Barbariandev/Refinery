"""Model-agnostic wrappers for the two Refiner-task primitives.

``LeanLlama`` implements both natively (fused, tiled head + CE; the weighted
loss is a branch of its compiled ``forward``). Any other causal LM
exposing ``model(ids).logits`` (the HF contract, the test ToyCausalLM) gets an
exact but unfused fallback, so the whole path is testable on CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_lm_loss(model, input_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """sum_t w_t * CE_t / sum_t w_t over positions t >= 1 (w aligned with input_ids)."""
    fn = getattr(model, "weighted_loss", None)
    if callable(fn):
        return fn(input_ids, weights)
    logits = model(input_ids).logits
    B, S, V = logits.shape
    w = weights[:, 1:].reshape(-1).to(torch.float32)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), input_ids[:, 1:].reshape(-1),
                         reduction="none")
    return (ce * w).sum() / w.sum().clamp_min(1e-12)


@torch.no_grad()
def token_losses(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position CE ``(B, S)``; column 0 is 0 (no prediction for the first token)."""
    fn = getattr(model, "token_losses", None)
    if callable(fn):
        return fn(input_ids)
    logits = model(input_ids).logits
    B, S, V = logits.shape
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), input_ids[:, 1:].reshape(-1),
                         reduction="none").view(B, S - 1)
    out = torch.zeros((B, S), dtype=torch.float32, device=logits.device)
    out[:, 1:] = ce
    return out
