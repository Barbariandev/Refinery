"""Refiner submission limits and the AST-only CAPABILITIES reader.

Kept free of torch so the round loop's pre-selection gate and the miner can
import them; ``refproc`` re-exports everything here.
"""
from __future__ import annotations

REFINER_TASK_ID = "RF1_smol_360M"

# ── submission limits (docs/REFINER_TASK.md) ────────────────────────────────
REFINER_MAX_PARAMS = 100_000_000          # total elements across all weight tensors
REFINER_MAX_WEIGHT_BYTES = 450 * 1024**2  # file size cap (bf16/fp16 at the param cap + headroom)
REFINER_MAX_TENSORS = 4096
REFINER_ALLOWED_DTYPES = ("F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL")
REFINER_GPU_MEMORY_FRACTION = 0.35        # share of the device the refiner process may allocate
DEFAULT_PLAN_TIMEOUT_S = 120.0
MAX_WINDOW_ENTRIES = 256


class RefProcError(RuntimeError):
    """Fatal error from / about the refiner process (miner fault unless noted)."""


class RefProcTimeout(RefProcError):
    """Refiner failed to answer within the deadline (process killed)."""


def extract_refiner_capabilities(source: str, strict: bool = False) -> dict:
    """AST-only read of the optional ``CAPABILITIES`` dict (never executes source).

    A ``CAPABILITIES`` that is not a literal dict cannot be read this way and
    falls back to the defaults; ``strict=True`` (the submission gate) raises
    instead, so a miner is told rather than silently run without its signals."""
    import ast
    caps = {"wants_token_losses": False, "window_entries": 16}
    found = False
    try:
        tree = ast.parse(source)
        raw = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                t = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(t, ast.Name) and t.id == "CAPABILITIES" and node.value is not None:
                    found = True
                    raw = ast.literal_eval(node.value)
                    break
        if found and strict and not isinstance(raw, dict):
            raise ValueError("not a dict")
        if isinstance(raw, dict):
            if "wants_token_losses" in raw:
                caps["wants_token_losses"] = bool(raw["wants_token_losses"])
            if "window_entries" in raw:
                caps["window_entries"] = max(1, min(MAX_WINDOW_ENTRIES, int(raw["window_entries"])))
    except Exception as e:
        if found and strict:
            raise RefProcError(f"CAPABILITIES must be a literal dict of literals ({type(e).__name__}: {e})")
        return caps
    return caps
