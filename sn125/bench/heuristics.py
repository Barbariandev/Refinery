"""Model-free floors for a suite: what a scorer with NO language knowledge gets.

``chance`` is the uniform model. ``copy_prior`` is a bag-of-context model: the
next token is, with probability ``mix``, a uniformly drawn token of the item's
context, otherwise uniform over the vocabulary. It knows nothing except "words
that already appeared are likely to appear again", which every language model
picks up early. A suite whose score moves a lot under this heuristic is
measuring in-context copying, not knowledge; the gap between the copy-prior
floor and a reference model is the dynamic range that is left for everything
else. Reported by ``sn125 bench-floor`` and recorded next to the reference
ladder (docs/REFINER_TASK.md §6).
"""
from __future__ import annotations

import math


def _pcn(lls: list[float], chars: list[float], gold: int) -> float:
    mean = sum(chars) / len(chars)
    z = [ll * mean / max(1.0, c) for ll, c in zip(lls, chars)]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return e[gold] / sum(e)


def suite_floors(suite: dict, *, vocab_size: int = 49152, mix: float = 0.5) -> dict:
    """Per-task and aggregate ``prob_correct_norm`` (MC) / in-context rate
    (completion) for the uniform and the copy-prior scorer."""
    out: dict = {"mix": float(mix), "vocab_size": int(vocab_size), "tasks": {}}
    u = (1.0 - mix) / vocab_size
    for name, task in suite["tasks"].items():
        items = task["items"]
        if task["kind"] != "mc":
            hit = sum(all(t in set(it["ctx"]) for t in it["target"]) for it in items)
            out["tasks"][name] = {"kind": task["kind"], "target_in_context": hit / max(1, len(items))}
            continue
        chance = copy = 0.0
        for it in items:
            ctx = it["ctx"]
            counts: dict[int, int] = {}
            for t in ctx:
                counts[t] = counts.get(t, 0) + 1
            n = max(1, len(ctx))
            chars = [float(c) for c in it["choice_chars"]]
            g = int(it["gold"])
            chance += _pcn([len(c) * math.log(1.0 / vocab_size) for c in it["choices"]], chars, g)
            copy += _pcn([sum(math.log(u + mix * counts.get(t, 0) / n) for t in c) for c in it["choices"]], chars, g)
        n_items = max(1, len(items))
        out["tasks"][name] = {"kind": "mc", "chance": chance / n_items, "copy_prior": copy / n_items}
    mc = [t for t in out["tasks"].values() if t["kind"] == "mc"]
    out["chance"] = sum(t["chance"] for t in mc) / max(1, len(mc))
    out["copy_prior"] = sum(t["copy_prior"] for t in mc) / max(1, len(mc))
    return out
