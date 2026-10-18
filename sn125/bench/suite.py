"""Build / load the pre-tokenized benchmark suite.

A suite is ONE JSON file, tokenized once with the production tokenizer so the
scorer never needs the tokenizer (or network) at scoring time, and so that the
exact token sequences every submission is judged on are pinned by a hash
(`suite_sha256`, recorded in every result).

Item kinds
  mc          {"ctx": [ids], "choices": [[ids], ...], "gold": k,
               "choice_chars": [...], "choice_bytes": [...]}
  completion  {"ctx": [ids], "target": [ids], "target_chars": n, "target_bytes": n}

Every scored sequence is ``[EOS] + ctx + continuation`` — EOS first so the
model sees the same "document start" it was trained on (shards join documents
with EOS). Context is left-truncated by the scorer to fit ``seq_len``.

Sources (text → items):
  * ``*.jsonl`` files with {"ctx","choices","gold"} or {"ctx","target"} lines
    (the pilot files under data/pilot_encdec_v0/bench are in this shape);
  * HF datasets (hellaswag, piqa, arc_easy, sciq, openbookqa, lambada) via the
    ``datasets`` library when cached/online — prompts follow lm-eval-harness.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

from ..fineweb import TOKENIZER_NAME

SUITE_VERSION = 1
DEFAULT_SUITE_ID = "easy_v0"
DEFAULT_LIMIT = 1000  # items per task (deterministic subsample, seed below)
SUBSAMPLE_SEED = 20261003
# The placeholder suite: easy, old, public. Intentionally not the frontier.
DEFAULT_TASKS = ("hellaswag", "piqa", "arc_easy", "sciq", "openbookqa", "lambada")


# ── text-level item construction (lm-eval-harness style prompts) ───────────

def _hellaswag_preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _hf(name: str, *args, split: str, **kw):
    from datasets import load_dataset
    return load_dataset(name, *args, split=split, **kw)


def text_items_from_hf(task: str) -> list[dict]:
    """Pull one benchmark from HF datasets and render lm-eval-style text items."""
    if task == "hellaswag":
        ds = _hf("Rowan/hellaswag", split="validation")
        out = []
        for d in ds:
            ctx = d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()
            out.append({"ctx": _hellaswag_preprocess(ctx),
                        "choices": [" " + _hellaswag_preprocess(e) for e in d["endings"]],
                        "gold": int(d["label"])})
        return out
    if task == "piqa":
        try:
            ds = _hf("ybisk/piqa", split="validation", trust_remote_code=True)
        except Exception:
            ds = _hf("baber/piqa", split="validation")
        return [{"ctx": f"Question: {d['goal']}\nAnswer:", "choices": [" " + d["sol1"], " " + d["sol2"]],
                 "gold": int(d["label"])} for d in ds]
    if task == "arc_easy":
        ds = _hf("allenai/ai2_arc", "ARC-Easy", split="test")
        out = []
        for d in ds:
            labels = d["choices"]["label"]
            out.append({"ctx": f"Question: {d['question']}\nAnswer:",
                        "choices": [" " + t for t in d["choices"]["text"]],
                        "gold": labels.index(d["answerKey"])})
        return out
    if task == "sciq":
        ds = _hf("allenai/sciq", split="test")
        out = []
        for d in ds:
            choices = [d["distractor1"], d["distractor2"], d["distractor3"], d["correct_answer"]]
            support = (d.get("support") or "").strip()
            ctx = (support + "\n" if support else "") + f"Question: {d['question']}\nAnswer:"
            out.append({"ctx": ctx, "choices": [" " + c for c in choices], "gold": 3})
        return out
    if task == "openbookqa":
        ds = _hf("allenai/openbookqa", "main", split="test")
        out = []
        for d in ds:
            labels = d["choices"]["label"]
            out.append({"ctx": d["question_stem"], "choices": [" " + t for t in d["choices"]["text"]],
                        "gold": labels.index(d["answerKey"])})
        return out
    if task == "lambada":
        ds = _hf("EleutherAI/lambada_openai", "default", split="test")
        out = []
        for d in ds:
            text = d["text"]
            ctx, _, target = text.rpartition(" ")
            out.append({"ctx": ctx, "target": " " + target})
        return out
    raise ValueError(f"unknown benchmark task {task!r}")


def text_items_from_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def subsample(items: list[dict], limit: int | None, seed: int = SUBSAMPLE_SEED) -> list[dict]:
    """Deterministic subsample: fixed-seed permutation of the source order, first ``limit``."""
    if limit is None or limit <= 0 or len(items) <= limit:
        return list(items)
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    return [items[i] for i in sorted(idx[:limit])]


# ── tokenization ────────────────────────────────────────────────────────────

class _Tok:
    def __init__(self, name: str = TOKENIZER_NAME):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.name = name
        eos = self.tok.eos_token_id
        self.eos_id = int(eos if eos is not None else 0)

    def encode(self, s: str) -> list[int]:
        return list(self.tok(s, add_special_tokens=False)["input_ids"])

    def encode_pair(self, ctx: str, cont: str) -> tuple[list[int], list[int]]:
        """Encode ctx and continuation so that ctx_ids + cont_ids == encode(ctx + cont)
        whenever the tokenizer splits cleanly at the boundary (lm-eval convention)."""
        whole = self.encode(ctx + cont)
        ctx_ids = self.encode(ctx)
        if whole[:len(ctx_ids)] == ctx_ids and len(whole) > len(ctx_ids):
            return ctx_ids, whole[len(ctx_ids):]
        return ctx_ids, self.encode(cont)


def tokenize_items(items: list[dict], tok: _Tok) -> tuple[str, list[dict]]:
    kind = None
    out = []
    for it in items:
        if "choices" in it:
            k = "mc"
            ctx_ids = None
            choices = []
            for c in it["choices"]:
                ci, cc = tok.encode_pair(it["ctx"], c)
                ctx_ids = ci if ctx_ids is None else ctx_ids
                choices.append(cc)
            if ctx_ids is None or any(len(c) == 0 for c in choices):
                continue
            out.append({"ctx": ctx_ids, "choices": choices, "gold": int(it["gold"]),
                        "choice_chars": [len(c) for c in it["choices"]],
                        "choice_bytes": [len(c.encode("utf-8")) for c in it["choices"]]})
        elif "target" in it:
            k = "completion"
            ctx_ids, tgt = tok.encode_pair(it["ctx"], it["target"])
            if not tgt:
                continue
            out.append({"ctx": ctx_ids, "target": tgt, "target_chars": len(it["target"]),
                        "target_bytes": len(it["target"].encode("utf-8"))})
        else:
            raise ValueError("item must have 'choices' or 'target'")
        if kind is None:
            kind = k
        elif kind != k:
            raise ValueError("mixed item kinds in one task")
    return kind or "mc", out


def suite_sha256(suite: dict) -> str:
    body = {k: v for k, v in suite.items() if k != "suite_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_suite(*, tasks=DEFAULT_TASKS, jsonl_dir: str | Path | None = None, limit: int | None = DEFAULT_LIMIT,
                tokenizer_name: str = TOKENIZER_NAME, suite_id: str = DEFAULT_SUITE_ID,
                allow_network: bool = True, log=print) -> dict:
    """Assemble + tokenize the suite. Per task: a ``<task>.jsonl`` in ``jsonl_dir`` wins;
    otherwise the HF dataset (if reachable); a task that resolves to nothing is skipped
    (and listed under ``skipped``) so a trimmed box still produces a scorable suite."""
    tok = _Tok(tokenizer_name)
    suite = {"version": SUITE_VERSION, "suite_id": suite_id, "tokenizer": tok.name,
             "eos_id": tok.eos_id, "created_unix": int(time.time()), "limit": limit,
             "subsample_seed": SUBSAMPLE_SEED, "tasks": {}, "skipped": {}}
    # Hash of the TEXT items (post-subsample, pre-tokenization): two suites built
    # from the same items with different tokenizers share it, which is what makes
    # their char/byte-normalized metrics comparable across model families.
    text_hash = hashlib.sha256()
    for task in tasks:
        items = None
        source = ""
        if jsonl_dir is not None and (Path(jsonl_dir) / f"{task}.jsonl").exists():
            items = text_items_from_jsonl(Path(jsonl_dir) / f"{task}.jsonl")
            source = f"jsonl:{Path(jsonl_dir) / f'{task}.jsonl'}"
        else:
            # `datasets` serves cached copies offline; with allow_network=False we
            # pin HF_HUB_OFFLINE so a missing cache fails fast instead of hanging.
            import os
            prev = os.environ.get("HF_HUB_OFFLINE")
            if not allow_network:
                os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                items = text_items_from_hf(task)
                source = f"hf:{task}"
            except Exception as e:  # noqa: BLE001 - network/cache absence is a normal condition
                suite["skipped"][task] = f"{type(e).__name__}: {str(e)[:200]}"
                log(f"[bench] skip {task}: {type(e).__name__}: {str(e)[:120]}")
                continue
            finally:
                if not allow_network:
                    if prev is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = prev
        items = subsample(items, limit)
        text_hash.update(json.dumps({task: items}, sort_keys=True, separators=(",", ":")).encode())
        kind, tok_items = tokenize_items(items, tok)
        suite["tasks"][task] = {"kind": kind, "n": len(tok_items), "source": source, "items": tok_items}
        log(f"[bench] {task}: {len(tok_items)} items ({kind}) from {source}")
    suite["text_sha256"] = text_hash.hexdigest()
    suite["suite_sha256"] = suite_sha256(suite)
    return suite


def save_suite(suite: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(suite, separators=(",", ":")))
    tmp.replace(p)
    return p


def load_suite(path: str | Path) -> dict:
    suite = json.loads(Path(path).read_text())
    if suite.get("version") != SUITE_VERSION:
        raise ValueError(f"unsupported suite version {suite.get('version')}")
    expect = suite_sha256(suite)
    if suite.get("suite_sha256") != expect:
        raise ValueError(f"suite hash mismatch: file says {suite.get('suite_sha256')}, content is {expect}")
    return suite


def default_suite_path(data_dir: str | Path | None = None) -> Path:
    from ..fineweb import default_data_dir
    base = Path(data_dir) if data_dir else default_data_dir()
    return base / "bench" / f"suite_{DEFAULT_SUITE_ID}.json"
