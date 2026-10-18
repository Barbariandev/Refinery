#!/usr/bin/env python3
"""Offline trainer for the `select` refiner's weight file: a bag-of-tokens
importance model (DSIR-style log-likelihood ratio).

    w[t] = log p_target(t) - log p_corpus(t)

`p_target` is the unigram distribution of the TRAIN splits of public
question-answering / commonsense datasets (never the test or validation items
a benchmark suite scores); `p_corpus` is the unigram distribution of the
FineWeb-Edu training shards. A document's value is the mean of `w` over its
tokens: how much more its vocabulary looks like "things people ask questions
about" than like the average web page. The refiner keeps the documents above
a quantile of that score (quantiles of the corpus score distribution are
stored next to `w`).

    python refiners/tools/build_domain_prior.py --data-dir "$SN125_FINEWEB_DIR" --out refiners/select.safetensors
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

V = 49152
HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))


def _parquet(repo: str, pattern: str):
    import pandas as pd
    hits = sorted(glob.glob(f"{HUB}/datasets--{repo.replace('/', '--')}/snapshots/*/{pattern}"))
    if not hits:
        raise FileNotFoundError(f"{repo}:{pattern} not in the local HF cache")
    return pd.read_parquet(hits[0])


def target_texts() -> dict[str, list[str]]:
    """TRAIN-split texts only, one list per source."""
    out: dict[str, list[str]] = {}
    df = _parquet("allenai/ai2_arc", "ARC-Easy/train-*.parquet")
    out["arc_easy"] = [f"{r.question} {r.choices['text'][list(r.choices['label']).index(r.answerKey)]}"
                       for r in df.itertuples() if r.answerKey in list(r.choices["label"])]
    df = _parquet("allenai/sciq", "data/train-*.parquet")
    out["sciq"] = [f"{r.support} {r.question} {r.correct_answer}" for r in df.itertuples()]
    df = _parquet("allenai/openbookqa", "main/train-*.parquet")
    out["openbookqa"] = [f"{r.question_stem} {r.choices['text'][list(r.choices['label']).index(r.answerKey)]}"
                         for r in df.itertuples()]
    df = _parquet("baber/piqa", "piqa_train.parquet")
    out["piqa"] = [f"{r.goal} {r.sol1 if int(r.label) == 0 else r.sol2}" for r in df.itertuples()]
    df = _parquet("Rowan/hellaswag", "data/train-*.parquet")
    out["hellaswag"] = [f"{r.ctx} {r.endings[int(r.label)]}" for r in df.itertuples()]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-shards", type=int, default=6)
    ap.add_argument("--sample-docs", type=int, default=200_000)
    ap.add_argument("--alpha", type=float, default=1.0, help="additive smoothing (counts)")
    ap.add_argument("--clip", type=float, default=4.0)
    args = ap.parse_args(argv)

    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from sn125.fineweb import TOKENIZER_NAME, load_manifest
    from sn125.refiner.entries import EntryStream

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # target: every source contributes the same probability mass
    p_t = np.zeros(V)
    for name, texts in target_texts().items():
        c = np.zeros(V)
        for i in range(0, len(texts), 2000):
            for ids in tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"]:
                np.add.at(c, np.asarray(ids, dtype=np.int64), 1)
        print(f"[prior] {name}: {len(texts)} train items, {int(c.sum())} tokens")
        p_t += (c + args.alpha) / (c.sum() + args.alpha * V)
    p_t /= p_t.sum()

    mf = load_manifest(args.data_dir)
    files = [s["file"] for s in mf["splits"]["train"]]
    c = np.zeros(V)
    for f in files[::max(1, len(files) // args.corpus_shards)][:args.corpus_shards]:
        c += np.bincount(np.asarray(np.load(Path(args.data_dir) / f, mmap_mode="r")).reshape(-1), minlength=V)
    p_c = (c + args.alpha) / (c.sum() + args.alpha * V)
    w = np.clip(np.log(p_t) - np.log(p_c), -args.clip, args.clip).astype(np.float32)
    w[c == 0] = 0.0                      # never seen in the corpus sample: no evidence either way
    w[int(mf["eos_token_id"])] = 0.0

    # score distribution over a document sample (a different seed than any eval run)
    st = EntryStream(args.data_dir, seed=20261010, verify=False)
    wt = torch.from_numpy(w)
    scores = torch.tensor([float(wt[st[i].tokens].mean()) for i in range(args.sample_docs)])
    qs = torch.quantile(scores.double(), torch.linspace(0, 1, 101, dtype=torch.float64)).float()
    print("[prior] doc score quantiles 0/10/25/50/75/90/95/99/100:",
          [round(float(qs[i]), 3) for i in (0, 10, 25, 50, 75, 90, 95, 99, 100)])
    save_file({"token_llr": wt.half(), "score_quantiles": qs}, args.out)
    print(f"[prior] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
