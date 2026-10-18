#!/usr/bin/env python3
"""Offline trainer for the per-task `select` weight file: one bag-of-tokens
importance table per target source instead of the pooled one
`build_domain_prior.py` writes.

    w[t, k] = log p_target_k(t) - log p_corpus(t)

A document is kept when its mean score reaches the per-table quantile for ANY
table, so every source contributes its own best-matching documents. Output
tensors: `w` (vocab x sources, f16) and `t` (one threshold per source).

    python refiners/tools/build_task_priors.py --data-dir "$SN125_FINEWEB_DIR" --out refiner.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_domain_prior import V, target_texts  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-shards", type=int, default=6)
    ap.add_argument("--sample-docs", type=int, default=100_000)
    ap.add_argument("--quantile", type=float, default=0.985, help="per-table keep threshold")
    ap.add_argument("--alpha", type=float, default=1.0, help="additive smoothing (counts)")
    ap.add_argument("--clip", type=float, default=4.0)
    args = ap.parse_args(argv)

    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from sn125.fineweb import TOKENIZER_NAME, load_manifest
    from sn125.refiner.entries import EntryStream

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    names, p_t = [], []
    for name, texts in target_texts().items():
        c = np.zeros(V)
        for i in range(0, len(texts), 2000):
            for ids in tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"]:
                np.add.at(c, np.asarray(ids, dtype=np.int64), 1)
        print(f"[priors] {name}: {len(texts)} train items, {int(c.sum())} tokens")
        names.append(name)
        p_t.append((c + args.alpha) / (c.sum() + args.alpha * V))

    mf = load_manifest(args.data_dir)
    files = [s["file"] for s in mf["splits"]["train"]]
    c = np.zeros(V)
    for f in files[::max(1, len(files) // args.corpus_shards)][:args.corpus_shards]:
        c += np.bincount(np.asarray(np.load(Path(args.data_dir) / f, mmap_mode="r")).reshape(-1), minlength=V)
    p_c = (c + args.alpha) / (c.sum() + args.alpha * V)
    w = np.stack([np.clip(np.log(p) - np.log(p_c), -args.clip, args.clip) for p in p_t], 1).astype(np.float32)
    w[c == 0] = 0.0                      # never seen in the corpus sample: no evidence either way
    w[int(mf["eos_token_id"])] = 0.0

    # score distribution over a document sample (a different seed than any eval run)
    st = EntryStream(args.data_dir, seed=20261010, verify=False)
    wt = torch.from_numpy(w)
    scores = torch.stack([wt[st[i].tokens].mean(0) for i in range(args.sample_docs)])
    t = torch.quantile(scores.double(), args.quantile, dim=0).float()
    keep = float((scores >= t).any(1).float().mean())
    print(f"[priors] sources {names}; per-table quantile {args.quantile} keeps {keep:.3%} of documents")
    save_file({"w": wt.half().contiguous(), "t": t.contiguous()}, args.out)
    print(f"[priors] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
