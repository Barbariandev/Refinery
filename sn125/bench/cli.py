"""CLI entry points: bench-build, bench-score, bench-floor, refiner-index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bench_build(argv) -> int:
    from ..fineweb import TOKENIZER_NAME
    from .suite import DEFAULT_LIMIT, DEFAULT_SUITE_ID, DEFAULT_TASKS, build_suite, default_suite_path, save_suite
    ap = argparse.ArgumentParser(prog="sn125 bench-build", description="Build the pre-tokenized benchmark suite")
    ap.add_argument("--data-dir", default=None, help="shard dir; suite goes to <data-dir>/bench/")
    ap.add_argument("--out", default="", help="explicit output path (overrides --data-dir placement)")
    ap.add_argument("--jsonl-dir", default="", help="dir of <task>.jsonl text files that take precedence over HF")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="items per task (deterministic subsample)")
    ap.add_argument("--suite-id", default=DEFAULT_SUITE_ID)
    ap.add_argument("--tokenizer", default=TOKENIZER_NAME,
                    help="tokenizer repo (default: production). Another tokenizer re-tokenizes the SAME text items "
                         "(same text_sha256) so non-SmolLM2 reference models can be scored; pass --out for it")
    ap.add_argument("--offline", action="store_true", help="only use the local HF cache")
    args = ap.parse_args(argv)
    if args.tokenizer != TOKENIZER_NAME and not args.out:
        ap.error("--tokenizer other than the production one needs an explicit --out (never the default suite path)")
    suite = build_suite(tasks=tuple(t for t in args.tasks.split(",") if t), jsonl_dir=args.jsonl_dir or None,
                        limit=args.limit, tokenizer_name=args.tokenizer, suite_id=args.suite_id,
                        allow_network=not args.offline)
    out = Path(args.out) if args.out else default_suite_path(args.data_dir)
    save_suite(suite, out)
    print(json.dumps({"path": str(out), "suite_id": suite["suite_id"], "suite_sha256": suite["suite_sha256"],
                      "text_sha256": suite["text_sha256"], "tokenizer": suite["tokenizer"],
                      "tasks": {k: v["n"] for k, v in suite["tasks"].items()}, "skipped": suite["skipped"]}, indent=2))
    return 0 if suite["tasks"] else 1


def _bench_score(argv) -> int:
    import torch
    from .score import (BASE_MODELS, HF_REFERENCE_MODELS, score_base_model_on_suite, score_checkpoint_on_suite,
                        score_hf_reference_on_suite)
    from .suite import default_suite_path, load_suite
    ap = argparse.ArgumentParser(prog="sn125 bench-score",
                                 description="Score a checkpoint, or an allowlisted pretrained base model, on the suite")
    ap.add_argument("checkpoint", nargs="?", default="", help="safetensors checkpoint (omit with --base/--hf)")
    ap.add_argument("--base", default="", choices=[""] + sorted(BASE_MODELS), metavar="NAME",
                    help="SmolLM-family reference model, loaded into the lean twin from the local HF cache "
                         "(safetensors only, never downloads): " + ", ".join(sorted(BASE_MODELS)))
    ap.add_argument("--hf", default="", choices=[""] + sorted(HF_REFERENCE_MODELS), metavar="NAME",
                    help="other-tokenizer reference model (native transformers class, no remote code, local "
                         "safetensors only); needs a suite built with `bench-build --tokenizer NAME`: "
                         + ", ".join(sorted(HF_REFERENCE_MODELS)))
    ap.add_argument("--revision", default="", help="with --hf: cached repo revision (e.g. a Pythia step tag)")
    ap.add_argument("--model-spec", default="hf:lean-llama-360m", help="hf:<config name> or toy:<json>")
    ap.add_argument("--suite", default="")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if sum(map(bool, (args.checkpoint, args.base, args.hf))) != 1:
        ap.error("give exactly one of: a checkpoint path, --base NAME, or --hf NAME")
    if args.hf and not args.suite:
        ap.error("--hf needs --suite pointing at a suite built with `bench-build --tokenizer NAME`")
    suite = load_suite(args.suite or default_suite_path(args.data_dir))
    log = lambda s: print(s, file=sys.stderr)  # noqa: E731
    if args.base:
        res = score_base_model_on_suite(args.base, suite, seq_len=args.seq_len, device=args.device,
                                        use_amp=not args.no_amp, rows_per_batch=args.rows, log=log)
    elif args.hf:
        res = score_hf_reference_on_suite(args.hf, suite, seq_len=args.seq_len, revision=args.revision or None,
                                          device=args.device, use_amp=not args.no_amp, rows_per_batch=args.rows,
                                          log=log)
    else:
        res = score_checkpoint_on_suite(args.checkpoint, suite, args.model_spec, seq_len=args.seq_len,
                                        device=args.device, use_amp=not args.no_amp, rows_per_batch=args.rows,
                                        log=log)
    text = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def _bench_floor(argv) -> int:
    from .heuristics import suite_floors
    from .suite import default_suite_path, load_suite
    ap = argparse.ArgumentParser(prog="sn125 bench-floor",
                                 description="Model-free floors of a suite: uniform chance and the bag-of-context "
                                             "copy prior (how much of the score needs no language knowledge)")
    ap.add_argument("--suite", default="")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--mix", type=float, default=0.5, help="probability mass the copy prior puts on context tokens")
    ap.add_argument("--vocab-size", type=int, default=49152)
    args = ap.parse_args(argv)
    suite = load_suite(args.suite or default_suite_path(args.data_dir))
    res = suite_floors(suite, vocab_size=args.vocab_size, mix=args.mix)
    res.update({"suite_id": suite["suite_id"], "suite_sha256": suite["suite_sha256"]})
    print(json.dumps(res, indent=2))
    return 0


def _refiner_index(argv) -> int:
    from ..refiner.entries import EntryStream, build_doc_index, doc_index_sha256
    ap = argparse.ArgumentParser(prog="sn125 refiner-index", description="Build the document index over the shards")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--verify", action="store_true", help="hash-verify the shards first")
    args = ap.parse_args(argv)
    p = build_doc_index(args.data_dir, args.split, verify=args.verify)
    stream = EntryStream(args.data_dir, split=args.split, seed=0, verify=False)
    print(json.dumps({"path": str(p), "sha256": doc_index_sha256(args.data_dir, args.split),
                      **stream.describe()}, indent=2))
    return 0


def main(command: str, argv) -> int:
    argv = list(argv or [])
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if command == "bench-build":
        return _bench_build(argv)
    if command == "bench-score":
        return _bench_score(argv)
    if command == "bench-floor":
        return _bench_floor(argv)
    if command == "refiner-index":
        return _refiner_index(argv)
    raise SystemExit(f"unknown bench command {command}")
