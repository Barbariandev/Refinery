"""FineWeb-Edu production-task data pipeline (DESIGN §1.1 + §5.4).

Pre-tokenized, hash-pinned shards for the production training task:

  - Dataset: HuggingFaceFW/fineweb-edu, config sample-100BT (streamed once,
    shuffled with a pinned seed).
  - Tokenizer: SmolLM2 family tokenizer. Tokenization is frozen at BUILD time —
    a run can never observe a tokenizer mismatch.
  - Packing: documents joined with EOS, chopped into fixed-length sequences.
    Production uses seq_len=2048 for the fixed-budget B200 task.
  - Storage: uint16 .npy per shard ([n_sequences, seq_len]; SmolLM2 vocab
    49152 < 65536), each sha256-pinned in MANIFEST.json.
  - Split: the HELD-OUT shards are written FIRST from the stream and the
    boundary is cut at a document boundary, so train/heldout are strictly
    document-disjoint by construction. The split is explicit in the manifest;
    production training boxes receive only the train shards (§5.4 — the
    held-out eval happens in the clean process).

The manifest's `manifest_hash` (sha256 of the canonical manifest body) is the
single pin a TaskSpec carries (`TaskSpec.data_manifest`) and is part of
`task_signature()` — switching dataset or rebuilding shards changes the
signature, so every stored reference/rolling-best curve from another task
(e.g. all wikitext research history) goes STALE automatically (S1).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("sn125.fineweb")

DATASET_REF = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-100BT"
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
SEQ_LEN = 256
SHUFFLE_SEED = 1337
SHARD_TOKENS = 2 ** 24

SHARD_DATASET_NAME = "fineweb-edu-shards"

MANIFEST_NAME = "MANIFEST.json"


def default_data_dir() -> Path:
    """Shard directory: $SN125_FINEWEB_DIR or
    <repo>/data/fineweb_edu_smollm2_seq{FIXED_BUDGET_SEQ_LEN}.

    The dir name tracks the PRODUCTION seq_len (2048), not the legacy seq-256 dev
    set — the validator loads the production shards by default, and the manifest
    seq_len is re-checked against the task in `_open_split`, so a stale seq-256 set
    left under data/ can never silently satisfy a seq-2048 task."""
    env = os.environ.get("SN125_FINEWEB_DIR", "")
    if env:
        return Path(env)
    name = f"fineweb_edu_smollm2_seq{FIXED_BUDGET_SEQ_LEN}"
    return Path(__file__).resolve().parent.parent / "data" / name


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(body: dict) -> str:
    """sha256 of the canonical manifest body (everything except 'manifest_hash')."""
    body = {k: v for k, v in body.items() if k != "manifest_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()




def build_shards(
    out_dir: str | Path,
    token_lists,
    *,
    train_tokens: int,
    heldout_tokens: int,
    seq_len: int = SEQ_LEN,
    shard_tokens: int = SHARD_TOKENS,
    meta: dict | None = None,
) -> dict:
    """Pack pre-tokenized documents into hash-pinned shards + MANIFEST.json.

    Held-out shards are filled FIRST; when the held-out target is reached
    mid-document, the remainder of that document is DROPPED so no document
    ever spans the split boundary (strict disjointness).

    Returns the manifest dict (also written to out_dir/MANIFEST.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o755)
    shard_seqs = max(1, shard_tokens // seq_len)
    targets = [("heldout", heldout_tokens // seq_len), ("train", train_tokens // seq_len)]

    shards: dict[str, list[dict]] = {"heldout": [], "train": []}

    def _flush(split: str, rows: list[np.ndarray]):
        if not rows:
            return
        arr = np.stack(rows).astype(np.uint16)
        fname = f"{split}_{len(shards[split]):04d}.npy"
        np.save(out_dir / fname, arr)
        os.chmod(out_dir / fname, 0o644)
        shards[split].append({
            "file": fname,
            "sha256": _sha256_file(out_dir / fname),
            "n_sequences": int(arr.shape[0]),
        })
        log.info(f"wrote {fname}: {arr.shape[0]} seqs")

    it = iter(token_lists)
    exhausted = False
    for split, target_seqs in targets:
        buf: list[int] = []
        rows: list[np.ndarray] = []
        n_done = 0
        while n_done < target_seqs and not exhausted:
            try:
                doc = next(it)
            except StopIteration:
                exhausted = True
                break
            buf.extend(doc)
            while len(buf) >= seq_len and n_done < target_seqs:
                rows.append(np.asarray(buf[:seq_len], dtype=np.uint16))
                buf = buf[seq_len:]
                n_done += 1
                if len(rows) >= shard_seqs:
                    _flush(split, rows)
                    rows = []
        _flush(split, rows)

    body = {
        "version": 1,
        "dataset": DATASET_REF,
        "dataset_config": DATASET_CONFIG,
        "tokenizer": TOKENIZER_NAME,
        "seq_len": seq_len,
        "shuffle_seed": SHUFFLE_SEED,
        "splits": shards,
        "totals": {s: sum(e["n_sequences"] for e in shards[s]) for s in shards},
        "note": ("heldout written first, split cut at a document boundary -> "
                 "strictly document-disjoint; heldout shards must never be "
                 "staged on production training boxes (DESIGN 5.4)"),
    }
    if meta:
        body.update(meta)
    body["manifest_hash"] = manifest_hash(body)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(body, indent=2, sort_keys=True))
    os.chmod(out_dir / MANIFEST_NAME, 0o644)
    log.info(f"manifest {body['manifest_hash'][:16]}: "
             f"train={body['totals']['train']} heldout={body['totals']['heldout']} seqs")
    return body


FIXED_BUDGET_TRAIN_TOKENS = 16_000_000_000
FIXED_BUDGET_HELDOUT_TOKENS = 350_000_000
FIXED_BUDGET_SEQ_LEN = 2048
PROD_BUDGET_SECONDS = 72_000.0
PROD_CONFIRM_STEPS = 188_000
PROD_BATCH_SIZE = 32


def build_fineweb_shards(
    out_dir: str | Path | None = None,
    *,
    train_tokens: int = FIXED_BUDGET_TRAIN_TOKENS,
    heldout_tokens: int = FIXED_BUDGET_HELDOUT_TOKENS,
    seq_len: int = FIXED_BUDGET_SEQ_LEN,
    seed: int = SHUFFLE_SEED,
) -> dict:
    """Stream FineWeb-Edu, tokenize with the SmolLM2 tokenizer, build shards.

    Network-touching production builder (run once, locally). Documents are
    joined with EOS; tokenization is batched for throughput. Defaults build the
    fixed-budget Chinchilla shard set (≥16B train tok, ≥300M held-out, seq 2048);
    ~32GB on disk — verify free space first (host ~86% full).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = Path(out_dir) if out_dir else default_data_dir()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    eos_id = tok.eos_token_id
    assert eos_id is not None and tok.vocab_size <= 65536, \
        f"uint16 packing requires vocab<=65536, got {tok.vocab_size}"

    ds = load_dataset(DATASET_REF, DATASET_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    def _doc_token_lists():
        batch, n_tok = [], 0
        for row in ds:
            batch.append(row["text"])
            if len(batch) >= 256:
                for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                    n_tok += len(ids) + 1
                    yield ids + [eos_id]
                batch = []
                log.info(f"tokenized ~{n_tok/1e6:.0f}M tokens")
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            yield ids + [eos_id]

    return build_shards(
        out_dir, _doc_token_lists(),
        train_tokens=train_tokens, heldout_tokens=heldout_tokens, seq_len=seq_len,
        meta={"eos_token_id": int(eos_id), "shuffle_seed": seed},
    )



_VERIFIED_DIRS: set[tuple[str, str]] = set()


def load_manifest(data_dir: str | Path | None = None) -> dict:
    """Load MANIFEST.json and validate its self-hash."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    mf = json.loads((data_dir / MANIFEST_NAME).read_text())
    expect = manifest_hash(mf)
    if mf.get("manifest_hash") != expect:
        raise ValueError(f"Manifest self-hash mismatch in {data_dir}: "
                         f"{mf.get('manifest_hash')} != {expect}")
    return mf


def verify_split(data_dir: str | Path, manifest: dict, split: str) -> None:
    """Verify every shard of a split against its pinned sha256. Raises on mismatch."""
    data_dir = Path(data_dir)
    for entry in manifest["splits"][split]:
        path = data_dir / entry["file"]
        actual = _sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Shard hash mismatch: {path} {actual} != {entry['sha256']}")


class ShardBatchSequence:
    """Lazy, deterministic, mmap-backed batch sequence.

    Indexable like a list of CPU long tensors [batch_size, seq_len], but a batch
    is materialized only on __getitem__ — so a multi-hundred-thousand-step
    fixed-budget run (§4.3: N≈760k steps) never holds more than one batch in
    host RAM. This is the train-side analogue of the §5 eval streaming fix that
    closed the 1.7B host-RAM OOM. Index b yields a tensor BYTE-IDENTICAL to
    load_shard_batches(...)[b] for the same (seed, batch_size, num_batches) —
    the CRN determinism / exact submission↔baseline pairing is unchanged.
    """

    def __init__(self, arrays, offsets, idx, batch_size, seq_len, num_batches):
        self._arrays = arrays
        self._offsets = offsets
        self._idx = idx
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._n = num_batches

    def __len__(self):
        return self._n

    def __getitem__(self, b):
        if b < 0:
            b += self._n
        if not 0 <= b < self._n:
            raise IndexError(b)
        bs = self._batch_size
        rows = self._idx[b * bs:(b + 1) * bs]
        out = np.empty((bs, self._seq_len), dtype=np.int64)
        for j, r in enumerate(rows):
            shard = int(np.searchsorted(self._offsets, r, side="right")) - 1
            out[j] = self._arrays[shard][r - self._offsets[shard]]
        return torch.from_numpy(out)


def _open_split(data_dir: str | Path | None, split: str,
                seq_len: int | None, verify: bool):
    """Resolve + (optionally) hash-verify a split; return its mmap'd arrays and
    a contiguous global row-offset table. mmap_mode='r' means only touched pages
    page in — building the sequence costs no host RAM proportional to data size."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    mf = load_manifest(data_dir)
    if seq_len is not None and seq_len != mf["seq_len"]:
        raise ValueError(f"seq_len {seq_len} != manifest seq_len {mf['seq_len']} — "
                         f"this manifest is pinned to seq {mf['seq_len']}")
    if split not in mf["splits"] or not mf["splits"][split]:
        raise ValueError(f"No shards for split {split!r} in {data_dir}")
    key = (str(data_dir), split)
    if verify and key not in _VERIFIED_DIRS:
        verify_split(data_dir, mf, split)
        _VERIFIED_DIRS.add(key)
    arrays = [np.load(data_dir / e["file"], mmap_mode="r") for e in mf["splits"][split]]
    counts = [a.shape[0] for a in arrays]
    total = sum(counts)
    offsets = np.cumsum([0] + counts)
    return mf, arrays, offsets, total


def shard_batch_sequence(
    data_dir: str | Path | None,
    *,
    split: str,
    batch_size: int,
    num_batches: int,
    seed: int,
    seq_len: int | None = None,
    verify: bool = True,
) -> ShardBatchSequence:
    """Deterministic CRN batch sequence — LAZY (O(batch) host RAM regardless of
    num_batches). Preferred for the fixed-budget 20h real-data run. Sequences
    are drawn from a seeded permutation of the whole split; if the request
    exceeds the split the permutation cycles (documented, deterministic)."""
    mf, arrays, offsets, total = _open_split(data_dir, split, seq_len, verify)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=g)
    need = num_batches * batch_size
    idx = perm[torch.arange(need) % total].numpy()
    return ShardBatchSequence(arrays, offsets, idx, batch_size, mf["seq_len"], num_batches)


def load_shard_batches(
    data_dir: str | Path | None,
    *,
    split: str,
    batch_size: int,
    num_batches: int,
    seed: int,
    seq_len: int | None = None,
    verify: bool = True,
) -> list[torch.Tensor]:
    """Eager wrapper over shard_batch_sequence: fully materializes the list of
    CPU long tensors [batch_size, seq_len]. Kept for back-compat / small draws
    (e.g. eval); long training runs should consume the lazy sequence directly."""
    seq = shard_batch_sequence(data_dir, split=split, batch_size=batch_size,
                               num_batches=num_batches, seed=seed,
                               seq_len=seq_len, verify=verify)
    return [seq[b] for b in range(len(seq))]



_FW_SHAPES = {
    "135M": ("FW1_smol_135M", "HuggingFaceTB/SmolLM2-135M", 4),
    "360M": ("FW2_smol_360M", "HuggingFaceTB/SmolLM2-360M", 4),
    "lean-360M": ("FW2_smol_360M", "lean-llama-360m", 4),
    "1.7B": ("FW3_smol_1.7B", "HuggingFaceTB/SmolLM2-1.7B", 2),
}


def make_fineweb_task(model_size: str, total_steps: int, manifest: str,
                      decay_fraction: float = 0.20, *,
                      seq_len: int = SEQ_LEN, batch_size: int | None = None,
                      compute_budget_seconds: float = 0.0,
                      stop_at_total_steps: bool = True):
    """Production FineWeb-Edu TaskSpec. `manifest` is the manifest_hash pin
    (REQUIRED — an unpinned production task is exactly the S1 hole).

    `seq_len`/`batch_size` override the per-scale defaults for the fixed-budget
    regime (production seq=2048, batch=32). Changing seq_len from 256 changes the
    task_signature (intended — old 256-era curves correctly drop out)."""
    from .training import TaskSpec
    if model_size not in _FW_SHAPES:
        raise ValueError(f"No FineWeb shape for {model_size}; known: {list(_FW_SHAPES)}")
    if not manifest:
        raise ValueError("manifest hash required for a production FineWeb task")
    tid, mcfg, bs = _FW_SHAPES[model_size]
    if batch_size is not None:
        bs = batch_size
    parameter_count = "360M" if model_size == "lean-360M" else model_size
    return TaskSpec(
        task_id=tid, model_config=mcfg, parameter_count=parameter_count,
        total_steps=total_steps, batch_size=bs, sequence_length=seq_len,
        eval_every=max(total_steps // 20, 5), eval_sequences=8192,
        warmup_fraction=0.05, decay_fraction=decay_fraction,
        task_weight=1.0, use_pretrained=False,
        compute_budget_seconds=compute_budget_seconds,
        dataset=SHARD_DATASET_NAME, data_manifest=manifest,
        stop_at_total_steps=bool(stop_at_total_steps and compute_budget_seconds > 0),
    )


PROD_MODEL_SIZE = "lean-360M"
PROD_MODEL_SIZE_ENV = "SN125_PROD_MODEL_SIZE"


def production_tasks(total_steps: int, *, data_dir: str | Path | None = None,
                     model_size: str | None = None,
                     decay_fraction: float = 0.20,
                     compute_budget_seconds: float = PROD_BUDGET_SECONDS,
                     stop_at_total_steps: bool = True) -> list:
    """The launch scoring task list, pinned to the live FineWeb-Edu shard manifest.

    Single source of truth for "what the production round trains on": the SmolLM2
    backbone (`PROD_MODEL_SIZE`) at the fixed-budget calibrated regime
    (seq=`FIXED_BUDGET_SEQ_LEN`, batch=`PROD_BATCH_SIZE`), data-pinned to the on-disk manifest_hash
    so a stale/legacy shard set can never silently satisfy the task (S1).

    Raises FileNotFoundError (with a build hint) when the shards are absent, so the
    validate path fails loudly with an actionable message rather than crashing."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    model_size = model_size or os.environ.get(PROD_MODEL_SIZE_ENV, PROD_MODEL_SIZE)
    if not (data_dir / MANIFEST_NAME).exists():
        raise FileNotFoundError(
            f"No production shards at {data_dir} (MANIFEST.json missing). "
            f"Build them first (scripts/build_prod_shards.py / build_shards()).")
    mf = load_manifest(data_dir)
    return [make_fineweb_task(model_size, total_steps=total_steps,
                              manifest=mf["manifest_hash"],
                              decay_fraction=decay_fraction,
                              seq_len=FIXED_BUDGET_SEQ_LEN, batch_size=PROD_BATCH_SIZE,
                              compute_budget_seconds=compute_budget_seconds,
                              stop_at_total_steps=stop_at_total_steps)]
