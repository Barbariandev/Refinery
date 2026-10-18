"""Entry stream for the Refiner task.

The production shards (fineweb.py) are the document stream of a split joined
with EOS and chopped into fixed ``seq_len`` rows IN ORDER, so document
boundaries are recoverable exactly: a document is the token range ending at
(and including) each EOS of the flattened split. ``build_doc_index`` scans a
split once and stores the exclusive end offset of every document as an int64
array next to the shards (``DOC_INDEX_<split>.npy``); its sha256 is part of
the task record.

``EntryStream`` turns that index into a seeded, deterministic sequence of
ENTRIES: whole documents, except that documents longer than
``max_entry_tokens`` are cut into consecutive chunks (each chunk is its own
entry, tagged with ``chunk``/``n_chunks``). Entries are read lazily from the
mmap'd shards, so host RAM is O(entry), not O(corpus).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..fineweb import _open_split, default_data_dir, load_manifest

DOC_INDEX_TEMPLATE = "DOC_INDEX_{split}.npy"
EXCLUSIONS_TEMPLATE = "DECONTAM_{split}_{suite_id}.npy"
DEFAULT_MAX_ENTRY_TOKENS = 8192
_SCAN_ROWS = 4096  # rows per block while scanning for EOS (bounds RAM during the index build)


def doc_index_path(data_dir: str | Path | None, split: str) -> Path:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    return data_dir / DOC_INDEX_TEMPLATE.format(split=split)


def _eos_id(manifest: dict) -> int:
    eos = manifest.get("eos_token_id")
    if eos is None:
        raise ValueError("manifest has no eos_token_id; cannot recover document boundaries")
    return int(eos)


def build_doc_index(data_dir: str | Path | None, split: str = "train",
                    out_path: str | Path | None = None, verify: bool = False) -> Path:
    """Scan ``split`` for EOS positions and write the document end-offset index.

    Idempotent: returns the existing path when the index is already present.
    Must run as a user that can write to the shard dir (the validator parent,
    not the sandboxed worker, which sees the shards read-only).
    """
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    out = Path(out_path) if out_path else doc_index_path(data_dir, split)
    if out.exists():
        return out
    mf, arrays, _offsets, _total = _open_split(data_dir, split, None, verify)
    seq = int(mf["seq_len"])
    eos = _eos_id(mf)
    ends: list[np.ndarray] = []
    base = 0
    for arr in arrays:
        n_rows = arr.shape[0]
        for r0 in range(0, n_rows, _SCAN_ROWS):
            blk = np.asarray(arr[r0:r0 + _SCAN_ROWS])
            pos = np.flatnonzero(blk.reshape(-1) == eos).astype(np.int64)
            if pos.size:
                ends.append(pos + (base + r0 * seq) + 1)  # exclusive end, EOS included
        base += n_rows * seq
    index = np.concatenate(ends) if ends else np.zeros(0, dtype=np.int64)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, index)
    tmp.replace(out)
    try:
        out.chmod(0o644)  # the nobody sandbox worker must be able to read it
    except OSError:
        pass
    return out


def load_doc_index(data_dir: str | Path | None, split: str = "train") -> np.ndarray:
    p = doc_index_path(data_dir, split)
    if not p.exists():
        raise FileNotFoundError(
            f"document index missing: {p} (build it with build_doc_index / `sn125 refiner-index`)")
    idx = np.load(p, mmap_mode="r")
    if idx.dtype != np.int64 or idx.ndim != 1:
        raise ValueError(f"malformed document index {p}: dtype={idx.dtype} ndim={idx.ndim}")
    return idx


def doc_index_sha256(data_dir: str | Path | None, split: str = "train") -> str:
    return _file_sha256(doc_index_path(data_dir, split))


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── decontamination: documents withheld from the entry stream ───────────────
# A refiner cannot author tokens, but it can find and up-weight text that is
# already in the corpus, and the pool is cheap to skim end to end. Documents
# that share long n-grams with benchmark items (scripts/refiner_decontam.py)
# are therefore removed from the stream for baseline and submission alike. The
# list is a sorted int64 array of document ids next to the shards; its sha256
# is part of the task record (EntryStream.describe()).

def exclusions_path(data_dir: str | Path | None, split: str, suite_id: str) -> Path:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    return data_dir / EXCLUSIONS_TEMPLATE.format(split=split, suite_id=suite_id)


def write_exclusions(data_dir: str | Path | None, split: str, suite_id: str, doc_ids) -> dict:
    p = exclusions_path(data_dir, split, suite_id)
    arr = np.unique(np.asarray(list(doc_ids), dtype=np.int64))
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(p)
    try:
        p.chmod(0o644)  # readable by the nobody sandbox worker
    except OSError:
        pass
    return {"path": str(p), "n_docs": int(arr.size), "sha256": _file_sha256(p)}


def load_exclusions(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.int64 or arr.ndim != 1 or (arr.size > 1 and bool((np.diff(arr) <= 0).any())):
        raise ValueError(f"malformed exclusion list {path}: need a sorted, unique, 1-D int64 array")
    return arr


@dataclass
class Entry:
    """One unit of work for the refiner: a document (or a chunk of a long one)."""
    doc_id: int
    chunk: int
    n_chunks: int
    tokens: torch.Tensor  # 1-D int64, CPU; ends with EOS iff chunk == n_chunks - 1

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.numel())


class EntryStream:
    """Seeded, lazy, deterministic sequence of entries over one split.

    ``self[i]`` is the i-th entry of the permutation for ``seed``; the same
    ``(data_dir manifest, split, seed, max_entry_tokens)`` always yields the
    same sequence (common random numbers between baseline and submission).
    Indexing past the end wraps deterministically (second epoch), mirroring
    ``shard_batch_sequence``.
    """

    def __init__(self, data_dir: str | Path | None, *, split: str = "train", seed: int = 0,
                 max_entry_tokens: int = DEFAULT_MAX_ENTRY_TOKENS, verify: bool = True,
                 exclusions: str | Path | None = None):
        data_dir = Path(data_dir) if data_dir else default_data_dir()
        self.data_dir = data_dir
        self.split = split
        self.seed = int(seed)
        self.max_entry_tokens = int(max_entry_tokens)
        if self.max_entry_tokens <= 0:
            raise ValueError("max_entry_tokens must be positive")
        mf, arrays, row_offsets, total_rows = _open_split(data_dir, split, None, verify)
        self.manifest = mf
        self.seq_len = int(mf["seq_len"])
        self.eos_id = _eos_id(mf)
        self._arrays = arrays
        self._row_offsets = row_offsets  # cumulative rows per shard file
        self._total_tokens = int(total_rows) * self.seq_len
        ends = np.asarray(load_doc_index(data_dir, split), dtype=np.int64)
        if ends.size and int(ends[-1]) > self._total_tokens:
            raise ValueError("document index does not match the shards (end offset past the stream)")
        starts = np.empty_like(ends)
        if ends.size:
            starts[0] = 0
            starts[1:] = ends[:-1]
        lengths = ends - starts
        n_chunks = np.maximum(1, np.ceil(lengths / self.max_entry_tokens)).astype(np.int64)
        # entry -> (doc, chunk) via repeat; chunk k covers [start + k*M, min(start+(k+1)*M, end))
        self._entry_doc = np.repeat(np.arange(ends.size, dtype=np.int64), n_chunks)
        first = np.cumsum(n_chunks) - n_chunks  # first entry index of each doc
        self._entry_chunk = np.arange(self._entry_doc.size, dtype=np.int64) - first[self._entry_doc]
        self._doc_start = starts
        self._doc_end = ends
        self._doc_n_chunks = n_chunks
        self.n_docs = int(ends.size)
        # Withheld documents (decontamination). The permutation is drawn over ALL
        # entries and the withheld ones are then dropped from it, so the order of
        # everything that remains does not depend on the list: rotating the suite
        # (a new list) leaves the stream otherwise identical.
        self.exclusions_path = ""
        self.exclusions_sha256 = ""
        self.n_excluded_docs = 0
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(int(self._entry_doc.size), generator=g).numpy()
        if exclusions:
            exclusions = Path(exclusions)
            if not exclusions.is_absolute():
                exclusions = data_dir / exclusions  # a file name next to the shards
            self.exclusions_path = str(exclusions)
            ex = load_exclusions(exclusions)
            if ex.size and (int(ex[0]) < 0 or int(ex[-1]) >= self.n_docs):
                raise ValueError("exclusion list does not match the document index (doc id out of range)")
            self.exclusions_sha256 = _file_sha256(exclusions)
            self.n_excluded_docs = int(ex.size)
            perm = perm[~np.isin(self._entry_doc[perm], ex)]
        self._perm = perm
        self.n_entries = int(perm.size)
        if self.n_entries == 0:
            raise ValueError(f"split {split!r} contains no complete documents")

    def __len__(self) -> int:
        return self.n_entries

    @property
    def manifest_hash(self) -> str:
        return str(self.manifest.get("manifest_hash", ""))

    def _read_range(self, start: int, end: int) -> np.ndarray:
        """Tokens [start, end) of the flattened split stream (crosses rows/shards)."""
        seq = self.seq_len
        out = np.empty(end - start, dtype=np.int64)
        pos = start
        o = 0
        while pos < end:
            row = pos // seq
            col = pos - row * seq
            take = min(seq - col, end - pos)
            shard = int(np.searchsorted(self._row_offsets, row, side="right")) - 1
            local = row - int(self._row_offsets[shard])
            out[o:o + take] = self._arrays[shard][local, col:col + take]
            pos += take
            o += take
        return out

    def entry_at(self, entry_index: int) -> Entry:
        """Entry by UNPERMUTED index (document order)."""
        d = int(self._entry_doc[entry_index])
        k = int(self._entry_chunk[entry_index])
        s = int(self._doc_start[d]) + k * self.max_entry_tokens
        e = min(int(self._doc_end[d]), s + self.max_entry_tokens)
        toks = torch.from_numpy(self._read_range(s, e))
        return Entry(doc_id=d, chunk=k, n_chunks=int(self._doc_n_chunks[d]), tokens=toks)

    def __getitem__(self, i: int) -> Entry:
        if i < 0:
            raise IndexError(i)
        return self.entry_at(int(self._perm[i % self.n_entries]))

    def describe(self) -> dict:
        return {
            "split": self.split, "seed": self.seed, "n_docs": self.n_docs,
            "n_entries": self.n_entries, "max_entry_tokens": self.max_entry_tokens,
            "seq_len": self.seq_len, "eos_id": self.eos_id,
            "manifest_hash": self.manifest.get("manifest_hash", ""),
            "doc_index_sha256": doc_index_sha256(self.data_dir, self.split),
            "excluded_docs": self.n_excluded_docs, "exclusions_sha256": self.exclusions_sha256,
        }


def ensure_doc_index(data_dir: str | Path | None = None, split: str = "train") -> Path:
    """Parent-side preflight: make sure the index exists (building it if needed)."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    load_manifest(data_dir)  # validates the shard set is present and self-consistent
    return build_doc_index(data_dir, split)


def estimate_entries_for_tokens(stream: EntryStream, n_tokens: int) -> int:
    """How many entries (on average) cover ``n_tokens`` — used to size the window."""
    avg = max(1.0, stream._total_tokens / max(1, stream.n_entries))
    return max(1, int(math.ceil(n_tokens / avg)))
