"""The Lesson contract: how a refiner tells the harness what the student trains on.

A refiner never emits token ids. It emits SPANS of the entries currently in its
window plus a weight per token, and the harness materializes the sequence from
the entries itself — provenance is enforced by construction ("it shapes, it
doesn't author"). The only token a refiner can add is the EOS separator
(span ``SEP``), so sequences can be delimited without inventing text.

Plan protocol (one call = one ``plan``):

    plan(window, signals, ctx) -> {"lessons": [...], "retire": [wid, ...], "rescore": [wid, ...]}

    window   list of {"wid": int, "doc_id": int, "chunk": int, "n_chunks": int,
                      "tokens": LongTensor[n]}          (CPU tensors)
    signals  {"token_losses": {wid: FloatTensor[n]}}    (if the refiner asked for them)
    ctx      {"step", "elapsed_frac", "tokens_trained", "seq_len", "batch_size",
              "eos_id", "budget_s", "plan_calls"}
    lesson   {"spans": [(wid, start, end), ...] | (SEP,) entries,
              "weights": FloatTensor[total_len]}        0 <= w <= W_MAX, finite;
                                                        w < W_MIN is read as 0

Each lesson becomes ONE training row of ``seq_len`` tokens: the spans
concatenated in order, padded at the END with EOS at weight 0 (padding after
the content is invisible to every real token under causal attention). A lesson
longer than ``seq_len`` is a protocol violation. Retired entries leave the
window and are replaced by the next entries of the stream; an entry that is
never retired is force-retired after ``MAX_WINDOW_AGE`` plan calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

SEP = -1              # span sentinel: one EOS token (the only token a refiner may add)
W_MAX = 8.0           # per-token weight ceiling (weights are normalized by their sum anyway)
W_MIN = 1.0 / 64.0    # smallest non-zero weight; anything below is read as 0 (context only)
MAX_WINDOW_AGE = 64   # plan calls an entry may stay in the window before forced retirement
MAX_LESSONS_PER_PLAN = 4096


class LessonError(ValueError):
    """Protocol violation by the refiner (miner fault — fails the run)."""


@dataclass
class Lesson:
    spans: list[tuple]            # (wid, start, end) or (SEP,)
    weights: torch.Tensor         # float32 [total_len]

    @property
    def n_tokens(self) -> int:
        return int(self.weights.numel())


@dataclass
class PlanResult:
    lessons: list[Lesson] = field(default_factory=list)
    retire: list[int] = field(default_factory=list)
    rescore: list[int] = field(default_factory=list)


def _as_int(x, what: str) -> int:
    if torch.is_tensor(x):
        if x.numel() != 1:
            raise LessonError(f"{what} must be a scalar")
        x = x.item()
    if isinstance(x, bool) or not isinstance(x, (int, float)) or int(x) != x:
        raise LessonError(f"{what} must be an integer, got {x!r}")
    return int(x)


def parse_plan(raw, window_ids: set[int]) -> PlanResult:
    """Validate the untrusted plan payload coming back from the refiner."""
    if not isinstance(raw, dict):
        raise LessonError(f"plan must be a dict, got {type(raw).__name__}")
    lessons_raw = raw.get("lessons", [])
    if not isinstance(lessons_raw, (list, tuple)):
        raise LessonError("plan['lessons'] must be a list")
    if len(lessons_raw) > MAX_LESSONS_PER_PLAN:
        raise LessonError(f"too many lessons in one plan: {len(lessons_raw)} > {MAX_LESSONS_PER_PLAN}")
    lessons: list[Lesson] = []
    for li, L in enumerate(lessons_raw):
        if not isinstance(L, dict):
            raise LessonError(f"lesson {li} must be a dict")
        spans_raw = L.get("spans")
        if not isinstance(spans_raw, (list, tuple)) or not spans_raw:
            raise LessonError(f"lesson {li}: spans must be a non-empty list")
        spans: list[tuple] = []
        total = 0
        for sp in spans_raw:
            if not isinstance(sp, (list, tuple)):
                raise LessonError(f"lesson {li}: span must be a tuple")
            if len(sp) == 1:
                if _as_int(sp[0], "span sentinel") != SEP:
                    raise LessonError(f"lesson {li}: unknown 1-tuple span {sp!r}")
                spans.append((SEP,))
                total += 1
                continue
            if len(sp) != 3:
                raise LessonError(f"lesson {li}: span must be (wid, start, end), got {sp!r}")
            wid = _as_int(sp[0], "span wid")
            start = _as_int(sp[1], "span start")
            end = _as_int(sp[2], "span end")
            if wid not in window_ids:
                raise LessonError(f"lesson {li}: span references wid {wid} not in window")
            if start < 0 or end <= start:
                raise LessonError(f"lesson {li}: bad span range [{start}, {end})")
            spans.append((wid, start, end))
            total += end - start
        w = L.get("weights")
        if not torch.is_tensor(w):
            raise LessonError(f"lesson {li}: weights must be a tensor")
        w = w.detach()
        if w.dim() != 1 or w.numel() != total:
            raise LessonError(f"lesson {li}: weights shape {tuple(w.shape)} != ({total},)")
        w = w.to(dtype=torch.float32, device="cpu")
        if not bool(torch.isfinite(w).all()):
            raise LessonError(f"lesson {li}: non-finite weight")
        if bool((w < 0).any()) or bool((w > W_MAX).any()):
            raise LessonError(f"lesson {li}: weights must lie in [0, {W_MAX}]")
        # The loss is normalized by the batch weight sum, so only RATIOS matter;
        # without a floor the [0, W_MAX] range would bound nothing (8 vs 1e-30).
        # With it the largest ratio between two learned tokens is W_MAX / W_MIN.
        w = torch.where(w < W_MIN, torch.zeros_like(w), w)
        lessons.append(Lesson(spans=spans, weights=w.clone()))
    retire = [_as_int(x, "retire wid") for x in (raw.get("retire") or [])]
    rescore = [_as_int(x, "rescore wid") for x in (raw.get("rescore") or [])]
    for wid in retire + rescore:
        if wid not in window_ids:
            raise LessonError(f"retire/rescore references wid {wid} not in window")
    return PlanResult(lessons=lessons, retire=retire, rescore=rescore)


def materialize_lesson(lesson: Lesson, window: dict[int, torch.Tensor], seq_len: int,
                       eos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one ``(ids[seq_len], weights[seq_len])`` row from a validated lesson.

    Provenance: every non-SEP token is copied out of the window entry the span
    names; ranges are bounds-checked against that entry. Padding (EOS, weight 0)
    fills the tail.
    """
    n = lesson.n_tokens
    if n > seq_len:
        raise LessonError(f"lesson of {n} tokens exceeds seq_len {seq_len}")
    if n == 0:
        raise LessonError("empty lesson")
    ids = torch.full((seq_len,), int(eos_id), dtype=torch.long)
    w = torch.zeros(seq_len, dtype=torch.float32)
    o = 0
    for sp in lesson.spans:
        if sp[0] == SEP and len(sp) == 1:
            ids[o] = int(eos_id)
            o += 1
            continue
        wid, start, end = sp
        toks = window.get(wid)
        if toks is None:
            raise LessonError(f"span references wid {wid} not in window")
        if end > toks.numel():
            raise LessonError(f"span [{start}, {end}) exceeds entry {wid} length {toks.numel()}")
        ids[o:o + (end - start)] = toks[start:end]
        o += end - start
    w[:n] = lesson.weights
    return ids, w


class IdentityPlanner:
    """The baseline refiner: pack entries in stream order into full rows, every
    token at weight 1. Reproduces the plain next-token recipe (documents joined
    with their EOS, chopped to ``seq_len``) through the same plan protocol a
    miner's refiner uses, so baseline and submission differ ONLY in the plan.
    """

    capabilities = {"wants_token_losses": False, "window_entries": 16}

    def __init__(self):
        self._offset: dict[int, int] = {}   # wid -> tokens already packed
        self._done: set[int] = set()        # fully packed, awaiting retirement
        self._open_spans: list[tuple] = []  # the row under construction (persists across calls)
        self._open_len = 0

    def plan(self, window: list[dict], signals: dict, ctx: dict) -> dict:
        seq_len = int(ctx["seq_len"])
        lessons: list[dict] = []
        for ent in window:
            wid = int(ent["wid"])
            n = int(ent["tokens"].numel())
            pos = self._offset.get(wid, 0)
            while pos < n:
                take = min(n - pos, seq_len - self._open_len)
                self._open_spans.append((wid, pos, pos + take))
                pos += take
                self._open_len += take
                if self._open_len == seq_len:
                    lessons.append({"spans": self._open_spans, "weights": torch.ones(seq_len)})
                    self._open_spans, self._open_len = [], 0
            self._offset[wid] = pos
            if pos >= n:
                self._done.add(wid)
        # An entry can leave the window only once no open (partial) row still
        # references it; the harness keeps the window stable across calls, so
        # the open row's spans stay valid until it closes on a later call.
        open_refs = {sp[0] for sp in self._open_spans}
        retire = sorted(w for w in self._done if w not in open_refs)
        for w in retire:
            self._done.discard(w)
            self._offset.pop(w, None)
        return {"lessons": lessons, "retire": retire, "rescore": []}
