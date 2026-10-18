"""Harness-side feeder: turns a planner's lessons into training batches.

The feeder owns the entry stream, the refiner's window, the signal
computation and the lesson row queue. The training loop asks it for one
``(ids[B,S], weights[B,S])`` batch per step; everything the planner does in
between — plan calls, token-loss scoring, materialization — runs inside the
same wall clock as training, so a heavier refiner buys fewer student steps.

Planner = either the trusted in-process ``IdentityPlanner`` (the baseline) or
a ``RefProcHandle`` wrapping the miner's isolated refiner process. Both speak
the plan protocol documented in lessons.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .entries import EntryStream
from .lessons import (IdentityPlanner, LessonError, MAX_WINDOW_AGE, PlanResult,
                      materialize_lesson, parse_plan)

MAX_IDLE_PLANS = 8  # consecutive plans with no lessons and no retirements => stall
MIN_SCORE_LEN = 128  # smallest padded row length when scoring entries for token losses


@dataclass
class FeederStats:
    entries_consumed: int = 0
    lessons: int = 0
    plan_calls: int = 0
    plan_seconds: float = 0.0
    score_seconds: float = 0.0
    tokens_in_rows: int = 0      # non-padding tokens placed in rows (read by the student)
    tokens_weighted: int = 0     # tokens with weight > 0 (learned from)
    weight_sum: float = 0.0
    forced_retirements: int = 0
    scored_entries: int = 0
    startup_seconds: float = 0.0  # refiner process start + Refiner.__init__, charged to the budget

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["learn_fraction"] = (self.tokens_weighted / self.tokens_in_rows) if self.tokens_in_rows else 0.0
        return d


class _WindowEntry:
    __slots__ = ("wid", "entry", "born", "scored")

    def __init__(self, wid, entry, born):
        self.wid = wid
        self.entry = entry
        self.born = born
        self.scored = False


class LessonFeeder:
    def __init__(self, stream: EntryStream, planner, *, batch_size: int, seq_len: int,
                 budget_s: float = 0.0, window_entries: int | None = None,
                 wants_token_losses: bool | None = None, isolated: bool = False):
        self.stream = stream
        self.planner = planner
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.eos_id = int(stream.eos_id)
        self.budget_s = float(budget_s)
        caps = getattr(planner, "capabilities", {}) or {}
        self.window_entries = int(window_entries if window_entries is not None
                                  else caps.get("window_entries", 16))
        self.wants_token_losses = bool(wants_token_losses if wants_token_losses is not None
                                       else caps.get("wants_token_losses", False))
        self.isolated = isolated  # planner is a RefProcHandle (already parses/validates)
        self._window: list[_WindowEntry] = []
        self._next_wid = 0
        self._next_entry = 0
        self._rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._losses: dict[int, torch.Tensor] = {}
        self._pending_signal: list[int] = []
        self._idle = 0
        self.stats = FeederStats()
        self.startup_seconds = 0.0
        self._closed = False

    # ── window management ────────────────────────────────────────────────
    def _refill(self):
        while len(self._window) < self.window_entries:
            ent = self.stream[self._next_entry]
            self._next_entry += 1
            we = _WindowEntry(self._next_wid, ent, self.stats.plan_calls)
            self._next_wid += 1
            self._window.append(we)
            if self.wants_token_losses:
                self._pending_signal.append(we.wid)

    def _retire(self, wids: list[int]):
        if not wids:
            return
        drop = set(wids)
        keep = []
        for we in self._window:
            if we.wid in drop:
                self._losses.pop(we.wid, None)
                self.stats.entries_consumed += 1
            else:
                keep.append(we)
        self._window = keep

    def _force_retire_stale(self):
        stale = [we.wid for we in self._window if self.stats.plan_calls - we.born >= MAX_WINDOW_AGE]
        if stale:
            self.stats.forced_retirements += len(stale)
            self._retire(stale)

    # ── signals ──────────────────────────────────────────────────────────
    def _compute_signals(self, score_fn):
        if not self.wants_token_losses or not self._pending_signal:
            return
        if score_fn is None:
            raise LessonError("refiner asked for token losses but the loop provided no scorer")
        by_wid = {we.wid: we for we in self._window}
        todo = [w for w in self._pending_signal if w in by_wid]
        self._pending_signal = []
        if not todo:
            return
        t0 = time.time()
        S = self.seq_len
        pieces: list[torch.Tensor] = []
        owners: list[tuple[int, int, int]] = []  # (wid, offset, length)
        for wid in todo:
            toks = by_wid[wid].entry.tokens
            n = int(toks.numel())
            for o in range(0, n, S):
                piece = toks[o:o + S]
                pieces.append(piece)
                owners.append((wid, o, int(piece.numel())))
        out: dict[int, torch.Tensor] = {wid: torch.zeros(int(by_wid[wid].entry.tokens.numel()),
                                                         dtype=torch.float32) for wid in todo}
        # Bucket pieces by padded length (powers of two up to seq_len) instead of
        # padding everything to seq_len: FineWeb-Edu entries average ~1k tokens,
        # so seq_len padding doubled the scoring FLOPs (a third of the template's
        # budget went to scoring). Causal attention makes the loss at a real
        # position independent of trailing padding, so this changes only the
        # cost. Each scoring batch carries ~batch_size*seq_len tokens.
        buckets: dict[int, list[int]] = {}
        for idx, (_wid, _o, length) in enumerate(owners):
            L = MIN_SCORE_LEN
            while L < length:
                L *= 2
            buckets.setdefault(min(L, S), []).append(idx)
        for L, idxs in sorted(buckets.items()):
            rows = max(1, (self.batch_size * S) // L)
            for i in range(0, len(idxs), rows):
                sel = idxs[i:i + rows]
                batch = torch.full((len(sel), L), self.eos_id, dtype=torch.long)
                for j, k in enumerate(sel):
                    batch[j, :owners[k][2]] = pieces[k]
                losses = score_fn(batch).detach().float().cpu()
                for j, k in enumerate(sel):
                    wid, o, length = owners[k]
                    out[wid][o:o + length] = losses[j, :length]
        for wid, l in out.items():
            self._losses[wid] = l
            by_wid[wid].scored = True
        self.stats.scored_entries += len(todo)
        self.stats.score_seconds += time.time() - t0

    # ── planning ─────────────────────────────────────────────────────────
    def _window_payload(self) -> list[dict]:
        return [{"wid": we.wid, "doc_id": we.entry.doc_id, "chunk": we.entry.chunk,
                 "n_chunks": we.entry.n_chunks, "tokens": we.entry.tokens} for we in self._window]

    def _plan_once(self, ctx: dict, score_fn) -> PlanResult:
        self._refill()
        self._compute_signals(score_fn)
        window = self._window_payload()
        signals = {"token_losses": {wid: self._losses[wid] for wid in self._losses}} \
            if self.wants_token_losses else {}
        t0 = time.time()
        if self.isolated:
            result = self.planner.plan(window, signals, ctx)
        else:
            raw = self.planner.plan(window, signals, ctx)
            result = parse_plan(raw, {int(e["wid"]) for e in window})
        self.stats.plan_calls += 1
        self.stats.plan_seconds += time.time() - t0
        by_wid = {we.wid: we.entry.tokens for we in self._window}
        for L in result.lessons:
            ids, w = materialize_lesson(L, by_wid, self.seq_len, self.eos_id)
            self._rows.append((ids, w))
            self.stats.lessons += 1
            self.stats.tokens_in_rows += L.n_tokens
            nz = int((L.weights > 0).sum())
            self.stats.tokens_weighted += nz
            self.stats.weight_sum += float(L.weights.sum())
        if result.rescore and self.wants_token_losses:
            # one fresh scoring per entry per call, however often it was named
            pending = set(self._pending_signal)
            self._pending_signal.extend(w for w in dict.fromkeys(result.rescore) if w not in pending)
        self._retire(result.retire)
        self._force_retire_stale()
        if not result.lessons and not result.retire:
            self._idle += 1
            if self._idle >= MAX_IDLE_PLANS:
                raise LessonError(f"refiner stalled: {MAX_IDLE_PLANS} consecutive plans produced "
                                  f"no lessons and retired no entries")
        else:
            self._idle = 0
        return result

    def next_batch(self, *, step: int, elapsed_s: float, tokens_trained: int,
                   score_fn=None) -> tuple[torch.Tensor, torch.Tensor]:
        """One ``(ids[B,S], weights[B,S])`` CPU batch; raises LessonError on miner fault."""
        ctx = {"step": int(step), "elapsed_s": float(elapsed_s),
               "elapsed_frac": (float(elapsed_s) / self.budget_s) if self.budget_s > 0 else 0.0,
               "tokens_trained": int(tokens_trained), "seq_len": self.seq_len,
               "batch_size": self.batch_size, "eos_id": self.eos_id,
               "budget_s": self.budget_s, "plan_calls": self.stats.plan_calls}
        while len(self._rows) < self.batch_size:
            self._plan_once(ctx, score_fn)
        rows, self._rows = self._rows[:self.batch_size], self._rows[self.batch_size:]
        ids = torch.stack([r[0] for r in rows])
        w = torch.stack([r[1] for r in rows])
        return ids, w

    def close(self):
        if self._closed:
            return
        self._closed = True
        close = getattr(self.planner, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def make_feeder(stream: EntryStream, cfg: dict, *, batch_size: int, seq_len: int,
                budget_s: float, device: str) -> LessonFeeder:
    """Build the feeder from a refiner config dict (see training.evaluate_submission).

    cfg["mode"]: "identity" (baseline) or "refiner" (miner source + weights).
    """
    mode = (cfg or {}).get("mode", "identity")
    init_ctx = {"seq_len": int(seq_len), "batch_size": int(batch_size), "eos_id": int(stream.eos_id),
                "budget_s": float(budget_s), "device": str(device),
                "vocab_size": int(stream.manifest.get("vocab_size", 49152))}
    if mode == "identity":
        return LessonFeeder(stream, IdentityPlanner(), batch_size=batch_size, seq_len=seq_len,
                            budget_s=budget_s, window_entries=int(cfg.get("window_entries", 64)),
                            wants_token_losses=False, isolated=False)
    if mode != "refiner":
        raise ValueError(f"unknown refiner mode {mode!r}")
    from .refproc import RefProcHandle, DEFAULT_PLAN_TIMEOUT_S
    t_spawn = time.time()
    handle = RefProcHandle(cfg["source"], cfg.get("weights_path") or None, device=device,
                           sandboxed=bool(cfg.get("sandboxed", True)),
                           plan_timeout=float(cfg.get("plan_timeout", DEFAULT_PLAN_TIMEOUT_S)),
                           init_ctx=init_ctx)
    feeder = LessonFeeder(stream, handle, batch_size=batch_size, seq_len=seq_len, budget_s=budget_s,
                          isolated=True)
    # Loading the weights and running Refiner.__init__ is refiner work: it is
    # charged to the run's clock (the training loop starts its budget this far in).
    feeder.startup_seconds = time.time() - t_spawn
    feeder.stats.startup_seconds = feeder.startup_seconds
    return feeder
