"""
SN125 Refiner Template — Starter Refiner (Challenge 2)

The optimizer task asks "how should the student update its weights given a
gradient?". The Refiner task asks the question before it: "given a document,
how should the student LEARN from it?". You ship a small neural module
(this file + a safetensors weight file) that watches the stream of FineWeb-Edu
entries and, for each one, decides how it is turned into training rows:

  * which spans of the entry go into a row, in what order, next to what
    (you may reorder, repeat, split, skip, or juxtapose spans of the entries
    currently in your window);
  * the per-token loss weight in [0, 8] — 0 means "the student reads this token
    as context but does NOT backprop through its prediction" (the key lever:
    choose what to learn from, keep everything as context);
  * when to request the student's current per-token losses on an entry
    (`wants_token_losses` / `rescore`), so your decisions can depend on what
    the student already knows.

The student (lean 360M Llama), its optimizer (pinned Challenge-1 frontier),
batch size, sequence length and the 20h budget are fixed and identical for the
baseline and for you. Your refiner runs in its own process on the SAME clock:
a slow plan() buys fewer student steps.

SCORING: the final student checkpoint is evaluated on a pre-tokenized suite of
easy public benchmarks (HellaSwag, PIQA, ARC-Easy, SciQ, OpenBookQA, LAMBADA);
score = suite score(your refiner) - suite score(identity baseline). Held-out
loss is reported but not scored.

RULES (enforced; see docs/REFINER_TASK.md):
  - Module-level `class Refiner` with `__init__(self, weights, ctx)` and
    `plan(self, window, signals, ctx)`; optional `CAPABILITIES` dict.
  - `weights`: dict[str, Tensor] loaded from your safetensors file (CPU; move
    them to ctx["device"] yourself). Caps: 100M parameters, 450 MB, dtypes
    f32/f16/bf16/int/bool. No pickle. The file may be omitted (empty dict).
  - Same restricted namespace as optimizer.py: imports limited to torch, math,
    dataclasses, collections, functools, itertools, triton, typing, enum, abc;
    no exec/eval/open/getattr/setattr/type/super/print. Build tensors with
    torch.zeros/ones/full/arange/stack/cat/randn (torch.tensor(list) is not
    available inside the sandbox). Use torch.nn.Module.__init__(self) instead
    of super().__init__() if you subclass nn.Module.
  - Every token you emit must come from an entry in your current window (span
    provenance is enforced; the only token you may add is the EOS separator).
  - A lesson is <= seq_len tokens; weights are a 1-D float tensor, one per
    emitted token (separators included), finite, in [0, 8].
  - Retire entries when you are done with them; entries older than 64 plan
    calls are retired for you. 8 consecutive plans that emit nothing and retire
    nothing fail the run. A plan() call that exceeds the per-call deadline
    (default 120 s) kills your process and fails the run.

PLAN PROTOCOL
  window:  list of {"wid": int, "doc_id": int, "chunk": int, "n_chunks": int,
                    "tokens": LongTensor[n]}        (the entries you can draw from)
  signals: {"token_losses": {wid: FloatTensor[n]}}  (present iff wants_token_losses;
                    entry t = student CE of predicting token t; t=0 is 0)
  ctx:     {"step", "elapsed_s", "elapsed_frac", "tokens_trained", "seq_len",
            "batch_size", "eos_id", "budget_s", "plan_calls"}
  return:  {"lessons": [{"spans": [(wid, start, end), ..., (-1,) for a separator],
                         "weights": FloatTensor[total emitted tokens]}, ...],
            "retire": [wid, ...],            # optional
            "rescore": [wid, ...]}           # optional; fresh token losses next call

This template reproduces the baseline recipe (pack entries in order, weight 1)
with two small learned twists so you can see every lever used: tokens the
student already predicts well (loss below a learned threshold) are kept as
context only, and the per-token weight is a learned function of the loss.
Without a weight file (threshold 0, slope 0) it IS the identity baseline, so
your first experiment is one safetensors file away:

    from safetensors.torch import save_file
    save_file({"threshold": torch.full((1,), 0.5), "slope": torch.full((1,), 0.3),
               "proj": torch.randn(8, 8)}, "refiner.safetensors")

Replace all of it.
"""
import torch

CAPABILITIES = {
    "wants_token_losses": True,  # ask the harness for the student's per-token losses on new entries
    # Entries kept in view (1..256). Also the scoring granularity: new entries
    # are scored together before each plan call, so a bigger window means
    # fuller GPU batches and fewer plan calls per unit of training.
    "window_entries": 128,
}


class Refiner:
    def __init__(self, weights, ctx):
        self.device = torch.device(ctx["device"])
        self.seq_len = int(ctx["seq_len"])
        self.eos_id = int(ctx["eos_id"])
        # Two learned scalars from the weight file (or defaults when absent).
        self.threshold = float(weights["threshold"].item()) if "threshold" in weights else 0.0
        self.slope = float(weights["slope"].item()) if "slope" in weights else 0.0
        # Example of a real tensor parameter living on the device.
        self.proj = (weights["proj"] if "proj" in weights else torch.zeros(8, 8)).to(self.device)
        self._open = []     # spans of the row under construction (persist across calls)
        self._open_w = []
        self._open_len = 0
        self._cursor = {}   # wid -> tokens already consumed

    def _weight_for(self, losses):
        # weight = 1 + slope * (loss - mean) squashed into [0, 8]; 0 where the
        # student is already confident (loss below threshold) => context-only.
        if losses is None:
            return None
        centered = losses - losses.mean()
        w = (1.0 + self.slope * centered).clamp(0.0, 8.0)
        return torch.where(losses < self.threshold, torch.zeros_like(w), w)

    def plan(self, window, signals, ctx):
        lessons = []
        retire = []
        # Losses for entries that entered the window this call are already here
        # (the harness scores new entries before calling plan).
        token_losses = signals.get("token_losses", None)
        for entry in window:
            wid = int(entry["wid"])
            toks = entry["tokens"]
            n = int(toks.numel())
            w_all = None
            if token_losses is not None:
                if wid not in token_losses:
                    continue  # defensive: every window entry is scored before plan() is called
                w_all = self._weight_for(token_losses[wid].float())
            pos = self._cursor.get(wid, 0)
            while pos < n:
                room = self.seq_len - self._open_len
                take = min(room, n - pos)
                self._open.append((wid, pos, pos + take))
                self._open_w.append(w_all[pos:pos + take] if w_all is not None
                                    else torch.ones(take))
                self._open_len += take
                pos += take
                if self._open_len >= self.seq_len:
                    lessons.append({"spans": list(self._open), "weights": torch.cat(self._open_w)})
                    self._open, self._open_w, self._open_len = [], [], 0
            self._cursor[wid] = pos
        # Retire fully packed entries that the (still open) row no longer
        # references. Decided after the whole window is packed: an entry's tail
        # usually sits in the open row right after it is packed, but later
        # entries flush that row within the same call.
        open_wids = {sp[0] for sp in self._open}
        for entry in window:
            wid = int(entry["wid"])
            if wid not in open_wids and self._cursor.get(wid, 0) >= int(entry["tokens"].numel()):
                retire.append(wid)
                self._cursor.pop(wid, None)
        return {"lessons": lessons, "retire": retire}
