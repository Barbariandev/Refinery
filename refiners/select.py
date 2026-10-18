"""
Select refiner -- "read what is worth reading".

The pool (16 B tokens) is larger than any run can read, and retiring an entry
costs nothing, so the cheapest decision a refiner can make is which documents
to read at all. This refiner scores every entry with a bag-of-tokens importance
model shipped in its weight file,

    score(doc) = mean_t  llr[token_t],     llr = log p_target - log p_corpus,

where p_target is the unigram distribution of the TRAIN splits of public
question-answering and commonsense datasets and p_corpus that of FineWeb-Edu
(DSIR-style importance resampling; built by refiners/tools/build_domain_prior.py,
which never reads a test or validation item). Entries in the top KEEP_TOP of
the corpus score distribution are packed exactly as the identity baseline
packs them (stream order, full rows, weight 1); the rest are retired unread.
High-scoring documents are plain, concrete, explanatory text -- how things
work, what things are for -- and low-scoring ones are citations, listings and
boilerplate.

Weights: token_llr [vocab] f16, score_quantiles [101] f32 (percentiles 0..100
of the document score over a corpus sample). No student-loss signal; planning
is one gather + mean per entry on the CPU.
"""
import torch

CAPABILITIES = {"wants_token_losses": False, "window_entries": 256}

# Share of documents read (by corpus score quantile). Measured on one-hour runs
# (1 500 steps, 100 M tokens): 0.20 -> +0.030 suite score over identity, 0.05 ->
# +0.045. A short run reads a small part of even the top 5 % (0.8 B tokens);
# a full 12 B-token run would re-read it ~15 times, so widen this with the
# budget (0.20 covers 3.2 B tokens, i.e. about four passes).
KEEP_TOP = 0.05


class Refiner:
    def __init__(self, weights, ctx):
        self.seq_len = int(ctx["seq_len"])
        self.llr = weights["token_llr"].float()
        q = weights["score_quantiles"].float()
        self.threshold = float(q[int(round((1.0 - KEEP_TOP) * 100.0))])
        self._spans = []    # row under construction
        self._len = 0
        self._cursor = {}   # wid -> tokens consumed (kept entries only)
        self._skip = set()  # wids judged not worth reading

    def plan(self, window, signals, ctx):
        lessons = []
        for entry in window:
            wid = int(entry["wid"])
            if wid in self._skip:
                continue
            toks = entry["tokens"]
            n = int(toks.numel())
            pos = self._cursor.get(wid, -1)
            if pos < 0:                                   # first sight: read it or not?
                if float(self.llr[toks].mean()) < self.threshold:
                    self._skip.add(wid)
                    continue
                pos = 0
            while pos < n:
                take = min(n - pos, self.seq_len - self._len)
                self._spans.append((wid, pos, pos + take))
                pos += take
                self._len += take
                if self._len == self.seq_len:
                    lessons.append({"spans": self._spans, "weights": torch.ones(self.seq_len)})
                    self._spans, self._len = [], 0
            self._cursor[wid] = pos
        open_wids = {sp[0] for sp in self._spans}
        retire = []
        for entry in window:
            wid = int(entry["wid"])
            if wid in self._skip:
                retire.append(wid)
                self._skip.discard(wid)
            elif wid not in open_wids and self._cursor.get(wid, -1) >= int(entry["tokens"].numel()):
                retire.append(wid)
                self._cursor.pop(wid, None)
        return {"lessons": lessons, "retire": retire}
