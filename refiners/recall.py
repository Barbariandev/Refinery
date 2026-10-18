"""
Recall refiner -- "read a little, recall a little".

The identity recipe shows the student every token once, left to right, and
nothing in a short run forces it to USE its context: looking a term up in what
it has just read is learned only as fast as natural text happens to repeat
itself. This refiner asks for that skill explicitly. While a document is being
read, every GAP tokens the student is asked to recall a short span (an "echo")
of the part of the document that is already in its context, and then the
document resumes:

    ... natural text [echo] natural text continues [echo][echo] ...

  * The first token of an echo cannot be predicted from the previous token,
    only from "what is in my context": it trains a context-is-likely prior at
    arbitrary positions of natural text. The rest of the echo trains exact
    continuation from context.
  * The token where the document resumes after an echo has no honest
    prediction, so it is read at weight 0 (context only, no gradient).
  * Echoes are spans of the same entry, already present in the same row:
    provenance-clean by construction, and never a copy of something the
    student cannot see.

No weight file and no student-loss signal (zero scoring cost on the shared
clock); planning is span bookkeeping on the CPU. Knobs are the constants below.
"""
import torch

CAPABILITIES = {"wants_token_losses": False, "window_entries": 96}

GAP_MIN = 8          # natural tokens between two echo events
GAP_MAX = 40
N_SUB_MIN = 1        # echoes per event
N_SUB_MAX = 1
SUB_MIN = 1          # tokens per echo
SUB_MAX = 4
MIN_CTX = 32         # document tokens that must be in context before the first echo
JUMP_W = 1.0         # weight of the first token of an echo (the jump)
CONT_W = 1.0         # weight of the remaining echo tokens (the copy)
RESUME_W = 0.0       # weight of the token where the document resumes
ECHO_UNTIL = 1.0     # stop echoing after this fraction of the budget (1.0 = never stop)
HEAD_TOKENS = 0      # > 0: read only the first HEAD_TOKENS of each document (cut at a line break), skip the rest
NEWLINE_ID = 198     # "\n" in the SmolLM2 tokenizer (used only to place the cut)
SEED = 1234


class Refiner:
    def __init__(self, weights, ctx):
        self.seq_len = int(ctx["seq_len"])
        self.gen = torch.Generator()
        self.gen.manual_seed(SEED)
        self._spans = []    # row under construction
        self._w = []        # weight tensors aligned with _spans
        self._len = 0
        self._cursor = {}   # wid -> tokens consumed
        self._rows = []
        self._pool = []     # pre-drawn random numbers in [0, 1)
        self._echo_on = True

    # -- randomness: one batched draw, consumed scalar by scalar -------------
    def _u(self):
        if not self._pool:
            self._pool = torch.rand(8192, generator=self.gen).tolist()
        return self._pool.pop()

    def _randint(self, lo, hi):
        return lo + int(self._u() * (hi - lo + 1)) if hi > lo else lo

    # -- row assembly --------------------------------------------------------
    def _emit(self, span, w):
        """Append a span; returns True when this completed (flushed) the row."""
        self._spans.append(span)
        self._w.append(w)
        self._len += int(w.numel())
        if self._len >= self.seq_len:
            self._rows.append({"spans": self._spans, "weights": torch.cat(self._w)})
            self._spans, self._w, self._len = [], [], 0
            return True
        return False

    def _plain(self, wid, a, b, first_w):
        w = torch.ones(b - a)
        w[0] = first_w
        return self._emit((wid, a, b), w)

    def _echo(self, wid, lo, hi, n_tok):
        s = self._randint(lo, hi - n_tok)
        w = torch.full((n_tok,), CONT_W)
        w[0] = JUMP_W
        return self._emit((wid, s, s + n_tok), w)

    def _teach(self, wid, n, body_end, pos):
        """Consume entry tokens [pos, n); body_end excludes a trailing EOS."""
        ctx_start = pos      # first token of this entry that is in the current row
        first_w = 1.0
        while pos < n:
            if self._len == 0:
                ctx_start = pos
            if pos >= body_end:                      # the trailing EOS
                self._plain(wid, pos, n, first_w)
                return n
            take = min(self._randint(GAP_MIN, GAP_MAX), body_end - pos, self.seq_len - self._len)
            flushed = self._plain(wid, pos, pos + take, first_w)
            pos += take
            first_w = 1.0
            if flushed or not self._echo_on:
                continue
            k = self._randint(N_SUB_MIN, N_SUB_MAX)
            for _ in range(k):
                avail = pos - ctx_start
                n_tok = min(self._randint(SUB_MIN, SUB_MAX), avail)
                # keep at least one free slot so the echo never ends the row
                if avail < MIN_CTX or self.seq_len - self._len <= n_tok + 1:
                    break
                self._echo(wid, ctx_start, pos, n_tok)
                first_w = RESUME_W
        return pos

    def _teach_head(self, wid, toks, n):
        """Read the head of a long document, then close it with a separator."""
        cut = HEAD_TOKENS
        nl = torch.nonzero(toks[HEAD_TOKENS // 2:HEAD_TOKENS] == NEWLINE_ID)
        if int(nl.numel()) > 0:
            cut = HEAD_TOKENS // 2 + int(nl[-1]) + 1
        self._teach(wid, cut, cut, 0)
        # the cut is ours, not the author's: the separator is context only
        self._emit((-1,), torch.zeros(1))
        return n

    # -- protocol ------------------------------------------------------------
    def plan(self, window, signals, ctx):
        eos = int(ctx["eos_id"])
        self._echo_on = float(ctx["elapsed_frac"]) < ECHO_UNTIL
        for entry in window:
            wid = int(entry["wid"])
            toks = entry["tokens"]
            n = int(toks.numel())
            pos = self._cursor.get(wid, 0)
            if pos >= n:
                continue
            body_end = n - 1 if int(toks[n - 1]) == eos else n
            if HEAD_TOKENS > 0:
                if int(entry["chunk"]) > 0:          # later chunk of a long document: skip
                    self._cursor[wid] = n
                    continue
                if body_end > HEAD_TOKENS:
                    self._cursor[wid] = self._teach_head(wid, toks, n)
                    continue
            self._cursor[wid] = self._teach(wid, n, body_end, pos)
        lessons, self._rows = self._rows, []
        open_wids = {sp[0] for sp in self._spans}
        retire = []
        for entry in window:
            wid = int(entry["wid"])
            if wid not in open_wids and self._cursor.get(wid, 0) >= int(entry["tokens"].numel()):
                retire.append(wid)
                self._cursor.pop(wid, None)
        return {"lessons": lessons, "retire": retire}
