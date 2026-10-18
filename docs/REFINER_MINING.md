# Mining the Refiner task (Challenge 2)

> This repository is the production code release. Paths under `scripts/` and `data/`,
> and result files under `docs/notes/`, refer to the operator's working tree (lab
> tooling, the shard directory, raw measurements) and are not included here.

**Status: opens with the round of Monday 2026-10-19 00:00 UTC; 50% of
miner emission.** From that round validators accept refiner submissions in
their ordinary rounds by default, train and score them on the
benchmark suite against the pinned identity baseline, and pay the task's
pool — half of miner emission — to confirmed frontier events by the same
progress-scaled rule as Challenge 1 ([`REFINER_TASK.md`](REFINER_TASK.md)
§9). The bar is **+0.02 suite points over the current frontier** (the
identity baseline's 0.5554 until someone beats it), confirmed by a second
run at the validator's cost with the worse of the two counting; until a
refiner clears it the pool burns. Submitting costs the normal round fee.

The contract (protocol, limits, scoring, isolation) is
[`REFINER_TASK.md`](REFINER_TASK.md); the reasoning behind the task is
[`REFINER_RESEARCH.md`](REFINER_RESEARCH.md). This document is the working
guide: what to ship, how to submit, what the numbers mean, and what the first
experiments found.

## 1. What a refiner is

Challenge 1 asks *how to update* a model (the optimizer). Challenge 2 asks
*what the model should read, in what order, and which tokens deserve a
gradient* — with the optimizer, model, data pool, budget and clock all fixed.
A refiner is a small program (plus an optional weight file) that sits between
the raw FineWeb-Edu document stream and the student's training batches. Every
plan call it sees a window of fresh documents (token ids), optionally the
student's per-token losses on them, and the run's progress; it answers with
the spans to train on next and a per-token weight.

The student is `lean-llama-360m` trained for 72 000 s / 188 000 steps at
batch 32 × 2048 with a pinned optimizer (the Challenge-1 frontier at the time
of pinning, `sn125/refiner/student/kairos_b_96fe905a.py`). The *identity*
refiner — read every document once, in order, every token weighted 1 — is the
baseline. Your score is

    suite(student trained through your refiner) − suite(student trained through identity)

on the benchmark suite `easy_v0`, same seed, same box class, same clock.

## 2. What you ship

`refiner.py` — a single file with a module-level `class Refiner`:

```python
CAPABILITIES = {"wants_token_losses": False}      # a literal dict; optional

class Refiner:
    def __init__(self, weights, ctx):             # weights: dict of tensors or None
        ...
    def plan(self, window, signals, ctx):         # -> spans + weights, see the template
        ...
```

plus an optional `refiner.safetensors`: at most 100 M parameters in total,
at most 450 MiB on disk, at most 4096 tensors, dtypes F32/F16/BF16 or integer.
The annotated starting point is `sn125/refiner_template.py`; `refiners/`
holds complete, tested submissions:

| file | idea | weights | signals |
|---|---|---|---|
| `refiners/select.py` | read only the documents whose vocabulary looks like "things people ask questions about" (bag-of-tokens importance model, DSIR-style) | 49 k (`select.safetensors`, built by `refiners/tools/build_domain_prior.py`) | none |
| `refiners/recall.py` | interleave short echoes of the text already in context, so the student is asked to recall what it just read | none | none |

Your source passes the same single-source gate as an optimizer (≤ 1 MiB,
restricted namespace, no imports beyond the allow-list, `CAPABILITIES` must be
a literal). Everything a refiner emits is `(wid, start, end)` spans plus a
weight tensor; the harness assembles the batches, the student never sees your
code.

## 3. How to submit

The miner serves one string — a canonical JSON envelope — through the same
commit/reveal flow as an optimizer; its SHA-256 is your commit:

```json
{"refinery_submission":1,"source":"<refiner.py>","task":"RF1_smol_360M",
 "weights":{"bytes":98436,"sha256":"<64 hex>","url":"https://…/refiner.safetensors"}}
```

```bash
# weight-free refiner
python -m sn125 mine --refiner refiner.py

# with weights: the file is hashed into the commit, the URL is revealed with the source
python -m sn125 mine --refiner refiner.py \
    --weights refiner.safetensors --weights-url https://your.host/refiner.safetensors
```

Rules for the weight file, all checked on the scoring box *before* your code
runs and *before* its network is cut:

- the URL must be **https** (redirects included) and serve exactly the
  committed **bytes** and **sha256** — a changed, missing or oversized file is
  a miner fault and burns the fee, like an invalid optimizer;
- keep it online through the round: the box downloads it at evaluation time
  (the validator then archives a verified copy under its hash, so later
  re-runs do not depend on your hosting);
- only the canonical serialization is accepted, so one submission has one
  hash — build the envelope with the CLI, do not hand-write it.

Fees and credits are the optimizer task's (`MINING.md` §7, "Track credits,
rounds and confirmation"): a harness fault
(suite, baseline, entry stream, bench scoring) refunds the credit; a fault in
your refiner (lesson contract violated, process crash, diverged loss, weight
file) burns it.

**When.** The task is live from the round that opens **Monday 2026-10-19 at
00:00 UTC** (commit window 00:00–03:00 UTC; rounds open on that 24-hour grid
whenever the previous round has finished). An envelope committed in an
earlier round is rejected at the gate — `refiner task opens at 2026-10-19
00:00:00 UTC` — and, like any gate rejection, costs its credit; wait for the
opening round.

## 4. What is published, and what the numbers mean

A scored run lands in the round file's signed `refiner_task` block: your
score, the suite and per-task metrics of both arms, the paired 95 % interval
(item-sampling noise), the student's held-out loss under both arms, the
quartile checkpoints of your run (public URIs), the archived weight file, and
a `signature` of everything that defines the regime (task, student, budget,
data manifest, seed, suite, decontamination list, pinned optimizer). Two
scores are comparable only within one signature. Anyone can check the block
with `python -m sn125.verify` and read it on the dashboard under
"Challenge 2".

**The frontier and the bar.** A score is a candidate when it beats the
refiner frontier by more than the bar: `max(0.02, half the mean of the last
three confirmed improvements, halving per stagnant week)`. The best candidate
of a round is re-run next round; it becomes a frontier event only if both
runs clear the bar, and the event's score is the worse run. Events earn
credit that halves every 14 days; the pool pays `min(1, Σ credit / R)` of
itself pro rata (R ramps 0.02 → 0.07 over 56 days, so one floor-clearing
event a week keeps the pool fully paid) and burns the rest. The dashboard's
"Challenge 2" panel shows the frontier, the bar, what is awaiting
confirmation and the payable share; `python -m sn125.verify` replays it.

**The baseline and its noise.** Six full-budget identity runs through the
production path (seeds 42–47, 2026-10-11 … 10-15) score

| suite | hellaswag | piqa | arc_easy | sciq | openbookqa | lambada | held-out |
|---|---|---|---|---|---|---|---|
| **0.5555 ± 0.0030** | 0.438 ± 0.004 | 0.669 ± 0.004 | 0.558 ± 0.009 | 0.768 ± 0.013 | 0.345 ± 0.005 | 0.391 ± 0.011 | 2.599 ± 0.005 |

(mean ± sd over seeds; every run step-limited at 188 000 steps). One run
lands within ±0.003 of the mean; the item-sampling SE of a paired delta is
about 0.004. A delta under ~0.01 is indistinguishable from luck on one seed;
the 0.02 floor is ~6.7 sd of a run and ~4.7 sd of a paired difference, and
the confirmation run makes a lucky draw worth nothing.
The suite's model-free floors are 0.308 (chance) and 0.403 (a copy prior:
SciQ ships its support passage). The pinned baselines themselves are in
`sn125/refiner/baselines/`.

**The one full-budget refiner result so far**: `refiners/select.py` keeping
the top 20 % of documents scores 0.602 against 0.555 (**+0.047**, one seed),
every multiple-choice task up — and the student's held-out loss *worse* by
0.27 nats (2.86 vs 2.59). That trade is the open design question of the
task: whether the suite is the target or a held-out non-regression gate
bounds how much ordinary-text quality a refiner may give up. The gate exists
(`SN125_REFINER_HELDOUT_MAX_REGRESSION`); whether a validator runs with it is
part of its published configuration. Build refiners whose gain survives both
readings if you can.

## 5. The loop that works

1. **Write the plan as span bookkeeping first.** Keep a cursor per `wid` and
   one open row that persists across `plan()` calls; retire an entry only
   when it is fully consumed *and* the open row no longer references it (the
   template and both reference refiners do this).
2. **Drive it on the CPU before it ever sees a GPU.** The gate, the restricted
   namespace and the pipe are the same on CPU:

   ```python
   from sn125.gate import validate_single_source
   from sn125.refiner.entries import EntryStream
   from sn125.refiner.feeder import LessonFeeder
   from sn125.refiner.refproc import RefProcHandle

   src = open("refiner.py").read()
   assert validate_single_source(src, path="refiner.py").ok
   stream = EntryStream(DATA_DIR, seed=42, verify=False)
   handle = RefProcHandle(src, None, device="cpu", sandboxed=True,
                          init_ctx={"seq_len": 2048, "batch_size": 32, "eos_id": 0,
                                    "budget_s": 3600.0, "device": "cpu", "vocab_size": 49152})
   feeder = LessonFeeder(stream, handle, batch_size=32, seq_len=2048, budget_s=3600.0, isolated=True)
   ids, w = feeder.next_batch(step=0, elapsed_s=0.0, tokens_trained=0)   # (32, 2048) each
   print(feeder.stats.as_dict())                                          # plan_seconds, learn_fraction, ...
   ```

   (Run it under `if __name__ == "__main__":` — the refiner process is
   spawned.) Decode a few rows with the tokenizer and read them; most plan
   bugs are visible in the first row.
3. **Measure one arm at a time** with `scripts/refiner_lab.py` (keeps and
   scores the 25/50/75/100 % checkpoints of a single arm, so one run is a
   learning curve) and compare arms and seeds with
   `scripts/refiner_lab_compare.py`. Set `--total-steps` to what the identity
   arm completes in the budget on your box. `python -m sn125 refiner-eval` is
   the official paired run; use it for the final number, and
   `python -m sn125 bench-floor` for the suite's model-free floors.
4. **Read the per-task columns, not just the suite score.** On `easy_v0`
   almost all early movement is SciQ, then ARC-Easy and PIQA; HellaSwag and
   OpenBookQA sit at chance for the first hour. Watch the held-out loss too
   (§4).
5. **Budget for the clock.** `plan()` and `__init__` run on the run's clock.
   Span bookkeeping for a full batch costs ~0.1–0.3 s per step-equivalent on
   one core, a few percent of a step; a refiner that reads five documents to
   keep one is CPU-bound and ends the 20 h with fewer steps than identity.
   `wants_token_losses` on every entry cost the template a quarter of its
   steps.

## 6. Things the sandbox will tell you the hard way

- No `torch.tensor(list)` / `torch.as_tensor(list)`: build weights with
  `torch.ones/zeros/full/cat` and scalar index assignment (`w[0] = 0.0`).
- No `random`, no `getattr`/`setattr`/`type`/`super`/`print`. For randomness
  use a `torch.Generator`; draw a batch
  (`torch.rand(8192, generator=g).tolist()`) and consume it scalar by scalar —
  one `torch.randint` call per decision is slow.
- `CAPABILITIES` must be a literal dict; anything else is rejected at the
  gate.
- Entries older than 64 plan calls are force-retired; a span that references
  one fails the run. Consume the window in order.
- The refiner process may use at most 35 % of the GPU's memory and has a
  120 s plan timeout; a crash, a timeout or a diverged student loss is a
  miner fault.
- The benchmark's documents are withheld from the stream (a 12-gram
  decontamination list, pinned per round). "Find the benchmark text" is not
  a strategy; the suite will also rotate.

## 7. What the first experiments found

One-hour runs on an A100 (seed 42, replicated at seed 43 for the main
result), then the full-budget runs above:

- **The identity student learns the suite's cheap part by itself, fast.**
  Suite 0.331 → 0.356 → 0.375 → 0.381 at 375 / 750 / 1 125 / 1 500 steps,
  almost all of it SciQ: the answer is in the support passage, and "tokens in
  my context are likely" is one of the first things a language model learns.
- **Teaching copying explicitly does not beat that.** Two echo refiners end
  within noise of the baseline (+0.002 and +0.001 at 1 500 steps) while
  giving up 10–25 % of the natural tokens — which also says a short run is
  not limited by how much text it reads.
- **What the student reads moves the score; how it is arranged did not
  (yet).** `refiners/select.py` keeping the top 5 % of documents scores 0.433
  against identity's 0.388 after 1 500 steps (+0.045; +0.030 for the top
  20 %; +0.031 on a second seed at half the budget), and the top-20 % version
  holds and grows at the full budget (+0.047, §4). It gains on every task
  that moves and pays for it in held-out loss.
- **Noise.** ±0.005 between checkpoints of one run is noise at one hour; the
  full-budget across-seed sd is 0.003. Do not chase a third decimal on a
  single seed.
