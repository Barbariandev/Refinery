# Refiner research memo — the next direction

> This repository is the production code release. Paths under `scripts/` and `data/`,
> and result files under `docs/notes/`, refer to the operator's working tree (lab
> tooling, the shard directory, raw measurements) and are not included here.

Status: **living draft** (started 2026-10-08). This is the research
rationale behind Challenge 2. The contract the code implements is
[`REFINER_TASK.md`](REFINER_TASK.md); the reference numbers it leans on are in
[`notes/bench-reference-2026-10-05/REFERENCE.md`](notes/bench-reference-2026-10-05/REFERENCE.md).
This document is about *why* — the idea, the intuitions, what we expect to
find, what we do not know yet, and the experiments that decide it.

## 1. Thesis

Challenge 1 asks miners how a model should **update** given a gradient. Every
submission sees the same tokens in the same order with the same loss weight;
the only freedom is the step. That is the most heavily explored corner of
pretraining, and the race there is now over hundredths of a nat (frontier
2.5869 → 2.5552 pending confirmation, bar 0.03).

Challenge 2 asks the question that comes *before* the gradient: given a
document, how should a model **learn from it**? Which spans deserve a
prediction loss and which should only be read as context; what should sit
next to what in the context window; what is worth reading twice; what should
come late, when the model can finally use it; and how should all of this
change as the model's own state changes. Today's pretraining recipe answers
all of these with a shrug: concatenate, chop at 2048, weight everything 1.0.
Nobody believes that is optimal. It is simply the default that nobody has had
an incentive and a credible measurement to replace.

The refiner is a learned module that replaces the shrug. It is the only moving
part: student, optimizer (the pinned Challenge-1 frontier), batch shape, data
stream, clock and benchmark are identical for the baseline and for every
submission. If the student ends up knowing more, it is because it was taught
better, not because it was bigger, trained longer, or saw different data.

## 2. Why this lever, why now

Three observations motivate the direction.

**The field has optimized the step and left the lesson alone.** Optimizers,
schedules, architectures and scaling laws have each absorbed years of
competitive effort. "How a document becomes training rows" has a handful of
papers with positive first results (selective token loss, related-document
packing, learnability-based prioritization, repetition under data
constraints) and no venue where the ideas compete against each other under one
clock. Unexplored problem spaces are where early gains are large and cheap;
that is exactly what a network of incentivized researchers should be pointed
at.

**At our scale, what is read matters more than who reads it.** The reference
ladder on the Challenge-2 suite (`easy_v0`, score = mean length-debiased
probability of the correct answer over five multiple-choice tasks, chance ≈
0.30) says this plainly:

| what changed | models | suite score |
|---|---|---|
| nothing (random init) | lean-llama-360m, seed 0 | 0.308 |
| 2.5× the parameters, same 12.6 B Pile tokens | Pythia-160m → Pythia-410m @ step 6000 | 0.428 → 0.441 (+0.013) |
| 24× the tokens, same model | Pythia-410m step 6000 → final (300 B) | 0.441 → 0.506 (+0.065) |
| **curated data instead of the Pile, a 3× smaller model** | Pythia-410m (300 B Pile) → SmolLM-135M (600 B curated) | 0.506 → **0.537 (+0.031)** |
| better curation, same size (SmolLM v1 → v2 data) | SmolLM-360M → SmolLM2-360M | 0.609 → 0.620 |

A full-budget Challenge-1 run sees ≈ 12.3 B tokens (188 k steps × 32 × 2048),
so by Pythia analogy the anchor for a Challenge-2 student would be the
0.43–0.45 band. The measured full-budget identity baseline (§7.2) is **0.555**
— FineWeb-Edu and the pinned optimizer put a 20 h student where the Pile put
a 300 B-token one — and size barely moves that band while data moves it a lot.
Document-level
curation (FineWeb-Edu is a classifier over whole documents) is what produced
SmolLM's edge. The refiner takes the same idea *inside* the document, down to
spans and tokens, and makes it adaptive to what the student already knows.

**We are data-rich and compute-poor, so selection is free.** The training pool
is 16.0 B unique tokens (7,812,500 sequences × 2048) and a 20 h run consumes
12.3 B. The identity baseline reads each token at most once and never reaches
the last quarter of the pool. A refiner that drops the lowest-value third of
every document loses nothing the baseline would have needed, and the freed
steps can be spent re-reading the spans that matter. Data-constrained scaling
work found that repeating good data for a few epochs costs almost nothing
relative to fresh data; here the choice is not "repeat or fresh" but "repeat
the best or read the worst once", which is a much easier trade to win.

## 3. What a refiner is

A refiner is a program plus up to 100 M parameters of weights that watches a
seeded stream of FineWeb-Edu entries (whole documents, chunked at 8192
tokens) through a window of up to 256 entries and, on each call, returns
**lessons**: training rows assembled from spans of the entries in its window,
each emitted token carrying a loss weight in [0, 8]. Weight 0 is the central
primitive: *read this, do not learn to predict it.* The refiner may reorder,
split, repeat, drop and juxtapose spans across entries, retire entries when it
is done with them, and ask the harness for the student's current per-token
losses on an entry (and ask again later to see what changed). It never sees
the student's parameters, the benchmark, or any token that is not in its
window; it cannot invent text (provenance by construction — the only token it
may add is the EOS separator). It runs in its own sandboxed process on the
same wall clock as training: thinking about data costs student steps.

Everything the refiner can do is a function of (window contents, student's
token losses, training progress) → (lessons). That is a policy, and the point
of asking for a *model* rather than a rule is that the good policies are not
writable by hand: the value of a span depends on its content, on what else is
in the window, on what the student already knows and on how much budget is
left. A module can embed documents, estimate learnability, carry memory across
calls, and be trained offline by the miner against whatever proxy the miner
believes in. The miner's edge becomes research on *teaching*, not tuning.

## 4. Intuitions

These are the mechanisms we believe carry gain, roughly in order of how cheap
they are to try. Each is a hypothesis the harness can test (§7).

### 4.1 Not every token is worth a gradient

A web document is a body of informative text wrapped in navigation, dates,
identifiers, boilerplate, repeated headers, URL fragments, and — inside the
body — stretches of text whose next token is either trivially predictable or
irreducibly random. A next-token loss on those positions still produces a
full gradient, and that gradient is mostly noise or redundancy. Selective
language modeling (Rho-1) showed that training only on the tokens that a
good reference model finds *learnable* beats uniform training by a wide
margin at the same token budget — larger gains on downstream tasks than on
loss, which is the pattern our scoring is built to see. The refiner has the
cleanest version of this lever: weight 0 keeps the token as context (the
backbone still reads it) and removes only its prediction loss. The compute
saving is modest (the LM head is ≈ 12 % of forward FLOPs for a 362 M model
with a 49 k vocabulary); the gain has to come from gradient quality, and that
is the bet.

### 4.2 The student's own loss is a free curriculum signal

"Learnable, worth learning, not yet learnt" (RHO-LOSS) is the right triage for
training points, and the refiner can approximate all three terms from signals
it already has. Tokens the student predicts well are learnt — down-weight.
Tokens whose loss stays high across rescoring are unlearnable at this
capacity, or noise — down-weight. Tokens in between, especially those whose
loss is *falling*, are where capacity is being built — up-weight, and revisit.
None of this needs a reference model: rescoring an entry after the student
has trained on neighbours turns learning *progress* into a per-token signal.
A miner who wants the Rho-1 recipe proper can ship a small reference LM inside
the 100 M-parameter weight file and compute excess loss; both approaches pay
for their forward passes on the shared clock, which is how it should be.

### 4.3 Juxtaposition: context engineering inside the corpus

The baseline packs documents in stream order, so what sits next to what is an
accident of the shard. In-context pretraining showed that packing *related*
documents into one context improves reading comprehension and in-context
learning. The window gives a refiner up to 256 entries to compose from: put a
definition next to its usage, a question-shaped passage next to the passage
that answers it, two documents on one topic in one row so the second is
predicted with the first in context. The multiple-choice suite rewards exactly
the ability to use a context to score a continuation, and this is the lever
that most directly trains it.

### 4.4 Repetition with intent

Fresh tokens are not scarce for us (§2), but *valuable* tokens are. Reading a
high-value span two or three times — ideally spaced, ideally after the student
has seen related material — is cheap, and the data-constrained scaling results
say repeated good data is nearly as useful as new data for the first few
repeats. The baseline repeats nothing. A refiner that re-reads the top decile
and skips the bottom third turns a single pass over 12 B tokens into a
curriculum over the 8 B that matter.

### 4.5 Order and recency

What a model trains on last, it remembers best. That is a liability for a
public evaluation (§6) and an asset for teaching: scheduling the densest,
most benchmark-relevant *kinds* of text late — after the student can parse
them — is legitimate curriculum, and "easy early, hard late" versus the
reverse is an open empirical question at this scale. The refiner sees
`elapsed_frac` and can plan the shape of the whole run, not just the next row.

### 4.6 One clock

A refiner that scores every entry with the student spends forward passes it
could have spent on steps: in the 300 s smoke the template trained 96 steps
against the identity baseline's 126 (−24 %). Whatever a plan buys must beat
what it costs, which is the real research question in this space — the same
question tokenizer-free models (Byte Latent Transformer, H-Net) answer when a
small model decides where the big model spends compute along the byte stream.
Deciding what to learn from is itself a compute-allocation problem, and the
shared wall clock is what keeps the answers honest.

### 4.7 The entry is the right unit

Dataset-level mixing (DoReMi, DSIR) is coarse and largely spent: FineWeb-Edu
already is a dataset-level decision. Token-level weighting alone (Rho-1)
ignores order, juxtaposition and repetition. The document is where
human-authored coherence lives — title, body, boilerplate, references, the
part worth reading and the part that is scaffolding — and it is the unit at
which a teacher naturally works. The refiner spans all three levels, anchored
on the entry.

## 5. What we expect miners to find, in order

1. **Loss masking by heuristics**: weight 0 on boilerplate, navigation,
   repeated lines, hashes and dates; weight 0 on tokens with very low or very
   high student loss. Cheap, no weights, likely the first measurable gain.
2. **Self-paced reweighting**: continuous weights from the student's own loss
   and its change under rescoring (§4.2); an EMA over calls to smooth it.
3. **Window-level selection**: retiring low-value entries immediately and
   skipping them entirely — zero cost, since the pool is larger than the
   budget.
4. **Repetition of top spans** with spacing (§4.4) and **related-span
   packing** (§4.3), both of which need a notion of similarity and so are the
   first places a learned embedding earns its weights.
5. **Run-level curricula** (§4.5) that change the mixture of lesson types with
   `elapsed_frac`.
6. **Learned value models**: a module trained offline by the miner to predict
   which spans will reduce future student loss or raise downstream probability
   — datamodel / influence-style estimators at span granularity, distilled
   into the ≤ 100 M-parameter budget. This is where the task stops being a bag
   of tricks and becomes a research programme.

Our prior on the size of the early gains is deliberately vague until §7.1 is
run, but the shape of the argument is: the optimizer race fights over 0.002
nats; FineWeb-Edu's own ablations and the Rho-1 results suggest that data
decisions move downstream scores by points, not tenths of a point. If even a
fraction of that survives the shared clock at 360 M / 12 B tokens, it is the
largest lever we have offered miners.

## 6. The measurement problem

Held-out FineWeb-Edu loss — the right metric for Challenge 1 — is the *wrong*
metric here, and not only because a refiner that can see the public held-out
recipe could target it. The best data selection changes the training
distribution and is therefore punished by a loss measured on the source
distribution; FineWeb-Edu itself raises loss on general web text while
improving every benchmark. So the score has to be downstream, and then the
hard part is signal-to-noise at 360 M parameters and 12 B tokens, where
accuracy on most benchmarks sits within a point or two of chance.

The playbook from DCLM, FineWeb, OLMo and DataDecide is: stop measuring
accuracy. We score the length-normalized probability of the correct choice
(`prob_correct_norm`), which is continuous, needs no argmax to flip, and is
the family of metrics that was found to make small-scale data decisions
predictive of larger-scale outcomes. The reference ladder confirms the suite
has dynamic range exactly where we need it: random 0.308, the full-budget
identity student 0.555 (§7.2), 0.62 at 4 T tokens, 0.70 for a 1.7 B model —
so a 20 h student sits 0.065 below SmolLM2-360M and 0.15 below the 1.7 B
reference, with a copy-prior floor of 0.40 (§7.1). The suite's text items
are pinned independently of tokenization (`text_sha256`), which is also what
lets a future encoder/decoder challenge change the tokenizer and keep the
benchmark.

What was the condition for emission (spec §6), and where it stands:

- **Noise floor — measured (§7.2).** Six paired identity runs at the full
  budget give sd 0.0030 suite points; the bar is set in those units, as
  Challenge 1's 0.03 nats was: a 0.02 floor (~6.7 sd single-run, ~4.7 sd
  paired) with the same confirmation rule, and the task pays half of miner
  emission from its own frontier (spec §9).
- **Decontamination.** Overlap between the suite's items and the FineWeb-Edu
  shards has not been measured. A refiner cannot inject text, but it can
  up-weight text that is already there.
- **Targeting within the corpus.** The remaining attack class: bias the
  student toward benchmark-shaped passages in the pool, schedule them last for
  recency, over-weight memorization. Mitigations under consideration: a sealed,
  rotating portion of the suite scored only in the confirmation run; a
  non-regression gate on a frozen secondary mixture (code, wiki, non-edu web);
  publishing each winner's weight maps so targeting is auditable.
- **The missing reference.** A full-budget Challenge-1 checkpoint scored on
  the suite would pin the band empirically rather than by Pythia analogy. It
  is blocked on checkpoint archiving, which the infra review found has never
  succeeded (presigned PUTs are SigV2; R2 wants SigV4) — fixing that unblocks
  this.

## 7. Experiments that decide the direction

All of these run on the existing harness (`sn125 refiner-eval`,
`sn125 bench-score`); the budgets are the main cost.

| id | question | design | decides |
|---|---|---|---|
| E1 | What is the noise floor? | ≥ 5 paired identity runs at 20 h; σ of suite score and of per-task scores | the bar; whether `easy_v0` is sufficient or needs more items |
| E2 | Do cheap masks help, and does the gain grow with budget? | heuristic masks (§5.1–2) at 1 h, 4 h, 20 h; Δ vs identity at each budget | whether early gains are real or a short-budget artefact |
| E3 | Is the loss signal worth its clock? | same plan with and without `wants_token_losses`; steps lost vs Δ gained | the economics of §4.2 |
| E4 | Repetition vs fresh | re-read top-k spans ×2/×3 vs identity | §4.4 |
| E5 | Order | high-value spans late vs early vs uniform | §4.5; also measures how dangerous recency is for targeting |
| E6 | Decontamination | n-gram overlap of suite items against the shards; score with and without matched documents | §6 |
| E7 | Does the suite separate what we care about? | score the pinned reference set plus the first Challenge-1 full-budget checkpoint; per-task correlation with the published numbers | suite validity before emission |
| E8 | Transfer (later) | best plan from 360 M applied to lean-llama-135m and SmolLM2-1.7B-shape at matched tokens | whether teaching policies scale, the claim that matters in a year |

E1 and E7 are prerequisites for turning emission on; E2–E5 are what we expect
the first miners to do for us, and running them ourselves first sets honest
expectations for the bar.

### 7.1 First results (2026-10-10, one-hour runs on an A100)

Numbers, tables and reproduction commands are in
[`notes/refiner-lab-2026-10-10/`](notes/refiner-lab-2026-10-10/README.md).
What they change in this memo:

- **§6, "the suite has dynamic range exactly where we need it" — it has less
  than the ladder suggests.** A knowledge-free copy prior scores 0.403
  (random init 0.308, 12 B-token band 0.43–0.45), because SciQ ships its
  support passage. The identity student is at 0.38 after one hour. Most of
  the ladder below 0.45 is in-context copying.
- **E6 is done.** SciQ overlaps the corpus heavily (489 / 1000 items share a
  12-gram, 214 are covered ≥ 50 %), the other tasks barely; 6 307 documents
  are now withheld from the entry stream. The whole pool can be skimmed by a
  refiner in about a quarter of an hour, so "find the benchmark text" was a
  practical attack, not a theoretical one.
- **§4.3–4.4 (juxtaposition, repetition with intent) did not pay at this
  budget.** Two echo designs that make the student recall spans of what it
  just read end within noise of identity (+0.001 / +0.002). The same runs
  show that dropping 10–25 % of the natural tokens costs nothing in one hour:
  a short run is step-limited, not data-limited.
- **§5.3 (window-level selection) is the first measurable gain, and it is
  large:** +0.030 to +0.045 suite score at equal steps from a 49 k-parameter
  bag-of-tokens importance model, monotone in selectivity, replicated on a
  second seed. It is also a warning about what the placeholder suite rewards
  (§6, targeting within the corpus): the gain comes with a worse held-out
  loss, i.e. from a narrower diet matched to the suite's domains.
- **E1 has a first data point, not an answer:** same-seed identity runs differ
  by up to 0.0065 in suite score at one hour; the item-sampling standard error
  of a paired delta is 0.004.
- **E2 is open in the direction that matters:** whether the selection gain
  grows, holds or inverts at 4 h and 20 h, where a narrow selection is re-read
  many times.

### 7.2 Full-budget results (2026-10-11/12, RunPod B200, the production path)

Through `scripts/refiner_pod_run.py` — the same staging, sandbox, evaluator
and verdict path a live round takes (`notes/refiner-lab-2026-10-10/README.md`
§7, `runs/pod/`):

- **The identity baseline at 72 000 s is 0.555** (188 000 steps, step-limited
  at 61 311 s; hellaswag 0.441, piqa 0.670, arc_easy 0.549, sciq 0.777,
  openbookqa 0.340; held-out 2.592, lambada 0.379). It supersedes the
  0.43–0.45 Pythia anchor in §2 and §6: a full run ends 0.17 above the
  one-hour student, not 0.05, so the suite's room above the baseline is 0.07
  to SmolLM2-360M and 0.15 to the 1.7 B reference.
- **E2 at 20 h: the selection gain holds and grows.** `select` top-20 % scores
  **0.602 vs 0.555 (Δ +0.047)** at equal steps, every MC task up (hellaswag
  +0.064, piqa +0.059, arc_easy +0.060, sciq +0.020, openbookqa +0.032) —
  against +0.030 for the same refiner at one hour. The warning from §7.1 grows
  with it: held-out loss 2.859 vs 2.592 (+0.27 nats), lambada 0.326 vs 0.379.
  A narrow diet matched to the suite's domains re-read for 20 h is a better
  suite student and a worse language model. This is the suite-design decision
  in its sharpest form (§6, §8): either `easy_v0` is the target and this is the
  intended outcome, or a held-out non-regression gate
  (`SN125_REFINER_HELDOUT_MAX_REGRESSION`, implemented, off by default) bounds
  how much ordinary-text quality a refiner may trade away.
- **E1 is done (2026-10-15): the full-budget noise floor is 0.0030.** Six
  identity runs (seeds 42–47) through the live path score 0.5555 ± 0.0030 on
  the suite (range 0.0084; held-out 2.599 ± 0.005), all step-limited at
  188 000 steps (`notes/refiner-calibration-2026-10-12/`). The one-hour
  same-seed spread (up to 0.0065 between repeats) and the 20 h across-seed
  spread (sd 0.0030, range 0.0084 over six seeds) are of one size: training
  noise does not grow with budget in suite units. Per task, sciq and lambada
  (±0.011–0.013) carry most of the spread; hellaswag, piqa and openbookqa sit
  at ±0.004–0.005. With the item-sampling SE of a paired delta at ~0.004, a
  single-seed bar below ~0.01 pays for luck; the one-seed +0.047 is ~15 sd.
  The seed-43 replicate of the top-20 % refiner was lost to a host reboot, so
  "does +0.047 reproduce" is still one ~$150 run away.

## 8. Where this leads

The refiner is the first step of a larger arc. If a learned module can decide
how a document becomes rows, the next modules can decide how text becomes
tokens (encoder) and how a model's output becomes text (decoder) — the
tokenizer-free direction where segmentation, importance and compute
allocation are one decision. The scoring principle that makes this safe is
already in place: the benchmark is defined on text, not on token ids, and
every emitted token must trace back to the corpus. Each challenge compounds
with the last: Challenge 2 trains with the Challenge 1 frontier, and a
Challenge 3 would train under the Challenge 2 plan.

The public artefact of this challenge is also unusual. A winning refiner is,
in effect, a *refined FineWeb* — a per-token weight map and curriculum over
a public corpus, reproducible from the seed. That is a research result anyone
can use at any scale, which is the standard we want every challenge to meet:
improvements that are scalable, substantial, and still matter in a year.

## 9. Reading

The ideas above are grounded in, and should be read against:

- Lin et al., 2024 — *Rho-1: Not All Tokens Are What You Need* (selective
  language modeling; token-level excess loss against a reference).
- Mindermann et al., 2022 — *Prioritized Training on Points that are
  Learnable, Worth Learning, and Not Yet Learnt* (RHO-LOSS).
- Muennighoff et al., 2023 — *Scaling Data-Constrained Language Models*
  (repetition for a few epochs is nearly free).
- Shi et al., 2023 — *In-Context Pretraining: Language Modeling Beyond
  Document Boundaries* (related-document packing).
- Sorscher et al., 2022 — *Beyond Neural Scaling Laws: Beating Power Law
  Scaling via Data Pruning*.
- Penedo et al., 2024 — *The FineWeb Datasets* (document-level curation;
  FineWeb-Edu); Li et al., 2024 — *DataComp-LM*; Magnusson et al., 2025 —
  *DataDecide* (small-scale data decisions and which metrics make them
  predictive).
- Xie et al., 2023 — *DoReMi* and *DSIR* (dataset-level mixing and selection,
  the coarse end of the spectrum).
- Evans et al., 2024 — *Data Curation via Joint Example Selection* (JEST;
  learnability-based batch selection).
- Ilyas et al., 2022 — *Datamodels* (predicting outcomes from training-set
  composition; the ancestor of learned value models).
- Pagnoni et al., 2024 — *Byte Latent Transformer*; Hwang et al., 2025 —
  *H-Net* (dynamic chunking; compute allocation along the byte stream).
- Bengio et al., 2009 — *Curriculum Learning*; Kumar et al., 2010 —
  *Self-Paced Learning*; Fan et al., 2018 — *Learning to Teach*.
