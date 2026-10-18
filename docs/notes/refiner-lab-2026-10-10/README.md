# Refiner lab notes — 2026-10-10

> This repository is the production code release. Paths under `scripts/` and `data/`,
> and result files under `docs/notes/`, refer to the operator's working tree (lab
> tooling, the shard directory, raw measurements) and are not included here.

First end-to-end research pass over the Refiner task (Challenge 2) on the
local A100-80GB: where the identity baseline sits on the suite as it trains,
what the suite actually measures, how much benchmark text is in the corpus,
and the first refiners written against it. Spec: [`REFINER_TASK.md`](../../REFINER_TASK.md);
rationale: [`REFINER_RESEARCH.md`](../../REFINER_RESEARCH.md).

Setup for every run below unless noted: lean-llama-360m from scratch, pinned
student optimizer (KAIROS-B `96fe905a`), batch 32 × 2048, bf16,
`SN125_FORCE_GC=1` (≈ 0.44 steps/s), budget 3600 s with `--total-steps 1500`
(the identity arm completes 1 500 steps in 3 423 s, so the student's schedule
runs to its end and arms are step-matched), seed 42, suite `easy_v0`
(`ad25712a…`). One arm per run via `scripts/refiner_lab.py`; the 25/50/75/100 %
checkpoints are all scored. Raw results: `results/*.json` (per-item scores
included where the run post-dates their introduction).

## 1. The identity baseline on the suite

| steps | tokens | held-out loss | suite | hellaswag | piqa | arc_easy | sciq | openbookqa | lambada acc |
|---|---|---|---|---|---|---|---|---|---|
| 0 (init) | 0 | 11.01 | 0.308 | 0.265 | 0.502 | 0.263 | 0.234 | 0.275 | 0.000 |
| 375 | 25 M | 4.584 | 0.331 | 0.239 | 0.524 | 0.308 | 0.334 | 0.250 | 0.005 |
| 750 | 49 M | 4.086 | 0.356 | 0.245 | 0.524 | 0.332 | 0.438 | 0.240 | 0.047 |
| 1 125 | 74 M | 3.827 | 0.375 | 0.248 | 0.543 | 0.353 | 0.480 | 0.252 | 0.096 |
| 1 500 | 98 M | 3.688 | 0.381 | 0.249 | 0.546 | 0.358 | 0.501 | 0.251 | 0.110 |

MC cells are `prob_correct_norm`. In one hour and 98 M tokens the identity
student covers more than half of the distance from random init (0.308) to the
12 B-token reference band (0.43–0.45), and three quarters of that movement is
SciQ.

## 2. What the suite measures: the copy-prior floor

`python -m sn125 bench-floor` scores two model-free "students":

| scorer | suite | hellaswag | piqa | arc_easy | sciq | openbookqa |
|---|---|---|---|---|---|---|
| uniform over the vocabulary | 0.310 | 0.267 | 0.508 | 0.262 | 0.232 | 0.282 |
| copy prior, mix 0.1 | 0.379 | 0.288 | 0.515 | 0.273 | 0.546 | 0.274 |
| copy prior, mix 0.5 | 0.403 | 0.292 | 0.515 | 0.276 | 0.663 | 0.268 |
| copy prior, mix 0.9 | 0.411 | 0.288 | 0.516 | 0.277 | 0.709 | 0.267 |

The copy prior predicts "a token of the context with probability *mix*,
otherwise uniform" and knows nothing else. It reaches 0.40 because SciQ items
carry their support passage: picking the choice with the largest share of
tokens present in the context is right on 77 % of SciQ items. LAMBADA's target is in
its context for 75 % of items. Consequences:

- of the 0.13 between random init and the 12 B-token band, about 0.09 is
  reachable by copying from context alone;
- the identity student is at the mix ≈ 0.1 level after one hour, so a full
  20 h run starts the remaining race from roughly 0.38–0.40 and has ≈ 0.05 of
  range left to separate refiners with;
- any change that makes SciQ less of a lookup (dropping the support passage,
  down-weighting the task, or scoring each task above its own copy-prior
  floor) gives the range back to knowledge.

## 3. Decontamination scan (E6)

`scripts/refiner_decontam.py`, n = 12 tokens, context + gold continuation of
every item against all 16.0 B training tokens (8 cores, 8–9 min). Full output:
`decontam_easy_v0_ngram12.json`.

| task | items | with a 12-gram in the corpus | coverage ≥ 50 % | ≥ 90 % | verbatim | too short to test |
|---|---|---|---|---|---|---|
| sciq | 1000 | 489 | 214 | 6 | 0 | 0 |
| hellaswag | 1000 | 44 | 6 | 0 | 0 | 0 |
| arc_easy | 1000 | 28 | 4 | 0 | 0 | 7 |
| piqa | 1000 | 18 | 3 | 0 | 0 | 45 |
| lambada | 1000 | 5 | 0 | 0 | 0 | 0 |
| openbookqa | 500 | 0 | 0 | 0 | 0 | 224 |

SciQ's support passages are textbook text that FineWeb-Edu also holds; the
question-and-answer part is not there, so no item is verbatim. 6 307 documents
hold at least one match and are now withheld from the entry stream
(`DECONTAM_train_easy_v0.npy`, sha256 `858ba2b4…`). Withholding them does not
move the baseline beyond noise (§5).

A refiner that retires every entry unread goes through the feeder at
≈ 18 000 entries/s, i.e. the whole 15.5 M-entry pool in about 14 minutes. A
refiner can therefore afford to look at every document in the corpus; with a
shipped table of suite n-gram hashes it would find the overlapping documents
by itself, which is why they are removed at the source.

## 4. Refiners

All arms complete 1 500 steps (step-matched with the baseline); planning costs
36–128 s of the 3 600 s.

| arm | idea | suite @375 | @750 | @1125 | @1500 | Δ vs paired identity @1500 | held-out loss |
|---|---|---|---|---|---|---|---|
| identity | — | 0.3309 | 0.3560 | 0.3753 | 0.3812 | | 3.688 |
| identity, decontaminated stream | — | 0.3296 | 0.3580 | 0.3793 | 0.3878 | | 3.688 |
| block echo (`recall.py` ancestor: 96–512-token passage, then 25 % echoes of 2–12 tokens) | recall in blocks | 0.3295 | 0.3585 | 0.3741 | 0.3829 | +0.002 | 3.902 |
| `refiners/recall.py` (echo of 1–4 tokens every 8–40 tokens, resume at weight 0) | recall inline | 0.3303 | 0.3624 | 0.3725 | 0.3823 | +0.001 | 3.831 |
| `refiners/select.py`, top 20 % | read the top documents of a DSIR-style importance model | 0.3542 | 0.3884 | 0.3999 | 0.4182 | **+0.030** [+0.023, +0.038] | 3.948 |
| `refiners/select.py`, top 5 % (shipped default) | same, narrower | 0.3575 | 0.3860 | 0.4155 | 0.4329 | **+0.045** [+0.037, +0.054] | 4.237 |

Per task at 1 500 steps (`prob_correct_norm`):

| arm | hellaswag | piqa | arc_easy | sciq | openbookqa | lambada acc |
|---|---|---|---|---|---|---|
| identity (decontaminated) | 0.259 | 0.557 | 0.365 | 0.511 | 0.248 | 0.127 |
| select top 20 % | 0.254 | 0.590 | 0.402 | 0.563 | 0.281 | 0.116 |
| select top 5 % | 0.271 | 0.605 | 0.410 | 0.587 | 0.291 | 0.093 |

Second seed, half the budget (1 800 s, 750 steps, full schedule, decontaminated
stream, seed 43): identity 0.3564, select top 20 % 0.3871, **Δ +0.031**
[+0.023, +0.038].

Brackets are 95 % bootstrap intervals over benchmark items (item-sampling
noise). The select arms are compared with the decontaminated identity run, the
echo arms with the plain one.

Reading:

- **Echo / recall: null.** Teaching in-context copying explicitly ends within
  noise of the baseline (SciQ +0.00 / +0.02, everything else ±0.01), although
  the student gives up 10–25 % of its natural tokens for it. The identity
  student already picks the copy prior up from natural text, and — the more
  useful half of the result — a one-hour run is not limited by the amount of
  text it reads.
- **Selection: a clear win on every task that moves.** Reading only documents
  whose vocabulary resembles public QA/commonsense *training* sets raises the
  suite score by 0.030 (top 20 %) to 0.045 (top 5 %) at equal steps, with
  verbatim suite text removed from the stream. The top-5 % arm reaches 0.433 —
  the 12 B-token Pythia band — on 98 M tokens. The gain is monotone in
  selectivity, appears at every checkpoint, and replicates on a second seed
  and budget.
- **It costs held-out loss** (3.95 / 4.24 vs 3.69): the student is trained on
  a narrower distribution than FineWeb-Edu. That is exactly why the task is
  not scored on held-out loss, and exactly why this result has to be read as
  *domain adaptation toward the suite*, not as "better teaching". A suite
  made of grade-school science QA and how-to commonsense pays for reading
  grade-school science and how-to text. Whether that is the behaviour the
  challenge wants to buy is a suite-design decision (broader suite, a
  non-regression gate on general text, or accepting it).
- Not measured here: whether the gain survives a 20 h / 12 B-token run, where
  the top 5 % (0.8 B tokens) would be re-read ~15 times.

## 5. Noise at this budget

- Two identity runs with the same seed whose streams differ by the 6 307
  withheld documents (0.04 %): suite 0.3812 vs 0.3878 at 1 500 steps
  (Δ 0.0065), and −0.001 / +0.002 / +0.004 at the earlier checkpoints.
- Item-sampling standard error of a paired delta: ≈ 0.004.
- So at one hour a single-seed difference below ≈ 0.01 is not evidence. The
  select deltas are 5–7× the same-seed spread; the echo deltas are inside it.
  This is two runs, not a calibration: E1 (≥ 5 paired runs at the real
  budget) is still what sets a bar.

## 6. The official paired command

`python -m sn125 refiner-eval` (identity baseline + submission, final
checkpoints, `--decontam require`), 1 800 s budget, 750 steps, seed 44 — a
third seed, and the path a validator would run:

| submission | baseline suite | submission suite | **score** | 95 % (items) | held-out Δ (diagnostic) |
|---|---|---|---|---|---|
| `refiners/select.py`, top 20 % | 0.3546 | 0.3875 | **+0.0329** | [+0.025, +0.041] | +0.23 |
| `refiners/select.py`, top 5 % (shipped; baseline reused via `--baseline-json`) | 0.3546 | 0.3893 | **+0.0347** | [+0.026, +0.043] | +0.51 |

Per-task deltas of the shipped refiner: PIQA +0.046, ARC-Easy +0.034, SciQ
+0.060, OpenBookQA +0.034, HellaSwag ±0.000. Result files:
`results/refiner_eval_select_1800_s44.json`,
`results/refiner_eval_select05_1800_s44.json`.

Across three seeds and two budgets the top-20 % selection scores +0.030,
+0.031, +0.033; the narrower selection pulls ahead only in the longer run
(+0.045 at one hour against +0.035 at half an hour).

## 7. Reproduce

```bash
export SN125_FINEWEB_DIR=/srv/sn125_fineweb SN125_USE_OPTPROC=1 SN125_FORCE_GC=1   # A100-80GB
python -m sn125 bench-floor
python scripts/refiner_decontam.py --data-dir $SN125_FINEWEB_DIR --write-exclusions --out runs/decontam.json
python refiners/tools/build_domain_prior.py --data-dir $SN125_FINEWEB_DIR --out refiners/select.safetensors
python scripts/refiner_lab.py identity --budget 3600 --total-steps 1500 --seed 42 --decontam \
    --bench q25,q50,q75,q100 --out runs/lab/identity_dc_3600_s42.json
python scripts/refiner_lab.py refiners/select.py --weights refiners/select.safetensors --budget 3600 \
    --total-steps 1500 --seed 42 --decontam --bench q25,q50,q75,q100 --out runs/lab/select_dc_3600_s42.json
python scripts/refiner_lab_compare.py runs/lab/*_dc_3600_s42.json --label-from-file --base identity_dc_3600_s42
```

The block-echo arm is `refiners/recall.py` with `GAP_MIN, GAP_MAX = 96, 512`,
`N_SUB_MIN, N_SUB_MAX = 8, 20`, `SUB_MIN, SUB_MAX = 2, 12` (the original file
drew echoes from the current passage only; the result file is
`results/echo_v1_3600_s42.json`). `select top 20 %` is `KEEP_TOP = 0.20`.

## 7. Update 2026-10-11

Full-budget identity baseline (RunPod B200, 72 000 s, 188 000 steps, pinned
student optimizer, decontaminated stream): **suite 0.555** (hellaswag 0.441,
piqa 0.670, arc_easy 0.549, sciq 0.777, openbookqa 0.340; lambada acc 0.379).
Result: `results/identity_full_188k_pod_baseline.json` (usable as
`SN125_REFINER_BASELINE_JSON`). This supersedes the "0.43–0.45 band" and
"≈ 0.05 of range left" statements in §1–§2: a full run ends 0.17 above the
one-hour identity student, not 0.05.

More one-hour arms (same setup as §4, seed 42, decontaminated stream):

| arm | suite @25 % | @50 % | @75 % | @100 % | steps |
|---|---|---|---|---|---|
| identity | 0.3296 | 0.3580 | 0.3793 | 0.3878 | 1 500 |
| select top 5 % | 0.3575 | 0.3860 | 0.4155 | 0.4329 | 1 500 |
| select top 1 % | 0.3488 | 0.3845 | 0.4152 | 0.4345 | 1 237 |
| per-task union (top 1.5 % of any of the five task scores, 4.9 % kept) | 0.3713 | 0.3954 | 0.4234 | 0.4362 | 1 486 |
| teacher-loss oracle: weight 0 where the 200k-step AdamW model's loss > 6 (14 % of tokens) | 0.3271 | 0.3548 | 0.3759 | 0.3815 | 1 500 |

- Selectivity saturates near 5 %: top 1 % scores the same and loses 18 % of
  its steps to scanning.
- The per-task union is ahead of the pooled top 5 % at every checkpoint, by
  0.003 at the end — inside the same-seed spread (§5), so not established.
- The oracle arm is a null, and not a test of "drop inherently unpredictable
  tokens": a 360 M teacher at loss 2.78 cannot separate inherently
  unpredictable tokens from ones it is too weak to predict.

Full-budget refiner arm (same pod setup, top 20 %, pinned baseline above):
**suite 0.602 vs 0.555, Δ +0.047**, 188 000 steps, all five MC tasks up
(hellaswag +0.064, piqa +0.059, arc_easy +0.060, sciq +0.020, openbookqa
+0.032); held-out loss 2.859 vs 2.592, lambada acc 0.326 vs 0.379. One seed,
no item-paired interval (the pinned baseline carries no per-item scores).
Result: `results/select20_full_188k_pod.json`.

Trading suite gain against held-out loss (one hour, seed 42):

| arm | suite | Δ suite | held-out loss | Δ held-out |
|---|---|---|---|---|
| identity | 0.3878 | | 3.688 | |
| select top 20 % | 0.4182 | +0.030 | 3.948 | +0.260 |
| select top 5 % | 0.4329 | +0.045 | 4.237 | +0.549 |
| top 5 % + every 57th other document (≈ 25 % ordinary text) | 0.4193 | +0.031 | 3.824 | +0.136 |
| top 5 % + every 170th other document (≈ 10 % ordinary text) | 0.4293 | +0.042 | 3.945 | +0.257 |

At equal suite score the 25 % mix has half the held-out cost of top 20 %; at
equal held-out cost the 10 % mix scores 0.011 more than top 20 %. A narrow
selection diluted with ordinary text is a better trade than a wide selection.

Second seed, half budget (1 800 s, 750 steps, seed 43): identity 0.3564
(held-out 4.099), top 20 % 0.3871 (4.315), 10 % mix 0.3926 (4.328). The mix
is ahead of top 20 % on both seeds (+0.011, +0.006) at equal held-out loss;
each difference alone is within the same-seed spread.

Per-task union + 10 % mix (one hour, seed 42): suite 0.4361, held-out 3.911,
1 469 steps. Same suite score as the per-task union alone (0.4362) and above
the pooled 10 % mix (0.4293, held-out 3.945) on both axes; one seed.

Second seed for the per-task + 10 % mix (`submissions/refiner_v3`, 1 800 s,
seed 43): 0.3901 on 734 steps, held-out 4.301 — between top 20 % (0.3871) and
the pooled 10 % mix (0.3926). Its seed-42 lead over the pooled mix (+0.007)
does not replicate; both stay ahead of top 20 % on both seeds.

Ordering (one hour, seed 42): ordinary stream for the first half, top 5 % for
the second half — suite 0.3278 / 0.3588 / 0.4014 / 0.4175 at the four
checkpoints, held-out 3.931. Half the selected text bought almost the suite
score of the 25 % mix (0.4193), but at a worse held-out loss (3.931 vs 3.824):
ending on the narrow distribution costs general text more than mixing does.

Reverse order (top 5 % for the first half, ordinary stream for the second):
suite 0.3554 / 0.3880 / 0.4041 / 0.4107, held-out 3.724, lambada acc 0.124.
The suite keeps rising through the ordinary half (+0.023 over identity at the
end, half of the pure top-5 % gain) while held-out loss returns to within
0.036 of identity. Per unit of held-out cost this is the cheapest gain of any
arm here; one seed.

Top 5 % for the first 75 %, ordinary stream for the last 25 % (one hour,
seed 42): suite 0.3621 / 0.3908 / 0.4197 / 0.4272, held-out 3.920, lambada
acc 0.103. Same point as the 10 % mix (0.4293, 3.945) within noise: ordering
and mixing trade suite for held-out loss along one line, and neither shape
gets off it. The 50/50 split remains the only arm near identity's held-out
loss.

Discriminative scorer (one hour, seed 42). A logistic regression over mean
token weights plus mean hashed token-pair weights (2^22 buckets, 4.2 M
parameters), trained to separate benchmark TRAIN items from length-matched
corpus windows (held-out AUC 0.9984 vs 0.9883 for the likelihood-ratio
table; its top 5 % shares 47 % of documents with the table's). Pure top 5 %:
suite 0.3599 / 0.3926 / 0.4128 / 0.4335 on 1 455 steps, held-out 4.020,
lambada acc 0.091. Same suite score as the table's top 5 % (0.4329) at 0.22
lower held-out loss and 3 % fewer steps (pair hashing costs planning time).
Task mix shifts toward arc_easy (0.427 vs 0.410) and sciq (0.610 vs 0.587)
and away from hellaswag (0.255 vs 0.271) and piqa (0.589 vs 0.605). One seed.

Discriminative scorer + 10 % ordinary text, two seeds. One hour, seed 42:
suite 0.4255, held-out 3.870 (1 470 steps) against the table's 10 % mix at
0.4293 / 3.945. Half budget, seed 43: 0.3934 / 4.262 (735 steps) against
0.3926 / 4.328. Suite score is level with `refiner_v2` on both seeds (−0.004,
+0.001, inside the same-seed spread); held-out loss is 0.07 lower on both.
The scorer buys general-text quality, not suite score.

Sum of the two scorers (table rescaled to the regression's spread and added
to its token weights; document-score correlation between the two is 0.69),
pure top 5 %, one hour, seed 42: suite 0.3639 / 0.3892 / 0.4248 / 0.4410 on
1 450 steps, held-out 4.127, lambada acc 0.088. Highest one-hour suite score
so far (table top 5 % 0.4329, per-task union 0.4362), +0.008 over either
scorer alone — just outside the same-seed spread; one seed.

Summed scorer + 10 % ordinary text, two seeds. One hour, seed 42: suite
0.3687 / 0.4009 / 0.4175 / 0.4349 on 1 452 steps, held-out 3.908 (table mix
`refiner_v2`: 0.4293 / 3.945). Half budget, seed 43: 0.3916 on 717 steps,
held-out 4.292 (v2: 0.3926 / 4.328, 750 steps). +0.006 and −0.001 against
v2: the pure-top-5 % lead does not replicate as a suite gain. Held-out loss
is 0.04 lower on both seeds, and the pair hashing costs 3–4 % of the steps.
Across scorers, mixes and orderings every one-hour variant now sits within
about one same-seed spread (0.0065) of 0.43; this proxy no longer separates
them.

Per-token relevance weights (`refiner_v2` + weight exp(0.25·llr[token]),
renormalised to mean 1 per row; range about 0.4–2.7), one hour, seed 42:
suite 0.3637 / 0.4007 / 0.4151 / 0.4320, held-out 4.005 (unweighted),
hellaswag 0.285 (highest of any arm), 1 500 steps. +0.003 over v2 (0.4293):
inside the same-seed spread.

Three seeds per candidate (half budget, 750 steps, seeds 43 / 44 / 45; all
top 5 % + 10 % ordinary text):

| arm | s43 | s44 | s45 | mean suite | mean held-out | steps |
|---|---|---|---|---|---|---|
| `refiner_v2` (likelihood-ratio table) | 0.3926 | 0.3901 | 0.3892 | 0.3906 | 4.281 | 750 |
| summed scorer | 0.3916 | 0.3888 | 0.3884 | 0.3896 | 4.244 | 717–726 |
| v2 + per-token relevance weights | 0.3908 | 0.3886 | 0.3933 | 0.3909 | 4.344 | 750 |

The three means span 0.0013. Neither the richer scorer nor token weighting
moves the suite score at this budget; `refiner_v2` stays the candidate as the
simplest of the three.
