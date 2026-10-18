# Refiner task (Challenge 2) — specification

> This repository is the production code release. Paths under `scripts/` and `data/`,
> and result files under `docs/notes/`, refer to the operator's working tree (lab
> tooling, the shard directory, raw measurements) and are not included here.

Status: **live from the round of Monday 2026-10-19 00:00 UTC** (§9
"Start"); harness built and exercised end to end (first refiners measured
2026-10-10, `docs/notes/refiner-lab-2026-10-10/`), wired into the round loop,
on by default, with its own frontier and half of miner emission (§9;
`SN125_REFINER_TASK=0` turns it off, `SN125_REFINER_EMISSION_SHARE=0` makes
it publish-only). This
document is the contract the code in `sn125/refiner/`, `sn125/bench/` and
`sn125/refiner_eval.py` implements. It is deliberately explicit about what is
placeholder (the benchmark suite) and what is deferred (sealed suite rotation,
raw-entry staging; §10). The research rationale — why this lever, the intuitions
behind it and the experiments that set the bar — is in
[`REFINER_RESEARCH.md`](REFINER_RESEARCH.md).

## 1. The idea in one paragraph

The optimizer task fixes the data and asks miners for a better update rule.
The Refiner task fixes the update rule and asks miners for a better way to
**learn from each document**. A refiner is a small neural module that watches
the stream of FineWeb-Edu entries and, for every entry, decides how it becomes
training rows for the student: which spans go where, in what order, next to
what, and — the key lever — the per-token loss weight, including **zero**
("read this token as context, do not backprop through its prediction"). The
student, its optimizer, the batch shape, the data and the wall-clock budget
are identical for every submission and for the baseline; the only thing that
changes is the plan.

## 2. What miners do and do not control

Miners control, per entry in their window:

- the spans of the entry (and of other entries in the window) that are
  emitted, their order and juxtaposition; spans may be repeated, split,
  dropped, or interleaved with spans of other entries;
- the per-token weight `w ∈ [0, 8]` for every emitted token; `w = 0` keeps the
  token as context without a prediction loss;
- when an entry is retired from the window, and when the student's current
  per-token losses on an entry are (re)computed for them (`wants_token_losses`,
  `rescore`);
- their own state and their own weights (≤ 100M parameters, safetensors).

Miners do **not** control: the student architecture, the student optimizer
(pinned, §5), batch size, sequence length, budget, the entry stream (seeded by
the harness, identical for baseline and submission), the benchmark suite, or
any token value — every emitted token must come from a span of an entry in the
current window (provenance by construction; the only token the harness adds is
the EOS separator a plan may request). The student's parameters are never
exposed to the refiner; only per-token losses are, on request.

## 3. Protocol

Submission = `refiner.py` (module-level `class Refiner`, optional `CAPABILITIES`)
+ optional `refiner.safetensors`. The template `sn125/refiner_template.py` is
a complete, valid submission that reproduces the baseline without weights.

```
Refiner(weights: dict[str, Tensor], ctx: dict)
Refiner.plan(window, signals, ctx) -> {"lessons": [...], "retire": [...], "rescore": [...]}
```

- `window`: list of `{"wid", "doc_id", "chunk", "n_chunks", "tokens": LongTensor[n]}`.
  Entries are whole documents; documents longer than 8192 tokens are cut
  into consecutive chunks, each its own entry.
- `signals["token_losses"][wid]`: `FloatTensor[n]`, entry `t` = student CE of
  predicting token `t` (position 0 is 0). Present only when
  `CAPABILITIES["wants_token_losses"]` is true; computed by the harness when an
  entry enters the window and on `rescore`, each entry scored on its own
  (chunked at 2048, batched by padded-length bucket). This costs roughly one
  forward pass per scored token, paid from the run's wall clock: the template,
  which scores every entry, spent ~70 s of a 300 s smoke on it (A100) and
  trained 96 steps against the identity baseline's 126.
- `ctx`: `step, elapsed_s, elapsed_frac, tokens_trained, seq_len, batch_size,
  eos_id, budget_s, plan_calls`.
- A lesson is `{"spans": [(wid, start, end) | (-1,)], "weights": FloatTensor}`
  with `(-1,)` an EOS separator; `len(weights)` equals the number of emitted
  tokens (separators included), ≤ `seq_len`; weights finite in `[0, 8]`;
  a weight below 1/64 is read as 0 (the loss is normalized by the batch weight
  sum, so only ratios matter and the floor is what bounds them).
  Short lessons are right-padded with EOS at weight 0.

Validation (`sn125/refiner/lessons.py::parse_plan`, `materialize_lesson`) runs
on the harness side; any violation fails the run as miner fault.

Limits: window ≤ 256 entries (`CAPABILITIES["window_entries"]`, default 16);
≤ 4096 lessons per plan; entries older than 64 plan calls are force-retired;
8 consecutive plans with no lessons and no retirements = stall = failure; a
`plan()` call over the deadline (default 120 s) kills the refiner process and
fails the run. Weights: ≤ 100M parameters, ≤ 450 MB, dtypes f32/f16/bf16/
int*/bool, validated structurally before loading (`validate_refiner_weights`).

## 4. Training semantics

Each training row is `(ids[S], w[S])`. The student loss is the selective
weighted CE

    L = Σ_t w_t · CE_t / Σ_t w_t      over positions t ≥ 1 with w_t > 0,

implemented as a branch of `LeanLlama.forward` over all positions with static
shapes, so the compiled forward pass covers it and a step costs the same
whatever the weights are (a zero-weight token is read as context and still
pays its LM-head FLOPs). An earlier position-gathered method ran eager under
`SN125_LEAN_COMPILE`: 1.8 steps/s on a B200 against 3.4 for the compiled
uniform path, measured on the first pod run (2026-10-10).
`LeanLlama.token_losses` provides the signal.

The student optimizer is the pinned Challenge-1 frontier (§5); it sees the
same `step`, schedule and hyperparameters as on the optimizer task. All
refiner work — process start-up and `Refiner.__init__` (recorded as
`refiner_stats.startup_seconds`), plan calls, signal scoring, materialization —
runs inside the run's wall clock, so a heavier refiner buys fewer student
steps.

Baseline = `IdentityPlanner`: pack entries in stream order into full rows at
weight 1. This reproduces the optimizer task's recipe (documents joined with
EOS, chopped to `seq_len`) through the same protocol, so baseline and
submission differ only in the plan.

## 5. Pinned student optimizer

`sn125/refiner/student/kairos_b_96fe905a.py`, sha256
`96fe905a538c32325bf550f0eed66e9ed5289489589752bbe0c8da81771ea383` (see
`ORIGIN.md` there). Recorded in every result as `student_optimizer.sha256`.
Changing the pin changes the task. Override for experiments:
`SN125_STUDENT_OPTIMIZER_PATH`.

## 6. Scoring (placeholder suite)

The Refiner task is **not** scored on held-out FineWeb-Edu loss: a refiner
chooses what the student learns from, so a loss on the training distribution
rewards teaching-to-the-test. Both final checkpoints (baseline and submission)
are scored by trusted code on a pre-tokenized suite of easy, old, public
benchmarks:

| task | items | kind | source |
|---|---|---|---|
| hellaswag | 1000 | MC | lm-eval prompt, validation |
| piqa | 1000 | MC | validation |
| arc_easy | 1000 | MC | test |
| sciq | 1000 | MC | test (with support passage) |
| openbookqa | 500 | MC | test |
| lambada | 1000 | completion | openai, test |

Per MC item with `ll_i = Σ log p(choice_i | ctx)`: `acc`, `acc_norm`
(per-char), `prob_correct_norm = softmax_i(ll_i · mean_chars / chars_i)[gold]`
(primary: smooth, length-debiased, no argmax), `prob_correct`, `bpb_gold`.
Suite score = mean `prob_correct_norm` over MC tasks; completion tasks are
diagnostic. **Submission score = suite score(submission) − suite
score(identity baseline).** Every scored sequence starts with EOS, as a
document does in training. Items, tokenization and the suite hash are pinned
in `data/.../bench/suite_easy_v0.json` (`sn125 bench-build`): `suite_sha256`
pins the token sequences, `text_sha256` pins the text items independently of
the tokenizer.

Reference points on `easy_v0` (A100, bf16, seq 2048; full numbers and
provenance in `docs/notes/bench-reference-2026-10-05/REFERENCE.md`):

| model | params | tokens seen | suite score | mean acc_norm |
|---|---|---|---|---|
| lean-llama-360m, random init (seed 0) | 362 M | 0 | 0.308 | 0.308 |
| identity baseline, 300 s smoke (126 steps) | 362 M | 8 M | 0.308 | 0.311 |
| Pythia-160m @ step 6000 † | 162 M | 12.6 B | 0.428 | 0.444 |
| Pythia-410m @ step 6000 † | 405 M | 12.6 B | 0.441 | 0.451 |
| gpt2 † | 124 M | ~10 B | 0.450 | 0.459 |
| Pythia-160m † | 162 M | 300 B | 0.449 | 0.463 |
| Pythia-410m † | 405 M | 300 B | 0.506 | 0.524 |
| SmolLM-135M (v1) | 135 M | 600 B | 0.537 | 0.559 |
| SmolLM2-135M | 135 M | 2 T | 0.551 | 0.568 |
| Qwen2.5-0.5B † | 494 M | 18 T | 0.602 | 0.616 |
| SmolLM-360M (v1) | 362 M | 600 B | 0.609 | 0.626 |
| Qwen3-0.6B-Base † | 596 M | 36 T | 0.614 | 0.630 |
| SmolLM2-360M | 362 M | 4 T | 0.620 | 0.636 |
| SmolLM2-1.7B | 1.7 B | 11 T | 0.698 | 0.716 |

SmolLM-family references are scored with `sn125 bench-score --base <name>`,
which loads the model **only** from a local, raw-byte-validated safetensors
file into the lean twin of the architecture (no framework loader, no remote
code, no download), so reference and student numbers share one forward pass.
SmolLM2-360M's per-task numbers (HellaSwag acc_norm 0.545, PIQA 0.721, OBQA
0.364) match the published card, which also validates the lean model as a
faithful SmolLM2 reimplementation. † Other-tokenizer models are scored with
`--hf <name>` on a suite rebuilt from the same text items with their own
tokenizer (`bench-build --tokenizer`; suites share `text_sha256`), in fp32
because Pythia degrades in bf16 (the lean student does not). The Pythia step
6000 checkpoints (12.6 B Pile tokens) are the nearest public analogue of a
full-budget Challenge-1 run: at ≈ 12 B tokens the references sit at 0.43–0.45,
and model size barely moves them (160m vs 410m: 0.013) while data does
(SmolLM-135M on 600 B curated tokens beats Pythia-410m by 0.03) — that gap
between what is read and how big the reader is, is the refiner's lever.
Missing reference: a full-budget (20 h, 12 B tokens) Challenge-1 checkpoint —
score one from R2 with
`sn125 bench-score checkpoints/<round>/<uid>/<task>/q100.safetensors` on a
box with R2 credentials.

Every result also carries `paired`: the item-paired bootstrap of the suite
delta (`bench.score.paired_delta`: delta, standard error and 95 % interval
over benchmark items, per-task deltas). It measures item-sampling noise only;
run-to-run training noise needs repeated runs.

**Model-free floors** (`sn125 bench-floor`): the uniform scorer gets 0.310 on
`easy_v0` and a bag-of-context *copy prior* — "the next token is a token of the
context with probability 0.5, otherwise uniform", no language knowledge at all
— gets **0.403** (SciQ 0.66, HellaSwag 0.29, the rest at chance). SciQ items
carry their support passage, the gold answer is almost always in it, and the
distractors almost never are. Two thirds of the distance between random init
and the 12 B-token reference band is therefore in-context copying, which an
identity student learns within the first hour (0.381 after 1 500 steps on an
A100, SciQ 0.50; `docs/notes/refiner-lab-2026-10-10/`). The suite has little
dynamic range left for a full-budget run to separate refiners with. Options,
in order of preference: score SciQ without its support passage, down-weight
it, or replace the aggregate by the mean over tasks of the score above each
task's copy-prior floor.

**Decontamination** (`scripts/refiner_decontam.py`): every 12-token n-gram of
every suite item (context + gold) is looked up in the training split. On
`easy_v0`: SciQ 489 / 1000 items share an n-gram with the corpus and 214 are
covered ≥ 50 % (its support passages are textbook text that FineWeb-Edu also
holds); HellaSwag 44, ARC-Easy 28, PIQA 18, LAMBADA 5, OpenBookQA 0; no item
appears verbatim. The 6 307 documents that hold a match are listed in
`DECONTAM_train_easy_v0.npy` next to the shards and **withheld from the entry
stream** for baseline and submission (`--decontam auto|require|off`, default
`auto` = use the list when present; paid runs should use `require`). The
permutation is drawn over all entries and the withheld ones dropped from it,
so a new list (suite rotation) leaves the rest of the stream in the same
order. The list's sha256 is recorded as `entry_stream.exclusions_sha256` and
re-checked inside the worker.

Known gaps, to resolve before this task carries emission:

- **noise floor / bar**: paired baseline-vs-submission runs at the full budget
  are needed to set the minimum detectable improvement (the optimizer task's
  bar came from exactly such calibration);
- **suite range**: see the copy-prior floor above — decide the aggregate before
  calibrating a bar on it;
- **the first thing the suite pays for is domain selection**: `refiners/select.py`
  reads only the documents whose vocabulary resembles public QA/commonsense
  *training* sets and gains +0.030 (top 20 %) to +0.045 (top 5 %) suite score
  over identity at equal steps in a one-hour run, with the overlapping
  documents withheld — while its held-out FineWeb-Edu loss is 0.26–0.55 nats
  worse. That is adaptation toward the suite's domains, legal under the
  current rules and not what a sealed item set prevents (the domains are
  known even when the items are not). If the challenge is meant to buy better
  teaching rather than a narrower diet, it needs a broader suite or a
  non-regression gate on general text;
- **the pool can be skimmed end to end**: retiring an entry is free, and a
  refiner that retires everything reads ~18 k entries/s, i.e. the whole
  16 B-token pool in about a quarter of an hour. Selection is a lever by
  design, but together with a public suite this makes "find the documents
  that look like the benchmark" cheap; decontamination removes verbatim text,
  not look-alikes. A cap on entries consumed per token trained is the direct
  limit if one is wanted;
- **schedule vs clock**: the pinned student optimizer's cooldown is keyed to
  `step / total_steps` while the run stops on wall clock, so an arm that
  completes fewer steps is also cut earlier on the cooldown (and an arm with
  cheaper steps anneals further). The delta then mixes "taught better" with
  "annealed more". Driving the student's schedule from `elapsed_frac` on this
  task would remove the confound;
- **suite rotation / sealed items**: a public, fixed suite invites targeting;
  the mitigation is a rotating sealed portion scored in the confirmation run.

## 7. Isolation and attack surface

- The refiner runs in its own process (`sn125/refiner/refproc.py`): source
  gated by the same `validate_single_source` as `optimizer.py` and loaded
  through the same restricted-namespace loader, per-call deadline, 35 % GPU
  memory fraction fence, no access to the student, optimizer or eval data.
- Weights: the file is validated on its raw bytes by the clean-box checkpoint
  validator (`engine.score_worker.validate_safetensors_file`: size cap, sane
  header length, JSON-object header — rejects every pickle — dtype
  allowlist, exact offset tiling, no smuggled bytes) before any parser
  touches it, then the refiner caps apply (≤ 100 M parameters, ≤ 450 MB,
  ≤ 4096 tensors, no F64). The validator stages a private copy, validates and
  hashes **that copy**, and the refiner process re-validates the bytes it
  actually loads.
- Pipe: lesson payloads from the untrusted process are size-capped (64 MB,
  checked from the length header before the body is read) and unpickled with
  an allowlist of exactly the types the protocol uses (numpy arrays and
  builtins); any other global, or an oversize message, kills the process and
  fails the run. Tensors cross as numpy arrays (no shared-memory or CUDA
  handles). Everything that arrives is then validated by `parse_plan`.
- The worker that hosts the student is the existing sandbox (namespaces or
  seccomp, uid nobody, no network, read-only project). The refiner's weight
  file is staged to a nobody-readable path by the validator before launch.
- Final checkpoints are scored by trusted code through the
  same raw-byte validator with the model's exact tensor inventory
  (`bench.score.score_checkpoint_on_suite`); SmolLM-family references go
  through the same validator into the lean twin (`--base`), never a framework
  loader. The sha256 of the file that is scored must equal the sha the
  training loop reported for its final export, or the run fails as a harness
  fault. Other-tokenizer references (`--hf`) are allowlisted by name, loaded
  from the local cache only, safetensors only, every file raw-byte validated
  first, through the native `transformers` class with remote code refused
  (`auto_map` in the config is rejected outright).
- Provenance by construction (§2) removes the "inject arbitrary tokens" class.
  What remains: a refiner can bias the student toward benchmark-like text
  within the corpus, over-weight memorization, or spend the budget — all
  measured by the same score and the same clock the baseline is measured on.
- Sandbox note for miners: `torch.tensor(list)` / `torch.as_tensor(list)` are
  not available inside the restricted namespace; use
  `torch.zeros/ones/full/arange/stack/cat/randn`.

A failed result carries `fault: miner | harness`. Only gate rejections,
protocol violations (`LessonError`), refiner-process failures (`RefProcError`,
`RefProcTimeout`) and a diverged loss are attributed to the miner; everything
else is a harness fault and must not settle as a score.

`--baseline-json` reuses a previous result's baseline only when that result is
a full refiner-eval record with an identical `task`, `entry_stream`,
`student_optimizer.sha256`, `execution` regime and suite.

## 8. Commands

```bash
# one-time per box (validator, as root): document index + suite
python -m sn125 refiner-index --data-dir "$SN125_FINEWEB_DIR"
python -m sn125 bench-build  --data-dir "$SN125_FINEWEB_DIR" --jsonl-dir data/pilot_encdec_v0/bench

# evaluate a submission (identity baseline + submission, then bench both)
SN125_USE_OPTPROC=1 python -m sn125 refiner-eval refiner.py --weights refiner.safetensors \
  --checkpoint-dir runs/rf-001 --out runs/rf-001/result.json
# smoke: --budget 300 --total-steps 400 --eval-every 100 --eval-sequences 256 --timeout 1800
# reuse a baseline: --baseline-json runs/rf-000/result.json

# model-free floors of the suite (chance, copy prior)
python -m sn125 bench-floor --data-dir "$SN125_FINEWEB_DIR"
# suite/corpus overlap + the exclusion list the entry stream honours (as root, ~9 min on 8 cores)
python scripts/refiner_decontam.py --data-dir "$SN125_FINEWEB_DIR" --write-exclusions --out runs/decontam.json
# research: ONE arm, every quartile checkpoint scored; then compare arms/seeds
SN125_USE_OPTPROC=1 python scripts/refiner_lab.py identity --budget 3600 --total-steps 1500 \
  --bench q25,q50,q75,q100 --out runs/lab/identity_s42.json
python scripts/refiner_lab_compare.py runs/lab/*.json --at q100

# score any checkpoint on the suite
python -m sn125 bench-score runs/rf-001/submission.safetensors --model-spec hf:lean-llama-360m
# reference base model (local HF cache, safetensors only, never downloads)
HF_HUB_OFFLINE=1 python -m sn125 bench-score --base HuggingFaceTB/SmolLM2-360M
# other-tokenizer reference: rebuild the suite with its tokenizer (explicit --out), then --hf
python -m sn125 bench-build --jsonl-dir data/pilot_encdec_v0/bench --tokenizer EleutherAI/pythia-410m --out /tmp/suite_p410.json
HF_HUB_OFFLINE=1 python -m sn125 bench-score --hf EleutherAI/pythia-410m --revision step6000 --suite /tmp/suite_p410.json --no-amp
```

Staging a reference model on a disk-tight box: `snapshot_download(name,
allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"])` into
`HF_HUB_CACHE` on tmpfs (`/dev/shm`), score offline, delete — see
`docs/notes/bench-reference-2026-10-05/REFERENCE.md`.

Smoke runs: set `--total-steps` to the number of steps the identity arm
completes in `--budget` on that box (≈ 0.43 steps/s on an A100 with
`SN125_FORCE_GC=1`), otherwise the student's schedule never reaches its
cooldown and neither arm is representative. The shard dir must be readable by
uid 65534 through every parent directory (a symlink into `/root` is not).

`--checkpoint-dir` must be on the real filesystem (not `/tmp`, which the
sandbox worker replaces with a private tmpfs, and not under `/root`, which it
sees read-only); the data dir must be readable by uid 65534. The student runs
in the production flash regime (no gradient checkpointing), which fits a B200
but not an 80 GB card; on A100-80GB set `SN125_FORCE_GC=1` (the refiner path
costs no extra memory over a plain run — both peak ~6.3 GB with checkpointing).

## 9. In live rounds (on by default, 50% of emission)

A stock validator accepts refiner submissions with no configuration: they ride
the same rounds as optimizer submissions — same fee and credit, same commit
and reveal windows, same FIFO selection into the same evaluation slots, same
boxes — and the task pays **`REFINER_EMISSION_SHARE` = 50% of miner emission**
from its own frontier (below). The inputs every evaluation is pinned to ship
with the code: the suite and decontamination list in `sn125/refiner/pinned/`,
the seed-42 identity baseline in `sn125/refiner/baselines/` (copies in the
shard directory, or the `SN125_REFINER_*` variables below, take precedence).
`SN125_REFINER_TASK=0` turns the task off: a refiner envelope then fails the
gate and the round record is byte-for-byte what it was without Challenge 2.

**Start.** The task is live from the round that opens at or after
`config.REFINER_START_TIMESTAMP` — **Monday 2026-10-19 00:00:00 UTC**
(`1792368000`). A round that opened earlier is Challenge 1 only: no
`refiner_task` block, no refiner jobs or confirmations, the Challenge-1
weight vector unblended, and an envelope fails the gate with
`refiner task opens at 2026-10-19 00:00:00 UTC` (a gate rejection, so its
credit is burned like any other). Rounds themselves open on a 24-hour grid
anchored on the same instant (`config.ROUND_GRID_ANCHOR_TIMESTAMP`,
`next_round_start`): a round that finishes before the next 00:00 UTC waits
for it, one that overran opens the next round at once, and before the anchor
no round opens that could not run its full period first (`round_hold_until`),
so the round that opens *at* the anchor is the launch round. Each round's
`round.start` audit event records `refiner_live`, the grid anchor and the
chain block at which it opened.

Two failure modes, chosen by whether the operator set the variable at all.
On by default, anything that would make the task unscorable — a missing input,
a packaged baseline the box would reject for this validator's regime (seed,
budget, student, shard manifest, sandbox flavour) — is logged once at start
and the task stays off for that process; Challenge 1 is never taken down by
Challenge 2's configuration. With `SN125_REFINER_TASK=1` set explicitly the
same conditions stop the validator. An emission share outside [0, 1] stops
it in either mode, and a paying share with no usable pinned baseline is a
configuration error (the pool pays against the baseline's frontier), so in
default mode it leaves the task off.

**What a miner commits to.** The transport carries one string of at most 1 MB
whose SHA-256 is the commit. A refiner miner serves a canonical JSON envelope
(`sn125/refiner/submission.py`):

```json
{"refinery_submission":1,"source":"<refiner.py>","task":"RF1_smol_360M",
 "weights":{"bytes":98436,"sha256":"<64 hex>","url":"https://…/refiner.safetensors"}}
```

`weights` is `null` for a weight-free refiner. The commit therefore binds the
source and the weight file's hash and size; the URL is revealed with the
envelope, after the commit window has closed. Only the canonical serialization
is accepted, so one submission has one hash.

```bash
python -m sn125 mine --refiner refiner.py \
    --weights refiner.safetensors --weights-url https://…/refiner.safetensors
```

**What the validator does with it.**

1. Gate before selection (`make_source_gate(refiner_enabled=True)`): envelope
   shape, the single-source gate on `refiner.py`, literal `CAPABILITIES`. A
   failure burns the fee, as for an invalid optimizer.
2. On the box, network still on (`cloud._stage_refiner_inputs`): upload the
   suite and the pinned identity baseline, put the decontamination list next to
   the shards, and `curl` the weight file — https only (including redirects),
   size-capped at the committed size — then compare size and SHA-256 with the
   commitment. A file that is missing or different is the miner's fault (fee
   burned); miner code has not run. A verified file is archived once,
   content-addressed, to R2 (`refiner_weights/<sha256>.safetensors`, through a
   short-TTL presigned PUT; `refiner_weights_archive` in the result), so a
   result can be re-run after the miner's URL is gone.
3. Network cut, then `refiner-eval … --decontam require
   --expected-suite-sha256 … --expected-decontam-sha256 … --baseline-json …`
   under the same sandbox flavour and launch environment as `prod-eval`. The
   box prints one `REFINER_EVAL_RESULT {json}` line (per-item scores stay in
   the `--out` file).
4. Verdict: a result marked `fault: "harness"` (suite, baseline, entry stream,
   bench scoring, no result line) refunds the credit; `fault: "miner"` (lesson
   contract, refiner process, diverged loss, weight file) burns it.
5. Record: scored refiner runs go to a signed `refiner_task` section of the
   round file and attestation (`roundsm/refiner_round.py`), and the envelope is
   published in `sources` under its commit hash. They never enter
   `compute_weights`; the task's own frontier settles them (below) and the
   block's `frontier` section carries the result. Each entry carries a `signature` — a hash of the task
   id, student model, budget, batch and sequence length, data manifest, seed,
   suite, decontamination list and pinned student optimizer, computed from
   what the box actually ran (`refiner_signature`) — and the public URIs of the
   submission run's q25/q50/q75/final student checkpoints, uploaded at
   teardown like Challenge 1's. Two entries are comparable only within one
   signature. `python -m sn125.verify` checks the block
   (`check_refiner_task`: envelope ↔ commit ↔ source hash ↔ weights
   commitment, score arithmetic, suite/decontamination pins, signature,
   emission share), replays the frontier (`check_refiner_frontier`, below)
   and fetches the refiner checkpoints with `--fetch-base`; the dashboard
   shows the block under "Challenge 2".

**What it pays: the refiner frontier** (`roundsm/refiner_frontier.py`). The
optimizer task's mechanism (`INCENTIVE_MECHANISM.md`) run a second time, in
the task's own unit and its own pool, then blended:

- *Unit.* Suite points: the absolute benchmark-suite score of the student a
  refiner produced (`bench.submission`, 0–1). The frontier is the pinned
  identity baseline's suite score (0.5554, the seed-42 run; the six-seed mean is 0.5555) until an event
  confirms; afterwards it is the confirmed event's score.
- *Bar.* A submission is a candidate when `suite(submission) − frontier`
  exceeds the adaptive bar: `max(0.02, 0.5 × mean of the last three events ×
  0.5^(stagnant days / 7))`. The 0.02 floor (`MIN_REFINER_IMPROVEMENT`) is
  ~6.7 sd of the measured full-budget noise (sd 0.0030, n = 6, §6 and
  `notes/refiner-calibration-2026-10-12/`) and ~4.7 sd of a paired
  submission-vs-frontier difference — the posture of Challenge 1's 0.03-nat
  floor (~4.8 sd). An entry whose held-out gate tripped scores 0 and is never
  a candidate. Only entries with the frontier's `signature` count; a change
  of suite, student, budget, data or seed starts a new frontier.
- *Confirmation.* The best candidate of a round is recorded as
  `pending_confirmation`; next round the validator re-runs its envelope at
  its own cost (one more 20 h evaluation, in the same batch as the round's
  evals). The event is created only if the confirmation also clears the bar
  against the frontier the candidate was measured against; the event's score
  is the WORSE of the two runs. A run the miner's code fails rejects the
  candidate; a run that could not happen (no box) is retried up to three
  times. All of it is in `refiner_task.frontier` (`confirmation_outcomes`,
  `new_event`, `pending_confirmation`) and the confirmation run itself in
  `refiner_task.confirmations`.
- *Pool.* Every confirmed event `i` with improvement `g_i` (suite points) at
  `t_i` carries `credit_i(t) = g_i × 0.5^((t − t_i) / 14 d)`;
  `P = Σ credit_i`; the pool pays `min(1, P / R(t)) × (1 − burn floor)` of
  itself pro rata to credit and burns the rest. `R` ramps from 0.02 at launch
  (the first floor-clearing event pays the whole pool) to 0.07 after 56 days
  (`REFINER_FULL_PAY_PROGRESS_LAUNCH`, `REFINER_FULL_PAY_PROGRESS`): one
  floor-clearing event a week keeps the pool fully paid, `0.02 / (1 − 0.5^0.5)
  = 0.068`. No leader bonus; a stagnant pool is mostly burning within four to
  six weeks.
- *Blend.* `weights = 0.5 × optimizer frontier + 0.5 × refiner pool`, each a
  burn-inclusive map summing to 1 (`blend_weights`); the burn UID receives
  both pools' unearned emission. A hotkey holding both frontiers is paid from
  both. `SN125_REFINER_EMISSION_SHARE` overrides the 0.5; 0 is publish-only
  (the frontier is still settled and published, nothing is paid).
- *Replay.* `python -m sn125.verify` recomputes the bar, the event, the
  pending list, the pool's progress and payable share and its weight map from
  the prior public records plus this round's entries and confirmation runs
  (`check_refiner_frontier`), and checks the published weights are the blend
  (`settlement.weights`). The frontier's state lives in the round records
  alone, as Challenge 1's does.

**Validator inputs** (resolved and hashed at start; see above for what a
missing or mismatched one does):

| variable | default | |
|---|---|---|
| `SN125_REFINER_TASK` | on | accept and evaluate refiner submissions (`0` turns the task off, `1` makes a half-configured task fatal) |
| `SN125_REFINER_EMISSION_SHARE` | 0.5 | the refiner pool's share of miner emission (`config.REFINER_EMISSION_SHARE`); 0 = publish-only; outside [0, 1] refuses to start; a paying share needs a usable pinned baseline |
| `SN125_REFINER_SUITE` | `<data>/bench/suite_easy_v0.json`, else `sn125/refiner/pinned/suite_easy_v0.json` | shipped to every box, sha pinned |
| `SN125_REFINER_DECONTAM` | `<data>/DECONTAM_train_<suite>.npy`, else `sn125/refiner/pinned/DECONTAM_train_<suite>.npy` | shipped to every box, sha pinned |
| `SN125_REFINER_BASELINE_JSON` | `sn125/refiner/baselines/RF1_smol_360M_<suite>_seed<eval seed>.json` | pinned identity-baseline result; with `SN125_REFINER_TASK=1` and no packaged baseline for the eval seed, every eval trains its own baseline first (two runs) |
| `SN125_REFINER_HELDOUT_MAX_REGRESSION` | unset (off) | non-regression gate, nats: a submission whose student ends worse than the baseline on held-out FineWeb-Edu by more than this scores 0 for the round; the suite delta stays on the record (`components.suite_delta`, `gates`) |
| `SN125_REFINER_EVAL_SEED` | unset (42) | the seed every refiner eval trains both arms at; must be the pinned baseline's seed (checked at start); recorded in `task.seed` and the entry `signature` |
| `SN125_REFINER_BASELINE_ONLY` | unset | calibration knob for operator drivers (`scripts/refiner_calibration.py`): train and score only the identity arm; never set on a live validator |

Produce the baseline once per (manifest, suite, decontamination list, student,
seed, sandbox flavour, budget) on a production box:

```bash
SN125_USE_OPTPROC=1 python -m sn125 refiner-eval sn125/refiner_template.py --baseline-only \
    --decontam require --checkpoint-dir /workspace/rf_baseline --out refiner_baseline.json
```

Reuse is strict (`_baseline_mismatch`): a baseline made under `namespaces`
does not serve a `seccomp` box, so a mixed provider chain needs one per flavour
— today one path is configurable, so keep the chain to one flavour while the
task is on. The validator applies the same test at start
(`pinned_baseline_regime_mismatch`: seed, budget and steps, student optimizer,
data manifest, sandbox flavour of the provider chain, optproc/AMP/force_gc) and
refuses to run with a baseline the box would reject — a mismatch found on the
box costs a full rental per submission and a round that publishes nothing.
The pinned baselines made so far (seed 42 and the calibration seeds 43–47,
`sn125/refiner/baselines/`) all pass it on the `runpod` chain; a seed-43+
baseline needs `SN125_REFINER_EVAL_SEED` set to its seed.

Or through the live path, which is what the calibration campaign does
(`scripts/refiner_calibration.py`: identity baselines at several seeds plus a
refiner replicate, on rented B200s, under a hard dollar cap, each driver with
its own workload prefix so the validator's orphan sweep and the drivers never
delete each other's pods). Its results are pinned baselines made by the exact
production staging, sandbox and evaluator; `docs/REMAINING_FOR_LAUNCH.md` §C2
lists the runs and what they decide.

## 10. Deferred

- Done 2026-10-17: the start instant and the 00:00 UTC round grid (§9
  "Start"; `config.ROUND_GRID_ANCHOR_TIMESTAMP`, `REFINER_START_TIMESTAMP`).
- Done 2026-10-15: the frontier — bar, confirmation, 50% emission split, the
  verifier replaying it (§9).
- The doc index and decontamination list in the manifest (their sha256 is
  recorded in each result and pinned per round, not verified by
  `stage-shards`; a box without the doc index rebuilds it).
- Staging raw entries with FineWeb metadata (URL, score) to R2 so refiners can
  be studied offline; today refiners see token ids only.
- Sealed suite rotation (§6); the suite-design decision itself (is `easy_v0`
  the target, or does `SN125_REFINER_HELDOUT_MAX_REGRESSION` guard it) and the
  aggregate-vs-single pinned baseline — operator decisions, listed in
  `REMAINING_FOR_LAUNCH.md` §C2.
- Done 2026-10-12: durable weight-file archive, refiner audit checkpoints,
  per-entry `signature`, `verify` and dashboard reading the block, the
  calibration campaign tooling.
