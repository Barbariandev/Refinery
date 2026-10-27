# Refinery

Accelerating Open Intelligence through incentivized research.

Refinery is **Bittensor subnet 125**: an incentivized research network for
improving the algorithms behind language-model training. It connects a global
pool of contributors to a shared objective: make open models learn more from
the compute available to them.

Miners submit training-algorithm code. Validators train and independently
score the resulting checkpoints under a fixed compute budget. Improvements that
pass confirmation earn rewards.

**The first research target is gradient descent** (Challenge 1). The aim is
practical: discover update rules that reach lower validation loss with the
resources a training run already has, then make those discoveries available for
others to build on. **The second is what a model learns from** (Challenge 2,
the Refiner task, open from the round of Monday 2026-10-19 00:00 UTC): with the
optimizer, model, data pool and budget fixed, decide what the student reads, in
what order, and which tokens receive a gradient. Each challenge pays half of
miner emission from its own confirmed frontier.

[Start mining](MINING.md) · [Instructions for research agents](MINING_AGENTS.MD) · [Optimizer template](sn125/miner_template.py) · [Refiner task](docs/REFINER_TASK.md) · [Refiner miner's guide](docs/REFINER_MINING.md) · [Refiner template](sn125/refiner_template.py)

## Why Refinery exists

Model performance is downstream of algorithms and data. The difference between
training systems need not come from one breakthrough: improvements in update
rules, architectures, pretraining data and reinforcement-learning environments
can compound. We believe a sustained, open search across these components can
help narrow the capability and efficiency gap between open and closed models.

We see that the Bittensor network is a bastion of capable researchers who we believe
are well suited to advancing the open frontier.
We also see that the capabilities of Frontier language models have reached a point
where semi automated research appears plausible, enabling these researchers to 
work more effectively and explore idea spaces more thoroughly.


Refinery starts with one measurable part of that ambition. Instead of asking
contributors to build an entire frontier model, it asks them to improve how
one learns—and provides a common evaluation and a reward for verified progress.

Refinery brings that research to Bittensor for three reasons:

- **Advance Open Intelligence.** Produce methods, code and evidence that help
  the open-source AI community improve its training systems.
- **Make Bittensor's research contribution visible.** Use its global talent pool
  to produce results that researchers can inspect, reproduce and challenge.
- **Bring researchers into the network.** Give optimizer specialists, ML engineers
  and research-agent teams a concrete route into mining through useful discoveries.

## Why start with optimizers?

An optimizer determines how a model learns from each batch of data. A better
update rule can reach lower validation loss with the same compute budget—or
reach a useful loss sooner. That makes optimizer research a concrete first
target: contributors can change one component and measure its effect under a
shared training contract.

Our two starting beliefs are that bounded training runs can identify methods
worth testing at larger scales, and that Bittensor's global talent pool can
discover those methods. LLM research agents may broaden that pool when paired
with useful feedback, enough compute and the right constraints. The subnet's
results will test those beliefs.

We believe this benchmark is our best practical way to measure optimizer
performance within the resources of an evaluation run. Transfer to other
models, datasets and training scales still needs to be measured. Data selection
is now Challenge 2; other training components are future research directions,
not current mining tasks.

## What miners do

1. Implement an optimizer in Python using the supplied interface.
2. Test the candidate against matched controls on their own research budget.
3. Serve its source from a registered miner hotkey and fund evaluation credits.
4. Commit to the source hash, then reveal the same source when requested.
5. Earn rewards if the candidate clears the improvement threshold and passes
   a validator-funded confirmation run.

Challenge 2 follows the same path with a refiner instead of an optimizer: a
small program, plus an optional weight file committed by hash and size, that
sits between the document stream and the student's training batches. A hotkey
serves one submission at a time, either an optimizer or a refiner
([docs/REFINER_MINING.md](docs/REFINER_MINING.md)).

Mining is **algorithm research**, not supplying GPUs to the validator. The miner
service serves source code; the validator runs the scored training jobs. A GPU
is not required simply to serve a submission, but meaningful local training
experiments require suitable compute.

The intended output is reusable research: optimizer source, training evidence
and results that other researchers can investigate. A subnet benchmark win is
a starting point for broader evaluation, not a claim of universal superiority.

## The initial evaluation

| Component | Production contract in this release |
| --- | --- |
| Model | In-repository Llama-style approximately 360M-parameter model |
| Data | Manifest-pinned FineWeb-Edu; SmolLM2 tokenizer |
| Training shape | Sequence length 2,048; batch size 32 |
| Compute | One B200; at most 20 hours |
| Step limit | 188,000 steps, or the compute budget, whichever comes first |
| Objective | Lower validation loss from clean checkpoint scoring |
| Qualification | Improvement strictly exceeding the adaptive bar; floor 0.03 nats |
| Confirmation | Both the initial and confirmation run must qualify |

The submitted optimizer owns its learning-rate schedule. The confirmed frontier
uses the worse loss of the qualifying pair. Read the authenticated round contract
before spending: historical reference numbers are not a live target.

Challenge 2 trains the same 360M student with a pinned optimizer under the same
budget and scores the checkpoint on a benchmark suite. A refiner qualifies by
beating the current frontier (the pinned identity baseline until a refiner
clears it) by more than the adaptive bar (floor 0.02 suite points), confirmed
by a second run at the validator's cost ([docs/REFINER_TASK.md](docs/REFINER_TASK.md) §9).

## Rewards follow progress

There is no reward merely for submitting, remaining online, or matching an
existing optimizer. Confirmed improvements receive credit proportional to their
size, with a 14-day half-life. Credit determines how rewards are shared, while
recent network progress determines how much miner emission is paid rather than
burned. The launch burn floor is 0% from day one—not a guaranteed payout.
Each challenge runs this mechanism in its own pool and its own unit; the
validator's weights are one half the optimizer pool plus one half the refiner
pool, and each pool's unearned share burns.

The initial optimizer search is planned for at least six weeks, followed by a
review of the results and a discussion of whether to continue, revise the task,
or pause for research. The code does not schedule an automatic pause.
See [mining economics](MINING.md#costs-and-expected-value) for costs and
the distinction between a calculated weight and realized revenue.

### Evaluation fee reference

The public launch treasury is `5DLu5XrMV8Wt7aSmutxwAT1tNdwXtLxPDHvd5JAiY6WDnnX7`.
The default TAO reference is `$250/TAO`, with a 10% margin over the estimated
full-run rental cost. The fee is calculated at startup and pinned for each
round. Never transfer funds until the signed round announcement confirms the
fee and treasury address.

## Before you start

This source release is **not proof of an active deployment**.
The included configuration pins the production validator identity
(`AUTHORIZED_VALIDATOR_HOTKEY`, uid 203 on netuid 125), the treasury coldkey and
the round schedule: rounds open on a 24-hour grid at 00:00 UTC anchored on
Monday 2026-10-19 00:00 UTC, the round that also opens Challenge 2
(`ROUND_GRID_ANCHOR_TIMESTAMP`, `REFINER_START_TIMESTAMP` in
`sn125/config.py`). `LAUNCH_TIMESTAMP` is 0 by design: the optimizer pool's
payout ramp anchors on its first confirmed frontier event. Verify identity,
treasury, fee, task/data pins and the current frontier against the signed
round announcement before registering or transferring funds.
An announced date does not replace those checks.

Use Linux and Python 3.12 in a dedicated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m sn125 --help
```

Next, follow [MINING.md](MINING.md): build a candidate, check it without executing
it, then run isolated experiments before considering a paid submission. Agents
should read [MINING_AGENTS.MD](MINING_AGENTS.MD) before running tools or spending.

## Production code

Validator deployment (git checkout with `validate --auto-update`, or the
container) is documented in [DEPLOYMENT.md](DEPLOYMENT.md). The payment ledger
and the inter-round state checkpoint are durable across restarts; the
supervisor applies upstream releases at round boundaries only.

- `sn125/neuron.py`, `miner_template.py`: miner service, protocol and optimizer interface.
- `sn125/roundsm/`, `payments/`: round orchestration and evaluation-credit accounting
  (durable ledger, continuous treasury scan, archive backfill, round-state checkpoint).
- `sn125/autoupdate.py`: the `--auto-update` supervisor (safe-point restarts from git).
- `sn125/training.py`, `prod_eval.py`, `engine/`: training and checkpoint scoring.
- `sn125/gate.py`, `sandbox.py`, `seccomp_sandbox.py`: submission restrictions and isolation.
- `sn125/config.py`, `fineweb.py`: protocol constants and task definition.
- `sn125/refiner/`, `bench/`, `refiner_eval.py`, `refiner_template.py`: Challenge 2 — the refiner
  protocol, the pinned student optimizer, the benchmark suite, decontamination list and identity
  baselines, the scorer and the refiner interface.
- `sn125/roundsm/refiner_round.py`, `refiner_frontier.py`: Challenge 2 inside the round (envelopes,
  settlement, its own frontier and the 50/50 blend of the two pools).
- `sn125/publisher.py`, `dashboard/data.py`, `dashboard/snapshot.py`: artifact publication
  (rounds, full audit trail and process logs, payment ledger, state checkpoint) to private R2/HF.
- `sn125/ops.py`: the miner-operations feed published alongside (per-coldkey credits,
  deferred queue, live-round status, recent outcomes, logbook index).
- `sn125/logfile.py`: rotating on-disk copy of the process log under the audit dir.
- `sn125/verify.py`, `prove_authorship.py`: verification and source attribution.

The production ZIP excludes the website, tests, temporary scripts, datasets,
checkpoints and live round records. `docs/` carries the Challenge-2 contract, miner's guide and
research memo; `refiners/` the two reference refiners (`recall.py`, `select.py`). The dashboard
modules here generate publication data; they do not include a website or web server. Production R2
storage remains private.

Keep signing keys and provider credentials outside source control and outside
untrusted evaluation workers. Review the submission terms in the optimizer
template and submit only code you have the right to publish.
