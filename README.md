# Refinery

### Accelerating Open Intelligence through incentivized research.

Refinery is **Bittensor subnet 125**: an open search for better algorithms for
training AI. Miners propose optimizers. A shared evaluation measures what they
achieve within a fixed compute budget. Confirmed improvements earn rewards.

**The first research target is gradient descent.** The aim is practical: discover
update rules that reach lower validation loss with the resources a training run
already has, then make those discoveries available for others to build on.

[Start mining](MINING.md) · [Instructions for research agents](MINING_AGENTS.MD) · [Optimizer template](sn125/miner_template.py)

## Why Refinery exists

Model performance is downstream of algorithms and data. Better optimizers are
one part of a broader opportunity: small advances across a training system can
compound into meaningful gains in capability and efficiency.

Refinery brings that research to Bittensor for three reasons:

- **Advance Open Intelligence.** Produce methods, code and evidence that help
  the open-source AI community improve its training systems.
- **Make Bittensor's research contribution visible.** Use its global talent pool
  to produce results that researchers can inspect, reproduce and challenge.
- **Bring researchers into the network.** Give optimizer specialists, ML engineers
  and research-agent teams a concrete route into mining through useful discoveries.

The ambition is broad; the initial experiment is deliberately specific. A win on
this benchmark establishes a result under its recorded conditions. Transfer to
other models, datasets and training scales remains a research question.

## What miners do

1. Implement an optimizer in Python using the supplied interface.
2. Test the candidate against matched controls on their own research budget.
3. Serve its source from a registered miner hotkey and fund evaluation credits.
4. Commit to the source hash, then reveal the same source when requested.
5. Earn rewards if the candidate clears the improvement threshold and passes
   a validator-funded confirmation run.

Mining is **algorithm research**, not supplying GPUs to the validator. The miner
service serves source code; the validator runs the scored training jobs. A GPU
is not required simply to serve a submission, but meaningful local training
experiments require suitable compute.

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

## Rewards follow progress

There is no reward merely for submitting, remaining online, or matching an
existing optimizer. Confirmed improvements receive credit proportional to their
size, with a 14-day half-life. Credit determines how rewards are shared, while
recent network progress determines how much miner emission is paid rather than
burned. The launch burn floor is 0% from day one—not a guaranteed payout.

The first research review is planned after six weeks. There is no automatic
pause. See [mining economics](MINING.md#costs-and-expected-value) for costs and
the distinction between a calculated weight and realized revenue.

## Before you start

This is a production-source distribution, **not proof of an active deployment**.
The included configuration still has a placeholder authorized validator hotkey
and an unset launch timestamp. Obtain the verified production identity, treasury,
fee, task/data pins and current frontier before registering or transferring funds.
The September 7, 2026 launch announcement does not replace those checks.

Use Linux and Python 3.12 in a dedicated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m sn125 --help
```

Next, follow [MINING.md](MINING.md). Agents should read
[MINING_AGENTS.MD](MINING_AGENTS.MD) before running tools or spending.

## What's in this repository

- `sn125/neuron.py`, `miner_template.py`: miner service, protocol and optimizer interface.
- `sn125/roundsm/`, `payments/`: round orchestration and evaluation-credit accounting.
- `sn125/training.py`, `prod_eval.py`, `engine/`: training and checkpoint scoring.
- `sn125/gate.py`, `sandbox.py`, `seccomp_sandbox.py`: submission restrictions and isolation.
- `sn125/config.py`, `fineweb.py`: protocol constants and task definition.
- `sn125/publisher.py`, `dashboard/data.py`, `dashboard/snapshot.py`: artifact publication.
- `sn125/verify.py`, `prove_authorship.py`: verification and source attribution.

The website, tests, temporary scripts, datasets, checkpoints and live round records
are not bundled. The dashboard modules here generate publication data; they do
not include a website or web server. Production R2 storage remains private.

Keep signing keys and provider credentials outside source control and outside
untrusted evaluation workers. Review the submission terms in the optimizer
template and submit only code you have the right to publish.
