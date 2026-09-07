# Mining on Refinery

Your job is to discover an optimizer that trains better than the current
confirmed frontier under the subnet's evaluation system. Passing the source
gate is necessary; it is not evidence that a submission will earn rewards.

This guide covers human-operated mining. Autonomous workflows must also follow
[MINING_AGENTS.MD](MINING_AGENTS.MD).

The path is: verify the incentive mechanism → develop and test → register and serve →
fund credits → commit/reveal → evaluation → confirmation. Research can begin
without registration; paid submission cannot begin without verified launch inputs.

Use this guide in order on your first attempt. For an existing miner, jump to
[serving](#6-register-and-serve), [credits and rounds](#7-track-credits-rounds-and-confirmation),
or [troubleshooting](#attribution-and-troubleshooting).

## 1. Decide whether this work fits you

Refinery is suited to researchers and engineers who understand tensor operations,
can run reproducible training experiments, and can afford unsuccessful attempts.
Research agents can contribute when their operators provide these capabilities,
useful feedback and bounded compute.

We do not advise mining if you do not believe you either have an existing path to improving existing optimizers,
or have the capital to use frontier autoresearch loops to attempt this. 
Serving an unchanged reference optimizer does not establish
a competitive advantage.

## 2. Verify the inputs before spending

| Required input | How it is used |
| --- | --- |
| Network / subnet | `finney` / `125` |
| Authorized validator hotkey | Public SS58 address; must match the local trust pin |
| Treasury coldkey and current fee | `5DLu5XrMV8Wt7aSmutxwAT1tNdwXtLxPDHvd5JAiY6WDnnX7`; fee pinned per round |
| Your registered hotkey and owning coldkey | Identity that serves source and owns evaluation credits |
| Public endpoint | Reachable IP and TCP port; default port 8091 |
| Task revision, data manifest and shard access | Reproduce the intended evaluation conditions |
| Current frontier, adaptive bar and deadlines | Decide whether and when a submission is competitive |

The configured public treasury is `5DLu5XrMV8Wt7aSmutxwAT1tNdwXtLxPDHvd5JAiY6WDnnX7`.
The default reference is `$250/TAO`; the validator adds a 10% margin to its
estimated full-run cost and pins the resulting TAO fee before commits open.
Confirm the signed round announcement and live fee before transferring funds.
Do not infer addresses from examples or pay an address supplied by an unverified
message. Registration costs and evaluation fees are separate.

In this source snapshot, `AUTHORIZED_VALIDATOR_HOTKEY` in
[`sn125/config.py`](sn125/config.py) is still a placeholder and `LAUNCH_TIMESTAMP`
is unset. The placeholder rejects commit/reveal requests. Obtain the correctly
pinned production release or update the pin only from verified release details;
do not remove the verification logic.

Create your own wallet. The miner needs its operational hotkey and public wallet
metadata, not the validator's cloud or R2 keys. Keep the coldkey signing material
on a trusted wallet machine. Optional research-provider and shard-read credentials
belong to your own accounts and should be least-privilege.

## 3. Prepare the environment and candidate

Use Linux and Python 3.12:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp -n sn125/miner_template.py my_optimizer.py
```

The package targets PyTorch 2.8.0 and Bittensor 10.2 to below 11. For a B200
research worker, use the matching CUDA 12.8 PyTorch wheel:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0' 'torchvision==0.23.0'
```

Read [`sn125/miner_template.py`](sn125/miner_template.py) before editing. Define
`HPARAMS` and `Optimizer`, with `__init__`, `step`, `state_dict` and
`load_state_dict`. `step` receives clipped gradients, parameter values and a
zero-indexed step number. Return a mapping of parameter names to update tensors;
the harness applies `parameter += update`.

Your optimizer owns the schedule. Optional `CAPABILITIES` change buffer delivery
and trusted weight-decay behavior; use them only after understanding the template.
Do not apply weight decay twice. State must survive save/restore.

The source limit is 1 MB. Imports and operations are restricted by
[`sn125/gate.py`](sn125/gate.py). File/network/process access, dynamic execution
and reflection are restricted. Standard-library `random` is forbidden; supported
torch RNG is available. Treat generated candidate source as untrusted code.

Before a long run, check these interface invariants:

- Update keys match the parameters being updated; tensors have compatible shapes
  and devices, and contain finite values.
- The sign is correct: the harness **adds** the returned update. An SGD-style
  descent update is `-lr * gradient`, not `lr * gradient`.
- The optimizer uses the supplied step number and schedule horizon rather than
  assuming that a short smoke run has the production horizon.
- Saving and restoring state preserves the next update, including any momentum,
  statistics and counters required by the method.

## 4. Check the source without executing it

This check parses the candidate; it does not run the optimizer or rent compute:

```bash
python - <<'PY'
from pathlib import Path
from sn125.gate import validate_source
candidate = Path('my_optimizer.py')
assert candidate.is_file(), 'Candidate file is missing'
violations = validate_source(candidate.read_text())
print(violations or 'Source gate passed')
raise SystemExit(bool(violations))
PY
```

Next, `python -m sn125 validate-file my_optimizer.py` runs an interface check on
a CUDA worker. Unlike parsing, this executes candidate code. Use an isolated
worker without wallet or provider secrets. This command's restricted Python
loader is not the full production OS isolation boundary. Its “Ready to submit”
message means only that this interface check
passed; it is not a loss result, an isolation attestation or a reward qualification.

## 5. Run controlled research evaluations

The initial production contract is a Llama-style 360M model, pinned FineWeb-Edu,
sequence length 2,048 and batch size 32. Training ends at 188,000 steps or 72,000
seconds on a B200, whichever comes first. Clean scoring uses 8,192 held-out
sequences. See [`sn125/fineweb.py`](sn125/fineweb.py) and the live round contract.

Set the following to verified values: `SN125_FINEWEB_DIR`,
`SN125_FINEWEB_HF_REPO`, `SN125_FINEWEB_HF_REVISION` and
`SN125_FINEWEB_MANIFEST_HASH`. Do not use an unpinned repository revision for
a production comparison. Dataset and checkpoint binaries are not in this package.

The commands below require these variables to be set; stop if any is missing:

```bash
: "${SN125_FINEWEB_DIR:?Set the local shard directory}"
: "${SN125_FINEWEB_HF_REPO:?Set the verified dataset repository}"
: "${SN125_FINEWEB_HF_REVISION:?Set the immutable dataset revision}"
: "${SN125_FINEWEB_MANIFEST_HASH:?Set the verified manifest hash}"
python -m sn125 stage-shards \
  --data-dir "$SN125_FINEWEB_DIR" \
  --hf-repo "$SN125_FINEWEB_HF_REPO" \
  --hf-revision "$SN125_FINEWEB_HF_REVISION" \
  --expected-manifest "$SN125_FINEWEB_MANIFEST_HASH"
```

Private shard access may need your own approved Hugging Face read token. Keep
held-out data out of miner-controlled training environments; follow the clean
scoring layout required by your worker. Never disable hash or isolation checks
to make an experiment pass.

After approving compute spend, use a short smoke run on a disposable CUDA worker:

```bash
mkdir -p runs/smoke-001
SN125_USE_OPTPROC=1 SN125_REQUIRE_SANDBOX=1 SN125_LEAN_COMPILE=1 \
python -m sn125 prod-eval my_optimizer.py \
  --data-dir "$SN125_FINEWEB_DIR" \
  --expected-manifest "$SN125_FINEWEB_MANIFEST_HASH" \
  --budget 300 --total-steps 100 --timeout 1800 \
  --checkpoint-out runs/smoke-001/candidate.safetensors \
  --out runs/smoke-001/result.json
```

Use a fresh output directory for each attempt. Supported container workers need
`SN125_SANDBOX_MODE=seccomp`; VM workers use the namespace path. A sandbox setup
failure is not permission to turn off protection. The validator's production
path additionally applies provider, throughput and isolation checks.

This shortened horizon is an execution smoke test, **not a prediction of the
20-hour result**. For meaningful comparisons, match the production model, horizon,
budget, seed and data, include a control, and repeat promising candidates. Use
`--seed` to select the intended seed. Record configuration, exact source hash,
loss, runtime, failures and checkpoint provenance.

`--baseline-json` expects the baseline bundle format, not arbitrary dashboard
JSON. Without it, evaluation can train a reference as well as the candidate,
increasing time and cost. Inspect the JSON result, including failure and clean
scoring fields; process exit alone does not establish a valid result.

## 6. Register and serve

Using trusted Bittensor wallet tooling compatible with the installed SDK:

1. Verify the current registration cost and register your hotkey on subnet 125,
   network `finney`, within your approved spending limit.
2. Start the source server with an existing candidate file.
3. Advertise its public endpoint on-chain and verify it in the metagraph.
4. Fund evaluation credits from the coldkey that owns the registered hotkey.

```bash
test -f my_optimizer.py && python -m sn125 mine \
  --wallet-name YOUR_WALLET_NAME \
  --wallet-hotkey YOUR_HOTKEY_NAME \
  --port 8091 --optimizer my_optimizer.py
```

**The miner starts its Axon but does not advertise it on-chain.** Registering a
hotkey and starting the process are not sufficient. Use the installed SDK's
`Subtensor.serve_axon` or compatible wallet tooling to publish your actual
external IP and port. Inspect method signatures and transaction results, then
re-query the metagraph; do not assume a submitted transaction succeeded. Allow
the authorized validator to reach the port through your firewall/NAT.

For the installed SDK, the endpoint advertisement looks like this. Replace the
placeholders and execute only after registration, with authorization for the
on-chain operation; the source server above must already be running:

```python
import bittensor as bt

wallet = bt.Wallet(name="YOUR_WALLET_NAME", hotkey="YOUR_HOTKEY_NAME")
subtensor = bt.Subtensor(network="finney")
assert subtensor.is_hotkey_registered(wallet.hotkey.ss58_address, netuid=125)
axon = bt.Axon(wallet=wallet, port=8091,
               external_ip="YOUR_PUBLIC_IP", external_port=8091)
result = subtensor.serve_axon(
    netuid=125, axon=axon,
    wait_for_inclusion=True, wait_for_finalization=True,
)
print(result)
```

This advertises an endpoint; it does not start a second source server. Check the
returned result and the metagraph entry before funding evaluation credits.

The service reads the source once at startup. A missing optimizer path currently
falls back to default AdamW, which is why the file-existence check above matters.
Do not change source or restart onto different bytes between commit and reveal.
There is no manual submission REST endpoint: the validator requests the hash and
source from your running miner.

## 7. Track credits, rounds and confirmation

Default cadence is a three-hour commit window, one-hour reveal window and roughly
20-hour evaluation. Operator settings, capacity and previous-round completion
determine actual deadlines. Do not schedule solely from these defaults.

- Each finalized transfer grants `floor(transfer_rao / fee_rao)` credits under
  the processed fee pin. Two half-fee transfers do not combine into one credit.
- The paying coldkey owns the credits; hotkeys with that owner share them. Avoid
  unrelated sender accounts or exchange withdrawals that credit another identity.
- One credit is debited when a commitment is accepted. Missing the reveal still
  costs that credit.
- Gate-passing capacity overflow retains its paid credit and FIFO priority.
  Check deferred status before paying again. Confirmation jobs and spending caps
  can reduce the available capacity from the configured maximum of eight.
- Miner-caused failures do not earn rewards. Infrastructure disqualification
  refunds an evaluation credit, not a TAO transfer.
- A candidate must improve validation loss by **more than** the adaptive bar.
  Its floor is 0.03 nats, but the actual threshold can be larger.
- Confirmation uses the same source, is validator-funded, and must also qualify.
  The worse loss of the pair becomes the confirmed frontier. Keep the hotkey
  registered: shares for absent hotkeys are routed to burn.

For example only: a frontier of 2.80 with a 0.03 bar requires loss below 2.77.
A loss of exactly 2.77 does not strictly clear the bar. Always obtain live values.

## Costs and expected value

Profitability is unknown until you have evidence and costs. Include research GPU
time, agent/API use, serving, registration, evaluation fees and transaction costs.
The validator's proposed weight is not a guaranteed token receipt or cash return.

Confirmed improvement credit halves every 14 days. With total active credit `P`,
your credit `C` and reference `R`, the default zero-floor weight is:

```text
your_weight = min(1, P / R) * C / P     when P > 0
your_weight = 0                       when P = 0
```

The reference ramps from 0.03 to 0.10 nats over 56 days. Competition, future
improvements, registration status and chain settlement affect actual proceeds.
No confirmed progress means no miner reward; full emission eligibility from day
one is not guaranteed income. The six-week research review is not an automatic pause.

Evaluate each additional attempt over an explicit horizon:

```text
incremental EV = expected realizable reward proceeds
                 - all additional research, submission and operating costs
```

Include the probability of qualifying **and** confirming, selection delays,
competition, token-price uncertainty, fees and slippage. Use timestamped chain
and liquidity data, not static dashboard estimates. Do not treat sunk spending
as a reason to fund another attempt. Stop when your budget is exhausted, the
frontier removes your expected advantage, or evidence no longer supports the cost.

## Attribution and troubleshooting

Read the submission license grant in the template. Submit only source you can
authorize for evaluation and publication. Retain the exact submitted bytes and
their SHA-256; do not put secrets in source, metadata, prompts or logs.

| Symptom | Check first |
| --- | --- |
| No validator requests | Production trust pin, registration, advertised endpoint, firewall and round window |
| Commit rejected | Finalized credits, paying coldkey, fee pin and source hash |
| Old code still served | Source loads at startup; change only outside commit/reveal obligations |
| Local smoke passes, evaluation fails | Correct task/data, GPU build, sandbox, memory and scoring output |
| No reward after a lower loss | Strict adaptive threshold, confirmation status and current registration |
| Credit seems missing | Accepted commit versus deferred carryover; inspect the ledger before repaying |

Verify received round records with `python -m sn125.verify --rounds-dir PATH`.
Inspect skipped checks: settlement replay and a valid signature do not prove
that training was independently reproduced. `python -m sn125.prove_authorship
--help` describes optional signed attribution.
