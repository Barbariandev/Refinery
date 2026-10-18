# Validator deployment

Two supported shapes:

1. **Git checkout with the auto-update supervisor** (recommended):
   `python -m sn125 validate --auto-update <the usual flags>`. The checkout
   follows its git upstream and the validator restarts itself only at round
   boundaries, where nothing is in flight. See "Auto-update" below.
2. **Container** (CPU coordinator image; RunPod supplies the B200 workers).
   Watchtower stays monitor-only, because an image swap lands mid-round.

Both shapes rely on the same durable state, all of it plain files:

- **Payment ledger** — credits, spent credits, carried remainders, processed
  treasury transfers and the chain scan cursor, written atomically to
  `<audit-dir>/payments/ledger.json` (`--payment-ledger` / `SN125_PAYMENT_LEDGER`)
  on every change. A restart neither loses balances nor re-credits a deposit.
  Deposits are scanned continuously (`--payment-scan-interval`, default 60 s); a
  fresh ledger backfills `--payment-backfill-hours` (default 72) of deposits,
  reading pruned blocks from the archive endpoint (`--payment-archive-network`).
- **Round state checkpoint** — `<audit-dir>/validator_state.json`, written at
  every round boundary: the round counter, the deferred (paid, not yet
  evaluated) submissions carried to the next round, the fairness window and the
  last weight map for the epoch refresh. A process started over this file
  continues where the previous one stopped.
- **Round files and audit logs** — `sn125/rounds/*.json`, `<audit-dir>/*.jsonl`
  and `.log`, plus a rotating copy of the process log itself
  (`<audit-dir>/validator.log`, and `autoupdate.log` for the supervisor;
  `SN125_PROCESS_LOG=0` disables, `SN125_PROCESS_LOG_MAX_BYTES` /
  `SN125_PROCESS_LOG_BACKUPS` size it, default 32 MiB × 5).

### Off-box mirror (private R2)

With `R2_ENDPOINT` / `R2_ACCESS_KEY` / `R2_SECRET_KEY` / `R2_BUCKET` in the
validator's `.env`, the continuous publisher mirrors **all of the above** to the
private bucket every 60 s (`SN125_PUBLISH_INTERVAL_S`), under `artifacts/`:
round files, the whole audit trail and process logs, the payment ledger, the
state checkpoint, and a derived miner-operations feed (`ops/credits.json`,
`ops/evaluations.json`, `ops/logbook.json`, `ops/ops.json`) that answers
"did my payment land / where is my submission / what happened" per coldkey and
hotkey. A dead box therefore loses no balance, no queued submission and no log
line older than one cycle; restoring is copying `payments/ledger.json` and
`state/validator_state.json` back into the new `<audit-dir>`. Key layout and
the feed's fields are documented in the `sn125/publisher.py` and `sn125/ops.py`
module docstrings.
The bucket must stay private:
the raw archive contains miner source. `SN125_PUBLISH_HF_REPO` +
`HF_WRITE_TOKEN` adds a private Hugging Face dataset as a second sink.

What a restart still costs: the GPU spend of any evaluation running at that
moment. The miners' side is covered: credits debited in a round that never
published are returned automatically when the validator starts again (the
ledger and the round's audit file show which commits were accepted and which
were burned by miner fault before the abort; see `_refund_abandoned_rounds`),
and the commits themselves are re-collected in the next round. That is why the
supervisor stops the validator mid-round only when a push explicitly asks for
it (`UPDATE_NOW`, below) and why a plain `docker restart` / Watchtower swap
remains an operator decision, not an automatic one.

Start-up also runs a second restitution pass, `_refund_no_gpu_dqs`: every
published round's audit is re-read, and a commit the round finalized as a
miner-fault DQ whose recorded reason says no GPU was ever rented (capacity
wait, provisioning failure) gets its credit back, once, with a `refund` ledger
event naming the hotkey. This corrects the 2026-09-08 incident (0.2.1 and
earlier classified a `CapacityWait` raised during rental as miner fault) and
stays in place as a guard: a verdict without a box is never the miner's. The
published round file is not rewritten; the ledger event is the record. Both
passes run from one place, `_reconcile_ledger_at_startup`, which appends a
line per start to `<audit-dir>/payments_reconciliation.jsonl` (mirrored to R2
as `audit/payments_reconciliation.jsonl`) with the counts, the ledger events it
wrote and any error, so "what did the restart change in the ledger" has one
answer even when the answer is nothing.

### Launch window: sporadic GPU capacity

From 0.2.3 a round no longer needs B200 inventory at one instant. After
selection the driver opens a launch window (`SN125_LAUNCH_WINDOW_S`, default
14400 = 4 h; `fsm_launch_window_s` on the validator object takes precedence).
Inside it the capacity gate waits at most for the remaining window instead of
indefinitely, and every selected submission's worker retries its rental
whenever inventory reappears (`SN125_LAUNCH_RETRY_S`, default 300 s poll).
Boxes already running are not affected — the window gates starts, not
finishes. When the window closes, whatever never started is carried to the
next round as DEFERRED with its credit and FIFO priority
(`RoundFSM.defer_unlaunched`), round after round; a commit carried
`MAX_UNLAUNCHED_DEFERRALS` (30) rounds that is also older than
`UNLAUNCHED_MAX_AGE_S` (30 days) is published as `expired` and its credit is
absorbed — neither refunded nor recorded as miner fault. The round publishes,
and if *nothing* launched
the loop skips the remaining period and opens the next round at once. Audit
events: `launch_window.opened/closed`, `capacity.wait_expired`,
`cloud_eval.awaiting_box`, `evaluation.deferred`; the miners feed shows the
statuses `awaiting_gpu`, `deferred` and `expired`.

From 0.2.4 the reveal window is polled the same way: the validator re-queries
every still-committed miner's axon for its source every `SN125_REVEAL_RETRY_S`
(default 300 s; `fsm_reveal_retry_s` takes precedence; `0` = single query at
window open) until the window deadline. A miner whose axon is down when the
window opens but back within the hour still reveals; only a commit that never
answers is consumed as unrevealed.

## Auto-update (git checkout)

```bash
git clone https://github.com/<org>/<refinery-repo>.git refinery && cd refinery
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp /path/to/.env .                     # secrets; ignored by git
# edit sn125/config.py if you must (kept across updates, see below)
python -m sn125 validate --auto-update \
  --wallet-name W --wallet-hotkey H --provider runpod --set-weights
```

`--auto-update` accepts every flag `validate` accepts and passes them through
unchanged; the only extra one is `--update-interval` (seconds between upstream
checks, default `SN125_UPDATE_INTERVAL_S` or 300). The process you start is a
small supervisor; the validator runs as its child.

Update cycle:

1. The supervisor fetches the upstream branch (`SN125_UPDATE_REMOTE`, default
   `origin`; `SN125_UPDATE_BRANCH`, default the checked-out branch).
2. If upstream is ahead it writes a restart-request flag and otherwise does
   nothing. The running validator is not touched.
3. The validator finishes its round, saves it, publishes it, checkpoints
   `validator_state.json`, sees the flag and exits with code 75.
4. With nothing running, the supervisor moves the checkout to the upstream
   commit, runs `pip install -r requirements.txt` **only if that file changed**
   (with the same interpreter, i.e. inside your venv), import-smokes the new
   code, and rolls back to the previous commit if either step fails.
5. It re-executes itself on the new code and starts the validator, which
   resumes from the ledger and the state checkpoint.

What updates never touch: the virtualenv, `.env`, wallets, `~/.sn125`,
`sn125/rounds/`, `sn125/audit/` and anything else untracked or ignored (there
is no `git clean`). Locally modified tracked files listed in
`SN125_UPDATE_KEEP` (default `sn125/config.py`) are re-applied on top of the
new commit; if the upstream change conflicts, your version is kept verbatim
and upstream's copy is written under `~/.sn125/autoupdate/backup/` for a manual
merge (the supervisor logs this at ERROR). Other locally modified tracked files
are backed up to the same place and then replaced.

Crashes: a validator that exits non-zero for any other reason is restarted
after a backoff (30 s, doubling to 10 min); a pending update is applied first,
so a broken release is replaced by its fix as soon as the fix is pushed. A
clean exit (0) stops the supervisor. Run the supervisor itself under systemd
(or any init) so a host reboot brings it back:

```ini
[Unit]
Description=Refinery validator (auto-updating)
After=network-online.target

[Service]
User=refinery
WorkingDirectory=/home/refinery/refinery
EnvironmentFile=/home/refinery/refinery/.env
ExecStart=/home/refinery/refinery/.venv/bin/python -m sn125 validate --auto-update \
  --wallet-name W --wallet-hotkey H --provider runpod --set-weights
Restart=always
RestartSec=30
KillSignal=SIGINT
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

### Urgent updates: skipping the round boundary

A normal push is applied at the next boundary, i.e. up to ~24 h later. To have
it applied **now**, make the push itself say so — either of:

- add or change the file `UPDATE_NOW` at the repository root (any content;
  e.g. `date > UPDATE_NOW`), or
- put `[update-now]` anywhere in a commit message of the push.

The marker is edge-triggered: `UPDATE_NOW` may stay in the tree, only a change
to it between the running commit and the pushed one counts, so bump its
contents each time. What happens on the validator host, within one
`--update-interval` (default 300 s) of the push:

1. The supervisor writes the restart flag with a `now` line.
2. The validator sees it at its next poll of the flag (every 60 s during the
   commit/reveal window waits) and abandons the round: the round's audit file
   gets a `round.aborted` event, the state checkpoint is written, and the
   process exits 75 — no publish, no weights, nothing on chain for that round.
3. If the validator is blocked inside the evaluation phase it cannot poll; the
   supervisor sends SIGINT after `SN125_UPDATE_NOW_GRACE_S` (default 120 s),
   which the validator handles the same way (in-flight cloud evaluations are
   abandoned, not waited for). SIGTERM follows 60 s later and SIGKILL 30 s after
   that as a last resort; the ledger and the previous boundary's checkpoint are
   already on disk, so even a kill loses no accounting.
4. The update is applied and the validator restarts as in the normal cycle. On
   start it returns every credit debited in the abandoned round (minus commits
   already burned by miner fault), then opens a fresh round in which those
   miners are collected again.

Cost of an urgent update: the GPU time of the abandoned evaluations and the
delay of one round for the miners in it. Use it for fixes that cannot wait;
for everything else a plain push at the boundary is free.

To force a restart at the next boundary without an upstream change, create
the flag yourself: `touch ~/.sn125/autoupdate/restart.requested`; for an
immediate one, `printf 'manual\nnow\n' > ~/.sn125/autoupdate/restart.requested`
(the validator exits 75 at its next 60 s poll and the supervisor restarts it
on whatever is checked out, applying any update that is available; if it is
in the evaluation phase, send the validator child SIGINT yourself after
writing the flag — the supervisor only escalates automatically for pushes
carrying the marker). To inspect
what the supervisor will do, `git -C refinery fetch && git -C refinery log
HEAD..origin/main --oneline`. Rollback is `git reset --hard <sha>` with the
validator stopped; the state files are forward- and backward-compatible
within this release line.

Why not Watchtower for this: Watchtower swaps the container the moment an
image appears, which is mid-round almost always (rounds are ~24 h), throwing
away the in-flight evaluations. Only the validator knows when a restart is
free, hence the flag protocol above. The same protocol works inside a
container whose checkout is a mounted volume, if you prefer a container as the
runtime shell.

## Container deployment

### Build and publish

Build from the production release directory, which includes Dockerfile and Compose:

```bash
docker build -t YOUR_REGISTRY/refinery-validator:RELEASE .
docker run --rm YOUR_REGISTRY/refinery-validator:RELEASE validate --help
docker push YOUR_REGISTRY/refinery-validator:RELEASE
```

Use an immutable release tag or digest until restart recovery is implemented.
The image is not yet built or published by this checkout. Container dependencies
have version ranges; capture a tested dependency lock and base-image digest before
promoting a release. The CPU image still needs a build/import smoke on a roomy host.

### Operator setup

Place runtime credentials in `validator.env` (chmod 600). Do not include treasury
private keys or unrelated account keys. Place Compose settings in `.env`:

```dotenv
REFINERY_IMAGE=YOUR_REGISTRY/refinery-validator:RELEASE
VALIDATOR_WALLET_NAME=refinery-validator
VALIDATOR_HOTKEY_NAME=YOUR_LOCAL_HOTKEY_NAME
VALIDATOR_WALLETS_DIR=/absolute/path/to/dedicated-validator-wallets
```

The wallet directory must exist and contain the authorized hotkey and required
public wallet metadata. It is mounted read-only. Encrypted hotkeys need a tested
noninteractive unlock mechanism; an unattended process cannot answer a prompt.
Run only one controller against this RunPod account.

Stage the existing private dataset into the shared data volume first:

```bash
docker compose run --rm --entrypoint /bin/sh validator -c \
  'HF_TOKEN="${SN125_FINEWEB_HF_READ_TOKEN:-$HF_WRITE_TOKEN}" exec python3 -m sn125 stage-shards --hf-repo "$SN125_FINEWEB_HF_REPO" --hf-repo-type dataset --expected-manifest "$SN125_FINEWEB_MANIFEST_HASH"'
docker compose up -d validator
docker compose logs -f validator
```

Default operation omits `--set-weights`, but still processes real payments and
rents GPUs. Add `--set-weights` to the validator command when enabling mainnet
weight transactions. Keep Compose project/directory names stable: changing them
selects new named volumes. Never use `docker compose down -v` on production data.

### Image updates

```bash
docker compose --profile updates up -d watchtower
```

This checks for image changes every five minutes and logs them. The optional
Watchtower service has Docker socket access, which gives it control of the host's
containers. The original containrrr project is archived; `WATCHTOWER_IMAGE` permits
a reviewed, compatible replacement pinned to a release/digest. Verify private
registry authentication and Docker API compatibility before enabling the service.

Round-boundary restart recovery (deferred submissions, round counter, weights)
and the durable ledger are implemented; what an image swap still cannot do is
pick the moment. Before enabling Watchtower's automatic replacement you would
need the container to honour the same restart-request protocol as the git
supervisor (exit at the boundary, then be replaced) and a tested publisher
pipeline promoting images after the offline suite passes. Until then keep it
monitor-only, or run the git checkout shape above.

References: https://github.com/containrrr/watchtower and
https://docs.docker.com/reference/compose-file/services/
