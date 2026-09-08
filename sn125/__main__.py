"""Production command-line interface for Refinery subnet 125."""
import argparse, logging, os, sys
from pathlib import Path

def _check_no_duplicate(label: str):
    """Refuse to start if another sn125 validate is already running."""
    import subprocess
    my_pid = os.getpid()
    exclude = {my_pid}
    p = my_pid
    for _ in range(5):
        try:
            with open(f"/proc/{p}/stat") as f:
                pp = int(f.read().split()[3])
        except (OSError, ValueError, IndexError):
            break
        if pp <= 1:
            break
        exclude.add(pp)
        p = pp
    try:
        out = subprocess.check_output(['pgrep', '-f', 'python.*-m sn125 validate'], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return
    for line in out.strip().split('\n'):
        pid = int(line.strip())
        if pid not in exclude:
            print(f"ERROR: refusing to spawn {label}: PID {pid} already running "
                  f"(`pgrep -f 'python.*-m sn125'`). Kill it first.", file=sys.stderr)
            sys.exit(1)

def _raise_nofile_limit() -> None:
    """Shard-backed tasks mmap ~1k files; container images ship a 1024 soft
    nofile limit. Lift the soft limit toward the hard one (capped at 65536)
    for this process and everything it spawns (the worker raises its own)."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = min(hard if hard != resource.RLIM_INFINITY else 1 << 20, 1 << 16)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception:
        pass

def main():
    _raise_nofile_limit()
    parser = argparse.ArgumentParser(prog='sn125', description='SN125 Optimizer Discovery Subnet')
    sub = parser.add_subparsers(dest='command')
    p_mine = sub.add_parser('mine', help='Serve optimizer via Axon')
    p_mine.add_argument('--wallet-name', default='default')
    p_mine.add_argument('--wallet-hotkey', default='default')
    p_mine.add_argument('--port', type=int, default=8091)
    p_mine.add_argument('--optimizer', help='Path to optimizer .py file')
    p_val = sub.add_parser('validate', help='Run the production FSM validator loop')
    p_val.add_argument('--wallet-name', default='default')
    p_val.add_argument('--wallet-hotkey', default='default')
    p_val.add_argument('--netuid', type=int, default=125, help='Fixed at 125 (sn125.config.NETUID); any other value is refused')
    p_val.add_argument('--network', default='finney')
    p_val.add_argument('--set-weights', action='store_true', help='Actually set weights on chain (default: read-only logging)')
    p_val.add_argument('--timeout', type=int, default=86400, help='Per-submission timeout in seconds (default: 86400 = 24h)')
    p_val.add_argument('--resource', default='b200-small', help='Eval SKU (default: b200-small = 1× B200; the lium provider maps it to the equivalent machine)')
    p_val.add_argument('--provider', '--providers', dest='provider', default='', help="Ordered comma-separated provider failover chain for B200 rentals (e.g. 'lambda,targon': rent from the first provider with capacity, fail over down the chain, pause only when all are dry). Default: SN125_CLOUD_PROVIDERS env, legacy SN125_CLOUD_PROVIDER, then 'runpod,lambda'. 'aws' (standardized p6-b200.48xlarge, opt-in — see cloud_aws.py) is trusted; 'lium' is refused for scoring (untrusted marketplace hosts).")
    p_val.add_argument('--burn-fraction-floor', type=float, default=None, help='Override sn125.config.LAUNCH_BURN_FRACTION_FLOOR. Default uses config.py; with no frontier winner, emission still burns 100%%.')
    p_val.add_argument('--audit-dir', default='', help="Directory for per-round validator JSONL audit logs (default: sibling 'audit' directory next to rounds).")
    p_val.add_argument('--treasury-coldkey', default=None, help='SS58 of the treasury coldkey miners pay the round fee to (defaults to the configured public treasury address).')
    p_val.add_argument('--round-fee-tao', type=float, default=None, help='Per-round evaluation fee in TAO. If omitted, calculate from the estimated full-run cost plus 10%% margin.')
    p_val.add_argument('--tao-usd-price', type=float, default=None, help='TAO/USD reference for automatic fee calculation (default: sn125/tao_price.json, currently $250).')
    p_val.add_argument('--finality-depth', type=int, default=10, help='Blocks behind chain head treated as final before crediting a treasury transfer (default: 10).')
    p_val.add_argument('--payment-ledger', default=None, help='Durable credit-ledger file (credits, debits, processed transfers, scan cursor). Default: SN125_PAYMENT_LEDGER env, else <audit-dir>/payments/ledger.json.')
    p_val.add_argument('--payment-scan-interval', type=float, default=None, help='Seconds between background treasury scans (default: SN125_PAYMENT_SCAN_S env, else 60). 0 disables the continuous watcher (round-start sync only).')
    p_val.add_argument('--payment-backfill-hours', type=float, default=72.0, help='On a ledger with no scan cursor, credit treasury deposits from this many hours before start-up (default: 72; 0 = current prune window only). Older blocks are read from the archive endpoint.')
    p_val.add_argument('--payment-rescan', action='store_true', help='Force the backfill window even when the ledger already has a scan cursor. Safe: processed transfers are never credited twice.')
    p_val.add_argument('--auto-update', action='store_true', help='Run under the auto-update supervisor (sn125/autoupdate.py): keep this git checkout current with its upstream and restart the validator only at round boundaries, resuming from the durable ledger and state checkpoint. All other flags are passed through unchanged. Requires running from a git clone.')
    p_val.add_argument('--update-interval', type=float, default=None, help='Seconds between upstream checks with --auto-update (default: SN125_UPDATE_INTERVAL_S env, else 300).')
    p_val.add_argument('--payment-archive-network', default='archive', help="bittensor network name or wss:// endpoint of an ARCHIVE node used for blocks the primary node has pruned (default: 'archive' = wss://archive.chain.opentensor.ai). '' disables archive reads (old deposits are then lost).")
    p_vf = sub.add_parser('validate-file', help='Check optimizer file passes sandbox')
    p_vf.add_argument('file', help='Path to optimizer .py file')
    p_bp = sub.add_parser('box-probe', help='Standardized GPU throughput probe (the per-box speed gate; prints one parseable result line)')
    p_bp.add_argument('--warmup', type=int, default=40)
    p_bp.add_argument('--steps', type=int, default=120)
    p_prod_eval = sub.add_parser('prod-eval', help='Production FineWeb eval for one optimizer')
    p_prod_eval.add_argument('file', help='Path to optimizer .py file')
    p_prod_eval.add_argument('--data-dir', default=None, help='Production FineWeb shard directory (default: SN125_FINEWEB_DIR or repo data)')
    p_prod_eval.add_argument('--total-steps', type=int, default=0, help='Override production schedule horizon; default fineweb.PROD_CONFIRM_STEPS')
    p_prod_eval.add_argument('--budget', type=float, default=0.0, help='Override production compute_budget_seconds for smoke runs')
    p_prod_eval.add_argument('--timeout', type=float, default=0.0, help='Sandbox timeout; default budget*1.15')
    p_prod_eval.add_argument('--seed', type=int, default=42)
    p_prod_eval.add_argument('--no-amp', action='store_true')
    p_prod_eval.add_argument('--baseline-json', default='', help='JSON baseline bundle; when set, only the submission is trained')
    p_prod_eval.add_argument('--expected-manifest', default='', help='Require production shards to match this manifest hash')
    p_prod_eval.add_argument('--checkpoint-out', default='', help='Optional safetensors path for final plus q25/q50/q75 checkpoints')
    p_prod_eval.add_argument('--worker-log', default='', help='Optional JSONL sink for live sandbox worker stdout/stderr progress')
    p_prod_eval.add_argument('--out', default='', help='Optional JSON output path')
    p_stage = sub.add_parser('stage-shards', help='Stage/verify production FineWeb shards')
    p_stage.add_argument('--data-dir', default='')
    p_stage.add_argument('--expected-manifest', default='')
    p_stage.add_argument('--hf-repo', default='')
    p_stage.add_argument('--hf-repo-type', default='dataset')
    p_stage.add_argument('--hf-revision', default='')
    p_stage.add_argument('--hf-subdir', default='')
    p_stage.add_argument('--no-verify-hashes', action='store_true')
    p_stage.add_argument('--allow-build', action='store_true')
    p_lb = sub.add_parser('leaderboard', help='Show aggregated leaderboard')
    p_lb.add_argument('--rounds-dir', default=None)
    p_av = sub.add_parser('attest-validate', help='Lightweight validator: trust attestations from R2')
    p_av.add_argument('--wallet-name', default='default')
    p_av.add_argument('--wallet-hotkey', default='default')
    p_av.add_argument('--netuid', type=int, default=125, help='Fixed at 125 (sn125.config.NETUID); any other value is refused')
    p_av.add_argument('--network', default='finney')
    p_av.add_argument('--r2-endpoint', default=None, help='R2 endpoint URL (or R2_ENDPOINT env)')
    p_av.add_argument('--r2-bucket', default=None, help='R2 bucket name (or R2_BUCKET env)')
    p_av.add_argument('--trusted-hotkey', default=None, help='Core validator hotkey to trust (or TRUSTED_VALIDATOR_HOTKEY env)')
    p_av.add_argument('--poll-interval', type=int, default=300, help='Seconds between R2 polls (default: 300)')
    p_av.add_argument('--set-weights', action='store_true', help='Actually set weights on chain')
    p_ex = sub.add_parser('export', help='Export top optimizer source code')
    p_ex.add_argument('--rank', type=int, default=1, help='Leaderboard rank to export (default: 1)')
    p_ex.add_argument('--output', '-o', help='Write to file instead of stdout')
    p_ex.add_argument('--rounds-dir', default=None)
    p_cal = sub.add_parser('calibrate', help='§3 throughput probe: pin step_time/MFU and compute N')
    p_cal.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p_cal.add_argument('--seq', type=int, default=2048)
    p_cal.add_argument('--batch', type=int, default=16)
    p_cal.add_argument('--steps', type=int, default=200, help='probe steps to time')
    p_cal.add_argument('--warmup', type=int, default=100, help='WSD warmup steps')
    p_cal.add_argument('--drop-warmup', type=int, default=20, help='leading steps dropped from median')
    p_cal.add_argument('--no-amp', action='store_true')
    p_cal.add_argument('--target-hours', type=float, default=20.0)
    p_cal.add_argument('--backend', choices=['local', 'targon', 'lium', 'lambda', 'aws'], default='local')
    p_cal.add_argument('--resource', default='b200-small')
    p_cal.add_argument('--timeout', type=int, default=1800)
    p_cal.add_argument('--out', default='', help='write result JSON to this path')
    p_cal.add_argument('--spend-ledger', default='', help='append spend record to this JSON ledger (targon backend)')
    p_cal.add_argument('--empty-cache-every', type=int, default=0, help='per-step torch.cuda.empty_cache() cadence (0=off/boundary-only; the default)')
    p_cal.add_argument('--profile', action='store_true', help='emit PROFILE_BREAKDOWN (forward/backward/grad_clone/empty_cache/opt_step per-step seconds)')
    p_cal.add_argument('--flash', action='store_true', help='enable flash/mem-efficient SDPA (faster, NON-deterministic backward — breaks audit replay)')
    p_cal.add_argument('--nondet', action='store_true', help='allow non-deterministic algorithms (pairs with --flash for the efficient A/B arm)')
    p_cal.add_argument('--compile', dest='compile_model', action='store_true', help='DIAG: torch.compile the model before timing')
    p_cal.add_argument('--chunked-ce', dest='chunked_ce', type=int, default=0, help='DIAG: chunk HF cross-entropy into N tiles; lean models use their own fused loss')
    p_cal.add_argument('--fp8', action='store_true', help='DIAG: torchao Float8 Linear training path; major model-kernel change')
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    if args.command == 'mine':
        from .neuron import Miner
        import bittensor as bt
        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        Miner(wallet, port=args.port, optimizer_path=args.optimizer).run()
    elif args.command == 'validate':
        _check_no_duplicate('validate')
        if args.auto_update:
            from .autoupdate import supervise
            sys.exit(supervise(sys.argv[1:], interval_s=args.update_interval))
        from . import config
        treasury_coldkey = args.treasury_coldkey or config.TREASURY_COLDKEY
        if not treasury_coldkey:
            parser.error('validate requires --treasury-coldkey')
        import math
        for label, value in (('--round-fee-tao', args.round_fee_tao), ('--tao-usd-price', args.tao_usd_price), ('--timeout', args.timeout)):
            if value is not None and (not math.isfinite(value) or value <= 0):
                parser.error(f"{label} must be finite and positive")
        if args.finality_depth < 1:
            parser.error('--finality-depth must be positive')
        if args.payment_backfill_hours < 0 or not math.isfinite(args.payment_backfill_hours):
            parser.error('--payment-backfill-hours must be >= 0')
        if args.payment_scan_interval is not None and (not math.isfinite(args.payment_scan_interval) or args.payment_scan_interval < 0):
            parser.error('--payment-scan-interval must be >= 0')
        from . import settings
        from .cloud import ensure_scoring_providers
        from .neuron import CommitHash, GetSubmission, Validator
        from .fineweb import production_tasks, PROD_CONFIRM_STEPS
        from .payments.registry import RAO_PER_TAO, PaymentRegistry
        from .payments.chain import BittensorChainView
        from .roundsm.live import run_fsm_validator
        import bittensor as bt
        providers = [p.strip().lower() for p in args.provider.split(',') if p.strip()] if args.provider else settings.cloud_provider_chain()
        try:
            providers = ensure_scoring_providers(providers)
        except ValueError as e:
            parser.error(str(e))
        try:
            tasks = production_tasks(total_steps=PROD_CONFIRM_STEPS)
        except FileNotFoundError as e:
            parser.error(f"production shards required for validate: {e}")
        print(f"[validate] production task → {tasks[0].task_id} "
              f"seq{tasks[0].sequence_length} manifest {tasks[0].data_manifest[:16]}")
        print(f"[validate] provider failover chain → {','.join(providers)} "
              f"(B200-only; round pauses when every provider is dry)")
        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        validator = Validator(wallet=wallet, netuid=args.netuid, network=args.network, set_weights=args.set_weights, tasks=tasks, num_trials=1, submission_timeout=args.timeout, mode='prod', backend=','.join(providers), cloud_resource=args.resource, audit_dir=args.audit_dir, burn_fraction_floor=args.burn_fraction_floor)
        if args.round_fee_tao is None:
            from .cloud import SKU_COSTS, SUBMISSION_MARGIN
            from .pricing import fee_tao_for_cost, tao_usd_price
            if validator._cloud_orch is not None:
                hourly = validator._cloud_orch._cost_per_hour(args.resource)
            else:
                hourly = SKU_COSTS.get(args.resource, 20.0)
            estimated_cost = (float(args.timeout) + SUBMISSION_MARGIN) / 3600.0 * float(hourly)
            price = tao_usd_price() if args.tao_usd_price is None else args.tao_usd_price
            try:
                args.round_fee_tao = fee_tao_for_cost(estimated_cost, price_usd=price)
            except ValueError as e:
                parser.error(str(e))
            print(f"[validate] automatic fee → {args.round_fee_tao:.3f} TAO "
                  f"(estimated ${estimated_cost:.2f} run × 1.10 ÷ ${price:.2f}/TAO)")
        if args.round_fee_tao <= 0:
            parser.error('validate requires --round-fee-tao > 0')
        from .payments.chain import blocks_for_hours
        from .payments.store import resolve_ledger_path
        owner_subtensor = validator.subtensor or bt.Subtensor(network=args.network)
        subtensor = bt.Subtensor(network=args.network)
        archive_factory = None
        archive_network = (args.payment_archive_network or '').strip()
        if archive_network:

            def archive_factory():
                return bt.Subtensor(network=archive_network)
        head = int(subtensor.get_current_block())
        start_block = None
        if args.payment_backfill_hours > 0:
            start_block = max(0, head - args.finality_depth - blocks_for_hours(args.payment_backfill_hours))
        view = BittensorChainView.from_subtensor(subtensor, treasury_coldkey, finality_depth=args.finality_depth, archive_subtensor_factory=archive_factory, start_block=start_block, owner_subtensor=owner_subtensor)
        ledger_path = resolve_ledger_path(args.payment_ledger, audit_dir=validator.audit_dir, rounds_dir=validator.rounds_dir)
        registry = PaymentRegistry(view, store_path=ledger_path)
        if args.payment_rescan and start_block is not None:
            registry.rewind_scan(start_block)
        print(f"[validate] payment ledger → {ledger_path} "
              f"({len(registry.events)} event(s), {len(registry.balances())} funded coldkey(s)); "
              f"scan resumes after block {view.scan_cursor()} of finalized "
              f"{head - args.finality_depth}" + (f"; archive endpoint {archive_network}" if archive_network else '; NO archive endpoint (pruned blocks are skipped)'))
        validator.payment_registry = registry
        if args.payment_scan_interval is not None:
            validator.payment_scan_s = args.payment_scan_interval
        validator.treasury_coldkey = treasury_coldkey
        validator.round_fee_rao = int(round(args.round_fee_tao * RAO_PER_TAO))
        validator.cloud_resource = args.resource
        validator.commit_synapse_cls = CommitHash
        validator.submission_synapse_cls = GetSubmission
        if run_fsm_validator(validator) == 'restart':
            from .roundsm.live import EXIT_RESTART
            sys.exit(EXIT_RESTART)
    elif args.command == 'calibrate':
        import json as _json
        from .cloud import CLOUD_PROVIDERS
        if args.backend in CLOUD_PROVIDERS:
            from .cloud import TargonOrchestrator, make_cloud_client
            client = make_cloud_client(args.backend, args.timeout)
            orch = TargonOrchestrator(client=client, resource=args.resource, timeout=args.timeout)
            orch.initialize()
            print(f"Calibration probe on {args.backend} ({args.resource}): {args.model} "
                  f"seq={args.seq} batch={args.batch} steps={args.steps}...")
            result = orch.calibrate_remote(model=args.model, seq=args.seq, batch=args.batch, steps=args.steps, warmup=args.warmup, target_hours=args.target_hours, timeout=args.timeout, spend_ledger=args.spend_ledger, empty_cache_every=args.empty_cache_every, profile=args.profile, flash=args.flash, nondet=args.nondet, compile_model=args.compile_model, chunked_ce=args.chunked_ce, fp8=args.fp8)
            status = orch.get_status()
            print(f"\nCloud cost: ${status['daily_spend']:.2f} today")
        else:
            if args.profile:
                os.environ['SN125_PROFILE'] = '1'
            if args.flash:
                os.environ['SN125_DIAG_FLASH'] = '1'
            if args.nondet:
                os.environ['SN125_DIAG_NONDET'] = '1'
            from .calibrate import run_probe
            result = run_probe(args.model, args.seq, args.batch, args.steps, warmup_steps=args.warmup, drop_warmup=args.drop_warmup, use_amp=not args.no_amp, target_hours=args.target_hours, empty_cache_every=args.empty_cache_every, compile_model=args.compile_model, chunked_ce=args.chunked_ce, fp8=args.fp8)
        if result.get('failed'):
            print(f"FAILED: {result.get('error', 'unknown')}")
            sys.exit(1)
        if args.backend == 'local':
            print('CALIBRATION_RESULT ' + _json.dumps(result), flush=True)
        if args.out:
            with open(args.out, 'w') as f:
                _json.dump(result, f, indent=2)
        print('\n=== Calibration ===')
        print(_json.dumps(result, indent=2))
        print(f"\n→ N={result.get('computed_N')} step_time={result.get('median_step_time_s')}s "
              f"tokens={result.get('projected_tokens')} chinchilla_ratio={result.get('chinchilla_ratio')} "
              f"mfu={result.get('mfu')}")
    elif args.command == 'box-probe':
        from .boxprobe import format_result, run_box_probe
        result = run_box_probe(warmup_steps=args.warmup, measure_steps=args.steps)
        print(format_result(result), flush=True)
    elif args.command == 'validate-file':
        from .sandbox import validate_source, load_optimizer_sandboxed, SandboxViolation
        from .references import extract_hparams
        src = Path(args.file).read_text()
        print(f"Source: {len(src)} bytes ({len(src.encode())} encoded)")
        violations = validate_source(src)
        if violations:
            print(f"FAILED — {len(violations)} violation(s):")
            for v in violations:
                print(f"  ✗ {v}")
            sys.exit(1)
        print('AST validation: PASS')
        try:
            cls = load_optimizer_sandboxed(src)
            print('Sandbox load:   PASS')
        except SandboxViolation as e:
            print(f"Sandbox load:   FAIL — {e}")
            sys.exit(1)
        hp = extract_hparams(src)
        hp_sc = extract_hparams(src, {'use_pretrained': False})
        print(f"HPARAMS:        lr={hp['lr']:.1e} wd={hp.get('weight_decay',0.01):.1e}")
        print(f"  scratch:      lr={hp_sc['lr']:.1e} wd={hp_sc.get('weight_decay',0.01):.1e}")
        import torch
        pg = [{'params': [('w', (32, 32), torch.float32)], 'lr': hp['lr'], 'weight_decay': hp.get('weight_decay', 0.01)}]
        cfg = {'total_steps': 100, 'warmup_steps': 5, 'decay_fraction': 0.2, 'max_grad_norm': 1.0}
        try:
            opt = cls(pg, cfg)
            g = {'w': torch.randn(32, 32, device='cuda')}
            p = {'w': torch.randn(32, 32, device='cuda')}
            u = opt.step(g, p, 0)
            assert 'w' in u and u['w'].shape == (32, 32), 'Bad update shape'
            print('Functional test: PASS (init + step OK)')
        except Exception as e:
            print(f"Functional test: FAIL — {e}")
            sys.exit(1)
        print('\nReady to submit.')
    elif args.command == 'prod-eval':
        if args.total_steps <= 0:
            from .fineweb import PROD_CONFIRM_STEPS
            args.total_steps = PROD_CONFIRM_STEPS
        from .prod_eval import main as _prod_eval_main
        sys.exit(_prod_eval_main([args.file, *(['--data-dir', args.data_dir] if args.data_dir else []), '--total-steps', str(args.total_steps), '--budget', str(args.budget), '--timeout', str(args.timeout), '--seed', str(args.seed), *(['--no-amp'] if args.no_amp else []), *(['--baseline-json', args.baseline_json] if args.baseline_json else []), *(['--expected-manifest', args.expected_manifest] if args.expected_manifest else []), *(['--checkpoint-out', args.checkpoint_out] if args.checkpoint_out else []), *(['--worker-log', args.worker_log] if args.worker_log else []), *(['--out', args.out] if args.out else [])]))
    elif args.command == 'stage-shards':
        from .stage_shards import main as _stage_main
        sys.exit(_stage_main([*(['--data-dir', args.data_dir] if args.data_dir else []), *(['--expected-manifest', args.expected_manifest] if args.expected_manifest else []), *(['--hf-repo', args.hf_repo] if args.hf_repo else []), *(['--hf-repo-type', args.hf_repo_type] if args.hf_repo_type else []), *(['--hf-revision', args.hf_revision] if args.hf_revision else []), *(['--hf-subdir', args.hf_subdir] if args.hf_subdir else []), *(['--no-verify-hashes'] if args.no_verify_hashes else []), *(['--allow-build'] if args.allow_build else [])]))
    elif args.command == 'leaderboard':
        from .neuron import print_leaderboard
        print_leaderboard(args.rounds_dir)
    elif args.command == 'attest-validate':
        from .neuron import AttestationValidator
        import bittensor as bt
        if not args.set_weights:
            print('WARNING: --set-weights not set. Will verify attestations but not set weights.')
            print('Add --set-weights to actually set weights on chain.\n')
        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        AttestationValidator(wallet=wallet, netuid=args.netuid, network=args.network, r2_endpoint=args.r2_endpoint, r2_bucket=args.r2_bucket, trusted_hotkey=args.trusted_hotkey, poll_interval=args.poll_interval, set_weights=args.set_weights).run()
    elif args.command == 'export':
        from .neuron import export_optimizer
        export_optimizer(rank=args.rank, output=args.output, rounds_dir=args.rounds_dir)
if __name__ == '__main__':
    main()
