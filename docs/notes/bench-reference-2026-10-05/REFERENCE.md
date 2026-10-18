# Bench suite `easy_v0` — reference evaluations (2026-10-05)

> This repository is the production code release. Paths under `scripts/` and `data/`,
> and result files under `docs/notes/`, refer to the operator's working tree (lab
> tooling, the shard directory, raw measurements) and are not included here.

Reference points for the Refiner task's placeholder scoring suite
(`docs/REFINER_TASK.md` §6), so a submission score can be read against known
models rather than against chance alone. All runs: local A100-80GB, `seq_len`
2048 (1024 for gpt2, its context limit), 8 rows per batch. The suite is 1000
items each of hellaswag / piqa / arc_easy / sciq / lambada and 500 of
openbookqa; the SmolLM2-tokenized production file is
`sha256 ad25712aa737339d935eeba65d6200a7314424adc304f356c43eff7053d27a44`.

MC cells are `acc_norm / prob_correct_norm`; the suite score is the mean
`prob_correct_norm` over the five MC tasks. Chance on the MC tasks is ≈ 0.30
(PIQA is binary, the rest 4-way), which is where the random init sits.
LAMBADA is `acc / bits-per-byte`; both are tokenizer-independent (token-level
perplexity is not, so it is omitted here).

## A. SmolLM2 tokenizer — scored on the production suite, in the production regime

Loaded into the lean twin of the architecture (`bench-score --base`), bf16
autocast exactly as student checkpoints are scored.

| model | params | tokens seen | hellaswag | piqa | arc_easy | sciq | openbookqa | lambada | suite score |
|---|---|---|---|---|---|---|---|---|---|
| lean-llama-360m, random init (seed 0) | 362 M | 0 | 0.267 / 0.265 | 0.508 / 0.502 | 0.266 / 0.263 | 0.227 / 0.234 | 0.272 / 0.275 | 0.000 / 3.383 | **0.3078** |
| SmolLM-135M (v1) | 135 M | 600 B | 0.425 / 0.418 | 0.685 / 0.670 | 0.573 / 0.534 | 0.785 / 0.746 | 0.326 / 0.319 | 0.384 / 0.762 | **0.5372** |
| SmolLM2-135M | 135 M | 2 T | 0.431 / 0.431 | 0.682 / 0.672 | 0.582 / 0.556 | 0.810 / 0.766 | 0.334 / 0.327 | 0.435 / 0.640 | **0.5506** |
| SmolLM-360M (v1) | 362 M | 600 B | 0.535 / 0.536 | 0.719 / 0.706 | 0.649 / 0.612 | 0.865 / 0.831 | 0.362 / 0.362 | 0.465 / 0.624 | **0.6094** |
| SmolLM2-360M | 362 M | 4 T | 0.545 / 0.548 | 0.721 / 0.711 | 0.676 / 0.636 | 0.876 / 0.845 | 0.364 / 0.358 | 0.539 / 0.499 | **0.6197** |
| SmolLM2-1.7B | 1.7 B | 11 T | 0.712 / 0.710 | 0.781 / 0.769 | 0.745 / 0.701 | 0.908 / 0.891 | 0.432 / 0.417 | 0.677 / 0.338 | **0.6976** |

## B. Other tokenizers — same text items, re-tokenized per model, fp32

Each model is scored on a suite rebuilt from the **same text items** with its
own tokenizer (`bench-build --tokenizer <name>`); all of those suites, and a
rebuild of the production suite, share `text_sha256
bf027e2d142d0adb25d0dec331569c02b3f237d83232b12cc53a57c721287600` (the
production file predates the field; its token items were verified identical to
the rebuild). Loaded through the native `transformers` class (`bench-score
--hf`) in fp32 — see the Pythia note below for why not bf16.

| model | params | tokens seen | hellaswag | piqa | arc_easy | sciq | openbookqa | lambada | suite score |
|---|---|---|---|---|---|---|---|---|---|
| Pythia-160m @ step 6000 | 162 M | 12.6 B | 0.281 / 0.274 | 0.606 / 0.594 | 0.383 / 0.370 | 0.678 / 0.642 | 0.270 / 0.257 | 0.323 / 0.870 | **0.4276** |
| Pythia-410m @ step 6000 | 405 M | 12.6 B | 0.304 / 0.302 | 0.600 / 0.591 | 0.404 / 0.393 | 0.686 / 0.655 | 0.262 / 0.263 | 0.371 / 0.752 | **0.4408** |
| gpt2 | 124 M | ~10 B (WebText) | 0.305 / 0.310 | 0.626 / 0.614 | 0.414 / 0.399 | 0.682 / 0.662 | 0.266 / 0.263 | 0.322 / 0.802 | **0.4495** |
| Pythia-160m (final) | 162 M | 300 B | 0.305 / 0.303 | 0.622 / 0.607 | 0.403 / 0.391 | 0.714 / 0.680 | 0.270 / 0.265 | 0.353 / 0.809 | **0.4493** |
| Pythia-410m (final) | 405 M | 300 B | 0.403 / 0.397 | 0.673 / 0.665 | 0.481 / 0.452 | 0.769 / 0.732 | 0.292 / 0.285 | 0.510 / 0.535 | **0.5063** |
| Qwen2.5-0.5B | 494 M | 18 T | 0.510 / 0.512 | 0.690 / 0.682 | 0.633 / 0.591 | 0.915 / 0.892 | 0.334 / 0.329 | 0.534 / 0.533 | **0.6015** |
| Qwen3-0.6B-Base | 596 M | 36 T | 0.533 / 0.532 | 0.698 / 0.686 | 0.673 / 0.629 | 0.919 / 0.892 | 0.326 / 0.333 | 0.541 / 0.512 | **0.6144** |

Pythia step 6000 = 6000 × 2,097,152 tokens ≈ 12.6 B — the closest public
analogue of a full-budget Challenge-1 run (20 h, ≈ 12 B FineWeb-Edu tokens,
362 M params), on the Pile rather than FineWeb-Edu.

## What the numbers say

- **Where the refiner task lives.** At ≈ 12 B tokens the public references
  score 0.43–0.45 (Pythia-160m/410m @ step 6000, gpt2): ≈ 0.13 above init
  and ≈ 0.10 below SmolLM2-135M. A full-budget Challenge-1 student should land
  in or somewhat above that band (FineWeb-Edu is far better matched to these
  benchmarks than the Pile or WebText), and refiner-vs-identity differences
  will be a fraction of it — the bar/noise calibration listed in §6 of the
  task spec is what turns that fraction into a payable signal.
- **Data, not size, is the lever at this budget.** Pythia-410m and -160m at
  the same 12.6 B tokens differ by 0.013; the same 410m model after 300 B
  tokens gains 0.065. SmolLM-135M (600 B curated tokens) beats Pythia-410m
  (300 B Pile tokens) by 0.03 with 3× fewer parameters. What the student
  reads matters more than how big it is, which is the premise of the task.
- **Family ladder** (same tokenizer, same forward pass): 135M → 360M is
  worth ≈ 0.07 at 2–4 T tokens; v1 → v2 data (600 B → 2–4 T tokens) ≈ 0.01;
  360M → 1.7B ≈ 0.08. The 1.7B model's HellaSwag 0.712 / PIQA 0.781 / OBQA
  0.432 bracket its card (68.7 / 77.6 / 42.1) within the 1000-item subsample.
- **Lean model is a faithful twin.** SmolLM-135M/360M (v1: RoPE θ 1e4) and
  SmolLM2-1.7B (MHA, θ 1.3e5) loaded via LeanConfig overrides reproduce
  their published cards (SmolLM-135M HellaSwag 41.2 / PIQA 68.4; SmolLM-360M
  51.8 / 71.6) as SmolLM2-360M did earlier; bf16 vs fp32 on SmolLM-135M
  moves every task by ≤ 0.002.
- **Pythia is fragile in bf16; our students are not.** Pythia-160m scored
  in bf16 autocast loses 20 points of LAMBADA (0.122 vs 0.353) and 0.02 of
  suite score (`pythia-160m-bf16.json`); Pythia-410m loses 0.025 LAMBADA;
  gpt2, Qwen and the SmolLM twins move ≤ 0.005. GPT-NeoX's parallel residual
  grows activation outliers that bf16 cannot resolve. The Llama-shaped lean
  student does not have this property, so scoring checkpoints in bf16 (the
  production regime) is sound — but any future student architecture should
  be checked the same way before its scores are trusted.
- **Sanity against published lm-eval-harness numbers** (acc_norm): gpt2
  HellaSwag 31.1 / PIQA 62.9 / ARC-e 43.8 / LAMBADA 32.6 → here 30.5 / 62.6
  / 41.4 / 32.2; Pythia-410m (paper) LAMBADA 51.6 / PIQA 66.9 / ARC-e 52.0 /
  SciQ 81.6 → here 51.0 / 67.3 / 48.1 / 76.9. ARC-easy and SciQ run a few
  points under lm-eval because the pilot jsonl prompt format differs
  slightly; the suite is internally consistent, which is all the task needs.
- **Still missing**: a full-budget 20 h Challenge-1 checkpoint. Score one
  from R2 on a box with credentials
  (`sn125 bench-score checkpoints/<round>/<uid>/<task>/q100.safetensors`).

## How these were produced

Nothing was downloaded onto the (99 %-full) disk: weights were staged with
`huggingface_hub.snapshot_download(allow_patterns=["*.safetensors", "*.json",
"*.txt", "tokenizer*", ...])` — no `.bin`/`.pt`/`.msgpack`/`.h5` — into
`HF_HUB_CACHE=/dev/shm/hf_ref/hub` (tmpfs), scored offline from there, and
deleted afterwards. Every weight file passed `validate_safetensors_file` on
its raw bytes before any framework loader touched it; `--hf` additionally
refuses configs with `auto_map` (remote code) and loads with
`use_safetensors=True, trust_remote_code=False, local_files_only=True`.

```bash
export HF_HUB_CACHE=/dev/shm/hf_ref/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# A. lean twin, production suite, bf16 (default)
python -m sn125 bench-score --base HuggingFaceTB/SmolLM-135M   --out smollm-135m.json
python -m sn125 bench-score --base HuggingFaceTB/SmolLM-360M   --out smollm-360m.json
python -m sn125 bench-score --base HuggingFaceTB/SmolLM2-1.7B  --out smollm2-1.7b.json

# B. per-tokenizer suites from the same text items (never the default suite path), then fp32 scoring
python -m sn125 bench-build --jsonl-dir data/pilot_encdec_v0/bench --tokenizer gpt2 --out /dev/shm/suite_gpt2.json
python -m sn125 bench-score --hf gpt2 --suite /dev/shm/suite_gpt2.json --seq-len 1024 --no-amp --out gpt2.json
python -m sn125 bench-build --jsonl-dir data/pilot_encdec_v0/bench --tokenizer EleutherAI/pythia-410m --out /dev/shm/suite_p410.json
python -m sn125 bench-score --hf EleutherAI/pythia-410m --revision step6000 --suite /dev/shm/suite_p410.json --no-amp --out pythia-410m-step6000.json
python -m sn125 bench-score --hf EleutherAI/pythia-410m                    --suite /dev/shm/suite_p410.json --no-amp --out pythia-410m.json
# likewise EleutherAI/pythia-160m (main, step6000), Qwen/Qwen2.5-0.5B, Qwen/Qwen3-0.6B-Base

# random init: export a seed-0 lean-llama-360m with engine.score_worker.export_checkpoint,
# then score it through the ordinary (validated) checkpoint path
python -m sn125 bench-score lean360m_init_seed0.safetensors --out lean-llama-360m-init-seed0.json
```

Provenance (sha256 of the scored `model.safetensors`; HF snapshot in brackets):

- `smollm2-360m.json`: `7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f` [f8027fd0]
- `smollm2-135m.json`: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` [93efa2f0]
- `smollm-135m.json`: `c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093` [1d461723]
- `smollm-360m.json`: `e91f05d8506ee5efbd8c0fbfc1799c49af2b2f2cce824bc2d801d5af2a716cc2` [59f7ef24]
- `smollm2-1.7b.json`: `1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0` [effd688a]
- `gpt2.json`: `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` [607a30d7]
- `pythia-160m.json` (+ `-bf16`): `29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270` [50f5173d]
- `pythia-160m-step6000.json`: `af817c48650ab2dc9e123444696812f6eeb07c93cd8ef6290a6a82f9279fa90f` [da2dcd59]
- `pythia-410m.json`: `3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b` [9879c9b5]
- `pythia-410m-step6000.json`: `482d18a109d050496f9a0a6be00fec088fb04042259ab2e29ff014251130296f` [037b5fec]
- `qwen2.5-0.5b.json`: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` [060db649]
- `qwen3-0.6b-base.json`: `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` [da87bfb6]
- `lean-llama-360m-init-seed0.json`: `91a18fff1ccf0c35f588b064eff8f8134a0c5d90dab06782133075cf45dbe982`

Each result JSON also carries the suite hash, `text_sha256`, tokenizer,
architecture class / lean config, parameter count and revision.
