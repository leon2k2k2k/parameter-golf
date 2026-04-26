# Spec 042A — slope anneal 0.707→0.5 at warmdown

**Slug:** `042A-slope-anneal-0707-to-05-screen`
**Created:** 2026-04-26
**Status:** READY (recompile bug fixed)
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `aff2de4`
**Links to:** `research/ideas/slope-annealing-sqrt-to-half.md`
**Predecessor smoke:** `runs/042A-recompile-smoke/notes.md`

## Recompile fix in this commit

The original 042A run (commit `2593982`) lost ~7-8 minutes mid-training to a
Dynamo recompile when looping_active flipped. The smoke run on `dd63a75`
(`runs/042A-recompile-smoke/notes.md`) identified the cause:

**Dynamo LRU cache eviction.** Pre-warm correctly generated all required graph
variants (cu_seqlens × looping_active × slope), including the slope=0.5 path.
But `cache_size_limit` defaulted to 8, and the pre-warm produced 17+ unique
graphs. Earlier graphs (including the slope=0.5 ones) were LRU-evicted before
training reached the slope-switch + loop-activation transition, forcing full
Dynamo retraces. **Fix in `4b29c64`:** `cache_size_limit = 32`, which holds
all pre-warm variants simultaneously.

A small additional fix in `aff2de4` extends the pre-warm to also cover
`compiled_forward_logits` (a separately-compiled function used by `eval_val`)
and the `looping_active=True` slope variant. Without this, the first val_loss
eval and the first val after the slope switch would each cold-compile
forward_logits (~30s each, one-time). Not critical, but cheap defense-in-depth.

Combined: all four (forward × forward_logits) × (slope=0.7071 × slope=0.5)
graph variants are pre-cached before training begins.

## Hypothesis

The fused backward bug (commits before 5bbf12f) accidentally used effective slope
√0.5≈0.707 on negative activations instead of 0.5. Contaminated runs showed dramatically
better mid-training signal but vibrated at warmdown due to forward/backward inconsistency.

This spec harvests that benefit intentionally: run with `NEGATIVE_SLOPE=0.7071` (≈√0.5)
for the first 75% of training, then switch cleanly to `NEGATIVE_SLOPE=0.5` at warmdown
start. The switch fires exactly once at `frac = 1 - WARMDOWN_FRAC = 0.25`.

- Phase 1 (frac 0–0.25): s=0.7071 — more gradient flow through negative activations,
  especially beneficial when the recurrent loop activates at frac=0.35
- Phase 2 (frac 0.25–1.0): s=0.5 — forward/backward consistent, clean convergence

The pre-warm now compiles both slope variants, so the slope kernel switch is a cache
hit (~0ms). Logged at startup as `slope_anneal: precompiled warmdown kernel slope=0.5000`,
and at the switch step as `slope_anneal: 0.7071→0.5000 step:N frac:0.250`.

## Predecessor signal

The original (broken) 042A run reached step 3034/20000 in 1196s — 6× fewer steps
than baseline — yet pre-quant EMA val_bpb was **1.0886**, only ~0.024 worse than
the fully-trained 039b baseline (1.06514). With the recompile fix, this run should
reach the full step count and the early-training advantage should compound.

## Config diff

Two env var changes from the 039b baseline:

```bash
NEGATIVE_SLOPE=0.7071
SLOPE_WARMDOWN=0.5
```

All loop/architecture settings identical to baseline (layers 3-5, NL=2, frac=0.35).

## Regime

- `4×H100`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=0`
- Regions: AP-JP-1 or US-NE-1

## Launch form

```bash
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=1000
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=1
export RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.7071 SLOPE_WARMDOWN=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0 VAL_LOSS_EVERY=1000
export RUN_ID="042A-slope-anneal-0707-to-05"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/042A-slope-anneal-0707-to-05-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/042A-slope-anneal-0707-to-05-screen/train.log 2>&1
```

## What to watch

- **Slope pre-warm at startup:** `slope_anneal: precompiled warmdown kernel slope=0.5000`
- **Slope switch:** `slope_anneal: 0.7071→0.5000 step:~1650 frac:0.250` in log
- **No mid-training pause:** step rate at step 1700 should match step 1600 (no 7-8 min gap)
- **Mid-run val trajectory (VAL_LOSS_EVERY=1000):** compare to baseline at matched steps
- **Final EMA val_bpb:** vs baseline 1.06514

## Acceptance

Baseline (039b): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641**
- **Noise zone**: 1.0641–1.0670 — run second seed
- **Kill**: pre-quant EMA val_bpb ≥ **1.0670**

Post-quant val_bpb also recorded (no TTT). Watch quant damage vs baseline.

## Prediction

The original (broken) 042A reached step 3034 with val_bpb 1.0886 — ~0.024 from
baseline at 6× fewer steps. Extrapolating naively, the early-training advantage
should compound. Optimistic: ~1.063x (clear win). Conservative: low end of noise
zone (~1.0645).
