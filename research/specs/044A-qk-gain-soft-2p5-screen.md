# Spec 044A — QK_GAIN_INIT=2.5 (soften attention init)

**Slug:** `044A-qk-gain-soft-2p5-screen`
**Created:** 2026-04-27
**Status:** READY (parallel-launchable with 044B)
**Branch:** `exp/042-slope-anneal-screen` (no code change needed)
**Commit:** `3d4776b`
**Links to:** `research/ideas/per-layer-qk-gain.md` (PR #1648 source)

## Hypothesis

Canonical baseline initializes `q_gain = full((num_heads,), 5.0)` uniformly
across all 11 layers. PR #1648 (mikeapedia, off-stack) ran a convergence loop
and consistently landed at per-layer values in the **2.0–3.0 range** across
seeds. Author quote:

> "The model consistently prefers much softer attention than the 5.0 default."

If 5.0 is genuinely too sharp, **early training wastes steps fighting the wrong
init** before Muon can soften it. With ~5000 steps in the 1200s wallclock budget,
anything that lets the model start closer to its natural operating point is
pure win.

This spec tests the **direction only** (uniform softer vs uniform 5.0) with a
single env-var change. If direction is right, Phase 2 (separate spec) finds
optimal per-layer values via convergence loop on our stack.

## Why test on our stack

mikeapedia's evidence was on #1586's stack (no CaseOps, no QuantGate, different
TTT). Our canonical baseline is #1736 + caseops + sparse gates + LQER. The
optimal `q_gain` could differ. But:
- The *direction* (softer is better) cross-validates across seeds, so it's
  unlikely to be stack-specific
- The gap (5.0 → 2.5) is too large to be pure noise

## Config diff (single env var change vs canonical baseline)

```bash
QK_GAIN_INIT=2.5    # was 5.0
```

All other settings identical to baseline.

## Regime

- `4×H100 JP`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=0`
- Cost: ~$3.50

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
# *** the only change: QK_GAIN_INIT=2.5 (was 5.0) ***
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=2.5
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
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0
export RUN_ID="044A-qk-gain-soft-2p5"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/044A-qk-gain-soft-2p5-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/044A-qk-gain-soft-2p5-screen/train.log 2>&1
```

## What to watch

- **q_gain log line at startup:** confirm `q_gain` is initialized at 2.5 across heads (look for any printout of attention init values)
- **Loop activation:** standard at frac=0.35
- **Pre-quant EMA val_bpb** — the headline

## Acceptance

Baseline: pre-quant EMA val_bpb **1.06514**

- **Strong win**: < **1.0635** (clear direction win, consistent with mikeapedia's claim)
- **Win**: < **1.0641** (acceptance threshold)
- **Noise zone**: 1.0641–1.0670 (signal exists but small; second seed needed)
- **Kill**: ≥ **1.0670** (direction wrong on our stack, or stack-specific reason 5.0 is right)

## Predicted outcome

Per the source's "consistent across seeds" framing, the direction is probably
right. Magnitude is uncertain on our stack. Most likely:
- Optimistic (matches PR #1648's direction strongly): **1.0640** (clear win)
- Realistic (smaller effect on our stack with caseops/etc): **1.0648** (noise zone)
- Pessimistic (TTT compensates, init doesn't matter much): **1.0655** (mid noise)

## After this run

- **Win** → run convergence loop / per-layer optimization (Phase 2 spec). Stack QK softening with whatever else we find.
- **Noise** → second seed; if confirmed, marginal but worth keeping.
- **Kill** → close direction; the per-layer claim is stack-specific to #1586.

## Cost

~$3.50.

## Cross-references

- 044B (parallel spec): combines QK softening with 041I's loop config
- `research/ideas/per-layer-qk-gain.md` — full background
