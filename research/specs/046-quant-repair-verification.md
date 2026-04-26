# Spec 046 — RESUME_FROM_CKPT verification on armD

**Slug:** `046-quant-repair-verification`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/046-quant-repair`
**Commit:** `0ea6a97`
**Parent idea:** `research/ideas/quant-repair.md`

## Hypothesis

The new `RESUME_FROM_CKPT` mode (added in commit `0ea6a97`) loads an existing
`final_model.pt`, runs GPTQ with current env vars, and emits both pre-quant
and quantized val_bpb. If the infrastructure works, loading 045 armD's checkpoint
with armD's exact env vars should reproduce armD's quantized number (1.07467).

## Why this is the family entry point

All 046A-F arms run via `RESUME_FROM_CKPT` on the same checkpoint. If 046
verification fails, every downstream sweep is invalid. **046 must pass before
any 046A+ runs.**

## Reference checkpoint

- **Run**: `045-loop-layer-improvements/armD`
- **Path**: `/workspace/runs/045-loop-layer-improvements/armD/final_model.pt` (135 MB, NE-1 volume)
- **Trained at commit**: `1c6cd7c` (spec 045 levers introduced; older than fc54262 optimizer fix)
- **armD's env vars**: `LOOP_ITER_EMBEDS=0 MLP_ONLY_FROM_PASS=0 LOOP_SCALE_INIT=recip`
- **armD's results**:
  - pre-quant EMA: **1.06547**
  - quantized: **1.07467**
  - quant tax: **+0.00920**

## Acceptance

- **PASS**: pre-quant EMA matches armD's 1.06547 (±0.0001 numeric noise) AND quantized matches 1.07467 (±0.0002 GPTQ stochastic noise)
- **FAIL**: anything else — investigate before launching 046A+

If pre-quant doesn't match: state_dict load issue (architecture mismatch). Likely cause: env var mismatch — re-check against armD's spec.
If pre-quant matches but quantized differs: GPTQ stochastic variance OR commit-level GPTQ logic difference between 1c6cd7c and 0ea6a97.

## Regime

- `4×H100 NE-1`, `SEED=42`, no training
- Cost: ~$1 (5-7 min: load + GPTQ + 2 evals)

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
# *** armD's spec 045 levers (must match training-time config) ***
export LOOP_ITER_EMBEDS=0 MLP_ONLY_FROM_PASS=0 LOOP_SCALE_INIT=recip
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=1000
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
# *** baseline quant config — DO NOT MODIFY for verification ***
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
# *** the new mode — load armD's checkpoint, skip training ***
export RESUME_FROM_CKPT=/workspace/runs/045-loop-layer-improvements/armD/final_model.pt
export RUN_ID="046-quant-repair-verification"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/046-quant-repair-verification

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/046-quant-repair-verification/train.log 2>&1
```

## What to watch

- `resume_from_ckpt=...: skipping train_model, loading checkpoint` line at startup
- Pre-quant eval runs first (no GPTQ yet) — should match 1.06547
- `Serialized model:` line indicates GPTQ ran
- Post-quant eval — should match 1.07467

## Decision tree

| Result | Next |
|---|---|
| **PASS** (both within tolerance) | Proceed to 046A (Q5 calib sweep). Infrastructure proven. |
| Pre-quant differs | Architecture/load issue. Diff env vars vs armD. Likely missing/extra LOOP_* env. |
| Quantized differs but pre-quant matches | GPTQ stochastic OR commit-level quant logic differs. Check if any quant logic changed between 1c6cd7c and 0ea6a97. |
| Both differ wildly | Catastrophic load failure. Check log for state_dict mismatch errors. |

## Cost

~$1 single 5-7 min run on 4×H100.
