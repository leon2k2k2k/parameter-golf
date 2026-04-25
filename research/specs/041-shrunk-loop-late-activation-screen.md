# Spec 041 — shrunk loop (layers 4-5) + activation at frac=0.46

**Slug:** `shrunk-loop-late-activation-screen`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/evaluations/040-no-loop-ablation.md`, `research/specs/040-no-loop-ablation-screen.md`

## Hypothesis

Spec 040 showed the no-loop model reaching a lower loss basin around step
5600–6100 (raw train loss 2.29 at step 6100 vs baseline's 2.31 at step 5100),
but without recurrence depth. The baseline with loop@0.35 never reaches that
basin — it stops at step 5156.

Two changes combined:

1. **Shrink the loop from 3 layers (3-5) to 2 layers (4-5).** This reduces
   effective passes from 17 to 15 (~12% fewer FLOPs), giving ~13% faster
   throughput with loop active (~3616K vs 3200K tok/s).

2. **Activate at frac=0.46 (552s wallclock).** Chosen so the total step count
   lands at ~6000 — more steps than baseline (5156) while keeping a meaningful
   recurrent phase (~3000 steps with loop active).

**Expected trajectory:**
- Phase 1 (0–552s): ~5.47 steps/s → ~3020 steps
- Phase 2 (552–1200s): ~4.60 steps/s → ~2980 recurrent steps
- Total: ~6000 steps vs baseline's 5156

The activation spike at step ~5580 will be smaller than baseline's (2 loop
layers instead of 3, ~2/3 the architectural shock). Low LR at that point is
not guaranteed (warmdown starts at step ~5000 in ITERATIONS space, so step
5580 is early in warmdown), but the model is already settled in a better basin.

## Config diff

Three changes from the 039b baseline (commit `5bbf12f`):

```bash
LOOP_START=4
LOOP_END=5
ENABLE_LOOPING_AT=0.46
RECUR_ALPHA_ENABLED=0
```

Layer 3 becomes a normal (non-looped) layer. Layers 4-5 run 3 passes total
(NUM_LOOPS=2 unchanged).

`RECUR_ALPHA_ENABLED=0` is required: the hardcoded `recur_alpha`/`recur_beta`
buffers are calibrated for 3 loop layers (3-5). With LOOP_START=4 those
values map to the wrong layers. Carry is neutral (established by 039bK/bL)
so disabling it is safe and correct.

## Regime

- `4×H100`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=1`
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
export NUM_LOOPS=2 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.46
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=0.5
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=0
export RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=1
export RUN_ID="041-shrunk-loop-late-activation"

mkdir -p /workspace/runs/041-shrunk-loop-late-activation-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/041-shrunk-loop-late-activation-screen/train.log 2>&1
```

## What to watch during the run

- **Throughput Phase 1 (0–552s):** should be stable ~4308K tok/s (no loop)
- **Loop activation:** expect around step ~3020, log should show `layer_loop:enabled`
- **Throughput Phase 2:** should drop to ~3600K tok/s (2-layer loop, faster than baseline's 3200K)
- **Step at wallclock cap:** expect ~6000 steps total
- **Activation spike:** should be smaller than baseline's (2 loop layers, not 3)
- **Post-activation loss:** watch for continued descent

## Acceptance

Baseline (039b, loop@0.35 on layers 3-5, commit `5bbf12f`): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641** — basin + recurrence combo works
- **Noise zone**: 1.0641–1.0670 — run a second seed
- **Kill**: pre-quant EMA val_bpb ≥ **1.0670** — late activation spike or shallow recurrence not worth it

## What each outcome means

**Win**: The basin crossing is real and recurrence in the better basin adds meaningful value even with only ~830 steps. Next step: tune the activation timing (try 0.80, 0.90) and loop size.

**Kill**: Either the basin crossing benefit is outweighed by the smaller loop depth, or the activation spike at high LR disrupts the better basin. Next step: try late activation without shrinking the loop (frac=0.85, layers 3-5) to isolate which factor is responsible.
