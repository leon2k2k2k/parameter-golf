# Spec 040 — no-loop ablation on 039b fixed baseline

**Slug:** `no-loop-ablation-screen`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/evaluations/043a-vs-038a-investigation.md`

## Hypothesis

The 039b arc established that LR shape (WSD warmdown) is the dominant training
signal — no LR schedule modification around loop activation beat the baseline.
The recurrence loop costs ~33% throughput once active (steps ~2300–5100:
~4300 → ~3200 tok/s), meaning the same 1200s wallclock budget delivers ~20%
fewer tokens with recurrence than without.

If the model is still on a steep region of the loss curve at end-of-run (which
the baseline trajectory suggests), those extra ~20% tokens could matter more
than the architectural depth gain from recurrence. This spec tests that directly:
same model, same WSD schedule, same everything — just never activate the loop.

A no-loop win means the throughput tax exceeds the depth benefit at this
budget. A loss means recurrence is genuinely earning its cost.

## Config diff

One change from the 039b baseline (commit `5bbf12f`, WSD, `ENABLE_LOOPING_AT=0.35`):

```bash
NUM_LOOPS=0
```

This causes `if h.num_loops > 0` to be false, skipping all loop setup,
recur_alpha, and carry state. The model is an unmodified 11-layer forward pass.

Everything else — architecture, LR schedule, quantization, seeds — is identical
to the baseline that produced pre-quant EMA val_bpb **1.06514**.

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
export NUM_LOOPS=0 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
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
export RUN_ID="040-no-loop-ablation"

mkdir -p /workspace/runs/040-no-loop-ablation-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/040-no-loop-ablation-screen/train.log 2>&1
```

## What to watch during the run

- **No loop activation event** in the log (should never see `layer_loop:enabled`)
- **Throughput** should be stable at ~4300 tok/s throughout (no mid-run drop)
- **Step count** at wallclock cap: expect ~6200–6500 steps (vs ~5100–5400 with loop)
- **Train loss trajectory**: the no-loop curve should not show the spike at ~step 2300

## Acceptance

Baseline (039b, loop@0.35, WSD, commit `5bbf12f`): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641** — throughput > depth at this budget; recurrence should be revisited
- **Noise zone**: 1.0641–1.0670 — ambiguous; run a second seed before concluding
- **Kill**: pre-quant EMA val_bpb ≥ **1.0670** — recurrence earns its cost; loop is load-bearing

## What each outcome means

**No-loop wins**: The ~20% extra tokens from not looping matter more than architectural depth. Next step: explore whether a deeper/wider no-loop model can use the recovered capacity budget, or whether loop@0.175 (activating late so most of the budget is unlooped) is even better than no-loop-at-all.

**No-loop loses**: Recurrence is genuinely earning its 33% throughput tax. The 039b arc's problem isn't the loop itself — it's the LR schedule modifications around activation. Loop@0.175 (from spec 042, on buggy code) becomes the cleaner next candidate to validate on fixed code.
