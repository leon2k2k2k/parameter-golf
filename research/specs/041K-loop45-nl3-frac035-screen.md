# Spec 041K — loop layers 4-5, NUM_LOOPS=3, frac=0.35

**Slug:** `041K-loop45-nl3-frac035-screen`
**Created:** 2026-04-26
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/ideas/041-late-loop-activation-basin-crossing.md`

## Hypothesis

041A (layers 4-5, NL=2, frac=0.46) nearly ties the baseline but falls short by 0.00031.
The frac sweep (041H/I/J) is testing earlier activation, but early-activation penalty appears
real. This spec tests a cleaner lever: **deeper recurrence on the same narrow band**.

**Key symmetry:** NL=3 on 2 layers → N_eff = 11 + 3×2 = **17** — identical to the baseline
(NL=2, layers 3-5). So loop-phase throughput matches baseline exactly (~3717 steps/s) and
we get the same ~2899 loop steps. But each step does **3 passes** through layers 4-5
instead of 2 passes through 3-5.

The "depth > coverage" lesson from 041 arc: NL matters more than which layers are looped.
If that holds, NL=3 on 2 layers should outperform NL=2 on 3 layers at matched loop steps.

| Config | N_eff | Loop steps | Depth per step |
|--------|-------|------------|----------------|
| Baseline (3-5, NL=2) | 17 | 2899 | 2× through 3-5 |
| **041K (4-5, NL=3)** | **17** | **~2899** | **3× through 4-5** |

Prediction: should beat baseline. Hard to quantify — 2.2e-6/step was calibrated for NL=2,
doesn't directly apply. Best case: noticeably below 1.06514.

Risk: 3 passes through 2 layers may be qualitatively different from 2 passes through 3 layers
in a bad way (e.g., layer 3 carries information that 4-5 can't compensate for). But 041A
(NL=2, 2-layer loop) nearly tying baseline (NL=2, 3-layer loop) suggests coverage isn't critical.

## Config diff

```bash
LOOP_START=4
LOOP_END=5
NUM_LOOPS=3
ENABLE_LOOPING_AT=0.35
RECUR_ALPHA_ENABLED=0
```

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
export NUM_LOOPS=3 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.35
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
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=60
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=0
export RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0 VAL_LOSS_EVERY=1000
export RUN_ID="041K-loop45-nl3-frac035"

mkdir -p /workspace/runs/041K-loop45-nl3-frac035-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/041K-loop45-nl3-frac035-screen/train.log 2>&1
```

## What to watch

- **Loop activation:** step ~2308 (same as baseline)
- **Throughput Phase 2:** ~3717 steps/s — should match baseline (N_eff=17)
- **Step at wallclock cap:** ~5207 steps
- **Loop steps:** ~2899 (same as baseline)

## Acceptance

Baseline (039b): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641**
- **Noise zone**: 1.0641–1.0670 — run second seed
- **Kill**: pre-quant EMA val_bpb ≥ **1.0670**

## Prediction

Should beat baseline if depth > coverage holds. No direct numerical prediction since
2.2e-6/step rate was calibrated for NL=2. Compare to baseline (1.06514) and 041A (1.06545).
