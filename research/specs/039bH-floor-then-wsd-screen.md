# Spec 039bH — floor_then_wsd LR schedule screen

**Slug:** `floor-then-wsd-screen`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `b92d802`
**Links to:** `research/specs/039bG-lr-floor-only-screen.md`, `research/specs/039bE-early-loop-lr-floor-screen.md`

## Background

From the 039bE/039bG experiments, two findings:

1. **LR is the dominant driver** — 039bG (loop@0.35 + first_half_floor) beat
   the baseline by −0.123 train loss at step 2800, nearly as much as 039bE
   (loop@0.175 + first_half_floor, −0.144) but without the 1.9m wallclock
   penalty from early loop. LR explains ~86% of the mid-training gain.

2. **first_half_floor stalls at the floor** — both 039bE and 039bG decay fast in
   the first half but then sit at MIN_LR=0.1 for the remaining 10 minutes.
   039bE pre-quant BPB (1.06916) was worse than baseline (1.06514), and quant
   BPB (1.08770) was 2× worse. The floor phase starves the model of gradient
   signal right when it needs to converge cleanly.

## Hypothesis

Replacing the second-half floor with the standard WSD curve evaluated at the
current frac gives the best of both phases:

- **First 35% of wallclock** (0–7min): first_half_floor — fast early decay,
  strong gradient signal early, model and recurrent stack adapt quickly
- **Remaining 65%** (7–20min): standard WSD at frac — same long linear decay as
  the baseline, proper convergence, clean weight distribution for quantization

The recurrent loop fires at 17.5% (3.5min, loop@0.175) inside the fast-decay
phase, adapts during declining LR, then benefits from the resumed standard
warmdown for proper settling.

## New schedule: `floor_then_wsd`

```
lr_mul(frac):
  if frac < LR_REWARM_AT:          # first_half_floor phase
      rf = frac / 0.5
      if rf >= 1 - WARMDOWN_FRAC: return max((1-rf)/WARMDOWN_FRAC, MIN_LR)
      return 1.0
  else:                             # standard WSD from current frac
      if frac >= 1 - WARMDOWN_FRAC: return max((1-frac)/WARMDOWN_FRAC, MIN_LR)
      return 1.0
```

Implemented in commit `f4a57c8`. Two new env vars:
- `LR_SCHEDULE_MODE=floor_then_wsd`
- `LR_REWARM_AT=0.35` (switch point; matches standard loop@0.35 firing time)

At frac=0.35 the schedule jumps from ~0.4 (orange) onto blue(0.35)=0.867 and
follows the standard WSD from there.

## Baseline

- branch: `exp/039b-loop-band-activation-screen`
- commit: `b92d802`
- reference: clean baseline, pre-quant EMA val_bpb=**1.06514**, quant=**1.07410**

## Config diff

Relative to the clean 039 baseline:

```bash
LR_SCHEDULE_MODE=floor_then_wsd   # new
LR_REWARM_AT=0.35                 # new
ENABLE_LOOPING_AT=0.175           # earlier loop onset
```

Everything else unchanged.

## Regime

- `4×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=1200`
- `TTT_ENABLED=0`
- training-only screen

## Resolved env block

```bash
DATA_DIR=/workspace/parameter-golf/data
DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin
VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin
VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin
VOCAB_SIZE=8192
NUM_LAYERS=11
XSA_LAST_N=11
MODEL_DIM=512
NUM_KV_HEADS=4
NUM_HEADS=8
MLP_MULT=4.0
TIE_EMBEDDINGS=1
LOGIT_SOFTCAP=30
ROPE_BASE=10000
ROPE_DIMS=16
ROPE_TRAIN_SEQ_LEN=2048
ROPE_YARN=0
LN_SCALE=1
QK_GAIN_INIT=5.0
NUM_LOOPS=2
LOOP_START=3
LOOP_END=5
ENABLE_LOOPING_AT=0.175
PARALLEL_START_LAYER=8
PARALLEL_FINAL_LANE=mean
MIN_LR=0.1
EMBED_LR=0.6
TIED_EMBED_LR=0.03
TIED_EMBED_INIT_STD=0.005
MATRIX_LR=0.026
SCALAR_LR=0.02
MUON_MOMENTUM=0.97
MUON_BACKEND_STEPS=5
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_ROW_NORMALIZE=1
BETA1=0.9
BETA2=0.95
ADAM_EPS=1e-8
GRAD_CLIP_NORM=0.3
ADAM_WD=0.02
MUON_WD=0.095
EMBED_WD=0.085
EMA_DECAY=0.9965
TRAIN_BATCH_TOKENS=786432
TRAIN_SEQ_LEN=2048
TRAIN_LOG_EVERY=100
ITERATIONS=20000
WARMDOWN_FRAC=0.75
WARMUP_STEPS=20
VAL_BATCH_TOKENS=524288
EVAL_SEQ_LEN=2048
EVAL_STRIDE=64
VAL_LOSS_EVERY=0
CASEOPS_ENABLED=1
COMPRESSOR=brotli
MATRIX_BITS=6
MATRIX_CLIP_SIGMAS=12.85
ATTN_CLIP_SIGMAS=13.0
MLP_CLIP_SIGMAS=12.0
EMBED_BITS=7
EMBED_CLIP_SIGMAS=15.0
GPTQ_CALIBRATION_BATCHES=16
GPTQ_RESERVE_SECONDS=0.5
SKIP_GATES_ENABLED=1
SPARSE_ATTN_GATE_ENABLED=1
SPARSE_ATTN_GATE_INIT_STD=0.0
SPARSE_ATTN_GATE_SCALE=1.0
GATED_ATTN_ENABLED=0
GATED_ATTN_INIT_STD=0.005
GATED_ATTN_QUANT_GATE=1
ATTN_OUT_GATE_ENABLED=0
ATTN_OUT_GATE_SRC=proj
GATE_WINDOW=12
RECUR_ALPHA_ENABLED=1
RECUR_DIAG_P2P_COS=0
SMEAR_GATE_ENABLED=1
LQER_ENABLED=1
LQER_RANK=4
LQER_TOP_K=3
LQER_FACTOR_BITS=4
LQER_ASYM_ENABLED=1
LQER_ASYM_GROUP=64
SPINQUANT_ENABLED=0
SPINQUANT_SEED=42
SPINQUANT_SITES=attn_in,attn_proj_in,mlp_in,mlp_proj_in
MLP_OUTER_ACTIVATION=leaky_relu_square
NEGATIVE_SLOPE=0.5
LR_SCHEDULE_MODE=floor_then_wsd
LR_REWARM_AT=0.35
SEED=42
MAX_WALLCLOCK_SECONDS=1200
TTT_ENABLED=0
```

## Launch form

```bash
env \
  ... (full env block above) ... \
  RUN_ID="039bH-floor-then-wsd" \
  torchrun --standalone --nproc_per_node=4 train_gpt.py \
  >> /workspace/runs/039-neg-slope-screen-on-1797-base/039bH/train.log 2>&1
```

## Compare against

| Arm | pre-quant BPB | quant BPB | Notes |
|---|---|---|---|
| baseline (loop@0.35, std WSD) | 1.06514 | 1.07410 | target to beat |
| 039bE (loop@0.175, first_half_floor) | 1.06916 | 1.08770 | LR stalls at floor |
| 039bG (loop@0.35, first_half_floor) | pending | pending | LR only, no early loop |
| **039bH** (loop@0.175, floor_then_wsd) | — | — | this spec |

## Acceptance

- **Win**: pre-quant EMA val_bpb < 1.0641 AND quant BPB ≤ 1.0750
- **Interesting**: pre-quant beats baseline even if quant is flat — worth
  investigating why (possible quant interaction with new LR shape)
- **Kill**: pre-quant ≥ 1.0660 — worse than baseline, floor_then_wsd doesn't help

## Open questions

- Does the jump from ~0.4 to blue(0.35)=0.867 at the switch point cause
  instability? Watch for loss spikes around 7min.
- Should LR_REWARM_AT be tuned (e.g. 0.25 or 0.40)? Defer to a follow-up if
  039bH wins.
- Does loop@0.35 + floor_then_wsd (no early loop cost) also win? Run 039bI if
  039bH wins and 039bG is inconclusive.
