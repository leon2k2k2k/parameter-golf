# Spec 044B — QK_GAIN softening combined with 041K's loop config

**Slug:** `044B-qk-gain-soft-on-041K-screen`
**Created:** 2026-04-27
**Status:** READY (parallel-launchable with 044A)
**Branch:** `exp/042-slope-anneal-screen` (no code change needed)
**Commit:** `a187ce4`
**Links to:** `research/ideas/per-layer-qk-gain.md`,
              041K result (4-5, NL=3, frac=0.35 → EMA 1.06563 — best single-lever to date)

## Hypothesis

Two independent levers from different lever classes:
1. **QK softening** (`QK_GAIN_INIT=2.5`) — attention init lever, per PR #1648
   cross-seed evidence.
2. **041K config** (loop 4-5, NL=3, frac=0.35) — best single-lever loop
   variant so far at EMA **1.06563** (+0.00049 vs baseline) — closest any
   screen has gotten to baseline.

Different lever classes (attention init vs loop architecture) should NOT have
the negative-interaction problem we hit with 041L (which tried to combine
two same-class loop levers and went anti-additive).

If the levers are even partially additive:
- 041K alone: −0.00049 (already this close to baseline)
- QK softening contribution (predicted): −0.001 to −0.003
- Combined: ~1.0635–1.0645 → potentially **win zone**

## Why 041K (not 041I)

041K is currently our **best individual screen result** (1.06563), beating
041I (1.06594), 043A (1.06610), and 042A (1.06661). Stacking the orthogonal
QK lever on top of the best loop config maximizes the chance of clearing the
win threshold.

041L showed combining same-class loop levers (NL=3 + early activation) is
anti-additive on this stack. But QK softening is a fundamentally different
mechanism — it doesn't change how many loop steps you get or where they
fire; it only changes how sharply attention computes from the start. So
the cross-class combination should be cleaner.

## Why this is interesting alongside 044A

- 044A (QK alone) tests the direction in isolation
- 044B tests whether QK stacks with the strongest known loop variant
- If 044B wins clearly while 044A is noise → the combo is the unlock
- If 044A wins and 044B doesn't beat it → QK doesn't stack with shrunk-band NL=3
- If both win → QK is a real lever and we have a new building block

## Config diff (vs canonical baseline)

```bash
QK_GAIN_INIT=2.5            # was 5.0 (QK softening)
LOOP_START=4                # was 3 (041K shrunk band)
LOOP_END=5                  # unchanged
NUM_LOOPS=3                 # was 2 (041K extended depth)
ENABLE_LOOPING_AT=0.35      # unchanged (041K kept baseline timing)
```

NEGATIVE_SLOPE=0.5 (no shock). WARMDOWN_FRAC=0.75 unchanged.

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
# *** QK softening ***
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=2.5
# *** 041K shrunk-band NL=3, baseline timing ***
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
export RUN_ID="044B-qk-gain-soft-on-041K"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/044B-qk-gain-soft-on-041K-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/044B-qk-gain-soft-on-041K-screen/train.log 2>&1
```

## What to watch

- **Loop activation:** `layer_loop:enabled step:N frac:0.350 depth:4` (NL=3 + 1 = 4 visits)
- **Final step count:** expect similar to 041K (~5157 steps)
- **Pre-quant EMA val_bpb** — the headline

## Acceptance

Both **pre-quant EMA val_bpb** and **quantized val_bpb** are tracked.

Baseline (canonical 039b):
- pre-quant EMA: **1.06514**
- quantized: **1.07410**
- quant cost: +0.00896

Pre-quant thresholds:
- **Strong win**: < **1.0635** (combo really stacks)
- **Win**: < **1.0641**
- **Noise zone**: 1.0641–1.0670 (check 044A and 041K to attribute)
- **Kill**: ≥ **1.0670** (negative interaction; QK kills loop benefit)

Quantized thresholds (must also pass):
- **Strong win**: < **1.0731**
- **Win**: < **1.0737**
- **Noise zone**: 1.0737–1.0766
- **Kill**: ≥ **1.0766**

Quant cost regression flag: if `quantized − pre_quant > 0.012` (vs baseline's
+0.009), investigate before promoting.

> **Bigger context (per memory update 2026-04-27):** current SOTA submission
> is **PR #1801 @ 1.06287**. Even a "win" against our screen baseline
> (< 1.06414) is still ~0.001 worse than current submission. To actually
> advance the leaderboard, we need post-quant total < 1.06287, which means
> screen pre-quant ~< 1.054 (accounting for ~+0.009 quant tax). Single
> levers won't get us there alone — we need stacking + tokenizer + larger-
> structural changes. 044 series is one piece of the stack.

## Predicted outcome

Starting from 041K (1.06563) and adding QK softening (predicted −0.001 to −0.003):
- Optimistic: **1.0635** (clear strong win, levers stack well)
- Realistic: **1.0648** (low noise zone)
- Pessimistic: **1.0655** (essentially same as 041K; QK doesn't help on this combo)

## Decision matrix (after both 044A and 044B return)

| 044A (QK alone) | 044B (QK + 041K) | Interpretation |
|---|---|---|
| Win | Win | QK lever real AND stacks → submission building block |
| Win | Noise | QK works alone but NL=3+shrunk band kills its benefit |
| Noise | Win | Levers genuinely complementary; combo is the lever |
| Noise | Noise | QK is small (or nothing) on our stack; close direction |
| Kill | * | QK kills training; close direction immediately |

## Cost

~$3.50 (parallel with 044A → ~$7 total for the pair).
