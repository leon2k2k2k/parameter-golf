# Spec 041M — combine 041I + 041K with more aggressive frac (loop 4-5, NL=3, frac=0.20)

**Slug:** `041M-loop45-nl3-frac020-screen`
**Created:** 2026-04-26 (queued for 2026-04-27, run AFTER 041L)
**Status:** READY (run after 041L; only worth running if 041L is in noise zone)
**Branch:** `exp/042-slope-anneal-screen` (no code change needed)
**Commit:** `23abb5e`
**Continues:** 041 series

## Hypothesis

Same combine as 041L (loop 4-5 + NL=3 + early activation), but with
**ENABLE_LOOPING_AT=0.20 instead of 0.25**. More aggressive: gives the loop
more total passes in the same wallclock budget, at the cost of slightly
under-trained loop-band layers when the loop activates.

The case for going more aggressive than 041L:
- 041I (4-5, NL=2, frac=0.25) and 043A (3-5, NL=2, frac=0.20) both worked
  cleanly. So frac=0.20 is plausibly safe on either band.
- More loop iterations in the budget = more total loop training time.
- If 041L's win is small, 041M might tip it lower.

The case AGAINST going more aggressive:
- 039bE went to frac=0.175 (with confounded LR drop) and lost decisively.
  There's a quality cliff somewhere below frac~0.20.
- Loop 4-5 layers might be even more sensitive to under-training than 3-5
  (fewer layers means each one carries more representational load per pass).

## When to run this

**Only run if 041L lands in noise zone (1.0641–1.0670).** If 041L wins
decisively (< 1.0635), 041M is unnecessary — we have a clear winner already.
If 041L kills (≥ 1.0670), 041M is unlikely to recover (saturation/cliff
issue isn't solved by going more aggressive).

## Config diff (vs 041L: just one knob)

```bash
ENABLE_LOOPING_AT=0.20    # 041L used 0.25
```

All other settings identical to 041L (loop 4-5, NL=3, no slope shock, etc).

## Regime

Same as 041L: 4×H100 JP, SEED=42, 1200s wallclock. Cost ~$3.50.

## Launch form

Same as 041L's launch form, with one change:
- `ENABLE_LOOPING_AT=0.20` (was 0.25)
- `RUN_ID="041M-loop45-nl3-frac020"`
- run dir: `041M-loop45-nl3-frac020-screen`

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
# *** vs baseline: shrunk band, NL=3, very early activation ***
export NUM_LOOPS=3 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.20
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
export RUN_ID="041M-loop45-nl3-frac020"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/041M-loop45-nl3-frac020-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/041M-loop45-nl3-frac020-screen/train.log 2>&1
```

## What to watch

Same diagnostics as 041L. Key comparison:
- 041M EMA vs 041L EMA — does going more aggressive help?
- 041M EMA vs baseline (1.06514) — direct submission test

## Acceptance

Same thresholds as 041L:
- Strong win: < 1.0635
- Win: < 1.0641
- Noise zone: 1.0641–1.0670
- Kill: ≥ 1.0670

## Decision tree after results

| 041L | 041M | Next |
|---|---|---|
| Win | — | Don't run 041M; promote 041L to 8×H100 |
| Noise | Win | Use 041M as candidate, frac sweep |
| Noise | Noise | Both close; run a third seed of the better one |
| Noise | Kill | Stick with 041L; close direction |
| Kill | — | Don't run 041M; close direction, pivot to other ideas |

## Cost

~$3.50 if run.
