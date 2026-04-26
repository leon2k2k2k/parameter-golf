# Spec 044D — frozen QK_GAIN at 3.5 (non-learnable, mid-range)

**Slug:** `044D-qk-gain-frozen-3p5-screen`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `e37a9ce`
**Links to:** `research/ideas/per-layer-qk-gain.md`,
              044A result (init=2.5 learnable → 1.06756, +0.00242 vs baseline)

## Hypothesis

044A revealed a sharp disconnect:
- **train_loss systematically better** than baseline (Δ −0.029 by step 4500, all 11 matched steps better)
- **pre-quant EMA val_bpb +0.0024 WORSE** than baseline

That ~0.03 train↔val mismatch needs explanation. Three candidate
mechanisms (from prior analysis):

1. **Loss-scale artifact:** softer attention shifts CE loss numerically
   without changing model quality
2. **Genuine train/val divergence:** softer init lets model overfit
   training distribution
3. **Gradient on q_gain itself is the issue:** Muon's gradient pressure
   on q_gain produces some pathology — wrong direction, oscillation, or
   spurious updates that hurt generalization

044D tests **#3** by FREEZING q_gain at a fixed mid-range value (3.5,
between 044A's 2.5 and baseline's 5.0). If frozen-at-3.5 wins or matches
baseline, the gradient pressure on q_gain was the culprit. If it tracks
044A's pattern (train better, val worse), value/scale is the culprit and
the QK direction is dead.

## Why 3.5 specifically

- Sits between 044A's losing init (2.5) and baseline's winning init (5.0)
- Slightly above 044A's trained mean (2.43), slightly below baseline-trained
  values (likely ~3.5-4.5 area given Muon pulls down from 5.0)
- A "compromise" value that doesn't commit to either extreme
- 044A showed gradients are gentle (most heads moved <±1.0 in 5000 steps),
  so a frozen value 1.0 above 044A's mean is reasonable

## What this run does NOT test

- Per-layer optimal init (would require QK_GAIN_PER_LAYER, separate spec)
- Other frozen values (3.0, 4.0) — could sweep if 3.5 is informative
- Whether frozen-vs-learnable matters at the WINNING init (would be
  frozen-at-5.0 vs learnable-at-5.0)

## Config diff (vs canonical baseline)

```bash
QK_GAIN_INIT=3.5            # was 5.0 (mid-range fixed value)
QK_GAIN_FROZEN=1            # NEW — disables learning on q_gain
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
# *** the two changes: QK_GAIN_INIT=3.5 + QK_GAIN_FROZEN=1 ***
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=3.5 QK_GAIN_FROZEN=1
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
export RUN_ID="044D-qk-gain-frozen-3p5"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/044D-qk-gain-frozen-3p5-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/044D-qk-gain-frozen-3p5-screen/train.log 2>&1
```

## What to watch

- **Verify q_gain stays at 3.5 throughout training** — execution can log
  `model.blocks[0].attn.q_gain.requires_grad` at startup; should be `False`.
- **Train_loss vs baseline at matched steps:**
  - If shows same artifact pattern as 044A (train better, val worse) →
    value/scale is the culprit (hypothesis #1 or #2)
  - If train_loss is closer to baseline, val_bpb closer to baseline →
    gradient pressure was the issue (hypothesis #3)
  - If train_loss is much closer to baseline AND val_bpb is BETTER than
    044A → big surprise, gradient pressure was hurting; frozen is the
    way

## Acceptance

Both pre-quant EMA val_bpb and quantized val_bpb tracked.

Baseline (canonical 039b):
- pre-quant EMA: **1.06514**
- quantized: **1.07410**

Pre-quant thresholds:
- **Strong win**: < **1.0635** (gradient was the issue; frozen at 3.5 is much better)
- **Win**: < **1.0641** (frozen approach has real benefit)
- **Noise zone**: 1.0641–1.0670 (probably similar pattern to 044A; informative but not winning)
- **Kill**: ≥ **1.0670** (value/scale is the issue, no benefit from freezing)

Quantized thresholds: Strong win < 1.0731 / Win < 1.0737 / Noise 1.0737–1.0766 / Kill ≥ 1.0766
Quant cost regression flag if `quantized − pre_quant > 0.012`.

## Predicted outcome

Realistically, this is a **research test, not a win attempt**. Possible outcomes:

- Optimistic (gradient pressure was the culprit): val_bpb ~1.0640 (clear win)
- Realistic (value/scale issue, frozen doesn't help): val_bpb ~1.0670 (similar to 044A, possibly worse since q_gain can't drift toward better values)
- Pessimistic (frozen at wrong value compounds problems): val_bpb ≥ 1.0680 (clear loss; q_gain needed Muon's adjustment)

**Most likely:** noise zone 1.066-1.068 with similar train↔val pattern as 044A.
That answers the research question (artifact lives in the value, not the gradient)
and lets us close the QK direction definitively.

## After this run

- **Win or strong win** → frozen-at-other-values sweep (3.0, 4.0)
- **Noise** → close uniform-QK direction; per-layer init is the only remaining QK shot
- **Kill** → close QK direction entirely; pivot to EMA decay, LQER, layer-0-attn (044C)

## Cost

~$3.50.
