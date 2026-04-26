# Spec 043B — early loop + slope shock simultaneously at frac=0.20

**Slug:** `043B-early-loop-and-slope-shock-screen`
**Created:** 2026-04-26
**Status:** READY (depends on 043A result for prioritization)
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `1796bb8`
**Links to:** `research/specs/043A-early-loop-frac02-screen.md`,
              `research/evaluations/042A-slope-anneal-0707-to-05-screen.md`

## Hypothesis

The broken 042A had **slope shock + loop activation fire within 2 steps of
each other** (step 1623 + 1625) and showed strong train_loss advantage at
matched steps. We've decomposed that into:
- "Earlier loop activation" (the dominant lever) → tested in 043A
- "Slope shock at any time" (zero contribution alone) → tested in 042A

But maybe the **interaction** matters: firing slope shock AT THE SAME TIME
as loop activation could help if the loop's first pass through MLPs lands
on slope=0.5 instead of slope=0.7071. The loop iterates layers 3–5, and
the activation behavior of those MLPs at the moment the loop kicks in
might shape early loop-pass dynamics.

This spec tests that interaction:
- `ENABLE_LOOPING_AT=0.20` (loop fires early)
- `NEGATIVE_SLOPE=0.7071`, `SLOPE_WARMDOWN=0.5`, **`SLOPE_WARMDOWN_AT=0.20`**
  → slope shock fires at the *same frac* as loop activation
- `WARMDOWN_FRAC=0.75` unchanged (LR schedule is unaffected, thanks to the
  `SLOPE_WARMDOWN_AT` env var added in commit `1796bb8`)

## Why bother given 042A killed slope-anneal alone?

Slope alone in 042A fired at frac=0.25 (slope) vs frac=0.35 (loop) — they
were *not* simultaneous. The "shock" landed BEFORE loop activation, in a
phase where the MLPs were operating non-recurrently. Maybe the slope value
during the loop's initial passes is what matters, and that wasn't tested.

This is a defensible hypothesis but a small one. Run only if 043A is
inconclusive (noise zone) and you want one more shot at finding signal in
the slope direction.

## Config diff (vs canonical baseline)

```bash
ENABLE_LOOPING_AT=0.20
NEGATIVE_SLOPE=0.7071
SLOPE_WARMDOWN=0.5
SLOPE_WARMDOWN_AT=0.20    # new env var, decouples from WARMDOWN_FRAC
```

All other settings identical to canonical 039b baseline.

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
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
# *** EARLY LOOP ***
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.20
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
# *** SHOCK FIRES SIMULTANEOUSLY WITH LOOP ***
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.7071 SLOPE_WARMDOWN=0.5 SLOPE_WARMDOWN_AT=0.20
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0
export RUN_ID="043B-early-loop-and-slope-shock"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/043B-early-loop-and-slope-shock-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/043B-early-loop-and-slope-shock-screen/train.log 2>&1
```

## What to watch

- **Both events at frac=0.20:** look for `slope_anneal: 0.7071→0.5000 step:N frac:0.200`
  followed within ~2 steps by `layer_loop:enabled step:N+~ frac:0.200 ...`
- **deserialize slope sync:** `deserialize: slope synced to warmdown value 0.5000` (from 72c7328)
- **Pre-quant EMA val_bpb** — the headline; compare to 043A and baseline

## Acceptance

Same thresholds as 043A.

- Strong win: < 1.0635
- Win: < 1.0641
- Noise zone: 1.0641–1.0670
- Kill: ≥ 1.0670

**Specifically interesting:** Δ vs 043A. If 043B − 043A < −0.001, the
slope-loop interaction is real. If ≈0, slope-shock-at-loop-time is also
null and we can finally close the slope direction.

## Prediction

If 043A already wins decisively → 043B might add a small marginal benefit
(~0.0005) or none. Probably not worth running unless 043A is inconclusive.

If 043A is in the noise zone → 043B might tip into the win zone if the
interaction is real. Worth running.

If 043A loses → 043B is unlikely to help (the dominant lever didn't move
the needle; adding a null effect on top doesn't change much).

## When to run

**After 043A.** Use 043A's result to decide if 043B is worth $3.50.
