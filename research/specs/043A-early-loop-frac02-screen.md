# Spec 043A — early loop activation (frac=0.20), clean isolation

**Slug:** `043A-early-loop-frac02-screen`
**Created:** 2026-04-26
**Status:** READY
**Branch:** `exp/042-slope-anneal-screen` (no code change needed; pins existing infrastructure)
**Commit:** `1796bb8`
**Links to:** `research/evaluations/042A-slope-anneal-0707-to-05-screen.md` (revised verdict),
              `research/ideas/early-loop-activation.md` (todo)

## Hypothesis

The broken 042A run looked great because its recompile pause shifted loop
activation from step 2289 (where frac=0.35 normally fires) to step 1625
(where the recompile-inflated frac happened to cross 0.35 first). That's
**~664 extra loop steps in the same wallclock budget**, which gave a
0.10–0.13 train_loss advantage at every matched step from 1700 onward.

The slope-anneal hypothesis itself contributed ≈0 (042A vs canonical baseline
at fixed loop timing was tied/slightly behind). **The actual lever was loop
activation timing.**

This spec isolates that lever cleanly: activate the loop earlier, change
nothing else.

- `ENABLE_LOOPING_AT=0.20` (vs baseline 0.35)
- Standard LR schedule (vs 039bE which dropped LR to floor — confounded)
- `NEGATIVE_SLOPE=0.5` throughout (vs 042A which added a slope shock — null effect)

## Predicted trade-off

Earlier loop activation = more loop steps but fewer total steps.

Per project memory:
- ~2.2e-6 val_bpb improvement per loop step
- Loop steps ~2× more valuable than non-loop steps for val_bpb
- "Delaying loop activation frac is always a losing trade at this scale" → conversely, advancing should help

Counterargument: loop on under-trained early layers (frac < 0.25) may have a
quality cliff. 039bE went all the way to frac=0.175 + LR drop and lost
(EMA 1.06916 vs baseline 1.06514). Picking 0.20 (less aggressive) and
holding LR schedule normal isolates whether early loop alone helps.

## Config diff (one env var change from canonical baseline)

```bash
ENABLE_LOOPING_AT=0.20
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
# *** the only change: ENABLE_LOOPING_AT=0.20 (was 0.35) ***
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
# Slope held at baseline (no shock)
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0
export RUN_ID="043A-early-loop-frac02"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/043A-early-loop-frac02-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/043A-early-loop-frac02-screen/train.log 2>&1
```

## What to watch

- **Loop activation log:** `layer_loop:enabled step:N frac:0.200 ...` — should fire around step ~1300 (vs baseline ~2289)
- **Final step count:** expect lower than baseline 5156 (more loop steps = slower per-step time). Estimate ~4400-4700.
- **Pre-quant EMA val_bpb** — the headline number

## Acceptance

Baseline: pre-quant EMA val_bpb **1.06514**

- **Strong win**: < **1.0635** (clearly real)
- **Win**: < **1.0641**
- **Noise zone**: 1.0641–1.0670 — run second seed
- **Kill**: ≥ **1.0670**

Compare also to 039bE (1.06916, early loop + LR drop) — if 043A beats 039bE,
isolates that the LR-drop-to-floor was the killer, not early loop.

## Prediction

The mechanism is real (loop steps are valuable), so I expect modest movement:
- Optimistic: 1.0635–1.0645 (clear win, ~0.001 improvement)
- Conservative: noise zone (1.0645–1.0660), with second seed needed
- Pessimistic: tied with baseline or slightly worse (loop on under-trained
  layers has a quality cliff at frac=0.20)

## After this run

If wins → frac sweep (043C: 0.15, 043D: 0.10) to find the optimum.
If noise → 043B (early loop + slope shock combo) might add the 0.001 needed.
If kills → close the loop-timing direction; both 042A's residual signal and
039bE's failure are explained by "early loop is genuinely bad on this stack."
