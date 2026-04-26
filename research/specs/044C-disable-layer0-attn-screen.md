# Spec 044C — disable layer-0 attention (forward-skip only)

**Slug:** `044C-disable-layer0-attn-screen`
**Created:** 2026-04-27
**Status:** READY (parallel-launchable with 044A and 044B)
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `a187ce4`
**Links to:** `research/ideas/disable-layer0-attn.md` (2026-03-31 SOTA used this)

## Hypothesis

Layer 0 attention operates on **fresh token embeddings with no context** —
no meaningful patterns to attend to. Empirically, layer-0 attention entropy
is near-uniform across positions in many architectures, meaning the model
spends compute + parameter capacity for ~zero information gain.

This spec tests V1: **skip the attention forward pass in layer 0** while
keeping the params allocated (still in qo_bank/kv_bank but unused). If
val_bpb holds (or improves), the next step (V2) is removing the params
entirely to shrink the artifact and trade for higher-precision quant.

## What this DOES vs DOESN'T do

**Does:**
- Block 0's forward becomes `x = x + MLP(norm(x))` (no attn term)
- Saves ~1-2% per-step compute (one attn block out of 11 layers)
- Tests "does the model NEED layer-0 attention?"

**Does NOT (V1 limitations):**
- Still allocates qo_bank[0:2], kv_bank[0:2] (~786K params)
- Doesn't change submission size
- Doesn't free quant budget

If V1 shows null/positive impact → V2 (separate spec) removes params for
actual size savings + quant headroom.

## Predicted outcome

The mechanism is well-known in the literature, but expected Δ is modest:

- Optimistic (model doesn't need layer-0 attn at all): ~−0.001 bpb (small win) +
  marginal step-count gain from saved compute
- Realistic (slight quality loss masked by step-count gain): ~tied with baseline
- Pessimistic (some real signal in layer-0 attn): ~+0.001 to +0.003 (small loss)

Even if pessimistic, the result is informative — tells us layer-0 attn IS doing
something, which matters for any future attention-allocation work.

## Config diff (one new env var)

```bash
DISABLE_LAYER0_ATTN=1
```

All other settings identical to canonical baseline.

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
# *** the only change: skip layer-0 attention forward ***
export DISABLE_LAYER0_ATTN=1
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0
export RUN_ID="044C-disable-layer0-attn"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/044C-disable-layer0-attn-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/044C-disable-layer0-attn-screen/train.log 2>&1
```

## What to watch

- **No new log line specifically** — _skip_attn is set on block 0 silently. If
  it's working, you should see slightly higher tok/s (saved 1-2% compute) and
  marginally more total steps than baseline (5156 → ~5200).
- **Pre-quant EMA val_bpb** — the headline
- **Quantized val_bpb** — confirms quant pipeline still works (no model state change here that affects quant)

## Acceptance

Both pre-quant EMA val_bpb and quantized val_bpb tracked.

Baseline (canonical 039b):
- pre-quant EMA: **1.06514**
- quantized: **1.07410**
- quant cost: +0.00896

Pre-quant thresholds:
- **Strong win**: < **1.0635** (model genuinely doesn't need layer-0 attn AND extra steps help)
- **Win**: < **1.0641**
- **Noise zone**: 1.0641–1.0670 (quality loss exactly compensated by step-count gain — V2 worth pursuing for size savings)
- **Kill**: ≥ **1.0670** (layer-0 attn is doing real work)

Quantized thresholds:
- Strong win < 1.0731 / Win < 1.0737 / Noise 1.0737–1.0766 / Kill ≥ 1.0766
- Quant cost regression flag if `quantized − pre_quant > 0.012`

## After this run

- **Win/strong win** → V2 spec: remove qo_bank[0:2] and kv_bank[0:2] allocations to actually shrink artifact, then either bump int6→int7 on some weights OR allocate to extra LQER rank.
- **Noise (tied)** → V2 still worth pursuing for size savings (no quality cost).
- **Kill** → close direction, layer-0 attn matters on this stack.

## Cost

~$3.50.

## Cross-references

- 044A (parallel): QK softening alone
- 044B (parallel): QK + 041K combo
- 041N (parallel): NL=4 alone
- All four can run on 4 separate pods simultaneously for ~$14 total
- All four pinned to commit `a187ce4` (with the new DISABLE_LAYER0_ATTN flag)
