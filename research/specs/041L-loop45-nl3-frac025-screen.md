# Spec 041L — combine 041I + 041K levers (loop 4-5, NL=3, frac=0.25)

**Slug:** `041L-loop45-nl3-frac025-screen`
**Created:** 2026-04-26 (queued for 2026-04-27 execution)
**Status:** READY
**Branch:** `exp/042-slope-anneal-screen` (no code change needed; pins existing infrastructure)
**Commit:** `23abb5e`
**Continues:** 041 series (loop 4-5 shrunk-band screen)
**Links to:** `research/evaluations/042A-slope-anneal-0707-to-05-screen.md`,
              041I/041K results (logs on JP volume),
              `diary/2026-04-26-recompile-train-loss-artifact.md`

## Hypothesis

The 041 series identified two independent levers that each individually
improve over the shrunk-band 041H baseline:

| Lever | Run | EMA val_bpb | Δ vs 041H (1.06693) |
|---|---|---|---|
| Early activation alone | 041I (4-5, NL=2, frac=**0.25**) | 1.06594 | **−0.00099** |
| NL=3 alone | 041K (4-5, **NL=3**, frac=0.35) | 1.06563 | **−0.00130** |

041L combines BOTH levers in one spec: **loop 4-5, NL=3, frac=0.25**.

If the effects are additive (or even half-additive), prediction:

```
041H base:       1.06693
- 0.00099 (early activation lever)
- 0.00130 (NL=3 lever)
≈ 1.06464  → win zone (< baseline 1.06514, < acceptance 1.06410 strong)
```

If effects only half-add: **~1.06578** (still close, just noise zone).

The 042A/043A/043B arc closed the slope-anneal direction (slope shock
is a small drag in every config tested). The remaining open lever is
this combination on the shrunk band.

## Why frac=0.25 (not 0.20)

- 041I tested 0.25 cleanly and won small (+0.00080 vs baseline).
- 043A tested 0.20 on canonical band and got +0.00096 vs baseline — slightly worse.
- 039bE tested 0.175 + LR drop and lost decisively (+0.00402).
- 0.25 is the known-good early-activation point. Stick to it; don't risk under-trained loop layers.

## Why NL=3 + early might NOT just add

Caveats that could prevent the predicted win:
1. **Saturation:** if the loop benefit plateaus quickly with more passes, NL=3
   on top of early activation might give less than NL=3 alone did.
2. **Throughput tax compounding:** NL=3 already costs ~12% throughput; combined
   with earlier activation = even fewer total steps. The math assumes loop
   steps are 2× more valuable than non-loop, per project memory. If that
   value ratio drops below ~1.5× under heavier loop use, the trade flips.
3. **Per-loop quality on shrunk band:** the 4-5 band already has known
   per-loop-step quality cost (vs canonical 3-5). NL=3 on that compounds.

If 041L lands in noise zone (1.0641–1.0670) → the levers don't fully stack;
we've found the local optimum on this stack. If 041L wins (< 1.0641) → next
step is a frac sweep (0.20, 0.30) to optimize.

## Config diff (vs canonical baseline)

```bash
LOOP_START=4               # was 3
LOOP_END=5                 # unchanged
NUM_LOOPS=3                # was 2
ENABLE_LOOPING_AT=0.25     # was 0.35
```

NEGATIVE_SLOPE=0.5 (no shock), WARMDOWN_FRAC=0.75 unchanged.

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
# *** the three changes vs baseline ***
export NUM_LOOPS=3 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.25
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
# Slope held at baseline (no shock — confirmed drag in 042A and 043B)
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0
export RUN_ID="041L-loop45-nl3-frac025"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/041L-loop45-nl3-frac025-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/041L-loop45-nl3-frac025-screen/train.log 2>&1
```

## What to watch

- **Loop activation:** `layer_loop:enabled step:N frac:0.250 depth:4 encoder:[...]` — depth:4 because NL=3 + 1 = 4 total passes. Should fire around step ~1333 (2×H100 throughput similar to 4×H100 in this regime).
- **Final step count:** expect ~4500-4700 (NL=3 throughput tax + early activation overhead). Lower than baseline 5156.
- **Pre-quant EMA val_bpb** — the headline:
  - **Strong win**: < 1.0635
  - **Win**: < 1.0641
  - **Noise zone**: 1.0641–1.0670
  - **Kill**: ≥ 1.0670

## Acceptance

Baseline (canonical 039b): pre-quant EMA val_bpb **1.06514**

- Win threshold: **< 1.0641** (Δ < −0.001)
- Noise zone: 1.0641–1.0670 — would suggest levers don't fully stack
- Kill: ≥ 1.0670 — combination is worse than individual variants (saturation hit hard)

## Predicted outcome

- Optimistic (effects mostly additive): **1.0646** (clear win)
- Realistic (effects partially overlap): **1.0656** (noise zone, marginal)
- Pessimistic (saturation eats most of one effect): **1.0660** (noise zone, mid)

## After this run

If wins → frac sweep (041M: 0.20, 041N: 0.30) to find the optimum.
If noise → close the loop-shape direction; pivot to EMA decay or
forward-pass speedups (cheaper loop iterations) per
`project_cheaper_loop_iterations_for_tomorrow.md`.
