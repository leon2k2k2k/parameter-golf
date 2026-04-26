# Spec 041N — push NL further (loop 4-5, NL=4, frac=0.35)

**Slug:** `041N-loop45-nl4-frac035-screen`
**Created:** 2026-04-26 (queued for 2026-04-27)
**Status:** READY
**Branch:** `exp/042-slope-anneal-screen` (no code change needed)
**Commit:** `a01c608`
**Continues:** 041 series (loop 4-5 shrunk-band screen)
**Links to:** `research/evaluations/042A-slope-anneal-0707-to-05-screen.md`,
              041H/041I/041K results

## Hypothesis

The 041 series showed monotonic improvement going from NL=2 to NL=3 on the
shrunk band:

| Run | Config | EMA val_bpb | Δ vs baseline |
|---|---|---|---|
| 041H | 4-5, NL=2, frac=0.35 | 1.06693 | +0.00179 |
| 041K | 4-5, **NL=3**, frac=0.35 | 1.06563 | +0.00049 |

**ΔNL=2→3 was −0.00130.** If we get even half as much going NL=3→4
(i.e., −0.00065), we land at **1.06498 — beats baseline by 0.0002 (win zone)**.
If the lever still has full strength, we land at 1.06433 (clear win).

This spec tests the simplest extension: same as 041K but `NUM_LOOPS=4`.
**No frac change, no slope shock, no other knobs touched.** Isolates "is
NL=4 better than NL=3?"

## Predicted trade-off

NL=4 adds ~24% throughput tax on the shrunk band (vs 041K's ~12%):
- 041K: 17 layer visits per token (11 base + 2×3 loop)
- 041N: **19 layer visits** per token (11 base + 2×4 loop)

Estimated post-loop step rate:
- 041K: ~3360K tok/s
- 041N: ~3000K tok/s (rough estimate, ~10% slower)

Estimated total step count:
- 041K reached: 5157 steps
- 041N expected: **~4700-4900 steps**

**Loop steps gained vs lost:**
- ~600 fewer total steps (cost ~0.0007 per project memory's ~1.1e-6 per non-loop step)
- ~2 extra loop passes per post-loop step × ~2900 post-loop steps × 2.2e-6 ≈ +0.013 (very optimistic)

The math is wildly favorable IF the "loop steps worth 2× non-loop" rule holds
linearly. It probably doesn't (saturation), but even partial value of extra
loop passes should net positive.

## Why 041N is the right next step (post-041L)

**041L result (came in 2026-04-26):** loop 4-5 + NL=3 + frac=0.25 landed at
pre-quant EMA **1.06615** — *worse* than 041K's NL=3 alone (1.06563). The
combine hypothesis FAILED: NL=3 and early-activation have anti-additive
interaction (probably "more loop passes need more pre-loop base training"
quality cliff). This kills the combine direction.

041N is orthogonal to the failed combine — it pushes ONLY the NL lever,
keeping the same frac=0.35 baseline timing that 041K used. If the NL lever
extends linearly (or even with mild diminishing returns), 041N could win.

If 041N also lands in noise zone → loop-shape direction is exhausted; the
local optimum is 041K (1.06563). We pivot to orthogonal ideas like EMA
decay sweep or cheaper-loop-iteration code work.

## Config diff (vs canonical baseline)

```bash
LOOP_START=4               # was 3
LOOP_END=5                 # unchanged
NUM_LOOPS=4                # was 2
```

ENABLE_LOOPING_AT=0.35 unchanged. NEGATIVE_SLOPE=0.5 unchanged. WARMDOWN_FRAC=0.75 unchanged.

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
# *** Two changes: shrunk band + NL=4. Frac unchanged. ***
export NUM_LOOPS=4 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.35
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
export RUN_ID="041N-loop45-nl4-frac035"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/041N-loop45-nl4-frac035-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/041N-loop45-nl4-frac035-screen/train.log 2>&1
```

## What to watch

- **Loop activation:** `layer_loop:enabled step:N frac:0.350 depth:5 encoder:[...]` — depth:5 because NL=4 + 1 = 5 total visits.
- **Final step count:** expect ~4700-4900 (NL=4 throughput tax)
- **Pre-quant EMA val_bpb** — the headline

## Acceptance

Baseline (canonical 039b): pre-quant EMA val_bpb **1.06514**

- **Strong win**: < **1.0635** (clear NL lever extension works)
- **Win**: < **1.0641**
- **Noise zone**: 1.0641–1.0670 (lever has saturation; close direction)
- **Kill**: ≥ **1.0670** (NL=4 over-shot; throughput tax killed it)

## Predicted outcome

- Optimistic (lever maintains strength): ~1.0643 (just barely win)
- Realistic (half diminishing returns from NL=3→4): ~1.0650 (low noise zone)
- Pessimistic (full saturation + throughput tax dominates): ~1.0660 (mid noise)

## Decision tree (041L already failed; this is just 041N's tree)

| 041N | Next |
|---|---|
| **Strong win** (< 1.0635) | Run second seed to confirm; if confirmed, promote to 8×H100 |
| **Win** (< 1.0641) | Same — second seed → 8×H100 |
| **Noise** (1.0641–1.0670) | Loop-shape direction exhausted; 041K (1.06563) is local optimum. Pivot to EMA decay sweep + cheaper-loop-iteration code work |
| **Kill** (≥ 1.0670) | NL lever has plateaued/regressed; same pivot as noise |

**Skip 041M** (NL=3 + frac=0.20). 041L's negative interaction kills the
"more aggressive combine" hypothesis. 041M would just push further in a
direction we now know doesn't work.

## Cost

~$3.50.
