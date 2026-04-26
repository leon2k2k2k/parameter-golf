# Spec 042D — eval-only GPTQ fix verification (1×H100, ~3 min)

**Slug:** `042D-eval-only-gptq-fix-verify`
**Created:** 2026-04-26
**Status:** READY
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `2719021`
**Predecessor run:** `runs/042A-slope-anneal-0707-to-05-screen/` (broken quantized val_bpb 2.05)

## Why

042A produced pre-quant EMA val_bpb 1.06661 (noise zone, marginal vs baseline)
but quantized val_bpb 2.0503 — catastrophic +0.98 vs baseline's quant cost of
~+0.009. Root cause identified: `deserialize()` re-instantiates `GPT(h)` with
`negative_slope=h.negative_slope=0.7071` (env var), but the trained quantized
weights were optimized for slope=0.5 (post-warmdown). Fix in commit `72c7328`
syncs the slope on every MLP after `load_state_dict`. Commit `2719021` adds
`EVAL_ONLY=1` mode to skip training and run only the deserialize + quantized
eval portion against existing artifacts.

This spec verifies the fix end-to-end without paying for a full training rerun
(~$4 → ~$0.25).

## Hypothesis

With the deserialize slope sync, the existing 042A quantized blob (which is
unchanged) should produce a quantized val_bpb in the **~1.07-1.08 range**
(comparable to baseline's 1.0741), not 2.05.

If it does → the GPTQ failure was purely a deserialize bug; original training
artifacts are fine; we can decide whether to re-run training (with the fix
permanently in `train_gpt.py`) or promote on existing artifacts.

If it doesn't → the bug is elsewhere; need deeper investigation.

## Regime

- **1×H100 JP**, `MAX_WALLCLOCK_SECONDS` irrelevant (no training)
- Cost: ~$0.25 (~5 min wall × $2.99/hr including pod startup)

## Required: artifact location

The launch must point at the existing 042A artifacts on the JP volume.
Execution should locate where `final_model.int6.ptz` was written for the
original 042A run (cwd of that torchrun, probably
`/workspace/parameter-golf/final_model.int6.ptz` since the spec didn't set
`ARTIFACT_DIR`). Then either:

a) Set `ARTIFACT_DIR` in this spec's launch to that exact directory, OR
b) Run torchrun from that directory (inherits cwd → `final_model.{pt,int6.ptz}`
   resolved relative)

If the artifacts are missing on the volume → STOP, hand back, this spec is not
runnable until the original run's artifacts are recovered or the run is redone.

## Launch form

```bash
# *** IMPORTANT: Verify artifact location before launching ***
# ssh into pod and check: ls /workspace/parameter-golf/final_model*
# If files exist there, the launch below will work.
# If they're elsewhere, set ARTIFACT_DIR to the correct path.

export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
# Architecture must EXACTLY match the original 042A — deserialize allocates
# a fresh GPT(h) and load_state_dict will fail on any shape mismatch.
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
# These don't affect deserialize but Hyperparameters reads them.
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
# Slope settings MUST match the original 042A run for h.slope_warmdown
# to trigger the fix path in deserialize.
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.7071 SLOPE_WARMDOWN=0.5
export SEED=42 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0
export RUN_ID="042D-eval-only-gptq-fix-verify"

# THE NEW FLAG
export EVAL_ONLY=1

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/042D-eval-only-gptq-fix-verify

# torchrun with 1 rank — eval_only doesn't benefit from sharding.
# Run from the directory containing the existing final_model.{pt,int6.ptz}
# so the script finds them via the relative-path defaults.
cd /workspace/parameter-golf

torchrun --standalone --nproc_per_node=1 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/042D-eval-only-gptq-fix-verify/train.log 2>&1
```

## What to harvest

1. **Confirm the new code paths fire:**
   - `eval_only=1: skipping train_model, loading existing artifacts from ...`
   - `deserialize: slope synced to warmdown value 0.5000`
   Both must appear in the log. If either is missing, the fix didn't run.

2. **Quantized val_bpb** — the headline number:
   - `diagnostic quantized val_loss:X val_bpb:Y`
   - Compare to:
     - 042A original (broken): **2.0503**
     - Baseline 039b (clean quant): **1.0741**
     - Baseline pre-quant EMA: 1.06514
     - Baseline quant cost: +0.0090
   - Expected with fix: pre-quant EMA was 1.06661, so quant cost ~+0.009 → **target ~1.075**

3. **No artifact-not-found error** — if the file path is wrong, the script
   raises `FileNotFoundError`. Means we need to fix the path and rerun.

4. **No state_dict shape mismatch** — confirms architecture env vars match
   the original 042A run.

## Acceptance

- **PASS** (fix confirmed): quantized val_bpb in [1.07, 1.08]. Quant cost
  (vs pre-quant EMA 1.06661) within ~0.01-0.02. Original 042A artifacts are
  recovered as a usable submission candidate.
- **PARTIAL** (fix helped but not fully): quantized val_bpb in [1.08, 1.20].
  Slope sync helped but other quantization issues remain. Investigate.
- **FAIL** (fix had no effect): quantized val_bpb still ≥ 2.0. Different
  root cause; need to dig further.

## After this run

If PASS:
- 042A submission candidate is recoverable from the existing artifacts (no
  retrain needed).
- Mark 042A eval as superseded by 042D; the *real* 042A result is
  pre-quant EMA 1.06661 + quantized ~1.075.
- Decision: this is still in the noise zone vs baseline. Run a second seed
  if we want to confirm; otherwise close out the 042 arc.

If FAIL or PARTIAL:
- The deserialize fix isn't the full story. Look at EMA next (user mentioned
  it might also be wrong) or other state mismatches.

## Cost

~$0.25 (1×H100 JP, ~5 min including pod startup).
