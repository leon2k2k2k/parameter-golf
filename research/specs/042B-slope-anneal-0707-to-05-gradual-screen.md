# Spec 042B — pre-warm speedup smoke (3 min)

**Slug:** `042B-slope-anneal-0707-to-05-gradual-screen` (slug retained for git history; this is no longer the gradual-anneal spec)
**Created:** 2026-04-26
**Status:** SMOKE / DIAGNOSTIC — 3-min run to measure pre-warm speedup from compile env vars
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `aff2de4`
**Links to:** `research/ideas/slope-annealing-sqrt-to-half.md`
**Predecessor:** `runs/042A-recompile-smoke/notes.md`

> **Note:** the original gradual-anneal blueprint that lived here is parked
> until a continuous-slope code path lands (see git history for the prior
> spec body). This 042B has been repurposed as a fast diagnostic for pre-warm
> compile speed.

## Why this smoke

The first 042A smoke (commit `dd63a75`) burned ~10-15 min in pre-warm because
`cache_size_limit=8` caused recompile loops. After fixing that (`4b29c64`,
limit=32) and adding forward_logits/looping-True coverage in `aff2de4`, the
executor estimates the full 042A pre-warm at **~15 min**. That's a lot of pod
cost (~$3 at 4×H100, $1.50 at 2×H100) before training even starts.

This spec tests two compile-speed env vars to see if pre-warm can be cut to
~4-6 min:

```bash
export TORCHINDUCTOR_COMPILE_THREADS=8     # parallel Inductor lowering across kernels
export TRITON_AUTOTUNE_NUM_RUNS=1          # 1 autotune sample instead of default ~5
```

Both are env-var-only, no code changes, and don't affect training correctness.
`TRITON_AUTOTUNE_NUM_RUNS=1` may yield kernels ~5-10% slower at runtime
(autotune picks the first config that compiles instead of the fastest), but
that's tolerable for a smoke and likely tolerable for the full run too.

## What we want to measure

After the run, compute pre-warm time = (time of first `warmup_step: 1/20`
log line) - (time of `Hyperparameters:` line at startup). Compare to:

| Run | Pre-warm time | Notes |
|---|---|---|
| 042A smoke (`dd63a75`, no env tweaks) | ~10-12 min | cache_size_limit=8 caused recompile flood |
| 042B smoke target (this run) | < 6 min | hopeful, with env tweaks |
| 042A full (`aff2de4`, no env tweaks) | ~15 min (executor estimate) | full pipeline |

Other things to capture:
- Total recompile events from `TORCH_LOGS=recompiles` — should be 0 mid-training
- Kernel runtime: tok/s post-warmup. If `TRITON_AUTOTUNE_NUM_RUNS=1` makes
  kernels measurably slower, we'll see lower tok/s than the 042A smoke's
  ~2.19M tok/s.

## Regime

- **2×H100 JP**, `SEED=42`, `MAX_WALLCLOCK_SECONDS=180`, `TTT_ENABLED=0`,
  `TRAINING_ONLY_SCREEN=1`
- Compressed schedule: loop activates ~50s in, slope switches ~50s in
- Cost: ~$0.50 (~5 min wall × $5.98/hr)

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
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.30
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=20
export ITERATIONS=20000 WARMDOWN_FRAC=0.72 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=999999
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
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.7071 SLOPE_WARMDOWN=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=180 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=1
export RUN_ID="042B-prewarm-speedup-smoke"

# Pre-warm speedup tweaks
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_AUTOTUNE_NUM_RUNS=1

# Diagnostic logging — confirm 0 mid-training recompile events
export TORCH_LOGS=recompiles

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/042B-prewarm-speedup-smoke

torchrun --standalone --nproc_per_node=2 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/042B-prewarm-speedup-smoke/train.log 2>&1
```

## What to harvest

1. **Pre-warm time** — timestamp(`warmup_step: 1/20`) − timestamp(`Hyperparameters:`).
2. **Slope pre-warm fired** — confirm log line:
   `slope_anneal: precompiled warmdown kernel slope=0.5000`
   appears at startup. Without this, the warmdown kernel was never compiled
   and the slope switch will hit a cold compile.
3. **Slope switch actually fired during training** — confirm log line:
   `slope_anneal: 0.7071→0.5000 step:N frac:0.XXX`
   appears at frac ≈ 0.28 (elapsed ~50s). If missing, the trigger condition
   in the training loop didn't fire — investigate before doing anything else.
4. **Loop activated** — confirm log line:
   `layer_loop:enabled step:N frac:0.XXX depth:3 encoder:[...] decoder:[...]`
   appears at frac ≈ 0.30 (elapsed ~54s). Same diagnostic value as #3.
5. **No mid-training pause at the switch step** — step rate stays steady before
   and after both events. From `train_loss` log lines (every 20 steps), compute
   step delta; should be roughly constant around 1-2s/step. A 30s+ gap at the
   slope switch or loop activation = the pre-warm didn't cover that variant
   and we still have a recompile bug.
6. **Recompile count from TORCH_LOGS** — `grep -cE "Recompiling|cache_size_limit|guard failed" train.log`
   → should be 0 mid-training (any pre-warm-time recompiles are fine).
7. **Throughput** — average tok/s after the loop activation. Compare to 042A
   smoke's ~2.19M (post-loop) and 042A original's ~1.7M (post-loop on 4×H100,
   so 2×H100 should be ~50-60% of that). Drop > 5% from the smoke baseline
   suggests `TRITON_AUTOTUNE_NUM_RUNS=1` is too aggressive.

## Acceptance

This is a diagnostic smoke, not judged on val_bpb.

- **Strong win**: pre-warm < 5 min, 0 mid-train recompiles, throughput drop < 5%
  → adopt the env tweaks for full 042A and any future spec
- **Weak win**: pre-warm 5-8 min → still adopt; small but real savings
- **No win**: pre-warm ≥ 10 min OR mid-train recompiles → revert env tweaks,
  accept the long pre-warm
- **Regression**: throughput drops > 10% → revert `TRITON_AUTOTUNE_NUM_RUNS=1`,
  keep only `TORCHINDUCTOR_COMPILE_THREADS=8`

## After this run

If the env tweaks work, add them to the full 042A spec and any future specs in
this branch. Keeping them at the spec level (not committed code) so we can
toggle without a re-spec.
