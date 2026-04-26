# Spec 042A — recompile-cause smoke (3 min)

**Slug:** `042A-slope-anneal-0707-to-05-screen`
**Created:** 2026-04-26
**Status:** SMOKE / DIAGNOSTIC — 3-min run to identify the recompile cause
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `dd63a75`
**Links to:** `research/ideas/slope-annealing-sqrt-to-half.md`

## Why this is a smoke

First 042A run (commit `2593982`) had a **~7-8 minute compile pause** between
step 1623 (slope switch) and step 1700, where the slope kernel switch
(0.7071→0.5) and the loop activation (looping_active=False→True at step 1625)
fired within 2 steps of each other. Training got cut to step 3034/20000 by the
1196s wallclock cap. Pre-quant EMA val_bpb landed at **1.0886** at step 3034 —
only ~0.024 worse than the fully-trained 039b baseline (1.06514) over 6× fewer
steps. **The early-training signal is real.** We need to fix the recompile to
realize it.

This smoke run answers: **what causes the 7-8 min pause?** Three suspects:
1. **Dynamo retrace** — `torch.compile(dynamic=False, fullgraph=True)` +
   flipping the Python attribute `looping_active` at step 1625 changes the
   for-loop iteration count (11 → 17 layers); Dynamo throws away the cached
   graph and re-traces.
2. **Triton autotune** — bucket warmup may not have covered the cu_seqlens
   shape that the first real loop-active batch encountered.
3. **Cache eviction** — Dynamo `cache_size_limit=8` (default) may LRU-evict
   the looping_active=True graphs during the 1600 pre-loop steps.

## What this run does

Compresses the run so the trigger events fire in the first ~60s:

| Knob | Original 042A | Smoke 042A | Why |
|---|---|---|---|
| `MAX_WALLCLOCK_SECONDS` | 1200 | **180** | 3-min total run |
| `ENABLE_LOOPING_AT` | 0.35 | **0.30** | Loop fires at ~54s elapsed |
| `WARMDOWN_FRAC` | 0.75 | **0.72** | Slope switch at frac=0.28 (~50s) |
| `TRAIN_LOG_EVERY` | 100 | **20** | Granular step timing around the event |
| `VAL_LOSS_EVERY` | 1000 | **999999** | Skip val — not measuring quality |
| `TRAINING_ONLY_SCREEN` | 0 | **1** | Skip EMA/GPTQ/quant/sliding/TTT |

Plus diagnostic env vars:
```bash
export TORCH_LOGS=recompiles,dynamo,inductor
export TORCHDYNAMO_VERBOSE=1
```

`TORCH_LOGS=recompiles` prints a line every time Dynamo invalidates a cache
entry, with the exact guard that fired. `dynamo,inductor` add tracing &
lowering events. **This will dump a lot of log volume** — that's the point.

## Regime

- **2×H100, JP region** (cheap; we don't need 4×H100 for a 3-min smoke; full
  spec was 4×H100 but the recompile is per-rank so 2×H100 is sufficient
  to see it)
- `SEED=42`, `MAX_WALLCLOCK_SECONDS=180`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=1`

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
export RUN_ID="042A-recompile-smoke"

# Diagnostic logging — captures the recompile reason at the loop activation
export TORCH_LOGS=recompiles,dynamo,inductor
export TORCHDYNAMO_VERBOSE=1

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/042A-recompile-smoke

torchrun --standalone --nproc_per_node=2 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/042A-recompile-smoke/train.log 2>&1
```

## What to harvest after the run

The log will be huge. After it finishes, harvest:

1. **Recompile events** — `grep -E "Recompiling|Cache miss|guard failed|GuardManager" train.log | head -50`
2. **Lines around loop activation** — `grep -B3 -A30 "layer_loop:enabled" train.log`
3. **Triton autotune events** — `grep -E "AUTOTUNE|autotuning" train.log | wc -l` (count) and `head -20`
4. **Step timing around the events** — `grep "train_loss" train.log` (cadence is every 20 steps)

## Acceptance

This run is **not** judged on val_bpb. Success = we identify which of the three
suspects causes the pause. Failure = logs are inconclusive → need finer
instrumentation (single-step repro or Dynamo's `torch._dynamo.config.verbose=True`).

## After this run

Next spec applies the targeted fix (e.g. tensor-gate `looping_active`, or
bump `cache_size_limit`, or pre-warm a wider cu_seqlens shape range) and
re-runs the full 042A.

## Cost

2×H100 JP × ~5 min wall (including SSH setup, pod boot, install, launch, run,
shutdown) ≈ **$0.50**.
