# Spec 034 — Frozen direct-carry from `031A`

**Slug:** `frozen-direct-carry-from-031a`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/034-frozen-direct-carry-from-031a`
**Commit:** `7c1b390`
**Links to:** `research/ideas/034-frozen-direct-carry-from-031a.md`, `research/specs/031-direct-carry-freefloat-neutral.md`, `research/specs/025c-cross-layer-carry-frozen-per-pass.md`, `research/specs/030-025b-seed314-new-ttt.md`

## Hypothesis

`031A` learned a meaningful per-pass direct-carry structure in the clean 2-loop
regime. If we freeze that learned structure as buffers and run a live-like 4×H
proxy, it should hold up better than older frozen carry baselines.

This is a deployment-style follow-up, not another calibration run.

## Source snapshot

Freeze from the healthy `031A-ratio0272` snapshot at `val_step_4000`.

Pinned frozen values:

```text
self =
[[0.92578125, 1.5390625, 1.921875],
 [2.0,       2.0,       1.4296875]]

edges_pass1 =
[[ 0.349609375,   0.06005859375,  0.0615234375 ],
 [ 0.1337890625, -0.369140625,   -0.04150390625],
 [-0.0247802734375, 0.3828125,   -0.353515625  ]]

edges_pass2 =
[[ 0.55859375,   -0.03515625,    0.3046875,     -0.59765625,   -0.027099609375,  0.0162353515625 ],
 [ 0.036376953125,-0.28125,     -0.1123046875,   0.30078125,   -0.251953125,     0.0164794921875 ],
 [ 0.052001953125, 0.1103515625,-0.01336669921875,0.0576171875, 0.2353515625,    -0.07275390625   ]]
```

## Storage semantics

Store the frozen direct-carry tensors as `register_buffer(...)`, not parameters.

Pinned objects:

- `direct_carry_self_frozen` shape `[2, 3]`
- `direct_carry_edges_frozen_pass1` shape `[3, 3]`
- `direct_carry_edges_frozen_pass2` shape `[3, 6]`

Requirements:

- not trainable
- not in optimizer param groups
- serialized in `final_model.pt`
- logged verbatim during training so execution can confirm they remain fixed

## Comparison targets

Primary practical target:

- competitive end-to-end result against the strong frozen carry line after TTT

Secondary structural comparison:

- does this frozen direct-carry line look better than older frozen carry baselines
  in the same 4×H proxy class?

## Regime

Pinned intent:

- `NUM_LOOPS=2`
- detached carry behavior preserved
- live-like `4×H100` proxy run
- full pipeline:
  - train
  - pre-quant diagnostic
  - quantized diagnostic
  - quantized phased TTT with the newer 3-phase path

## Timing

This spec is meant to proxy the live-style pacing, not the extended calibration
regime.

So use:

- `MAX_WALLCLOCK_SECONDS=1200`
- `ENABLE_LOOPING_AT=0.35`

This is intentional here.

Do **not** reuse the `031A-ratio0272` onset correction for this run, because that
correction was only needed to preserve step-space onset under a longer effective
wallclock.

## Checkpoint requirements

Must preserve persistent artifacts under `/workspace/runs/...`, including:

- `final_model.pt`
- `final_model.int6.ptz`
- training log
- diagnostics

This is already supported by the current training code when `ARTIFACT_DIR` is
set, but the spec requires it explicitly because we may want to reuse the float
checkpoint later.

Important correction:

- `TTT_ENABLED=1` alone is not sufficient to match the newer phased TTT path.
- In this codepath, `PHASED_TTT_NUM_PHASES` defaults to `1`.
- So this spec must explicitly set:
  - `PHASED_TTT_PREFIX_DOCS=2000`
  - `PHASED_TTT_NUM_PHASES=3`

If a prior `034` training rung completed with inline TTT but omitted those
envs, then its post-TTT result is not the intended `034` result. In that case,
do not retrain solely for TTT. Reuse the saved `final_model.pt` and rerun the
post-training quantized phased-TTT stage through `spinquant_hotstart.py`, the
same way `034b` consumes a saved checkpoint.

## Hardware ladder

1. `4×H100`, `1200s`, full pipeline

This spec is already close to a decision rung, so no separate reduced smoke rung
is required if the code patch is small and obvious.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 7c1b390

# Sanity verify
grep -n "frozen_edge_self" train_gpt.py
grep -n "direct_carry_self_frozen" train_gpt.py
grep -n "direct_carry_edges_frozen_pass1" train_gpt.py
grep -n "direct_carry_edges_frozen_pass2" train_gpt.py

mkdir -p /workspace/runs/034-frozen-direct-carry-from-031a/seed_314
mkdir -p /tmp/torch_inductor_cache_034_4h

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/034-frozen-direct-carry-from-031a/seed_314/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/parameter-golf/data \
ARTIFACT_DIR=/workspace/runs/034-frozen-direct-carry-from-031a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_034_4h \
CASEOPS_ENABLED=1 \
TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
DIRECT_CARRY_MODE=frozen_edge_self \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/034-frozen-direct-carry-from-031a/seed_314/train.log 2>&1

kill $NVSMI_PID
```

## Required logging

Must log the frozen direct-carry tensors repeatedly during training:

- frozen `self`
- frozen `edges_pass1`
- frozen `edges_pass2`
- `diagnostic pre-quantization post-ema`
- `diagnostic quantized`
- `quantized_ttt_phased`

And the log should make it obvious that:

- values remain fixed
- no optimizer drift occurs

## Accept criteria

Strong success:

- competitive post-TTT result
- no throughput/pathology surprise
- frozen direct-carry looks like a real promotion candidate

Weak success:

- base/quantized path is healthy, but TTT result is only neutral

Failure:

- frozen direct-carry underperforms the relevant frozen carry baseline
- or the richer `031A` structure does not survive freezing well

## Open questions

- whether to compare primarily against `025c`-style frozen baselines or directly
  against the stronger modern `8×H` post-TTT line in the writeup
- whether a shared compression of this direct-carry object is worth trying later
  if the native frozen version works
