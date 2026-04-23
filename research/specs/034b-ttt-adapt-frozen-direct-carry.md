# Spec 034b — TTT-only adaptation of frozen direct-carry from `034`

**Slug:** `ttt-adapt-frozen-direct-carry`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/034b-ttt-adapt-frozen-direct-carry`
**Commit:** `<PINNED_COMMIT>`
**Links to:** `research/ideas/034b-ttt-adapt-frozen-direct-carry.md`, `research/specs/034-frozen-direct-carry-from-031a.md`, `research/specs/033-ttt-adapt-alpha-beta.md`, `research/specs/033b-ttt-adapt-alpha-beta-high-lr.md`

## Hypothesis

If `034` yields a healthy frozen direct-carry checkpoint, then letting TTT make
small adjustments to that richer carry object may help more than the old
alpha/beta TTT line did.

This is specifically a **TTT-only** test, not a retrain.

## Base checkpoint

Base artifact is the saved float checkpoint from `034`:

```text
/workspace/runs/034-frozen-direct-carry-from-031a/seed_314/final_model.pt
```

## Comparison target

Primary comparison:

- the normal post-TTT result from `034`

Interpretation:

- if `034b` > `034`, then TTT wants to refine direct-carry
- if `034b` ~= `034`, then frozen direct-carry is already good enough
- if `034b` < `034`, then TTT carry adaptation is harmful or unnecessary

## Mechanism

Keep the same frozen direct-carry base as `034`.

During TTT only:

- load the frozen direct-carry values from the `034` checkpoint
- make them trainable
- include them in a separate TTT optimizer param group

Pinned trainable tensors:

- `direct_carry_self_frozen`
- `direct_carry_edges_frozen_pass1`
- `direct_carry_edges_frozen_pass2`

## Initial LR stance

Pinned first probe:

- `TTT_DIRECT_CARRY_ENABLED=1`
- `TTT_DIRECT_CARRY_LR_SCALE=0.25`

Interpretation:

- LoRA keeps the normal `TTT_LORA_LR`
- direct-carry uses `0.25x` of that LR

Start conservative. Do not jump straight to an aggressive carry LR.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout <PINNED_COMMIT>

# Sanity verify
grep -n "TTT_DIRECT_CARRY_ENABLED" train_gpt.py
grep -n "TTT_DIRECT_CARRY_LR_SCALE" train_gpt.py
grep -n "ttt_direct_carry:" train_gpt.py
grep -n "frozen_edge_self" train_gpt.py

mkdir -p /workspace/runs/034b-ttt-adapt-frozen-direct-carry/seed_314
mkdir -p /tmp/torch_inductor_cache_034b

SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/workspace/runs/034-frozen-direct-carry-from-031a/seed_314/final_model.pt \
ARTIFACT_DIR=/workspace/runs/034b-ttt-adapt-frozen-direct-carry/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_034b \
DATA_DIR=/workspace/parameter-golf/data \
CASEOPS_ENABLED=1 \
DIRECT_CARRY_MODE=frozen_edge_self \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
TTT_DIRECT_CARRY_ENABLED=1 TTT_DIRECT_CARRY_LR_SCALE=0.25 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=16 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 spinquant_hotstart.py \
  > /workspace/runs/034b-ttt-adapt-frozen-direct-carry/seed_314/ttt.log 2>&1
```

## Required logging

Must log:

- `ttt_direct_carry: enabled=1 ...`
- `ttt_direct_carry: before_self=...`
- `ttt_direct_carry: before_edges_pass1=...`
- `ttt_direct_carry: before_edges_pass2=...`
- live snapshots during phased TTT, for example:
  - `ttt_direct_carry: live_b17_self=...`
  - `ttt_direct_carry: live_b17_edges_pass1=...`
  - `ttt_direct_carry: live_b17_edges_pass2=...`
  - `ttt_direct_carry: live_b17 ..._max_drift=...`
- `ttt_direct_carry: after_self=...`
- `ttt_direct_carry: after_edges_pass1=...`
- `ttt_direct_carry: after_edges_pass2=...`
- `ttt_direct_carry: after ..._max_drift=...`

## Hotstart validation contract

Use the same style of hotstart validation as the `033` family:

- verify the checkpoint path exists
- verify the branch/commit contains the TTT direct-carry path
- verify the normal hotstart diagnostics are sane before interpreting TTT delta

Exact expected baseline values should be filled from the actual `034` result once
that run completes.

## Stop-early criteria

- NaN in TTT loss → halt
- missing `ttt_direct_carry:` logs → halt
- checkpoint path missing or wrong → halt
- post-TTT clearly worse than the corresponding `034` result → flag hard

## Execution order

Only run after:

- `034` completed
- the `034` checkpoint path exists
- the base result is healthy enough to justify refinement
