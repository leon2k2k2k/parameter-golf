# Spec 043B — Faithful `038A` replay on `4×H100`, `1200s`, training-only

**Slug:** `038a-faithful-4h-20m-replay`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `1a7a814`
**Links to:** `research/specs/043-038a-4h-20m-baseline.md`, `research/evaluations/043a-vs-038a-investigation.md`

## Hypothesis

`043A` was invalid as a same-rung `038A` control because it was launched from
the `039b` code line with a hand-written partial env.

`043B` fixes that by replaying the **actual `038A` launch contract** from:

- [runs/038-smear-lqer-asym-8h/seed_42/config.json](/home/claude-user/ai-workspace/projects/parameter-golf/runs/038-smear-lqer-asym-8h/seed_42/config.json)

and changing only the rung-control variables.

## Baseline

Real reference run:

- `038A`
- `8×H100`
- `600s`
- pre-quant post-EMA `val_bpb = 1.06541920`

Expected tracking behavior from historical recurrence families:

- a faithful `4H/1200s` replay should land **close** to the `8H/600s` line
- not identical, but in the normal historical drift band (`~<=0.002` BPB)

## Execution contract

Replay the full `038A` config key-for-key from:

- `runs/038-smear-lqer-asym-8h/seed_42/config.json`

Use the **actual `038` code line**:

- remote branch: `fork/exp/038-fullfloat-smear-lqer-asym`
- pinned code commit: `c8620b68fc6c53c9cf8c0a28a05587bd4fdca9ea`

Override only:

- `RUN_ID=043B-038a-faithful-4h-20m-replay`
- `MAX_WALLCLOCK_SECONDS=1200`
- `TTT_ENABLED=0`
- `TRAINING_ONLY_SCREEN=1`
- `torchrun --nproc_per_node=4`

Do **not** hand-copy or simplify the rest of the env surface.

## Regime

- `4×H100`
- `1200s`
- `SEED=42`
- training-only
- stop after pre-quant post-EMA diagnostic

## Acceptance

Primary pass condition:

- `043B` tracks `038A` closely enough to be a credible same-rung control

Practical target:

- pre-quant post-EMA within roughly `0.002` BPB of `038A`

Failure:

- large mismatch similar to `043A`

If it fails again, that points away from a simple launch-contract bug and toward
a deeper `4H` recurrence-regime issue.

## Monitoring

Primary comparison:

- `043B` vs `038A`

Secondary:

- `043B` vs broken `043A`

The point is to answer one question cleanly:

- was `043A` just a bad reproduction, or does `038A` genuinely fail to track on
  the `4H/1200s` rung?
