# Investigation — Why `043A` diverged from `038A`

## Update — 2026-04-25: 4H/8H tracking confirmed on clean code

The 039b baseline run (4×H100, 1200s, fixed commit `5bbf12f`) produced:

- pre-quant post-EMA val_bpb: **1.06513927**

vs. the real 038A (8×H100, 600s):

- pre-quant post-EMA val_bpb: **1.06541920**

Delta: **0.00028** — essentially perfect. The 4H/20min rung faithfully tracks
the 8H/10min rung on fixed code. The 043A divergence was entirely due to the
launch contract bug and the backward bug, not a real 4H/8H regime mismatch.
The 043A divergence was entirely due to the wrong activation backward formula
(`2·s·x` instead of `2·s²·x`) — not a real 4H/8H regime mismatch, and not
just a launch-contract issue. The 043B spec is also answered by this run — no
separate replay needed.

---

## Bottom line

`043A` was **not** a faithful `038A` same-rung control.

The main failure happened at the **launch contract**, not in the loop math:

- `038A` was launched from a validated script that **literally inherited** the
  full `036` seed-42 config and then overrode only the intended `038` diffs.
- `043A` was launched from the `039b` code line with a **hand-written partial
  env file**.
- That means `043A` did **not** carry forward the real hidden non-defaults and
  branch behavior that the actual `038A` run used.

So the `043A` vs `038A` mismatch is not strong evidence that “4H recurrence
tracking broke” or that “the loop itself is bad now.” It is strong evidence
that `043A` was under-specified.

## Exact place it went wrong

Compare the actual `038A` launch script:

- [launch_038a_seed42.sh](/home/claude-user/ai-workspace/projects/parameter-golf/tmp_exec/launch_038a_seed42.sh)

with the `043A` launcher:

- [043-038a-4h-20m-baseline.sh](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/039b-loop-band-activation/research/specs/043-038a-4h-20m-baseline.sh)

`038A` did this:

- loaded the validated `036` config from `runs/036-035e-8h-promotion/seed_42/config.json`
- replayed it key-for-key into env vars
- then overrode only:
  - `RUN_ID`
  - `SMEAR_GATE_ENABLED`
  - `GATE_WINDOW`
  - `LQER_*`

`043A` did this instead:

- sourced a manually written `.env`
- launched from the `039b` branch/worktree

That manual `.env` only captured the obvious top-line knobs. It did **not**
faithfully reproduce the original `038A` launch contract.

## Concrete config drift

From the real `038A` log:

- [038A train.log](/home/claude-user/ai-workspace/projects/parameter-golf/runs/038-smear-lqer-asym-8h/seed_42/train.log)

the following hyperparameters were non-default / materially pinned:

- `fused_ce_enabled=True`
- `ttt_lora_alpha=144`
- `ttt_lora_rank=96`
- `ttt_weight_decay=1.0`
- `phased_ttt_num_phases=3`
- `phased_ttt_prefix_docs=2000`
- `global_ttt_lr=0.001`
- `global_ttt_epochs=1`

In `043A`:

- these were **not** materialized in the env file
- so they fell back to whatever the `039b` code line defaults were

Some of these do not matter when `TTT_ENABLED=0`, but the larger point is that
the baseline was not reproduced by construction.

## Structural regime drift

Both `038A` and `043A` use:

- `grad_accum_steps = 8 // world_size`

So:

- `038A` on `8H` ran with `grad_accum_steps=1`
- `043A` on `4H` ran with `grad_accum_steps=2`

This is a real regime difference. However, older recurrence-family `4H` vs `8H`
runs tracked each other reasonably well despite the same structural difference,
so this alone does **not** explain the `043A` blow-up.

It is a secondary source of mismatch, not the primary one.

## Why the divergence shows up around the loop

The visible failure centered around the loop-on region:

- real `038A`: `layer_loop:enabled step:2209 frac:0.350`
- `043A`: nominally also `ENABLE_LOOPING_AT=0.35`

This does **not** mean the loop code itself was wrong.

More likely:

- the under-specified `043A` control reached the loop boundary in a different
  optimizer/state regime than real `038A`
- once recurrence turned on, that mismatch became amplified

So the loop region is where the baseline mismatch became obvious, not
necessarily where it originated.

## Comparison with older recurrence lines

This is why the result looked suspicious:

- older recurrence-family `4H` and `8H` runs *did* track reasonably well
- for example `032 4H` vs `036/038 8H` stayed fairly aligned on matched-step
  train loss

That historical behavior argues against a simple story like:

- “4H never tracks 8H”

and supports the actual diagnosis:

- `043A` was not a clean reproduction of `038A`

## What should happen next

To get a real same-rung `038` control:

1. Launch from the **actual `038` code line**, not the `039b` code line.
2. Reuse the **full validated `038A` launch contract**, not a hand-written
   partial env.
3. Change only:
   - `nproc_per_node: 8 -> 4`
   - `MAX_WALLCLOCK_SECONDS: 600 -> 1200`
   - `TTT_ENABLED: 1 -> 0`
   - `TRAINING_ONLY_SCREEN: 1`

Only after that rerun exists should `042C` be judged against a “same-rung
038 baseline.”

## Decision

- Do **not** trust the current `043A` run as the `038A` control.
- Do **not** interpret its loop-region divergence as direct evidence against the
  recurrence stack.
- Treat this as a launch-contract / baseline-reproduction bug.
