# Spec 060L — PHASED_TTT_PREFIX_DOCS 2500 → 3000 on 060A baseline (eval-only)

**Date:** 2026-04-29
**Slug:** `060L-ttt-prefix-up`
**Idea:** `research/ideas/ttt-budget-reinvestment.md`
**Branch:** `research` (config-only; no code change)
**Pinned SHA:** `da50cd6` (060A's commit)

## Hypothesis

PR #1855's greedy hyperparameter search (codemath3000) explicitly raised `PHASED_TTT_PREFIX_DOCS` from #1851's **2000 → 2500** and reported a real Δ on this stack — which means the prefix-doc → bpb gradient was negative (more prefix helps) at the 2000→2500 step on the #1855 stack. **#1855 stopped greedy at 2500; 3000+ is untested.**

This spec extends the same lever +20%: 2500 → 3000. Cost: ~+50-80s eval (one-shot prefix forward, scales linearly with docs). Predicted **−0.0003 to −0.0010 BPB** if the 2000→2500 trend continues; ~0 if 2500 was already at saturation.

**Why conservative (3000, not 4000):** larger jumps risk blowing the 600s eval cap on the slowest seed (139s headroom on #1855 seed 42), and we have no measurement past 2500 on this stack — better to take a small step in a known-good direction first, then escalate.

## Baseline

060A (`seed_42_4h`, 4H matched-FLOPs).

## Expected Δ

**−0.0003 to −0.0010 BPB** post-TTT. (Higher confidence than 060J/K because the 2000→2500 step was greedy-validated on this exact stack by #1855.)

## Accept criteria

- post-TTT val_bpb ≤ (060A_seed_42_4h post-TTT − 0.0002)
- eval wallclock ≤ 600s
- TTT loss curve smoother (less per-batch noise) than 060A
- artifact size unchanged

## Config diff vs 060A

```
PHASED_TTT_PREFIX_DOCS: 2500 → 3000
```

All other TTT knobs unchanged.

**Phase 2 (separate spec, conditional):** if 060L wins, run 060L-aggressive at PREFIX_DOCS=4000 to find the saturation point.

## Code changes

None.

## Hardware ladder

- 4×H100 eval-only; ~8-12 min wall, ~$1-2.

## Seed plan

Single seed (42).

## Inputs

Same as 060J. Parent ckpt: `/workspace/runs/060A-1855-port/seed_42_4h/final_model.pt`.

## Checkpoints to emit

- `runs/060L-ttt-prefix-up/seed_42/final_model.int6.ptz`
- `runs/060L-ttt-prefix-up/seed_42/eval.log`

## Stop-early criteria

- eval wallclock projects > 600s → kill
- post-TTT val_bpb > 060A → kill
- TTT data-source error (e.g., not enough prefix docs in val stream) → kill, fall back to 3000

## Cost estimate

~$2.

## Run command

```bash
SEED=42 RUN_LABEL=seed_42 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42_4h/final_model.pt \
  ARM=060L-ttt-prefix-up \
  PHASED_TTT_PREFIX_DOCS=3000 \
  bash tmp_exec/launch_060_eval.sh
```

## Extra artifacts

None beyond defaults.

## Open questions for interview

1. **Source of extra prefix:** verify train_gpt.py's TTT path can pull beyond 2500 docs from val without leaking into the eval set itself.
2. **Per-phase split or shared:** if all 3 phases each use 4000 docs (shared), prefix forward grows ~+60%; if disjoint slices (12000 docs total), more like 3× — bust budget. Read code first.
3. **Bigger jumps:** if 4000 wins, 6000 next? Memory/wallclock implications.

## Phase 2

If 060L wins: stack with 060J (phases up) → "more phases × more prefix" combo, eval ~+130s. Verify cap on 3-seed cap-safety screen.
