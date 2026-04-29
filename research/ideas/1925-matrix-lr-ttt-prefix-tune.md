# Idea — Port PR #1925: MATRIX_LR + PHASED_TTT_PREFIX_DOCS hyperparam tweak on #1855

**Date:** 2026-04-29
**Source:** PR #1925 (simon-marcus), opened 2026-04-29.
**Claimed:** 3-seed mean **1.06109 BPB** vs #1855 base 1.06108 (essentially neutral on author's own report, but below our baseline #1736 @ 1.06549 and below the likely-legal frontier #1797 @ 1.06157).

## What changes

Hyperparam-only on the #1855 stack:

- `MATRIX_LR=0.028` (was 0.026)
- `PHASED_TTT_PREFIX_DOCS=3500`

That's it. No code. No checkpoint shape change. No tokenizer change.

## Why it might help (and why it might not)

The PR's headline number is essentially a tie with #1855 (1.06109 vs 1.06108), so this is *not* a frontier-mover on the author's own stack. The reason it's worth a quick spec is:

1. **Hyperparam-only.** Cost to test = single 8×H100 run (or 2 seeds for noise) on our existing pinned baseline commit. ~$3.
2. **Both knobs are independently plausible.** `MATRIX_LR` was the lever that opened up #1586 → #1626 (~−0.003 BPB), and we've never re-tuned it after stacking CaseOps + gates. `PHASED_TTT_PREFIX_DOCS` controls how much of the val prefix the phased TTT consumes — increasing it gives TTT more context before quant, at no eval-budget cost.
3. **Compounds with our quant repair work.** If 060A-class repair gives us another small wedge of BPB to recover from quantization, more TTT prefix may amplify it.

The risk is that simon-marcus's stack (#1855 base) has different blend algebra than our exp/046-quant-repair branch — so the hyperparam optimum may not transfer.

## Suggested test

- Single run on current research baseline commit at 8×H100 with the two env-var changes.
- Compare step-matched val_bpb at the matched stop step against the most-recent matching baseline.
- Kill-criterion: if delta is +0.0005 BPB or worse vs that baseline at midpoint, abort.

Estimated cost: ~$3 for single run, ~$10 for 3-seed if signal is positive.

## What this is not

This is a hyperparam refresh, not a research lever. If it doesn't help on our stack, no follow-up — drop and move on.
