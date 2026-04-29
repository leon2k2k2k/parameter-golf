# Idea — Per-document LoRA TTT (no global SGD), from PR #1915

**Date:** 2026-04-29
**Source:** PR #1915 (AidenGeunGeun), opened 2026-04-29.
**Claimed:** 3-seed mean **1.06504 BPB** on a CaseOps + LQER stack (no MIN_LR, no #1797-class TTT).

## What's new (vs our phased-TTT baseline)

The PR strips eval-time TTT down to a different recipe:

1. **No global SGD pass.** All TTT updates are per-document only.
2. **`TTT_WARM_START_A=0`** — LoRA-A starts at zero rather than identity, so the first forward is the un-adapted predictor.
3. **LoRA reset between documents** — each doc starts from the post-train state, not from the LoRA produced by the previous doc.
4. **Physical length bucketing across independent docs** — to fill batch lengths without any cross-document state contamination.
5. Score-first ordering retained (same as legal-TTT family).

The combination is meant to be a "minimum-legal" TTT discipline: every adaptation is provably document-local and provably gradient-only, no information leak between docs.

## Why this matters strategically (independent of the BPB number)

The author's headline (1.06504) sits between our baseline (1.06549) and the current likely-legal frontier (#1797 @ 1.06157). On its own this is not a frontier-mover.

But it's a useful **fallback discipline** if the #1797 / #1855 family of TTT methods continues to attract validity disputes. cocohearts has already flagged #1797/#1787/#1801 as validity-pending. If those rulings go against the multi-phase-global-SGD class, we'd want a "definitely legal" TTT recipe to fall back to. PR #1915's per-doc-only protocol is one such candidate.

## Levers that aren't in our baseline

- `TTT_WARM_START_A=0` (vs current default identity init)
- `PHASED_TTT_GLOBAL_SGD_OFF` (we always run global-SGD phases)
- Physical doc-length bucketing (we batch by token count, not doc count)

## Suggested next step

Don't spec yet. Park as a known fallback. Re-evaluate after the Issue #1872 / cocohearts validity round resolves. If global-SGD TTT is ruled out, this becomes the spec base. If global-SGD survives, kill this idea.

## Cost if specced later

~$3 for a single 8×H100 screening run with three env-var changes — no code.
