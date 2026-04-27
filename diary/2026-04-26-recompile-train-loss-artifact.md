# 2026-04-26 — research methodology lesson: post-recompile train_loss is unreliable

## What happened

Spent considerable analysis time on the "broken 042A had 0.10–0.13 train_loss
advantage over baseline at steps 1700–3000" finding. We attached the gap to
"earlier loop activation" because the recompile pause shifted loop activation
to step 1625 instead of 2289. This led us to spec 043A as a clean test of
"early loop activation" on the canonical 3-5 band.

When 043A actually ran — clean, no recompile, loop activated at step ~1333
(frac=0.20) — its train_loss vs baseline showed only **−0.036 max gap**, not
−0.13. Same band, more loop steps, a third of the apparent advantage.

Going back to v1_bak (the broken-with-different-recompile-timing run) confirmed:
the train_loss gap appeared **immediately at the recompile boundary** (step
2300 in v1_bak, step 1625 in the original broken 042A) and stayed at ~0.10
indefinitely after.

The 0.10 gap was NOT real model improvement — it was a measurement artifact
tied to the recompile.

## What was actually different

- Data loader is fully deterministic (no shuffle, cursor advances by step).
  Same step number → same batch. So it's not batch noise.
- Pre-quant EMA val_bpb (which uses a separately-compiled forward_logits path)
  showed 042A at +0.00147 vs baseline. That's the real model quality gap —
  small loss, not −0.10 advantage. The eval kernel measures consistently
  across runs; the train kernel doesn't.
- Hypothesis (unverified): the recompile changes either Triton autotune
  configs or kernel numerical paths (sigmoid precision, LSE reduction order,
  etc.) for the fused softcapped CE op. Same logits + same targets give
  systematically different CE values pre- vs post-recompile.
- We didn't pin down the exact mechanism — could be one of: per-row sigmoid
  precision shift, LSE reduction order from autotune, mid-step recompile
  leaving inconsistent buffers, or something else. Verification would need
  state-dict-replay across compile cache states, which we didn't do.

## The methodology rule

**When a single-batch metric jumps at the same time a system event fires
(recompile, optimizer change, scheduler change, kernel switch), the metric
is probably measuring the system event, not the model.**

Always confirm an apparent training-side improvement with a downstream metric
that doesn't share the same measurement path. In this codebase that means:
**val_bpb is the only believable signal; train_loss across recompile boundaries
is suspect.**

We were primed by "loop activation early" as the obvious story and attached
the 0.10 gap to it. The recompile was the *coincident* cause and we hadn't
separated the variables. This cost us ~half a day of speculative analysis and
inflated expectations for spec 043A.

## How to avoid this next time

1. **Before celebrating a train_loss gap, check val_bpb at matched step.**
   If val_bpb doesn't reflect the gap, the train_loss is lying.
2. **Annotate any recompile / kernel-state event in the log with a marker.**
   Then any train_loss reading near that marker should be flagged unreliable.
3. **Per-step train_loss is noisy enough that a gap of 0.05+ in a single
   measurement should ALWAYS be cross-checked** with cumulative metrics
   (sliding-window mean, val_bpb, loss-over-1k-steps).

## Spec 043A result (for context)

Final EMA val_bpb 1.06610 vs baseline 1.06514 = +0.00096 (noise zone).
The "early loop on canonical band" hypothesis turned in a small loss, not
a win. Confirms that the broken 042A's 0.10 train_loss "advantage" did
NOT translate to val improvement when the mechanism was tested cleanly.

## See also

- `research/evaluations/042A-slope-anneal-0707-to-05-screen.md` — original
  (and corrected) 042A eval
- `runs/042A-slope-anneal-0707-to-05-screen/train.log.v1_bak` — the broken
  attempt that exhibits the recompile artifact at step 2300
- `runs/042A-recompile-smoke/notes.md` — the smoke that pinned cache_size_limit
  as the recompile cause
