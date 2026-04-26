---
name: VAL_LOSS_EVERY=1000 default for 4H screens
description: Always set VAL_LOSS_EVERY=1000 in 4×H100 screen specs; currently set to 0 (disabled)
type: feedback
---

Always set `VAL_LOSS_EVERY=1000` in 4×H100 screen specs. Current default is `VAL_LOSS_EVERY=0` (disabled).

**Why:** Train loss is noisy and doesn't reliably predict val_bpb. Mid-run val checkpoints let us catch bad runs early and understand training dynamics. Cost is ~9s per eval → ~90s overhead for a ~5500-step run (~7% fewer steps), acceptable tradeoff.

**How to apply:** All new screen specs (TRAINING_ONLY_SCREEN=1, 4×H100, 1200s) should include `VAL_LOSS_EVERY=1000`. Note: mid-run evals show raw weights (not EMA), so they'll be ~0.012 worse than final EMA val_bpb — use for trajectory/early-kill, not absolute comparison.
