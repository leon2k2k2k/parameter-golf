# No `VAL_LOSS_EVERY=4000` on 8×H runs

For future 8×H runs, disable mid-train validation checkpoints (`VAL_LOSS_EVERY=0`) unless the user explicitly asks otherwise.

Reason:
- mid-train full validation costs real wallclock on the 10-minute budget
- for 8×H submission/promotion runs, we want the extra train time instead of a step-4000 val pass
- default future 8×H spec/launch contracts to no `val_4000`
