# Spec 060M — TTT epochs 3 → 4 on 060A baseline (eval-only)

**Date:** 2026-04-29
**Slug:** `060M-ttt-epochs-up`
**Idea:** `research/ideas/ttt-budget-reinvestment.md`
**Branch:** `research` (config-only; no code change)
**Pinned SHA:** `da50cd6` (060A's commit)

## Hypothesis

PR #1812 reported **−0.008 BPB** on a weaker pre-phased base (04-09 stack, ~1.0810 → 1.0729) by bumping TTT epochs 3 → 4 (combined with split MLP/attn WD; epoch bump credited as the main lever). **This has never been tested on a phased-TTT + SmearGate stack like #1855.** The technique-timeline review found a clear gap: epoch bump is the lever with the largest single reported delta on the leaderboard, and the only one in our shortlist that's never been ablated on a phased-TTT base.

If the per-chunk LoRA SGD inside each phase is undersaturated, +1 epoch should still buy partial Δ. Even with 5× absorption from #1812's −0.008, that's **−0.0016 BPB** — biggest predicted Δ in the 060J/L/M family.

## Baseline

060A (`seed_42_4h`, 4H matched-FLOPs).

## Expected Δ

**−0.001 to −0.003 BPB** post-TTT. (Wider band because no measurement on phased base; #1812's −0.008 on weaker base is the only data point.)

## Accept criteria

- post-TTT val_bpb ≤ (060A_seed_42_4h post-TTT − 0.001)
- eval wallclock ≤ 600s (HARD)
- TTT loss curve in `eval.log`: epoch 4 should reduce train loss further than epoch 3 (capacity check)
- artifact size unchanged

## Config diff vs 060A

```
TTT_EPOCHS: 3 → 4   (verify exact env var name in train_gpt.py at SHA da50cd6;
                     may be `TTT_NUM_EPOCHS` or `TTT_EPOCHS_PER_PHASE`)
```

All other TTT knobs unchanged. **Critical:** verify env var name during preflight — if epochs is computed implicitly (e.g. from PHASED_TTT_NUM_PHASES + PREFIX_DOCS chunk size), this spec needs a small code patch and should be re-routed.

## Code changes

None *expected*; verify env var exists before launch. If no env var: spec stops, user decides whether to add one (~5 LOC) or kill.

## Hardware ladder

- 4×H100 eval-only; ~8-12 min wall, ~$1-2.

## Seed plan

Single seed (42).

## Inputs

Same as 060J/L. Parent ckpt: `/workspace/runs/060A-1855-port/seed_42_4h/final_model.pt`.

## Checkpoints to emit

- `runs/060M-ttt-epochs-up/seed_42/final_model.int6.ptz`
- `runs/060M-ttt-epochs-up/seed_42/eval.log`

## Stop-early criteria

- Eval wallclock projects > 600s → kill
- post-TTT val_bpb > 060A → kill (regression / overfit)
- TTT NaN / divergence → kill
- Env var doesn't exist → halt, escalate to user

## Cost estimate

~$2 (assuming env var exists; otherwise +30 min dev for the patch).

## Run command

```bash
# CONDITIONAL: verify env var name first (TTT_EPOCHS vs TTT_NUM_EPOCHS vs PHASED_TTT_EPOCHS)
SEED=42 RUN_LABEL=seed_42 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42_4h/final_model.pt \
  ARM=060M-ttt-epochs-up \
  TTT_EPOCHS=4 \
  bash tmp_exec/launch_060_eval.sh
```

## Open questions for interview

1. **What is the actual env var name?** PR #1812 used "epochs" terminology but the env var name in current train_gpt.py is unverified. Grep for `epoch` in the SHA-`da50cd6` train_gpt.py before launch.
2. **Per-phase or global epochs?** If "epochs" applies per-phase, +1 epoch costs ~+33% TTT compute (~+65s); if global across all phases, ~+11% (~+22s). Different cap-safety implications.
3. **Interaction with `PHASED_TTT_PREFIX_DOCS=2500` chunking:** if epochs scales the # of passes over the chunk grid, doubling epochs shouldn't change which chunks are seen — pure SGD-pass count. Verify.

## Phase 2 (NOT in 060M)

If 060M wins ≥ −0.001 BPB and 060L (prefix up) also wins, stack as 060N: `PREFIX_DOCS=3000 + TTT_EPOCHS=4`. Total TTT cost ~+115s; needs cap-safety screen on 3 seeds.
