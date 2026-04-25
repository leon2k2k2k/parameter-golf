# Spec 041B — 1801 plus penalized-tanh full run

**Slug:** `1801-plus-penalized-tanh-8h`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `submission/036-sparse-updated-carry-clean`
**Commit:** `372c5f1`
**Links to:** `research/specs/039b-loop-band-activation-screen.md`

## Hypothesis

The cleaner comparison to the merged `038` line is your `1801` carry-focused
submission line with the same loop-band `penalized_tanh` activation applied on
top.

This asks whether the activation gain is:

- strongest on the fully merged `038` stack
- or already strong on the cleaner `1801` carry line alone

## Baseline

Base line:

- `submission/036-sparse-updated-carry-clean`
- commit `372c5f1`

This spec is intentionally not yet runnable because the `1801` code line does
not currently have the loop-band activation API patched in.

Required code change before execution:

- add the `039b` activation-band support onto the `1801` train file
- then pin a new runnable `exp/...` branch and commit before launch

## Intended config diff

Apply the same activation split as `039bA`:

- `MLP_OUTER_ACTIVATION=leaky_relu_square`
- `MLP_MIDDLE_ACTIVATION=penalized_tanh`
- `MLP_MIDDLE_NEGATIVE_SLOPE=0.5`
- `MLP_MIDDLE_LAYERS=3,4,5`

Keep the rest of the `1801` stack unchanged.

## Intended regime

Full run:

- `8×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=600`

## Status note

`041B` is frozen as a research target, but **not execution-ready yet**.

The next step is a small branch patch to port the `039b` middle-band activation
API onto the `1801` code line, then repin this spec to that runnable branch.
