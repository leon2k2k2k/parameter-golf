# Idea 042 — tune `039bA` promotion from the strong `4H` base

## Thesis

`039bA` is the strongest live training-side signal:

- `4×H100`
- `600s`
- training-only
- middle loop band `3,4,5` uses `penalized_tanh`
- pre-quant post-EMA `val_bpb = 1.12148667`

The failed transfer into `041A` does not look like evidence that
`penalized_tanh` is wrong. It looks like a **promotion / schedule interaction**.

So the `042` family should not invent a new architecture. It should tune around
the `039bA` win.

## Current best hypothesis

The most likely missing lever is **loop timing**.

`039bA` only changes the loop band activation. That activation matters most when
the repeated middle-band computation is actually active. On longer or different
regimes, the model can spend too much of training in the pre-loop state before
the special activation has a chance to matter.

So the first `042` move is:

- keep `039bA` exactly the same
- start looping earlier

## What not to change yet

- do not broaden `penalized_tanh` to earlier physical layers
- do not change width allocation
- do not change carry / smear / LQER
- do not change activation family again

The point is to preserve the `039bA` signal, not bury it under extra variables.

## Core proposal

Run a longer `4×H100` training-only screen:

- `MAX_WALLCLOCK_SECONDS=1200`
- compare:
  - `039bA` control with `ENABLE_LOOPING_AT=0.35`
  - earlier-loop candidate with `ENABLE_LOOPING_AT=0.175`

Why `0.175`:

- it is the cleanest first approximation to "turn the recurrent band on about
  twice as early"
- it directly tests the user's intuition that the promotion failure may come
  from the activation arriving too late in the longer regime

## Decision rule

If earlier loop onset improves:

- train-loss trajectory after loop onset
- stop `val_bpb`
- pre-quant post-EMA `val_bpb`

then the next `042` sibling can tune loop timing more finely, for example:

- `0.20`
- `0.15`
- maybe loop-count or loop-depth timing later

If `0.175` is clearly worse, then the next suspect is not timing but broader
regime interaction, and the family should pause before adding more `042`
variants.
