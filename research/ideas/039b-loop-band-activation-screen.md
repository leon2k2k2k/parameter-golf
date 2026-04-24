# Idea 039b — loop-band activation screen on the `039` base

## Thesis

The current stack uses the same MLP activation in every physical layer:

- `LeakyReLU(0.5)^2`

That is probably too uniform for a model whose middle physical layers `3,4,5`
are already privileged by recurrence. Reused loop-band computation may want a
more stable or more bounded nonlinearity than the outer trunk.

The recurrent-activation literature is not strong for looped transformers, but
the nearest signals point in a consistent direction:

- classic recurrent systems often prefer `tanh`-like bounded dynamics for
  stability
- penalized-tanh has unusually good NLP results and is a cleaner recurrent-like
  alternative than a random smooth activation swap
- reused middle-band blocks amplify activation behavior, so band-local
  nonlinearities are a more plausible lever than another global sweep

## Core proposal

Keep outer layers fixed at the current frontier-standard activation:

- outer layers `0,1,2,6,7,8,9,10` -> `LeakyReLU(0.5)^2`

Change only the looped middle physical layers:

- middle layers `3,4,5`

Screen these middle-band alternatives:

1. baseline: `LeakyReLU(0.5)^2`
2. `penalized_tanh`
3. `tanh`
4. `LeakyReLU(0.3)^2`

## Why this is interesting

- isolates activation *where recurrence lives*
- keeps the outer trunk identical to the current strong stack
- much cheaper and cleaner than redesigning the whole MLP
- gives a direct answer to whether recurrent-band activation wants to be more
  bounded than the feedforward trunk

## Expected ranking

Current guess:

1. `penalized_tanh`
2. `LeakyReLU(0.3)^2`
3. baseline `LeakyReLU(0.5)^2`
4. `tanh`

Reason:

- `penalized_tanh` is the most plausible “stable but not too saturating” choice
- `0.3` is the conservative same-family variant
- plain `tanh` may be too harsh / too saturating for this stack, but is still
  worth one direct test

## Main risks

- activation-family change may need eager fallback in the middle band, making
  throughput differences part of the result
- a short training-only screen may favor easier optimization but not survive
  quantization later
- plain `tanh` could simply be too saturated for this MLP path

## Cheapest next step

Freeze a training-only `4×H100` screen on the corrected `039` base:

- outer activation fixed
- middle activation varied across the four arms
- `600s`
- one seed
- stop after the pre-quant diagnostic
