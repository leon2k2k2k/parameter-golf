# Idea 039bE — loop-band tanh plus 040C width split

## Thesis

`039bA` (`penalized_tanh` in the loop band) is currently the strongest
training-only signal, but plain `tanh` is still worth testing in composite form
because it is the cleaner bounded recurrent nonlinearity.

If the loop band really wants more stable recurrent-style dynamics, the width
split from `040C` may combine with plain `tanh` as well as or better than the
penalized variant.

## Composite proposal

Keep the same width split as `040C`:

- early `4.0`
- middle `5.0`
- late `3.4`

Keep outer layers on the current activation:

- outer -> `LeakyReLU(0.5)^2`

Change only the loop band `3,4,5` to:

- middle -> `tanh`

## Decision logic

This is not a broad sweep. It is a sibling to `039bD`.

Compare:

1. baseline
2. `039bB`
3. composite `039bE`

Interpretation:

- if `039bE > 039bB`, the width split adds on top of the plain `tanh` signal
- if `039bE ~= 039bB`, activation is doing the work
- if `039bE < 039bB`, the `040C` width split is not helping this activation

## Main risk

Plain `tanh` may be too saturating in the loop band even if it looked
promising enough to keep alive. This composite tests that directly without
polluting the board with more width variants.
