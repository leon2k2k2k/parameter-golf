# Idea 039bD — loop-band penalized-tanh plus 040C width split

## Thesis

Two training-side signals are currently alive:

- `039bA`: loop-band `penalized_tanh` is a strong middle-band activation win on
  the `039` base
- `040C`: widening the loop band while shrinking late-layer MLP width survived
  triage and did not get worse in the off-spec width screen

The right next question is not another sweep. It is whether those two signals
stack.

## Composite proposal

Keep the loop band `3,4,5` special in two ways at once:

- width split from `040C`
  - early `4.0`
  - middle `5.0`
  - late `3.4`
- activation split from `039bA`
  - outer layers: `LeakyReLU(0.5)^2`
  - middle layers: `penalized_tanh`

## Why this is the correct next move

- `039bA` is the dominant training-side signal, so it should be preserved
- `040C` is not proven, but it is the only width-allocation family that
  survived triage
- a single composite answers the real additive question without exploding the
  search space

## Decision logic

Compare only:

1. baseline
2. `039bA`
3. composite `039bD`

Interpretation:

- if `039bD > 039bA`, width reallocation adds on top of the activation win
- if `039bD ~= 039bA`, the activation is doing the real work
- if `039bD < 039bA`, width split is hurting the better activation path

## Main risks

- `039bA` may already capture most of the available training-side gain
- the `040C` width split may interact badly with penalized-tanh in the loop
  band
- training-only improvement may still fail later under quantization
