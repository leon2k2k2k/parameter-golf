# Idea 039 — negative-slope screen on the `#1797` base

## Thesis

The current frontier-family MLP uses `LeakyReLU(0.5)^2`. That choice was a
good local upgrade over `ReLU^2`, but the exact leak value does not appear to
be strongly justified beyond repo-local empiricism.

On the `#1797` stack, quantization damage is still large enough that even a
small activation-shape change could matter if it changes hidden activation
tails in the MLP:

- `#1787`: `1.06699 -> 1.07632` (`+0.00933`)
- `#1801`: `1.06671 -> 1.07598` (`+0.00927`)

So the cheapest discriminating experiment is not a new MLP topology. It is a
small sweep of the negative slope around the current `0.5` default.

## Hypothesis

Reducing the negative leak slightly below `0.5` may improve the current stack
by:

- making the negative-side hidden activations less permissive
- slightly reducing MLP activation tail mass / quantization burden
- retaining some negative-side gradient flow rather than collapsing all the way
  back to `ReLU^2`

The most plausible direction is that `0.4` or `0.3` beats `0.5` modestly. A
larger leak such as `0.6` is included mainly as a guardrail against
overfitting the intuition.

## Why this is a good first experiment

- Tiny code change.
- No parameter-count increase.
- No artifact-size increase from extra learned weights.
- Clean attribution: same model, same quant pipeline, same TTT, just one
  activation constant.
- Good answer even if it fails:
  - if all nearby values are flat, stop spending time on leak micro-tuning
  - if one value helps pre-quant but not post-quant, the activation is not
    solving the real quant hole

## Proposed screen

Screen these negative slopes on the `#1797` / `038` base:

- `0.3`
- `0.4`
- `0.5` baseline
- `0.6`

Recommended order if compute is tight:

1. `0.4`
2. `0.3`
3. `0.6`

## What to measure

For each run, capture:

- pre-quant BPB
- quantized BPB
- post-TTT BPB
- artifact size
- train / eval time

The important read is not only final BPB. We want to know whether any gain
comes from:

- better float checkpoint quality
- smaller quant damage
- better TTT interaction

## Expected delta

Small. This is a tuning screen, not a new architecture.

Plausible outcomes:

- null / noise-level
- slight win: roughly `0.0003` to `0.0010` BPB
- slight loss if `0.5` was already near-optimal

## Main risks

- Effect is too small to distinguish at tiny scale.
- Improvement in pre-quant does not survive quant + TTT.
- The fused MLP path may currently bake in `0.5`, so the code change must touch
  both fused and eager paths or the experiment is invalid.

## Current judgment

- **Mechanism:** shrink or expand negative-side MLP contribution without
  changing the rest of the stack
- **Expected delta:** low-to-moderate, but cheap enough to be worth screening
- **Confidence:** medium that this is worth one quick screen; low that it is a
  frontier-sized standalone lever
- **Best next step:** freeze a small screen spec on top of the `038` Smear+LQER
  base, expose `NEGATIVE_SLOPE` cleanly in both fused and eager MLP paths, and
  run a `4×H` smoke ladder before any `8×H` promotion
