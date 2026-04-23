# 035g — freeze the learned `035f` `alpha/beta` back into the `#1779` family

## Thesis

If `035fA` teaches us that the old frozen `025b` `alpha/beta` values are no
longer ideal for the modern `#1779` / `030` stack, then the obvious next move
is not to keep them learnable forever. It is to measure the learned endpoint,
freeze it, and see whether that measured carry is itself a better deployable
object.

That is the point of `035g`.

## Why this is a good follow-up

- `035f` is a probe
- `035g` is the practical carry artifact

If `035f` wins because the current stack wants a different carry geometry, then
`035g` tells us whether that geometry can be baked back into the standard
frozen-alpha/beta line.

## Main rung

`035gA`

- exact learned `alpha/beta` from `035fA`
- frozen again during training
- same `030` family `4×H` screen stack
- no TTT first

## Optional robustness rung

`035gB`

- same learned values
- rounded to 4 significant digits before freezing

Reason:

- if 4 digits behave the same, the learned carry is probably robust / coarse
- if 4 digits hurt a lot, then the gain may depend on fragile exact values

## Why the rounding idea is interesting

This is really a robustness test, not a compression project.

Questions it answers:

- are we learning a clean carry geometry?
- or are we relying on very fine-grained coefficient placement?

The right default is:

- exact freeze first
- rounded freeze second only if the exact freeze is promising

## Promotion rule

If `035gA` is strong, then the natural next promotion is:

- full `8×H` run
- normal `030` / `#1779` phased LoRA-TTT stack

If `035gB` stays close, use the rounded version only if it buys practical
simplicity and does not materially cost quality.
