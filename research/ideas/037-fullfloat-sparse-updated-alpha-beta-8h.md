# 037 — full-float sparse carry promotion to `8×H` + TTT

## Thesis

`036` answers whether the sparse learned carry survives heavy rounding.
`037` is the paired check for the opposite question:

- same sparse-gate stack
- same fixed TTT path
- same `8×H` promotion
- but keep the learned `035h` recurrent carry at full float

## Why this exists

The `035h` learned carry has one obviously strong structural pattern:

- `beta` rises above the neutral init at all three looped layers
- `alpha[1,1]` stays strongly negative
- several smaller off-diagonal terms are nonzero but modest

`036` intentionally rounds those small terms away.
`037` keeps them.

So the comparison is:

- `036`: is coarse carry structure enough?
- `037`: do the small learned coefficients matter?

## Intended comparison

Compare:

- `036` rounded carry
- `037` full-float carry

under the same:

- sparse gate
- `MIN_LR=0.10`
- fused CE
- `GPTQ_RESERVE_SECONDS=0.5`
- phased LoRA-TTT
- `8×H`, `600s`

## Carry values

Full-float frozen values:

- `beta = [1.5610, 1.8531, 2.1320]`
- `alpha = [[0.2314, 0.0388, 0.0347], [0.1260, -0.3438, 0.0145], [0.0557, 0.1934, -0.0172]]`

## Code line

- runnable code branch: `exp/037-fullfloat-sparse-updated-alpha-beta`
- runnable code commit: `74b7dea`

This inherits the sparse-gate TTT bugfix from the corrected `036` code line.
