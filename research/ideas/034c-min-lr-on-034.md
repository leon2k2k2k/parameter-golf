# Idea 034c — `MIN_LR` floor on frozen `034`

## Thesis

`034` is now the cleanest frozen direct-carry line in the `NUM_LOOPS=2` regime.

The next cheap lever is not more carry complexity. It is a schedule tweak:

- keep late-training LR above zero with `MIN_LR`

This follows the ranked note in:

- `research/ideas/1779-next-adds-ranked.md`

## Why this is worth trying

- no code change
- one isolated schedule variable
- easy to compare against the existing `034` line
- more likely to transfer cleanly than another carry-mechanism rewrite

`034b` already suggests TTT-side learning on top of frozen direct-carry is not
the interesting next lever. A hotter tail is a cleaner bet.

## Regime

Keep the `034` stack fixed:

- frozen direct-carry from `031A`
- `NUM_LOOPS=2`
- live-like `4×H100`, `1200s`
- `ENABLE_LOOPING_AT=0.35`
- full pipeline including quantized and phased-TTT evaluation

Only change:

- `MIN_LR`

## Ladder

Three reasonable rungs:

1. `MIN_LR=0.05`
2. `MIN_LR=0.10`
3. `MIN_LR=0.15`

If only one rung is run first, the center rung is the right first bet:

- `MIN_LR=0.10`

## Main question

Does a hotter tail improve the frozen direct-carry line in a way that survives:

- pre-EMA
- post-EMA
- post-quant
- post-TTT

## Success condition

Any clear positive move over the current corrected `034` post-TTT result would
be interesting, because this is such a cheap additive change.

If the line is flat or harmful, that is still useful: it means the next effort
should go back to structural changes like `035`, not schedule tweaks on `034`.
