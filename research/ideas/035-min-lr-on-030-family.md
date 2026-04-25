# Idea 035 — `MIN_LR` on the original `030` alpha/beta family

## Thesis

The strong schedule result so far is:

- `034cB` improved a weaker frozen direct-carry branch by a large margin

That makes the obvious next question:

- does `MIN_LR=0.10` transfer to the stronger original frozen alpha/beta line?

This should now be treated as the main frontier question, ahead of more
direct-carry variants.

## Why this is the right next line

- `030` is the historically stronger family
- `026` gives a real `4×H` reference point for the same family
- `MIN_LR` now looks like a schedule-level add, not a direct-carry-specific fix

## Reference numbers

Best relevant `4×H` alpha/beta-family benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

Relevant `8×H` `030` family results:

- seed `314`: pre-quant `1.06821629`, post-TTT `1.06471941`
- seed `2025`: pre-quant `1.06821738`, post-TTT `1.06438348`
- seed `777`: pre-quant `1.06798687`, post-TTT `1.06428960`

## Main question

Can a `4×H` `030`-family `MIN_LR=0.10` screen beat the established `026`
`4×H` pre-quant reference of `1.06770372`?

If yes, then `MIN_LR` is a serious candidate to port onto the real `8×H` `030`
line.

## Execution shape

First rung should be:

- exact `030`-family `4×H` screen stack
- only `MIN_LR=0.10` changed
- pre-quant gate only

Then, if positive:

- decide whether to run a small `MIN_LR` ladder
- or jump directly to a fuller `8×H` `030`-family follow-up
