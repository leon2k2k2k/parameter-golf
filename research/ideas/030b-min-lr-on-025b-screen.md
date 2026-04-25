# Idea 030b — `MIN_LR` on the original `025b` / `030` carry line

## Thesis

`034cB` showed that a simple `MIN_LR=0.10` floor can materially improve the
final artifact on a frozen-carry line.

The next natural transfer test is to port that schedule tweak back to the
original stronger frozen carry family:

- `025b` / `030`

## Why this is interesting

- this is the carry family that actually matters most historically
- the mechanism looks like a schedule effect, not a direct-carry-specific trick
- if the gain transfers, it is probably a real stack-level add

## Why a 4×H pre-quant screen is enough first

For this line, the cheap first question is:

- does `MIN_LR` improve the float checkpoint at all?

That can be screened with the existing `4×H100` style pre-quant gate instead of
going straight to a full 8×H TTT pipeline.

## Reference numbers from `030`

Observed 8×H seed results:

- seed `314`: pre-quant `1.06821629`, post-TTT `1.06471941`
- seed `2025`: pre-quant `1.06821738`, post-TTT `1.06438348`
- seed `777`: pre-quant `1.06798687`, post-TTT `1.06428960`

So the practical screen question is simple:

- can `MIN_LR=0.10` push the pre-quant screen clearly below the usual
  `~1.0682` seed-314 level?

## First test

Start with one rung only:

- `MIN_LR=0.10`

If that is directionally positive on the 4×H pre-quant screen, then it becomes
worth deciding whether to:

- finish the local ladder (`0.05`, `0.15`)
- or jump straight to a full `030`-family run
