# Idea 035b — LR plateau on the original `030` alpha/beta family

## Thesis

If `035` is the global hotter-tail version of the schedule idea on the stronger
`030` family, then `035b` should be the targeted-transition version:

- plateau the LR around loop onset
- keep the rest of the `030` `4×H` screen stack fixed

This is the `030`-family analogue of the earlier `034dA` plateau idea.

## Why this belongs next to `035`

- `035` asks whether a global `MIN_LR=0.10` floor helps
- `035b` asks whether the gain is really about the recurrence transition itself

So the pair becomes:

- `035` = hotter tail everywhere
- `035b` = targeted loop-onset support

## Comparison target

Primary `4×H` alpha/beta-family benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

And directly against:

- `035` if that is run first

## First shape

Same plateau semantics as the `034dA` idea:

- `LR_PLATEAU_ENABLED=1`
- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`

Paused-time semantics:

- hold LR flat in the window
- after the window, resume with schedule time shifted by the plateau width

## Main question

On the stronger `030` alpha/beta family, is it better to:

- make the whole tail hotter (`035`)

or

- specifically support the loop-onset transition (`035b`)?
