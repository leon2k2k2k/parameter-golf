# Idea 035d — `#1787`-lite bundle on the `#1779` / `030` alpha/beta family

## Core idea

Take our stronger frozen recurrent `alpha/beta` line and stack the likely
transferable public `#1787` adds:

- `MIN_LR`
- Polar NS
- fused CE
- budget polish

But explicitly leave out:

- sparse gate

## Why this is interesting

`034cB` suggested `MIN_LR=0.10` is a real training-side lever for us.

`#1787` suggests that `MIN_LR`, Polar NS, fused CE, and budget recovery can
stack into a stronger training recipe. Sparse gate is the main invasive
architecture-ish change, so the faster read is to test the rest first.

## Why this should come before sparse gate

- more likely to transfer
- lower integration risk
- mostly training-side refinements
- cleaner “same model family, better recipe” test

## Immediate goal

Run a `4×H` screen on the `030` family and compare against:

- `026` screen seed `314`: `1.06770372`
- `035` if available

If positive, promote to fuller testing and only then add sparse gate.
