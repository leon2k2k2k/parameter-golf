# Idea 035e — sparse gate on the `#1779` / `030` alpha/beta family

## Core idea

After testing the non-sparse `#1787`-lite bundle, separately test the sparse
attention-output gate on our stronger frozen `alpha/beta` line.

## Why this is separate

Sparse gate is the main architecture-ish change in `#1787`.

Unlike:

- `MIN_LR`
- Polar NS
- fused CE
- budget polish

it changes the model path and is more likely to create integration mistakes or
train/eval/TTT mismatch.

## Immediate goal

Run a `4×H` screen after `035d`, using the same base stack plus sparse gate.

The point is to answer:

- does sparse gate add anything on top of the transferred training-side recipe?
