# 036 — promote `035e` to the real `8×H` + TTT path

## Thesis

`035eA` is now the best completed `4×H` screen in the current `035` family, so
the relevant next question is no longer whether sparse gate helps on the small
rung. It is whether that stack survives promotion to the actual `8×H` full
pipeline with the standard `#1779` / `030` TTT path.

## Why this is the right promotion candidate

Current `035` ranking by post-EMA pre-quant:

- `035eA`: `1.06617649`
- `035A`: `1.06679052`
- `035dA`: `1.06730362`
- `026`: `1.06770372`

So:

- `MIN_LR` is real
- sparse gate is real
- the non-sparse `#1787`-lite bundle alone was not enough

That makes `035e`, not `035d`, the natural `8×H` promotion target.

## What this promotion should preserve

Training-side stack from the successful `035eA` run:

- `MIN_LR=0.10`
- Polar NS lineage
- fused CE
- budget polish
- sparse gate

Then add:

- the normal `030` / `#1779` full quantized eval
- phased LoRA-TTT
- no `val@4000` on the `8×H` promotion

## Why leave parts open right now

We already know the base branch and successful code line:

- branch: `exp/035e-sparse-gate-on-1779-family`
- successful code commit: `0e13ad0`

What we may still want to decide with execution later:

- exact seed choice from the approved shortlist (`314`, `2025`, `777`)
- whether to mirror the current best `030` TTT contract literally or update any
  small operational defaults
- exact accept threshold for the first `8×H` promotion rung

Operational note:

- an optional `2`-minute `8×H` no-TTT smoke rung is fine as a compile/path
  warmup
- but it should stay clearly separate from the real `036A` quality run

So `036` should exist now as a real draft shell, but not fake certainty on the
last-mile launch choices.
