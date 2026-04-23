# 036 — promote sparse gate plus updated rounded recurrent carry to `8×H` + TTT

## Thesis

`035eA` is still the best completed `4×H` sparse-gate screen, but we no longer
want a plain `035e` promotion. The promotion target is now:

- sparse gate from `035e`
- updated recurrent `alpha/beta` learned later and then frozen again
- those coefficients rounded coarsely to 2 significant digits
- standard `#1779` / `030` `8×H` full pipeline with phased LoRA-TTT

## Why this is the right promotion candidate

Current relevant `4×H` ranking by post-EMA pre-quant:

- `035eA`: `1.06617649`
- `035A`: `1.06679052`
- `035gA`: `1.06711750`
- `035dA`: `1.06730362`
- `026`: `1.06770372`

So:

- `MIN_LR` is real
- sparse gate is real
- the non-sparse `#1787`-lite bundle alone was not enough

That makes sparse gate the natural `8×H` promotion base. The open question is
whether swapping in the later updated recurrent carry helps once promoted.

## What this promotion should preserve

Training-side stack from the successful `035eA` run:

- `MIN_LR=0.10`
- Polar NS lineage
- fused CE
- budget polish
- sparse gate
- updated rounded recurrent carry:
  - `beta = [1.7, 2.0, 2.2]`
  - `alpha = [[0.28, -0.026, 0.046], [0.068, -0.42, -0.0033], [0.11, 0.25, -0.0048]]`

Then add:

- the normal `030` / `#1779` full quantized eval
- phased LoRA-TTT
- no `val@4000` on the `8×H` promotion

## Why leave parts open right now

We already know the runnable code line:

- branch: `exp/036-sparse-updated-alpha-beta`
- runnable code commit: `38e44e4`

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
