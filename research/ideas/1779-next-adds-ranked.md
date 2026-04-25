# #1779 next adds — ranked

Base assumption for this note:

- Start from **PR #1779** style stack
- Current public score: **1.06421**
- Lineage: `#1736` CaseOps base + `#1767` warm-start-A LoRA-TTT side + frozen recurrent alpha/beta carry blend

This note ranks the most plausible public add-ons to try on top of `#1779`.

## Rank 1 — MIN_LR floor

What it is:

- Keep late-training LR above zero instead of annealing all the way down
- Public example: `#1787` uses **`MIN_LR=0.10`**

Why it ranks first:

- Very small implementation change
- Easy to isolate
- Most likely to transfer across nearby stacks
- Parameter Golf runs are short and often still underconverged at the wallclock cap, so a hotter tail is plausible

What it might buy:

- Unknown in isolation; in `#1787` it is bundled with several other tweaks
- Treat as a plausible **small positive late-training gain**, not a guaranteed win

How to test:

1. `#1779 + MIN_LR=0.05`
2. `#1779 + MIN_LR=0.10`
3. `#1779 + MIN_LR=0.15`

What to watch:

- pre-EMA endpoint loss
- post-EMA bpb
- post-quant bpb
- post-TTT bpb
- seed variance

Main risk:

- A hotter tail may improve raw training loss but hurt the final quantized or TTT-evaluated checkpoint

## Rank 2 — Polar NS

What it is:

- Replace Muon's repeated fixed Newton-Schulz coefficients with the per-iteration optimized "Polar Express" tuples
- Public source used in `#1787`, attributed back to `#1344`

Why it ranks second:

- Also small and self-contained
- Optimizer-quality refinement rather than a big stack change
- Likely orthogonal to carry blend

What it might buy:

- Unknown in isolation; bundled in `#1787`
- Plausible **small optimizer-quality gain**

How to test:

1. `#1779 + PolarNS`
2. `#1779 + PolarNS + MIN_LR=0.10`

Main risk:

- Gain may be real but tiny enough to get lost in seed noise unless the stack is already very stable

## Rank 3 — Polar NS + MIN_LR combo

Why it ranks here:

- This is the most direct extraction from `#1787`
- But it should come **after** the singles so attribution is not lost

Why it is still worth trying:

- If both singles are neutral-to-positive, the combo is the natural bundled follow-up
- If only one moves, the combo still tells you whether they interfere

Main risk:

- If tested first, it hides whether the gain came from schedule or optimizer refinement

## Rank 4 — Gate-path audit

What it is:

- Verify that any existing gate logic is actually active in all forward paths used by training, quantized eval, and TTT / LoRA eval

Why it ranks high despite not being a new lever:

- `#1784` and `#1787` both suggest silent forward-path mismatch is easy
- This can create fake regressions or fake non-additivity

What to check:

- Does the same gate apply in normal forward?
- Does the same gate apply in any TTT helper path that manually unrolls attention?
- Is quantization routing still hitting the gate weights correctly?

Main risk:

- Low technical risk, but it may produce no gain by itself; it is mainly a correctness guard

## Rank 5 — Sparse gate variant

What it is:

- Replace a denser attention-output gate with a narrower-input sparse variant
- Public source: `#1787`

Why it ranks below the others:

- More entangled with existing gate implementation choices
- Harder to know if the gain came from the gate shape itself or from bundle interactions in `#1787`
- `#1779` already sits on a gated CaseOps family, so this is not obviously additive

What it might buy:

- Potential artifact savings
- Maybe a small regularization / efficiency effect

Main risk:

- Easy to spend time on without proving it is better than the current gate design

## Lower priority / not immediate

### Recurrence depth curriculum

Status:

- Already being tried
- Public source: `#1756`

Why not ranked higher here:

- It is already on the table
- Harder to execute robustly than the schedule / optimizer micro-tweaks above

### Full `#1787` bundle

Status:

- Do not port blindly

Why:

- Total `#1779 -> #1787` public gap is only about **`-0.00043`**
- That gain is spread across several changes
- Bundling first destroys attribution

### PPM / cache-family methods

Status:

- Not in scope for this branch

Why:

- They are not the same kind of model-side improvement
- They conflict with the repo's current frontier-scan legality policy

### `#1790` gate bundle

Status:

- Watch only; do not port wholesale

Why:

- `#1790` is mostly a different branch recomposition, not an obvious improvement over `#1779`
- Public score is **`1.06991`**, which is materially worse than **`#1779 = 1.06421`**
- Much of that gap is plausibly tokenizer-driven because `#1790` is plain `SP8192` while `#1779` already has `CaseOps`

What `#1790` actually contributes relative to `#1779`:

- `SmearGate`
- `AttnOutGate` widened to 24

How to think about it:

- `SmearGate` is real historically, but on the modern `#1779` stack it is not an obvious first-order add
- `AttnOutGate(w24)` is the more interesting isolated piece, but still trails the cleaner schedule/optimizer bets above
- If we ever test anything from `#1790`, test the gate-width idea alone rather than importing the full non-CaseOps branch

Practical takeaway:

- `MIN_LR` and `PolarNS` stay ahead of `#1790`-style gate ports
- `AttnOutGate(w24)` is a medium-priority watch item
- full `#1790` port is not recommended

## Recommended experiment order

1. `#1779 + MIN_LR`
2. `#1779 + PolarNS`
3. `#1779 + PolarNS + MIN_LR`
4. gate-path audit
5. sparse gate variant

## Practical summary

If the goal is to move the `#1779` branch quickly with the least engineering:

- **First bet:** `MIN_LR floor`
- **Second bet:** `Polar NS`
- **Third bet:** combine them if singles do not hurt

If the goal is to avoid silent nonsense:

- do the **gate-path audit** before spending much time on any gate variant
