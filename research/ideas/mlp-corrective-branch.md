# MLP Corrective Branch On The Current Frontier Stack

## Core idea

The current repo has spent much more architectural creativity on:

- recurrence
- carry / alpha-beta
- gating
- TTT

than on the MLP itself.

That is notable because the MLP is both:

- a major source of model capacity
- a likely major source of quantization damage

So a promising training-side direction is to redesign the MLP slightly, not by
making it much larger, but by giving it a **small corrective branch**.

## Proposed shape

Keep the existing main MLP path, but add one tiny secondary path whose job is to
handle the part of the transformation the main path misses.

Candidate forms:

- a low-rank corrective branch
- a gated small-width corrective path
- a tiny residual branch that is only active on a subset of features/tokens

The high-level pattern is:

- main path does the bulk of the work
- small branch learns a correction

This is attractive because it echoes what `#1797` is doing on the quant side
with LQER, but moves that logic into the training architecture itself.

## Why it might help

- The MLP seems under-explored relative to the rest of the stack.
- A corrective branch may improve raw model quality without requiring a large
  parameter increase.
- It may also make the model more quantization-friendly if the hard-to-quantize
  behavior gets concentrated into a small structured branch.
- This is more structurally interesting than another attention-output gate or
  another carry refinement.

## Why this is interesting now

Current strong stacks like `#1787` / `#1801` still show large quant gaps:

- `#1787`: `1.06699 -> 1.07632` (`+0.00933`)
- `#1801`: `1.06671 -> 1.07598` (`+0.00927`)

That suggests the architecture is still producing representations that are not
especially easy to pack. A better MLP design might help twice:

- improve the float checkpoint
- reduce downstream quantization damage

## Risks

- It could end up behaving like a tiny parameter bump rather than a genuinely
  different architecture.
- If the correction branch is too wide, the byte cost kills the benefit.
- If the branch is too weak, it may do nothing measurable.
- It is easy to make an MLP change that improves pre-quant metrics but not final
  post-TTT BPB.

## Good first versions

The best first versions are likely:

- small
- easy to ablate
- easy to quantize separately if needed

In particular, a **low-rank corrective branch** is attractive because:

- it is tiny
- it is easy to isolate in ablations
- it naturally connects training architecture to later quant/repair questions

## Current judgment

- **Mechanism:** add a small corrective branch to the MLP rather than continuing
  to leave the MLP almost unchanged while innovating elsewhere
- **Expected delta:** plausibly frontier-relevant if the MLP is a real quality
  and quant bottleneck, but completely unproven in this stack
- **Confidence:** medium that the MLP is under-explored; low on any exact branch
  design before testing
- **Main risks:** byte cost, weak effect, or gain that disappears after quant
  and TTT
- **Cheapest next step:** define one minimal corrective-branch variant, ideally
  low-rank, and test it alone on the current sparse-gate family before mixing it
  with any routing or carry changes
