# Unified Residual Routing On The Current CaseOps Frontier Family

## Core idea

Stop treating sparse gate, smear gate, and related residual-control objects as
separate knobs.

Instead, introduce a **single small residual-routing module** that decides how
much of several candidate update paths should flow into the residual stream.

The current stack already contains multiple partial versions of this idea:

- dense attention-output gating in `#1736`
- sparse attention-output gating in `#1787` / `#1801`
- residual-window `SmearGate` in `#1797`

These all look like local answers to the same broader question:

- what information should be preserved in the residual stream?
- what should be attenuated?
- what should be injected from local content versus attention output?

## Why this is worth separating

The repo has repeatedly found that small routing / gating objects matter, but we
are still composing them as independent patches:

- one gate on attention outputs
- another gate on residual content
- maybe another object on carry or passthrough later

That is useful for quick frontier moves, but it may be the wrong long-term
architecture. A cleaner design may be to unify these into one explicit routing
object rather than stacking more local control surfaces.

## Architectural sketch

For each block (or each recurrent pass of a block), compute a tiny routing
object over a few candidate update paths:

- residual passthrough
- attention output
- MLP output
- local-window residual feature (`smear`-like)

The router should be very small:

- scalar, per-head, or small per-channel routing weights
- conditioned on lightweight local features only
- no large controller MLP

The point is not to build a second model. The point is to make the residual
update rule more explicit and more coherent than the current pile of gates.

## Why it might help

- Could subsume sparse gate and smear gate into one cleaner object.
- Might improve additivity by removing the need to manually compose multiple
  routing knobs.
- Could create a more quant-friendly control surface if the routing object is
  tiny and well-structured.
- Gives a more interesting training-side direction than just adding one more
  gate variant.

## Risks

- Easy to accidentally reinvent "another gate" without enough real novelty.
- If the router is too expressive, it will eat artifact budget and become hard
  to reason about.
- If the router is too weak, it may collapse to the current sparse-gate result
  and not justify the implementation complexity.
- It may interfere with recurrence in a messy way unless pass/block semantics
  are kept simple.

## What would count as a good first version

- One small routing object
- No new carry mechanism
- Compatible with the current sparse-gate stack
- Cheap enough that the quant path and artifact budget remain manageable

## Current judgment

- **Mechanism:** unify existing residual-control ideas into one routing module
- **Expected delta:** unclear, but plausibly larger than another isolated gate
  tweak if the current separate objects are all pointing at the same missing
  architecture
- **Confidence:** low-to-medium; conceptually attractive, but easy to execute in
  a way that is too vague or too incremental
- **Main risk:** degenerates into "just another gate"
- **Cheapest next step:** sketch one very constrained routing design and compare
  it against the current sparse-gate baseline before combining it with any other
  architectural change
