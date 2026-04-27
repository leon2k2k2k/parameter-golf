# Idea: Reduce attention heads on layer 5 (deepest loop layer)

**Date:** 2026-04-27  
**Context:** Spec 045 loop-layer improvements series

## Motivation

Layer 5 is the deepest layer in the loop window [3,4,5]. In the encoder-decoder
split, it sits at the boundary between encoder and decoder — it's the last encoder
step AND the first decoder step. It's the most "overworked" layer: visited on
every pass, at a position where the representation should already be rich.

Hypothesis: layer 5 doing full attention on every pass is wasted capacity. Its
attention budget could be redistributed — either to MLP (wider/cheaper) or freed
entirely and given to deeper layers (6–10) which currently only run once.

## Lever options

**A. Reduce NUM_HEADS for layer 5 only**  
Give layer 5 fewer attention heads (e.g. NUM_HEADS//2 = 4 heads vs 8 elsewhere).
The freed parameter budget can go to MLP_MULT (wider MLP on layer 5) or just
reduce total params. Since layer 5 is weight-shared across all passes, halving
its heads saves attention compute on every pass.

**B. MLP-only on last loop pass for layer 5**  
Skip attention on layer 5 during the last (deepest) pass. Acts like the
LOOP_ITER_EMBEDS idea but specifically targets attention on the deepest layer.
`MLP_ONLY_FROM_PASS` already supports this pattern.

**C. Redistribute freed params to deeper layers (6–10)**  
If layer 5 attention is halved, expand MLP or add attention heads on layers 6–10
which only run once per forward pass but handle the final representations.
This is a parameter reallocation, not just ablation.

## Implementation notes

- Per-layer head counts require changes to the attention bank dimensions — not
  trivial but feasible. The bank approach already has LOOP_START/LOOP_END logic.
- Option B (MLP_ONLY_FROM_PASS) is already implemented — just test it on layer 5
  specifically. Low implementation cost.
- Option C requires a more significant architectural change.

## Prior art

- Spec 045B (`MLP_ONLY_FROM_PASS`) was specced but never run — test that first.
- The loop's +33% throughput tax is mainly from the attention on loop layers;
  reducing attention heads on layer 5 could reduce this tax.

## Suggested next step

Try MLP_ONLY_FROM_PASS=1 (skip attention on passes 1+ for all loop layers)
as a quick proxy. If positive, target layer 5 specifically.
