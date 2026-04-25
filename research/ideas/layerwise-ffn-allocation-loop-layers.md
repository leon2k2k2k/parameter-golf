# Layerwise FFN Allocation For Looped Middle Layers

## Trigger

Recent external paper:

- **Layerwise Importance Analysis of Feed-Forward Networks in Transformer-based Language Models**
  - arXiv: https://arxiv.org/abs/2508.17734
  - finding: with total parameter count held fixed, concentrating FFN capacity
    into the consecutive middle ~70% of layers outperforms the standard uniform
    FFN layout

This is unusually relevant to the current repo because our architecture already
has a privileged middle-layer region:

- the looped middle layers
- plus an encoder-ish / decoder-ish split around that region

## Core thesis

We should stop assuming every block deserves the same MLP.

A plausible better allocation is:

- **looped middle layers get wider / richer MLPs**
- outer non-looped layers get standard or slightly reduced MLPs

This could be implemented with:

- larger `mlp_mult` only on looped layers
- or a small corrective branch only on looped layers
- or a different activation only on looped layers

## Why this fits our stack

The recent paper says FFN importance is not uniform by depth. The strongest
capacity concentration is in the middle layers.

Our model already treats the middle differently through recurrence / looping.
That makes the translation especially natural:

- the looped layers are the place where repeated refinement happens
- if any part of the network deserves extra FFN capacity, it is likely there

This is more compelling than globally increasing `MLP_MULT` everywhere.

## Relevant supporting paper

- **Transformer Feed-Forward Layers Are Key-Value Memories**
  - https://arxiv.org/abs/2012.14913
  - useful intuition: FFNs are not just generic width; they store different
    kinds of “memory” at different depths

## Research directions

1. **Wider loop-only MLP**
   - example: looped layers `4.5x`, outer layers `4.0x` or `3.5x`

2. **Budget-neutral reallocation**
   - widen looped layers
   - narrow outer layers
   - keep total parameter count near-flat

3. **Loop-only corrective branch**
   - keep standard MLP globally
   - add a tiny branch only on looped layers

4. **Loop-only activation change**
   - if activation slope matters at all, maybe it only matters in looped layers

## Why this may be better than a global width sweep

- cleaner architectural story
- matches recent external evidence
- better use of parameter budget
- more likely to preserve artifact size than widening every layer

## Current judgment

- **Mechanism:** non-uniform FFN capacity allocation, concentrated in looped
  middle layers
- **Expected delta:** potentially more interesting than a global MLP width bump
- **Confidence:** medium; external evidence is strong enough to justify a real
  screen
- **Cheapest next step:** after the activation slope screen, define a
  budget-aware loop-only width reallocation experiment on the `038` / `039`
  family
