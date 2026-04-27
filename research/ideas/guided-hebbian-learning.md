# Guided Hebbian Learning (GHL) — auxiliary recurrence consistency loss

**Source:** PR #1819 (DuoNeural, 2026-04-25)
**Lever class:** training-loss
**Status:** prototype only (A6000, 32.8M params, 600 steps) — no competitive bpb

## What it is

During the loop-active training phase, attach a small MLP head that takes the
hidden state after recurrence pass `k` and predicts the hidden state after pass
`k+1`. Train this head with an MSE loss weighted by a schedule that anneals to
zero before the warmdown phase:

```
ghl_loss = w(t) * MSE(head(h_k), h_{k+1}.detach())
total_loss = ce_loss + ghl_loss
```

The head and its loss are discarded at eval time (zero cost). The intent is to
force each recurrence pass to produce representations that are "useful inputs"
for the next pass — Hebbian-style co-activation pressure between adjacent loop
iterations.

## Why it might help

Our Loop45 recurrence iterates the same layers 45 times. The gradient signal
from CE loss reaches each pass through BPTT, but early passes see highly
diluted gradients. The GHL auxiliary loss provides a local Hebbian training
signal to each pass — "make your output look like what the next pass wants to
see" — which could accelerate the representation tower from stabilizing and
improve the quality of the final pass.

## Estimated Δ and feasibility

- No credible bpb on a competitive stack exists. Prototype is too weak to extrapolate.
- Implementation cost: low. Small 2-layer MLP head, one extra forward per loop
  pass (or over just the last N-1 pairs to save compute). Annealing schedule
  needs tuning.
- Risk: on our 45-loop stack, computing the GHL loss for all 44 adjacent pairs
  is expensive — likely need to sample 1-2 pairs per step, or apply only to the
  last recurrence pair. Per-step throughput penalty needs measurement.
- TTT interaction: GHL is training-time only; cannot be absorbed by eval-time TTT.

## Key questions before speccing

1. Does GHL on Loop45 (45 passes, same layer set) behave differently from the
   2-pass context in #1819? Our loop iterates the same block repeatedly, not
   distinct encoder/decoder blocks.
2. What's the throughput penalty on our 8×H100 stack with 45 loop passes?
3. What annealing schedule (linear, cosine, step) works best?

## Credibility

DuoNeural is a first-time submitter. No H100 results. Idea is conceptually
sound but completely unvalidated at competition scale. Treat as a "watch and
spec if someone posts a full-scale result."
