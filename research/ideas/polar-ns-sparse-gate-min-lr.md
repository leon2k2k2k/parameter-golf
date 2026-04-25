# Polar NS + Sparse Gate + MIN_LR

Source: PR #1787 by `nprime06`.

What it is:

- Replace Muon's fixed Newton-Schulz coefficients with the per-iteration Polar Express tuples from PR #1344.
- Keep LR from decaying to zero by flooring warmdown at `MIN_LR=0.10`.
- Replace dense GatedAttn with a much smaller sparse head-output gate while preserving the same quantization path.

Why it matters:

- Claimed `val_bpb = 1.06378` on top of the `#1736` CaseOps stack, which is below our `1.06549` baseline and below the prior public likely-legal CaseOps leader `#1779`.
- The gain is bundled, so this is not a clean single-knob proof. Still, it is the strongest public evidence today that small optimizer/schedule/gating refinements can still move the #1736 family.

Estimated delta / feasibility:

- Plausible total delta bucket: `~0.001-0.002 BPB` on a strong CaseOps stack.
- Feasibility: high. All three pieces are implementation-light compared with a new architecture or tokenizer.

Risks:

- The PR is bundled, so attribution is unclear. One of the knobs may be carrying most of the gain.
- Because this is a CaseOps descendant, public legality still sits in the tokenizer-disputed bucket.
- Sparse gate may interact with our existing gate/carry experiments in non-additive ways.
