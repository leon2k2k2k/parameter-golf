# Evaluation 034b — TTT Adapt Frozen Direct-Carry

## Result

Negative / null.

`034b` ran two `4×H100` hotstart-only `3`-phase TTT passes from the same
`034` float checkpoint:

- `034bA` frozen direct-carry baseline:
  - `diagnostic_pre_quant_post_rotation val_bpb = 1.2375012699`
  - `diagnostic_quantized val_bpb = 1.0806150627`
  - `quantized_ttt_phased val_bpb = 1.0675957597`
- `034bB` learnable direct-carry during TTT:
  - `diagnostic_pre_quant_post_rotation val_bpb = 1.2375012699`
  - `diagnostic_quantized val_bpb = 1.0806150627`
  - `quantized_ttt_phased val_bpb = 1.0675986668`

Primary comparison:

- `034bB - 034bA = +0.0000029071` bpb

So the learnable direct-carry probe is effectively a tie, with a tiny regression.

## Delta Vs Baseline

Internal `034b` baseline:

- `034bA` is the corrected `3`-phase hotstart baseline for the `034` checkpoint
- `034bB` does not improve over it

Broader `4H` phase-3 hotstart references:

- `033`: `1.06648781`
- `033b`: `1.06666734`
- `034bA`: `1.06759576`
- `034bB`: `1.06759867`

So the whole `034b` checkpoint family is worse than the stronger `026`-line
`4H` hotstart TTT runs, and making direct-carry learnable during TTT does not
recover that gap.

## Signal Vs Noise

The run is informative, not flaky.

Why the result is trustworthy:

- both runs completed cleanly
- hotstart-side diagnostics matched exactly between `A` and `B`
- the TTT direct-carry path in `B` was definitely active
- tensors moved during TTT and the required logs were emitted
- there was no NaN, crash, or silent no-op

This was not a “didn’t work” failure. It was a clean null result.

## Trajectory Notes

`034bB` briefly looked mildly promising if compared only against the final
`034bA` number, but the step-matched view showed it was actually worse than
`034bA` through the comparable post-phase region.

Direct-carry drift in `034bB`:

- `direct_carry_self_frozen_max_drift = 0.000000`
- `direct_carry_edges_frozen_pass1_max_drift ≈ 0.007690`
- `direct_carry_edges_frozen_pass2_max_drift ≈ 0.026733`

Interpretation:

- the adaptation happened almost entirely through the frozen edge tensors
- especially pass-2 edges
- that movement did not translate into better TTT quality

## Decision

Kill this line.

More precisely:

- do not promote learnable direct-carry-during-TTT as a promising mechanism on
  this checkpoint family
- do not spend more cycles on LR sweeps for this exact `034b` setup unless a
  separate argument emerges

## Next Step

If we revisit this family at all, the next experiment should not be “same
mechanism, different LR”.

More plausible next directions:

- change the base checkpoint family instead of the TTT adaptation rule
- or drop the hotstart-TTT lane for this direct-carry checkpoint family

The current result says:

- frozen direct-carry from `034` is not a strong hotstart-TTT base
- making it learnable during TTT does not rescue it
