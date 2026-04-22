# Idea 033b — TTT alpha/beta adaptation with aggressive LR

## Thesis

`033` allowed TTT to adapt frozen `alpha/beta` on top of the same `026 seed_42`
checkpoint used by `028B`, but the effect was effectively negligible:

- `028B`: `1.0664948109`
- `033`: `1.0664878103`
- delta: about `-7e-06` bpb

That pattern is consistent with an underpowered adaptation:

- `recur_alpha` moved a little
- `recur_beta` did not move at measurable precision
- outcome barely changed

So the next cheap question is not architectural. It is simply:

- did `033` fail because `TTT_ALPHA_BETA_LR_SCALE=0.25` was too small?

## Mechanism

Keep everything from `033` the same:

- same checkpoint
- same hotstart path
- same TTT setup
- same LoRA warm-start behavior
- same code commit

Only change:

- `TTT_ALPHA_BETA_LR_SCALE=10.0`

Since the base `TTT_LORA_LR` in this codepath is `1e-4`, this gives an effective
alpha/beta LR of:

```text
1e-4 * 10.0 = 1e-3
```

## Why this is worth one run

`033` at `2.5e-5` was so conservative that it was close to a no-op.

An aggressive rerun answers the question quickly:

- if `beta` still does not move and result still does not improve, the line is
  probably exhausted
- if `alpha/beta` move materially and TTT improves, then `033` was just
  under-tuned
- if it destabilizes, we also learn that immediately

## Expected outcomes

### Positive

- `recur_beta_max_drift` becomes clearly nonzero
- post-TTT beats `033` by more than noise

### Null

- drift increases but result stays flat
- or `beta` still stays effectively frozen

### Negative

- TTT becomes noisy or regresses
- post-TTT degrades beyond `028A` territory

## Recommendation

Worth exactly one follow-up run.

This should stay in the `033` family as a small LR rerun, not become a broad new
research line.
