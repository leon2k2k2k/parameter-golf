# Evaluation 034c — `MIN_LR` floor on frozen `034`

## Result

`034cB` is a clear win.

Corrected final result:

- `034` final `quantized_ttt_phased val_bpb`: `1.06669374`
- `034cB` final `quantized_ttt_phased val_bpb`: `1.06391360`
- delta: `-0.00278014`

TTT eval time:

- `684648ms` (`684.6s`)

## Important validation note

An earlier comparison was invalid because a mislabeled `034` artifact from a
different stack family was used. The corrected result above is the relevant
like-for-like comparison against the actual `034` baseline.

This matters because `034c` is only meaningful if the inherited stack is truly
the same as `034` except for `MIN_LR`.

## Side-by-side

Stop validation:

- `034`: `1.0704`
- `034cB`: `1.0782`
- delta: `+0.0078` worse

Post-EMA pre-quant:

- `034`: `1.06975736`
- `034cB`: `1.06720956`
- delta: `-0.00255` better

Quantized diagnostic:

- `034`: `1.07912355`
- `034cB`: `1.07642793`
- delta: `-0.00270` better

Final post-TTT:

- `034`: `1.06669374`
- `034cB`: `1.06391360`
- delta: `-0.00278014` better

## Interpretation

This is the interesting pattern:

- stop-val got worse
- but the final checkpoint got better everywhere that matters:
  - post-EMA
  - quantized
  - post-TTT

So `MIN_LR=0.10` is not just making training look better early. It appears to
improve the actual final artifact quality.

That makes this a real schedule lever, not a cosmetic training-loss tweak.

## Decision

Promote.

`MIN_LR=0.10` should now be treated as a serious add-on candidate, not just a
cheap speculative test.

## Next step

1. Run the neighboring `034c` rungs:
   - `034cA = 0.05`
   - `034cC = 0.15`
2. If `0.10` remains best or near-best, port the winning `MIN_LR` value back to
   the stronger original carry line.
3. Keep `034d` alive as the more targeted schedule follow-up, but it is now
   less urgent than finishing the simple `MIN_LR` ladder.
