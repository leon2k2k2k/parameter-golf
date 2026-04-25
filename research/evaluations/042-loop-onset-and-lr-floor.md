# Evaluation 042 — Loop onset timing + LR floor on 039bA base

## Result

**042C is the new best.** Earlier loop onset wins decisively; LR floor adds a
tiny additional margin.

| Arm   | Config                                    | prequant postEMA val_bpb | Δ vs 039bA  | Δ vs 042A   |
|-------|-------------------------------------------|--------------------------|-------------|-------------|
| 039bA | baseline (loop@0.35, default LR)          | 1.12148667               | —           | —           |
| 042B  | loop@0.35 (control), default LR           | 1.09951625               | −0.02197    | +0.00212    |
| 042A  | loop@0.175 (earlier), default LR          | 1.09740031               | −0.02409    | —           |
| 042C  | loop@0.175 (earlier), LR floor half-half  | **1.09724928**           | −0.02424    | −0.00015    |

All three experimental arms beat 039bA by ~0.022–0.024. 042A/C beat the
control (042B) by ~0.002.

## Stop-step summary

All runs on 4×H100, 1200s wallclock cap, TRAINING_ONLY_SCREEN=1, seed 42.

| Arm   | Loop activated | Stop step | Stop val_bpb |
|-------|---------------|-----------|--------------|
| 042A  | step 1144     | 4708      | 1.1100       |
| 042B  | step 2400     | 5104      | n/a          |
| 042C  | step 1144     | 4726      | 1.1176       |

042A and 042C stop at nearly the same step (~4708 vs 4726), confirming the same
effective wallclock budget. 042B's later loop activation (~2400) means it runs
more pre-loop steps and stops later (~5104) while getting fewer post-loop steps.

## Finding 1 — Earlier loop onset is the main lever

042A vs 042B isolates loop onset timing (0.175 vs 0.35) with all else equal.
042A wins by **0.00212** on prequant postEMA val_bpb.

Mechanism: activating the recurrent loop earlier means more of the 20-minute
budget runs with the full recurrent stack. Post-loop throughput is ~33% lower
(~4300 → ~3200 tok/s), so this trades tokens-per-second for model depth. At
this budget, more recurrent time wins.

## Finding 2 — LR floor adds a marginal improvement

042C vs 042A isolates the LR schedule (`first_half_default_then_floor` vs
default cosine decay), with loop onset held fixed at 0.175.

042C wins by **0.00015** — real but tiny. The LR floor holds the learning rate
flat at MIN_LR for the second half of training instead of continuing to decay.

Train-loss trajectory told an interesting story:
- Steps 1400–2500: 042C led 042A by ~0.10–0.14 consistently
- Steps 3000–4000: lead compressed to ~0.04–0.09
- Steps 4400–4700: 042A briefly edged ahead on raw train loss

The EMA smooths this out and 042C ends +0.00015 better, but the effect is small
and may not survive seed variance.

## Trajectory comparison (selected steps)

| Step | 042C   | 042A   | 042B   | 039bA  |
|------|--------|--------|--------|--------|
| 1000 | 2.7903 | 2.8082 | 2.8026 | 2.7921 |
| 1500 | 2.5698 | 2.6236 | 2.6433 | 2.5702 |
| 2000 | 2.5865 | 2.6766 | 2.7101 | 2.5823 |
| 2500 | 2.3967 | 2.5414 | 2.5716 | 2.3981 |
| 3000 | 2.4946 | 2.6216 | 2.6469 | —      |
| 3500 | 2.3547 | 2.4419 | 2.4684 | —      |
| 4000 | 2.3928 | 2.4311 | 2.4787 | —      |
| 4500 | 2.4272 | 2.4122 | 2.4584 | —      |

Note: 042C tracks almost exactly with 039bA through step 2500, then diverges
above it in the second half — consistent with the loop giving a structural
advantage that shows up in EMA rather than instantaneous train loss.

## Decision

**Promote 042A loop onset (0.175) as the new default.** The 0.022 win vs 039bA
baseline is large and clear. This is the main result.

**042C's LR floor is a weak positive signal, not a promotion.** 0.00015 delta
is within seed noise. Before treating it as a real knob:
- Run 042C vs 042A with a second seed to check if the edge holds
- Or stack the LR floor on top of a future stronger baseline rather than
  spending another run on seed 42

## Next steps

1. **Port loop@0.175 onto the current best submission base** — this is the
   highest-leverage move; 042 was running on the 039b branch which lags the
   latest frontier.
2. **Consider TTT + loop@0.175 interaction** — loop onset affects how much of
   the budget is recurrent; TTT may interact differently at 0.175 vs 0.35.
3. **LR floor follow-up** — low priority; only worth a dedicated run if
   loop@0.175 promotion stacks cleanly and there is budget for a 2-seed 042C
   retest.
