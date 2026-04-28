# Idea — Reinvest unused TTT eval budget

**Date:** 2026-04-29

## Observation

Per memory `project_eval_time_quant_repair_idea.md`, leaderboard eval runs use 423-495s of the 600s cap (PR #1797 measurements). That's **100-180s of unused deploy-time compute** on every submission. Levers that consume more eval-time TTT compute should land within budget on most seeds, with stop-early discipline for the high-tail cases.

## Family of levers

Three orthogonal axes for "do more TTT":

1. **More phases** (`PHASED_TTT_NUM_PHASES 3 → 4`): one extra round of LoRA fitting on the prefix. ~+65s eval cost. → Spec **060J**.
2. **Larger adapter** (`TTT_LORA_RANK 80 → 100`): more parameters in the LoRA module → more capacity to fit prefix. ~+50s eval cost. → Spec **060K**.
3. **More prefix data** (`PHASED_TTT_PREFIX_DOCS 2500 → 4000`): more held-out tokens per phase. ~+50-80s eval cost. → Spec **060L**.

## Why post-training, not pre-training

Spec 055 (our PR #1885 lineage) is ~0.0013 BPB behind #1855 on prequant val_bpb. We don't have time to retrain a frontier-tying base before deadline (2026-04-30). Post-training levers can stack on existing checkpoints without retraining — the actual question is which of them transfers.

## Why TTT specifically

- Largest reported deltas in the post-training space (PR #1812 reports −0.008 BPB from a single +1 epoch bump on a weaker base; even with 5× absorption that's −0.0016)
- Pure config change — no code port, no test risk
- Eval-only via `RESUME_FROM_CKPT`, ~$1-2 per arm
- Naturally stacks with quant-side levers (AWQ-lite, SDClip) since TTT runs after quantization

## Risks

- **Wallclock cap.** TTT compute scales linearly in phases × rank × prefix; combined arms can blow the 600s eval cap. Single-arm tests first; only stack if individual arms have ~30s+ headroom.
- **Diminishing returns.** TTT may already be near its capacity ceiling on this base; a +25% rank bump might give 0 BPB and burn $2.
- **Per-seed wallclock variance.** A safe-on-average +50s lever might bust 600s on one unlucky seed and disqualify the submission. Need 3-seed cap-safety verification before submitting any tuned config.

## Plan

Run 060J → 060K → 060L sequentially, single-seed each, on 060A `seed_42_4h` final_model.pt. Pick winners (Δ ≥ accept threshold AND wallclock margin ≥ 30s on probe). If any win, scale to 3 seeds for cap-safety verification, then stack survivors.

## Open questions

1. Do all three phases share the 2500-doc prefix, or split? Determines whether 060J's +1 phase costs ~33% more or just one more pass.
2. Does TTT_LORA_RANK affect adapter init? Rank-bumped adapter may not warm from rank-80 checkpoint — could change behavior unpredictably.
3. Is there a `TTT_MAX_SECONDS` knob already, or do we need to add cap-safety as a stop-early?
