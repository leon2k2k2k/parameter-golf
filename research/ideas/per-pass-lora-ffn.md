# Per-pass LoRA on the looped FFN

**Date:** 2026-04-27

## Idea

The Loop45 band currently runs `{3,4,5} × NL=2` (3 passes total) with **fully tied
weights**. Same `mlp_up`, same `mlp_down` on every pass. Each pass has zero
freedom to specialize.

Add a small per-pass LoRA delta on both `mlp_up` and `mlp_down`, one (A, B) pair
per (pass, layer, site). This gives each pass a tiny, controlled amount of freedom
to deviate from the shared base — without untying the full FFN.

## Lit basis

- **ALBERT (1909.11942) Table 4.** Tying the FFN across uses costs ~1.4–2.8 avg
  points. Tying the attention is approximately free (+0.1 at E=128). FFN is the
  axis where "make each use the same" hurts most.
- **Relaxed Recursive Transformers (2410.20672, DeepMind 2024).** Take a tied-block
  recurrent transformer, recover most of the dense-baseline gap by adding a
  per-iteration LoRA. Direct empirical kin of this idea.
- **MoLoRA / "ModernALBERT" (2512.12880, Dec 2025).** Per-pass LoRA experts inside
  the tied FFN. Confirms FFN-side per-pass differentiation as the dominant lever.

Our existing frozen α/β work (#1779, #1801) is the rank-0 (scalar) version of this.
Going to rank ≥ 1 is the natural lift.

## Why now

- Frontier #1797 (1.06157) has ~46 KB headroom on the 16 MB cap.
- Rank-2 LoRA stored at FP would not fit a submission, but a screen at training-
  endpoint val_bpb does not need quant — we only test the quality signal.
- If quality signal is positive, follow-up: int4 LQER-style storage (matches #1797's
  trick) which fits ~r=2 on the #1797 base.

See `research/specs/047C-per-pass-lora-ffn.md` for the frozen experiment.
