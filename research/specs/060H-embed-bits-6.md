# Spec 060H — EMBED_BITS=6 with LQER recovery on 060A baseline (eval-only)

**Status: DEPRECATED 2026-04-29 — implied refutation by PR #1898.**

PR #1898 ran EMBED_BITS=6 *with* SpinQuant rotation as protection against
INT6 noise and got a **+0.00486 BPB regression** vs their base. EMBED_BITS=6
*without* SpinQuant (this spec's H1, H2, H3 arms) has *more* INT6 noise on
`tok_emb`, so all arms here would likely regress further. Don't run on its
own — the pessimistic scenario in our prediction table is the most likely
outcome.

Document kept for reference. If we ever build deploy-time repair (060C)
that specifically targets `tok_emb` precision recovery, this spec becomes
worth re-examining as a stack candidate — but only after that's measured
and shown to work.

---

**Date:** 2026-04-29
**Branch:** `research` (config-only)
**Parent:** 060A `final_model.pt` + `RESUME_FROM_CKPT` infrastructure (commit `a7c0ed8`).

## Hypothesis

The embedding table (tied with output head, ~4.2M params on SP8192×512) is the largest single contributor to the artifact byte budget. At INT7 it occupies ~3.7 MB raw, at INT6 ~3.1 MB raw — saving ~520 KB raw / ~150-300 KB after `pergroup` compression. PR #1898 made INT6 viable by adding SpinQuant rotation (which makes the embedding distribution more outlier-free).

**Without SpinQuant**, INT6 alone introduces ~2× per-element quantization noise on `tok_emb`. The hypothesis: LQER asym (already covering `tok_emb` as the highest-rank correction tensor in 046's measurements) can recover most of that noise IF we expand its capacity using the freed bytes. **Net effect could be quality-neutral with bytes saved**, or quality-positive if LQER-side gain exceeds INT6 noise.

If the test is positive without SpinQuant, 060G (Partial SpinQuant) should compound it further.

## Baseline

060A.

## Expected Δ

Three rough scenarios:
- **Pessimistic** (LQER doesn't recover): +0.003 to +0.008 BPB (worse than 060A; INT6 noise dominates)
- **Neutral** (LQER recovers ~half): +0.000 to +0.002 BPB but with ~150 KB freed → reinvest still possible
- **Optimistic** (LQER recovers + reinvest gains): −0.002 to −0.005 BPB

PR #1898 reports their EMBED_BITS=6 + SpinQuant arm at 1.06614 (vs ~1.06128 baseline → +0.00486 worse), suggesting that *without* SpinQuant the regression is significant. Need to actually measure on our base.

## Accept criteria

- val_bpb ≤ (060A + 0.001)  ← weak, just looking for "not catastrophic"
- artifact ≤ 16,000,000 bytes (must fit; failing this means even more savings to reinvest)
- bytes saved (vs 060A artifact) ≥ 100 KB

## Config diff vs 060A

Three arms (sequential, eval-only via RESUME_FROM_CKPT):

**Arm H1 — EMBED_BITS=6 standalone:**
```
EMBED_BITS: 7 → 6
EMBED_CLIP_SIGMAS: 14.0 → 14.0   (unchanged; INT6 will clip more aggressively as a side effect)
```

**Arm H2 — EMBED_BITS=6 + LQER reinvest (rank bump):**
```
EMBED_BITS: 7 → 6
LQER_RANK: 4 → 5         (~+50 KB; covers higher residual on tok_emb)
LQER_TOP_K: 3            (unchanged)
```

**Arm H3 — EMBED_BITS=6 + LQER reinvest (rank + top_k):**
```
EMBED_BITS: 7 → 6
LQER_RANK: 4 → 5
LQER_TOP_K: 3 → 4        (~+30-50 KB; one more tensor gets correction)
```

## Code changes

None. Pure env-var override + RESUME_FROM_CKPT.

## Hardware ladder

- 4×H100, eval-only via RESUME_FROM_CKPT (no re-train).
- ~5-10 min wall per arm, ~$1-3 each.

## Seed plan

1 seed (42) per arm.

## Stop-early criteria

- post-quant val_bpb > 1.080 (catastrophic INT6 noise) → kill that arm
- LQER serialize fails → check that LQER has tok_emb in its top-K (default yes); kill if no

## Cost estimate

~$5 for all three arms.

## Run command (per arm)

```bash
# Arm H1
SEED=42 RUN_LABEL=seed_42_H1 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
  EMBED_BITS=6 \
  bash tmp_exec/launch_060_eval.sh

# Arm H2 (recommended baseline test)
SEED=42 RUN_LABEL=seed_42_H2 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
  EMBED_BITS=6 LQER_RANK=5 \
  bash tmp_exec/launch_060_eval.sh

# Arm H3
SEED=42 RUN_LABEL=seed_42_H3 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
  EMBED_BITS=6 LQER_RANK=5 LQER_TOP_K=4 \
  bash tmp_exec/launch_060_eval.sh
```

## Phase 2 followup (NOT in 060H)

If H2 or H3 lands at val_bpb ≤ 060A + 0.002 with bytes saved, **stack with 060G (Partial SpinQuant)** — SpinQuant rotation should recover most or all of the INT6 quality regression while keeping the byte savings. Combined target: 060A val_bpb − 0.005, fitting comfortably under cap.

## Open questions

1. **LQER actually covers tok_emb?** From spec 046K: "tok_emb has biggest residual" → it's at the top of LQER_TOP_K=3 at default. Confirm in our 060A logs.
2. **Pergroup compression on INT6 embed**: per-group block size may need adjustment for the smaller tensor footprint. Verify after first H1 run.
