# Spec 060D — 046G-tighter SDClip on 060A baseline (eval-only via RESUME_FROM_CKPT)

**Date:** 2026-04-29
**Branch:** `research` (config-only; no code change)
**Parent:** 060A `final_model.pt` + 060B if 060B passes (suggests SDClip direction holds).

## Hypothesis

046G measured **−0.00146 BPB** by tightening SDClip another step (each clip −1.0σ further). 060B applies one step (ATTN 13.0→12.5); 060D applies the next step (everything −1.0σ from 060A defaults). 046G arms were ALL ILLEGAL (+428 KB over cap on 045-armD); with #1855's lrzip headroom we have ~280 KB of budget — likely still over. If 060B fits with margin, we can probably afford only ATTN+EMBED tighter, not MLP.

## Baseline

060A (or 060B if 060B passes).

## Expected Δ

**−0.0015 BPB** vs 060A if all three clips tighten; **−0.0010 BPB** if only two (artifact-fit constraint).

## Accept criteria

- post-quant + post-TTT val_bpb ≤ (060A − 0.0010)
- artifact size ≤ 16,000,000 (HARD cap — must fit)
- no GPTQ failure or NaN

## Config diff vs 060A

Two arms (run sequentially; pick whichever fits + wins):

**Arm D-aggressive** (each clip −1.0σ further):
```
MLP_CLIP_SIGMAS:    11.5 → 10.5
ATTN_CLIP_SIGMAS:   13.0 → 12.0
EMBED_CLIP_SIGMAS:  14.0 → 13.0
```

**Arm D-conservative** (only ATTN+EMBED):
```
ATTN_CLIP_SIGMAS:   13.0 → 12.0
EMBED_CLIP_SIGMAS:  14.0 → 13.0
(MLP_CLIP_SIGMAS unchanged at 11.5)
```

## Code changes

None. Pure env-var override + RESUME_FROM_CKPT.

## Hardware ladder

- 4×H100, eval-only mode (RESUME_FROM_CKPT)
- ~5-7 min wall per arm, ~$1-2 each

## Seed plan

1 seed (42) per arm.

## Stop-early criteria

- Artifact > 16 MB → fail this arm; fall back to next arm
- post-quant val_bpb > 1.080 → kill

## Cost estimate

~$3-4 for both arms.

## Run command

```bash
# Try aggressive first; if oversize, fall back to conservative.
SEED=42 RUN_LABEL=seed_42_D_agg \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
  MLP_CLIP_SIGMAS=10.5 ATTN_CLIP_SIGMAS=12.0 EMBED_CLIP_SIGMAS=13.0 \
  bash tmp_exec/launch_060_eval.sh
```
