# Spec 060J — TTT phases 3 → 4 on 060A baseline (eval-only)

**Date:** 2026-04-29
**Slug:** `060J-ttt-phases-up`
**Idea:** `research/ideas/ttt-budget-reinvestment.md`
**Branch:** `research` (config-only; no code change)
**Pinned SHA:** `da50cd6` (060A's commit; train_gpt.py at `records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py` on that SHA)

## Hypothesis

Adding one TTT phase (`PHASED_TTT_NUM_PHASES 3 → 4`) consumes ~50-65s of the unused 100-180s eval budget. PR #1812 reported −0.008 BPB from a +1 TTT epoch bump on a weaker base; phase increment is a similar lever. Even with heavy absorption on our stronger base, **−0.0005 to −0.002 BPB** is plausible.

## Baseline

060A (single seed, `seed_42_4h`, 4H matched-FLOPs research run; prequant val_bpb ~1.0644, post-quant ~1.0721, post-TTT ~1.0592 — read off `runs/060A-1855-port/seed_42_4h/train.log`).

## Expected Δ

**−0.0005 to −0.002 BPB** post-TTT (vs `seed_42_4h` post-TTT number).

## Accept criteria

- post-TTT val_bpb ≤ (060A_seed_42_4h post-TTT − 0.0005)
- eval wallclock ≤ 600s (HARD; disqualifies submission if blown)
- artifact size unchanged (~15.9 MB; no quant change since same .pt)
- TTT loss curve shows phase 4 reduces train loss further (sanity)

## Config diff vs 060A

```
PHASED_TTT_NUM_PHASES: 3 → 4
```

All other TTT knobs unchanged from 060A baseline:
```
TTT_ENABLED=1
PHASED_TTT_ENABLED=3
PHASED_TTT_PREFIX_DOCS=2500
TTT_BETA2=0.99
TTT_WEIGHT_DECAY=0.5
TTT_LORA_RANK=80
```

## Code changes

None. Pure env-var override + RESUME_FROM_CKPT.

## Hardware ladder

- 4×H100 eval-only via RESUME_FROM_CKPT; ~7-12 min wall, ~$1-2.
- No 8H rung — single-arm screen.

## Seed plan

Single seed (42). If accept criteria hit, escalate to 3-seed cap-safety screen as 060J-phase2 (only after 060A 8H baseline runs exist).

## Inputs

- **Parent ckpt:** `/workspace/runs/060A-1855-port/seed_42_4h/final_model.pt` (from `runs/060A-1855-port/seed_42_4h/`; 4H matched-FLOPs research run)
- **Tokenizer:** `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- **Val data:** `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin`
- **Val bytes:** `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin`
- **Train script:** `records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py` at SHA `da50cd6`

## Checkpoints to emit

- `runs/060J-ttt-phases-up/seed_42/final_model.int6.ptz` — re-quantized artifact (same int6 weights as 060A, just re-serialized; cap check ≤ 16 MB)
- `runs/060J-ttt-phases-up/seed_42/eval.log` — full stdout/stderr from torchrun

## Stop-early criteria

- `eval_time_s` projects > 600s mid-run (e.g., phase 1 takes >150s) → kill arm
- post-TTT val_bpb > 060A_seed_42_4h post-TTT → kill (regression; signal that more TTT overfits)
- TTT NaN or divergence → kill
- artifact > 16,000,000 bytes → fail (shouldn't happen since no quant change, but enforce)

## Cost estimate

~$2 (single 4×H100 eval-only run).

## Run command

```bash
SEED=42 RUN_LABEL=seed_42 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42_4h/final_model.pt \
  ARM=060J-ttt-phases-up \
  PHASED_TTT_NUM_PHASES=4 \
  bash tmp_exec/launch_060_eval.sh
```

## Extra artifacts

None beyond defaults.

## Open questions for interview

1. **Phase 4 prefix slice:** does TTT slice prefix disjointly across phases, or all share the 2500-doc prefix? If shared, +1 phase = one more pass over same data; if disjoint, +1 phase consumes more prefix forward time.
2. **Phase 4 LR schedule:** does TTT LR decay across phases? If LR is already small in phase 3, phase 4 may underfit and add cost without gain.
3. **Failure mode if eval > 600s on the probe:** stop and report wallclock breakdown so we know whether to halve another knob, or kill the entire phases-up direction.
