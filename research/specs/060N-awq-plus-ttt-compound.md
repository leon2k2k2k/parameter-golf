# Spec 060N — AWQ-lite + 4 TTT phases + 3000 prefix + 2 global-SGD epochs (compound)

**Date:** 2026-04-29
**Slug:** `060N-awq-plus-ttt-compound`
**Idea sources:** `1908-awq-lite-mixed-bit-gptq.md`, `ttt-budget-reinvestment.md`, `1925-matrix-lr-ttt-prefix-tune.md`
**Branch:** `exp/060N-awq-ttt-compound` — forks from `research`. Code change is the AWQ-lite port from PR #1908 (same as 060I); TTT bumps are env-var only.
**Pinned SHA:** TBD on first commit of `exp/060N-awq-ttt-compound` (will pin after AWQ-lite port merges into a clean commit).
**Parent artifact:** `runs/060A-1855-port/seed_42/final_model.pt` on the NA-1 volume (or `seed_42_4h/` if 8H seed-42 hasn't been promoted yet — confirm at preflight).

## Hypothesis

Stack four predicted-additive levers on top of 060A:

1. **AWQ-lite mixed-bit GPTQ** (from PR #1908): top-1 saliency-scored 64-col group per quantized matrix at int8, rest int6. Quant-time change.
2. **`PHASED_TTT_NUM_PHASES`: 3 → 4.** One additional cumulative global-SGD update inside the prefix.
3. **`PHASED_TTT_PREFIX_DOCS`: 2500 → 3000.** More val tokens fed into global SGD (extends the lever #1855 greedy-validated at 2000→2500).
4. **`GLOBAL_TTT_EPOCHS`: 1 → 2.** Each chunk inside each global-SGD pass runs two epochs instead of one.

Predicted independent Δ for each is below seed noise (~−0.0002 to −0.002 BPB); the bet is that they compound additively into a single signal-detectable Δ above noise (~−0.001 to −0.005 BPB).

## Baseline

060A `seed_42_4h` post-TTT val_bpb (~1.0592 from `runs/060A-1855-port/seed_42_4h/train.log`). Compound target: clear our project baseline 1.06549 with margin (≤ 1.0625 on a single 8H seed would be a green-light to 3-seed).

## Expected Δ

**−0.001 to −0.005 BPB** post-TTT vs 060A `seed_42_4h`. Wide band because (a) phased base has no measurement of any single lever yet, (b) compound interaction unknown, (c) seed-noise floor ~0.0009 means single-seed reads will be noisy.

## Accept criteria

- **Phase 1 (smoke, 1 seed @ 8×H100):**
  - eval wallclock ≤ 580 s (HARD; if > 580 s, abort and re-run with safe fallback).
  - post-TTT val_bpb ≤ 1.0625 → green-light Phase 2.
  - post-TTT val_bpb in (1.0625, 1.0660] → ambiguous; consult before Phase 2.
  - post-TTT val_bpb > 1.0660 → kill, no Phase 2.
- **Phase 2 (3-seed @ 8×H100):**
  - 3-seed mean ≤ 1.06108 (#1855 published mean) → submission candidate.
  - 3-seed mean > 1.06549 (project baseline) → kill.

## Config diff vs 060A launch env

```
+ AWQ_LITE_ENABLED=1
+ AWQ_LITE_BITS=8
+ AWQ_LITE_GROUP_TOP_K=1
+ AWQ_LITE_GROUP_SIZE=64
  PHASED_TTT_ENABLED=3              (unchanged)
  PHASED_TTT_NUM_PHASES=3 → 4
  PHASED_TTT_PREFIX_DOCS=2500 → 3000
  GLOBAL_TTT_EPOCHS=(unset, default 1) → 2     # default verified in train_gpt.py line 334
```

All other env vars identical to `tmp_exec/launch_060A_4h_run.sh` (with `--nproc=8` and 8H matched bytes for 8H rung).

## Code changes

Same as spec 060I, lifted onto a clean commit on `exp/060N-awq-ttt-compound`:
- `_awq_lite_group_candidates(w, act_rms, group_size)` helper — ~15 LOC.
- Activation-RMS hook in GPTQ calibration — ~10 LOC.
- Per-group bit override in the GPTQ quantize loop — ~30–40 LOC.
- 4 hyperparam fields wired to env vars — ~10 LOC.

Total: ~80–100 LOC, all inside the GPTQ/serialize path. **No training-time code touched.**

## Hardware ladder

- **Smoke (1 seed @ 8×H100):** EVAL_ONLY via RESUME_FROM_CKPT off 060A's `final_model.pt`. Re-quantize + run phased TTT eval. Estimated wallclock ≤ ~480 s + GPTQ recalibration overhead. ~$2–3.
- **Phase 2 (3 seeds @ 8×H100):** EVAL_ONLY × 3 over the same parent ckpt. Total ~$6–9.

**Skip 4H smoke.** Eval-only on the parent .pt is already cheap; 4H wallclock doesn't translate to the 600 s submission cap.

## Safe-fallback variant — 060N-safe

If Phase 1 wallclock > 580 s on 8H, rerun with **`GLOBAL_TTT_EPOCHS=1`** (drop only that lever; keep AWQ-lite + phases=4 + prefix=3000). Predicted wallclock ~440–450 s on 8H. All other settings identical. No additional spec doc — same artifact, single env-var flip.

## Seed plan

- Phase 1: seed 42 only (smoke).
- Phase 2 (conditional): seeds 42, 0, 1234 — same suite as 060A and #1855 published.

## Inputs

- Parent ckpt: `/workspace/parameter-golf/runs/060A-1855-port/seed_42/final_model.pt` (NA-1; verify with execution session at preflight — fall back to `seed_42_4h/` if 8H ckpt isn't yet on volume).
- Train code: `exp/060N-awq-ttt-compound` HEAD, fork of `records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py` from SHA `da50cd6`.
- Tokenizer + dataset: same NA-1 paths as 060A (CaseOps SP8192).

## Checkpoints to emit

- `final_model.int6.ptz` — re-quantized artifact (the actual submission file, with AWQ-lite int8 groups).
- `final_model.pt` — copy of 060A's pt (chain-of-custody).
- `act_stats.pt` — per-layer activation RMS used for AWQ saliency (debug, <100 KB).
- `awq_selected.json` — which (layer, col_start, col_end) groups got int8 (debug, <10 KB).
- Normal `train.log` with `ttp:`/`ttpp:`/`ttpr:` lines for per-phase wallclock attribution.

## Stop-early criteria

- post-quant val_bpb > 1.080 → kill (catastrophic AWQ port regression).
- GPTQ NaN / serialize failure → kill, debug locally.
- artifact > 16,000,000 B → fail-hard. Reduce TOP_K or kill arm.
- eval wallclock > 580 s on the smoke run → abort & switch to 060N-safe (drop epochs).

## Cost estimate

- Smoke: ~$2–3.
- Phase 2 (3-seed eval-only): ~$6–9.
- Worst case incl. one 060N-safe re-run: ~$13.

## Extra artifacts beyond EXECUTION.md defaults

- `eval_phase_timings.txt` — extracted `ttp:`/`ttpp:`/`ttpr:` lines and per-phase wallclock breakdown. Auto-grep'd from `train.log` post-run.

## Open questions for interview

1. **Parent ckpt selection.** Is there a published 060A `seed_42` (8H, leaderboard-valid) on the volume, or is `seed_42_4h` all we have? If only 4H exists, smoke runs on the 4H ckpt and Phase 2 re-trains from scratch on 8H first — adds ~$24 to Phase 2 budget and pushes timeline past the 2026-04-30 deadline. **If 8H ckpt isn't on volume, skip 060N entirely and ship 060I solo.**
2. **AWQ-lite port: cherry-pick or hand-port?** PR #1908's diff is ~80 LOC. Cherry-pick the commits onto `exp/060N-awq-ttt-compound` if their tree is clean off our base; otherwise hand-port for a smaller diff (we want minimal blast radius for traceability).
3. **Failure-case for the smoke:** if Phase 1 lands in the (1.0625, 1.0660] grey zone — **do we still try the safe fallback** to see if dropping epochs helps, or do we kill? Recommend: kill grey-zone results — too close to deadline for two iterations on one seed.
4. **Failure-case for Phase 2:** if the 3-seed mean lands in (1.06108, 1.06549) — better than baseline but worse than #1855 — **do we ship as our submission anyway**, or hold for a tighter result? Recommend: ship if it's our best clean number on the 8H rung; we have no time to iterate further.
