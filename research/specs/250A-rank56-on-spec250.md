# Spec 250A — Eval-only TTT_LORA_RANK 80 → 56 on spec 250 outputs

**Date:** 2026-05-01 (deadline window)
**Idea:** combination of PR #1935's rank=56 lever + spec 250's V21+slope0.3 base.
**Lineage:** Spec 250 outputs (PR #1953 verbatim + LeakyReLU² slope 0.3, 3 seeds)
+ single env var override at eval time. **Eval-only — no retrain.**

## Hypothesis

PR #1935 ablated `TTT_LORA_RANK ∈ {48, 56, 64, 80, 96}` on the PR #1855 base
(at QK_GAIN=6.0, ctx=2048, default TTT mask, TTT_LR=1.0) and found rank=56 was
the inverted-U optimum (1.05997 single-seed s42, claimed −0.00111 vs #1855's
3-seed mean). PR #1953/spec 250 changed four of those conditions
(QK=5.25, ctx=2560, TTT_MASK=no_qv, TTT_LR=0.75). The rank optimum may have
shifted on this stack — or may not transfer at all.

This spec tests the cheapest possible version of the combination: **same
trained checkpoint as spec 250, re-evaluated with TTT_LORA_RANK=56 at eval
time only.** No retrain. No code change. One env var.

## Baseline

**Spec 250 same-seed post-TTT BPB.** Per-seed comparison only — not vs
3-seed mean of either parent. The eval-only override produces a paired
delta on the *same* `final_model.pt`, so noise cancels.

Reference 3-seed targets (from spec 250 hypothesis): mean 1.0578 ± 0.0006.
PR #1953 reference: 1.05855 (3-seed). PR #1935 reference: 1.05997 (s42).

## Expected Δ vs spec 250 (same seed)

**0 to −0.0005**, low confidence.

- For: rank=56 was a clean optimum on #1855 base, smaller LoRA tightens TTT
  regularization, mechanism plausibly transfers.
- Against: spec 250 already adds a TTT regularizer (`TTT_LR=0.75` was 1.0),
  so rank=56 is a *second* regularizer in the same direction — possible
  over-regularization. Also, `TTT_MASK=no_qv` removes Q/V LoRA paths;
  remaining K and MLP LoRAs may already be over-budgeted at rank 80, in
  which case the optimum is even lower than 56. Not testing other ranks
  this round to keep cost bounded.

Author estimate of P(win ≥ −0.0003 same-seed): **~25-35%**.

## Accept criteria

- **Per-seed:** post-TTT BPB at rank=56 ≥ −0.0003 below same-seed spec-250
  result (i.e., ≥ noise threshold below). Compute paired Δ per seed; report
  mean Δ across 3 seeds.
- **Mean across 3 seeds:** if mean Δ ≤ −0.0005, **promote**: submit the
  rank=56 version. If 0 ≥ mean Δ > −0.0005, **wash** — submit spec 250
  unchanged. If mean Δ > 0, **kill** — don't submit rank=56.
- All seeds clear 600s eval cap and 16 MB artifact cap (artifact size
  unchanged from spec 250 since LoRA is eval-only and scratch).

## Config diff (vs spec 250)

Single env var override; everything else verbatim from spec 250:

```
TTT_LORA_RANK = 56     # was 80
```

All other env vars identical to `tmp_exec/launch_<spec250>_run.sh` (or whatever
spec 250's execution session names its launch script). In particular keep:
- `EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560 TTT_MASK=no_qv TTT_Q_LORA=0
  TTT_V_LORA=0 TTT_LOCAL_LR_MULT=0.75 QK_GAIN_INIT=5.25` (unchanged from 250)

Add:
- `RESUME_FROM_CKPT=/path/to/runs/250-1953-leakyrelu03/seed_<N>/final_model.pt`
- `EVAL_ONLY=1` (skip train; per memory `feedback_no_git_stash_u_with_artifacts`,
  EVAL_ONLY only loads `final_model.int6.ptz`, but training-side env vars
  already in launch script must remain set so model construction matches the
  saved checkpoint).

## Code changes

**None.** Same train code as spec 250 (`exp/250-1953-leakyrelu03` @ commit
pinned by spec 250). No `exp/<slug>` branch needed. This spec lives on
`research`.

## Hardware ladder

- **Mini:** SKIP. Eval-only override of a single env var — pattern validated
  in spec 113 ("eval-only via RESUME_FROM_CKPT works correctly when
  training-side code is unchanged"). No 2×H100 mini exercises the full
  600s eval + Phased TTT regime that's the actual signal here.
- **Official (8×H100):** 3 seeds (42, 0, 1234) **eval-only**. Each seed
  re-runs only the eval+TTT pipeline against spec 250's saved
  `final_model.pt`. ~150-180s of GPTQ application + ~470s of phased TTT eval
  per seed. Wallclock ~10-12 min per seed.

## Seed plan

Three seeds matching spec 250's seeds (42, 0, 1234) for direct paired
comparison. **Run seed 42 first as smoke**: if seed 42 paired Δ ≥ +0.0010
(meaningfully worse than spec 250 same-seed), abort 0 and 1234 — the lever
isn't transferring to V21. Otherwise launch 0 and 1234 sequentially or in
parallel given pod availability.

## Inputs

- **Required:** spec 250 must have completed and produced
  `runs/250-1953-leakyrelu03/seed_{42,0,1234}/final_model.{pt,int6.ptz}`
  before this spec can launch. Hard dependency. If 250 failed or any seed
  is missing, this spec can only run on whichever seeds completed.
- Tokenizer + data paths: same as spec 250 (CaseOps, fineweb10B sp8192).
- Train script: same as spec 250 (`records/track_10min_16mb/2026-04-30_spec250_LongCtx_NoQV_LeakyRelu03/train_gpt.py`).

## Checkpoints to emit

- Per-seed `submission.json` matching spec 250's schema, with the
  `val_bpb` and `quantized_val_bpb` fields reflecting the rank=56 eval.
- Per-seed `train.log` (eval-only — no actual training output, but the
  GPTQ + TTT + final eval lines should all be present).
- Final artifact bytes should be ≈ identical to spec 250 (LoRA is fitted
  fresh at eval; not stored). Sanity-check artifact size matches within
  ~10 KB.

Destination: `runs/250A-rank56-on-spec250/seed_{42,0,1234}/`.

## Stop-early criteria

- **Per-seed:**
  - Eval-only run errors during model load → kill seed (likely cause:
    LoRA module shape mismatch from rank change interacting with quantized
    weight load). If this fires, the eval-only assumption is false and the
    spec needs a code change — escalate to user before launching other
    seeds.
  - Eval wallclock > 580s → kill seed (cap-safety).
  - Post-TTT BPB sanity floor: > 1.080 → kill seed (something about LoRA
    rank construction is broken; spec 250 baseline is ~1.058).
- **Across seeds:** if seed 42 paired Δ ≥ +0.0010, abort 0 and 1234.

## Cost estimate

- ~$2-3 per seed re-eval (8×H100 × 12 min × $24/hr ≈ $4.8/seed; if pod
  start+stop overhead amortized across all 3 seeds, closer to $3/seed).
- 3 seeds total: **~$9-15** depending on pod-restart pattern.
- Keep pod warm between seeds (per memory
  `feedback_keep_pod_warm_between_specs`) — single pod, sequential.

## Extra artifacts

- Per-seed `paired_delta.txt`: one line `seed=<N> spec250_bpb=<X>
  rank56_bpb=<Y> delta=<Y-X>`.
- Mean Δ summary at end for promote/wash/kill decision.

## Open questions for interview

1. **Spec 250 status.** This spec depends on 250's checkpoints. Is 250
   already running, queued, or finished? Halt this spec's interview if
   250 hasn't produced any seeds yet — there's nothing to eval against.
2. **Pod region & capacity.** Same constraints as 250: NE-1 first, JP
   fallback, 8×H100 only. If neither has 8×H100, halt — eval-only doesn't
   justify hardware substitution either, since GPTQ application + Phased
   TTT both scale with GPU count and the 600s eval cap was sized for
   8×H100.
3. **Pod re-use from spec 250.** If 250's pod is still up after its run
   completes, can we re-use it directly for 250A's eval? Saves
   provisioning time.
4. **Seed scheduling.** Run all 3 seeds sequentially on one pod (~35-40
   min total wallclock), OR run paired-with-spec-250 (i.e., as soon as
   250 seed 42 finishes, start 250A seed 42 in parallel on a second pod
   while 250 seed 0 trains). Default: sequential after 250 fully completes.
5. **Failure handling.** If 250's one of three seeds failed and we only
   have 2 final_model.pts, do we (a) run 250A on the 2 we have, (b) wait
   to see if 250 retries, or (c) skip 250A entirely? Default: (a), but
   accept criteria mean-of-3 becomes mean-of-2.
6. **Submission decision.** If 3-seed mean Δ ≤ −0.0005 (promote), do we
   submit the rank=56 version? Default: yes, it's the better number on
   our 3 seeds. Confirm before submitting.

## Notes

- This spec is a **hedge**, not a frontier-mover. Expected upside is small
  (~−0.0005); expected downside is null (don't submit rank=56). EV is
  positive only because the cost is bounded at $9-15.
- This spec **strictly downstream** of spec 250 — cannot delay or break
  it. Even if 250A regresses, spec 250's submission is unchanged.
- No code change means no compile burst, no prewarm needed, no autotune
  rebuild. Eval-only via RESUME_FROM_CKPT is a well-trodden pattern (specs
  113, 060B-F all used it).
