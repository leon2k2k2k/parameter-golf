# Spec 046L — deploy-time quant repair (use eval headroom, bypass byte cap)

**Slug:** `046L-deploy-time-quant-repair`
**Created:** 2026-04-27
**Status:** DRAFT — needs ~50-100 lines code (mostly wiring existing 046E/F)
**Branch:** `exp/046-quant-repair`
**Commit:** TBD (after code lands)
**Parent:** `research/ideas/quant-repair-fundamentally-new.md` (Idea A — highest EV)

## The fundamental shift

Every quant-repair lever we tried (046A-K) was **artifact-time**: the repair must
fit in the 16,000,000 byte cap. SDClip wins (-0.00086 to -0.00216 BPB) all
overflow because each tightening step costs ~250 KB of artifact size.

**This spec moves the repair to deploy-time** — runs on the leaderboard hardware
at eval time, using the **100-180s of unused eval budget** that PR #1797 leaves
on the table (PR uses 423-495s of 600s cap).

Deploy-time repair pays compute, not bytes. **Bypasses the byte-cap problem entirely.**

## Hypothesis

After deserialize loads the quantized model, run a small fit step that updates
the in-memory model's passthrough fp16 params (`attn_scale`, `mlp_scale`,
`resid_mix`, etc.) to compensate for accumulated quant error. Use AR self-generated
text as both calibration data AND optimization target (next-token CE loss).

This is similar in spirit to TTT but:
- Updates passthrough params (not LoRA adapters)
- Uses AR-self-gen text (not val data — no leak)
- Runs ONCE at deserialize, not per-document
- Total deploy-time cost: ~60s (fits in 100-180s headroom)

## Why this should work where 046E failed

046E's `fit_passthrough_params_to_match_base` fit AT ARTIFACT TIME on training
data, then DISCARDED the fitted values (in-memory only, didn't propagate to
artifact). Result: cost +0.0004 BPB with no shipping value.

Deploy-time fit:
- Values applied to the CURRENTLY-RUNNING model
- Directly impacts the eval being scored
- TTT works exactly this way and recovers ~-0.013 BPB

Whether the small per-channel param fit can do anything similar to TTT's full
LoRA fit is the open question.

## Code components

We already have most of the pieces from 046E + 046F:

1. **`fit_passthrough_params_to_match_base`** (046E, in train_gpt.py at commit
   381baf2) — currently fits to match `base_model.forward_logits`. Need to
   refactor: fit against AR-self-gen data + CE loss on next tokens.

2. **`ARSelfGenCalibLoader`** (046F, same commit) — already generates AR
   samples without val data leak. Reuse directly.

3. **New wiring** in `train_and_eval()` after `deserialize()`:
   ```python
   eval_model = deserialize(h, device)
   if h.num_loops > 0:
       eval_model.looping_active = True
       eval_model.looping_depth = h.num_loops

   # NEW: deploy-time quant repair
   if h.deploy_time_repair_enabled:
       repair_calib = generate_ar_calib(eval_model, h, n_batches=8, seq_len=512)
       fit_passthrough_to_self_consistency(eval_model, repair_calib, h)

   # then proceed to TTT + eval
   ```

4. **New objective**: instead of MSE-vs-teacher, use cross-entropy on next-token
   prediction over AR-generated tokens. The model trained on next-token
   prediction; if quant noise degraded that, fitting on its own AR samples
   should recover.

## Env vars

```bash
DEPLOY_TIME_REPAIR_ENABLED=1            # gate (default 0)
DEPLOY_TIME_REPAIR_ITERS=5              # AdamW iter count
DEPLOY_TIME_REPAIR_LR=1e-3              # learning rate
DEPLOY_TIME_REPAIR_BATCHES=8            # batches per iter
DEPLOY_TIME_REPAIR_AR_SEQ_LEN=512       # AR generation length
DEPLOY_TIME_REPAIR_AR_TEMP=1.0          # AR sampling temperature
```

Default OFF so existing runs unaffected.

## Rules legality (CRITICAL — verify before committing)

Deploy-time techniques are a gray area. Need to verify in the challenge rules:

1. **Is updating model parameters at eval time allowed?** TTT does this clearly. This is similar in spirit but updates a different set of params.
2. **AR-self-gen calibration data**: should be unambiguously legal (model generates from BOS, no val data touched)
3. **Total eval time**: must fit in 600s budget. Repair cost ~60s + TTT ~300-450s + final eval ~30s = ~390-540s. Fits.
4. **Determinism**: AR-gen with fixed seed should be deterministic. Verify.

**Action item before code work**: read `records/track_10min_16mb/README.md` and
the official challenge rules for what's legal at eval time. If unclear, ask in
the openai/parameter-golf discussions.

## Arms (after code lands)

### Phase 1 — sanity / cost check (1 arm)

| Arm | Config | Tests |
|---|---|---|
| **046L-baseline** | DEPLOY_TIME_REPAIR_ENABLED=1, defaults (5 iter × 8 batches × 512 seq) | does it run cleanly? does val_bpb improve at all? |

### Phase 2 — sweep iters/lr (3 arms)

| Arm | Iters | LR | Batches |
|---|---|---|---|
| **046L-iters3-lr1e3** | 3 | 1e-3 | 8 |
| **046L-iters5-lr3e4** | 5 | 3e-4 | 8 |
| **046L-iters10-lr1e3** | 10 | 1e-3 | 16 |

### Phase 3 — combine with SDClip win (1 arm)

If 046L baseline shows real BPB improvement at deploy time, we can use
deploy-time repair to compensate for an OVER-CAP SDClip variant if we can
also save bytes elsewhere. But without bytes, just test:

| Arm | Config |
|---|---|
| **046L-on-baseline-stack** | deploy-time repair on current legal baseline |

Total: 5 arms, ~$5, ~30 min wallclock.

## Acceptance

Reference = 046 verification (1.07467 quantized, no deploy repair).

Per arm:
- **Strong win**: quantized < 1.0735 (-0.0012)
- **Win**: quantized < 1.0739 (-0.0008)
- **Marginal**: 1.0739–1.0746
- **Null**: 1.0746–1.0750
- **Hurts**: > 1.0750
- **Bug**: NaN or fails to deserialize

Eval time check: `eval_time` log line MUST stay < 600s on the leaderboard simulation.

## Risk assessment

- **Rules legality**: medium — could be ruled out, kills direction
- **Implementation**: low-medium — most code exists in 046E/F; mainly refactoring
- **Outcome uncertainty**: medium-high — 046E artifact-time was negative; deploy-time MIGHT also be null if the passthrough params genuinely can't compensate for matrix quant error
- **Eval-time budget overflow**: low — 60s addition fits in 100-180s headroom

## What success looks like

If 046L-baseline gives ANY net positive (>0.0005 BPB win), this is a paradigm
unlock:
- Future submissions can spend ~60s of eval time for free BPB
- Combined with eventual SDClip+byte-saving wins, could be substantial
- New direction worth deeper exploration (different objectives, more iters,
  different param sets)

If 046L-baseline gives ~null:
- Confirms passthrough param capacity is too small to meaningfully repair
  quant error (matches 046E artifact-time finding)
- Direction closes — deploy-time only helps if you have MORE expressive
  params to update (e.g., LoRA, which is what TTT does)
- We then know definitively that quant repair via passthrough params is a
  dead direction in any setting

## Cost summary

- Code: ~30-60 min (refactoring existing 046E/F into eval path + new objective)
- Test: ~$5 across 5 arms
- Total time: ~1 hour engineering + ~30 min wallclock

## Decision tree

| Outcome | Next |
|---|---|
| Rules-illegal | Direction closed; pivot to other fundamentally-new ideas (B, C, E from quant-repair-fundamentally-new.md) |
| 046L-baseline wins ≥ -0.001 | Sweep params (iters, LR); explore combining with TTT |
| 046L-baseline ~null | Try a richer param set (e.g., a tiny LoRA-style adapter); else close direction |
| 046L hurts | Bug or fundamental incompatibility; investigate |

## Pre-requisite checklist

- [ ] Verify rules-legality of deploy-time param updates (beyond TTT's LoRA)
- [ ] Read challenge README for any restrictions
- [ ] Confirm AR-self-gen calib qualifies as no-val-leak
- [ ] Plan code: refactor `fit_passthrough_params_to_match_base` to use CE-on-AR objective
- [ ] Verify deserialize leaves passthrough params trainable (currently they're loaded as fp16 buffers; need to mark as Parameter)
