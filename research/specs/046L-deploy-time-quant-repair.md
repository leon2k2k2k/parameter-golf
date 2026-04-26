# Spec 046L — deploy-time quant repair (use eval headroom, bypass byte cap)

**Slug:** `046L-deploy-time-quant-repair`
**Created:** 2026-04-27
**Status:** READY (code landed in commit `fcb816f`)
**Branch:** `exp/046-quant-repair`
**Commit:** `fcb816f`
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

## Pre-requisite checklist (all complete)

- [x] **Verified rules-legality** — challenge README §"What are the restrictions on evaluation?" says: "you're free to evaluate however" + "we encourage competitors to push the bounds of evaluation methods"
- [x] **AR self-gen confirmed no-val-leak** — generates from BOS, no external data needed
- [x] **Code written and pushed** — commit `fcb816f`:
  - `_generate_ar_batch_for_repair()`: AR sampling with eager forward (no KV cache)
  - `fit_passthrough_on_ar_gen()`: full pipeline (gen → fit → return)
  - Wired in `train_and_eval()` after `deserialize()`, before compile
- [x] **Verified passthrough params are nn.Parameter** — `named_parameters()` picks them up correctly

## Launch form (Phase 1 — single 64K-token arm)

Reference checkpoint: `/workspace/runs/045-loop-layer-improvements/armD/final_model.pt`

```bash
# Standard armD env vars (load checkpoint architecture matches)
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
# armD's spec 045 levers (must match training-time)
export LOOP_ITER_EMBEDS=0 MLP_ONLY_FROM_PASS=0 LOOP_SCALE_INIT=recip
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=1000
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
# baseline quant config (don't change for the test)
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=1
export RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0

# resume from armD checkpoint
export RESUME_FROM_CKPT=/workspace/runs/045-loop-layer-improvements/armD/final_model.pt

# *** the new lever ***
export DEPLOY_TIME_REPAIR_ENABLED=1
export DEPLOY_TIME_REPAIR_BATCHES=16        # 16 × 8 × 512 = 65,536 AR tokens
export DEPLOY_TIME_REPAIR_ITERS=5
export DEPLOY_TIME_REPAIR_LR=1e-3
export DEPLOY_TIME_REPAIR_AR_SEQ_LEN=512
export DEPLOY_TIME_REPAIR_AR_TEMP=1.0

export RUN_ID="046L-arself-64k-iters5-lr1e3"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/046L-arself-64k-iters5-lr1e3

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/046L-arself-64k-iters5-lr1e3/train.log 2>&1
```

## What to watch in the log

- `postquant_fit:ar_gen progress=N/16 tokens_so_far=...` — AR generation progress
- `postquant_fit:ar_gen done 65536 tokens in Xs` — verify total tokens generated
- `postquant_fit: fitting N params (M elements) over 5 iters` — verify ~32K elements
- `postquant_fit:iter=N/5 avg_ce=X.XXXX` — loss trace; expect monotonically decreasing
- `diagnostic quantized val_loss:Y val_bpb:Z` — the headline number
- Total wallclock: expect ~10-12 min

## Acceptance

Reference = 046 verification quantized = **1.07467**.

- **Strong win**: quantized < 1.0735 (-0.0012 BPB) — paradigm unlock
- **Win**: quantized < 1.0739 (-0.0008) — meaningful real improvement
- **Marginal**: 1.0739–1.0746 — directionally positive but tiny
- **Null**: 1.0746–1.0750 — passthrough capacity insufficient (matches 046E lesson)
- **Hurts**: > 1.0750 — AR-gen distribution mismatch with real val
- **Diverges**: NaN or `avg_ce` increasing — bug in fit (kill arm, debug)

## Cost

Single arm: ~$2-3 (10-12 min on 4×H100 NE-1).

## Phase 2 (only if Phase 1 wins)

If Phase 1 shows ≥-0.0008 BPB win, follow up with parameter sweep:

| Arm | Iters | LR | Batches (× 4096 tokens each) |
|---|---|---|---|
| 046L-iters3-lr1e3 | 3 | 1e-3 | 16 |
| 046L-iters10-lr1e3 | 10 | 1e-3 | 16 |
| 046L-iters5-lr3e4 | 5 | 3e-4 | 16 |
| 046L-iters5-lr3e3 | 5 | 3e-3 | 16 |
| 046L-batches8 | 5 | 1e-3 | 8 (32K tokens) |
| 046L-batches32 | 5 | 1e-3 | 32 (128K tokens) |

Phase 2 cost: ~$15-20 across 6 arms.

## Phase 3 (if Phase 2 finds meaningful winner)

Productionize: re-serialize artifact with fitted passthrough values baked in, so
the win is permanent and doesn't need to re-run at deploy time on every eval.
Spec separately as 046M after Phase 1+2 results.

## Post-arm checklist

- [ ] Pre-quant val_bpb unchanged at 1.06546 (sanity — fit shouldn't affect pre-quant)
- [ ] Quantized val_bpb logged
- [ ] Total wallclock ≤ 15 min (else AR gen too slow; use smaller batches or KV cache)
- [ ] Submission size still ≤ 16,000,000 bytes (this should be unchanged; fit doesn't add bytes)
