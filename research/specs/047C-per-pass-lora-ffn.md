# Spec 047C — Per-pass LoRA on looped FFN (rank-2, both sites)

**Slug:** `per-pass-lora-ffn`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/047C-per-pass-lora-ffn` (forked from `exp/045-loop-layer-improvements` @ `ece7b76`)
**Commit:** `0826944` (activation-side LoRA fix; 70013c5 = launch SHA update)
**Links to:** `research/ideas/per-pass-lora-ffn.md`

## Hypothesis

Loop45 band currently uses fully tied FFN weights across all 3 passes. ALBERT
(1909.11942 Table 4) shows FFN-tying is the dominant cost in tied/recurrent
transformers (~1.4–2.8 avg-task drop), while attention-tying is approximately free.
Relaxed Recursive (2410.20672) and MoLoRA (2512.12880) recover most of the
FFN-tying gap on tied stacks by adding a small per-pass LoRA delta.

Adding rank-2 LoRA on both `mlp_up` and `mlp_down` of every (pass, layer) in the
band gives each pass a small subspace of freedom on top of the shared FFN. We
expect this to recover a measurable fraction of the loss that pure FFN tying
imposes.

This is the matrix-valued generalization of frozen α/β (rank-0 scalar per pass).

## Baseline

**Spec 045 armAC-fix-rerun**, 4×H100, 1 seed, post-EMA pre-quant:
- val_bpb = **1.06479** at step 5126
- val_loss = 2.33028
- ~20 min wallclock

This is the matched-rung baseline. Comparison is at training endpoint, post-EMA,
**no quant** (per `[Screen via training-endpoint val_bpb]` memory).

## Expected Δ

- Pre-quant post-EMA: **−0.0005 to −0.0030** vs 1.06479
- Confidence: medium-low. Lit signal is positive but at much larger scale (Relaxed
  Recursive on Gemma-2B; MoLoRA on ALBERT-base). Small-LM regime + 3-pass-only
  recurrence is novel.
- Negative result is informative: would confirm pure tying is not the binding
  constraint at K=3, send us back to schedule/band-position levers.

## Accept criteria

- Wait-and-see — no fixed kill threshold (per user direction).
- Read the val_bpb curve at matched steps + post-EMA bpb. Compare to baseline 1.06479.
- Decision rubric (post-hoc):
  - **Promote:** post-EMA Δ ≤ −0.0010 → write 047C-mini eval, run a second seed,
    plan int4 storage variant for submission rung.
  - **Iterate:** Δ ∈ (−0.0010, 0): try `r=4` if signal direction is right but
    weak, or up-only (cheaper) if compute is the constraint.
  - **Kill:** Δ ≥ 0 — pure tying is not the binding constraint at K=3.

## Code changes

**Branch:** `exp/047C-per-pass-lora-ffn` from `exp/045-loop-layer-improvements`.
**Commit:** `0826944` (activation-side LoRA).

**New parameters** added to `GPT.__init__`:
```python
# r = LOOP_FFN_LORA_RANK; 0 = disabled, byte-identical to baseline
# num_passes = num_loops + 1; num_looped = loop_end - loop_start + 1
self.loop_ffn_up_lora_A   = nn.Parameter(  # (P, NL, hidden_dim, r)
    torch.empty(num_passes, num_looped, hidden_dim, r))
self.loop_ffn_up_lora_B   = nn.Parameter(  # (P, NL, r, model_dim)
    torch.empty(num_passes, num_looped, r, model_dim))
self.loop_ffn_down_lora_A = nn.Parameter(  # (P, NL, model_dim, r)
    torch.empty(num_passes, num_looped, model_dim, r))
self.loop_ffn_down_lora_B = nn.Parameter(  # (P, NL, r, hidden_dim)
    torch.empty(num_passes, num_looped, r, hidden_dim))
```

**Init (standard LoRA):** A ~ Kaiming-uniform, B = 0 → initial delta is exactly 0,
training-loss neutral at step 0.

**Forward — activation-side LoRA (compile-safe):**

Weight-side LoRA (`up_w + delta`) changes the kernel identity passed to the
fused MLP, triggering a full Triton re-autotune at loop activation. Instead,
deltas are pre-folded in eager (`A@B` via `torch.bmm` before `model.forward`),
passed as forward() kwargs (guarded by shape/dtype only — no per-step recompile),
and applied AFTER the base MLP on the activation:

```python
# In Block.forward — base MLP uses original nn.Parameter (cache hit)
mlp_out = self.mlp(x_normed, up_w, down_w)
if lora_delta_up is not None:
    x_flat = x_normed.reshape(-1, x_normed.shape[-1])
    h_lora = self.mlp._activate(F.linear(x_flat, lora_delta_up))
    mlp_out = mlp_out + F.linear(h_lora, lora_delta_down).reshape_as(mlp_out)
```

K=512 and K=2048 are standard Triton-friendly shapes — autotuned once, then
cached. TTT eager path (`forward_ttt`) keeps weight-side LoRA via
`_apply_loop_ffn_lora` (runs outside `torch.compile`, no Triton concern).

**Optimizer routing — divergence from interview answer.** Spec interview said
"Muon for matrix params." On implementation, the LoRA matrices have extreme
aspect ratio at small r (e.g. up_A is `(9, 2048, 2)` — only r=2 columns).
Muon's NS5 orthogonalization is meaningless at r=2 and the per-bank scale
factor `sqrt(2048/2) = 32` would be aggressive. The original LoRA paper uses
Adam. Routed to `scalar AdamW` group instead, matching the `recur_alpha` /
`loop_resid_mixes` precedent for tiny matrix params on the GPT root. If r is
raised to ≥16 in a follow-up, revisit Muon routing.

**Activation flag:**
- `LOOP_FFN_LORA_RANK=0` → disabled, byte-identical to baseline. Default.
- `LOOP_FFN_LORA_RANK=2` → activates rank-2 LoRA on both sites.

## Config diff vs 045 armAC-fix-rerun

```
LOOP_FFN_LORA_RANK=2     # new; default 0
```

All other env vars inherit armAC-fix-rerun.

## Param accounting (FP)

| Site | Per (pass, layer) | × 3 passes × 3 layers | Total |
|---|---|---|---|
| `up_A` (hidden_dim, r) | 2048×2 = 4096 | ×9 | 36,864 |
| `up_B` (r, model_dim) | 2×512 = 1024 | ×9 | 9,216 |
| `down_A` (model_dim, r) | 512×2 = 1024 | ×9 | 9,216 |
| `down_B` (r, hidden_dim) | 2×2048 = 4096 | ×9 | 36,864 |
| **Total** | 10,240 | ×9 | **92,160** |

At FP16 = ~184 KB; at int4 (LQER-style, future) = ~46 KB.

For this screen we don't quantize or submit, so FP is fine.

## Hardware ladder

**4×H100 mini, 1 seed only.** No 8×H100 in this spec — promotion to 8H is a
follow-up after eval.

Per `[Smaller-H smoke before full trial]` and `[Use Parameter Golf pod template]`:
- Pod template: `--template-id y5cejece4j`
- Region: **NE-1 preferred** (per `[Prefer NE-1 region for pods]`, flipped 2026-04-27); fall back to JP
- Speed check at step 500–1000: ≥ 4.30M tok/s pre-loop (per `[Pod throughput is per-pod not per-region]`)

**Prewarm:** new commit → loop-active autotune graph is cold → must prewarm or
accept `TRITON_AUTOTUNE_NUM_RUNS=1` (~6% throughput hit). For a screen, accept
the throughput hit; full prewarm is wasted on a non-submission run.

## Launch plan

Two scripts on the worktree (`exp/047C-per-pass-lora-ffn` @ `e2f7a3d`):

- `tmp_exec/launch_047C_smoke.sh` — 200 steps, 180s wallclock, ~$0.30. Exercises
  the LoRA path past loop-activation (frac=0.35 of 180s ≈ 63s).
- `tmp_exec/launch_047C.sh` — full screen, ~22 min, ~$2.30. Restores fc54262
  cache, sets `TRITON_AUTOTUNE_NUM_RUNS=1`, background-stashes to
  `/workspace/.inductor_cache_5cf60f9` after 10 min for promotion-path warm start.

**Smoke gate before full screen:**
1. No NaN / divergence
2. step_time pre-loop within 10% of baseline (≥4.30M tok/s on 4×H100)
3. Loss at step 0 matches AC-fix-rerun within rounding (LoRA delta = 0 at init)

**Compile watch during full run:**
- Expected: 10–30s step_time spike when looping flips on at frac=0.35
  (~step 1750). Some inductor recompile on the cold LoRA subgraph paths.
- KILL if step_time > 5× baseline for >10 steps post-spike — graph is mis-
  cached; need a proper prewarm before retrying.

## Failure-case branches

1. Smoke fails → stop pod, debug code, do not proceed.
2. Mid-run compile pause >60s → stop, build `prewarm_5cf60f9.sh`, retry hot.
3. Δ ≥ 0 → KILL 047C; per-pass FFN freedom thesis wrong on our stack.
4. Δ ∈ (-0.001, 0) → ITERATE: try `r=4` (after a proper prewarm).
5. Δ ≤ -0.001 → PROMOTE: prewarm 5cf60f9, run multi-seed (3 seeds), then
   plan int4 storage wiring for submission.

## Seed plan

- Single seed for screen (seed = 314 to match 045 AC-fix-rerun lineage).
- Multi-seed deferred to promotion path.

## Inputs

- Data: `/workspace/parameter-golf/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/`
- Tokenizer: `/workspace/parameter-golf/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- Hotstart: same as armAC-fix-rerun (path TBD from launch script)

Path layout matches both JP and NE-1 per `[JP and NE-1 volume layout]`.

## Checkpoints to emit

- `final_model.pt` (FP, post-EMA) — for promotion analysis
- `train.log` — full training log
- (No INT6 quant artifact this spec — screen only)

Retention: under `runs/047C-per-pass-lora-ffn/seed_314/`.

## Stop-early criteria

- NaN in train_loss or val_loss → stop, file in tmp_exec, alert.
- step_time > 1.5× baseline at step 1000 → stop and investigate (likely LoRA
  forward inefficiency — fix `A@B@x` ordering).
- val_bpb at step 4500 > baseline + 0.005 → likely a divergence; stop and inspect.

Otherwise: run to wallclock cap (matched to baseline ~20 min).

## Cost estimate

- 4×H100 ≈ $5.50/hr
- ~20 min training + ~5 min eval/cleanup = 25 min
- **~$2.30 per arm × 1 arm = ~$2.30** (no prewarm, no second seed in this spec)

## Monitoring

Per `[1-min polling with comparison table]` and `[30s polling cadence]`:
- 1-min cron from launch.
- Compare table: 047C arm vs 045 AC-fix-rerun baseline (cached log) at matched
  steps. Show train_loss, step_time, tok/s.
- Auto-stop on completion + report final EMA pre-quant val_bpb.
- Alert on stopping_early or NaN.

## Open questions for interview

1. **LoRA fold-in.** Resolved: additive on the linear weight (`up_w + A@B`,
   `down_w + A@B`), reusing the existing fused MLP kernel. Implementation in
   `_apply_loop_ffn_lora` helper.
2. **Hotstart compatibility.** New LoRA params are zero-init at construction;
   if checkpoint omits them they keep the zero-init. Verify load_state_dict
   uses `strict=False` or that `_rebank_state_dict` ignores missing keys.
   Pre-flight check before launch.
3. **Prewarm strategy.** New code path → loop-active graph is cold. For this
   single-arm screen, accept `TRITON_AUTOTUNE_NUM_RUNS=1` (~6% throughput hit).
   If 047C wins, prewarm before any follow-up arm.
4. **Quant/serialize wiring.** Out of scope for this screen
   (`TRAINING_ONLY_SCREEN=1` skips serialize/quant). Follow-up work if 047C
   wins: passthrough fp16 (or int4 LQER-style) for the 4 LoRA banks. They sum
   to ~92K params at r=2 → above the 65536 auto-passthrough threshold, so
   explicit handling will be needed.

## Extra artifacts

- Run command saved under `tmp_exec/launch_047C.sh` (to be created with code).
- Inductor cache stash to `/workspace/inductor_cache_stash/047C/` for follow-up arms.
