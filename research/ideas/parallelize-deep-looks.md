# Parallelizing the deeper recurrent looks — research thread

**Status:** OPEN, autonomous-loop-driven brainstorm. New ideas appended each
wake cycle (10 min). Mature ideas promoted to `research/specs/08x-*.md`.

## Hard rules for every spec promoted from this file

**EVERY spec written from this thread must be FROZEN before the wake ends.**
"Frozen" per `CLAUDE.md` checklist:
1. Spec file committed.
2. Code commit pushed (if code change), pinned hash in spec.
3. Spec change pushed (commit + `git push fork exp/046-quant-repair`).
4. No silent code changes (no uncommitted edits in worktree).
5. Verify clean: `git status` clean + `git ls-remote fork <branch>` matches.

**EVERY spec must explicitly verify NO mid-run recompile path.** Per memory
`feedback_no_mid_run_recompile`: mid-run recompile = run is dead. Per
memory `feedback_always_tensor_block_kwargs` and `feedback_spec_compile_checklist`:

- ITEM 0: **Always-tensor rule.** No Python `if`/`None` passthroughs of
  loop-only kwargs in the compile region. Use registered identity buffers
  (zeros/ones) for inactive steps. Single graph variant.
- ITEM 1: **No weight slicing in compile.** Column/row slices of an
  `nn.Parameter` (e.g. `weight[:, :h2]`) produce non-contiguous tensors
  that hang Triton silently. Fix: separate `nn.Parameter` banks in
  `__init__`.
- ITEM 2: **No narrow-K matmul** (K<16) inside compile — pre-fold in eager.
- ITEM 3: **No new shape dimensions appearing post-loop-activation** that
  weren't compile-warmed. (Loop-warmup of the looped graph is required if
  shapes change at activation.)
- ITEM 4: **`@dynamo.disable` for backward kwargs** that flow into
  control-flow conditionals.

For each spec, the **Code changes** section must include an explicit line:
`Compile-graph audit: <one-paragraph verification that no mid-run recompile
is possible>`. Reject any spec where this audit cannot be written
honestly.

**Cluster-level audit done at file open (W0):** the LOOP_PATTERN env var
(used by 071-074) is __init__-time only — it changes the
`encoder_indices`/`decoder_indices` Python lists once at module
construction. The compiled forward iterates the list using a Python `for`
(not a torch op), so each unique pattern produces one graph variant total
(at first invocation), no mid-run recompile. Verified.



**Date opened:** 2026-04-29 (overnight session)
**Goal:** find ways to compute "more recurrent depth" without paying linear
wallclock. Exploit the unspent compute resources we've identified:
- ~50% VRAM headroom on 4H *and* 8H (per spec 064b run analysis)
- ~180s unused eval-time wallclock (per `project_eval_time_quant_repair_idea`)
- NCCL 0.5% of GPU time (comm bandwidth nearly free)
- Matmul roofline at ~25% — meaning we're FLOPs-bound by the GEMMs themselves,
  not bandwidth or launch overhead

**Constraint to keep in mind:** the loop tax = layer-pass arithmetic. Adding
sequential passes is linear. The angle is finding **non-sequential ways to
extract extra recurrence value**.

## Wake log

- **W0 (2026-04-29 initial):** seed file, broad brainstorm
- (next wake will append below)

---

## W0 — Initial brainstorm (broad)

### Cluster A — Token-axis parallelism (run pass k on token i in parallel with pass k+1 on token i-1)

**A1. Parallel Loop Transformer port (arXiv 2510.24824).** Cross-Loop
Parallelism (CLP): build a "displaced micro-batch" where token i runs loop
pass 1 while token i-1 runs loop pass 2 simultaneously. Single forward pass
collapses L sequential loop iterations into L parallel slots.
- **Mechanism:** breaks the sequential dependency by recognizing that
  loop pass k of token i only depends on token i's previous-layer state,
  not on token i's loop pass k-1 (in some formulations).
- **Catch:** in our looped-block setup, pass k+1 of token i *does* depend
  on pass k of token i (the residual is updated each pass). So pure CLP
  may not apply — needs adaptation.
- **Adapted version:** if we cache token i's pass-k output, then pass-k+1
  of token i and pass-k of token i+1 can run in parallel (different
  computations, both on now-stale dependencies).
- **Eval-only variant:** at eval time, we have 180s headroom. Could
  prefill the eval sequence's loop-pass-1 outputs in one parallel pass,
  then loop-pass-2 in another, etc. Wallclock comes out same as
  sequential — but if we use it to *enable* deeper recurrence (NL=4
  or 5) at eval time, we get free depth.

**A2. Speculative-execution parallelism.** Compute pass k+1 from pass k-1's
state speculatively (in parallel with pass k), accept if pass k's output
matches "expected" stale state. Trade: extra compute, no extra wallclock if
we have headroom (we have 50% VRAM idle).
- **In our setup:** would need to run loop pass k twice in parallel —
  one with the actual incoming residual, one with a speculative version.
  Then merge based on a confidence score.
- **Concrete win:** if speculative pass produces a usable output 50% of
  the time, we get effectively 1.5× recurrence depth at 1.0× wallclock.
- **Risk:** adds compute; if we're roofline-bound, the parallel speculative
  pass blocks the actual pass (since both compete for SMs).

**A3. Wave-front / diagonal parallelism.** Schedule layer-passes on a
diagonal: at time t, token i is at pass (t-i) mod NL. The model still
computes the full loop band per token, but *concurrent* with it, the next
token's earlier pass is also computing. This is pipeline parallelism in
"depth × token" 2D space.
- **Not a quality lever — purely a wallclock lever.** If implemented well,
  effectively halves the per-step wallclock for the loop band.
- **Implementation difficulty: HIGH.** Needs custom kernel or careful
  CUDA-stream choreography.

### Cluster B — Lane / multi-stream parallelism

**B1. Speculative dual-pass with merge.** Run two parallel recurrence chains
on the same input: chain A does NL=2 with the canonical band, chain B does
NL=2 with the 2,3,3,2 spec-073 band (different shape). Merge outputs with
a learned gate.
- **Mechanism:** model gets two attempts at iterative refinement, with
  diverse pass distributions. The gate learns which to trust per-token.
- **Catch:** doubles compute. Useful only if we have spare FLOPs; at
  matmul roofline this slows training 2x.
- **Eval-only version:** at eval time, run both, ensemble outputs.

**B2. Independent multi-rank recurrence.** Each of 8 GPUs runs the same
loop band but on a *different token subset*. Concatenate at the end. Already
how DDP works — but currently DDP processes the *batch* in parallel, all
ranks doing the same recurrence depth on different sequences.
- **What's new:** if we let one rank run NL=3 while others run NL=2, we
  get a "mixed-depth ensemble." During backward, gradient averaging over
  ranks naturally regularizes toward the depth-mixture.
- **No compute cost.** Could be a free quality-bump from inducing depth
  noise.

**B3. Per-channel parallel recurrence.** Split d=512 channels into K groups,
each group does its own recurrence chain. The residual stream becomes
group-parallel during the loop band, recombined after.
- **Like multi-head attention but for loop iteration.** Each group does
  iterative refinement on its slice of features.
- **Not parallel in the wallclock sense** (same compute on the same GPU)
  — but parallel in the sense that gradient flow is independent per
  group, allowing the model to specialize different feature groups for
  different iteration counts.

### Cluster C — Eval-time-only parallelism (the cleanest unspent budget)

**C1. Eval-time NL=4 with cached intermediate.** Train at NL=2 (fits 600s
training cap). At eval, run NL=4 — uses the unused 180s eval headroom to
compute deeper recurrence on each token. The trained model has only seen
NL=2, but recurrence is just "apply the same block more times" — extra
applications should refine, not break.
- **Empirical risk:** memory `project_eval_time_quant_repair_idea` confirms
  the eval headroom; the question is whether out-of-distribution NL at eval
  (NL=4 when trained on NL=2) helps or hurts.
- **Test cost:** zero training-time. Just changes a NUM_LOOPS or eval-time
  override env var. ~$1 to confirm.
- **HIGH PRIORITY** — cheapest, most directly motivated by spare-compute
  finding.

**C2. Eval-time pattern shape sweep.** Train on canonical {3,4,5}NL=2, then
at eval try the 073/074 patterns (already specced for training). If the
*trained representation* benefits from a different recurrence pattern at
inference, we get a free quality bump.
- **Mechanism:** the layer weights are fixed, only the iteration order
  changes. Different patterns may extract different aspects of the
  representation.
- **Negligible cost.**

**C3. Eval-time multi-pattern ensemble.** Run k different LOOP_PATTERN
variants at eval, average logits. Costs k× the eval compute but uses the
180s headroom.
- For k=2: 2× eval cost = 1000s, exceeds budget.
- Only viable if we drop other eval-side levers (sliding window, multiple
  eval batches).

### Cluster D — Memory-for-compute trades exploiting VRAM headroom

**D1. Cache all loop-pass outputs in VRAM, refeed selectively.** Right now
each pass overwrites the prior pass's hidden state (residual stream).
Instead, store every pass's intermediate state in VRAM (we have 40 GiB
spare per GPU). At pass N, the model can access *all* prior pass outputs
via a learned gate or attention.
- **Mechanism:** gives later passes structured access to earlier-pass
  representations. Like K-V cache but for recurrence depth.
- **VRAM cost:** for 17 layer-passes × seq_len 2048 × d 512 × bf16 ×
  per-GPU microbatch 48 = ~3 GiB per pass × 17 passes = ~50 GiB.
  TOO MUCH. Would need to reduce or chunk.
- **Reduced version:** only cache pass outputs for the last 3 passes.
  ~9 GiB. Fits comfortably.
- **Param cost:** small attention or learned-gate mechanism over cached
  states. ~50K params.

**D2. Activation rematerialization-free training.** If we currently use
activation checkpointing (recomputing activations during backward), the
recompute is "wasted" forward compute. With VRAM headroom, we could
disable checkpointing entirely — saves ~30% of backward compute time.
- **Need to verify** what the current setup does. Check the train_gpt.py
  for `torch.utils.checkpoint` calls.

**D3. Gradient accumulation reduction.** Lower GRAD_ACCUM_STEPS from 2
to 1 with doubled microbatch (uses VRAM headroom). Saves ~5% wallclock
on 4H runs.
- Already discussed; bounded gain, but pure win if it fits.

### Cluster E — NCCL bandwidth headroom

**E1. Inter-GPU loop-pass pipelining.** GPU 0 runs loop pass 1, sends
output to GPU 1 which runs loop pass 2 (in parallel with GPU 0 starting
pass 1 of the next batch). Effectively turns the recurrence into a
multi-GPU pipeline.
- **Trade:** synchronous pipeline = wallclock recovery; async = harder.
- **NCCL is at 0.5% — tons of headroom.** Sending hidden states between
  GPUs is cheap relative to FLOPs.
- **Implementation:** custom inter-GPU forward, complex. Not realistic
  for one-day homestretch.

### Cluster F — Algorithm-level (rethink what "iteration" means)

**F1. Newton-style second-order recurrence.** Instead of fixed-point
iteration (which is what NL passes do), run a Newton update at each
pass: f(x) - f'(x)·f(x). Converges quadratically — 3 passes ≈ NL=8 of
fixed-point.
- **Theoretical** — implementation requires Jacobian-vector products
  per pass. Approximately 2× compute per pass; needs to converge ≥ 2×
  faster to win. Possible if the loop function is sufficiently smooth.

**F2. Krylov-subspace recurrence.** Treat the loop as an iterative linear
solver. Build a Krylov subspace from successive powers of the loop
operator, project the solution into it, solve. Converges in O(√k) where
k is condition number — could mean 4 passes of Krylov ≈ NL=16 of vanilla.
- Speculative; recurrence is nonlinear so the Krylov framing is loose.

**F3. Monte-Carlo dropout pass averaging.** Run NL=2 with dropout on the
recurrent layers. At inference, run NL passes with K different dropout
masks, average. K=4 → effectively 4× recurrence diversity at NL=2 cost
per inference pass. Eval-only.

### Cluster G — Compile-time / kernel-level

**G1. CUDA Graphs for the loop body.** Capture the entire loop band as
one CUDA graph; replay K times. One launch per loop activation instead
of per-block. With ~6 blocks × NL=3 visits = ~18 launches per loop region
currently. Combine into 1.
- **Wallclock recovery:** depends on how much launch overhead dominates;
  per spec 064b, at matmul roofline launch overhead is small (~1-2%).
- **Risk:** torch.compile already does some of this. Verify what's left
  to capture.

**G2. Triton fused loop block.** Write a custom Triton kernel that does
RMSNorm + attention + MLP + residual add for the loop block, all fused.
Cuts memory bandwidth by avoiding HBM round-trips between sub-ops.
- **Engineering cost: HIGH.** Probably 2-3 days of work.
- **Recovery:** ~5-10% wallclock if bandwidth-limited, less if FLOPs-limited.

---

## Top-3 to develop further (initial pick)

1. **C1 — Eval-time NL=4 with cached intermediate.** Cheapest, motivated by
   spare-compute finding, no training risk.
2. **D1 (reduced) — Cache last 3 pass outputs, learned gate.** Uses VRAM
   headroom, adds small param budget, structurally novel.
3. **A1 (eval-only) — Parallel loop transformer at eval.** Adapts a 2025
   paper's mechanism to our spare eval budget.

Spec drafts will land at `research/specs/080-*.md` and onward when any of
these matures past the brainstorm threshold.
