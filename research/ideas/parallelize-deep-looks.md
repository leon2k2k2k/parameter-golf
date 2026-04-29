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
- **W1 (2026-04-29 +10min):** fresh ideas in clusters H/I/J/K; spec 080 frozen
- **W2 (2026-04-29 +20min):** clusters L/M/N/O + opposite-direction tests; spec 081 frozen
- **W3 (2026-04-29 +30min):** clusters P/Q/R/S/T (TTT×recurrence, position-dependent depth, cross-rank pattern divergence, fixed-point cache); spec 082 frozen
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

---

## W1 — fresh angles + spec 080 frozen

### Cluster H — Eval-time-only "more passes" leveraging trained checkpoint

**H1. Eval at NL=3 on a model trained at NL=2.** Extends C1. Implementation
is *config-only* on top of `exp/071-loop-pattern @ e7ccda2` (no new code):
set `LOOP_PATTERN` to encode 4 passes through {3,4,5} when resuming the
060A checkpoint via `launch_060_eval.sh`. The trained model has bank-
indexed weights — extra visits to layer 3,4,5 just reuse those banks.
- **Why "parallelism":** no actual parallelism in the wallclock sense, but
  conceptually parallel to the C1 idea: it tests whether the trained
  iterative-refinement function generalizes to deeper iteration.
- **Compile audit:** eval-only run. Model construction reads LOOP_PATTERN
  at __init__, builds longer encoder/decoder index lists once. Compiled
  forward sees one new graph variant on first eval call (different
  iteration count from training-time graph), no mid-run recompile.
- **Frozen as spec 080.**

**H2. Eval at NL=3 with progressive "warmup."** Variant of H1: at eval,
run forward with NL=2 first (matches training), then run a *second* forward
with NL=3, blend logits (0.7 × NL=2 logits + 0.3 × NL=3 logits). Tests
whether trained-distribution and OOD-deeper-distribution outputs combine
better than either alone. ~2× eval compute — only viable if eval headroom
allows.

**H3. Per-document NL adaptation.** Long documents may benefit from extra
passes more than short ones (more context to iterate over). At eval, set
NL=2 for short docs, NL=3 for long docs (length threshold). Free at eval.
Risk: doc-length distribution may make the gain negligible if eval is
mostly short docs.

### Cluster I — Pass-axis weight-sharing variants

**I1. Pass-shared LoRA injection.** Instead of per-pass LoRA (047C, 050B,
050D — all null), share a single LoRA across all passes but apply it
*only on the recurrent passes* (not pass 0). Tests whether it's the
"per-pass differentiation" that fails or the "LoRA-on-loop" that fails.
Single LoRA, low param cost.

**I2. Asymmetric pass-1 vs pass-N+ weights.** Add a single learnable
ΔW on the loop layers that's applied ONLY for the *first* pass (pass 0)
and ZEROED for later passes. Tests whether "the first pass is doing
encoding, later passes are doing refinement" hypothesis. If yes, giving
pass 0 dedicated capacity helps; if no, this is null. Costs the same as
047C but inverts the asymmetry.

### Cluster J — Speculative-input parallelism (cheap, no quality risk)

**J1. Bigram-prefix speculative parallel pass.** Compute pass k+1 *in
parallel* with pass k by feeding pass k+1 a *bigram-predicted* input
(estimated from the prior token's hidden state). When pass k completes,
rectify pass k+1's output with a delta correction. Speculatively-
parallel; cost is one extra pass per step but it runs concurrent
with pass k.
- VRAM headroom can absorb the extra activations.
- The bigram prediction is cheap: a 2-gram table lookup (or fast linear
  layer) gives a near-deterministic next-residual prediction for common
  bigrams.
- Implementation is real engineering work — not for tonight.

**J2. Pass-k-output reuse from prior step.** At training time, the same
input batch goes through the model. The *same residual at layer 3* on
*sequential training steps* should be similar (slowly-changing weights,
similar inputs). At step t, predict pass-3-of-layer-5 using pass-3-of-
layer-5 from step t-1 (cached). Use cached value as a *speculative input*
to pass-4-of-layer-5 in parallel with computing pass-3 freshly. Merge.
- Speculative; if the prediction is close, we save one sequential pass.
- Hard to verify without testing — but cheap to add as a flag if we
  build the infrastructure.

### Cluster K — Mathematical reformulations of the recurrence

**K1. Anderson acceleration on the loop.** Anderson mixing accelerates
fixed-point iteration: instead of x_{k+1} = f(x_k), use
x_{k+1} = β·f(x_k) + (1-β)·linear-combination-of-prior-iterates.
Converges 2-5× faster on smooth fixed points. Tested for years on
DeepEqQ/DEQs (Bai et al. 2019) — known to work.
- Implementation: store m=2-3 prior pass outputs, compute coefficients
  via least-squares, mix. Adds ~3 small matmuls per pass.
- Compile audit: stateful cache across passes inside compile region —
  compiles to one graph variant if cache size is fixed and tensors are
  always-tensor. Need careful design.

**K2. Implicit fixed-point via Newton iteration.** As K1 but with
explicit Jacobian-vector product. Converges quadratically. ~2× compute
per pass. Theoretical home run if it works (3 passes ≈ NL=8).
- Implementation: requires backward pass during forward (`torch.autograd.grad`
  inside compile) — almost certainly forces graph break. Skip for the
  homestretch.

### Decisions for tonight

- **H1 → frozen as spec 080.** Cheapest, safest, eval-only, config-only,
  uses LOOP_PATTERN already implemented at e7ccda2. Tests "deeper is free"
  hypothesis directly.
- **K1 (Anderson)** is the most exciting algorithmic angle but needs
  careful engineering. Defer to W2 or W3 wakes if time permits.
- **I2 (asymmetric pass-0 weights)** is interesting but contradicts
  prior 047 nulls. Lower priority for the homestretch.
- **J1/J2** require infrastructure that doesn't exist yet. W3+.

---

## W2 — opposite-direction tests + sensitivity sweeps + premature exits

### Cluster L — "Premature head" intermediate prediction

**L1. Auxiliary intermediate-pass logit head.** During training, project
the hidden state at the *end of pass 1* (instead of the final pass) to
vocab via the tied embedding head. Compute a *small* auxiliary CE loss
on it. The model is then trained to make pass 1's output already
predictive — which means at eval, you could potentially skip later
passes for tokens whose pass-1 prediction is high-confidence.
- **Param cost:** zero new params (uses tied head).
- **Quality risk:** the auxiliary loss may compete with the main loss;
  weight carefully (probably 0.1× main loss).
- **Eval flexibility:** tokens with H(p_pass1) < threshold can skip
  passes 2/3 → free wallclock. Tokens with high entropy continue.
- **Compile audit complication:** the per-token early-exit gate is
  data-dependent control flow inside compile — would force graph
  break. So at training time we'd just compute both losses every step
  (no early exit). The early-exit win is eval-only.

**L2. Simpler variant: pass-1 logit ensemble at eval.** Train normally;
at eval, compute logits at end of pass 1 AND end of final pass, average
them. Doesn't require any training-side change — uses spare eval compute
to ensemble two depths. Bypasses the data-dependent control flow problem.

### Cluster M — Opposite-direction sensitivity tests

**M1. Eval at NL=1 (FEWER passes) on 060A checkpoint.** The dual to
spec 080. If the model already reaches good output at NL=1 (one pass
through {3,4,5}), we know the recurrence isn't load-bearing for *most*
tokens. If it crashes, recurrence is essential.
- LOOP_PATTERN body for NL=1 equivalent: the trained pattern minus
  two passes. Body = "1,2,3,4,5,6,7" (7 visits). Pre [0] + post
  [8,9,10] = 11 total visits = no-loop equivalent on layer band.
  Layers 3,4,5 visited only ONCE each. Drastic reduction.
- Spec 082 candidate: this is the cleanest companion to 080/081.

**M2. Eval at NL=2 with {3,5} band only.** Skip layer 4 in the loop
band. body = "1,2,3,5,3,5,3,5,6,7" — visits: 3:3, 5:3, 4 in non-loop
visits only. Tests whether layer 4's loop-pass contribution is
load-bearing or just filler.
- Empirical hole; never tested.

**M3. Eval-pattern sensitivity sweep.** On 060A checkpoint, eval with
all four already-specced patterns (071/072/073/074) plus 080/081's
patterns. Build a "pattern × bpb" table to see how robust the trained
representation is to pattern shape.
- Cost: ~4 × $1 = $4 for the full sweep.

### Cluster N — Loop-pass dropout regularization

**N1. Random pass-skip at training time.** Per training step, with
probability p (e.g., 0.1), randomly skip ONE of the three loop passes.
Forces the model to be functional with NL ∈ {2, 3}. At eval, always
use NL=3 (full power). Free at inference; cheaper at training (10%
faster on average); regularization upside.
- Compile-graph hazard: per-step random pass count → variable
  iteration depth → graph variant explosion. Could be fixed with
  always-tensor pattern: instead of skipping, multiply the skipped
  pass's output by 0 (a learned-dropout mask tensor). Then iteration
  count is fixed, graph is one variant, but the skipped pass has
  identity behavior.
- This makes it not a throughput lever (still pays for the pass), only
  a regularization lever. Drop the throughput angle; keep the regu-
  larization framing.

**N2. Pass-index conditional modulation (= 047D AdaLN done right).**
047D failed; the failure mode was likely that per-pass γ/β has too
many degrees of freedom for the optimizer. Cleaner: a *single* learnable
scalar per pass index, applied uniformly across the entire residual
(not per-channel). 5 scalars × 3 loop layers = 15 params total. Tested
indirectly by 053 ("per-pass FFN scale") — null. Confirmed dead.

### Cluster O — Eval-time auxiliary refinement

**O1. Final-pass MLP-only refinement.** At eval, after the trained NL=2
loop completes, run ONE additional pass through just the MLP of the loop
band (skip attention). Cheaper than a full extra pass (~50% of pass cost),
since MLP is half the loop-block compute. Tests whether the residual
benefits from one more "feature update" without re-attending.
- Compile audit: requires a code change to add an "MLP-only refinement"
  flag that conditionally runs `mlp(x)` after the main loop. Must use
  always-tensor pattern (zero-buffer for inactive case).

**O2. Final-pass attention-only refinement.** Inverse of O1: extra
attention pass without MLP. Tests the orthogonal hypothesis. Same
compile-audit complexity.

### Decisions for W2

- **Spec 081 = NL=4 equivalent eval on 060A** (5 passes through {3,4,5}).
  Direct upward extension of 080. Pattern-only, config-only, same code
  as 080. **FREEZE THIS WAKE.**
- **Spec 082 candidate (W3): NL=1 equivalent eval (M1).** Downward dual.
- **L2 (pass-1 logit ensemble at eval)** is interesting but requires a
  small code change (forward returns intermediate logits). Defer.
- **N1 with always-tensor masking** is novel but the regularization
  angle is the only justification; throughput-side framing was wrong.
  Lower priority for homestretch.
- **O1/O2** require code changes for the MLP-only / attn-only
  refinement modes. Defer to W3 or W4.

---

## W3 — TTT × recurrence interactions, position-dependent depth, cross-rank divergence

### Cluster P — TTT × deeper-recurrence composition

The model has Phased TTT enabled (`PHASED_TTT_ENABLED=3`,
`PHASED_TTT_NUM_PHASES=3`, `PHASED_TTT_PREFIX_DOCS=2500`). TTT adapts
LoRA weights to the eval distribution at inference time. The 060A
checkpoint already pays ~423-495s of the 600s eval budget on TTT +
quant + scoring.

**P1. TTT-on with NL=3 equivalent (compose 080 with TTT).** Spec 080
disabled TTT to isolate the deeper-recurrence effect on pre-quant bpb.
But the *real* leaderboard number includes TTT. So we need to know
whether the deeper-recurrence gain (if any) survives TTT. Implement-
ation: same as 080 but `TTT_ENABLED=1 PHASED_TTT_ENABLED=3`. Tests
whether the two eval-time-compute levers compose.
- **Compile-graph audit:** TTT path (`forward_ttt`) is a separate
  compiled function from `forward_logits`. Both share the same
  encoder/decoder index lists at module level. Setting LOOP_PATTERN
  means both compile once for the longer iteration count, on first
  call each. No mid-run recompile.
- **Risk:** TTT was tuned for NL=2 trained model. Deeper recurrence
  may distort what TTT sees — could compose well, could destructively
  interfere.

**P2. TTT phases at different recurrence depths.** Run phase 1 of TTT
at NL=2 (matches training distribution; warms LoRA on the canonical
model), then phases 2-3 at NL=3 (uses the LoRA-adapted weights with
deeper recurrence to refine final logits). Stages compute across
phases like a curriculum.
- **Code change:** requires per-phase LOOP_PATTERN switching inside
  the TTT phase loop. Modest but real engineering. Defer.

### Cluster Q — Position-dependent recurrence depth

**Q1. Depth-tail-only.** For autoregressive bpb, the loss-dominant
positions are typically the last k tokens of each window (or the
"hard" positions per-document). If we only apply NL=3 to the LAST 25%
of the sequence and NL=2 to the rest, we recover roughly 75% of the
extra cost while still helping where it matters most.
- **Implementation:** requires a position-mask in the forward; not a
  pure config change. Compile-graph hazard: position-conditional
  iteration depth = data-dependent control flow inside compile = graph
  break or recompile.
- **Defer until we have a clean always-tensor design.**

**Q2. Depth-by-doc-length at eval.** Choose NL based on document length
once at the start of each eval window (deterministic, not data-
dependent). Short docs use NL=2; long docs use NL=3. This is per-doc,
not per-token, so the iteration count is fixed within a window.
- **Compile audit:** still involves variable-iteration-count, would
  produce two distinct graph variants (one per NL). Both compile once,
  cache, no mid-run recompile if the *number* of distinct variants is
  bounded.
- **Marginally implementable** but engineering-heavy. Defer.

### Cluster R — Cross-rank pattern divergence (multi-pattern ensemble for free)

**R1. Per-rank LOOP_PATTERN at eval.** The 4-GPU eval setup distributes
sequences across ranks via DDP. Each rank processes its own subset of
the val set. If we set a *different* LOOP_PATTERN on each rank
(rank-conditional via LOCAL_RANK env var), each subset of sequences
is evaluated under a different recurrence pattern. The aggregate
val_bpb averages across patterns naturally — implicit ensemble at
zero extra wallclock.
- **Cost:** zero — uses existing DDP infrastructure.
- **Implementation:** read LOCAL_RANK in launch script, choose
  pattern. Or per-rank LOOP_PATTERN env via systemd-style launch.
- **Catch:** per-rank pattern means per-rank graph variant. With 4
  ranks running 4 patterns, each rank compiles its own pattern variant
  once, no recompile after that. **Compile-safe.**
- **Hypothesis:** if patterns 080/081/canonical/no-loop produce
  different bpbs that average lower than canonical-everywhere, we get
  a free ensemble effect. If they degrade, we lose.
- **Spec candidate:** could freeze as 084 in W4.

**R2. Sequence-level sharding to specific patterns.** Within a single
rank, group sequences by length and route long-docs to NL=3 LOOP_PATTERN
batches, short-docs to NL=2 batches. Maintains throughput while
specializing pattern per-doc.
- More complex than R1; defer.

### Cluster S — Speculative fixed-point caching (refines D1)

**S1. Cache-and-extend.** Train at NL=2. Eval forward = run NL=2 normally;
cache the residual-stream output of the loop band; THEN run k more
passes from the cached state through ONLY the loop block, using the
cached state as input. Append the k-extended output to the rest of the
forward (decoder).
- **VRAM cost:** one residual cache per loop layer, ~3 GiB total. Fits.
- **Code change:** a single hook that grabs the residual at end of
  loop band, then re-feeds it through the loop block k times before
  passing to layers 6+. Modest implementation.
- **Quality story:** lets us adjust the *number* of additional passes
  at eval-time without rerunning the whole forward — if we're already
  computing NL=2, adding 2 more passes through the loop block costs
  roughly 2 × loop-band-time, not 2 × full-forward-time.
- **Compile-graph audit:** the extension passes are inside compile
  region; but iteration count is fixed (k is a constant env var, not
  data-dependent). One graph variant per (k, base NL) combination,
  compiles once, cached.
- **Spec candidate for W4 with code change**

### Cluster T — TTT-aware deeper recurrence

**T1. TTT learns the extended recurrence.** TTT phases adapt LoRA. If we
*also* extend the recurrence during TTT (not just at scoring), the LoRA
adapts the model to deeper iteration. In effect, we "fine-tune at
inference" for NL=3 even though training was NL=2.
- **Hypothesis:** P1 (deeper recurrence at scoring only, TTT at canonical
  NL) may show neutral effect because TTT doesn't see the deeper
  recurrence. T1 has TTT learn the deeper recurrence pattern, possibly
  unlocking the gain.
- **Implementation:** identical to P1 (both TTT and scoring use
  LOOP_PATTERN with extra passes). Same compile-audit story.
- **Effectively a renaming of P1 — they're the same spec.**

### Decisions for W3

- **Spec 082 = NL=1 equivalent eval (M1).** Bookends the 080/081
  scaling curve from below. Pattern body = "1,2,3,4,5,6,7" giving
  layers 3,4,5 each visited ONCE. Equivalent to non-looping forward,
  but using LOOP_PATTERN explicitly so the comparison is matched
  via the same code path. Config-only on `e7ccda2`. **FREEZE THIS WAKE.**
- **P1 / T1 (TTT-on with NL=3 eq)** is the next obvious freeze for W4
  — composes 080 with TTT, real leaderboard-relevant.
- **R1 (per-rank pattern divergence)** is the most novel idea this wake
  — free ensemble effect at no extra cost. Worth specifying for W5.

