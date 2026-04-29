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
- **W4 (2026-04-29 +40min):** TTT compositional risks; clusters U/V/W (eval-time logit blends, encoder-only/decoder-only NL skew, post-quant deeper recurrence); spec 083 frozen (TTT-on × NL=3 eq)
- **W5 (2026-04-29 +50min):** R1 framing correction; clusters X/Y/Z (warmup-skip eval, prefix-only deeper recurrence, mixed-precision recurrence); spec 084 frozen (TTT-on × NL=4 eq)
- **W6 (2026-04-29 +60min):** V1/V2 design challenge (U-Net midpoint constraint); clusters AA/BB/CC (matched-compute pattern shape, TTT-prefix length sweep, hot-pass conditioning); spec 085 frozen (matched-compute broad pattern eval on 060A)
- **W7 (2026-04-29 +70min):** clusters DD/EE/FF (LR schedule × deeper-NL interactions, rolling-window eval, low-rank residual stream); spec 086 frozen (TTT-on × broad-pattern, leaderboard-relevant)
- **W8 (2026-04-29 +80min):** clusters GG/HH/II (band-position shifts, TTT batch size sensitivity, gradient-aligned recurrence); spec 087 frozen (matched-compute stepped pattern eval, 073-shape sibling)
- **W9 (2026-04-29 +90min):** clusters JJ/KK/LL (post-quant only TTT-extension, KV-projection truncation, expanded smear-gate window); spec 088 frozen (TTT-on stepped-pattern, leaderboard cell)
- **W10 (2026-04-29 +100min):** clusters MM/NN/OO (eval-only embedding-LR rescale, recurrence + sliding window interaction, multiple residual streams during loop band); spec 089 frozen (band-position shift {2,3,4} eval — novel positional axis)
- **W11 (2026-04-29 +110min):** clusters PP/QQ/RR (band {4,5,6} symmetry test, training-data-free hotstart from canonical, partial-rank weight slicing); spec 090 frozen (band-position shift {4,5,6} — completes position pair with 089)
- **W12 (2026-04-29 +120min):** clusters SS/TT/UU (TTT-compute axis isolation, eval-time-deterministic seeding, multi-checkpoint averaging); spec 091 frozen (more TTT phases at canonical — TTT compute axis test)
- **W13 (2026-04-29 +130min):** clusters VV/WW/XX (position × TTT cell completion, GPTQ calibration sweep, dropout-style residual perturbation); spec 092 frozen (TTT-on band {2,3,4}, leaderboard cell for 089)
- **W14 (2026-04-29 +140min):** clusters YY/ZZ/AAA (TTT-LR sweep, eval-time SmearGate alteration, repeated-context test); spec 093 frozen (TTT-on band {4,5,6}, completes position × TTT grid)
- **W15 (2026-04-29 +150min):** clusters BBB/CCC/DDD (TTT-momentum sensitivity, eval-time logit softcap variant, full-attention vs XSA on loop layers); spec 094 frozen (TTT_LORA_LR=5e-5, smaller-LR sweep)
- **W16 (2026-04-29 +160min):** clusters EEE/FFF/GGG (XSA on/off subset of layers, validation-set sub-sampling for fast iteration, eval-time mixed-precision); spec 095 frozen (TTT_LORA_LR=2e-4, larger-LR sweep, completes LR direction)
- **W17 (2026-04-29 +170min):** clusters HHH/III/JJJ (TTT-momentum activation, prefix-doc count sweep, 1-shot TTT post-hoc); spec 096 frozen (TTT_BETA1=0.9, novel TTT-momentum axis)
- **W18 (2026-04-29 +180min):** clusters KKK/LLL/MMM (TTT batch granularity, TTT chunk size effect, TTT_BETA2 variant); spec 097 frozen (TTT_BATCH_SIZE=32, TTT-batch-granularity axis)
- **W19 (2026-04-29 +190min):** clusters NNN/OOO/PPP (TTT-WD effect, eval-time GPTQ-only test, TTT-LR-warmup ramp); spec 098 frozen (TTT_BETA2=0.999, completes Adam β-mapping)
- **W20 (2026-04-29 +200min):** clusters QQQ/RRR/SSS (TTT vs no-TTT calibration, GPTQ-without-TTT pre-quant comparison, single-phase TTT diagnostic); spec 099 frozen (TTT-disabled, calibrates TTT lever weight)
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

---

## W4 — TTT compositional risks + post-quant variants

### Cluster U — Eval-time logit blends (cross-NL ensembling without retraining)

**U1. Logit average across NL=2 + NL=3 forwards on the same checkpoint.**
060A's `final_model.pt` can be loaded twice in the same eval (or
in two passes, output cached and combined). Run forward at canonical
NL=2 → save logits L₂; run forward at NL=3 (LOOP_PATTERN per 080) →
save logits L₃. Final logits = α·L₂ + (1-α)·L₃ with α tuned via a
short grid {0.3, 0.5, 0.7}.
- **Mechanism:** if 080 lands close-to-baseline-noise, neither NL=2
  nor NL=3 dominates uniformly — but they may have *complementary*
  errors. Logit averaging exploits that.
- **Cost:** 2× eval forwards = ~2× wallclock. Risk: blows the eval-
  time budget if both forwards take ~500s. Need to verify total
  wallclock fits.
- **Compile audit:** two distinct eval forwards = two cached graph
  variants, both compiled at first invocation, no mid-run recompile.
- **Spec candidate for W5/W6**.

**U2. Token-position-conditional NL via post-hoc selection.** Run BOTH
forwards (NL=2 and NL=3), but per-token select whichever has lower
cross-entropy on the validation byte stream. Optimistic upper bound:
"oracle NL selector" gives the best-of-both at every position. Real
implementation: select based on prediction entropy of NL=2 (proxy for
"am I sure?" — if sure, use NL=2; if uncertain, use NL=3).
- **Practical version:** run both, average; ignore selection logic.

### Cluster V — Encoder-only vs decoder-only NL skew

The U-Net structure splits encoder_indices and decoder_indices at the
midpoint of the expanded loop list. Currently NL=2 means the loop
band {3,4,5} is repeated 3 times across the *combined* list. The
encoder visits some of those repetitions, the decoder visits the
rest. We could break the symmetry deliberately.

**V1. Encoder-deeper, decoder-canonical.** Build LOOP_PATTERN such
that the encoder side has 4 visits to {3,4,5} but the decoder side has
only 2-3. Tests whether the iterative refinement happens during the
encoder-side or decoder-side processing.
- For example: pre [0,1,2] + body [3,4,5,3,4,5,3,4,5,3,4,5,5,6] + post [7..10]
  → encoder visits {3,4,5} four times, decoder visits twice.
- Cleanest specification: tweak the body string so the midpoint split
  produces asymmetric counts.
- **Hypothesis:** if encoder-side recurrence dominates, V1 wins. If
  decoder-side, V1 loses. Either way, informative.

**V2. Decoder-deeper, encoder-canonical.** Inverse of V1. Asymmetric
in the other direction.

**V3. Joint sweep V1+V2.** Run both as the natural pair-test.

### Cluster W — Post-quantization recurrence variants

Spec 080/081/082 measure pre-quant val_bpb. The leaderboard cares
about post-quant val_bpb (pre-quant is just a proxy). Quantization
error is an additional source that may interact with deeper
recurrence non-linearly:

- Each loop pass amplifies whatever quant error is present in the
  weight banks. Trained NL=2 model's quant calibration was tuned for
  3 visits per loop layer; running NL=3 means each weight is used 4
  times → quant error compounds 4/3× more.
- Or: extra passes "average out" quant noise (de-noising effect).
  Unclear which dominates.

**W1. Spec 080 with GPTQ ON.** Same as 080 but enable
`GPTQ_CALIBRATION_BATCHES=16` and run the full quant pipeline. Tests
whether the pre-quant gain (if any) survives quantization.
- **Subsumes part of P1** (TTT × NL=3) which is being frozen this wake.
- W1 is *quant-only* without TTT; P1 is *TTT-only* without recompute.
  The full leaderboard test (TTT + GPTQ) is W2-style.

**W2. Spec 080 with full submission pipeline (TTT + GPTQ).** This is
the actual leaderboard-relevant measurement. After running, you have
a real submittable artifact at NL=3 inference.
- **Cost:** ~$3-5 (full eval pipeline including TTT phases).
- **Hypothesis:** if pre-quant gain survives both TTT and GPTQ, this
  is a leaderboard-promotable result.
- **Subsumes both P1 and W1.**

### TTT compositional risks (the reason 083 needs careful framing)

When freezing P1 (083 = TTT-on with NL=3 eq), there are two distinct
sub-questions:

- **Q1: Does TTT operate over the deeper recurrence?** TTT updates LoRA
  weights based on prefix tokens. If the recurrence depth changes
  during TTT (eval at NL=3 instead of NL=2), the LoRA gradients are
  computed against the deeper-recurrence forward — meaning TTT learns
  to adapt the deeper-recurrence model. Should be fine.

- **Q2: Are TTT phases compatible with longer iteration counts?**
  PHASED_TTT runs 3 phases. Each phase: TTT update, then score next
  chunk. If LOOP_PATTERN extends the recurrence, all phases see it.
  Each phase scores against the deeper graph — same compile burst at
  first phase, then cached.

Both should be safe under the always-tensor / no-mid-run-recompile
constraint, IF the LOOP_PATTERN is set BEFORE the model is constructed
(env var read at __init__). Setting it via env var in the launch
script does exactly this.

### Decisions for W4

- **Spec 083 = P1 = TTT-on × NL=3 eq.** Real leaderboard-relevant
  test. Eval-only on 060A checkpoint, config-only. **FREEZE THIS WAKE.**
- **W2 (full pipeline, 080+TTT+GPTQ) candidate for W5.**
- **V1/V2 (encoder/decoder NL skew)** is interesting and config-only —
  good W5 candidate too.
- **U1 (logit-blend ensemble)** needs careful eval-time wallclock
  accounting — defer until we know if there's headroom for 2× eval
  cost.

---

## W5 — correction to R1 framing + new clusters X/Y/Z

### Correction: R1 (per-rank pattern divergence) is NOT a free ensemble

W3 proposed R1 as "free implicit ensemble at zero extra wallclock" via
per-rank LOOP_PATTERN. **Re-examined this wake — the framing was wrong.**

In our DDP setup, each rank processes a *different subset* of validation
sequences. So per-rank pattern means rank 0 scores subset A under
pattern P_A, rank 1 scores subset B under pattern P_B, etc. Aggregating
their bpb gives **mixture-of-patterns aggregate bpb across the dataset**,
not a per-token ensemble of patterns on the same sequences.

That's a different (and less useful) measurement. The aggregate is
unbiased only if all patterns produce identical per-sequence bpb
expectations — exactly what we're trying to test.

For a *real* per-token ensemble we'd need each sequence scored under
multiple patterns and logits averaged — that's U1, which costs ~2× eval
wallclock. There's no zero-cost free-lunch on this axis.

R1 demoted: drop from "novel zero-cost test" to "interesting comparison
data point at no extra cost, but not an ensemble."

### Cluster X — Warmup-skip eval (a one-line win possibility)

**X1. Skip the loop-warmup at eval-only runs.** During training, the
`loop_warmup` phase runs ~20 forward+backward passes through the looped
graph to populate the inductor cache before training begins. For
**eval-only** runs (RESUME_FROM_CKPT), the looping is set to active
immediately on first forward — but the loop_warmup may still run if not
explicitly skipped. That's wasted compute (~20 × loop forward pass times).
- **Verify:** check 080's `train.log` to see if loop_warmup_step runs
  appear during the eval-only path.
- **Fix:** if yes, add a `SKIP_LOOP_WARMUP_FOR_EVAL=1` env var that
  short-circuits the warmup when there's no training. ~5 min savings on
  every eval-only spec (082-085 all benefit).
- **Compile audit:** the warmup itself is what creates the compiled
  graph cache. Skipping warmup means first eval batch triggers the
  compile burst. Same total work, just shifted from warmup phase to
  first batch. No mid-run recompile.
- **Savings real?** Probably not net — warmup compile is the same
  compile that would happen on first eval batch anyway. False savings.
  **Drop this idea.**

**X2. Pre-cache the inductor cache on the pod.** Memory has
`feedback_prewarm_before_new_commit` — for new code commits, do a full
autotune prewarm. For eval-only runs on a *different* SHA than 060A's
trained run (we're on `e7ccda2` vs 060A's `a0a48b7`), the cached graphs
may not match. Stash + restore the inductor cache from the original 060A
training run before launching the eval. Memory has utilities for this.
- **Concrete:** `tmp_exec/cache_restore.sh a0a48b7` would restore the
  cache from the training run. The launch script already does this for
  matching SHA; need to verify behavior for different SHA.
- **If it works:** saves ~5 min on every eval spec (no first-batch
  compile burst). Real savings.
- **Implementation:** verify cache_restore behavior, possibly modify
  launch_060_eval.sh to use the training SHA's cache.
- **Tractable for an execution-time tweak, not a research-spec.**

### Cluster Y — Prefix-only deeper recurrence

**Y1. Run NL=3 only on the *prefix* tokens of each eval sequence.**
Autoregressive bpb is computed by predicting each token from its
preceding context. For long contexts, the prefix length grows. Apply
extra loop passes ONLY to prefix processing (longer chain of context
integration), then revert to NL=2 for the prediction-tail.
- **Catch:** in our forward, the model processes the *whole* sequence
  in one shot, then computes per-token loss via shifted logits. There
  is no separate "prefix-encoding" phase. The model is parallel over
  position. So this idea doesn't directly apply.
- **Variant Y1':** in the TTT pipeline, the prefix docs ARE processed
  separately (used to update LoRA). If TTT prefix processing uses
  NL=3 but scoring uses NL=2, that's a clean test of "TTT learns
  better with deeper recurrence even if scoring is shallow."
- **Compile audit:** TTT and scoring already use separate compiled
  functions. Different NL for each = two distinct graph variants,
  both compile once, no recompile. Implementable as two LOOP_PATTERN
  env vars: `TTT_LOOP_PATTERN` for `forward_ttt` and `LOOP_PATTERN`
  for `forward_logits`. Requires small code change to read both.

### Cluster Z — Mixed-precision recurrence (compile-safe)

**Z1. fp32 residual stream during loop band only.** Currently the
residual stream is bf16 throughout. The recurrent passes accumulate
small per-pass updates; bf16's ~7-bit mantissa may quantize away
useful signal across many passes.
- **Hypothesis:** fp32 residual during the loop band, cast back to
  bf16 after. Tests whether mantissa precision limits recurrence
  refinement.
- **VRAM cost:** doubles activation memory for loop-band layers.
  We have headroom (50% spare).
- **Compile audit:** dtype change at the loop entry/exit is two
  static cast ops baked into the compile graph. One graph variant
  (with-fp32-loop), no mid-run recompile.
- **Code change:** small, targeted. ~30 LOC in train_gpt.py to wrap
  the loop band in `with torch.autocast(dtype=torch.float32)`.
- **Spec candidate for W6+ if a code-change spec is desired.**

**Z2. Per-pass dtype hierarchy.** First pass in fp32 (precise
integration), later passes in bf16 (cheaper refinement). Asymmetric
precision per pass.
- **Compile audit:** would need per-pass dtype routing inside compile.
  Risky — likely produces multiple graph variants. Skip unless we
  carefully always-tensor the dtype selection.

### Decisions for W5

- **Spec 084 = TTT-on × NL=4 eq.** Direct extension of 083. Composes
  081 with TTT. Tests whether the deeper-recurrence gain (if any)
  scales with depth under TTT. Config-only on `e7ccda2`. **FREEZE
  THIS WAKE.**
- **Y1' (separate TTT vs scoring LOOP_PATTERN)** is novel and worth
  doing if 083 shows ambiguous TTT-NL coupling. Defer for code-change.
- **Z1 (fp32 loop residual)** is a code-change candidate for W6+.
  Higher engineering cost than the eval-only specs we've been freezing.
- **R1 demotion finalized** — not a free ensemble; lower priority.

---

## W6 — U-Net midpoint constraint + matched-compute pattern shape + TTT-prefix sweep

### V1/V2 design challenge — U-Net midpoint forces near-symmetry

W4 proposed V1 (encoder-deeper) and V2 (decoder-deeper) NL skews. This
wake's investigation: **the existing GPT.__init__ splits all_indices at
the midpoint** (`num_enc = len(all_indices) // 2`), so attempting to
skew {3,4,5} visits per side is constrained.

**Concrete attempts (matched 17-pass compute):**

| Pattern body | Pre+post | enc {3,4,5} visits | dec {3,4,5} visits |
|---|---|---|---|
| `3,4,5,3,4,5,3,4,5,6,7` | [0,1,2]+[8,9,10] | 5 | 4 |
| `3,4,5,3,4,5,3,4,5,4,5` | [0,1,2]+[6..10] | 5 | 4 |
| canonical `3,4,5,3,4,5,3,4,5` | (auto) | 5 | 4 |

In all matched-17-pass body designs, the midpoint split lands at the
same encoder/decoder distribution. **Truly skewed V1/V2 require a
code change** to override the midpoint split (e.g., `ENC_DEC_SPLIT`
env var that forces split at a different index).

- **Implication:** V1/V2 demoted to "code-change required, lower
  priority." If a future wake produces this, the change is small
  (~5 LOC: read ENC_DEC_SPLIT, override `num_enc` if set).
- **Compile audit for the code change:** trivial — `num_enc` is a
  Python int set at __init__. Different value → different
  encoder/decoder split lists. Same compile-time fixed iteration
  count, no mid-run recompile.

### Cluster AA — Matched-compute pattern shape at eval

Sibling to the (NL eq) scaling curve specs 080/081/082. Those vary
*compute* (number of layer-passes). This cluster varies *shape* at
**fixed compute** (17 passes = canonical).

We already specced 071/072/073/074 as TRAINING-time pattern variants.
Reusing those patterns at *eval* (no retraining) on 060A's checkpoint
tests a different question: **how robust is the trained representation
to runtime pattern shape changes?**

**AA1. Eval at 074's broad pattern.** Pattern body
`1,2,3,3,4,5,4,5,6,5,6,7,7` — 17 total layer-passes. Layers 3,4,5,6,7
all visited 2-3 times each (broader band).

The trained model saw layers 3,4,5 visited 3× each and layers 6,7
visited only 1× each. AA1 visits layers 6 and 7 *twice*, demanding
they iterate in the recurrent role they weren't trained for.

- **Hypothesis:** if recurrence is robust to pattern shape (the
  iterated function generalizes across band shapes), AA1 gets
  bpb similar to canonical. If shape-sensitive, AA1 worsens.
- **Compile audit:** same as 080. Eval-only, LOOP_PATTERN sets
  the index lists at __init__, one new graph variant on first
  eval call. No mid-run recompile.
- **Spec candidate: 085 = AA1.** **FREEZE THIS WAKE.**

**AA2. Eval at 073's stepped 2-3-3-2 pattern.** Companion to AA1
testing a different shape (2-3-3-2 across {3,4,5,6}, layer 5 not
peaked).

**AA3. Full sweep AA1+AA2+canonical with bpb table.** Reads as a
2D map: compute axis (080/081/082) × shape axis (AA1/AA2/canonical).

### Cluster BB — TTT prefix length × recurrence

PHASED_TTT_PREFIX_DOCS=2500 was tuned for the canonical NL=2 setting.
With deeper recurrence at eval (083, 084), the LoRA may benefit from
longer prefix (more adaptation budget for the deeper iteration) or
shorter (less overfit to the deeper-recurrence eval distribution).

**BB1. NL=3 eval × prefix_docs={1500, 2500, 3500} sweep.** Three
configs, each a config-only spec. Each tests how TTT prefix length
interacts with deeper-recurrence eval. Cost: ~$3-4 each, $9-12 total.

**BB2. NL=2 eval × prefix_docs sweep.** Baseline check — does the
canonical NL benefit from non-default prefix lengths? Important
for separating "TTT-prefix lever" from "TTT-prefix × deeper-NL
interaction."

Defer this to W7+ if time permits — these tests are most informative
*after* 083/084's results land.

### Cluster CC — Hot-pass conditioning (variant of N1, no compile risk)

W2's N1 (random pass-skip at training) was rejected for compile-graph
hazards. This wake: a static analog that's compile-safe.

**CC1. Per-pass dropout MASK (not a skip).** During training, with
some probability p, multiply a random pass's output by a small
factor (e.g. 0.5 instead of 1.0). Iteration count stays fixed; the
"skipped" pass contributes less. Always-tensor: a precomputed mask
tensor controls per-pass scaling.
- **Forces robustness:** model learns to function across different
  per-pass weight distributions.
- **Eval:** all passes at full weight (1.0).
- **Compile audit:** mask is a tensor, indexed deterministically per
  step. Single graph variant. No mid-run recompile.
- **Code change:** ~30 LOC. Defer.

### Decisions for W6

- **Spec 085 = AA1 (matched-compute broad pattern eval on 060A).**
  Eval-only, config-only, completes the (compute, shape) grid.
  **FREEZE THIS WAKE.**
- **AA3 (full sweep)** is the natural followup but requires running
  3+ eval configs. Single-spec with multi-arm structure for W7?
- **BB1 (TTT-prefix × NL sweep)** worth doing AFTER 083/084 land —
  conditional value.
- **V1/V2 demoted** — code change required.

---

## W7 — TTT × pattern shape, rolling-window eval, low-rank residual

### Cluster DD — LR schedule × deeper-NL interactions

The 060A model trained at NL=2 with a specific LR schedule (warmdown
fraction 0.85, ITERATIONS=20000, etc.). The Adam/Muon momentum
warmup steps (1500) are tied to the trained iteration count.

**DD1. Anti-finding: nothing to do here at eval.** The LR schedule
only affects training. Eval-time recurrence depth (specs 080-085)
doesn't interact with optimizer state. So this cluster collapses to
"the LR schedule is fixed at the trained model's settings; eval-time
NL changes don't see it."

**DD2. If we ever retrain with deeper NL:** then a longer warmup
(2000+ steps) might be needed because the deeper recurrence is
harder to optimize early. This is a *training-time* spec, not
eval-only. Defer to a future training-time recurrence-deepening
spec series.

### Cluster EE — Rolling-window eval (varies eval-time stride)

The current eval uses `EVAL_STRIDE=64` which means consecutive
evaluation windows overlap by stride=64 tokens. This is a fixed
hyperparameter at eval time.

**EE1. Smaller EVAL_STRIDE.** STRIDE=32 means more overlap → more
forward passes per evaluation token → more accurate per-token bpb
estimate. Costs ~2× eval compute (must process ~2× more windows).
- **Compile audit:** stride doesn't change the model graph; just
  changes how many forward calls happen. No mid-run recompile.
- **Wallclock:** doubles eval time. Fits in the 180s eval headroom?
  Need to verify by checking 060A's actual eval time vs cap.
- **Compose with deeper recurrence:** spec 080 ate ~7 min eval at
  NL=3. Add 2× from stride=32 → ~14 min. Pushes against eval cap.
- **Standalone test:** a config-only spec running 060A canonical
  with stride=32. Spec candidate.

**EE2. Adaptive stride per document.** Long docs use larger stride
(less overlap, faster); short docs use smaller stride (more overlap,
better measurement). Conditional eval scheduling. Implementation
non-trivial.

### Cluster FF — Low-rank residual stream during loop band

Refines Z1 (W5) — fp32 residual is one option; another is
*low-rank* residual structure during the loop band.

**FF1. SVD-projected residual at loop entry.** Compute the SVD of the
loop-entry residual once per layer-pass. Use only the top-k singular
modes for the residual computation, drop the rest. The "noise floor"
of the residual is removed; recurrence iterates on the cleaner signal.
- **Mechanism:** if late-pass updates are mostly orthogonal but small,
  they may live in the top-k singular subspace. Filtering keeps the
  signal-bearing modes and drops noise.
- **Implementation:** SVD inside compile region — risky. PyTorch's
  `torch.svd` may not be torch.compile-friendly. Could use a simpler
  truncation (e.g., zero-out small singular values via thresholding,
  using torch.svd on bf16 might break gradient flow).
- **Compile audit:** SVD adds significant compile complexity; one new
  graph variant per pass × layer combination. Likely safe if all
  shapes are static, but the SVD kernel itself may not be compiled.
  This idea needs prototyping; defer to W8+.

**FF2. Top-k subspace projection (lighter version).** Instead of full
SVD, maintain a learned k-dimensional projection matrix that maps the
residual to a k-dim subspace and back. Train k as a small parameter.
Does the recurrence iterate better in the projected space?
- **Code change:** ~50 LOC. Adds k×d and d×k matrices per loop layer.
  Configurable via env var.
- **Compile audit:** static shapes; one graph variant; no mid-run
  recompile.
- Defer.

### Decisions for W7

- **Spec 086 = TTT-on version of 085.** Direct extension. Composes
  the matched-compute broad-pattern test with TTT for leaderboard
  relevance. Config-only on `e7ccda2`. **FREEZE THIS WAKE.**
- **EE1 (stride=32 eval)** is a small, clean variant — config-only.
  Spec candidate for W8.
- **FF1/FF2 (low-rank residual)** are code-change candidates with
  modest implementation difficulty. Defer.
- **DD cluster collapses** — LR schedule isn't relevant for eval-only
  specs.

---

## W8 — band-position shifts, TTT-batch sensitivity, gradient alignment

### Cluster GG — Band-position shifts (matched compute, shifted location)

Sibling axis to AA (matched compute, different shape). GG asks: what
if we keep the *shape* (3 layers, NL=2 each = 9 visits) but shift
the *position* of the band?

**GG1. Loop band {2,3,4} (one layer earlier).** body =
"2,3,4,2,3,4,2,3,4,5,6,7" — 12 visits. pre [0,1] + post [8,9,10]
= 17 total. Layer 2 visited 4× (3 in body + 1 in pre — wait, with
pre = range(min(body)) = range(2) = [0,1], layer 2 not in pre).
Visits: {0:1, 1:1, 2:3, 3:3, 4:3, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1}
= 17 ✓. Tests "is the loop band {3,4,5} optimal positionally, or
would {2,3,4} also work?"
- Memory has prior negative result on this from #1726 (heavy reuse
  {2..7} catastrophic; layer band shifts harmful). But #1726 didn't
  test {2,3,4} specifically with NL=2.
- **Compile audit:** same as 080. Eval-only LOOP_PATTERN, one new
  graph variant, no mid-run recompile.
- **Spec candidate.**

**GG2. Loop band {4,5,6} (one layer later).** body =
"4,5,6,4,5,6,4,5,6,7" — 10 visits. pre [0,1,2,3] + post [7,8,9,10]
= 18 total. Hmm 18 not 17. Adjust: body = "4,5,6,4,5,6,4,5,6"
= 9, pre [0..3] + post [7..10] = 17 ✓. Visits: {0:1,1:1,2:1,3:1,
4:3,5:3,6:3,7:1,8:1,9:1,10:1}. Tests "later band positioning."

**GG3. Joint sweep GG1+GG2+canonical.** Three-point band-position
test at fixed compute. Each spec ~$1, total $3.

### Cluster HH — TTT batch size × deeper-NL interaction

The TTT phases use `TTT_BATCH_SIZE=64` and `TTT_CHUNK_SIZE=48`. With
deeper recurrence at eval, each TTT chunk takes longer to score
(more layer-passes per chunk). The TTT optimizer (Adam over LoRA)
sees fewer chunks per wall-second.

**HH1. Smaller TTT batch with deeper recurrence (082 + smaller TTT_BATCH_SIZE).**
Trade fewer chunks for finer-grained TTT updates. Tests whether the
TTT LoRA benefits from higher-frequency / lower-batch updates when
the underlying scoring is more expensive.
- Config-only spec; modifies TTT_BATCH_SIZE in addition to LOOP_PATTERN.
- **Compile audit:** TTT_BATCH_SIZE doesn't change the model graph;
  it changes the data loading. No new graph variant. Safe.

**HH2. TTT chunk size sweep × NL.** TTT_CHUNK_SIZE=48 was tuned at
NL=2. At NL=3 maybe larger chunks (96) work better (the deeper
recurrence already saw enough context). Or smaller (24) for finer
updates. Sweep candidate; multiple specs.

### Cluster II — Gradient-alignment recurrence (training-time, code change)

Speculative: during training, after the loop band's k-th pass, compute
the gradient of pass-k's output with respect to pass-0's output
(within the same forward). Add a regularizer that aligns these
gradients across passes — ensuring later-pass updates point in the
same direction as earlier passes. Smooths the iteration.

- **Engineering:** requires `torch.autograd.grad` *during* forward.
  Forces graph break in compile. **Skip for homestretch.**
- Documenting for completeness; not a near-term spec.

### Decisions for W8

- **Spec 087 = matched-compute stepped pattern eval (073-shape).**
  Direct sibling to 085 — different matched-compute shape. Together
  with 085 builds the shape sweep at fixed compute. Config-only on
  `e7ccda2`. **FREEZE THIS WAKE.**
- **GG cluster (band-position shifts)** good for W9-W10 if we have
  more wakes. Each is config-only.
- **HH (TTT batch size sensitivity)** is config-only and complementary
  to the (compute, shape) grid; defer to W9.
- **II demoted** — autograd-in-forward isn't compile-safe.

---

## W9 — TTT-extension on existing artifact, KV-projection truncation, smear-gate window

### Cluster JJ — Post-quant-only TTT extension

The full eval pipeline does: pre-quant eval → GPTQ → post-quant eval
with TTT phases. The TTT phases adapt LoRA on the *quantized* model.

**JJ1. More TTT phases at eval.** Current `PHASED_TTT_NUM_PHASES=3`.
Try 4 or 5 phases. Each phase costs ~1-2 min. Eval headroom (~180s)
might fit one extra phase, definitely fits 1-2 more if we drop other
overhead.
- **Compile audit:** number of phases is a Python loop bound, not a
  graph parameter. Same compiled `forward_ttt` graph, just called
  more times. **No mid-run recompile.**
- **Spec candidate.**

**JJ2. Larger TTT_LORA_RANK at eval-time.** Currently 80 (came from
060A canonical). Try 96 or 128. The LoRA matrices are bigger; uses
some of the 98 KB byte budget at the cost of capacity. But this is
EVAL ONLY — the LoRA weights are computed on-the-fly during TTT, not
baked into the submission artifact. So byte budget isn't affected.
- **Catch:** the LoRA was *initialized* during training at rank 80;
  loading the checkpoint and switching to rank 128 means new (mostly
  zero) LoRA banks. The model may not have the capacity to use the
  extra rank effectively in 3 TTT phases.
- **Compile audit:** rank changes the LoRA tensor shapes, which
  changes the compiled forward_ttt graph. ONE new graph variant on
  first TTT phase. No mid-run recompile.
- **Defer:** uncertain whether this helps; needs careful analysis.

### Cluster KK — KV-projection truncation at eval

Memory mentioned 047B (loop KV shrink) trained-time was killed
on size, +0.0028 quality cost at +10% throughput. **Eval-only**
KV truncation hasn't been tested.

**KK1. Use only top-N KV-rank at eval.** Project K and V through a
truncated SVD (top-r modes) before attention. Reduces attention
softmax noise floor. Eval-time: weights unchanged; just a runtime
projection.
- **Compile audit:** SVD inside compile is risky (PyTorch SVD has
  graph-break behavior). Truncated projection via a fixed mask
  could work but requires picking the mask offline. Engineering cost.
- **Defer for code change.**

### Cluster LL — Expanded smear-gate window at eval

The model has a SmearGate mechanism with `gate_window=12` (from
hyperparam dump). The gate creates a per-token forward-1 smear of
the embedding lane, with a sigmoid gate over the first 12 dims.

**LL1. Wider gate_window at eval.** Try gate_window=24 (use more
embedding dims for the gate signal). Eval-only override; the
trained gate weights operate on the first 12 dims; expanding to 24
means using untrained dims as gate input. Almost certain to be
neutral or harmful since the trained weights expect 12 dims.
- **Drop.** Unlikely to help; high blast radius.

**LL2. Shrink gate_window at eval to 6.** Inverse: smaller window,
narrower gate. Same problem in reverse — trained weights expect 12.
- **Drop.**

### Decisions for W9

- **Spec 088 = TTT-on stepped pattern eval.** Direct sibling to 087
  (TTT-off stepped pattern). Composes 087 with TTT for the
  leaderboard-relevant matched-compute stepped-shape test. Config-
  only on `e7ccda2`. **FREEZE THIS WAKE.**
- **JJ1 (more TTT phases)** is config-only and complementary; spec
  candidate for W10.
- **JJ2 (larger LoRA rank at eval)** has uncertain value; defer.
- **KK / LL clusters demoted** — code-change-required or low-value.

---

## W10 — Band-position shift, eval-only LR-style overrides, lane-split during loop

### Cluster MM — Eval-only LR-style overrides

These overrides apply only at eval time when no training optimizer
state exists. Mostly null since LR doesn't affect eval forward — but
some apply to TTT phases:

**MM1. TTT-LR sweep at NL=2 baseline.** `TTT_LORA_LR=0.0001` is the
default. Try 5e-5 (smaller, more conservative LoRA updates) or 2e-4
(larger, faster adaptation). Tests whether canonical TTT is at the
right LR for our checkpoint.
- **Compile audit:** TTT_LORA_LR is a Python float used by Adam in
  TTT phases; no graph effect. Safe.
- **Spec candidate** for W11+ if simpler than other ideas.

**MM2. TTT-LR scaling with NL.** When eval-NL doubles (NL=2→4 = 23 vs
17 passes), the LoRA gradients flow through more layer-passes per
chunk. Effective LR per Adam step is amplified by chain-rule depth.
Could either need smaller LR (to compensate) or larger (to keep up).
- **Companion to 084 (NL=4 + TTT)** — would clean up that spec's
  result if 084 lands ambiguous.

### Cluster NN — Recurrence × sliding window attention

The model has `gate_window=12`. Memory and code mention attention is
basic (no sliding window in 060A). For deeper recurrence at eval, we
could *introduce* a sliding window only on the loop layers, with
window size matched to the recurrence depth.

**NN1. Sliding-window attention only on loop layers, eval-only.** Per
loop pass, the attention attends over a window of N tokens (not full
context). Smaller window = cheaper attention compute, partially offsets
the deeper-recurrence cost.
- **Engineering:** real code change to `CausalSelfAttention.forward`
  to accept a window-mask argument, gated by env var on loop layers.
  ~50 LOC. Compile audit: mask is a static tensor, single graph
  variant.
- **Defer for code change.**

**NN2. Per-pass increasing window size.** Pass 0 uses window=64,
pass 1 uses window=256, pass 2 uses window=full. Mimics "iterative
attention expansion." More speculative.

### Cluster OO — Multiple residual streams during loop band

The architecture already has parallel lanes after `parallel_start_layer=8`
(decoder side, two lanes merged at end). Idea: extend lane structure
to the loop band itself.

**OO1. Lane-split inside loop band.** At loop entry, split residual
into 2 lanes; each lane runs its own NL=2 recurrence; merge at loop
exit.
- **Mechanism:** lanes carry different views of the residual (e.g.,
  one for syntactic features, one for semantic). Independent
  recurrence per lane.
- **VRAM cost:** 2× activation memory in the loop band. Have headroom.
- **Param cost:** adds lane-split / lane-merge weights (~2 × d² for
  the 2D affine projection). ~500K params. Doesn't fit byte budget.
- **Reduced version:** lane-split is a fixed orthogonal projection
  (no learned weights); merge is sum or learned-weighted sum (~d
  params). Cheaper.
- **Defer code change.**

### Decisions for W10

- **Spec 089 = GG1 band-position shift to {2,3,4} eval.** Novel
  positional axis — never tested at NL=2 on canonical band shape.
  Memory's #1726 only tested wider bands {2..7}, not narrower
  shifted band {2,3,4}. Config-only on `e7ccda2`. **FREEZE THIS WAKE.**
- **MM1 (TTT-LR sweep)** is config-only; possible W11 spec.
- **NN1 (sliding-window on loop layers)** is the most novel idea
  this wake — code-change required, defer.
- **OO1 (lane-split in loop band)** has param-budget issues at full
  rank; reduced version is interesting future work.

---

## W11 — Position symmetry, hotstart-free verification, partial-rank weight reads

### Cluster PP — Position-axis symmetry (companion to 089)

**PP1. Band shift {4,5,6} (one layer later, GG2 from W8).** Sibling
to 089. Together with 089 brackets the canonical {3,4,5} band on
both sides. Reads as a 3-point position curve: {2,3,4} → {3,4,5} →
{4,5,6}.
- Pattern body: `4,5,6,4,5,6,4,5,6` (9 visits). pre [0,1,2,3] +
  post [7,8,9,10] = 17 total.
- **Spec candidate: 090 = PP1.** **FREEZE THIS WAKE.**

**PP2. Position symmetry hypothesis test.** If 089 ({2,3,4}) and 090
({4,5,6}) lose by similar magnitudes, the canonical band is
positionally optimal. If they lose asymmetrically, there's a "right
direction" to shift — informs future band-design specs.

### Cluster QQ — Hotstart-free verification

A concern with all the eval-only 080+ specs: they all depend on
060A's saved checkpoint. If that checkpoint differs subtly from what
080+ specs assume (e.g., parameter shapes, version of LOOP_PATTERN
in __init__), bad results would be from the dependency, not the
hypothesis.

**QQ1. Sanity-check spec for 060A canonical at e7ccda2.** Run 060A's
canonical config (LOOP_PATTERN unset, NUM_LOOPS=2, LOOP_START=3,
LOOP_END=5) at the e7ccda2 commit (which adds LOOP_PATTERN env var
but doesn't otherwise change the path). Should produce identical
val_bpb to 060A's reported number.
- Tests: does the LOOP_PATTERN code change introduce any unintended
  side effects when LOOP_PATTERN is empty?
- Cost: ~$1, ~10 min eval-only.
- **Sanity check, not headline science.** Worth doing before
  promoting any 080+ spec. Spec candidate for W12.

### Cluster RR — Partial-rank weight reads at eval (training-time orthogonal)

**RR1. SVD-truncated weight reads on loop layers.** At eval, before
each Block.forward call on a loop layer, SVD-decompose the weight
banks (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank for that layer)
and use only the top-k singular components.
- **Mechanism:** removes "noise rank" from weights — top-k components
  carry signal; tail-rank may add quantization noise especially in
  bf16.
- **Engineering:** SVD per Block call inside compile region — risky
  graph break. Alternative: do SVD ONCE at startup (after checkpoint
  load), replace weights with truncated reconstructions. Then forward
  uses standard weights. **Static** transformation, compile-safe.
- **Code change:** ~30 LOC for the startup SVD truncation, gated by
  env var (e.g., `EVAL_WEIGHT_RANK=64`). One-time SVD per loop
  layer's banks; runtime forward unchanged.
- **Compile audit:** completely safe — weights replaced before first
  forward call.
- **Spec candidate for W12+ if a code-change spec is wanted.**

### Decisions for W11

- **Spec 090 = PP1 = band shift {4,5,6} eval.** Completes the
  positional axis pair with 089. Config-only on `e7ccda2`. **FREEZE
  THIS WAKE.**
- **QQ1 (sanity-check 060A at e7ccda2)** is a good "due diligence"
  spec — defer to W12 only if no other novel idea matures.
- **RR1 (startup-SVD-truncated weights)** is the most novel idea
  this wake — code change required, defer.

---

## W12 — TTT compute axis, eval determinism, multi-checkpoint averaging

So far the eval-time specs (080-090) have varied loop pattern (compute,
shape, position) and TTT on/off. Today's wake explores a different
eval-time-headroom lever: **the TTT compute axis itself**.

### Cluster SS — TTT compute axis (orthogonal to LOOP_PATTERN)

We have ~100-180s of unused eval wallclock. Most of 080-090 spent
that on extra layer-passes via LOOP_PATTERN. SS asks: spend it on
TTT instead.

**SS1. More TTT phases at canonical NL.** PHASED_TTT_NUM_PHASES=3
default. Each phase: TTT update → score next chunk → repeat. Adding
phase 4 means ~30-50% more TTT compute, more LoRA adaptation
opportunities.
- **Hypothesis:** 4 phases lets the LoRA adapt further to the eval
  distribution; fits in eval-time headroom.
- **Compile audit:** number of phases is a Python loop bound. Same
  compiled `forward_ttt` graph called more times. **No new graph
  variant. No mid-run recompile.**
- **Cost:** ~$3-4 (canonical eval pace + 1 extra phase ≈ +1-2 min).
- **Spec candidate: 091 = SS1.** **FREEZE THIS WAKE.**

**SS2. Larger TTT prefix at canonical NL.** Memory has spec 060L
(`PHASED_TTT_PREFIX_DOCS` up). May be already specced — check before
duplicating. The idea: longer prefix per phase = more adaptation
data per LoRA step.

**SS3. Combined TTT-extension + canonical recurrence.** No deeper
recurrence; just spend the eval budget on TTT compute. Pure TTT-axis
test isolates whether the headroom is best spent on TTT or on
recurrence depth.

### Cluster TT — Eval-time determinism (sanity)

The 080+ specs all use seed=42 to match 060A. But the eval-time path
may have nondeterministic operators (e.g., flash-attention tile
order, all_reduce ordering across ranks). Verifying determinism
helps disambiguate "real signal" from "rng noise."

**TT1. Run 060A canonical at e7ccda2 twice; verify exact match.** If
identical to 5+ decimal places, eval-time path is deterministic.
If not, all 080+ "Δ ≤ 0.0005 noise floor" thresholds need revisiting.
- **Cost:** $2 for two eval runs.
- **Engineering value: HIGH** but not a science result.
- Defer to user-discretion spec.

### Cluster UU — Multi-checkpoint averaging

A more speculative lever: load *multiple* trained checkpoints
(if available — e.g., 060A and 060A-with-different-seed) and
average their weights before eval.

**UU1. Weight-averaged 060A across seeds.** Memory says baseline
should be multi-seed. If we have seed-42 + seed-43 + seed-44
checkpoints, average their weights → run eval. Should produce a
smoother model. Common technique in DL ensembles.
- **Catch:** we may not have multiple seeds of 060A. Memory implies
  only seed-42 is saved. Skipped.

**UU2. Inter-step EMA outside training.** During training, EMA decay
is 0.9965. After training, we have the EMA-applied weights. Could
we keep the *non-EMA* weights too and run a different decay at
eval? Speculative; would need both saved which we likely don't have.

### Decisions for W12

- **Spec 091 = SS1 (more TTT phases at canonical).** Tests the
  TTT-compute axis directly. Config-only on `e7ccda2`. Real
  leaderboard relevance (different lever from LOOP_PATTERN).
  **FREEZE THIS WAKE.**
- **TT1 (determinism check)** is good engineering hygiene but
  low-priority for science output. Defer.
- **UU cluster** depends on having multiple checkpoints we likely
  don't have. Drop.

---

## W13 — Position × TTT cell completion + GPTQ-side levers

### Cluster VV — Position-axis × TTT cell completion

089 (band {2,3,4} TTT-off) and 090 (band {4,5,6} TTT-off) cover the
two position shifts at TTT-off. **TTT-on versions complete the
position-axis × TTT grid** for leaderboard relevance.

**VV1. TTT-on version of 089 (band {2,3,4} + TTT).** Same pattern as
089 (`2,3,4,2,3,4,2,3,4,5,6,7`) but with TTT and GPTQ enabled.
Real leaderboard test of "earlier band shift."
- **Compile audit:** same as 089 + 083 — forward_logits and
  forward_ttt each compile once with new index lists, no mid-run
  recompile. Bank weights unchanged from 060A.
- **Spec candidate: 092.** **FREEZE THIS WAKE.**

**VV2. TTT-on version of 090 (band {4,5,6} + TTT).** Sibling. May be
W14's freeze.

### Cluster WW — GPTQ calibration sweep at canonical NL

`GPTQ_CALIBRATION_BATCHES=16` is the default. Memory has spec 046A
in the `GPTQ_CALIBRATION_BATCHES` sweep family — likely already
explored at canonical 060A. Verify before duplicating.

**WW1. GPTQ_CALIBRATION_BATCHES=32 at canonical.** More calibration
data → potentially better quantization → smaller pre→post quant gap.
- Already may be specced; defer.

### Cluster XX — Eval-time residual perturbation

A novel direction: at eval, *perturb* the residual stream slightly
between loop passes. Tests whether the trained recurrence is
robust to noise (it should be — it's already a contraction).

**XX1. Pre-pass dropout at eval.** Apply Bernoulli dropout (p=0.05)
to the residual at the entry to each loop pass. Forces the model
to be robust; at high enough p, may improve generalization
(implicit regularizer at eval).
- **Eval-time only**, no training change. The trained model never
  saw dropout, so this is a regularization-at-test technique.
- **Compile audit:** dropout adds randomness inside the compile
  region. With `torch.dropout` and a fixed seed, the kernel is
  deterministic per-step but produces different masks per call. One
  graph variant. **No mid-run recompile** if seed-controlled.
- **Code change required:** ~10 LOC in `_forward_hidden` — wrap the
  block call with a dropout call gated by an env var.
- **Defer for code change.** Risk: trained models without dropout
  often degrade with eval-time dropout; this is a "wide-net" test.

**XX2. Quantization-noise-style perturbation at eval.** Add
deterministic small noise to weights (matching INT6 quant noise
scale) at eval-time. Tests whether trained weights are robust to
their own quantization. Speculative; not clearly useful.

### Decisions for W13

- **Spec 092 = VV1 (TTT-on band {2,3,4}).** Direct extension of 089
  to leaderboard relevance. Config-only on `e7ccda2`. **FREEZE THIS
  WAKE.**
- **VV2 (TTT-on band {4,5,6})** for W14.
- **WW (GPTQ calibration sweep)** likely already specced under 046A;
  verify before duplicating.
- **XX (eval-time residual perturbation)** is novel but code-change
  required; defer.

---

## W14 — TTT-LR sweep, SmearGate sensitivity, repeated-context

### Cluster YY — TTT_LORA_LR sweep at canonical NL

`TTT_LORA_LR=0.0001` is the default. Eval-time hyperparameter that
controls how aggressively the LoRA adapts during TTT phases.

**YY1. TTT_LORA_LR=5e-5 at canonical.** Smaller LR → more conservative
adaptation. Tests "is canonical LR too aggressive for the eval
distribution?"
- Config-only spec; no graph effect.
- Cost: ~$3-4 (full pipeline eval).

**YY2. TTT_LORA_LR=2e-4 at canonical.** Larger LR → faster adaptation,
risk of overshoot. Tests "is canonical LR too conservative?"

**YY3. Joint sweep** with the deeper-recurrence specs (083/084).
LR may need to scale with effective recurrence depth (deeper
recurrence = longer chain rule = effectively larger gradients).
Combination spec.

### Cluster ZZ — Eval-time SmearGate alteration

The model has a SmearGate mechanism (`smear_gate_enabled`) at the
embedding level. It's tied to `gate_window=12` and a learned gate
weight. Memory cluster LL (W9) considered changing gate_window
and dropped. But there's another SmearGate lever:

**ZZ1. Disable SmearGate at eval (`SMEAR_GATE_ENABLED=0`).** The
model trained with SmearGate; turning it off at eval removes its
forward-1 smear. Tests whether the trained model relies on the
smear or whether it's a redundant mechanism the model has learned
around.
- **Compile audit:** SMEAR_GATE_ENABLED gates a Python `if` in
  `_forward_hidden` (around line 1332). When disabled, the
  embedding `x` doesn't go through the smear path. This is a
  graph-level change → one new graph variant on first call. Same
  iteration count, no mid-run recompile.
- Almost certainly hurts (the model trained with it). But cheap
  test.

### Cluster AAA — Repeated-context evaluation (data-side)

A different "free eval-compute" usage: at eval, present each
sequence to the model TWICE (concatenate sequence with itself).
The model autoregressively predicts. The second copy benefits from
the first as in-context. Logits on the second copy may be lower
loss → free bpb improvement.

**AAA1. Repeated-context eval.** Concat each val sequence with itself
before scoring. Doubles eval data per sequence, requires careful
loss-mask to score only the second half (first half is context).
- **Engineering: real change to validation data preparation.** Not
  a config-only spec.
- **Compile audit:** sequence length doubles → graph variant on first
  call. One new variant; cached. No mid-run recompile.
- **Score-mask handling:** the existing eval scores all positions in
  the window with stride=64. We need to ignore positions in the
  first-copy half. Adds complexity to the eval loop.
- **Defer.**

### Decisions for W14

- **Spec 093 = VV2 = TTT-on band {4,5,6}.** Completes the position ×
  TTT grid (089/092 covered {2,3,4}; 090/093 cover {4,5,6}).
  Config-only on `e7ccda2`. **FREEZE THIS WAKE.**
- **YY (TTT-LR sweep)** is config-only and worth doing if budget
  allows. Multiple specs (YY1, YY2). Defer to W15+.
- **ZZ1 (disable SmearGate at eval)** is config-only but predicted
  to lose. Low-priority diagnostic.
- **AAA (repeated-context eval)** is genuinely novel but requires
  validation-side code change. Defer.

---

## W15 — TTT-momentum, logit softcap, XSA on/off

### Cluster BBB — TTT-momentum sensitivity at canonical NL

The TTT optimizer uses Adam with `TTT_BETA1=0.0` and `TTT_BETA2=0.99`
(per memory `project_baseline_1851_post_1872`'s notes on the
060A/#1855 stack). β1=0 means no momentum on the gradient itself;
β2=0.99 is the variance-tracking term.

**BBB1. TTT_BETA1=0.9 at canonical.** Add momentum back into TTT
Adam. Tests whether the canonical β1=0 (which is unusual — typically
0.9) was an active design choice or a default that didn't get tuned.
- Config-only spec; no graph effect (Adam state is Python-side).
- Cost: ~$3-4.

**BBB2. TTT_BETA2 sweep at canonical.** Try 0.95, 0.999. Same axis
of "is the TTT optimizer right-tuned?" Less novel than BBB1 since
β2 sweeps are common.

### Cluster CCC — Eval-time logit softcap variant

`LOGIT_SOFTCAP=30` is the default. Memory's spec 044 family (qk-gain
soft) explored similar territory. Eval-time softcap can be different
from training (model is still applied; softcap is a tanh on logits).

**CCC1. LOGIT_SOFTCAP=20 at eval.** Tighter softcap → more
suppression of extreme logits → potentially smoother bpb.
- **Compile audit:** softcap is a Python float used as
  `logit_softcap * tanh(logits / logit_softcap)`. Static. No graph
  effect when value changes (one cached graph variant per value).
  Safe.
- Lower priority — already partially explored in spec 044 family.

### Cluster DDD — Eval-time XSA on/off

The model has XSA (eXclusive Self-Attention) per memory; it's
implementation-related to the modified attention computation in 060A.
`XSA_LAST_N=11` means all 11 layers use XSA.

**DDD1. Disable XSA at eval (`XSA_LAST_N=0`).** Tests whether the
model relies on XSA at inference or whether canonical attention
is fine.
- Almost certainly hurts (the model trained with XSA throughout).
- Diagnostic value low; defer.

### Decisions for W15

- **Spec 094 = YY1 = TTT_LORA_LR=5e-5 at canonical.** Direct
  sibling to 091 (same TTT-compute axis, different parameter).
  Tests whether the canonical TTT LR is over-aggressive. Config-only.
  **FREEZE THIS WAKE.**
- **BBB1 (TTT_BETA1=0.9)** is interesting and config-only; spec
  candidate for W16.
- **CCC, DDD demoted** — already partially explored or low-value
  diagnostics.

---

## W16 — XSA layer subset, val sub-sampling, eval-time mixed precision

### Cluster EEE — XSA on a layer subset (refines DDD)

DDD1 (W15) considered disabling XSA entirely. More refined: disable
XSA only on the loop layers (where recurrence may interact differently
with XSA's exclusivity correction).

**EEE1. XSA only on non-loop layers.** Set `XSA_LAST_N=11` but
override on loop layers via a hypothetical env var. Tests "does XSA
help or hurt during recurrence?"
- Code change required: ~10 LOC adding per-layer XSA flag.
- Compile audit: trivial — adds a static if/else per layer at
  __init__. One graph variant.
- Defer.

### Cluster FFF — Validation sub-sampling for fast iteration

For diagnostic specs (where we want a fast bpb estimate, not the
full leaderboard number), use a subset of the val set.

**FFF1. Eval on first 1024 sequences only.** Full val set has many
more; using 1024 is faster but still gives a reasonable variance
estimate.
- **Cost reduction:** ~5-10× faster eval (depending on full set size).
- **Use case:** would let us run dozens of variant specs at $0.10
  each instead of $1+.
- **Caveat:** sub-sampled val_bpb has higher variance; not reliable
  for sub-0.001 deltas.
- **Compile audit:** sequence count is data-loop-bound; no graph
  effect.
- Could be a meta-spec ("use this for cheap exploration") more than
  a science result. Defer to research-time tooling.

### Cluster GGG — Eval-time mixed-precision

The model uses bf16 throughout. Could we eval with parts of the model
in fp32 for higher precision?

**GGG1. fp32 head + softmax at eval.** The output projection (tied
embedding) and the cross-entropy computation can run in fp32 for
better numerical precision in the bpb measurement.
- The training-time path already uses softcapped CE which controls
  numerics; this would be a redundant change.
- Actually MEMORY mentions spec 060A uses `fused_ce_enabled` and
  `softcapped_cross_entropy` — already optimized. Lower priority.

**GGG2. fp32 attention softmax on loop layers.** During recurrence,
each loop pass's softmax is computed in bf16. fp32 could improve
precision in the iterative refinement.
- Same theme as Z1 (W5) but more targeted.
- Code change required, defer.

### Decisions for W16

- **Spec 095 = YY2 = TTT_LORA_LR=2e-4 (larger LR).** Direct sibling
  to 094. Completes the LR direction sweep. Together with canonical
  (1e-4) and 094 (5e-5) gives a 3-point LR scan. Config-only.
  **FREEZE THIS WAKE.**
- **EEE / FFF / GGG clusters** all require code changes or have
  unclear value. Defer.

---

## W17 — TTT momentum activation + prefix-doc sweep + 1-shot TTT-post-hoc

### Cluster HHH — TTT-momentum activation (BBB1 reframe)

`TTT_BETA1=0.0` in 060A is unusual — Adam normally uses β1=0.9 to
dampen gradient noise via momentum. β1=0 means each LoRA update
is purely the (gradient × LR) signal, no smoothing.

**HHH1. TTT_BETA1=0.9 at canonical.** Add momentum back. Tests
whether 060A's β1=0 was a tuned choice or a stripped-down default.
- Config-only spec; Adam state lives Python-side.
- Cost ~$3-4 (full pipeline eval).
- **Spec candidate: 096.** **FREEZE THIS WAKE.**

**HHH2. TTT_BETA1=0.5 (intermediate).** If β1=0.9 hurts but β1=0.0
is also suboptimal, an intermediate value may be best. Defer to
followup if HHH1 is suggestive.

### Cluster III — Prefix-doc count sweep

`PHASED_TTT_PREFIX_DOCS=2500` controls how many prefix documents the
TTT optimizer adapts on per phase. More docs = more adaptation data
per phase but proportionally more time per phase.

**III1. PHASED_TTT_PREFIX_DOCS=1500 (smaller, ~60% of canonical).**
Less adaptation per phase; faster phases. Trade-off: less context per
LoRA step vs more "phases-per-second" effective.
- Config-only.
- Already partially explored in spec 060L if it exists; verify.

**III2. PHASED_TTT_PREFIX_DOCS=3500 (larger).** More adaptation per
phase. Risk: phases run longer, may push past eval budget.

### Cluster JJJ — 1-shot TTT-post-hoc on top of canonical

The 060A model has TTT applied during eval. A novel idea: BEFORE
running TTT, run a lightweight one-shot adaptation step that warms
the LoRA on a small global pre-prefix.

**JJJ1. Add a single "warmup" TTT step with smaller LR before phase 1.**
- Code change: ~30 LOC in eval pipeline. Extra Adam step on a
  pre-pre-prefix.
- Compile audit: extra forward_ttt call, same compiled graph, called
  once more. No new graph variant.
- Defer for code change.

### Decisions for W17

- **Spec 096 = HHH1 = TTT_BETA1=0.9 at canonical.** Novel axis test
  (TTT momentum). Config-only on `e7ccda2`. **FREEZE THIS WAKE.**
- **III1 (smaller prefix)** is config-only and complementary; spec
  candidate W18.
- **JJJ (1-shot warmup TTT)** requires a code change; defer.

---

## W18 — TTT batch granularity, chunk size, β2 variant

### Cluster KKK — TTT batch granularity (HH1 reframe)

`TTT_BATCH_SIZE=64` is the default. The TTT optimizer accumulates
gradients over batches of this many sequences per Adam step.

**KKK1. TTT_BATCH_SIZE=32 at canonical.** Smaller batches = more
frequent Adam updates per phase, finer-grained adaptation.
- Hypothesis: smaller batches give the optimizer more signal-to-noise
  rolls; could improve adaptation quality.
- Caveat: smaller batches also mean noisier gradients per step.
- Config-only spec.
- **Spec candidate: 097.** **FREEZE THIS WAKE.**

**KKK2. TTT_BATCH_SIZE=128 at canonical.** Larger batches = fewer
updates, smoother gradients. Tests inverse direction.

### Cluster LLL — TTT chunk size sweep (HH2 reframe)

`TTT_CHUNK_SIZE=48` controls the per-chunk forward token count
during TTT. Each chunk runs a forward, then the loss is used for the
LoRA update.

**LLL1. TTT_CHUNK_SIZE=24 (smaller).** Each chunk has less context;
more chunks per phase. Trade-off: more updates vs less per-update
context.

**LLL2. TTT_CHUNK_SIZE=96 (larger).** Each chunk has more context;
fewer chunks. More-context-per-update.

These are config-only and conceptually separate from KKK
(batch granularity vs context length).

### Cluster MMM — TTT_BETA2 variant

Sibling to 096 (TTT_BETA1=0.9). Canonical TTT_BETA2=0.99. Adam's
default is 0.999 (more variance smoothing).

**MMM1. TTT_BETA2=0.999 at canonical.** More smoothing of the
variance estimate. Standard Adam practice.

**MMM2. TTT_BETA2=0.95 at canonical.** Less smoothing; faster
adaptation to local variance.

Both config-only.

### Decisions for W18

- **Spec 097 = KKK1 = TTT_BATCH_SIZE=32 at canonical.** Novel axis
  (TTT batch granularity). Config-only. **FREEZE THIS WAKE.**
- **LLL (chunk size sweep)** is config-only and orthogonal; W19
  candidate.
- **MMM (β2 variant)** is config-only; could complete the Adam
  hyperparameter mapping started by 096. W19+ candidate.

---

## W19 — TTT weight decay, GPTQ-only, LR-warmup ramp

### Cluster NNN — TTT weight decay (orthogonal to LR/momentum)

`TTT_WEIGHT_DECAY=0.5` per the 060A canonical config. Adam-W weight
decay regularizes LoRA weights toward zero. This is a third
optimizer dimension besides LR and β.

**NNN1. TTT_WEIGHT_DECAY=0.0 at canonical.** No regularization. Tests
whether the LoRA needs WD to behave well in 3 phases.
- Hypothesis: at small phase counts, WD pull is small per phase; may
  not matter. If it doesn't, we can remove the regularizer.
- Config-only.

**NNN2. TTT_WEIGHT_DECAY=1.0 at canonical.** Stronger regularization.
Tests inverse direction.

### Cluster OOO — GPTQ-only eval (skip TTT entirely)

Cleanest TTT-vs-canonical test: at eval, run GPTQ but skip TTT
phases entirely. Measures the model's intrinsic post-quant val_bpb.

**OOO1. PHASED_TTT_NUM_PHASES=0 at canonical.** Disable TTT phases
(0 iterations). Eval the GPTQ-quantized 060A model without any
adaptation. Compare to 060A canonical (which has 3 phases).
- The delta tells us: how much value does TTT contribute on top of
  pre-quant + GPTQ?
- Config-only.
- **Diagnostic value HIGH** — calibrates the leverage of TTT in the
  full pipeline.

### Cluster PPP — TTT-LR-warmup ramp

Currently TTT runs at fixed `TTT_LORA_LR=1e-4` from phase 1. Warmup-
ramp could start at small LR (1e-5) and grow to canonical (1e-4) over
the 3 phases.

**PPP1. Warmup-ramp from 1e-5 to 1e-4 over 3 phases.** Each phase
uses an exponentially-increasing LR.
- **Code change required:** ~20 LOC in TTT phase loop to compute
  per-phase LR.
- **Compile audit:** LR is Python-side; no graph effect.
- **Defer for code change.**

### Decisions for W19

- **Spec 098 = MMM1 = TTT_BETA2=0.999 at canonical.** Completes the
  Adam β-mapping started by 096 (TTT_BETA1=0.9). 060A canonical has
  β1=0.0, β2=0.99 — both unusual. Spec 096 tests β1=0.9 direction;
  098 tests β2=0.999 direction. Independent axes. Config-only.
  **FREEZE THIS WAKE.**
- **NNN1 (TTT_WEIGHT_DECAY=0.0)** is config-only and orthogonal —
  good W20 candidate.
- **OOO1 (TTT-disable diagnostic)** is high-info and config-only;
  W20 candidate.
- **PPP (LR warmup)** requires code change; defer.

---

## W20 — TTT vs no-TTT calibration, GPTQ-only baseline

### Cluster QQQ — TTT contribution diagnostic (OOO1 reframed)

The cleanest "what does TTT actually buy?" measurement: at eval, run
GPTQ but skip TTT entirely. Compares to canonical post-TTT
post-quant val_bpb to size the TTT lever.

**QQQ1. TTT-disabled at canonical NL.** `TTT_ENABLED=0` and/or
`PHASED_TTT_NUM_PHASES=0`. Eval the GPTQ-quantized model directly
without LoRA adaptation.
- The delta tells us: how much value does TTT contribute on top of
  pre-quant + GPTQ?
- Calibration baseline for all the other TTT-axis specs (091, 094,
  095, 096, 097, 098). If TTT contributes 0.005 bpb total, then any
  TTT-axis improvement of 0.001 is a 20% relative improvement. If
  TTT contributes 0.001 bpb total, then 0.001 axis-improvement is a
  100% relative improvement.
- Config-only.
- **Spec candidate: 099 = QQQ1.** **FREEZE THIS WAKE.**

### Cluster RRR — Single-phase TTT diagnostic

A finer-grained calibration than QQQ1: run TTT with exactly 1 phase
instead of 0 or 3. Tells us:
- 099 vs 060A canonical: 3 phases of TTT contribution
- 100 vs 099 (RRR1): 1 phase of TTT contribution
- 060A canonical vs 100: marginal phases 2-3 contribution

**RRR1. PHASED_TTT_NUM_PHASES=1 at canonical NL.** Config-only.
- Combined with 099 and canonical: gives a 3-point phase-count
  calibration (0, 1, 3). Memory-cited 091 already adds the 4-point.
- W21 candidate.

### Cluster SSS — TTT × deeper recurrence as a "TTT-amplification" test

If 091 (TTT phases up) wins on canonical NL, the question is whether
TTT-axis amplification compounds with recurrence-axis amplification.

**SSS1. PHASED_TTT_NUM_PHASES=4 + LOOP_PATTERN at NL=3 eq.** Combines
091's TTT extension with 080's recurrence extension. If both win
individually, this composes.
- Cost ~$4-5 (full pipeline + extra TTT phase).
- Compile audit: same as 083 — forward_logits and forward_ttt
  compile once each, no mid-run recompile.
- W21 or W22 candidate.

### Decisions for W20

- **Spec 099 = QQQ1 = TTT-disabled at canonical.** Calibration
  diagnostic; sizes the TTT lever. Config-only on `e7ccda2`.
  **FREEZE THIS WAKE.**
- **RRR1 (1-phase TTT)** is the natural finer-grained companion;
  spec candidate W21.
- **SSS1 (compound TTT × recurrence)** is config-only and
  potentially leaderboard-relevant; W22 candidate.

