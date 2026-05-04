# Parameter Golf — Writeup Outline

Substack blog post. Tone: formal for Parts 0–1, semi-formal + funny for Parts 2–3.
Audience: half-technical, half non-technical.

---

## Part 0 — What Is This Competition?

- **The premise:** 16 MB artifact, 10 minutes training on 8×H100, scored by
  byte-level BPB on the FineWeb validation set (tokenizer-agnostic). OpenAI
  sponsoring $1M in compute grants.
- **The framing:** L(N) optimization — fixed parameters, unconstrained
  compute/data/architecture. Sister challenges: NanoGPT Speedrun (L(T)),
  NanoGPT Slowrun (L(D)).
- **The baseline:** 9-layer 512-dim 1024-vocab tied-embedding transformer →
  **1.2244 bpb**
- **Scoring:** what "byte-level BPB" means and why it matters — the model
  thinks in tokens, but the judge counts bytes, making the tokenizer choice a
  first-class compression lever.
- **The C1–C4 rules** (emerged mid-competition from Issue #1017, not present
  on day one):
  - C1: causal eval only — no peeking at future tokens
  - C2: full normalized distribution — the scoring must be a real probability
    distribution over bytes, summing to exactly 1
  - C3: score-before-update — in TTT, a chunk must be fully scored *before*
    any gradient step touches it
  - C4: single pass — each validation token is scored exactly once, no
    multi-pass re-scoring
- **Statistical significance bar:** new records require a 3-seed mean beating
  prior SOTA with a p-value threshold; borderline submissions land as
  "non-record" entries.

---

## Part 1 — The Key Techniques

Four pillars. Each gets: the core insight, who pushed it, representative BPB
numbers, and what it contributed to the final model.

### 1. Tokenizer / Vocabulary

- SP1024 (baseline) → SP4096 → SP8192 — counterintuitive: a bigger vocabulary
  *helps* in a tight parameter budget because better tokenization is free
  compression before the model runs.
- **CaseOps** (romeerp, PR #1729): a lossless bijective case transform that
  folds upper/lowercase into a compact token set, recovering original bytes via
  a sidecar. Drops the frontier from ~1.07 to ~1.065 in one shot.
- Why byte-level BPB makes tokenizer choice load-bearing: the scorer charges
  per byte regardless of how the model tokenizes, so every token that encodes
  more bytes is a free win.

### 2. Training Architecture

Key innovations in roughly the order they appeared:

- **XSA (Cross-Sequence Attention)** — lets the model attend across document
  boundaries within a batch; biggest single jump of the middle phase (~0.01 bpb).
- **Depth recurrence (Loop45 / Loop45×2)** — run layers 4–5 multiple times per
  forward pass; free test-time compute within the 16 MB budget.
- **Parallel residuals** — two-lane attention+MLP routing; stacks cleanly with
  recurrence.
- **SmearGate + BOS fix** — boundary handling at document starts; the BOS bug
  that went unnoticed for weeks and cost ~0.002 bpb when fixed.
- **LQER (Low-rank Quantization Error Reduction)** — corrects GPTQ rounding
  error with a low-rank residual trained after quantization.
- **MuonEq-R** — equivariant variant of the Muon optimizer; improves
  generalization under tight parameter constraints.

*End of section: describe the final stacked model architecture — what the
winning submission actually looks like.*

### 3. Quantization

- The steady thread from day 1 to the end; every record touches quantization.
- int6 → all-int6 → full Hessian GPTQ (Cholesky error compensation, 64-batch
  calibration).
- **AR self-generated calibration data** — use the model's own outputs as GPTQ
  calibration data instead of a fixed corpus; measurable improvement.
- **GPTQ embeddings** — quantize the embedding table too; previously left in
  fp16.
- **Artifact compression** — lrzip, brotli, zstd on the final 16 MB; the last
  few KB matter at the margin.
- QAT (quantization-aware training) as an alternative path.

### 4. TTT (Test-Time Training)

- The most contested technique in the competition.
- **The legality fight:** early LoRA TTT submissions were training on validation
  data in ways that violated the spirit of the rules. C3 and C4 emerged
  directly from this dispute (Issue #1017).
- **Score-first TTT** — the clean formulation: score each validation chunk
  *completely* before running any gradient step. Settled by ~April 9.
- **Phased TTT** — break the validation pass into phases; each phase scores
  then updates; multi-phase global SGD + per-doc LoRA. The final legal form.
- TTT contribution to the final model: empirically ~0.003–0.005 bpb.

---

## Part 2 — The Disqualification Zoo

Tone: semi-formal, wry.

### PPM-D: "Free Bits From Arithmetic That Doesn't Sum to 1"

- Background: PPM (Prediction by Partial Matching) is a classical byte-level
  compressor. Mix it with a neural model in probability space and you get a
  byte-level hybrid scorer.
- The cluster: ~8 PRs (including our own #1885) claiming 0.90–1.014 bpb via a
  NN + PPM-D mixture. Flagged in Issue #1872.
- The C2 violation: the uniform-spread construction (`p(t)^{1/n}` per byte)
  does not normalize. A toy example: two 2-byte tokens each with p=0.25 gives
  `p(first_byte) = 0.5 + 0.5 = 1.0`, not sharing space with any other byte.
- The punchline: under the correct conditional byte distribution, PPM is *not*
  better than the baseline — it's worse by ~0.038 bpb. The apparent 0.051 bpb
  gain is entirely from the broken scoring rule, not from PPM.

### Scylla: "0.9485 BPB (For Three Days)"

- PR #1184, merged April 23, claiming 0.9485 bpb — roughly 0.12 below the
  frontier at the time.
- Used a custom TokenMonster-derived "Scylla" vocabulary (998 tokens) and
  retokenized all of FineWeb with it, including the val split.
- Removed April 26 as "invalid record." Almost certainly: the custom val shard
  from retokenization didn't correspond to the official competition validation
  documents.

### GatedDeltaNet / FLA Byte-Bug Cluster

- Several PRs using the `flash-linear-attention` library double-counted the
  byte denominator in BPB accounting by ~17.46%, inflating scores by ~18%.
- Flagged in Issue #1719; affected PRs self-closed or were ruled out.

### N-gram Eval Cache

- The within/word boundary gate in several n-gram TTT implementations used
  `boundary_lut[tokens[i]]` — same bug as Issue #1420, partially fixed in
  #1514 by disabling. ~95% of gated mass was leaky (C1 violation).

---

## Part 3 — Last Day Chaos: The CaseOps Val-Set Leak

Tone: funny, forensic.

- **CaseOps** arrives April 18–19 and is genuinely good (+0.003 bpb). Everyone
  adopts it immediately.
- **The bug:** `prepare_caseops_data.py` has a `--val-docs=10000` default.
  Nobody overrides it. All 34 CaseOps-lineage PRs. Zero overrides.
- **What leaked:** training starts at canonical-stream document 10,000, but the
  validation set is documents 0–49,999. Documents 10,000–49,999 (80% of the
  val set) are in both train and val.
- **The timeline:**
  - Introduced: PR #1736 (dexhunter, April 19) — first to use
    `prepare_caseops_data.py` with the default, while reporting a separately-
    regenerated 50k-doc val set
  - Fixed: PR #1851 (aquariouseworkman, April 27) — switched to the official
    HF dataset (`romeerp/parameter-golf-caseops-v1`), disjoint by construction
  - Re-introduced same day: PR #1855 (codemath3000, April 27) — rebuilt
    locally with the default, propagating the leak forward
- **The numbers:** claimed frontier #2118 at **1.04350** vs clean frontier
  #1851/#1855 at **~1.061**. A 0.016 bpb gap from memorizing 80% of your exam.
- **Our own position:** our research baseline (#1736) was leaky. All internal
  spec measurements (specs 008–302) are internally consistent but live in the
  leaked world. Absolute numbers not comparable to the clean leaderboard.

---

## Part 4 — Final Leaderboard and Some Thoughts

- The cocohearts PR (#1902, April 29): OpenAI's official retroactive leaderboard
  update applying the BOS fix and clarifying which CaseOps records are clean.
- **Official clean SOTA:** PR #1855 at **1.0611** / PR #1851+#1868 at **1.0614**
- What the clean number tells us about headroom: we went from 1.2244 to 1.0611,
  a drop of 0.163 bpb in 6 weeks.
- Brief reflection on what worked (stacking is everything, early tokenizer bets
  paid off), what surprised (vocabulary size), and what didn't (PPM, pre-quant
  TTT, anything that required cheating to look good).
