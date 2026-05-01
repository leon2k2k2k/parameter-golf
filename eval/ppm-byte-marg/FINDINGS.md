# Findings: PPM byte-mixing on spec 250 — corrected analysis

**Date:** 2026-05-01
**Base:** spec 250 seed_0 (#1953 + LeakyReLU² slope=0.3, val_bpb 1.0680 sidecar-normalized on full val).
**TL;DR:** The original "−0.097 BPB" finding was wrong. PPM byte-mixing **hurts** on this base under correct sidecar-normalized accounting, by **+0.044 BPB on full val** at the canonical PR-#1145 hyperparameters (α=15, β=0.80). No (α, β) combo yields a positive gain.

---

## What we got wrong before

The Python implementation (`proper_ppm_mixer_rigorous.py`) and the original C port had two compounding bugs that made the mixer look like it was helping by ~−0.097 BPB:

### Bug 1 — Wrong denominator
Mixer reports `bpb = total_mix_nll / total_canonical_bytes / log(2)`.

`total_canonical_bytes` is computed from tokenizer pieces + leading-space prepend rule. On spec 250 / SP8192 CaseOps tokenizer:

- Sidecar file bytes (leaderboard convention): 148.4M (full val)
- Canonical with LS prepend: 162.9M (full val)
- Ratio: **1.10× over-count**

Dividing NLL by canonical (which over-counts bytes) makes BPB look ~10% smaller than the leaderboard sidecar normalization.

### Bug 2 — Single-byte chain-rule term missing
For single-byte byte-content tokens (e.g. ".", ",", " a", "1") the code skipped the chain-rule remainder NLL:

```c
if (toklen > 1) { compute nll_rem; }
total_nll += nll_b0 + nll_rem;  // nll_rem = 0 for single-byte
```

But chain rule requires:
```
P_NN(token) = P_NN(b_0) · P_NN(rem | b_0)
```

Even for single-byte tokens, the conditional `P_NN(rem = empty | b_0)` ≠ 1 because many tokens share the same b_0. The "rem" term for single-byte tokens is `−log(P_NN(token) / P_NN(b_0))` — non-zero and non-trivial.

On 1M tokens, this missing term accounts for ~80,000 nats (~0.04 BPB sidecar-normalized).

### Combined effect

| metric | reported (buggy) | actual (sidecar-normalized) |
|---|---|---|
| Mix BPB on 1M | 0.976 | **1.112** |
| Mix BPB on 4.7M | 0.977 | ~1.110 |
| Mix BPB on full 47M | 0.981 | **~1.112** |
| NN baseline | 1.068-1.073 | 1.068-1.073 |

So instead of −0.087 BPB **gain** on full val, mix is +0.044 BPB **loss**.

---

## Verification of the fix

After fixing the C mixer to (a) handle single-byte chain rule, (b) report sidecar BPB:

```
Mix check: mix_b0 + mix_rem + ctl = 2,395,055.6 nats
  vs total_mix_nll                = 2,395,055.8 nats
  diff = 0.1  (fp32 noise)        ✓
NN check: nn_b0 + nn_rem + ctl    = nn_total                ✓
```

Now totals align with chain-rule decomposition exactly.

Sample 1M (sidecar-normalized):
- NN-only: 1.072962
- PPM-mix (α=15, β=0.80, LOOSE): 1.112251
- Δ = **+0.039290** (mixer hurts)

**Full val (47M tokens, sidecar-normalized):**

```
                                  byte_0 BPB    rem BPB    ctl BPB   total BPB    Δ vs NN
NN-only                             0.245125   0.821561   0.001297    1.067984   +0.000000
PPM-only (w/ NN control)            0.657781   1.465599   0.001297    2.124678   +1.056694
Actual mix (α=15, β=0.80, LOOSE)    0.267064   0.843362   0.001297    1.111723   +0.043739
PROJ: λ_b0=1 (drop byte_0 mix)      0.245125   0.843362   0.001297    1.089785   +0.021801
PROJ: byte_0 oracle + mix rem       0.220373   0.843362   0.001297    1.065033   −0.002951
PROJ: full oracle (b0 + rem)        0.220373   0.761390   0.001297    0.983061   −0.084923
```

- **Mixer is +0.044 BPB worse than NN-only** at canonical PR-#1145 hyperparameters.
- byte_0 oracle (per-position min of NN/PPM): saves only −0.025 BPB if achievable.
- Full oracle (b0 + rem): saves −0.085 BPB. This is the absolute upper bound.
- The actual mix gate captures **−88.6% of byte_0 oracle headroom** (i.e., moves 88% backward) and **−36.2% of remainder oracle headroom**.

Conclusion: even with a perfect gate, byte-level PPM mixing can save at most −0.085 BPB on this base. With any realistic gate (PPM confidence-based), the mix cannot recover its own losses from confident-but-wrong PPM predictions.

---

## Why the mixer hurts (mechanistic)

The byte_0 oracle analysis showed PPM at byte_0 is much worse than NN on average (PPM 2.54 bits/byte_0 vs NN 0.78 bits/byte_0). The gate (lambda based on PPM confidence) puts weight on PPM exactly when PPM is *confident*, but PPM's confidence is a poor signal for accuracy at byte_0 — confident on common bigrams (like " " after content), often wrong because content tokens follow.

At the remainder level, our earlier analysis (using buggy normalization) suggested the remainder mix was helping. Re-checking with the fix: the remainder mix is also worse. The chain-rule "rem" term for single-byte tokens we were ignoring is large because P(byte_0) >> P(specific 1-byte token), so the chain-rule decomposition surfaces a lot of NLL that the broken accounting was hiding.

---

## Hyperparameter sweep

Sweep on 1M tokens (sidecar-normalized Δ vs NN, lower is better):

```
          β=0.50   β=0.70   β=0.80   β=0.85   β=0.90   β=0.95   β=0.99 
α=   5:   +0.2356  +0.1157  +0.0764  +0.0612  +0.0486  +0.0381  +0.0312
α=  15:   +0.3142  +0.0889  +0.0393  +0.0243  +0.0141  +0.0075  +0.0042
α=  30:   +0.4087  +0.1017  +0.0409  +0.0231  +0.0109  +0.0038  +0.0012
```

Zoom on best region:

```
          β=0.95   β=0.97   β=0.98   β=0.99   β=0.995  β=0.999
α=  30:   +0.0038  +0.0022  +0.0017  +0.0012  +0.0010  +0.0009
α=  50:   +0.0030  +0.0013  +0.0008  +0.0005  +0.0003  +0.0002
α= 100:   +0.0030  +0.0010  +0.0004  +0.0001  +0.0001  +0.0000
α= 200:   +0.0042  +0.0012  +0.0004  +0.0001  +0.0000  +0.0000
```

**Every (α, β) combo hurts.** The optimum is the trivial limit α→∞, β→1, which makes the gate so conservative that PPM is never used — the mix degenerates into NN-only. The actual headline:

**There is no positive Δ available from PPM byte-mixing on this base.**

---

## Implications

1. **PR #2039's claimed −0.04 BPB gain from "conditional PPM mixture" is suspect.** Its construction has a C1/C2 leak and likely also has a normalization issue. A clean implementation of the same idea (this repo) yields negative gain on the same compute regime.

2. **PR #1145's PPM-D byte mixer category** likely doesn't help on a strong base like #1853-derived stack. The original demonstration was on a weaker base where PPM had headroom; on spec 250 NN is strong enough that PPM dilutes rather than augments.

3. **The normalization bug was widespread.** Anyone reading our earlier README will see "0.976 BPB" for mix and assume that's leaderboard-comparable. It isn't. The bug is in this repo's `proper_ppm_mixer_rigorous.py`. We need to either fix it or add a prominent warning.

4. **Oracle analysis is also affected** — earlier byte_0 oracle showed −0.023 BPB available; that was with the same buggy normalization. With the fix, the available oracle gain at byte_0 alone might still be positive (PPM is sometimes right when NN is wrong) but the remainder mix is fundamentally bad.

---

## What needs to be redone

1. ✅ C mixer: fixed (chain-rule for single-byte; sidecar BPB reported separately)
2. ❌ Python mixer (`proper_ppm_mixer_rigorous.py`): same single-byte bug, needs fix
3. ❌ README: claims `0.976` mix BPB; needs correction or removal of those numbers
4. ❌ Oracle analysis on full val: rerun with fixed accounting to get TRUE oracle headroom
5. ❌ Memory items: `project_ppm_analysis_047B.md`, `project_ppmd_leading_space_lever.md` need updates flagging the prior numbers were buggy

---

## Files

All paths relative to `eval/ppm-byte-marg/`.

### Top-level (original, contains the buggy mixer for diff reference)
- **`README.md`** — original headline (now contains incorrect "−0.097 BPB" claim, needs update)
- **`proper_ppm_mixer_rigorous.py`** — Python mixer with the single-byte chain-rule bug + canonical-bytes denominator. **Do not use for new BPB claims.** Kept for diff against the C port.
- **`byte_bpb_proper.py`** — NN-only byte scoring baseline (also has the same denominator subtlety; not used in this analysis)
- **`show_big_gains.py`** — token-level gain inspection (uses old buggy mixer)
- **`test_byte0_3way.py`** — early 5-config byte_0 distribution comparison (predates these bugs being noticed)

### `c_port/` — Fixed C implementation
- **`ppm_byte_mixer.c`** — **fixed** C mixer. Handles single-byte chain-rule. Outputs both canonical and sidecar-normalized BPB via wrapper.
- **`ppm_mixer_c_wrapper.py`** — Python ctypes wrapper. Returns full decomposition (mix/NN/PPM/oracle at byte_0 and rem) plus raw `total_nll_nats` for sidecar normalization.
- **`build.sh`** — gcc -O3 build script + self-test
- **`run_c_only.py`** — driver: forward + C mixer, prints sidecar BPB
- **`run_c_oracle_fullval.py`** — driver: forward + C mixer + decomposition table including projections (drop byte_0 mix, byte_0 oracle, full oracle)
- **`verify_c_vs_python.py`** — bit-exact comparison: C output vs Python on small slice (validates port correctness modulo the shared single-byte bug)

### `analysis/` — Diagnostic + sweep scripts (Python, slow but transparent)
- **`diag_canonical_vs_sidecar.py`** — counts canonical-with-LS / canonical-pieces-only / sidecar bytes for any slice; surfaces the denominator bug
- **`diag_mixer_totals.py`** — verifies decomposition consistency: NN check, mix check (should sum exactly to total NLL after the C fix)
- **`sweep_alpha_beta.py`** — α/β grid sweep; reports sidecar-normalized BPB per combo
- **`oracle_byte0.py`** — Python byte_0 oracle (NN/PPM/mix/oracle NLL accumulators); slow but useful for sanity-checking C implementation
- **`verify_distributions_sum_to_one.py`** — verifies P_NN(byte_0 = b | history, prev) sums to 1 over 256 bytes after renormalization, for both case A (prev_is_boundary) and case B (prev_not_boundary)
- **`verify_remainder.py`** — verifies remainder probabilities are non-negative and ≤ 1 (chain-rule shortcut sanity)
- **`verify_mix_dist.py`** — dumps top-N positions with full NN/PPM/mix top-5 distributions; used for "where does PPM help" inspection (but built on the buggy mixer; gain numbers are inflated but the *positions* are still meaningful)
- **`verify_mix_distrib.py`** — per-decile savings breakdown (head vs middle vs tail of stream)
- **`top50_categorize.py`** — top-N high-savings positions with byte context for manual categorization (verbatim repeat / tokenization-gap / UTF-8 / etc.)

### What's pinned to the pod
On `psev0odnsqdetw` (216.243.220.226:11058) all scripts live in `/tmp/ngram_analysis/` (Python) and `/tmp/ngram_analysis/c_port/` (C). The pod is ephemeral; the canonical copies are in this repo.

### How to reproduce a single number

```bash
# On the inspect pod, in /tmp/ngram_analysis/c_port:
bash build.sh

export SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
       LQER_ENABLED=1 LQER_ASYM_ENABLED=1 ASYM_LOGIT_RESCALE=1 CASEOPS_ENABLED=1

python3 run_c_oracle_fullval.py \
  --ckpt /workspace/runs/250-1953-leakyrelu03/seed_0/final_model.pt \
  --train_gpt /tmp/train_gpt_spec250.py \
  --tokenizer /workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  --val_tokens_bin /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin \
  --val_bytes_bin  /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin \
  --n_tokens 47000000
```

~3 min total: 105s forward + 53s C mixer + small Python overhead. Expect:
```
NN-only:                       1.067984
PPM-mix LOOSE (sidecar):       1.111723   Δ=+0.043739   (mix HURTS)
```
