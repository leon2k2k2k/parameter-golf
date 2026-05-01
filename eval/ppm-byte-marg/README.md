# Proper byte-level marginalization for NN + PPM-D mixing

Self-contained, rigorous, **C1-clean** implementation of byte-level NN scoring and
optional byte-level PPM-D mixing — without the include_space C2/C1 leak that
PR #2039 / PR #1967 / PR #2018 / PR #2041 ship.

**Empirical result on spec 250 seed_0** (1M val tokens, 1H H100):

| | val_bpb | Δ vs NN-only |
|---|---|---|
| NN-only (token-level) | 1.072962 | — |
| PPM-mix LOOSE (no renorm) | 0.975982 | −0.097 |
| **PPM-mix RIGOROUS (with renorm)** | **0.976086** | **−0.097** |

Rigorous − Loose difference: **+0.000104 BPB** (negligible).

PPM provides ~−0.097 BPB legitimate gain on this base from byte-level
autoregressive scoring of repeated patterns. None of it depends on the
include_space leak that #2039 et al. exploit.

---

## Files

- **`proper_ppm_mixer_rigorous.py`** — canonical reference implementation.
  Forward pass + per-position byte-level marginalization (3 matmuls) +
  PPM-D byte mixer with both LOOSE and RIGOROUS modes for comparison.
- **`byte_bpb_proper.py`** — NN-only byte-level BPB scoring with proper
  marginalization (no PPM mixing). Useful as a clean baseline.
- **`show_big_gains.py`** — finds tokens where PPM-mix gives largest savings
  vs NN-alone, dumps byte-by-byte breakdown. Use to inspect WHERE the gain
  comes from.
- **`test_byte0_3way.py`** — compares 5 different byte_0 distribution
  formulations (flat, #2039-hybrid, no-if, LS-marginal, truly-proper) at
  byte_0 only. Used to identify the C1/C2 leak in #2039.

## Quick run (on a 1H H100 pod with the volume)

```bash
# Required env: gate flags so all model weights load
export SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_ASYM_ENABLED=1 ASYM_LOGIT_RESCALE=1 CASEOPS_ENABLED=1

python3 proper_ppm_mixer_rigorous.py \
  --ckpt /workspace/runs/250-1953-leakyrelu03/seed_0/final_model.pt \
  --train_gpt /path/to/spec250/train_gpt.py \
  --tokenizer /workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  --val_tokens_bin /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin \
  --val_bytes_bin /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin \
  --n_tokens 1000000
```

Time: ~16s for 1M tokens (forward 2.6s, PPM mixer 13s). Full val (~47M
tokens) projects to ~12 minutes.

---

## The math (rigorous version)

### Three byte_0 distributions per position

Built from the model's softmax via three matmuls against precomputed masks:

```python
byte_dist        = probs @ first_byte_mask        # (B*T, 256) — Σ_T P(T) · 𝟙[fb(T)=b]
non_ls_byte_dist = probs @ non_ls_first_byte_mask # (B*T, 256) — same but excludes leading-space tokens
p_leading_space  = probs @ leading_space_mask     # (B*T,) — Σ_{T:leading_space(T)} P(T)
```

Where:
- `first_byte_mask[T, b] = 1 iff first_byte_id(T) = b` (after stripping `▁`)
- `non_ls_first_byte_mask[T, b] = 1 iff ¬leading_space(T) AND first_byte_id(T) = b`
- `leading_space_mask[T] = 1 iff piece(T) starts with ▁`

### Case-split selection (C1-clean)

For each position with prior token `prev` and realized target `y`:

```
P_NN(byte_0 = b | history, prev) =
  if prev_is_boundary:                       # ~1% of positions
      byte_dist[b]
  else (prev_not_boundary):
      if b = SPACE:
          leading_space_sum
      else:
          non_ls_byte_dist[b]
```

The if-branches condition only on `prev` (a prefix property — fine for C1).
The realized target `y` is used only to look up which slot of this fixed
distribution corresponds to the realized byte_0.

This contrasts with PR #2039's construction, which selects between two
**different distribution formulas** based on `has_leading_space[tokens[i]]`
(a property of the realized target). That's the C1 leak.

### Renormalization (rigorous mode)

`Σ_b P_NN(byte_0 = b | history, prev)` summed over the 256-byte alphabet
gives `sum_check ≈ 0.999`. The missing ~0.001 is mass on control/operator
tokens (BOS, CaseOps reserved tokens, etc.) that have no byte representation.

Rigorous renormalization makes the byte_0 distribution proper over the byte
alphabet:

```
P_proper(byte_0 = b | history, prev, target_has_bytes) = P_NN(byte_0 = b) / sum_check
NLL_b0_rigorous = −log(P_proper) = NLL_b0_loose + log(sum_check)
```

The conditioning correction `−log(sum_check)` (a small positive number,
~0.001 nats) is charged once per byte-content token, representing the cost
of "the model committed to a byte-content token at this position rather
than a control."

### Chain rule for remainder

```
P_NN(remainder | byte_0, history) = P_NN(token | history) / P_NN(byte_0 | history)
```

This holds for both loose and rigorous modes (the renormalization factor
cancels). For multi-byte tokens, the remainder is the (k−1)-byte joint
distribution and is mixed at the joint alphabet level with PPM-D's byte
chain.

### PPM-D byte mixer (vendored from PR #1145)

At each byte position, mix gate based on PPM context confidence:

```
λ_t = 1 − sigmoid(α · (PPM_conf_t − β))         α=15, β=0.80
P_mix(byte_t) = λ_t · P_NN(byte_t) + (1−λ_t) · P_PPM(byte_t | byte_history)
```

PPM state advances after scoring each byte (score-before-update).

### Total NLL accounting

```
For each scored position:
  if target has no bytes (control token):
    total_NLL += full token NLL
    (no byte contribution)
  else:
    total_NLL += NLL_b0_mix + NLL_remainder_mix + renorm_correction
    (renorm_correction = −log(sum_check), charged once)

BPB = total_NLL / total_bytes_via_sidecar / log(2)
```

Total bytes uses the `fineweb_val_bytes_000000.bin` sidecar (true byte
counts, not piece-byte counts — handles CaseOps operators correctly).

---

## Comparison to PR #2039

| Aspect | PR #2039 (as-shipped) | This implementation |
|---|---|---|
| C1 (causality) | ✗ — `if has_leading_space[tokens[i]]` selects formula based on realized target | ✓ — selection conditions on `prev_is_boundary` only |
| C2 (normalization) | ✗ — flat `exp(−nll/n_bytes)` for include_space (~14% of byte mass) is not a proper distribution | ✓ — proper marginal `Σ_{T:LS} P(T)`, renormalized to byte alphabet |
| C3 (score-before-update) | ✓ | ✓ |
| C4 (single-pass) | ✓ | ✓ |
| Headline gain inflation from leak | ~0.047 BPB on flat-vs-proper at byte_0 | 0 (no leak) |
| Source code visibility | lzma+base85 wrapped in train_gpt.py (hard to audit) | plain Python in this repo |

## Empirical leak validation

See `test_byte0_3way.py` output:

```
Config                                     byte_0 NLL (bits/tok)   Δ vs flat
A flat everywhere                          2.800                   —
B #2039 hybrid (with if)                   2.872                   +0.071
C no-if (byte_dist for all)                4.012                   +1.212
D LS-marginal (not C1-clean)               2.051                   −0.750
E TRULY PROPER (C1-clean)                  2.389                   −0.411
```

The truly-proper construction (E) gives **−0.41 bits/tok better byte_0
prediction** than #2039's hybrid (B). The chain-rule conservation means
this advantage gets reallocated to remainder bytes; in the post-PPM-mix
output, the net effect on total BPB is what's reported above.

---

## Findings: where the −0.097 BPB gain actually comes from

Inspecting top-gain tokens via `show_big_gains.py`:

1. **Verbatim phrase repetition** (~40% of gain): val contains repeating
   proper nouns, brand names, technical terms. PPM memorizes the byte
   sequence after the first occurrence, every subsequent is ~free.
   Examples: "jockey club" appears 5+ times in 1M tokens; "razumih"
   appears 3+ times.
2. **Common byte-bigram statistics within val** (~30%): "at", "ic", "ity",
   "ff", "tion" — common letter-pair transitions PPM learns from the
   byte stream.
3. **CaseOps operator UTF-8 multi-byte expansion** (~20%): `` is
   3 bytes; bytes 1-2 are deterministic given byte 0. PPM gets bytes
   1-2 at near-zero NLL.
4. **Common word-suffix patterns** (~10%): "less", "ate", "ing", "tion".

This is byte-level entropy coding of val-internal redundancy — legitimate
under C1-C4 but conceptually distinct from "language modeling" gains.

---

## Compliance status

The implementation satisfies all of Issue #1017's C1-C4 conditions
explicitly:

- **C1 (causal):** byte_0 distribution is a function of `(history, prev)`
  alone. PPM state advances after scoring each byte.
- **C2 (normalized):** byte_0 distribution sums to 1 over the byte alphabet
  (verified empirically; sum_check renormalization handles the small
  control-token leak).
- **C3 (score-before-update):** explicit in code; PPM state updates after
  each byte's mix log-prob is recorded.
- **C4 (single L→R pass):** each val byte scored exactly once.

Whether byte-level PPM mixing as a *category* is admissible is the open
PR #1872 ruling. This implementation gives the cleanest possible version
of that question — if cocohearts/valerio rule the category in, this is
ready to ship; if they rule it out, the byte-level scoring framework here
is still valid for any other byte-level technique (e.g., a small byte LM
mixed in at byte_0).
