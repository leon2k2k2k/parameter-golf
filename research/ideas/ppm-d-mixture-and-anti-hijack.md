# PPM-D byte mixture: scoring formula and anti-hijack gate

**Date:** 2026-04-28
**Related:** `research/specs/052-ppm-port-tuned.md`, `eval/2026-04-28_ppm-rigorous-full-val.md`, `research/ideas/ppm-port-on-047B.md`

This note writes down exactly how the PPM-D byte mixture scores each byte, the gate logic (1850's original + our anti-hijack mod), and why each piece is there. Future specs / read-throughs should be able to start here.

## 1. Per-token NN probability → per-byte NN probability

The NN predicts TOKENS, but PPM operates on BYTES. We bridge by spreading the per-token log-prob uniformly across the token's bytes (this is 1850's modeling choice; we adopt it verbatim).

For target token $T_t$ at position $t$:

1. **NN per-token cross-entropy** (in nats):
   $$\text{nll}_t = -\ln p_{NN}(T_t \mid T_{<t})$$

2. **Token decodes to $n_t$ bytes**: piece bytes from the SP tokenizer, plus an optional leading SPACE byte if `has_leading_space[T_t]` AND `prev` is not a boundary:
   $$\text{bytes}(T_t) = (b_1, b_2, \dots, b_{n_t})$$

3. **NN per-byte log-prob** (uniform spread):
   $$\ln p_{NN,\text{byte}}(b_j) \;=\; -\frac{\text{nll}_t}{n_t} \quad \forall j \in \{1,\ldots,n_t\}$$
   Equivalently, $p_{NN,\text{byte}} = p_{NN}^{1/n_t}$ — the geometric mean over the token's bytes.

This is what the C scorer reads as `nn_logp` (a negative number, closer to 0 = NN more confident on the byte).

### Why this isn't a true marginal

The NN doesn't actually predict bytes; it predicts the next token. Spreading log-prob uniformly is a heuristic that says "if NN gave this token 90% probability and the token has 4 bytes, treat each byte as if NN gave it $0.9^{1/4} \approx 0.974$." This is geometric-mean spreading, not true byte-marginalization. PPM corrects positions where bytes are predictable from byte context but the NN-per-byte spreading would underweight that predictability (see §6).

## 2. Per-byte PPM-D probability

Walk byte history $h$ (the previously scored byte stream, single context for the entire val) at contexts of decreasing length $K, K-1, \ldots, 0$.

### Howard escape-D allocation

At context length $k$ with current k-byte suffix $\text{ctx}_k$:
- $N_k(\text{ctx}_k, b)$ = count of (context, next byte) pairs.
- $N_k(\text{ctx}_k, *) = \sum_b N_k(\text{ctx}_k, b)$ = total observations.
- $U_k(\text{ctx}_k)$ = number of unique followers.

Probability allocated to seen byte $b$ at order $k$:
$$p_k(b \mid \text{ctx}_k) = \frac{N_k(\text{ctx}_k, b)}{N_k(\text{ctx}_k, *) + U_k(\text{ctx}_k)}$$

Escape probability (mass for unseen bytes, passed to order $k-1$):
$$\text{esc}_k = \frac{U_k(\text{ctx}_k)}{N_k(\text{ctx}_k, *) + U_k(\text{ctx}_k)}$$

If $b_j$ is found at order $k$, $\ln p_{PPM}(b_j) = \ln p_k(b_j) + \sum_{k'>k} \ln \text{esc}_{k'}$. If never found at any order, fall through to uniform $\ln(1/256)$.

In the C scorer, this is `ppm_log` — the cumulative log-prob in nats with all the escape multipliers folded in.

### PPM's confidence

$$c \;=\; \max_b\, p_{PPM}(b \mid \text{ctx}_K)$$

at the deepest context that has any data. This is the C scorer's `conf`. **Critical:** $c$ is PPM's max probability over the alphabet — NOT $p_{PPM}(b_j)$ specifically. PPM might be 95% sure on some OTHER byte, in which case $c=0.95$ but $p_{PPM}(b_j)$ could be tiny.

## 3. The gate

### 3a. Original 1850 rule

$$\text{hi}_{\text{raw}} \;=\; \mathbb{1}[c \geq \tau]$$

If PPM is confident in *something*, gate raw-fires. For 1850's $\tau = 0.9$: ~17% of bytes. For our tuned $\tau = 0.76$: ~27% of bytes.

### 3b. Anti-hijack (our addition)

Define "NN confident on the actual byte":
$$\text{nn\_confident\_on\_actual} \;=\; \mathbb{1}\big[\ln p_{NN,\text{byte}}(b_j) > -\tau_{NN}\big]$$

Equivalently: $p_{NN,\text{byte}}(b_j) > e^{-\tau_{NN}}$. For $\tau_{NN} = 0.277$ nats $= 0.40$ bits → NN gave the actual byte > 0.758 probability.

The effective gate:
$$\text{hi} \;=\; \text{hi}_{\text{raw}} \;\land\; \neg\,\text{nn\_confident\_on\_actual}$$

In words: gate fires only when PPM is confident AND NN wasn't already nailing the actual byte.

**The asymmetry is crucial:**
- $c$ (PPM's confidence) is NN-agnostic of the actual byte
- The NN-confidence check is on the ACTUAL byte $b_j$
- This is exactly the anti-hijack lever — we suppress only when NN was right *on the right byte*

After anti-hijack at our tuned settings ($\tau=0.76, \tau_{NN}=0.40$): effective gate-fire drops from 17.3% (raw at $\tau=0.9$) to 9.9% — about 42% of would-be fires get suppressed, all of them hijacks.

### 3c. Lambda selection

$$\lambda = \begin{cases}
\lambda_{\text{lo}} & \text{if hi}=1 \\
\lambda_{\text{hi}} & \text{if hi}=0
\end{cases}$$

$\lambda$ is **the weight on the NN side**. So `hi=1` (gate fires) means $\lambda$ is small → PPM dominates the mix.

1850's defaults: $\lambda_{\text{lo}} = 0.05$, $\lambda_{\text{hi}} = 0.9$.

## 4. The mixture itself

In probability space:
$$p_{\text{mix}}(b_j) = \lambda \cdot p_{NN,\text{byte}}(b_j) + (1-\lambda) \cdot p_{PPM}(b_j \mid h)$$

In log-space (numerically stable, log-sum-exp), which is what the C scorer computes:

```
a = log(λ)   + nn_logp      // = log(λ · p_NN)
c = log(1-λ) + ppm_log      // = log((1-λ) · p_PPM)
m = max(a, c)
log_mix = m + log( exp(a - m) + exp(c - m) )
```

Edge cases the code handles explicitly:
- $\lambda \leq 0$ → `log_mix = ppm_log` (pure PPM)
- $\lambda \geq 1$ → `log_mix = nn_logp` (pure NN)

Pre-cached at function entry: `lhi = log(λ_hi)`, `llo = log(λ_lo)`, `l1hi = log(1-λ_hi)`, `l1lo = log(1-λ_lo)`. Hot-path multiplexes on `hi`:
```
a = (hi ? llo : lhi) + nn_logp
c = (hi ? l1lo : l1hi) + ppm_log
```

## 5. Bits charged + score-before-update

For byte $b_j$:
$$\text{bits}(b_j) = \frac{-\text{log\_mix}}{\ln 2}$$

After scoring, update PPM tables (increment $N_k$ at all context lengths $1 \ldots K$ with $b_j$). This is the **score-before-update** legality discipline (Issue #1017 cond 3).

## 6. Why anti-hijack works — the 4.32-bit geometry

When `hi=1`:
- Mix weight on PPM is $1 - \lambda_{\text{lo}} = 0.95$.
- If PPM's top-1 disagrees with $b_j$, then $p_{PPM}(b_j)$ is tiny (PPM's mass is on a different byte).
- Mix becomes:
  $$p_{\text{mix}}(b_j) \approx \lambda_{\text{lo}} \cdot p_{NN}(b_j) + 0.95 \cdot \text{tiny} \approx \lambda_{\text{lo}} \cdot p_{NN}(b_j)$$
- If NN was right ($p_{NN}(b_j) \approx 1$), the mix is $\approx 0.05$.
- Cost: $-\log_2(0.05) = 4.32$ bits per byte.

This is the **canonical hijack signature**: every hijack costs exactly 4.32 bits, regardless of NN's exact confidence (as long as NN was very confident). On 047B post-quant pre-anti-hijack: ~376K hijack positions × 4.32 bits = ~1.6M bits of damage.

Anti-hijack catches all of them: NN-confident-on-actual + PPM-confident-on-something-else → suppress gate → mix mostly trusts NN → charge ≈ 0 bits.

### What anti-hijack does NOT catch

If `hi_raw=1` AND NN was UNCERTAIN on $b_j$ (e.g. $p_{NN}(b_j) = 0.1$, `nn_logp = -2.3`), and PPM is wrong → mix takes ~4.3 bits anyway. Anti-hijack only protects high-NN-confidence positions.

This is fine because: NN-uncertain positions are exactly where we WANT PPM to fire (in-doc rare-term recall, repetitive boilerplate). Even if PPM is occasionally wrong there, the average gain over those positions is still positive (D1-D5 of NN difficulty: +9-11% rescue rate per `eval/2026-04-28_ppm-rigorous-full-val.md`).

## 7. Tuned hyperparameters (from 2D sweep on cached post-quant 047B per-byte dump)

| Parameter | Value | Note |
|---|---:|---|
| $\tau$ (`ppm_conf_threshold`) | **0.76** | Lower than 1850's 0.9 → gate fires more often. Acceptable because anti-hijack catches the new hijacks. |
| $\tau_{NN}$ (`ppm_nn_skip_thr_nats`) | **0.277 nats** | = 0.40 bits → NN p_byte > 0.758 threshold for suppress. |
| $\lambda_{\text{hi}}$ | 0.9 | Unchanged from 1850. |
| $\lambda_{\text{lo}}$ | 0.05 | Unchanged from 1850. |
| $K$ (`ppm_order`) | 4 | Unchanged from 1850. |

The two knobs interact strongly:
- $\tau$ alone (lowered to 0.85, no anti-hijack): −0.002 BPB
- Anti-hijack alone ($\tau$=0.9, $\tau_{NN}$=0.20): −0.007 BPB
- **Both tuned**: **−0.026 BPB** (vs 1850 defaults)

The interaction means our anti-hijack discovery is most valuable in combination with a more aggressive PPM threshold, not as a standalone fix.

## 8. Aggregated val_bpb

$$\text{val\_bpb}_{\text{sidecar}} = \frac{1}{N_{\text{bytes}}^{\text{sidecar}}} \sum_t \sum_{j=1}^{n_t} -\log_2 p_{\text{mix}}(b_j^{(t)})$$

where $N_{\text{bytes}}^{\text{sidecar}}$ is the total byte count from the val_bytes sidecar (uint16 per token, summed). This is the **leaderboard-comparable** number.

The "piece-bytes" version uses the same numerator but $N_{\text{bytes}}^{\text{piece}} = \sum_t n_t$ (sum of piece-encoded byte counts). On CaseOps val, sidecar < piece (~151M vs ~165M bytes) because piece-encoding includes structural marker bytes (the `\xee\x80\x81` etc. CaseOps prefixes) that aren't in the source text.

## 9. Validation status

- Cached-NLL prediction (post-quant 047B, OMP-chunked): **mix_sidecar = 1.00505**
- End-to-end run (047B + patched train_gpt.py with PPM hook + EVAL_ONLY=1): **mix_sidecar = 1.00506** (matches cache to 0.0001)
- 1850's submission: 1.00495 (essentially tied — within seed noise)

So the mechanism + tuning + integration are validated. The remaining uncertainty is whether the 0.0001 BPB margin to 1850 holds across seeds, and whether moving to a stronger NN base (e.g. 052) would shift the gain delta.
