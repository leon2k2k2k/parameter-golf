# Parameter Golf: Six Weeks to Build the Best LLM

---

> **What does a 14% compression improvement actually look like?**
> Both models below were trained on the same dataset, in the same 10 minutes,
> on the same hardware. The only difference is what the community built in six weeks.

*From the FineWeb validation set, same 150-token context fed to both models:*

| | SP1024 Baseline · 1.22 BPB | Near-SOTA Model · 1.06 BPB |
|---|---|---|
| *"…his driver license and auto insurance had been cancelled in October. This poses a problem for"* | the estate, though internalieness to gets the pass. It is a fat check that's gives ges auto insurance company… that's a **fairyyyyyyyyyyyyyyyyyyyyy** | the estate's creditors who need to work out a payment, especially with the new insurance regulations in october… |
| *"…New host Steve Byrnes was feeling his way through this"* | season. He had just had a short, **bad weekkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk** | season's show, and while he played one of the best nights of his career at the same time, he hadn't been able to really shine since the first Nashville game… |
| *"…Prosecutors now have until"* | Jan. 2, the **Supreen House** prosecutor sentences on a **murded** case… **Mazzzzaglia**… **charies's''''''** | Dec. 18 to sign off on a case after the new hampshire attorney general's office says they have more time… |

*Prompted with "You are a helpful AI assistant. Please answer the following question." (not very good at general conversation):*

| | SP1024 Baseline · 1.22 BPB | Near-SOTA Model · 1.06 BPB |
|---|---|---|
| *"The history of the internet began in the 1960s when"* | the Netherlands had established a strong supply of information about the **Canadian government and its supply chains**. The German government, in its own right, is commonly established as a matter of legal… | the internet became the **first electronic commerce tool** that was used by many people, including children, to connect online. the internet is now used by more than 3 billion users worldwide… |
| *"What is 15 multiplied by 13? Let me work it out:"* | **1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 14. 13.** 14. 15. 16. 17. 18. 19. **19.** 20.1 | 15 multiples 13 have 15, for example, 15, for **1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000** |
| *"To bake a chocolate cake, you need flour, eggs, butter, and"* | sugar… you need to have **an infinite wooden cake**. Otherwise, you need to be taken as close to an… | salt… Add 1 tablespoon of flour to 1/4 cup of milk and pour over your cake. Add 1/4 teaspoon of sugar… Cook a lightly toasted chocolate cake. Make cupcakes Ã 3 |

---

In March 2026, OpenAI released a public competition with a deceptively simple premise: train the best language model you can, but it has to fit in 16,000,000 bytes, and you only get 10 minutes of training time on 8 H100s. Call it parameter golf — every byte counts, every second counts.

What followed over the next six weeks looked, from the outside, like just another competitive coding challenge. It turned out to be something more: techniques stacking on each other in ways nobody planned, innovations that shouldn't have worked but did, controversial submissions that looked like miracles, mayhem on the last day, and a picture-perfect finish. In the end, starting from a model that produces the gibberish above, the community built one that speaks coherently — just don't ask it anything that isn't in the training data. This post goes through some of the highlights — both the technical and the dramatic. 

---

## 1. The Competition

At the core of this competition is a simple question: how well can a model predict text?

A language model is, at its heart, a probability distribution over text. Given a sequence of words — or more precisely, tokens — the model assigns a probability to every possible next token. A well-trained model should assign high probability to tokens that actually appear in real text, and low probability to tokens that don't. Think of it like a well-read person trying to complete sentences. Given "The president signed the —", they would confidently predict "bill" or "order" and be surprised by "banana." A bad model treats every next word as equally likely. A good model has internalized the patterns of language well enough to be right, or at least close, most of the time.

The models in this competition are trained and scored on FineWeb, a large dataset of cleaned web text. The score is computed on a held-out validation slice that the models never see during training. For each token in that slice, we ask: what probability did the model assign to the token that actually appeared? The cost for a single token is $-\log_2 p(t)$, where $t$ is the correct token. If the model is perfectly confident — $p(t) = 1$ — it pays zero cost. If it assigns $p(t) = 0.5$, it pays 1 bit. If it assigns $p(t) = 0.01$, it pays about 6.6 bits. The total score is this cost summed across all tokens, normalized by the number of bytes in the original text:

$$\text{BPB} = \frac{-\sum_k \log_2 p(t_k \mid t_1, \ldots, t_{k-1})}{\text{number of bytes}}$$

This is called bits-per-byte (BPB). To put it in perspective: a model that assigns completely uniform probability across all 256 possible bytes — knowing nothing at all — scores exactly 8 BPB. The baseline OpenAI provided started at **1.2244 BPB**, already far below that, meaning the model had learned real structure in language. Six weeks later, the community had pushed it to **1.0565** — a 14% reduction, achieved purely through algorithmic improvements with no change to the hardware or the data.

---

## 2. Model Evolution

A modern language model is built from a stack of identical blocks. Each block has two components — attention first, then MLP:

```
# One transformer block
def block(x):
    x = x + attention(x)    # tokens communicate: each looks at all others
    x = x + mlp(x)          # tokens think: each processes what it gathered
    return x

# Attention: tokens communicate
def attention(x):
    Q = x @ W_Q    # what am I looking for?
    K = x @ W_K    # what do I contain?
    V = x @ W_V    # what do I share?
    weights = softmax(Q @ K.T / sqrt(d))
    return weights @ V

# MLP: tokens think
def mlp(x):
    return W_2 @ activation(W_1 @ x)    # expand, activate, contract
```

The full model is just these blocks chained one after another:

```
x → block_1 → block_2 → block_3 → ... → block_N → output
```

Attention captures how words interact with each other — which tokens are relevant to which. The MLP then enriches the meaning of each token individually, using what attention gathered as context. Each block refines the representation a little further. Stack 9 to 11 of them and you have a language model. We will come back to this picture when we discuss depth recurrence.

---

### The Baseline

OpenAI's starting model was already not a plain stack of blocks. Let's break it down across five components.

**Tokenizer.** The baseline used SentencePiece with a 1024-token vocabulary (SP1024). A 1024-token vocabulary is small by modern standards — GPT-2 uses 50,000 — but keeps the embedding table compact, which matters when the entire model has to fit in 16,000,000 bytes.

**Model architecture.** A 9-layer, 512-dimensional transformer with U-Net skip connections (each decoder layer receives a residual from its mirror encoder layer), grouped-query attention (GQA) to reduce parameter count, and rotary positional embeddings (RoPE).

**Training.** The Muon optimizer with a linear warmup-then-warmdown learning rate schedule.

**Quantization.** After training at bfloat16, MLP weights were rounded to int6 (6 bits per weight), shrinking what would otherwise be a ~70 MB model into the 16 MB budget. Everything else was left at higher precision.

**Post-training adaptation.** None. The 10-minute eval window was used only for scoring.

This baseline scored **1.2244 BPB**. Here is what changed.

---

### The Evolution

#### Tokenizer: CaseOps

The vocabulary grew in two steps: SP1024 → SP4096 (PR #1218) → SP8192 (PR #1394). Counterintuitive, since a larger vocabulary means a larger embedding table. The payoff comes from the BPB denominator: it counts *bytes*, not tokens. A token that encodes two bytes contributes two bytes to the denominator, so a model that packs more bytes per token earns lower BPB for the same prediction quality.

An SP8192 vocabulary still wastes slots on case variants: "the", "The", and "THE" are three separate entries. PR #1578 introduced casefold — lowercase everything before tokenizing — but this permanently destroys casing information the model needs to score correctly. Ruled illegal in Issue #1604.

CaseOps (PR #1729) solved this losslessly. Four control tokens are reserved (TITLE, ALLCAPS, CAPNEXT, ESC) and capitalization is encoded inline:

```
"The NASA launched."  →  "TITLE the ALLCAPS nasa launched."
```

The original text is fully recoverable. The ~8188 remaining vocabulary slots are now entirely free of case duplication, and the control tokens are cheap to predict — capitalization follows clear patterns — so the model pays very little BPB on them. SP8192+CaseOps remained the tokenizer frontier for the rest of the competition.

---

#### Model Architecture: Depth Recurrence

In a standard transformer, each layer runs exactly once per token. The final model loops layers 3–5 three times per forward pass — the same three layers, the same weights, applied three times in sequence:

```
standard:  1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

with loop: 1 → 2 → 3 → 4 → 5 → 3 → 4 → 5 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
                     └──────────────── ×3 ────────────────┘
```

17 effective processing steps from 11 physical layers, at zero additional parameter cost. This is free test-time compute: the model gets to think harder without growing larger. PR #1344 introduced the structure; the final configuration of 3 passes over layers 3–5 was established in later records.

| | Baseline | Final |
|---|---|---|
| Physical layers | 9 | 11 |
| Effective processing steps | 9 | 17 |
| Parameters | ~16 MB | ~16 MB |

Other architectural changes are summarized in the table at the end of this section.

---

#### Training: EMA and Loop Curriculum

**EMA (PR #287).** Instead of evaluating the final checkpoint directly, the model maintains an exponential moving average of all past weight iterates throughout training. The eval model is this running average, not the last step. SGD iterates are noisy; the average sits in a flatter, more stable region of the loss landscape and generalizes significantly better. The decay value introduced in PR #287 was never revisited through the final SOTA.

**Loop curriculum (PR #1420).** The recurrence loop does not activate from the start of training. For the first 35% of wallclock (~3.5 minutes), the model trains as a standard 11-layer network. The loop then switches on for the remainder. The reason is throughput: 17 effective layers is significantly slower per step than 11. By delaying the loop, the model gets more gradient steps within the 10-minute budget before paying the cost.

The final training run is a choreographed 10 minutes: fast 11-layer passes early, the loop switching on at the 35% mark, context growing from 1024 to 2048 to 3072 tokens as the clock runs down (PR #2014).

---

#### Quantization: GPTQ and LQER

**GPTQ (PR #535).** When you round a weight, you introduce an error. Instead of ignoring that error, GPTQ compensates for it by adjusting the remaining unquantized weights in the same layer, using second-order information about how sensitive the output is to each weight. The result is a quantized model that stays much closer to the original's predictions than naive rounding. This evolved to cover all weights including attention (PR #1285) and embeddings at int7 (PR #1586).

**LQER (PR #1797).** After GPTQ, some quantization error remains. LQER stores a correction: compute the residual between the original and quantized weights, take a rank-4 low-rank approximation, and pack those correction factors into the artifact. The model reconstructs a better approximation at inference time. The correction costs ~30 KB of artifact space and recovers a meaningful fraction of the remaining quantization damage.

---

#### Post-Training: TTT

The 10-minute eval window is not just for scoring. The final model uses most of it to actively adapt its weights to the validation text it is about to score — test-time training (TTT).

**Per-document LoRA (PR #1530).** The base model weights are frozen. For each validation document, a set of low-rank adapter matrices are attached to the model's projections. The document is processed in chunks: score each chunk first, then take a gradient step to update the adapters. By the final chunk, the model has already adapted to that document's style and vocabulary. The adapters reset after each document; nothing carries over.

**Global SGD phase (PR #1610/#1626).** After an initial batch of documents has been scored, a full SGD pass runs on the base model weights themselves — not just the adapters — using all the already-scored documents as training data. The base model is updated, the adapters reset, and the remaining documents are scored on top of this improved base.

The LoRA handles fast local adaptation per document; the global SGD step shifts the base model toward the distribution of the validation set as a whole. The eval budget splits roughly as ~120 seconds for the baseline scoring pass and ~480 seconds for the TTT loop.

---

### Other Changes

| Component | Change | PR |
|---|---|---|
| Architecture | XSA: removes self-copy bias from attention outputs | #287 |
| Architecture | Parallel residuals from layer 8: `h = x + Attn(x) + MLP(x)` | #1204, #1529 |
| Architecture | SmearGate: learned blend of each token with its neighbor | #162, #1667 |
| Architecture | LeakyReLU² replacing relu² in MLP | #493 |
| Architecture | Partial RoPE + layer-norm scaling | #315 |
| Quantization | AWQ-lite: sensitive weight columns promoted to int8 | #1908 |
| Quantization | Calib32: doubled calibration batches for better Hessian | #2135 |
| Quantization | Artifact compression: lrzip+ZPAQ+L1 row reordering | #1855 |
| TTT | Progressive context 1024→2048→3072 during TTT chunks | #2014 |

Many other eval-time methods were attempted. Several were ruled illegal. We examine two of them next.

---

## 3. Too Good to Be True

*(draft pending)*

---

## 4. Drama on the Last Day

*(draft pending)*
