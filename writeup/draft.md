# Parameter Golf: Six Weeks of Compressing Language Models to Their Limits

---

## Part 0 — The Competition

In March 2026, OpenAI announced Parameter Golf: a community competition to build
the best language model that fits inside 16,000,000 bytes, trained in exactly
10 minutes on 8 H100 GPUs. The competition ran for six weeks, attracted over
2000 PR submissions, and ended with the community pushing a baseline model from
1.22 bits-per-byte down to 1.056 — a compression improvement of roughly 14%
through pure algorithmic ingenuity, with no change to the hardware budget or
the data.

Parameter Golf sits within a family of challenges sometimes called L(N)
optimization. Sister challenges include the NanoGPT Speedrun (fixed time,
minimize loss) and the NanoGPT Flowrun (fixed data, minimize loss). Here
the constraint is different: fixed size, with both training time and model
capacity tightly capped.

Before we look at the models, let us first understand how the competition works.

**The setup.** The dataset is a slice of FineWeb, a large corpus of filtered
web text. Competitors train on a fixed training split and are scored on a
held-out validation set they cannot access during training. The training window
is 10 minutes of wall-clock time on 8 H100 GPUs — Triton kernel compilation
happens beforehand (competitors pre-warm their kernel caches as a separate
step and it does not count against the clock), and GPTQ quantization runs after
the clock stops as a post-processing step. The final artifact — which must fit
inside 16,000,000 bytes — is the compressed model weights plus the
`train_gpt.py` code that runs them.¹ Evaluation gets a separate 10-minute
window; test-time training (TTT) happens within that window, consuming some of
the budget before the scoring pass begins.

**How scoring works.** The task is next-word prediction. The model sees text
one token at a time, left to right, and at each position must output a
probability distribution over all possible next tokens. If the next token is t,
the model earns log₂ p(t) bits — close to zero when confident and right,
increasingly negative when wrong. Over a full document t₁, t₂, …, tₙ, the
total score is:

  −∑ₖ log₂ p(tₖ | t₁, …, tₖ₋₁)

Dividing by the total number of bytes in the document gives **bits-per-byte**
(BPB). The byte-level denominator matters: a token that encodes two bytes
counts as two bytes in the denominator, so longer tokens are neither rewarded
nor penalized just for being large — something that turns out to be important
later. Lower BPB is better.

A useful analogy: imagine a multiple-choice exam with 256 options at every
question. Instead of circling one answer, you must assign a probability to each
option, with all 256 summing to 1. Your score on each question is −log₂ of the
probability you assigned to the correct answer. Assign 100% to the right answer:
score 0 (perfect). Assign equal weight to all 256 options: score log₂(256) = 8.
A model with no knowledge of text does exactly that — uniform over 256 bytes —
scoring 8 BPB. A perfect oracle scores 0 BPB. The baseline transformer the
competition started from scored **1.2244 BPB**.

**The rules: C1–C4.** The competition launched without a complete ruleset. As
participants found increasingly creative — and occasionally questionable — ways
to improve their scores, four constraints were codified mid-competition through
community discussion in Issue #1017:²

- **C1 — Causal eval:** the probability assigned to token tₖ must depend only
  on the tokens before it. No peeking at what comes next. In the multiple-choice
  analogy: you must commit to your probability distribution before seeing which
  answer is correct. What happened in practice: certain submissions fed the
  answer token into the model as part of the context while scoring that very
  token — effectively choosing all options, seeing which one is right, then
  going back and erasing the wrong ones.

- **C2 — Normalized distribution:** the output must be a valid probability
  distribution summing to exactly 1. You cannot manufacture artificially low
  entropy by assigning more than 100% total mass across tokens. In the analogy:
  you cannot assign 90% probability to each of four options.

- **C3 — Score before update:** in TTT, a chunk of validation text must be
  fully scored *before* any gradient step is applied to it. Training on a chunk
  and then scoring it means the model has already adapted to those exact tokens.
  In the analogy: you cannot read the exam questions, go study them, then come
  back and sit the exam.

- **C4 — Single pass:** each validation token is scored exactly once. No
  re-scoring after adaptation. In the analogy: you cannot retake the same
  question after seeing the answer.

In the six weeks the competition ran, the score fell from 1.2244 to 1.0565 — a
drop of 0.168 BPB. [TODO: score progression graph]

In the next section we trace the key techniques that drove this improvement.

---
¹ The artifact is the compressed model weights plus all code in `train_gpt.py`.
The cap is 16,000,000 bytes (decimal) — not 16 MiB (16,777,216 bytes), a
distinction that matters: the actual budget is about 4.5% smaller than a
"16 megabyte" headline implies. No external downloads or network calls are
permitted during evaluation.
² Issue and PR numbers refer to the openai/parameter-golf GitHub repository:
`https://github.com/openai/parameter-golf/issues/{number}` or `/pull/{number}`.

---

## Part 1 — Model Comparison and Techniques

There are five components to a Parameter Golf submission: the tokenizer,
the model architecture, the training setup, the quantization pipeline, and
post-training adaptation. Let's start with the base model that OpenAI
provided at the start of the competition.

*(This section assumes basic familiarity with transformer architecture at
the level of NanoGPT — attention, residual stream, MLP layers. If that's
new territory, Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
is the right starting point.)*

---

### The Baseline

OpenAI's starting model was already not a simple transformer — not a plain
NanoGPT-style stack of attention and MLP layers. The baseline was more
carefully engineered than that.

**Tokenizer.** The baseline used SentencePiece with a 1024-token vocabulary
(SP1024), trained on the same FineWeb corpus used for scoring. A 1024-token
vocabulary is quite small by modern standards; GPT-2 uses 50,000 tokens.
The small vocabulary keeps the embedding table compact, which matters when
your entire model has to fit in 16 MB.

**Model architecture.** The baseline was a 9-layer, 512-dimensional transformer
with three structural additions worth calling out:

- **U-Net skip connections** — in a standard transformer, each layer feeds
  only into the next. The U-Net pattern adds direct connections that skip the
  middle of the network. The 9 layers are split into an encoder half (layers
  1–4), a bottleneck (layer 5), and a decoder half (layers 6–9). Each decoder
  layer receives the output of its mirror encoder layer as an additional
  residual, weighted by a learned scalar *w*:

  ```
  Standard:  h_l = Block_l(h_{l-1})

  U-Net:     h_l = Block_l(h_{l-1})               for l ≤ 5
             h_l = Block_l(h_{l-1} + w · h_{10-l}) for l ∈ {6,7,8,9}
  ```

  Concretely: layer 6 gets a skip from layer 4, layer 7 from layer 3, layer 8
  from layer 2, layer 9 from layer 1. The early-layer representations — which
  tend to capture local, surface-level patterns — are fed directly into the
  late layers alongside the deep representations. This is the same idea that
  made U-Net famous in image segmentation.
- **Grouped-query attention (GQA)** and **rotary positional embeddings (RoPE)**
  — both standard in modern LLMs, just not in NanoGPT.
  GQA shares key-value heads across query heads to cut parameter count; RoPE
  encodes position by rotating query and key vectors rather than adding learned
  position embeddings.

**Training.** The baseline used the **Muon optimizer** — a popular choice
over AdamW. The learning rate follows a
warmup-then-warmdown schedule, rising over the first few steps then decaying
to zero by the end of the 10-minute window.

**Quantization.** After training, the model weights are rounded from their
training precision (bfloat16, 16 bits per weight) down to 6 bits per weight
(int6). This is the core quantization step that makes the 16 MB constraint
achievable: a bfloat16 copy of this model would be around 70 MB. The baseline
applied int6 quantization to MLP weights only; attention weights were left at
higher precision.

The key concept for readers unfamiliar with quantization: every weight is a
number stored with some number of bits. More bits means more precision, but
also a larger file. The game is managing the tradeoff — round the weights
aggressively enough to fit the size budget, but not so aggressively that the
model's predictions degrade. The baseline's approach was simple: round the
biggest chunk of parameters (the MLP weights) and leave the rest alone.

**Post-training adaptation.** The baseline did none. During the 10-minute
evaluation window, it simply ran the model forward on the validation text
and recorded the scores. Later models would use this window to actively update
their weights in response to what they were seeing — a technique called
test-time training (TTT). The baseline serves as the clean reference point
before that complication enters.

