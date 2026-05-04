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
¹ Specifically: the compressed model weights plus all code in `train_gpt.py`.
² Issue and PR numbers throughout this post refer to the openai/parameter-golf
GitHub repository: `https://github.com/openai/parameter-golf/issues/{number}`
or `/pull/{number}`.
The cap is 16,000,000 bytes (decimal) — not 16 MiB (16,777,216 bytes), a
distinction that matters: the actual budget is about 4.5% smaller than a
"16 megabyte" headline implies. No external downloads or network calls are
permitted during evaluation.

---
