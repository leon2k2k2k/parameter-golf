# Quant Damage On The Current CaseOps Frontier Stack

## Thesis

Quantization damage looks like one of the largest remaining reservoirs on the
modern strong CaseOps family.

This is not a vague "quantization is hard" claim. It is visible directly in the
public frontier numbers:

- `#1787`: pre-quant `1.06699` -> quantized `1.07632` = **+0.00933 BPB**
- `#1801`: pre-quant `1.06671` -> quantized `1.07598` = **+0.00927 BPB**

That is enormous relative to frontier-sized gains. Most meaningful record-track
moves are on the order of `0.001-0.003 BPB`, so a `~0.0093` quant hole is a
large reservoir if any part of it can be recovered cheaply.

## Why this matters now

The public sequence makes the argument stronger:

- `#1787` proved the training/eval stack still had room below `#1736`
- `#1801` showed the carry line also pushes below baseline
- `#1797` then attacked the same family from the quant side, explicitly framing
  part of the gain as recovery of int6 quantization tax

So the interesting question is no longer "does quant damage exist?" It clearly
does. The question is:

- where is the damage concentrated?
- how much survives TTT?
- what is the cheapest byte-efficient way to buy it back?

## Mechanism hypothesis

The stack is good enough pre-quant that the post-quant gap is probably not
coming from one gross modeling failure. More likely it is concentrated in a few
high-leverage regions:

- MLP rows with large outliers / bad int6 fit
- gate-related weights whose small magnitudes or narrow dynamic range quantize
  awkwardly
- a small number of especially sensitive tensors where the current clip /
  bit-allocation policy is still crude

The DexHunter-style read is:

1. training recipe is already strong
2. quantized checkpoint is still much worse than pre-quant
3. therefore local repair on the quant path is one of the few remaining places
   where a sub-2e-3 frontier gain can plausibly come from

## What makes this attractive

- The signal is already visible in public numbers; we are not inventing a
  theory first.
- Quant repair can be orthogonal to carry / schedule / optimizer improvements.
- It is possible to screen quant ideas more cheaply than a whole new
  architecture if we keep the experiment focused on the pre-quant -> post-quant
  gap.

## Main risks

- Quant repair does not translate 1:1 into final post-TTT gain.
- Some apparent quant damage may already be recoverable by TTT, so repairing it
  directly could show little end-to-end benefit.
- Byte-efficient repair is the real constraint. A method that buys back a lot
  but costs too much artifact budget is not frontier-usable.
- It is easy to jump too quickly to a fancy mechanism (`LQER`, special routing,
  mixed exceptions) before measuring where the hole actually lives.

## Recommended research order

### 1. Measurement before invention

First build a clean local diagnostic for the current family:

- post-EMA pre-quant `val_bpb`
- quantized `val_bpb`
- post-TTT `val_bpb`
- tensor- or module-level attribution of quant loss if feasible

The immediate goal is to answer:

- does most of the `~0.0093` live in MLP, attention, embeddings, gates, or a
  tiny set of outlier tensors?

### 2. Cheap discriminating experiment

Before attempting a full DexHunter-style repair, do the smallest experiment that
can tell us whether the hole is concentrated:

- slightly relax quantization only for one suspected bucket
- keep everything else fixed
- measure how much post-quant BPB moves and whether post-TTT follows

Examples of the right style:

- one targeted clip change
- one targeted precision exception
- one targeted repair path on MLP only

Examples of the wrong style:

- port an entire external quant bundle all at once
- mix clip changes, precision changes, and low-rank correction in one run

### 3. Only then freeze a real spec

If the first diagnostic says the hole is concentrated and byte-efficiently
recoverable, then freeze a proper numbered spec around the best mechanism.

## Current judgment

- **Mechanism:** quant damage reservoir on the modern CaseOps family
- **Expected delta:** plausible frontier-relevant upside if even `15-30%` of the
  current `~0.0093` gap can be recovered under budget
- **Confidence:** medium that the reservoir is real, low-to-medium on any
  specific repair method before diagnostics
- **Main risks:** TTT overlap, byte budget, and misidentifying where the loss
  actually lives
- **Cheapest next step:** write a measurement-first quant-damage screen for the
  current `#1801`-style family rather than jumping straight to a repair bundle
