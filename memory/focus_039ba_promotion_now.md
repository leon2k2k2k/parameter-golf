---
name: Current Focus — Preserve 039bA Signal
description: The main research focus right now is preserving the strong 039bA 4H penalized-tanh training signal when promoting to 8H, with loop timing as the first lever
type: focus
---

Current top-priority idea:

- `039bA` (`4×H100`, `600s`, training-only) is the strongest live training-side signal.
- It reached `1.12148667` pre-quant post-EMA val_bpb and beat the `039b` 4H baseline by `-0.03346342`.

What to optimize around:

- Treat `039bA` as the best short-run training base.
- The main question is how to preserve that gain when promoting to `8×H100` full runs.
- The first lever to test is **loop timing / loop-onset schedule**, not widening the activation region.

Current interpretation:

- `041A` showed that naïvely promoting the same loop-band `penalized_tanh` idea to the `8H` full regime does not automatically inherit the `4H` gain.
- This looks like a **regime / schedule interaction**, not evidence that the `039bA` signal was fake.
- The likely next spec direction is:
  - keep the `039bA` activation placement the same
  - move looping earlier, or otherwise make the recurrent middle band active earlier in the promoted run

Do not lose focus:

- primary live path: preserve `039bA`
- secondary exploratory path: other MLP allocation ideas
- do not broaden the activation change to earlier layers before testing loop-timing promotion fixes
