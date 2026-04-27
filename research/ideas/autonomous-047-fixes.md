# Autonomous 047 Fixes — Session Notes

**Date:** 2026-04-27
**Purpose:** Fix 047C (mid-run hang) and 047D (silent AdaLN gradients), study TTT LoRA, spec 047E/F.

---

## Task 1: Read REPORT and logs — DONE

**047D REPORT summary:**
- Attempts 1+2 (`eea2c67`): hung at loop-active backward pass — NCCL/autograd deadlock at step ~2300.
- Attempt 3 (`40a59db`, fix2: allow_in_graph module-level fn): completed cleanly, val_bpb=1.06520. But AdaLN is SILENT — γ/β never received gradients. Loss curve tracks 045 armD within noise.
- Root cause: `@allow_in_graph` marks the function opaque to dynamo/autograd → backward bypasses γ/β parameter nodes. Adam update = 0 for these params.

**047C train.log (last 20 lines):**
- Train was running fine pre-loop at ~4.30M tok/s (steps 1300-2200).
- `recur_alpha` printing shows it was alive and computing.
- But we NEVER see steps past 2200 — the log ends at step 2200 which is the last pre-loop step.
- Confirms: hang was at loop activation (step ~2300, frac=0.35).

---

## Task 2: Fix 047C — DONE

**Change:** `ENABLE_LOOPING_AT="${ENABLE_LOOPING_AT:-0.35}"` → `"${ENABLE_LOOPING_AT:-0.0}"`
- File: `worktrees/047C-per-pass-lora-ffn/tmp_exec/launch_047C.sh`
- Smoke script: removed `export ENABLE_LOOPING_AT=0.05` (no longer needed; main defaults to 0.0)

**Why this fixes it:** Loop active from step 1 means the loop-active graph compiles at step 1, not mid-training. No cold-graph backward pass encountered during training budget.

**Risk:** Training starts harder (loop active immediately). But since LoRA delta=0 at init, this is effectively the same as baseline loop-active behavior at step 1.

**Commits:**
- `d9d6bcb` on `exp/047C-per-pass-lora-ffn` — fix ENABLE_LOOPING_AT
- Pushed to `fork exp/047C-per-pass-lora-ffn`
- Spec updated: commit → d9d6bcb, launch plan notes new behavior

---

## Task 3: Fix 047D — DONE

**Change:** Remove `@torch._dynamo.allow_in_graph` from `_adabn_apply` function.
- File: `worktrees/047D-loop-adabn/records/.../train_gpt.py`
- Also: added `tmp_exec/launch_047D_fix3.sh` with `ENABLE_LOOPING_AT=0.0` and `SHA=ac6598b`

**Why this fixes gradients:** `allow_in_graph` tells dynamo to treat the function as an opaque call — so autograd never traverses into it to compute gradients for γ/β. Removing the decorator allows dynamo to inline the ops natively (γ*x+β), which are trivially differentiable.

**Why ENABLE_LOOPING_AT=0.0 also needed:** Without this, fix3 would still hang at step ~2300 (same as prior attempts 1+2) because the backward pass of the loop-active graph would be cold at activation time.

**Commits:**
- `ac6598b` on `exp/047D-loop-adabn` — remove allow_in_graph + add fix3 launch script  
- `7b38f8d` — update SHA in launch script to ac6598b
- Pushed to `fork exp/047D-loop-adabn`
- Spec updated: fix history added, commit → 7b38f8d

**Remaining concern:** Need to verify γ/β grad norm > 0 at step 1. The launch_047D_fix3.sh notes this as a key diagnostic. If grad is still 0, there may be a deeper autograd issue (e.g., torch.compile disabling backward for some ops).

---

## Task 4: Study TTT LoRA gradient flow — DONE

**Reference:** `records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py`

**Key patterns:**
1. `BatchedLinearLoRA`: plain `nn.Module` with `nn.Parameter` A (kaiming-uniform init) and B (zero init). Forward: `(x @ Aᵀ) @ Bᵀ`. No dynamo decorators anywhere.
2. `BatchedTTTLoRA`: contains a ModuleList of LoRA adapters. No custom autograd.
3. Gradients flow via standard autograd — no decorators, no opaque wrappers.
4. The key insight: LoRA backward in TTT code uses plain matrix multiplications that autograd traces natively. This confirms that decorators are NOT needed and are counterproductive.

**Application to 047C:**
- 047C's activation-side LoRA (pre-folded A@B in eager, applied as additive delta on activation) follows the same principle — plain ops, no wrappers.
- The TTT LoRA is per-batch, per-step (different A/B for each test batch). Our 047C LoRA is per-pass (shared across training steps). Both should be autograd-transparent.
- **Robustness concern:** 047C uses `torch.bmm` to fold A@B in Python before the compiled forward. This is eager, so it gets recomputed every step. At r=2, this is negligible cost. The TTT code also uses non-compiled loops — consistent pattern.

---

## Task 5: Spec 047E/F — DONE

### 047E — Per-pass FFN LoRA on 039bL base

**File:** `research/specs/047E-lora-ffn-on-039bL-base.md`
**Branch/commit:** same as 047C (`d9d6bcb`)
**Baseline:** 039bL pre-quant 1.06594
**Config diff vs 039bL:** `LOOP_FFN_LORA_RANK=2`, `ENABLE_LOOPING_AT=0.0`
**Cost:** ~$2.30 for 4×H100 screen

**Rationale:** 
- Tests LoRA additivity on the simpler 039 code stack (no LOOP_ITER_EMBEDS, no RECUR_ALPHA, floor-then-linear LR)
- 039bL baseline (1.06594) is weaker than 045 armAC (1.06479), so beating it is a lower bar
- But if LoRA helps on the simpler stack, it's stronger evidence for the technique itself

### 047F — Per-pass AdaLN on 039bL base

**File:** `research/specs/047F-adabn-on-039bL-base.md`
**Branch/commit:** same as 047D fix3 (`7b38f8d`)
**Baseline:** 039bL pre-quant 1.06594
**Config diff vs 039bL:** `LOOP_ADABN=1`, `ENABLE_LOOPING_AT=0.0`
**Cost:** ~$2.30-3.65 for 4×H100 screen (depends on cache availability)

**Rationale:**
- 039bL never had LOOP_ITER_EMBEDS, so there's no confound — clean additive test
- Key diagnostic: verify γ/β grad norm > 0 at step 1 before committing to full run
- Pairs with 047E for simultaneous screening (~$5 total for both)

---

## Decision logic for next steps

1. **Run 047C first** (cheapest, most likely to work — ENABLE_LOOPING_AT fix is straightforward).
2. **Run 047D fix3** next (slightly higher risk — need to verify grad flow diagnostic).
3. **Run 047E+F together** (same pod, sequential, ~$5 total). Use as a sanity check that 047C/D approaches are technique-grade (not stack-specific).

If 047C shows Δ ≤ -0.001 AND 047D fix3 shows the AdaLN params actually moved:
- Spec 047G = 047C + 047D fix3 combined on 045 armAC stack

---

## Commits made this session

| Branch | Commit | Description |
|---|---|---|
| `exp/047C-per-pass-lora-ffn` | `d9d6bcb` | Fix ENABLE_LOOPING_AT=0.0 |
| `exp/047D-loop-adabn` | `ac6598b` | Remove allow_in_graph + fix3 launch |
| `exp/047D-loop-adabn` | `7b38f8d` | Fix3 launch SHA update |
| `exp/046-quant-repair` (research) | (pending) | Spec updates: 047C/D/E/F |

---

## Key findings

1. **allow_in_graph blocks gradient flow** — any time we need a parameter to actually train, avoid any dynamo decorator that makes a function opaque. Plain ops are always preferable.
2. **ENABLE_LOOPING_AT=0.35 causes hang** — applies to ANY spec with a new loop-active backward graph that's cold at activation time. Always use 0.0 for new code changes that touch the loop path.
3. **Pre-folding A@B in eager is the right pattern** — matching the TTT LoRA reference implementation. No Triton autotune for rank-2 matmuls.
4. **039bL (1.06594) is a weaker baseline than 045 armAC (1.06479)** — 047E/F on 039bL are lower-bar tests. A pass there is necessary but not sufficient.
