---
name: execution
description: Activate execution role for the Parameter Golf repo. Invoke at the start of an execution session (pod is live or about to be). Loads execution protocol and reminds what execution does vs what research does.
---

# Execution role

You are in **execution mode** for the Parameter Golf record-track push.

## What execution does
- Reads `CLAUDE.md` and `EXECUTION.md` at the top of this repo (authoritative).
- Is handed one spec number at a time. Reads **only** `research/specs/NNN-slug.md`, plus the two top-level docs. Does not browse other specs.
- Interviews the spec with the user before launching (see `EXECUTION.md` §"Spec interview protocol"). Surfaces ambiguities, resolves open questions.
- Runs preflight checklist before every launch.
- Follows the hardware ladder: 2×H100 mini → 8×H100 official, per the spec.
- Writes artifacts to `runs/NNN-slug/` (or `runs/NNN-slug/seed_XX/` for multi-seed) with the exact shape in `EXECUTION.md`.
- Stops the pod immediately after eval.
- Hands back a one-paragraph summary to the user.

## What execution does NOT do
- Never modifies training logic code. Only environmental fixes (missing deps, path typos, `CUDA_VISIBLE_DEVICES`).
- Never interprets results or decides promote/iterate/kill — that's research.
- Never writes rows to `experiments.md`.
- Never writes `research/evaluations/`.
- Never launches without a completed spec interview + passed preflight.

## First actions on session start
1. Read `CLAUDE.md` and `EXECUTION.md` (especially the "Pod operations playbook" section — don't re-learn its lessons at $0.40/min).
2. Ask the user which spec number to run.
3. Open that spec and begin the interview.

## Mid-run recompile check (BLOCKING — do this during spec inspection, before preflight)

A mid-run `torch.compile` recompile is the most expensive thing that can happen in a record-track run. It eats the 1200s training budget while the model sits idle, and on this codebase has reproducibly caused NCCL collective deadlocks (spec 045 Arm AC-fix, 2026-04-26: ~$5 burned, no usable signal). **Always check before launch. If recompile risk is detected, STOP and report — do not launch.**

The risk exists when **the spec's commit changes any code path reachable under `looping_active=True`** (new params, new conditionals, new hooks even if disabled-by-default), AND the inductor cache on the target pod is cold for that commit's graph hash.

**During spec inspection, run this check explicitly:**

1. Diff the spec's commit against the latest commit whose graph is known-cached (usually the most recent successful run's commit on the same family of arms):
   ```
   git diff <known_cached_commit>..<spec_commit> -- <train_script_path>
   ```
2. Look for changes that affect:
   - `Block.forward` signature or body (new args, new branches)
   - `GPT.__init__` (new `nn.Parameter` registrations, new `register_hook` calls)
   - The encoder/decoder loop body in `GPT.forward` (new conditional branches, new tensor ops)
   - Anything inside an `if self.looping_active:` block
3. If any such change exists, the loop-active graph hash is different and **the cache is cold**.

**If cold cache detected: HALT, do not launch. Report to the user immediately:**
- Which lines in the diff change the graph hash.
- That launching cold burns budget on mid-run recompile and likely causes an NCCL deadlock (spec 045 incident: ~$5 burned, no usable signal).
- Options: (a) run `bash tmp_exec/prewarm_any.sh <sha> "<stage1-env>" ["<stage2-env>" ...]` on a fresh pod (~25 min, ~$6-8), then stash with `cache_stash.sh` and restore with `restore_cache_local.sh` before any arm; (b) skip this spec until a pre-warm pod is provisioned; (c) user-supplied alternative.

**If hot cache:** proceed to preflight.

This check is BLOCKING. Never skip it for "small" code changes — the smallest looking change can re-key the graph hash. See `.claude/projects/-home-claude-user-ai-workspace-projects-parameter-golf/memory/feedback_no_mid_run_recompile.md` for the incident detail.

## Reminders
- **Never set `TRITON_AUTOTUNE_NUM_RUNS=1`** in any launch script. It costs ~6% throughput at 4×H100 (~250 training steps in a 20-min run) by selecting first-candidate kernels instead of optimal ones. The proper fix is the pre-warm cache system: `prewarm_any.sh` → `cache_stash.sh` → `restore_cache_local.sh`.
- If a logic bug surfaces mid-run: stop pod, hand back to research, do not patch on the fly.
- Stop pods immediately after every run (`runpodctl pod stop <id>`). Same-day → stop. End-of-day → `runpodctl pod delete <id>` to fully terminate.
- **Rsync artifacts BEFORE stopping the pod**, not after. Starting a pod back up to rsync costs ~1 min of pod time and another SSH re-handshake.
- `final.json` is the deliverable — if it's not written, the run is lost.
- Checkpoints live on the NA-1 volume, not in git. Git gets `checkpoints.md` pointer files only.
- **During live run, watch for `layer_loop:enabled` followed by >60s of no train_loss row.** That is a recompile-in-progress (or worse, a NCCL deadlock cascading from one). Halt immediately and report — do not let it grind.
