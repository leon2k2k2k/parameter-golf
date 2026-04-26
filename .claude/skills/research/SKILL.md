---
name: research
description: Activate research role for the Parameter Golf repo. Invoke at the start of a research session (no pod live). Loads repo conventions and reminds what research does vs what execution does.
---

# Research role

You are in **research mode** for the Parameter Golf record-track push.

## What research does
- Reads `CLAUDE.md` at the top of this repo (authoritative; defer to it for all conventions).
- Thinks about ideas, writes free-form notes in `research/ideas/<slug>.md`.
- When the user says "spec this one," freezes an idea into `research/specs/NNN-<slug>.md` using the spec template.
- Writes code diffs on `exp/<slug>` branches of the training code, in a `worktrees/<slug>/` worktree. CPU-only sanity checks if possible; otherwise leave validation to execution's 2×H100 mini rung. Pins a commit hash into the spec.
- After an execution session completes a run, reads `runs/NNN-slug/`, writes `research/evaluations/NNN-slug.md`, and appends a row to `experiments.md`.
- Writes diary entries in `diary/` as the session progresses.

## What research does NOT do
- Never launches pods, never runs training on real hardware.
- Never writes to `runs/` during a run (only syncs small artifacts if asked).
- Never skips the interview step — always flag open questions in the spec so execution surfaces them.

## First actions on session start
1. Read `CLAUDE.md` at the top of this repo.
2. Check recent `experiments.md`, `research/specs/`, `research/evaluations/` to see where the loop is.
3. Ask the user what they want to work on: new idea, freeze an existing idea into a spec, or evaluate a completed run.

## Reminders
- Specs are contracts. Short, unambiguous, ~1 page max. Hypothesis essays go in `research/ideas/`.
- Numbering is linear: `000`, `001`, `002`, …. Assigned at spec-freeze time.
- Multi-seed runs are one spec, not many.
- Accept criteria and stop-early criteria must be in every spec.

## Spec freeze checklist — a spec is NOT ready until ALL pass

Whenever you write or update a spec, the spec is not "ready" / "frozen" /
"handed off to execution" until **every** item below is true. Run through this
before claiming the spec is done, before pausing the conversation, and before
the user has to ask "is it pushed?".

1. **Spec file committed.** `git status` shows the spec file is not in the
   working tree as modified or untracked. If it's modified, `git add` + commit
   it now with a descriptive message.
2. **Code commit pushed.** The commit hash the spec pins must exist on
   `fork`. Run `git ls-remote fork <branch>` and compare to the local HEAD or
   the spec's pinned hash. If stale, `git push fork <branch>` immediately —
   no permission prompt.
3. **Spec change pushed.** If the spec lives on a branch that's tracked on
   fork (typical for `exp/<slug>` branches that hold both code and the spec),
   the spec commit must also be pushed.
4. **No silent code changes.** If you edited training code, the diff must be
   in the pinned commit. If you have uncommitted code edits sitting in the
   worktree, the pod will check out the pinned commit and run **without**
   them — silently producing the wrong run.
5. **Verify with one command:** `git status` (clean) AND `git ls-remote fork <branch>` (HEAD matches local).

Reason: An execution session reads the spec and immediately preflights a pod
(~$0.20–0.40/min). Any unfrozen state (uncommitted spec, unpushed code) either
blocks the pod or, worse, runs the wrong code. Pushing is publishing what was
already authored intentionally — never ask permission.

After hand-off, the spec is immutable until the run completes. Edits during a
run silently change the contract under execution's feet.
