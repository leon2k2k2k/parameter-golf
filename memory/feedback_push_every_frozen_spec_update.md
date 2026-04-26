---
name: Push every frozen spec update
description: If I change a frozen runnable spec or its execution branch, I must push before reporting readiness again
type: feedback
---

If I freeze or update a runnable spec, branch, or launch contract, I must push that change before I say it is ready.

Rules:

- Do not stop after a local edit to a frozen spec.
- Do not say a spec is updated if the remote branch or remote-tracked spec still reflects the old state.
- If I change wallclock, env, branch pin, commit pin, or launch code on a runnable spec, I must push the corresponding branch before reporting the new frozen state.
- If the spec lives outside the execution branch, I must say that clearly and avoid claiming the execution branch already contains the update.

Minimum check before reporting readiness:

1. confirm which branch actually owns the runnable contract
2. commit the update on that branch if needed
3. push that branch
4. only then report the spec/branch as updated
