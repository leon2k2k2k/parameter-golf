---
name: Commit on the spec branch
description: When a spec names a branch, commits must be created and pushed on that exact branch, not left only on research
type: feedback
---

If a spec pins a branch, create the commit on that exact branch and push that exact branch to the remote named in the spec.

Do **not**:

- commit only on local `research`
- assume a cherry-pick or later branch fix-up is acceptable
- leave the spec branch missing the pinned commit

For example, if the spec says:

- branch: `exp/031-direct-carry-freefloat`

then the commit must exist on:

- local `exp/031-direct-carry-freefloat`
- remote `fork/exp/031-direct-carry-freefloat`

**Why:** Execution uses the branch named in the spec. If the commit exists only on `research`, the spec is effectively wrong and the executioner will pull stale or missing code.

**How to apply:** Before finalizing a spec as `READY`, verify:

1. you are on the spec branch when committing
2. the commit is pushed to the matching remote branch
3. the spec’s pinned branch and commit are both actually reachable together
