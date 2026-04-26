---
name: Executioner must validate launch code
description: Launch commands in specs are not blind paste targets; execution must understand and validate them against the real environment
type: feedback
---

The executioner must understand the launch code, not just paste it.

Spec command blocks are a proposed executable contract, but execution must still validate that they are coherent with:

- the current environment
- the pinned branch and commit
- the actual code's env vars and flags
- the repo's current path conventions
- the referenced directories, inputs, and outputs

Execution should catch and report command/spec mistakes such as:

- wrong mount paths
- wrong branch/commit reachability
- stale or nonexistent env vars
- inconsistent launch assumptions
- missing files or directories

Execution may fix environmental mismatches when the intended command is still clear, but must not silently change the experiment itself.

Do **not**:

- treat the command block as sacred text if it is obviously inconsistent
- silently change major settings
- silently upgrade hardware
- patch training logic during execution

If command validation fails, execution should hand back a concrete report of what is wrong and what needs to be fixed in research/spec land.
