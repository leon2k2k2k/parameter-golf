# testing/

Hands-on model inspection. Scripts that load checkpoints and *show us what the
model is actually doing* — not just aggregate val_bpb numbers.

## What this is for

`research/` and `runs/` give us numbers. `testing/` gives us **intuition**:

- "What does the model predict for the next token at this exact position?"
- "Where in the val stream does NN's NLL spike?"
- "What kinds of bytes does PPM catch that NN misses?"
- "Does our model fail in the same places dexhunter's model does?"

These questions can't be answered from `val_bpb=1.0654`. They require running
inference on the val set and *looking at the output*.

## What goes here

| Subpath | Purpose |
|---|---|
| `*.py` (top-level) | Inspection / probe scripts. Self-contained, runnable. |
| `outputs/<date>_<descr>/` | Captured outputs from running a script. Markdown reports, decoded text, prediction tables. Reviewed manually. |

## What does NOT go here

- **Training scripts** → `tmp_exec/`
- **Model code changes** → `records/track_10min_16mb/<spec>/`
- **Spec / eval docs** → `research/specs/`, `research/evaluations/`
- **Run artifacts** (final.json, train.log) → `runs/`

If a `testing/` script reveals something worth iterating on, that's an idea
(`research/ideas/`) or a spec (`research/specs/`), not a `testing/` artifact.

## How to run scripts

Most scripts are designed to run on a pod (the volume has the val data and
checkpoints). Pattern:

```bash
# 1. Copy script to pod
scp testing/inspect_val_text.py root@<pod-ip>:/tmp/

# 2. Run on pod (CPU is fine for small inspections; GPU only if doing inference)
ssh root@<pod-ip> 'python3 /tmp/inspect_val_text.py > /tmp/output.md'

# 3. Pull output back into testing/outputs/<date>_<descr>/
scp root@<pod-ip>:/tmp/output.md testing/outputs/2026-04-28_val_text_first_200/output.md
```

CPU is the default — if a script needs GPU, mark it explicitly in the docstring
and avoid running it on a pod where training is live.

## Don't pollute the index

Outputs in `testing/outputs/` are reference material — committed for sharing
across sessions, but kept tight. If an output gets large (>500 lines) or
loses relevance, prune it. This is not a permanent archive; it's a scratchpad
that the team reads and discards.
