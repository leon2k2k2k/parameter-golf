# eval/

Methodology notes, failure-mode analyses, and cross-spec qualitative findings.

## What goes here vs other dirs

| Dir | Role | Granularity |
|---|---|---|
| `research/specs/NNN-slug.md` | Frozen run contracts | Per-spec |
| `research/evaluations/NNN-slug.md` | Post-run quantitative summary | Per-spec |
| `research/ideas/<slug>.md` | Forward-looking proposals | Per-idea |
| `research/literature/<slug>.md` | Background reference reading | Reference |
| `runs/<spec>/` | Raw run artifacts (final.json, train.log) | Per-run |
| `testing/<script>.py`, `testing/outputs/<date>_<descr>/` | Inspection scripts and their captured outputs | Per-inspection |
| **`eval/<date>_<descr>.md`** | **Methodology + cross-cutting findings** | **Cross-spec / methodology** |

## Examples of what belongs in `eval/`

- **Methodology notes** — how we measure val_bpb, how to read the sidecar bytes, why we use varlen attention at eval time, etc.
- **Failure-mode analyses** — qualitative deep dives into where a model fails on val, what kinds of bytes it gets wrong, how that connects to potential downstream levers (PPM, TTT).
- **Cross-spec comparisons** — e.g. "spec 047B vs spec 050: failure-profile diff."
- **Calibration studies** — does the model's softmax probability match its actual accuracy?
- **Eval-time lever studies** — does PPM help on URL-like bytes more than on prose? quantified.
- **Tokenizer / preprocessing studies** — what does the SP8192+CaseOps tokenizer split into in practice?

## Examples of what does NOT belong here

- Running scripts → `testing/`
- Per-spec post-run quantitative analysis → `research/evaluations/`
- Forward-looking idea proposals → `research/ideas/`
- Specs (frozen contracts) → `research/specs/`

## Naming convention

`<YYYY-MM-DD>_<short-descr>.md`. Date-prefixed so order is obvious.

## Git policy

Markdown notes are committed and pushed to `research`. Big artifacts (`.npz` files,
multi-MB tables, etc.) live in `testing/outputs/` and are gitignored — `eval/` notes
should reference them by relative path but not include them inline.
