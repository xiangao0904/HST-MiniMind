# Repository Inspection

Date: 2026-05-15

## Current State

- Project root: `/data/aoxiang/repo/HST-test`
- Existing files: `PRD.md`, `proxy.txt`
- No MiniMind source tree is present in the current project root.
- The platform provides read-only `.git`, `.agents`, and `.codex` directories in the root. Git metadata for this working tree is therefore stored in `hst_git/` and used with `GIT_DIR=hst_git GIT_WORK_TREE=.`.

## MiniMind Integration Points

Because MiniMind code is not present yet, these PRD inspection tasks cannot be resolved against real source files:

- MiniMind model class: not found.
- Tokenizer loading code: not found.
- Dataset loading code: not found.
- Existing pretraining loop: not found.
- Existing loss calculation: not found.

## Implementation Decision

The initial implementation will be additive and isolated:

- Provide a small GPT-style causal LM for local smoke tests and remote training scaffolding.
- Keep HST modules independent under `model/`, `trainer/`, `utils/`, and `scripts/`.
- Design the training entrypoint so a future MiniMind model can replace the local model without changing the superposition composer or losses.
- Do not modify files outside `$PROJECT_ROOT`.
