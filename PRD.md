# PRD: HST-MiniMind Next Phase

## 0. Document Purpose

This document is the handoff PRD for the next conversation / next implementation round.

It replaces the earlier broad PRD with a narrower and more execution-oriented version based on what has already been built and what has already been learned from running the first batches.

Primary goal now:

```text
Do not expand methods yet.
First make vanilla TST replication clean, stable, and interpretable.
```

---

## 1. Project Snapshot

### 1.1 Project Name

**HST-MiniMind**  
Structure-aware and paper-aligned Token Superposition Training experiments.

### 1.2 What Exists Already

The repository already contains:

- A small causal LM training stack for local/remote validation.
- `ntp_baseline`, `vanilla_tst`, `order_aware_tst`, `boundary_aware_tst`, `hierarchical_tst`.
- Recovery phase switching.
- W&B logging support.
- Path safety checks.
- Tiny smoke scripts.
- MiniMind tokenizer support via `tokenizer/minimind_tokenizer`.
- Tokenized cache support.
- Packed tokenized cache generation.
- Equal-FLOPs style raw sequence expansion for TST.

### 1.3 Current Constraint

The current trainer is still a lightweight GPT-style implementation, not the original MiniMind backbone.

That is acceptable for the current phase only if the goal is:

```text
stabilize the TST protocol and recover a believable vanilla TST signal
```

It is not yet sufficient to claim a full MiniMind-level reproduction.

---

## 2. Current Findings

### 2.1 What Was Wrong Before

An earlier "paper-aligned" batch used a bad cache construction:

- each JSONL example was tokenized independently
- then padded to a fixed length such as 3072

That introduced large amounts of padding into long-sequence TST runs, especially `s=4`.
As a result, loss became artificially easy and invalid for comparison.

This issue is now understood and fixed by packed cache generation.

### 2.2 Why Packed Cache Is Required

For formal TST comparison, examples must be built from a continuous token stream:

```text
doc1 + eos + doc2 + eos + doc3 + ...
```

Then cut into fixed-length blocks:

```text
block_0: 3072 tokens
block_1: 3072 tokens
...
```

This avoids pad-dominated training and makes TST runs comparable to NTP runs.

### 2.3 What the Latest Clean 20k Run Showed

Using packed cache and equal-FLOPs setup, the most recent clean batch produced:

```text
P0 NTP baseline:    3.8574
P1 vanilla TST s=2: 3.8818
P2 vanilla TST s=4: 3.8313
```

Interpretation:

- `s=4` shows a weak positive signal versus baseline.
- `s=2` does not.
- The signal is too small to call this a reliable reproduction yet.

### 2.4 Key Paper Interpretation

The main paper insight is:

```text
TST phase does not optimize standard AR next-token loss.
Its value is as a coarse-grained pre-pretraining phase.
Recovery phase is where that coarse representation gets translated back into standard LM performance.
```

Therefore:

- do not judge TST only by first-phase loss
- do not expect standard LM loss gains to appear directly during superposition training
- do measure recovery behavior explicitly

---

## 3. Next-Phase Objective

### 3.1 Primary Objective

Recover a stable and convincing vanilla TST replication signal before touching structure-aware variants again.

### 3.2 Success Criteria

The next phase is successful only if all of the following are true:

1. The evaluation protocol is stable enough that small differences are not dominated by noise.
2. The packed-cache / equal-FLOPs protocol is consistently used for all formal runs.
3. At least one vanilla TST setting shows a repeatable advantage or a repeatable recovery-behavior difference versus baseline.
4. Results can be explained in terms of:
   - TST phase
   - recovery start
   - recovery gap
   - final NTP eval loss

### 3.3 Non-Goals for the Next Phase

Do not prioritize the following yet:

- order-aware ablations
- boundary-aware ablations
- hierarchical ablations
- switching to large hyperparameter sweeps
- new visualizations unrelated to recovery behavior
- pushing for stronger claims than the current setup supports

---

## 4. Required Experimental Logic

### 4.1 Paper-Aligned Comparison Rule

All formal comparison runs must use:

- MiniMind tokenizer
- packed tokenized cache
- equal-FLOPs style TST sequence construction
- explicit recovery schedule
- standard NTP evaluation

### 4.2 Equal-FLOPs Convention

Use:

```text
baseline_seq_len = 768
```

Then:

- NTP baseline uses `raw_seq_len = 768`
- TST `s=2` uses `raw_seq_len = 1536`
- TST `s=4` uses `raw_seq_len = 3072`

Meaning:

```text
TST reads more raw data tokens per step while keeping latent processed length roughly aligned.
```

### 4.3 Recovery Ratio Convention

Current code semantics:

```text
recovery_ratio = fraction of total steps spent in recovery
```

So:

- `recovery_ratio = 0.7` means `30% TST + 70% recovery`
- on a 20k run, that means:
  - `6000` TST steps
  - `14000` recovery steps

This matches the intended interpretation for the current vanilla paper-aligned runs.

---

## 5. Required Metrics

### 5.1 Metrics That Must Be Logged Locally

Keep full `metrics.jsonl` logging with at least:

- `step`
- `phase`
- `loss_train`
- `loss_eval_ntp`
- `loss_eval_phase`
- `raw_tokens_seen`
- `latent_tokens_seen`
- `effective_data_tokens_seen`
- `wall_time_sec`
- `gpu_mem_gb`

### 5.2 Metrics That Must Be Logged to W&B

W&B should stay minimal and only log:

- `loss_train`
- `loss_eval_ntp`
- `loss_eval_phase`
- `lr`
- `raw_tokens_seen`
- `effective_data_tokens_seen`
- `gpu_mem_gb`

Do not spam W&B with constant or low-value step fields like:

- `run_name`
- `method`
- `superpose_size`
- `phase`
- `recovery_ratio`

These should go into W&B config or summary instead.

### 5.3 Metrics That Matter for Interpretation

The next round should focus on:

1. final NTP eval loss
2. best NTP eval loss
3. NTP recovery gap
4. wall-clock to a loss threshold
5. data tokens per second
6. raw tokens per second

---

## 6. Immediate Next Experiments

### 6.1 Priority Batch

These are the primary runs that matter now:

```text
P0_ntp_baseline_20k
P1_vanilla_tst_s2_r03_20k
P2_vanilla_tst_s4_r03_20k
```

These are already the correct template for the current phase.

### 6.2 What To Do After the Current 20k Batch

Do this in order:

1. Run a stronger offline eval over final checkpoints using a much larger eval token budget.
2. Compare `P0/P1/P2` using that offline eval, not the tiny online eval only.
3. If the result is still noisy, repeat the same three runs once more with the same setup.
4. If the signal survives repetition, then extend total steps.

### 6.3 Second Batch If Needed

If the first 20k batch is inconclusive, run:

```text
P0_ntp_baseline_40k
P1_vanilla_tst_s2_r03_40k
P2_vanilla_tst_s4_r03_40k
```

Do not launch structure-aware runs before this decision point.

---

## 7. Required Code Tasks

### 7.1 Evaluation Stabilization

The next coding task after the current run analysis should be:

```text
Add a stronger offline evaluation path for checkpoints.
```

Requirements:

- accept a checkpoint path or run dir
- run standard NTP evaluation only
- use the tokenized packed cache or tokenizer backend consistently
- support larger eval budget than the current online eval
- write results into project-local outputs

Suggested output:

```text
hst_outputs/offline_eval_summary.json
hst_outputs/offline_eval_summary.md
```

### 7.2 Recovery Analysis

Add a small analysis tool that extracts:

- last NTP eval before recovery start
- first NTP eval after recovery start
- recovery gap
- number of steps needed after switch to beat baseline-at-same-step

### 7.3 Summary Cleanup

Keep `scripts/hst_collect_metrics.py` as the simple run summary tool, but do not overload it with every analysis.
If richer analysis is needed, add a separate script.

---

## 8. Local Workflow

### 8.1 Local Machine Role

Local machine is for:

- code editing
- unit tests
- smoke verification
- syncing with remote
- Git commit management

Local machine is not for full training.

### 8.2 Local Validation Commands

Basic checks:

```bash
python3 -m py_compile trainer/train_hst_pretrain.py \
  scripts/hst_collect_metrics.py \
  scripts/hst_tokenize_dataset.py

python3 -m unittest \
  tests/test_hst_path_safety.py \
  tests/test_hst_token_types.py \
  tests/test_hst_shapes.py \
  tests/test_hst_losses.py \
  tests/test_hst_training_protocol.py
```

### 8.3 Local Git Rules

- Always inspect `git status` before committing.
- Never commit generated run artifacts.
- Never commit `image.png` or ad hoc screenshots unless explicitly needed.
- Use one commit per problem.

---

## 9. Remote Workflow

### 9.1 Remote Machine Role

Remote machine is for:

- tokenized cache generation
- long-running training
- W&B-backed experiment execution
- checkpointed formal comparisons

### 9.2 Remote Environment

Use:

```bash
conda run -n lm-test ...
```

Do not install into `base`.

### 9.3 Remote Data and Cache Locations

Use:

```text
dataset/pretrain_t2t_mini.jsonl
hst_tmp/tokenized/pretrain_t2t_mini_packed_seq3072.pt
```

### 9.4 Remote Training Safety

- all run outputs must remain under `hst_runs/`
- tokenized caches must remain under `hst_tmp/`
- do not write outside project root
- do not mutate shared data files

---

## 10. Git Commit Format

### 10.1 Commit Style

All commits must be in English and problem-based.

Good examples:

```bash
git commit -m "Add optional wandb metric logging"
git commit -m "Add paper-aligned tokenizer and TST protocol"
git commit -m "Pack tokenized pretraining cache"
git commit -m "Reduce wandb metric noise"
git commit -m "Add offline checkpoint eval for NTP comparison"
git commit -m "Measure recovery gap from NTP eval checkpoints"
```

Bad examples:

```bash
git commit -m "update"
git commit -m "fix stuff"
git commit -m "more changes"
```

### 10.2 Commit Command Workflow

Use command-line git, not editor-driven interactive flows:

```bash
git status --short
git add <files>
git commit -m "Meaningful English commit message"
```

### 10.3 Push Policy

If the agent is told not to push, do not push.
If the user wants to push manually later, keep the local history clean and self-explanatory.

---

## 11. Known Facts to Carry Into the Next Conversation

The next assistant must assume the following are true unless re-checked:

1. W&B support is implemented.
2. W&B logging has already been reduced to important metrics only.
3. Packed tokenized cache generation is required for valid long-sequence TST comparison.
4. Earlier pad-heavy `P*` runs are invalid and should not be interpreted.
5. Current valid paper-aligned run family is based on:
   - MiniMind tokenizer
   - packed cache
   - equal-FLOPs sequence lengths
   - recovery ratio interpreted as recovery fraction

---

## 12. Final Instruction for the Next Conversation

When starting the next conversation, the priority is:

```text
1. Analyze the clean P0/P1/P2 batch.
2. Make checkpoint evaluation more stable.
3. Decide whether vanilla TST signal is real.
4. Only then consider structure-aware variants again.
```

If there is any uncertainty between:

- "expand methods"
- "improve evaluation fidelity"

choose:

```text
improve evaluation fidelity
```
