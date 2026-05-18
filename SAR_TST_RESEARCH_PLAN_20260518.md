# SAR-TST Research Plan

Date: 2026-05-18

## Goal

Design and test a residual-based TST variant that is more likely to improve wall-clock step speedup than the current `residual_structured_tst`, while preserving the main stability benefit of a vanilla-mean backbone.

Important baseline constraint:

- the reference vanilla must be the paper-style vanilla TST protocol
- `SAR-TST` should be treated as a modification on top of that paper-vanilla backbone, not as a tweak on top of the older structured residual line

Working name:

- `SAR-TST`
- full repo method name: `sparse_anchor_residual_tst`

## Core Idea

Current residual TST keeps the optimization stable by using:

- `z_mean + small residual`

But it is still dense in two places:

- every chunk can carry structured residual information
- supervision still tries to recover all target tokens with a heavy objective

`SAR-TST` changes the structure from dense residual supervision to sparse residual supervision:

1. Use vanilla mean as the backbone latent.
2. Add order residual only when a cheap gate says the chunk is worth the extra structure.
3. Predict one anchor token with full-vocabulary CE.
4. Predict the remaining `s-1` slots with a small residual-code objective instead of full-token recovery.

This is intended to improve the speed-quality tradeoff rather than raw representational power.

Operationally, the intended backbone is:

- paper-style vanilla TST
- same equal-FLOPs setup
- same repeated CE target family
- same optimizer/LR family
- same WSD scheduler shape when using the paper-aligned long-run configs

## Method Definition

For a source chunk with token embeddings `E(x_1 ... x_s)`:

- `z_mean = mean(E(x_1 ... x_s))`
- `r_order = order_residual(E, z_mean)`
- `g = 1[ rms(r_order) > tau ]`
- `z = z_mean + g * alpha * r_order`

In the current minimal implementation:

- `alpha = order_alpha`
- `tau = sar_gate_threshold`
- hierarchy is disabled by default for the first experiment

Optional extension kept in code path:

- `z = z + g * beta * r_hier`

but the initial debug/full configs set `hier_alpha: 0.0` to keep the first ablation clean.

## Target Construction

For each target chunk:

- anchor slot: `anchor_slot_idx`, default `0`
- anchor target: token id at the anchor slot
- residual targets: quantized embedding deltas from each non-anchor slot to the anchor slot

Residual code target construction:

1. Embed target chunk tokens.
2. Compute `delta_j = E(x_j) - E(x_anchor)` for non-anchor slots.
3. Assign each delta to the nearest vector in a learnable residual codebook of size `K = residual_codebook_size`.

This keeps the dense token supervision only on the anchor path.

## Loss

Total stage1 loss:

- `L = L_anchor + lambda_res * L_residual`

Where:

- `L_anchor`: standard full-vocabulary CE on the anchor token
- `L_residual`: CE over residual code indices for non-anchor slots
- `lambda_res = residual_loss_weight`

Residual code loss is only applied on gated positions. If no positions are active in a batch, the residual branch contributes zero loss for that batch.

## Why This Should Be Faster

Compared with dense structured TST:

- full-vocabulary prediction pressure is reduced to one anchor token per chunk
- non-anchor supervision uses a small codebook classifier instead of direct token recovery
- residual structure is activated sparsely by a cheap gate

In this toy repo, step speed differences will be modest because the transformer forward still dominates. In a more realistic setup with larger decoding heads and richer slot supervision, the reduction should matter more.

## Minimal Experiment Sequence

### Phase A: Logic and stability

Run a local CPU debug experiment with:

- `method: sparse_anchor_residual_tst`
- `superpose_size: 4`
- `loss_mode: sparse_anchor_residual`
- `order_alpha: 0.05`
- `hier_alpha: 0.0`
- `anchor_slot_idx: 0`
- `residual_codebook_size: 64`
- `sar_gate_threshold: 0.05`
- `residual_loss_weight: 0.5`

Purpose:

- verify code path
- verify W&B init
- verify loss stays finite

### Phase B: First full-size comparison

Compare against the existing 120k family with:

- `P2_vanilla_tst_s4_r03_full_120k`
- `S3_residual_structured_s4_r03_full_120k`
- `S4_sparse_anchor_residual_s4_r03_full_120k`

Keep fixed:

- `s = 4`
- `recovery_ratio = 0.7`
- same tokenizer/cache
- same optimizer/LR
- same WSD scheduler shape
- same model size

Interpretation rule:

- `P2_vanilla_tst_s4_r03_full_120k` is the true control
- `S4_sparse_anchor_residual_s4_r03_full_120k` should differ from that control only by the SAR residual path and its auxiliary target

Primary questions:

1. Is stage1 NTP eval as good as or close to vanilla/residual?
2. Does stage1 wall time per effective token improve?
3. Does recovery erase or preserve the stage1 advantage?

## Metrics To Watch

- `loss_train`
- `loss_eval_ntp`
- `loss_eval_phase`
- `wall_time_sec`
- `raw_tokens_seen`
- `effective_data_tokens_seen`

Additional recommended metric to add later if needed:

- `sar_gate_rate`

The current implementation computes gate rate in composer metadata, but it is not yet logged into `metrics.jsonl` or W&B.

## Ablation Order

1. `SAR` without hierarchy: current baseline
2. `SAR + hierarchy residual`
3. codebook size `64 -> 128 -> 256`
4. gate threshold sweep
5. residual loss weight sweep

## Self-Check

### What is logically consistent

- The backbone latent remains close to vanilla mean, so optimization should be at least as stable as the current residual family.
- The residual branch is sparse by construction, which matches the speedup goal.
- Anchor prediction and residual-code prediction are separated cleanly.

### What is still approximate

- Residual code targets are built from the current embedding table, so the quantization target moves during training.
- Gate sparsity is heuristic, not learned.
- This implementation reduces supervision cost more than transformer FLOPs, so any step-speed gain may be smaller than the conceptual gain.

### Expected failure modes

- gate too strict: model collapses to near-vanilla and residual branch does nothing
- gate too loose: method degenerates toward dense residual TST
- codebook too small: residual branch underfits
- codebook too large: auxiliary head gets heavier and may eat the speed benefit

## Files Added or Changed

- `model/hst_superposition.py`
- `model/hst_losses.py`
- `trainer/train_hst_pretrain.py`
- `configs/hst/sparse_anchor_residual_s4_debug.yaml`
- `configs/hst/structure_sparse_anchor_residual_s4_r03_full_120k.yaml`
- `tests/test_hst_losses.py`
- `tests/test_hst_shapes.py`
- `tests/test_hst_training_protocol.py`
- `scripts/hst_local_verify.sh`

## Launch Command

Local debug:

```bash
conda run -n tiny-jepa-debug python trainer/train_hst_pretrain.py --config configs/hst/sparse_anchor_residual_s4_debug.yaml
```

Remote/full:

```bash
bash scripts/hst_remote_train.sh configs/hst/structure_sparse_anchor_residual_s4_r03_full_120k.yaml
```
