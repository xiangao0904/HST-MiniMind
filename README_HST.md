# HST-MiniMind

This repository contains an additive, isolated MVP for structure-aware Token Superposition Training experiments. The current project root did not include MiniMind source code, so the training entrypoint uses a small GPT-style causal LM while keeping the HST composer and loss interfaces separate for later MiniMind integration.

## Git Note

The execution platform exposes a read-only `.git` directory in the project root. Git metadata for this workspace is stored in `hst_git/`.

Use:

```bash
git --git-dir=hst_git --work-tree=. status
git --git-dir=hst_git --work-tree=. log --oneline
```

## Local Verification

Install dependencies in a project/local environment, not in conda base:

```bash
pip install -r requirements.txt
bash scripts/hst_local_verify.sh
```

The local smoke script creates `hst_tmp/tiny_pretrain.jsonl` and runs 3-step dry runs for:

- `ntp_baseline`
- `vanilla_tst`
- `order_aware_tst`
- `boundary_aware_tst`
- `hierarchical_tst`

All outputs are constrained to `hst_runs/`.

## Remote Training

Place the MiniMind-style pretraining JSONL at `dataset/pretrain_t2t_mini.jsonl` or update the config to a read-only dataset path inside the project policy.

```bash
bash scripts/hst_remote_train.sh configs/hst/remote_hst_s4_short_recovery.yaml
```

If `NPROC_PER_NODE` is set, the script uses `torchrun`; otherwise it uses `python3`. Each remote run writes under:

```text
hst_runs/{timestamp}_{run_name}/
```

## Metrics

```bash
python3 scripts/hst_collect_metrics.py --runs_dir ./hst_runs --output_dir ./hst_outputs
```

This writes `summary.csv` and `summary.md` with final eval loss, best eval loss, recovery gap, and throughput estimates.

## Probe Template

```bash
python3 scripts/hst_eval_probes.py --output ./hst_outputs/probe_results.jsonl
```

The current probe script writes deterministic order and boundary probe pairs. Model-scored probe losses can be added once a trained checkpoint is available.
