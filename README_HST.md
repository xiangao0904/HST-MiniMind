# HST-MiniMind

This repository contains an additive, isolated MVP for structure-aware Token Superposition Training experiments. The current project root did not include MiniMind source code, so the training entrypoint uses a small GPT-style causal LM while keeping the HST composer and loss interfaces separate for later MiniMind integration.



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

For paper-style TST notation, `r` is the fraction of total steps spent in the superposition phase.
The trainer keeps the older `recovery_ratio` field, so `tst_ratio = 1 - recovery_ratio` for TST runs.

For stronger post-run checkpoint evaluation, use standard NTP offline eval:

```bash
python3 scripts/hst_offline_eval.py --run_dir ./hst_runs/P2_vanilla_tst_s4_r03_20k --device cuda --eval_max_batches 200
```

For recovery-specific analysis against a baseline run:

```bash
python3 scripts/hst_recovery_analysis.py \
  --run_dir ./hst_runs/P2_vanilla_tst_s4_r03_20k \
  --baseline_run_dir ./hst_runs/P0_ntp_baseline_20k
```

## Probe Template

```bash
python3 scripts/hst_eval_probes.py --output ./hst_outputs/probe_results.jsonl
```

The current probe script writes deterministic order and boundary probe pairs. Model-scored probe losses can be added once a trained checkpoint is available.
