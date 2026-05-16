#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_PATH="$PROJECT_ROOT/hst_tmp/tiny_pretrain.jsonl"

"$PYTHON_BIN" scripts/hst_make_tiny_dataset.py --output "$DATA_PATH" --num_examples 64

COMMON_ARGS=(
  --data_path "$DATA_PATH"
  --max_steps 3
  --eval_interval 1
  --save_interval 100
  --batch_size 2
  --max_seq_len 64
  --learning_rate 0.0003
  --dry_run 1
  --debug 1
  --device cpu
)

"$PYTHON_BIN" trainer/train_hst_pretrain.py --method ntp_baseline --run_name verify_ntp --output_dir "$PROJECT_ROOT/hst_runs/verify_ntp" "${COMMON_ARGS[@]}"
"$PYTHON_BIN" trainer/train_hst_pretrain.py --method vanilla_tst --run_name verify_vanilla --output_dir "$PROJECT_ROOT/hst_runs/verify_vanilla" --superpose_size 2 "${COMMON_ARGS[@]}"
"$PYTHON_BIN" trainer/train_hst_pretrain.py --method order_aware_tst --run_name verify_order --output_dir "$PROJECT_ROOT/hst_runs/verify_order" --superpose_size 2 --slot_gate_type embedding "${COMMON_ARGS[@]}"
"$PYTHON_BIN" trainer/train_hst_pretrain.py --method boundary_aware_tst --run_name verify_boundary --output_dir "$PROJECT_ROOT/hst_runs/verify_boundary" --superpose_size 2 "${COMMON_ARGS[@]}"
"$PYTHON_BIN" trainer/train_hst_pretrain.py --method hierarchical_tst --run_name verify_hier --output_dir "$PROJECT_ROOT/hst_runs/verify_hier" --superpose_size 4 --loss_mode ordered_slot --recovery_ratio 0.34 "${COMMON_ARGS[@]}"
"$PYTHON_BIN" trainer/train_hst_pretrain.py --method residual_structured_tst --run_name verify_residual_structured --output_dir "$PROJECT_ROOT/hst_runs/verify_residual_structured" --superpose_size 4 --loss_mode ordered_slot --order_alpha 0.05 --hier_alpha 0.05 --recovery_ratio 0.34 "${COMMON_ARGS[@]}"
