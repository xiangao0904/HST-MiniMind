#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/hst_remote_train.sh configs/hst/remote_hst_s4_short_recovery.yaml" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [[ -z "$PROJECT_ROOT" || "$PROJECT_ROOT" == "/" ]]; then
  echo "PROJECT_ROOT must be set and must not be /" >&2
  exit 2
fi
export PROJECT_ROOT
cd "$PROJECT_ROOT"

CONFIG="$1"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("run_name", "hst_run"))
PY
)"
RUN_DIR="$PROJECT_ROOT/hst_runs/${TIMESTAMP}_${RUN_NAME}"

echo "host: $(hostname)"
echo "project_root: $PROJECT_ROOT"
echo "python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" -V
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi: not found"
fi

"$PYTHON_BIN" - "$PROJECT_ROOT" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
run = Path(sys.argv[2]).resolve()
if str(root) == "/":
    raise SystemExit("refusing PROJECT_ROOT=/")
if root not in run.parents:
    raise SystemExit(f"run dir escapes project root: {run}")
if (root / "hst_runs").resolve() not in run.parents:
    raise SystemExit(f"run dir must be under hst_runs: {run}")
for child in ("checkpoints", "outputs", "plots", "artifacts"):
    (run / child).mkdir(parents=True, exist_ok=True)
PY

cp "$CONFIG" "$RUN_DIR/config.yaml"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  torchrun --nproc_per_node "$NPROC_PER_NODE" trainer/train_hst_pretrain.py --config "$CONFIG" --output_dir "$RUN_DIR" \
    >"$RUN_DIR/stdout.log" 2>"$RUN_DIR/stderr.log"
else
  "$PYTHON_BIN" trainer/train_hst_pretrain.py --config "$CONFIG" --output_dir "$RUN_DIR" \
    >"$RUN_DIR/stdout.log" 2>"$RUN_DIR/stderr.log"
fi
