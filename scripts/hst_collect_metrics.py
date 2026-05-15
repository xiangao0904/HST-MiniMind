#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hst_path_safety import ensure_within_project, safe_mkdir


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def recovery_gap(rows: list[dict]) -> float | None:
    last_super = None
    for row in rows:
        if row.get("phase") == "superposition" and row.get("loss_eval") is not None:
            last_super = row["loss_eval"]
        if row.get("phase") == "recovery" and row.get("loss_eval") is not None and last_super is not None:
            return row["loss_eval"] - last_super
    return None


def summarize(path: Path, threshold: float | None) -> dict:
    rows = load_metrics(path)
    evals = [r for r in rows if r.get("loss_eval") is not None]
    final = evals[-1]["loss_eval"] if evals else None
    best = min((r["loss_eval"] for r in evals), default=None)
    time_to = None
    if threshold is not None:
        for row in evals:
            if row["loss_eval"] <= threshold:
                time_to = row["wall_time_sec"]
                break
    elapsed = rows[-1]["wall_time_sec"] if rows else 0.0
    tokens = rows[-1]["tokens_seen"] if rows else 0
    examples = rows[-1]["step"] if rows else 0
    return {
        "run": path.parent.name,
        "method": rows[-1].get("method") if rows else "",
        "final_eval_loss": final,
        "best_eval_loss": best,
        "time_to_loss": time_to,
        "recovery_gap": recovery_gap(rows),
        "tokens_per_sec": tokens / elapsed if elapsed else None,
        "steps_per_sec": examples / elapsed if elapsed else None,
        "metrics_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="./hst_runs")
    parser.add_argument("--output_dir", default="./hst_outputs")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    runs_dir = ensure_within_project(args.runs_dir)
    out_dir = safe_mkdir(args.output_dir)
    metrics = sorted(runs_dir.glob("**/metrics.jsonl"))
    rows = [summarize(path, args.threshold) for path in metrics]
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    fieldnames = ["run", "method", "final_eval_loss", "best_eval_loss", "time_to_loss", "recovery_gap", "tokens_per_sec", "steps_per_sec", "metrics_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Run | Method | Final Eval Loss | Best Eval Loss | Recovery Gap |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(f"| {row['run']} | {row['method']} | {row['final_eval_loss']} | {row['best_eval_loss']} | {row['recovery_gap']} |\n")
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
