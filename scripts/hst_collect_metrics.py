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


def _eval_value(row: dict) -> float | None:
    return row.get("loss_eval_ntp", row.get("loss_eval"))


def recovery_gap(rows: list[dict]) -> float | None:
    last_super = None
    for row in rows:
        eval_loss = _eval_value(row)
        if row.get("phase") == "superposition" and eval_loss is not None:
            last_super = eval_loss
        if row.get("phase") == "recovery" and eval_loss is not None and last_super is not None:
            return eval_loss - last_super
    return None


def _tst_ratio(row: dict) -> float | None:
    if "tst_ratio" in row:
        return row["tst_ratio"]
    if row.get("method") == "ntp_baseline":
        return 0.0
    recovery_ratio = row.get("recovery_ratio")
    return 1.0 - recovery_ratio if recovery_ratio is not None else None


def summarize(path: Path, threshold: float | None) -> dict:
    rows = load_metrics(path)
    evals = [r for r in rows if _eval_value(r) is not None]
    final = _eval_value(evals[-1]) if evals else None
    best = min((_eval_value(r) for r in evals), default=None)
    time_to = None
    if threshold is not None:
        for row in evals:
            if _eval_value(row) <= threshold:
                time_to = row["wall_time_sec"]
                break
    elapsed = rows[-1]["wall_time_sec"] if rows else 0.0
    tokens = rows[-1].get("raw_tokens_seen", rows[-1].get("tokens_seen", 0)) if rows else 0
    data_tokens = rows[-1].get("effective_data_tokens_seen", rows[-1].get("effective_tokens_seen", 0)) if rows else 0
    examples = rows[-1]["step"] if rows else 0
    return {
        "run": path.parent.name,
        "method": rows[-1].get("experiment_method", rows[-1].get("method")) if rows else "",
        "tst_ratio": _tst_ratio(rows[-1]) if rows else None,
        "recovery_ratio": rows[-1].get("recovery_ratio") if rows else None,
        "final_eval_ntp_loss": final,
        "best_eval_ntp_loss": best,
        "time_to_loss": time_to,
        "recovery_gap_ntp": recovery_gap(rows),
        "raw_tokens_per_sec": tokens / elapsed if elapsed else None,
        "data_tokens_per_sec": data_tokens / elapsed if elapsed else None,
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
    metrics = sorted(runs_dir.glob("*/metrics.jsonl"))
    rows = [summarize(path, args.threshold) for path in metrics]
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    fieldnames = ["run", "method", "tst_ratio", "recovery_ratio", "final_eval_ntp_loss", "best_eval_ntp_loss", "time_to_loss", "recovery_gap_ntp", "raw_tokens_per_sec", "data_tokens_per_sec", "steps_per_sec", "metrics_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Run | Method | TST Ratio | Recovery Ratio | Final NTP Eval Loss | Best NTP Eval Loss | Recovery Gap NTP |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(f"| {row['run']} | {row['method']} | {row['tst_ratio']} | {row['recovery_ratio']} | {row['final_eval_ntp_loss']} | {row['best_eval_ntp_loss']} | {row['recovery_gap_ntp']} |\n")
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
