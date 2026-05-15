#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hst_path_safety import ensure_within_project, safe_mkdir


def load_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_value(row: dict) -> float | None:
    return row.get("loss_eval_ntp", row.get("loss_eval"))


def eval_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if eval_value(row) is not None]


def first_recovery_step(rows: list[dict]) -> int | None:
    for row in rows:
        if row.get("phase") == "recovery":
            return int(row["step"])
    return None


def baseline_loss_at_same_step(rows: list[dict], step: int) -> float | None:
    for row in eval_rows(rows):
        if int(row["step"]) == step:
            return eval_value(row)
    return None


def steps_to_beat_baseline(rows: list[dict], baseline_rows: list[dict] | None, switch_step: int | None) -> int | None:
    if baseline_rows is None or switch_step is None:
        return None
    for row in eval_rows(rows):
        step = int(row["step"])
        if step < switch_step:
            continue
        baseline_loss = baseline_loss_at_same_step(baseline_rows, step)
        if baseline_loss is not None and eval_value(row) <= baseline_loss:
            return step - switch_step
    return None


def analyze(rows: list[dict], baseline_rows: list[dict] | None = None) -> dict:
    switch_step = first_recovery_step(rows)
    last_before = None
    first_after = None
    for row in eval_rows(rows):
        if row.get("phase") == "superposition":
            last_before = row
        elif row.get("phase") == "recovery" and first_after is None:
            first_after = row

    before_loss = eval_value(last_before) if last_before is not None else None
    after_loss = eval_value(first_after) if first_after is not None else None
    return {
        "run": rows[-1].get("run_name", "") if rows else "",
        "method": rows[-1].get("method", "") if rows else "",
        "recovery_start_step": switch_step,
        "last_ntp_eval_before_recovery": before_loss,
        "last_ntp_eval_before_recovery_step": last_before.get("step") if last_before else None,
        "first_ntp_eval_after_recovery": after_loss,
        "first_ntp_eval_after_recovery_step": first_after.get("step") if first_after else None,
        "recovery_gap": after_loss - before_loss if before_loss is not None and after_loss is not None else None,
        "steps_to_beat_baseline_same_step": steps_to_beat_baseline(rows, baseline_rows, switch_step),
    }


def write_output(result: dict, output: Path) -> None:
    safe_mkdir(output.parent)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = output.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Run | Recovery Start | Last Before | First After | Recovery Gap | Steps To Beat Baseline |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        f.write(
            f"| {result['run']} | {result['recovery_start_step']} | "
            f"{result['last_ntp_eval_before_recovery']} | {result['first_ntp_eval_after_recovery']} | "
            f"{result['recovery_gap']} | {result['steps_to_beat_baseline_same_step']} |\n"
        )
    print(f"wrote {output} and {md_path}")


def metrics_path(run_dir: Path) -> Path:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"metrics not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baseline_run_dir")
    parser.add_argument("--output", default="./hst_outputs/recovery_analysis.json")
    args = parser.parse_args()

    run_dir = ensure_within_project(args.run_dir)
    rows = load_metrics(metrics_path(run_dir))
    baseline_rows = None
    if args.baseline_run_dir:
        baseline_dir = ensure_within_project(args.baseline_run_dir)
        baseline_rows = load_metrics(metrics_path(baseline_dir))
    result = analyze(rows, baseline_rows)
    write_output(result, ensure_within_project(args.output))


if __name__ == "__main__":
    main()
