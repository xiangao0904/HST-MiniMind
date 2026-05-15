#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from trainer.train_hst_pretrain import TrainConfig, recovery_start_step, tst_ratio
from utils.hst_path_safety import ensure_run_output_dir, ensure_within_project, safe_mkdir


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = TrainConfig(**{**asdict(TrainConfig()), **data})
    if cfg.method == "ntp_baseline":
        raise ValueError("two-phase paper training requires a TST method, not ntp_baseline")
    if not 0.0 < tst_ratio(cfg) < 1.0:
        raise ValueError("two-phase paper training requires 0 < tst_ratio < 1")
    return asdict(cfg)


def write_config(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def run_stage(python_bin: str, config_path: Path, run_dir: Path) -> None:
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        subprocess.run(
            [python_bin, "trainer/train_hst_pretrain.py", "--config", str(config_path)],
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )


def combine_metrics(parent: Path, phase1_dir: Path, phase2_dir: Path) -> None:
    output = parent / "metrics.jsonl"
    with output.open("w", encoding="utf-8") as out:
        for metrics_path in (phase1_dir / "metrics.jsonl", phase2_dir / "metrics.jsonl"):
            if not metrics_path.exists():
                continue
            with metrics_path.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_root", default="./hst_runs")
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    config_path = ensure_within_project(args.config)
    base = load_config(config_path)
    total_steps = int(base["max_steps"])
    tst_steps = recovery_start_step(TrainConfig(**base))
    recovery_steps = total_steps - tst_steps
    run_name = base.get("run_name") or "hst_two_phase"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parent = ensure_run_output_dir(Path(args.output_root) / f"{timestamp}_{run_name}_paper_two_phase")
    safe_mkdir(parent)

    phase1_dir = safe_mkdir(parent / "phase1_tst")
    phase2_dir = safe_mkdir(parent / "phase2_recovery")
    safe_mkdir(parent / "configs")

    phase1 = dict(base)
    phase1.update(
        {
            "experiment_method": base["method"],
            "run_name": f"{run_name}_phase1_tst",
            "output_dir": str(phase1_dir),
            "max_steps": tst_steps,
            "global_step_offset": 0,
            "recovery_ratio": 0.0,
            "phase_override": "superposition",
            "init_checkpoint_path": "",
            "from_resume": 0,
        }
    )
    phase1_config = parent / "configs" / "phase1_tst.yaml"
    write_config(phase1_config, phase1)

    phase1_final = phase1_dir / "checkpoints" / f"step_{tst_steps}.pt"
    phase2 = dict(base)
    phase2.update(
        {
            "method": "ntp_baseline",
            "experiment_method": base["method"],
            "run_name": f"{run_name}_phase2_recovery",
            "output_dir": str(phase2_dir),
            "max_steps": recovery_steps,
            "global_step_offset": tst_steps,
            "recovery_ratio": 0.0,
            "phase_override": "recovery",
            "init_checkpoint_path": str(phase1_final),
            "from_resume": 0,
        }
    )
    phase2_config = parent / "configs" / "phase2_recovery.yaml"
    write_config(phase2_config, phase2)

    manifest = {
        "source_config": str(config_path),
        "run_name": run_name,
        "total_steps": total_steps,
        "tst_steps": tst_steps,
        "recovery_steps": recovery_steps,
        "paper_tst_ratio": tst_ratio(TrainConfig(**base)),
        "phase1_dir": str(phase1_dir),
        "phase2_dir": str(phase2_dir),
    }
    write_config(parent / "manifest.yaml", manifest)

    print(f"paper two-phase run: {parent}")
    print(f"phase1 TST steps: {tst_steps}")
    run_stage(args.python_bin, phase1_config, phase1_dir)
    if not phase1_final.exists():
        raise FileNotFoundError(f"phase1 final checkpoint not found: {phase1_final}")
    print(f"phase2 recovery steps: {recovery_steps}")
    run_stage(args.python_bin, phase2_config, phase2_dir)
    combine_metrics(parent, phase1_dir, phase2_dir)
    print(f"completed paper two-phase run: {parent}")


if __name__ == "__main__":
    main()
