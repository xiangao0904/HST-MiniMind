#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for offline evaluation.") from exc

from trainer.train_hst_pretrain import (
    CharTokenizer,
    JsonlTextDataset,
    MiniMindTokenizer,
    TinyCausalLM,
    TokenizedTensorDataset,
    TrainConfig,
    baseline_seq_len,
    evaluate,
    load_texts,
    tst_ratio,
    checkpoint_step,
)
from utils.hst_path_safety import ensure_within_project, safe_mkdir


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def load_config_for_eval(run_dir: Path | None, checkpoint_data: dict) -> TrainConfig:
    data = asdict(TrainConfig())
    if run_dir is not None:
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            data.update(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    data.update(checkpoint_data.get("config") or {})
    return TrainConfig(**data)


def build_eval_loader(cfg: TrainConfig):
    seq_len = baseline_seq_len(cfg)
    if cfg.use_tokenized_cache:
        dataset = TokenizedTensorDataset(ensure_within_project(cfg.tokenized_cache_path), seq_len)
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False), dataset.vocab_size

    data_path = ensure_within_project(cfg.data_path)
    texts = load_texts(data_path)
    if cfg.tokenizer_backend == "minimind":
        tokenizer = MiniMindTokenizer(ensure_within_project(cfg.tokenizer_path))
    else:
        tokenizer = CharTokenizer(texts)
    dataset = JsonlTextDataset(data_path, tokenizer, seq_len)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False), tokenizer.vocab_size


def eval_checkpoint(checkpoint: Path, run_dir: Path | None, device_name: str, eval_max_batches: int) -> dict:
    data = torch.load(checkpoint, map_location="cpu")
    cfg = load_config_for_eval(run_dir, data)
    cfg.device = device_name
    loader, vocab_size = build_eval_loader(cfg)
    device = torch.device(device_name)
    model = TinyCausalLM(vocab_size, baseline_seq_len(cfg), cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.dropout).to(device)
    model.load_state_dict(data["model"])
    loss = evaluate(model, None, loader, cfg, device, "ntp", eval_max_batches)
    batches = min(eval_max_batches, len(loader))
    return {
        "run": run_dir.name if run_dir is not None else checkpoint.parent.parent.name,
        "checkpoint": str(checkpoint),
        "step": int(data.get("step", 0)),
        "method": cfg.method,
        "loss_eval_ntp": loss,
        "eval_max_batches": eval_max_batches,
        "eval_batches": batches,
        "eval_tokens_budget": batches * cfg.batch_size * baseline_seq_len(cfg),
        "batch_size": cfg.batch_size,
        "baseline_seq_len": baseline_seq_len(cfg),
        "superpose_size": cfg.superpose_size,
        "tst_ratio": tst_ratio(cfg),
        "recovery_ratio": cfg.recovery_ratio,
        "paper_equal_flops": cfg.paper_equal_flops,
    }


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    safe_mkdir(output_dir)
    json_path = output_dir / "offline_eval_summary.json"
    md_path = output_dir / "offline_eval_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Run | Method | Step | NTP Eval Loss | Batches | TST Ratio | Recovery Ratio |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['run']} | {row['method']} | {row['step']} | {row['loss_eval_ntp']} | "
                f"{row['eval_batches']} | {row['tst_ratio']} | {row['recovery_ratio']} |\n"
            )
    print(f"wrote {json_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--run_dir")
    parser.add_argument("--output_dir", default="./hst_outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_max_batches", type=int, default=50)
    args = parser.parse_args()

    if args.eval_max_batches <= 0:
        raise ValueError("eval_max_batches must be positive")

    run_dir = ensure_within_project(args.run_dir) if args.run_dir else None
    checkpoint = latest_checkpoint(run_dir) if run_dir is not None else ensure_within_project(args.checkpoint)
    row = eval_checkpoint(checkpoint, run_dir, args.device, args.eval_max_batches)
    write_outputs([row], ensure_within_project(args.output_dir))


if __name__ == "__main__":
    main()
