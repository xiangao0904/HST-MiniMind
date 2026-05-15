#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from utils.hst_path_safety import ensure_run_output_dir, ensure_within_project, safe_mkdir

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - exercised on machines without torch.
    raise SystemExit("PyTorch is required for training. Install it in a project/local environment.") from exc

from model.hst_losses import ntp_loss, ordered_slot_loss, repeated_token_ce_loss
from model.hst_superposition import SuperpositionComposer, SuperpositionConfig
from model.hst_token_types import build_token_type_cache


METHODS = {"ntp_baseline", "vanilla_tst", "order_aware_tst", "boundary_aware_tst", "hierarchical_tst"}


@dataclass
class TrainConfig:
    method: str = "ntp_baseline"
    data_path: str = "./hst_tmp/tiny_pretrain.jsonl"
    run_name: str = "debug"
    output_dir: str = "./hst_runs/debug"
    max_steps: int = 3
    eval_interval: int = 1
    save_interval: int = 100
    seed: int = 42
    superpose_size: int = 2
    superpose_mode: str = "mean"
    loss_mode: str = "repeated_ce"
    recovery_ratio: float = 0.0
    learning_rate: float = 3e-4
    batch_size: int = 2
    max_seq_len: int = 128
    device: str = "cpu"
    dry_run: int = 0
    debug: int = 0
    from_resume: int = 0
    block_mode: str = "fixed"
    chunks_per_block: int = 8
    hier_alpha: float = 0.1
    slot_gate_type: str = "embedding"
    type_vocab_size: int = 11
    log_jsonl: str = "metrics.jsonl"
    use_wandb: int = 0
    use_swanlab: int = 0
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0


class CharTokenizer:
    def __init__(self, texts: list[str]) -> None:
        chars = sorted(set("".join(texts)))
        self.id_to_text = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        for ch in chars:
            if ch not in self.id_to_text.values():
                self.id_to_text[len(self.id_to_text)] = ch
        self.text_to_id = {v: k for k, v in self.id_to_text.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_text)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.text_to_id["<bos>"]]
        ids.extend(self.text_to_id.get(ch, self.text_to_id["<unk>"]) for ch in text)
        ids.append(self.text_to_id["<eos>"])
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids.extend([self.text_to_id["<pad>"]] * (max_len - len(ids)))
        return ids


class JsonlTextDataset(Dataset):
    def __init__(self, path: Path, tokenizer: CharTokenizer, max_seq_len: int) -> None:
        self.rows = load_texts(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(self.rows[idx], self.max_seq_len), dtype=torch.long)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            x = self.token_embedding(input_ids)
        else:
            x = inputs_embeds
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(pos).view(1, seq_len, -1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        hidden = self.blocks(x, mask=mask)
        hidden = self.ln_f(hidden)
        return {"logits": self.lm_head(hidden), "hidden_states": hidden}


def load_texts(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(str(row.get("text", "")))
    if not texts:
        raise ValueError(f"empty dataset: {path}")
    return texts


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    for field, value in asdict(TrainConfig()).items():
        arg_type = type(value)
        parser.add_argument(f"--{field}", type=arg_type, default=None)
    ns = parser.parse_args()
    data = {}
    if ns.config:
        config_path = ensure_within_project(ns.config)
        data.update(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for key, value in vars(ns).items():
        if key != "config" and value is not None:
            data[key] = value
    cfg = TrainConfig(**data)
    validate_config(cfg)
    return cfg


def validate_config(cfg: TrainConfig) -> None:
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method: {cfg.method}")
    if cfg.dry_run and cfg.max_steps > 20:
        raise ValueError("dry_run requires max_steps <= 20")
    if not 0.0 <= cfg.recovery_ratio <= 1.0:
        raise ValueError("recovery_ratio must be in [0, 1]")
    if cfg.use_wandb or cfg.use_swanlab:
        raise ValueError("wandb/swanlab integration is intentionally disabled in this MVP")


def phase_for_step(cfg: TrainConfig, step: int) -> str:
    if cfg.method == "ntp_baseline":
        return "ntp"
    recovery_start = int(cfg.max_steps * (1.0 - cfg.recovery_ratio))
    return "superposition" if step < recovery_start else "recovery"


def method_to_mode(cfg: TrainConfig) -> str:
    return {
        "vanilla_tst": "mean",
        "order_aware_tst": "order_aware",
        "boundary_aware_tst": "boundary_aware",
        "hierarchical_tst": "hierarchical",
    }.get(cfg.method, cfg.superpose_mode)


def evaluate(model, composer, loader, cfg: TrainConfig, device: torch.device, phase: str, max_batches: int = 2) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            losses.append(compute_loss(model, composer, batch, cfg, phase).detach())
    model.train()
    return float(torch.stack(losses).mean().cpu())


def compute_loss(model, composer, batch: torch.Tensor, cfg: TrainConfig, phase: str) -> torch.Tensor:
    if phase in {"ntp", "recovery"}:
        return ntp_loss(model(input_ids=batch)["logits"], batch)
    composed = composer.compose(batch)
    out = model(inputs_embeds=composed["inputs_embeds"])
    if cfg.loss_mode == "ordered_slot":
        return ordered_slot_loss(out["hidden_states"], composed["chunk_targets"], model.lm_head, composer.slot_embed)
    return repeated_token_ce_loss(out["logits"], composed["chunk_targets"])


def save_checkpoint(path: Path, model, optimizer, step: int, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(cfg)}, path)


def load_latest_checkpoint(ckpt_dir: Path, model, optimizer) -> int:
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    if not checkpoints:
        return 0
    data = torch.load(checkpoints[-1], map_location="cpu")
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return int(data.get("step", 0))


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data_path = ensure_within_project(cfg.data_path)
    output_dir = ensure_run_output_dir(cfg.output_dir)
    safe_mkdir(output_dir)
    ckpt_dir = safe_mkdir(output_dir / "checkpoints")
    safe_mkdir(output_dir / "outputs")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=True), encoding="utf-8")

    texts = load_texts(data_path)
    tokenizer = CharTokenizer(texts)
    dataset = JsonlTextDataset(data_path, tokenizer, cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    device = torch.device(cfg.device)
    model = TinyCausalLM(tokenizer.vocab_size, cfg.max_seq_len, cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.dropout).to(device)
    token_types = torch.tensor(build_token_type_cache(tokenizer.id_to_text, output_dir / "token_type_cache.json"), dtype=torch.long, device=device)
    sp_cfg = SuperpositionConfig(
        mode=method_to_mode(cfg),
        superpose_size=cfg.superpose_size,
        hidden_size=cfg.hidden_size,
        vocab_size=tokenizer.vocab_size,
        slot_gate_type=cfg.slot_gate_type,
        type_vocab_size=cfg.type_vocab_size,
        block_mode=cfg.block_mode,
        chunks_per_block=cfg.chunks_per_block,
        hier_alpha=cfg.hier_alpha,
    )
    composer = SuperpositionComposer(model.token_embedding, cfg.hidden_size, tokenizer.vocab_size, sp_cfg, token_types).to(device)
    model_param_ids = {id(p) for p in model.parameters()}
    composer_params = [p for p in composer.parameters() if id(p) not in model_param_ids]
    optimizer = torch.optim.AdamW(list(model.parameters()) + composer_params, lr=cfg.learning_rate)
    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer) if cfg.from_resume else 0
    metrics_path = output_dir / cfg.log_jsonl
    start_time = time.time()
    data_iter = iter(loader)

    print(yaml.safe_dump(asdict(cfg), sort_keys=True))
    for step in range(start_step, cfg.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = batch.to(device)
        phase = phase_for_step(cfg, step)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, composer, batch, cfg, phase)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()

        eval_loss = None
        if (step + 1) % cfg.eval_interval == 0 or step == 0:
            eval_loss = evaluate(model, composer, eval_loader, cfg, device, phase)
        if (step + 1) % cfg.save_interval == 0 or step + 1 == cfg.max_steps:
            save_checkpoint(ckpt_dir / f"step_{step + 1}.pt", model, optimizer, step + 1, cfg)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "run_name": cfg.run_name,
            "method": cfg.method,
            "step": step + 1,
            "phase": phase,
            "loss_train": float(loss.detach().cpu()),
            "loss_eval": eval_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "tokens_seen": (step + 1) * cfg.batch_size * cfg.max_seq_len,
            "effective_tokens_seen": (step + 1) * cfg.batch_size * cfg.max_seq_len / max(1, cfg.superpose_size),
            "superpose_size": cfg.superpose_size,
            "recovery_ratio": cfg.recovery_ratio,
            "wall_time_sec": time.time() - start_time,
            "gpu_mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

    shutil.copy2(metrics_path, output_dir / "outputs" / "metrics.jsonl")


if __name__ == "__main__":
    main()
